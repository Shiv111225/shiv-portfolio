"""Render QC: checks every 5-second segment (120 frames) of the animation.

    python qc.py --segment 3      # check one segment
    python qc.py                  # check all segments + write the summary report

Checks (per segment):
  frames      all 120 frames exist, 1280x720 RGB, no exact duplicates
  exposure    mean luma in range, no blown/black frames, letterbox bars pure black
  pops        temporal residual |F_i - (F_{i-1}+F_{i+1})/2| spikes (a frame that does
              not sit between its neighbours = glitch / pop / dropped pose)
  freeze      pixels stop changing while the rig is still moving
  bones       rigid bone lengths constant (< 0.1 % drift)
  ik          no IK hyperextension (reach < 0.995)
  feet        planted feet do not slide (< 0.05 px/frame) or sink below the ground
  jerk        joint acceleration spikes outside the declared action events
  cloth       cloak finite, never stretched > 25 %, never under the ground
  seam        continuity with the neighbouring segment's boundary frame
  framing     head, every horn tip, heart, hands, feet stay inside the letterboxed frame
  determinism one frame re-rendered from scratch must match (<= 10 isolated px may flip under parallel SwiftShader)
Spikes inside config.EVENTS windows are listed as intentional, not failures.
"""
import argparse
import hashlib
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import (W, H, FPS, NSEG, SEG_FRAMES, SEGMENTS, GROUND, LETTERBOX, QC_DIR, OUT, ROOT,
                    RIG_FILE, EVENTS, frame_path, seg_range, in_event)
PROJ_DIR = os.path.join(OUT, "proj")
import anim

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def small(path):
    im = Image.open(path).convert("L").resize((W // 8, H // 8), Image.BOX)  # averages out grain
    return np.asarray(im, np.float32) / 255.0


def check_segment(s, rerender=True):
    fr = list(seg_range(s))
    res = {"segment": s, "name": SEGMENTS[s][0], "frames": [fr[0], fr[-1]],
           "time": [fr[0] / FPS, (fr[-1] + 1) / FPS], "checks": {}, "intentional": []}
    C = res["checks"]

    def put(name, ok, detail, **kw):
        C[name] = dict(ok=bool(ok), detail=detail, **kw)

    # ---------------- frames
    missing = [f for f in fr if not os.path.exists(frame_path(f))]
    bad, hashes = [], {}
    dups = []
    for f in fr:
        if f in missing:
            continue
        im = Image.open(frame_path(f))
        if im.size != (W, H) or im.mode not in ("RGB", "RGBA"):
            bad.append(f)
        h = hashlib.md5(im.tobytes()).hexdigest()
        if h in hashes:
            dups.append([hashes[h], f])
        hashes[h] = f
    put("frames", not missing and not bad and not dups,
        f"{len(fr) - len(missing)}/{len(fr)} present, {len(bad)} wrong format, {len(dups)} duplicates",
        missing=missing, bad=bad, duplicates=dups)
    if missing:
        res["ok"] = False
        return res

    # pad with neighbours for seam / residual checks
    lo, hi = max(0, fr[0] - 1), min(NSEG * SEG_FRAMES - 1, fr[-1] + 1)
    have = [f for f in range(lo, hi + 1) if os.path.exists(frame_path(f))]
    G = {f: small(frame_path(f)) for f in have}

    # ---------------- exposure
    lum = np.array([G[f][LETTERBOX // 8 + 1:-LETTERBOX // 8 - 1].mean() for f in fr])
    bars_bad = []
    for f in fr[::10]:
        a = np.asarray(Image.open(frame_path(f)))
        if a[:LETTERBOX].max() > 0 or a[-LETTERBOX:].max() > 0:
            bars_bad.append(f)
    put("exposure", lum.min() > 0.03 and lum.max() < 0.75 and not bars_bad,
        f"mean luma {lum.min():.3f}..{lum.max():.3f}, letterbox breaches {len(bars_bad)}",
        luma=lum.round(4).tolist())

    # ---------------- temporal pixel analysis
    diff, resid = {}, {}
    for f in have[1:]:
        diff[f] = float(np.abs(G[f] - G[f - 1]).mean())
    for f in have[1:-1]:
        resid[f] = float(np.abs(G[f] - 0.5 * (G[f - 1] + G[f + 1])).mean())
    rv = np.array(list(resid.values()))
    base = float(np.median(rv)) if len(rv) else 0.0
    pops, intent = [], []
    for f, r in resid.items():
        loc = [resid[g] for g in range(f - 12, f + 13) if g in resid and g != f]
        ref = max(np.median(loc) if loc else base, 0.001)
        if r > 3.0 * ref and r > 0.003:
            ev = in_event(f / FPS)
            (intent if ev else pops).append(dict(frame=f, residual=round(r, 4), ratio=round(r / ref, 1), event=ev))
    put("pops", not pops, f"{len(pops)} unexplained pops (median residual {base:.4f}), "
        f"{len(intent)} spikes inside action events", pops=pops)
    res["intentional"] += [dict(check="pops", **p) for p in intent]

    # ---------------- rig
    with np.load(RIG_FILE) as z:
        R = {k: z[k] for k in z.files}
    names = list(R["names"])
    J = R["joints"]
    idx = {n: i for i, n in enumerate(names)}
    seg = slice(fr[0], fr[-1] + 1)
    # bones
    worst = 0.0
    for a, b, L in anim.BONES:
        d = np.hypot(*(J[seg, idx[a]] - J[seg, idx[b]]).T)
        worst = max(worst, float(np.abs(d / L - 1).max()))
    put("bones", worst < 1e-3, f"max bone length drift {worst * 100:.4f} %")
    # ik
    reach = R["reach"][seg]
    put("ik", reach.max() < 0.995, f"max IK reach {reach.max():.3f} (arms+legs)")
    # feet
    slide, sink = 0.0, 0.0
    for k, side in enumerate(("n", "f")):
        toe = J[:, idx[f"{side}_toe"]]
        pl = R["planted"][:, k]
        for f in fr:
            if f > 0 and pl[f] and pl[f - 1]:
                slide = max(slide, float(np.hypot(*(toe[f] - toe[f - 1]))))
        sink = max(sink, float((toe[seg, 1] - GROUND).max()))
    put("feet", slide < 0.05 and sink < 0.5, f"planted-foot slide {slide:.4f} px/frame, max sink {sink:.3f} px")
    # jerk (2nd difference of joint positions)
    acc = np.hypot(*(J[2:] - 2 * J[1:-1] + J[:-2]).transpose(2, 0, 1))    # (F-2, J)
    vel = np.hypot(*(J[1:] - J[:-1]).transpose(2, 0, 1))
    jerk_bad = []
    for f in fr:
        if f < 1 or f >= len(J) - 1:
            continue
        a = acc[f - 1]
        lo_, hi_ = max(0, f - 13), min(len(acc), f + 12)
        ref = np.median(acc[lo_:hi_], axis=0) + 0.6
        spikes = np.where((a > 6.0 * ref) & (a > 12.0))[0]
        for j in spikes:
            ev = in_event(f / FPS, pad=0.15)
            item = dict(frame=f, joint=names[j], accel=round(float(a[j]), 1), event=ev)
            if ev:
                res["intentional"].append(dict(check="jerk", **item))
            else:
                jerk_bad.append(item)
    put("jerk", not jerk_bad, f"{len(jerk_bad)} unexplained joint acceleration spikes; "
        f"peak accel {acc[max(fr[0]-1,0):fr[-1]].max():.1f} px/frame^2", spikes=jerk_bad[:20])
    # freeze: pixel diff ~0 while rig moves
    frozen = [f for f in fr if f in diff and diff[f] < 0.0005 and vel[f - 1].max() > 0.5]
    put("freeze", not frozen, f"{len(frozen)} frozen frames while rig moving", frozen=frozen)
    # cloth
    cl = R["cloak"][seg]
    m = R["cloak_mask"]
    finite = bool(np.isfinite(cl).all())
    stretch = 0.0
    for c in range(cl.shape[1]):
        n = int(m[c].sum())
        d = np.hypot(*(cl[:, c, 1:n] - cl[:, c, :n - 1]).transpose(2, 0, 1))
        stretch = max(stretch, float((d / R["seg_l"][c]).max()))
    under = float((cl[..., 1] - GROUND).max())
    put("cloth", finite and stretch < 1.25 and under < 0.5,
        f"finite={finite}, max segment stretch {stretch:.3f}x, max below ground {under:.2f} px")
    # seam continuity with neighbour segments
    seam = []
    for f in (fr[0], fr[-1] + 1):
        if f - 1 in G and f in G and f + 1 in G:
            r = resid.get(f, resid.get(f - 1, 0))
            seam.append(dict(boundary=f, residual=round(r, 4), ratio=round(r / max(base, 1e-4), 2)))
    seam_ok = all(x["ratio"] < 4 or in_event(x["boundary"] / FPS) for x in seam)
    put("seam", seam_ok, ", ".join(f"f{x['boundary']}: {x['ratio']}x median" for x in seam) or "n/a",
        seams=seam)
    # framing: every tracked part (head, horn tips, heart, hands, feet) inside the visible frame
    worst = None
    out_frames = []
    for f in fr:
        pj = os.path.join(PROJ_DIR, f"{f:05d}.json")
        if not os.path.exists(pj):
            out_frames.append(f); continue
        P = json.load(open(pj))
        pts = [P[k] for k in P if k != "horns"] + P["horns"]
        m = min(min(x, W - x, y - LETTERBOX, H - LETTERBOX - y) for x, y in pts)
        if worst is None or m < worst[0]:
            worst = (m, f)
        if m < 0:
            out_frames.append(f)
    put("framing", not out_frames, f"tightest margin {worst[0]:.0f}px at f{worst[1]} "
        f"(head, 7 horn tips, heart, hands, feet); {len(out_frames)} frames out of frame",
        out=out_frames[:20])
    # determinism: re-render one frame from scratch through the 3D renderer
    if rerender:
        import subprocess, tempfile
        f = fr[len(fr) // 2 + 7]
        with tempfile.TemporaryDirectory() as td:
            subprocess.run(["node", os.path.join(ROOT, "r3d", "render3d.js"), "--frames", str(f),
                            "--workers", "1", "--out", td], check=True, capture_output=True)
            a = np.asarray(Image.open(os.path.join(td, f"{f:05d}.png")).convert("RGB")).astype(int)
        b = np.asarray(Image.open(frame_path(f)).convert("RGB")).astype(int)
        d = np.abs(a - b).max(-1)
        n = int((d > 2).sum())
        # parallel software-GPU (SwiftShader) rendering can flip a handful of isolated pixels;
        # anything beyond 10 px (0.001 %) means real state leakage between frames
        put("determinism", n <= 10, f"re-rendered f{f}: {n} px differ (> 2/255) of {W * H}, max diff {int(d.max())}")

    res["ok"] = all(c["ok"] for c in C.values())
    res["diff"] = {int(k): round(v, 5) for k, v in diff.items() if k in fr}
    res["resid"] = {int(k): round(v, 5) for k, v in resid.items() if k in fr}
    contact_sheet(s, res, J, idx)
    return res


def contact_sheet(s, res, J, idx):
    """Thumbnails every 6 frames + motion-spacing chart + temporal residual graph."""
    fr = list(seg_range(s))
    tw, th = 256, 144
    cols, rows = 5, 4
    sheet = Image.new("RGB", (cols * tw, rows * th + 360), (14, 14, 18))
    d = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(FONT, 12)
    for k, f in enumerate(fr[::6]):
        im = Image.open(frame_path(f)).resize((tw, th), Image.BILINEAR)
        x, y = (k % cols) * tw, (k // cols) * th
        sheet.paste(im, (x, y))
        d.text((x + 4, y + 4), f"f{f} {f / FPS:5.2f}s", font=font, fill=(255, 230, 120))
    y0 = rows * th
    # spacing chart: dots per frame along key arcs (animator's spacing check)
    d.rectangle([0, y0, cols * tw, y0 + 240], fill=(8, 8, 12))
    seg = slice(fr[0], fr[-1] + 1)
    pts = np.concatenate([J[seg, idx[n]] for n in ("head_tip", "n_hand_tip", "heart", "n_toe", "f_toe")])
    mn, mx = pts.min(0) - 20, pts.max(0) + 20
    sc = min((cols * tw - 20) / (mx[0] - mn[0]), 220 / (mx[1] - mn[1]))
    for n, col in (("head_tip", (120, 200, 255)), ("n_hand_tip", (255, 120, 80)),
                   ("heart", (255, 200, 80)), ("n_toe", (160, 255, 160)), ("f_toe", (90, 160, 90))):
        P = J[seg, idx[n]]
        q = [(10 + (p[0] - mn[0]) * sc, y0 + 10 + (p[1] - mn[1]) * sc) for p in P]
        d.line(q, fill=tuple(c // 3 for c in col), width=1)
        for x, y in q:
            d.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=col)
        d.text((q[0][0], q[0][1] - 12), n, font=font, fill=col)
    d.text((6, y0 + 224), "motion arcs - one dot per frame (even spacing = smooth, gaps = fast)", font=font,
           fill=(170, 170, 170))
    # residual graph
    g0 = y0 + 250
    d.rectangle([0, g0, cols * tw, g0 + 110], fill=(8, 8, 12))
    rv = [res["resid"].get(f, 0) for f in fr]
    dv = [res["diff"].get(f, 0) for f in fr]
    mxv = max(max(rv + dv), 1e-4)
    for vals, col in ((dv, (90, 120, 200)), (rv, (255, 140, 60))):
        q = [(10 + i * (cols * tw - 20) / (len(fr) - 1), g0 + 95 - v / mxv * 80) for i, v in enumerate(vals)]
        d.line(q, fill=col, width=1)
    for et, label, hw in EVENTS:
        if fr[0] / FPS <= et < (fr[-1] + 1) / FPS:
            x = 10 + (et * FPS - fr[0]) * (cols * tw - 20) / (len(fr) - 1)
            d.line([(x, g0 + 5), (x, g0 + 100)], fill=(200, 60, 200))
            d.text((x + 3, g0 + 4), label, font=font, fill=(200, 120, 200))
    status = "PASS" if res["ok"] else "FAIL"
    d.text((6, g0 + 96), f"blue = frame diff, orange = temporal residual (pop detector)   segment {s} "
           f"{res['name']}  {status}", font=font, fill=(170, 170, 170))
    os.makedirs(QC_DIR, exist_ok=True)
    sheet.save(os.path.join(QC_DIR, f"seg{s}_{res['name'].lower()}.jpg"), quality=88)


def print_result(res):
    print(f"\n=== segment {res['segment']} [{res['name']}] frames {res['frames'][0]}-{res['frames'][1]} "
          f"({res['time'][0]:.0f}-{res['time'][1]:.0f}s): {'PASS' if res['ok'] else 'FAIL'}")
    for k, c in res["checks"].items():
        print(f"  [{'ok' if c['ok'] else 'XX'}] {k:<12} {c['detail']}")
    if res["intentional"]:
        evs = sorted({i['event'] for i in res['intentional']})
        print(f"  (intentional spikes: {len(res['intentional'])} in {evs})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", type=int)
    ap.add_argument("--no-rerender", action="store_true")
    a = ap.parse_args()
    segs = [a.segment] if a.segment is not None else range(NSEG)
    os.makedirs(QC_DIR, exist_ok=True)
    out = []
    for s in segs:
        r = check_segment(s, rerender=not a.no_rerender)
        print_result(r)
        with open(os.path.join(QC_DIR, f"seg{s}.json"), "w") as fh:
            json.dump(r, fh, indent=1)
        out.append(r)
    return all(r["ok"] for r in out)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
