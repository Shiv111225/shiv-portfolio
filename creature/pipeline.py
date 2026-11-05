"""Full pipeline: solve rig -> for each 5 s segment: render + QC (stop on failure)
-> encode MP4 -> verify the encode (frame count, duration, PSNR vs. source frames).

    python pipeline.py                # everything
    python pipeline.py --from-seg 3   # resume rendering at segment 3
"""
import argparse
import json
import os
import re
import subprocess
import time

from config import FPS, NFRAMES, NSEG, QC_DIR, VIDEO, FRAMES_DIR
import anim
import qc


def encode():
    cmd = ["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", os.path.join(FRAMES_DIR, "%05d.png"),
           "-frames:v", str(NFRAMES), "-c:v", "libx264", "-preset", "slow", "-crf", "16",
           "-pix_fmt", "yuv420p", "-tune", "grain", "-movflags", "+faststart", VIDEO]
    subprocess.run(cmd, check=True)


def verify_encode():
    pr = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                         "-show_entries", "stream=nb_read_frames,r_frame_rate,width,height:format=duration",
                         "-of", "json", VIDEO], capture_output=True, text=True, check=True)
    info = json.loads(pr.stdout)
    st = info["streams"][0]
    n = int(st["nb_read_frames"])
    dur = float(info["format"]["duration"])
    ps = subprocess.run(["ffmpeg", "-i", VIDEO, "-framerate", str(FPS), "-i",
                         os.path.join(FRAMES_DIR, "%05d.png"), "-lavfi", "[0:v][1:v]psnr", "-f", "null", "-"],
                        capture_output=True, text=True)
    m = re.search(r"average:([\d.]+|inf).*min:([\d.]+|inf)", ps.stderr)
    avg, mn = (float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0)
    ok = n == NFRAMES and abs(dur - NFRAMES / FPS) < 0.05 and mn > 32
    r = dict(ok=ok, frames=n, duration=dur, fps=st["r_frame_rate"], size=[st["width"], st["height"]],
             psnr_avg=avg, psnr_min=mn, bytes=os.path.getsize(VIDEO))
    print(f"\n=== encode: {'PASS' if ok else 'FAIL'}  {n} frames, {dur:.3f}s @ {st['r_frame_rate']}, "
          f"PSNR avg {avg:.1f} dB / min {mn:.1f} dB vs source PNGs, {r['bytes'] / 1e6:.1f} MB")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-seg", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(QC_DIR, exist_ok=True)
    t0 = time.time()
    anim.solve()
    results = []
    for s in range(NSEG):
        if s >= a.from_seg:
            ts = time.time()
            subprocess.run(["node", os.path.join(os.path.dirname(os.path.abspath(__file__)), "r3d", "render3d.js"),
                            "--frames", f"{s * 120}-{(s + 1) * 120 - 1}", "--workers", "3"], check=True)
            print(f"rendered segment {s} in {time.time() - ts:.1f}s")
        # QC the previous segment again too, now that its right-hand seam frame exists
        for q in ([s - 1] if s > 0 else []) + [s]:
            r = qc.check_segment(q, rerender=(q == s))
            if q == s or not r["ok"]:
                qc.print_result(r)
            with open(os.path.join(QC_DIR, f"seg{q}.json"), "w") as fh:
                json.dump(r, fh, indent=1)
            if not r["ok"]:
                print(f"\nSTOP: segment {q} failed QC - fix and rerun with --from-seg {q}")
                raise SystemExit(1)
        results.append(r)
    encode()
    enc = verify_encode()
    summary = dict(segments=[dict(segment=r["segment"], name=r["name"], ok=r["ok"],
                                  checks={k: c["detail"] for k, c in r["checks"].items()},
                                  intentional=len(r["intentional"])) for r in results],
                   encode=enc, ok=all(r["ok"] for r in results) and enc["ok"])
    with open(os.path.join(QC_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\nALL {'PASS' if summary['ok'] else 'FAIL'} in {time.time() - t0:.0f}s -> {VIDEO}")
    return summary["ok"]


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
