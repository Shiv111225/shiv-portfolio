"""Deterministic frame renderer for THE ASHEN CANTOR.

    python render.py --segment 2          # render frames 240..359
    python render.py --frames 0,119,500   # render specific frames
"""
import argparse
import math
import os
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from config import (W, H, SS, FPS, NFRAMES, GROUND, LETTERBOX, RIG_FILE, FRAMES_DIR,
                    frame_path, seg_range)
import anim
from anim import v, PI

FONT = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
_CACHE = {}


# ---------------------------------------------------------------- helpers
def rng(seed):
    return np.random.default_rng(seed)


def tile_noise(w, h, cells, seed, octaves=4):
    """Tileable fBm value noise in [0,1]."""
    r = rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        cx, cy = cells[0] * 2 ** o, cells[1] * 2 ** o
        g = r.random((cy, cx)).astype(np.float32)
        g = np.tile(g, (3, 3))
        im = Image.fromarray((g * 255).astype(np.uint8)).resize((w * 3, h * 3), Image.BICUBIC)
        a = np.asarray(im, np.float32)[h:2 * h, w:2 * w] / 255.0
        out += a * amp
        tot += amp
        amp *= 0.5
    out /= tot
    return (out - out.min()) / (out.max() - out.min())


def layer_affine(img, Lc, zl):
    a = 1.0 / zl
    return img.transform((W, H), Image.AFFINE,
                         (a, 0, Lc[0] - (W / 2) * a, 0, a, Lc[1] - (H / 2) * a),
                         resample=Image.BILINEAR)


def to_np(img):
    return np.asarray(img, np.float32) / 255.0


# ---------------------------------------------------------------- background assets
LW, LH = 2200, 1000
CAM0 = np.array([700.0, 385.0])


def build_cathedral(seed=7):
    """Far gothic cathedral silhouette + lit windows (premultiplied RGB, alpha, glow)."""
    r = rng(seed)
    col = Image.new("L", (LW, LH), 0)
    d = ImageDraw.Draw(col)
    win = Image.new("RGB", (LW, LH), 0)
    dw = ImageDraw.Draw(win)
    cx, base = 1160, 760

    def spire(x, top, w, body_top):
        d.rectangle([x - w / 2, body_top, x + w / 2, base], fill=255)
        d.polygon([(x - w / 2 - 3, body_top), (x + w / 2 + 3, body_top), (x, top)], fill=255)
        for px in (x - w / 2, x + w / 2):  # pinnacles
            d.polygon([(px - 5, body_top + 6), (px + 5, body_top + 6), (px, body_top - 34)], fill=255)

    def lancet(x, y, w, h, lit):
        c = (255, 178, 92) if lit else (40, 52, 80)
        dw.rectangle([x - w / 2, y, x + w / 2, y + h], fill=c)
        dw.polygon([(x - w / 2, y), (x + w / 2, y), (x, y - w)], fill=c)

    # central facade
    d.rectangle([cx - 140, 330, cx + 140, base], fill=255)
    d.polygon([(cx - 150, 330), (cx + 150, 330), (cx, 210)], fill=255)
    spire(cx, 230, 16, 330)
    for sx, top in ((-170, 60), (170, 88)):
        spire(cx + sx, top, 78, 240)
        for k in range(4):
            lancet(cx + sx, 300 + k * 90, 12, 34, r.random() < 0.55)
    # naves, buttresses and side spires
    for side in (-1, 1):
        for k in range(9):
            x = cx + side * (260 + k * 92 + r.integers(-10, 10))
            h = r.integers(380, 520)
            d.rectangle([x - 42, h, x + 42, base], fill=255)
            d.polygon([(x - 46, h), (x + 46, h), (x, h - r.integers(60, 190))], fill=255)
            d.polygon([(x - 46 * side, h + 60), (x - 120 * side, base - 40), (x - 46 * side, base)], fill=255)
            for wy in range(h + 40, base - 60, 70):
                lancet(x, wy, 10, 30, r.random() < 0.35)
    # rose window glow ring (emissive)
    ry = 305
    for rr, c in ((52, (90, 150, 255)), (44, (170, 210, 255)), (30, (215, 235, 255))):
        dw.ellipse([cx - rr, ry - rr, cx + rr, ry + rr], fill=c)
    for k in range(12):
        a = k * PI / 6
        dw.line([(cx, ry), (cx + 50 * math.cos(a), ry + 50 * math.sin(a))], fill=(20, 28, 48), width=4)
    dw.ellipse([cx - 9, ry - 9, cx + 9, ry + 9], fill=(20, 28, 48))
    alpha = to_np(col)
    # atmospheric colour: lighter toward the foggy base
    yy = np.linspace(0, 1, LH, dtype=np.float32)[:, None]
    tint = np.stack([0.10 + 0.10 * yy, 0.13 + 0.13 * yy, 0.22 + 0.16 * yy], -1) * np.ones((1, LW, 1), np.float32)
    w = to_np(win)
    rgb = tint * alpha[..., None] * (1 - w.max(-1, keepdims=True)) + w * 0.85 * alpha[..., None]
    glow = to_np(win.resize((LW // 4, LH // 4)).filter(ImageFilter.GaussianBlur(10)).resize((LW, LH)))
    return rgb, alpha, glow


def build_near(seed=11):
    r = rng(seed)
    col = Image.new("L", (LW, LH), 0)
    d = ImageDraw.Draw(col)
    for x0, x1 in ((0, 360), (1880, LW)):
        x = x0
        while x < x1:
            w = r.integers(60, 120)
            h = r.integers(140, 380)
            d.rectangle([x, h, x + w, LH], fill=255)
            d.polygon([(x - 4, h), (x + w + 4, h), (x + w / 2, h - r.integers(80, 220))], fill=255)
            x += w - 10
    alpha = to_np(col)
    rgb = np.ones((LH, LW, 3), np.float32) * np.array([0.045, 0.055, 0.09], np.float32) * alpha[..., None]
    return rgb, alpha


def assets():
    if "a" in _CACHE:
        return _CACHE["a"]
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    sky = (np.array([0.03, 0.045, 0.09]) * (1 - yy) + np.array([0.11, 0.15, 0.26]) * yy).astype(np.float32)
    sky = np.broadcast_to(sky, (H, W, 3)).copy()
    stars = np.zeros((H, W), np.float32)
    r = rng(3)
    for _ in range(160):
        x, y = r.integers(0, W), r.integers(0, int(H * 0.45))
        stars[y, x] = r.random() * 0.6
    cat_rgb, cat_a, cat_glow = build_cathedral()
    near_rgb, near_a = build_near()
    to_img = lambda a: Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8))
    fog1 = tile_noise(1024, 512, (6, 3), 21)
    fog2 = tile_noise(1024, 512, (4, 2), 22, octaves=5)
    A = dict(sky=sky, stars=stars,
             cat_rgb=to_img(cat_rgb), cat_a=to_img(cat_a), cat_glow=to_img(cat_glow),
             near_rgb=to_img(near_rgb), near_a=to_img(near_a),
             fog1=Image.fromarray((np.tile(fog1, (2, 2)) * 255).astype(np.uint8)),
             fog2=Image.fromarray((np.tile(fog2, (2, 2)) * 255).astype(np.uint8)),
             font=ImageFont.truetype(FONT, 46), font_s=ImageFont.truetype(FONT, 17))
    yy2 = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    xx2 = np.linspace(-1, 1, W, dtype=np.float32)[None, :]
    A["vignette"] = (1 - 0.55 * np.clip((xx2 ** 2 * 0.8 + (yy2 * 2 - 1) ** 2 * 0.6), 0, 1) ** 1.2)[..., None]
    _CACHE["a"] = A
    return A


def rig():
    if "rig" not in _CACHE:
        with np.load(RIG_FILE) as z:
            _CACHE["rig"] = {k: z[k] for k in z.files}
    return _CACHE["rig"]


def fog_layer(tex, off_x, off_y, scale):
    a = 1.0 / scale
    img = tex.transform((W, H), Image.AFFINE,
                        (a, 0, off_x % 1024, 0, a, 128 + off_y), resample=Image.BILINEAR)
    return to_np(img)


# ---------------------------------------------------------------- creature drawing
BODY = (17, 11, 13, 255)
FAR = (9, 10, 16, 255)
CLOAK = [(13, 12, 19, 255), (19, 15, 23, 255)]
VEIN_BODY = (70, 8, 10, 255)


class Pen:
    """Draws every primitive into the body layer AND the emission layer, so
    later (opaque) shapes correctly occlude earlier glow."""

    def __init__(self, cam):
        self.body = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
        self.emit = Image.new("RGB", (W * SS, H * SS), (0, 0, 0))
        self.db = ImageDraw.Draw(self.body)
        self.de = ImageDraw.Draw(self.emit)
        self.cam = cam

    def P(self, p):
        cx, cy, z, sx, sy = self.cam
        return (((p[0] - cx) * z + W / 2 + sx) * SS, ((p[1] - cy) * z + H / 2 + sy) * SS)

    def S(self, r):
        return r * self.cam[2] * SS

    def poly(self, pts, fill, emit=None):
        q = [self.P(p) for p in pts]
        self.db.polygon(q, fill=fill)
        self.de.polygon(q, fill=emit or (0, 0, 0))

    def limb(self, a, b, wa, wb, fill, emit=None):
        d = b - a
        L = math.hypot(*d) + 1e-9
        n = np.array([-d[1], d[0]]) / L
        self.poly([a + n * wa, b + n * wb, b - n * wb, a - n * wa], fill, emit)
        self.circle(a, wa, fill, emit)
        self.circle(b, wb, fill, emit)

    def circle(self, c, r, fill, emit=None):
        x, y = self.P(c)
        rr = self.S(r)
        self.db.ellipse([x - rr, y - rr, x + rr, y + rr], fill=fill)
        self.de.ellipse([x - rr, y - rr, x + rr, y + rr], fill=emit or (0, 0, 0))

    def glow(self, c, r, color, k=6):
        x, y = self.P(c)
        for i in range(k):
            f = 1 - i / k
            rr = self.S(r) * f
            col = tuple(int(cc * (0.25 + 0.75 * (i + 1) / k)) for cc in color)
            self.de.ellipse([x - rr, y - rr, x + rr, y + rr], fill=col)

    def line(self, pts, width, body=None, emit=None):
        q = [self.P(p) for p in pts]
        wpx = max(1, int(self.S(width)))
        if body:
            self.db.line(q, fill=body, width=wpx, joint="curve")
        if emit:
            self.de.line(q, fill=emit, width=wpx, joint="curve")


def scale_col(c, k):
    return tuple(int(min(255, x * k)) for x in c[:3])


def draw_arm(pen, arm, far, heart, smear=None):
    col = FAR if far else BODY
    if smear:
        for sa in smear:          # smear frames: union of earlier poses of the forearm+hand
            pen.limb(sa["el"], sa["wr"], 7, 5, col)
            pen.limb(sa["wr"], sa["kn"], 5, 4, col)
            for fp in sa["fingers"]:
                pen.line(fp, 3.0, body=col, emit=(0, 0, 0))
    pen.limb(arm["sh"], arm["el"], 11, 8, col)
    pen.limb(arm["el"], arm["wr"], 8, 5, col)
    pen.limb(arm["wr"], arm["kn"], 5, 4.5, col)
    vein = scale_col((255, 40, 20), (0.25 if far else 0.55) * heart)
    mid = (arm["el"] + arm["wr"]) / 2 + v(arm["fa"] + PI / 2, 2.5)
    pen.line([arm["sh"], (arm["sh"] + arm["el"]) / 2 + v(arm["ua"] - PI / 2, 3), arm["el"], mid, arm["wr"]],
             1.4, body=VEIN_BODY, emit=vein)
    for fp in arm["fingers"]:
        for i in range(len(fp) - 1):
            w0 = 3.2 - i * 1.0
            pen.limb(fp[i], fp[i + 1], max(w0, 0.6), max(w0 - 1.0, 0.35), col)
        pen.line(fp[-2:], 1.0, emit=scale_col((255, 90, 40), 0.35 * heart))


def draw_leg(pen, leg, far):
    col = FAR if far else BODY
    pen.limb(leg["hip"], leg["knee"], 16, 10, col)
    pen.limb(leg["knee"], leg["ankle"], 9, 6, col)
    pen.limb(leg["ankle"], leg["toe"], 6, 4, col)
    for k, a in enumerate((PI - 0.25, PI + 0.05, PI + 0.35)):
        tip = leg["toe"] + v(a + leg["ma"] - anim.META_ANG, 16 - 3 * k)
        pen.poly([leg["toe"] + np.array([0, -3]), tip, leg["toe"] + np.array([3, 2])], col)


def draw_head(pen, r, ch, horn_lag):
    n2 = r["neck"][2]
    hd = r["head_dir"]
    fwd = v(hd)
    up = np.array([-fwd[1], fwd[0]])
    if up[1] > 0 and abs(fwd[0]) > 0.2:
        up = -up                          # keep the skull crown on top in profile

    def L(u, w):
        return n2 + fwd * u + up * w

    skull = [L(-10, 4), L(-2, 22), L(18, 28), L(42, 20), L(64, 9), L(76, 2), L(72, -4),
             L(46, -8), L(22, -10), L(2, -16), L(-12, -6)]
    jaw_a = -ch["jaw"] * 0.85
    hinge = L(12, -8)

    def J(u, w):
        p = L(u, w) - hinge
        c, s = math.cos(jaw_a), math.sin(jaw_a)
        # rotate in the (fwd, up) frame so it always opens downward
        pu, pw = p @ fwd, p @ up
        return hinge + fwd * (pu * c - pw * s) + up * (pu * s + pw * c)

    jaw = [J(12, -8), J(40, -12), J(66, -8), J(70, -4), J(44, -6), J(18, -4)]
    heart = ch["heart"]
    if ch["jaw"] > 0.02:   # inner mouth fire
        pen.poly([L(14, -6), L(70, -4), J(70, -4), J(14, -6)], BODY,
                 emit=scale_col((255, 70, 20), (0.25 + 0.4 * heart) * min(1, ch["jaw"] * 2)))
    # horns: a crown of five spires, lagging behind head motion
    for k, (u, ang, ln) in enumerate(((-6, 0.35, 82), (4, 0.62, 98), (14, 0.86, 106),
                                       (24, 1.08, 88), (34, 1.3, 66))):
        base = L(u, 20 if u < 30 else 16)
        # direction rotated from 'back' toward 'up'
        bvec = -fwd * math.cos(ang) + up * math.sin(ang)
        cur = math.atan2(bvec[1], bvec[0])
        pts = [base]
        p = base
        crs = (-fwd[0]) * up[1] - (-fwd[1]) * up[0]
        bend = -0.06 * (1 if crs > 0 else -1)   # curl further back along the spire
        for i in range(6):
            cur += bend + horn_lag * 0.18 * (i + 1) / 6 * (1.2 - 0.1 * k)
            p = p + v(cur, ln / 6)
            pts.append(p)
        for i in range(6):
            w0 = 4.2 * (1 - i / 6) + 0.4
            pen.limb(pts[i], pts[i + 1], w0, max(0.35, w0 - 0.7), BODY)
        pen.circle(pts[-1], 1.6, BODY, emit=scale_col((255, 110, 60), 0.5 * heart + 0.15))
    pen.poly(skull, BODY)
    pen.poly(jaw, BODY)
    # teeth
    for u in range(20, 68, 8):
        pen.poly([L(u, -8), L(u + 4, -8), L(u + 2, -15 + ch["jaw"] * 0)], BODY)
    # crimson brow stroke + eye
    pen.line([L(26, 14), L(44, 12), L(58, 7)], 1.3, body=VEIN_BODY,
             emit=scale_col((255, 30, 20), 0.35 * heart))
    e = ch["eyes"]
    if e > 0.01:
        pen.poly([L(42, 8), L(52, 11), L(60, 6), L(50, 5)], BODY,
                 emit=scale_col((255, 120, 40), e))
        pen.glow(L(51, 8), 5, scale_col((255, 70, 20), e * 0.8), k=3)


def draw_creature(pen, r, cloak, smear):
    ch = r["ch"]
    heart = ch["heart"]
    sp = r["spine"]
    # cloak (behind everything)
    for c in range(len(cloak) - 1):
        a, b = cloak[c], cloak[c + 1]
        n = min(len(a), len(b))
        pts = list(a[:n]) + [a[-1] * 0.5 + b[-1] * 0.5 + np.array([0, 6])] + list(b[:n][::-1])
        pen.poly(pts, CLOAK[c % 2])
    for c in range(len(cloak)):
        ch_pts = cloak[c]
        tip = ch_pts[-1] + (ch_pts[-1] - ch_pts[-2]) * 0.9
        pen.poly([ch_pts[-2] + np.array([-5, 0]), tip, ch_pts[-2] + np.array([5, 0])], CLOAK[c % 2])
    draw_leg(pen, r["legs"]["f"], True)
    draw_arm(pen, r["arms"]["f"], True, heart, smear.get("f"))
    # torso
    widths_back = [26, 20, 30, 34]
    widths_front = [22, 16, 30, 36]
    back, front = [], []
    for i, p in enumerate(sp):
        ang = r["dirs"][min(i, 2)]
        d = v(ang)
        bn = np.array([-d[1], d[0]])
        back.append(p + bn * widths_back[i])
        front.append(p - bn * widths_front[i])
    top = sp[-1] + v(r["chest_dir"], 14)
    pen.poly(back + [top] + front[::-1], BODY)
    pen.circle(sp[0], 26, BODY)
    # dorsal spikes along the back
    for i in range(3):
        for u in (0.3, 0.75):
            p = back[i] * (1 - u) + back[i + 1] * u
            d = v(r["dirs"][i])
            bn = np.array([-d[1], d[0]])
            pen.poly([p - d * 6, p + bn * 13 - d * 4, p + d * 5], BODY)
    # ember heart through the ribs
    hp = r["J"]["heart"]
    pen.glow(hp, 26 + 10 * heart, scale_col((255, 70, 20), 0.35 + 0.65 * heart), k=7)
    pen.glow(hp, 9 + 4 * heart, scale_col((255, 210, 150), 0.3 + 0.7 * heart), k=4)
    # ribs (occlude glow, leaving gaps)
    d = v(r["chest_dir"])
    bn = np.array([-d[1], d[0]])
    for k in range(5):
        u = 0.15 + k * 0.19
        p = sp[2] * (1 - u) + sp[3] * u
        a = p + bn * 22
        b = p - bn * 30 - d * 10
        mid = p - bn * 8 + d * 6
        pen.line([a, mid, b], 4.0, body=BODY, emit=(0, 0, 0))
    pen.line([sp[2] - bn * 26, sp[3] - bn * 34], 3.0, body=BODY, emit=(0, 0, 0))
    # spine vein
    pen.line([sp[0], sp[1], sp[2]], 1.4, body=VEIN_BODY, emit=scale_col((255, 40, 20), 0.4 * heart))
    # neck + head
    nk = r["neck"]
    pen.limb(nk[0], nk[1], 14, 10, BODY)
    pen.limb(nk[1], nk[2], 10, 9, BODY)
    draw_head(pen, r, ch, smear["horn_lag"])
    draw_leg(pen, r["legs"]["n"], False)
    draw_arm(pen, r["arms"]["n"], False, heart, smear.get("n"))


def smear_poses(t, side, r):
    """Sub-frame samples for fast-moving hands (classic 2D smear frames)."""
    dt = 1.0 / FPS
    prev = anim.build(t - dt)["arms"][side]["kn"]
    speed = float(np.hypot(*(r["arms"][side]["kn"] - prev)))
    if speed < 28:
        return None, speed
    n = int(min(7, speed / 14))
    return [anim.build(t - dt * k / n * 1.1)["arms"][side] for k in range(1, n + 1)], speed


# ---------------------------------------------------------------- FX
def embers(t, cam, emit_np):
    """Closed-form particles (position is a function of t, so frames are order-independent)."""
    r = rng(99)
    n = 170
    spawn = r.random(n) * 34 - 4
    life = 3 + r.random(n) * 4
    x0 = r.random(n) * 1500 - 50
    vx = 10 + r.random(n) * 25
    vy = 25 + r.random(n) * 45
    ph = r.random(n) * 6.28
    age = (t - spawn) % 40
    alive = age < life
    xs = x0 + vx * age + 14 * np.sin(age * 1.3 + ph)
    ys = GROUND + 20 - vy * age
    a = np.sin(np.clip(age / life, 0, 1) * PI) * alive
    # burst at the roar + sparks on the slash
    pts = [(xs, ys, a * 0.7, 1.0)]
    for t0, cnt, cx, cy, spd, seed in ((16.55, 110, None, None, 420, 5), (21.32, 60, None, None, 520, 6)):
        if t >= t0:
            rr = rng(seed)
            ang = rr.random(cnt) * 2 * PI
            sp = spd * (0.3 + rr.random(cnt))
            lifeb = 0.8 + rr.random(cnt) * 1.6
            age_b = t - t0
            k = 2.2
            dist = sp * (1 - math.exp(-k * age_b)) / k
            origin = _CACHE.get(("burst", t0))
            if origin is None:
                b = anim.build(t0)
                origin = b["J"]["heart"] if t0 < 20 else b["J"]["n_hand_tip"]
                _CACHE[("burst", t0)] = origin
            bx = origin[0] + np.cos(ang) * dist
            by = origin[1] + np.sin(ang) * dist - 30 * age_b * age_b
            ab = np.clip(1 - age_b / lifeb, 0, 1) ** 1.5
            pts.append((bx, by, ab, 1.6))
    cx, cy, z, sx, sy = cam
    h, w = emit_np.shape[:2]
    for xs, ys, a, size in pts:
        X = ((xs - cx) * z + W / 2 + sx).astype(int)
        Y = ((ys - cy) * z + H / 2 + sy).astype(int)
        m = (a > 0.01) & (X >= 1) & (X < w - 1) & (Y >= 1) & (Y < h - 1)
        col = np.array([1.0, 0.45, 0.15], np.float32)
        for x, y, aa in zip(X[m], Y[m], a[m]):
            emit_np[y, x] += col * aa * 1.6 * size
            emit_np[y - 1:y + 2, x - 1:x + 2] += col * aa * 0.25 * size


def slash_marks(t, emit_rgb):
    """Screen-space claw streaks after the strike."""
    out = []
    for t0, base_y, tilt, flip in ((21.32, 330, 0.28, 1), (23.28, 300, -0.22, -1)):
        age = t - t0
        if 0 <= age < 0.9:
            grow = min(1.0, age / 0.06)
            fade = (1 - age / 0.9) ** 2
            d = ImageDraw.Draw(emit_rgb)
            for k in range(3):
                pts = []
                for i in range(int(22 * grow) + 2):
                    u = i / 22
                    x = (220 + 820 * u) if flip > 0 else (1060 - 820 * u)
                    y = base_y + k * 38 + tilt * (x - 640) + 60 * math.sin(u * PI) * flip
                    pts.append((x * SS, y * SS))
                w = max(1, int((6 - k) * SS * fade + 1))
                d.line(pts, fill=scale_col((255, 120, 80), fade), width=w)
                d.line(pts, fill=scale_col((255, 240, 220), fade), width=max(1, w // 3))
    return out


# ---------------------------------------------------------------- frame
def render_frame(f):
    A = assets()
    R = rig()
    t = f / FPS
    r = anim.build(t)
    ch = r["ch"]
    z = ch["zoom"]
    cam = (ch["cam_x"], ch["cam_y"], z, ch["shake_x"], ch["shake_y"])
    camv = np.array([cam[0], cam[1]])

    img = A["sky"].copy()
    img += A["stars"][..., None] * np.array([0.7, 0.8, 1.0], np.float32) * 0.7

    # far cathedral (parallax 0.18)
    p = 0.18
    Lc = np.array([1160, 470]) + (camv - CAM0) * p
    zl = 1 + (z - 1) * p
    Lc = Lc + np.array([cam[3], cam[4]]) * -p
    ca = to_np(layer_affine(A["cat_a"], Lc, zl))[..., None]
    crgb = to_np(layer_affine(A["cat_rgb"], Lc, zl))
    cglow = to_np(layer_affine(A["cat_glow"], Lc, zl))
    pulse = 0.85 + 0.15 * math.sin(t * 1.7)
    img = img * (1 - ca) + crgb * pulse
    img += cglow * 0.55 * pulse

    # far fog
    f1 = fog_layer(A["fog1"], t * 14 + cam[0] * 0.25, -cam[1] * 0.1, 1.6)
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    band = np.clip((yy - 0.28) * 2.2, 0, 1) * (1 - np.clip((yy - 0.95) * 4, 0, 1) * 0.3)
    fa = (f1 * 0.75 * band)[..., None]
    img = img * (1 - fa * 0.8) + np.array([0.24, 0.32, 0.52], np.float32) * fa * 0.8

    # blue sorcery ribbons
    rib = Image.new("L", (W // 4, H // 4), 0)
    dr = ImageDraw.Draw(rib)
    for k, (yb, amp, sp) in enumerate(((95, 14, 0.35), (72, 10, -0.22))):
        pts = [(x, yb + amp * math.sin(x * 0.03 + t * sp * 3 + k) + 8 * math.sin(x * 0.011 - t * 0.4))
               for x in range(0, W // 4 + 4, 4)]
        dr.line(pts, fill=255, width=3)
    rib = to_np(rib.filter(ImageFilter.GaussianBlur(3)).resize((W, H), Image.BILINEAR))
    rib *= 0.6 + 0.4 * fog_layer(A["fog2"], t * 40, 0, 1.0)
    img += rib[..., None] * np.array([0.35, 0.6, 1.0], np.float32) * 0.45

    # near buildings (parallax 0.45)
    p = 0.45
    Lc = np.array([1100, 470]) + (camv - CAM0) * p - np.array([cam[3], cam[4]]) * p
    zl = 1 + (z - 1) * p
    na = to_np(layer_affine(A["near_a"], Lc, zl))[..., None]
    img = img * (1 - na) + to_np(layer_affine(A["near_rgb"], Lc, zl))

    # ground plane (world space)
    gy = (GROUND - cam[1]) * z + H / 2 + cam[4]
    ys = np.arange(H, dtype=np.float32)[:, None]
    g = np.clip((ys - gy + 6) / 12, 0, 1)
    depth = np.clip((ys - gy) / (H - gy + 1), 0, 1)
    gcol = np.array([0.05, 0.06, 0.1], np.float32) * (1 - depth * 0.6)[..., None]
    img = img * (1 - g[..., None]) + gcol * g[..., None]

    # creature
    pen = Pen(cam)
    cl = R["cloak"][f]
    mask = R["cloak_mask"]
    cloak = [cl[c][mask[c]] for c in range(cl.shape[0])]
    smear = {}
    for side in ("n", "f"):
        s, _ = smear_poses(t, side, r)
        smear[side] = s
    smear["horn_lag"] = wrap_lag(t)
    # contact shadow
    sh = Image.new("L", (W, H), 0)
    ds = ImageDraw.Draw(sh)
    for side in ("n", "f"):
        toe = r["legs"][side]["toe"]
        x, y = pen.P(toe)
        hgt = max(0.0, GROUND - toe[1])
        rr = (34 - hgt * 0.4) * z
        ds.ellipse([x / SS - rr, y / SS - rr * 0.22, x / SS + rr, y / SS + rr * 0.22],
                   fill=int(150 * max(0.2, 1 - hgt / 40)))
    sh = to_np(sh.filter(ImageFilter.GaussianBlur(6)))[..., None]
    img *= 1 - sh * 0.7

    draw_creature(pen, r, cloak, smear)
    body = pen.body.reduce(SS)
    emit_img = pen.emit
    slash_marks(t, emit_img)
    emit = to_np(emit_img.reduce(SS))
    b = to_np(body)
    ba = b[..., 3:4]
    brgb = b[..., :3]
    # rim light from the rose window (behind, upper right) + heart underglow
    m = ba[..., 0]
    shifted = np.roll(np.roll(m, -3, axis=1), 3, axis=0)   # sample toward the light
    rim = np.clip(m - shifted, 0, 1)
    rim = to_np(Image.fromarray((rim * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2)))
    rim = np.minimum(rim, m)
    hx, hy = pen.P(r["J"]["heart"])
    xx = np.arange(W, dtype=np.float32)[None, :]
    under = np.exp(-(((xx - hx / SS) ** 2 + (ys - hy / SS) ** 2) / (2 * (65 * z) ** 2)))
    brgb = brgb + rim[..., None] * np.array([0.42, 0.58, 0.95], np.float32) * 0.85 * pulse
    brgb = brgb + (under * m)[..., None] * np.array([0.5, 0.06, 0.03], np.float32) * (0.05 + 0.3 * ch["heart"])
    img = img * (1 - ba) + brgb * ba

    # front fog (thickens at the end to swallow the creature)
    f2 = fog_layer(A["fog2"], t * 26 + cam[0] * 0.6, -cam[1] * 0.2, 1.2)
    ff = ch["fog_front"]
    lowband = np.clip((yy - 0.45) * 2.0, 0, 1)
    dens = np.clip(f2 * (0.35 + ff) * (0.35 + 0.65 * lowband) + ff * 0.55, 0, 1)[..., None]
    img = img * (1 - dens * 0.85) + np.array([0.2, 0.27, 0.44], np.float32) * dens * 0.85

    # emission + embers + bloom (emission sits above the fog for the eyes, dimmed by fog)
    embers(t, cam, emit)
    emit *= (1 - dens * 0.55)
    ei = Image.fromarray(np.clip(emit * 255, 0, 255).astype(np.uint8))
    small = ei.resize((W // 4, H // 4), Image.BILINEAR)
    b1 = to_np(small.filter(ImageFilter.GaussianBlur(3)).resize((W, H), Image.BILINEAR))
    b2 = to_np(small.filter(ImageFilter.GaussianBlur(12)).resize((W, H), Image.BILINEAR))
    img += emit * 0.9 + b1 * 1.1 + b2 * 1.4

    # roar shockwave ring + flash
    age = t - 16.55
    if 0 <= age < 0.8:
        rad = 40 + age * 900
        rx, ry = hx / SS, hy / SS
        dist = np.sqrt((xx - rx) ** 2 + (ys - ry) ** 2)
        ring = np.exp(-((dist - rad) ** 2) / (2 * 14 ** 2)) * (1 - age / 0.8) ** 2
        img += ring[..., None] * np.array([1.0, 0.35, 0.15], np.float32) * 0.5
        img += np.array([0.5, 0.08, 0.03], np.float32) * math.exp(-age / 0.12) * 0.35

    # foreground: balustrade (left) and hooded statue (right), parallax 1.3
    fg = Image.new("L", (W, H), 0)
    dfg = ImageDraw.Draw(fg)
    ox = -(cam[0] - CAM0[0]) * 1.3 * z + cam[3] * 1.3
    oy = -(cam[1] - CAM0[1]) * 1.3 * z + cam[4] * 1.3
    sc = 1 + (z - 1) * 1.3
    def F(x, y):
        return ((x - 640) * sc + 640 + ox, (y - 360) * sc + 360 + oy)
    dfg.polygon([F(-200, 610), F(330, 625), F(330, 640), F(-200, 628)], fill=255)
    dfg.polygon([F(-200, 700), F(330, 700), F(330, 800), F(-200, 800)], fill=255)
    for k in range(7):
        x = -150 + k * 70
        dfg.ellipse([F(x - 14, 640), F(x + 14, 672)], fill=255)
        dfg.rectangle([F(x - 8, 636), F(x + 8, 702)], fill=255)
    dfg.rectangle([F(300, 560), F(350, 800)], fill=255)
    dfg.polygon([F(1190, 800), F(1210, 360), F(1240, 250), F(1290, 205), F(1340, 225),
                 F(1370, 300), F(1400, 800)], fill=255)
    dfg.polygon([F(1205, 430), F(1150, 520), F(1170, 540), F(1215, 480)], fill=255)
    fgm = to_np(fg)[..., None]
    fg_rim = np.clip(fgm[..., 0] - np.roll(np.roll(fgm[..., 0], -2, 1), 2, 0), 0, 1)[..., None]
    img = img * (1 - fgm) + (np.array([0.012, 0.014, 0.022], np.float32) + fg_rim * np.array([0.2, 0.28, 0.45], np.float32)) * fgm

    # grade: cool shadows, warm highlights, vignette, grain
    img = np.clip(img, 0, None)
    img = img / (1 + img * 0.35)                 # soft shoulder
    img = img * A["vignette"]
    lum = img.mean(-1, keepdims=True)
    img = img + (0.012 - lum * 0.03) * np.array([-0.3, 0.1, 0.6], np.float32)
    grain = rng(1000 + f).normal(0, 0.012, (H // 2, W // 2)).astype(np.float32)
    grain = np.repeat(np.repeat(grain, 2, 0), 2, 1)[..., None]
    img = img + grain * (0.5 + 0.5 * (1 - lum))
    assert np.isfinite(img).all(), f"non-finite pixels in frame {f}"
    out = Image.fromarray(np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8))

    # title + letterbox
    d = ImageDraw.Draw(out)
    tl = ch["title"]
    if tl > 0.001:
        txt = "T H E   A S H E N   C A N T O R"
        tw = d.textlength(txt, font=A["font"])
        c = int(225 * tl)
        d.text(((W - tw) / 2, H / 2 - 40), txt, font=A["font"], fill=(c, int(c * 0.93), int(c * 0.85)))
        sub = "creature concept  —  30 s animation test"
        sw = d.textlength(sub, font=A["font_s"])
        c2 = int(150 * max(0.0, tl * 1.4 - 0.4))
        d.text(((W - sw) / 2, H / 2 + 22), sub, font=A["font_s"], fill=(c2, int(c2 * 0.85), int(c2 * 0.8)))
        lw = int(180 * tl)
        d.line([(W / 2 - lw, H / 2 + 12), (W / 2 + lw, H / 2 + 12)], fill=(int(160 * tl), 40, 30), width=1)
    d.rectangle([0, 0, W, LETTERBOX - 1], fill=(0, 0, 0))
    d.rectangle([0, H - LETTERBOX, W, H], fill=(0, 0, 0))
    return out


def wrap_lag(t):
    a0 = anim.head_abs(anim.channels(t))
    a1 = anim.head_abs(anim.channels(t - 0.12))
    return anim.wrap(a1 - a0) * 1.5


def _job(f):
    img = render_frame(f)
    img.save(frame_path(f), compress_level=1)
    return f


def render_frames(frames, procs=4):
    os.makedirs(FRAMES_DIR, exist_ok=True)
    with Pool(procs) as p:
        for _ in p.imap_unordered(_job, frames, chunksize=4):
            pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", type=int)
    ap.add_argument("--frames")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    if a.segment is not None:
        fr = list(seg_range(a.segment))
    else:
        fr = [int(x) for x in a.frames.split(",")]
    render_frames(fr, a.procs)
