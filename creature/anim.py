"""Animation + rig for THE ASHEN CANTOR.

Everything here is a pure function of time except the cloak, which is a
verlet simulation solved once, in order, by `solve()` and cached to rig.npz so
frames can then be rendered in any order / in parallel and still be identical.

World space: pixels, y grows down, ground at GROUND. The creature faces LEFT.
"""
import math
import numpy as np
from config import FPS, NFRAMES, GROUND, RIG_FILE

PI = math.pi

# --------------------------------------------------------------------------
# Curves
# --------------------------------------------------------------------------
class Curve:
    """Monotone cubic Hermite (Fritsch-Carlson): C1 continuous, never
    overshoots between keys, so holds stay perfectly still and fast moves
    don't wobble past their target."""

    def __init__(self, keys):
        ks = sorted(keys)
        self.t = np.array([k[0] for k in ks], float)
        self.v = np.array([k[1] for k in ks], float)
        n = len(ks)
        self.m = np.zeros(n)
        if n < 2:
            return
        d = np.diff(self.v) / np.diff(self.t)
        m = np.zeros(n)
        m[1:-1] = (d[:-1] + d[1:]) / 2.0
        for i in range(1, n - 1):
            if d[i - 1] * d[i] <= 0:
                m[i] = 0.0
        for i in range(n - 1):
            if d[i] == 0:
                m[i] = m[i + 1] = 0.0
                continue
            a, b = m[i] / d[i], m[i + 1] / d[i]
            s = a * a + b * b
            if s > 9.0:
                tau = 3.0 / math.sqrt(s)
                m[i], m[i + 1] = tau * a * d[i], tau * b * d[i]
        self.m = m

    def __call__(self, t):
        T, V, M = self.t, self.v, self.m
        if t <= T[0]:
            return float(V[0])
        if t >= T[-1]:
            return float(V[-1])
        i = int(np.searchsorted(T, t)) - 1
        h = T[i + 1] - T[i]
        s = (t - T[i]) / h
        s2, s3 = s * s, s * s * s
        return float((2 * s3 - 3 * s2 + 1) * V[i] + (s3 - 2 * s2 + s) * h * M[i]
                     + (-2 * s3 + 3 * s2) * V[i + 1] + (s3 - s2) * h * M[i + 1])


def bump(t, t0, dur):
    """Smooth 0->1->0 pulse (sin^2), C1 continuous at both ends."""
    if t <= t0 or t >= t0 + dur:
        return 0.0
    return math.sin(PI * (t - t0) / dur) ** 2


def smootherstep(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (s * (6 * s - 15) + 10)


def wrap(a):
    return (a + PI) % (2 * PI) - PI


def v(ang, length=1.0):
    return np.array([math.cos(ang) * length, math.sin(ang) * length])


# --------------------------------------------------------------------------
# Skeleton dimensions
# --------------------------------------------------------------------------
SPINE = [58.0, 58.0, 58.0]
NECK = [38.0, 34.0]
HEAD_LEN = 74.0
UPPER, FORE, HAND = 105.0, 118.0, 32.0
FING = [30.0, 26.0, 18.0]           # two phalanges + claw
THIGH, SHIN, META = 100.0, 110.0, 55.0
META_ANG = -PI / 3                  # toe -> ankle (ankle is up/behind the toe)
HIP_OFF = {"n": np.array([-4.0, 0.0]), "f": np.array([8.0, -3.0])}
FOOT_REST = {"n": -40.0, "f": 20.0}  # toe x relative to root when standing

# --------------------------------------------------------------------------
# Keyframes  (time s, value)
# --------------------------------------------------------------------------
C = {}
C["root_x"] = Curve([(0, 830), (5, 830), (8, 826), (10.2, 820), (10.9, 802), (14.0, 612),
                     (14.6, 600), (15.0, 600), (16.2, 612), (16.6, 596), (18.5, 600),
                     (20.0, 600), (20.9, 607), (21.35, 560), (22.2, 555), (24.5, 560),
                     (25.4, 563), (27.4, 640), (30, 642)])
C["root_y"] = Curve([(0, 478), (5.0, 478), (5.6, 484), (7.6, 392), (8.2, 385), (10.2, 388),
                     (14.6, 388), (15.0, 392), (16.2, 372), (16.6, 380), (18.5, 392),
                     (20.0, 392), (20.9, 398), (21.35, 412), (22.3, 402), (24.5, 392),
                     (25.4, 390), (27.4, 388), (30, 390)])
C["lean"] = Curve([(0, 1.05), (5, 1.05), (5.6, 1.1), (7.4, 0.6), (8.6, 0.33), (10.2, 0.4),
                   (11.0, 0.6), (14.6, 0.6), (15.2, 0.5), (16.2, -0.26), (16.6, 0.12),
                   (18.4, 0.15), (19.2, 0.38), (20.0, 0.4), (20.9, 0.05), (21.35, 0.8),
                   (22.2, 0.62), (23.0, 0.55), (23.25, 0.7), (24.5, 0.42), (25.5, 0.3),
                   (27.5, 0.22), (30, 0.2)])
C["curl"] = Curve([(0, 0.26), (5, 0.26), (7.4, 0.16), (8.6, 0.09), (10.2, 0.1), (11, 0.14),
                   (14.6, 0.14), (16.2, -0.09), (16.6, 0.02), (18.4, 0.03), (20, 0.09),
                   (20.9, 0.0), (21.35, 0.16), (24.5, 0.1), (30, 0.07)])
C["neck"] = Curve([(0, 0.7), (5, 0.7), (6.4, 0.75), (8.0, 0.4), (8.8, 0.27), (10.2, 0.45),
                   (11, 0.5), (14.6, 0.5), (15.4, 0.35), (16.2, 0.44), (16.6, 0.55),
                   (18.4, 0.5), (19.2, 0.45), (20, 0.45), (20.9, 0.4), (21.35, 0.55),
                   (24.5, 0.4), (30, 0.3)])
C["head"] = Curve([(0, 0.6), (5, 0.6), (6.8, 0.75), (8.4, 0.85), (8.9, 1.0), (10.2, 0.45),
                   (14.6, 0.43), (15.4, 0.6), (16.2, 0.25), (16.6, 0.85), (18.4, 0.8),
                   (19.2, 0.6), (20, 0.55), (20.9, 0.6), (21.35, 0.55), (24.5, 0.7), (30, 0.85)])
C["jaw"] = Curve([(0, 0), (8.6, 0), (9.0, 0.18), (9.6, 0.05), (10.0, 0), (15.6, 0),
                  (16.2, 0.25), (16.55, 1.0), (18.2, 0.9), (18.9, 0.1), (19.3, 0), (21.1, 0),
                  (21.35, 0.55), (21.9, 0.1), (22.3, 0), (30, 0)])
C["eyes"] = Curve([(0, 0), (3.95, 0), (4.1, 0.65), (4.22, 0.15), (4.4, 0.9), (4.55, 0.45),
                   (4.9, 1.0), (27.6, 1.0), (28.4, 0.55), (29.2, 0.0), (30, 0)])
C["heart"] = Curve([(0, 0.12), (5, 0.16), (8.8, 0.65), (10.2, 0.45), (15.2, 0.5), (16.2, 0.72),
                    (16.55, 1.0), (18.4, 0.9), (19.5, 0.55), (20.9, 0.7), (21.35, 1.0),
                    (22.5, 0.6), (25, 0.5), (28.5, 0.15), (30, 0.08)])
C["breath_amp"] = Curve([(0, 0.6), (5, 0.6), (8, 1.0), (15, 0.8), (18.5, 1.6), (20, 1.0),
                         (25, 1.2), (30, 0.6)])
C["fog_front"] = Curve([(0, 0.18), (5, 0.12), (10, 0.06), (25, 0.06), (28.6, 0.78), (30, 0.86)])
C["title"] = Curve([(0, 0), (27.0, 0), (28.6, 1.0), (30, 1.0)])
C["cam_x"] = Curve([(0, 700), (5, 730), (10, 770), (14.6, 640), (16.6, 625), (19, 630),
                    (21.35, 600), (24, 620), (30, 680)])
C["cam_y"] = Curve([(0, 385), (5, 380), (10, 352), (14.6, 362), (16.6, 350), (19, 358),
                    (21.35, 366), (24, 370), (30, 372)])
C["zoom"] = Curve([(0, 1.0), (5, 1.04), (10, 1.15), (14.6, 1.28), (16.0, 1.24), (16.6, 1.33),
                   (19, 1.26), (21.35, 1.36), (24, 1.3), (30, 1.08)])

# Arms: a = raise angle from hanging (forward +), e = elbow, w = wrist,
# c = finger curl 0..1, s = finger spread, ik = IK weight, ix/iy = IK target
ARM = {
    "n": dict(
        a=Curve([(0, 0.6), (5.8, 0.6), (6.6, 0.5), (8.0, 0.12), (8.5, 0.5), (9.4, 2.25),
                 (9.9, 2.2), (10.5, 0.35), (14.6, 0.35), (15.2, 0.1), (16.2, -0.65),
                 (16.6, 1.05), (18.4, 0.95), (19.2, 0.3), (20.0, 0.4), (20.9, 3.4), (21.18, 3.3),
                 (21.32, 1.0), (21.42, 0.35), (22.0, 0.25), (24.5, 0.3), (30, 0.18)]),
        e=Curve([(0, 0.55), (5.8, 0.55), (8.0, 0.3), (9.4, 0.3), (10.5, 0.35), (15.2, 0.4),
                 (16.2, 1.25), (16.6, 0.25), (18.4, 0.3), (19.2, 0.35), (20.9, 0.65),
                 (21.18, 0.6), (21.32, 0.08), (21.6, 0.2), (22.4, 0.4), (30, 0.4)]),
        w=Curve([(0, 0.1), (8.0, 0.15), (9.4, -0.15), (10.5, 0.2), (16.2, 0.4), (16.6, -0.1),
                 (19.2, 0.2), (20.9, 0.35), (21.32, -0.15), (22.2, 0.2), (30, 0.2)]),
        c=Curve([(0, 0.7), (5.8, 0.7), (8.0, 0.75), (9.4, 0.05), (10.0, 0.1), (10.6, 0.55),
                 (15.2, 0.6), (16.2, 0.85), (16.6, 0.0), (18.4, 0.1), (19.2, 0.55),
                 (20.9, 0.15), (21.32, 0.25), (22.0, 0.7), (30, 0.8)]),
        s=Curve([(0, 0.6), (8.0, 0.5), (9.4, 1.4), (10.4, 0.6), (16.2, 0.4), (16.6, 1.5),
                 (18.4, 1.3), (19.2, 0.6), (20.9, 1.2), (21.32, 1.0), (22.2, 0.6), (30, 0.5)]),
        ik=Curve([(0, 1.0), (5.5, 1.0), (6.4, 0.0), (30, 0.0)]),
        ix=Curve([(0, -150), (30, -150)]), iy=Curve([(0, 572), (5.5, 572), (6.4, 535), (30, 535)]),
    ),
    "f": dict(
        a=Curve([(0, 0.55), (5.8, 0.55), (6.8, 0.45), (8.2, 0.1), (8.7, 0.5), (9.6, 2.0),
                 (10.0, 1.95), (10.7, 0.3), (14.6, 0.3), (15.2, 0.05), (16.2, -0.75),
                 (16.6, 0.8), (18.4, 0.7), (19.2, 0.2), (20.9, -0.2), (21.35, 0.4),
                 (22.4, 0.3), (23.0, -0.75), (23.15, 0.4), (23.3, 2.0), (23.45, 2.3),
                 (24.2, 0.6), (30, 0.15)]),
        e=Curve([(0, 0.5), (5.8, 0.5), (8.2, 0.3), (9.6, 0.35), (10.7, 0.35), (16.2, 1.3),
                 (16.6, 0.3), (18.4, 0.35), (20.9, 0.6), (22.4, 0.5), (23.0, 1.1),
                 (23.3, 0.15), (24.2, 0.45), (30, 0.4)]),
        w=Curve([(0, 0.1), (9.6, -0.2), (10.7, 0.2), (16.2, 0.4), (16.6, -0.1), (19.2, 0.2),
                 (23.0, 0.5), (23.3, -0.2), (24.2, 0.2), (30, 0.2)]),
        c=Curve([(0, 0.7), (5.8, 0.7), (8.2, 0.75), (9.6, 0.05), (10.2, 0.1), (10.8, 0.55),
                 (16.2, 0.85), (16.6, 0.0), (18.4, 0.1), (19.2, 0.55), (23.0, 0.3),
                 (23.3, 0.1), (24.2, 0.7), (30, 0.8)]),
        s=Curve([(0, 0.6), (9.6, 1.3), (10.4, 0.6), (16.2, 0.4), (16.6, 1.4), (18.4, 1.2),
                 (19.2, 0.6), (23.3, 1.3), (24.2, 0.6), (30, 0.5)]),
        ik=Curve([(0, 1.0), (5.7, 1.0), (6.6, 0.0), (30, 0.0)]),
        ix=Curve([(0, -110), (30, -110)]), iy=Curve([(0, 574), (5.7, 574), (6.6, 540), (30, 540)]),
    ),
}
ELBOW_SIGN = -1.0   # IK elbow side for arms (elbows up/back)
KNEE_SIGN = 1.0     # IK knee side for legs (knee forward)

TWITCHES = [(7.1, 0.3, -0.35), (12.35, 0.28, 0.3), (13.4, 0.22, -0.2),
            (19.6, 0.3, 0.25), (26.2, 0.35, -0.25)]

# Camera shake impulses (t0, amplitude px, decay s)
SHAKES = [(16.5, 9.0, 0.9), (21.32, 12.0, 0.45), (23.28, 7.0, 0.4)]

# --------------------------------------------------------------------------
# Footsteps (t0, t1, lift px); target x is derived from where the root will be
# --------------------------------------------------------------------------
_STEP_PLAN = {
    "n": [(10.25, 10.9, 22), (11.65, 12.3, 22), (13.05, 13.7, 22), (20.95, 21.32, 30),
          (26.1, 26.6, 18), (27.05, 27.45, 14)],
    "f": [(10.95, 11.6, 22), (12.35, 13.0, 22), (13.75, 14.45, 22), (25.5, 26.0, 18),
          (26.6, 27.1, 16)],
}


def _build_steps():
    steps = {}
    for side, plan in _STEP_PLAN.items():
        x = C["root_x"](0) + FOOT_REST[side]
        out = []
        for t0, t1, lift in plan:
            nx = C["root_x"](t1 + 0.3) + FOOT_REST[side]
            out.append((t0, t1, lift, x, nx))
            x = nx
        steps[side] = (C["root_x"](0) + FOOT_REST[side], out)
    return steps


STEPS = _build_steps()


def foot(side, t):
    """Returns toe position, swing phase bump (0 planted .. 1 mid-swing), planted flag."""
    x, plan = STEPS[side]
    for t0, t1, lift, x0, x1 in plan:
        if t < t0:
            return np.array([x, GROUND]), 0.0, True
        if t <= t1:
            s = (t - t0) / (t1 - t0)
            b = math.sin(PI * s) ** 2
            return np.array([x0 + (x1 - x0) * smootherstep(s), GROUND - lift * b]), b, False
        x = x1
    return np.array([x, GROUND]), 0.0, True


# --------------------------------------------------------------------------
# Channels (all scalar animation values at time t)
# --------------------------------------------------------------------------
BREATH_PERIOD = 2.6


def channels(t):
    ch = {k: c(t) for k, c in C.items()}
    br = math.sin(2 * PI * t / BREATH_PERIOD) * ch["breath_amp"]
    ch["breath"] = br
    _, bn, _ = foot("n", t)
    _, bf, _ = foot("f", t)
    ch["swing_n"], ch["swing_f"] = bn, bf
    ch["root_y"] += -5.0 * (bn + bf) - 1.2 * br
    ch["curl"] += 0.012 * br
    ch["heart"] = min(1.0, max(0.0, ch["heart"] + 0.05 * br))
    ch["head"] += sum(a * bump(t, t0, d) for t0, d, a in TWITCHES)
    for side in ("n", "f"):
        for k, c in ARM[side].items():
            ch[f"{side}_{k}"] = c(t)
    # arm swing while walking (opposite leg)
    ch["n_a"] += 0.2 * bf - 0.12 * bn
    ch["f_a"] += 0.2 * bn - 0.12 * bf
    sx = sy = 0.0
    for t0, amp, dec in SHAKES:
        if t >= t0:
            env = amp * math.exp(-(t - t0) / dec) * min(1.0, (t - t0) / 0.05)
            sx += env * (math.sin(2 * PI * 7.3 * t + 1.0) + 0.6 * math.sin(2 * PI * 11.1 * t + 2.0))
            sy += env * (math.sin(2 * PI * 6.1 * t + 3.0) + 0.6 * math.sin(2 * PI * 9.7 * t + 4.0))
    ch["shake_x"], ch["shake_y"] = sx, sy
    return ch


# --------------------------------------------------------------------------
# IK
# --------------------------------------------------------------------------
def two_bone(a, target, l1, l2, sign):
    d_vec = target - a
    d = float(np.hypot(*d_vec))
    reach = d / (l1 + l2)
    d = min(max(d, abs(l1 - l2) + 1e-3), l1 + l2 - 1e-3)
    base = math.atan2(d_vec[1], d_vec[0])
    cosA = (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d)
    alpha = math.acos(min(1.0, max(-1.0, cosA)))
    ang1 = base + sign * alpha
    mid = a + v(ang1, l1)
    end = a + v(base, d)
    ang2 = math.atan2(end[1] - mid[1], end[0] - mid[0])
    return ang1, ang2, mid, end, reach


# --------------------------------------------------------------------------
# Rig build
# --------------------------------------------------------------------------
def head_abs(ch):
    chest = -PI / 2 - (ch["lean"] + 3 * ch["curl"])
    return chest - ch["neck"] - ch["head"]


def build(t):
    """Return dict with joints (name -> np.array([x,y])), lists for drawing, and scalars."""
    ch = channels(t)
    J = {}
    root = np.array([ch["root_x"], ch["root_y"]])
    J["hip"] = root
    # spine (angle measured from up, forward = toward -x)
    p = root.copy()
    sp = [p.copy()]
    dirs = []
    for i, L in enumerate(SPINE):
        ang = -PI / 2 - (ch["lean"] + (i + 1) * ch["curl"])
        dirs.append(ang)
        p = p + v(ang, L)
        sp.append(p.copy())
    for i, q in enumerate(sp):
        J[f"spine{i}"] = q
    chest_dir = dirs[-1]
    neck_dir = chest_dir - ch["neck"]
    n1 = sp[-1] + v(neck_dir, NECK[0])
    n2 = n1 + v(neck_dir - ch["neck"] * 0.35, NECK[1])
    J["neck1"], J["neck2"] = n1, n2
    hd = neck_dir - ch["head"]          # head relative to neck base
    J["head_tip"] = n2 + v(hd, HEAD_LEN)

    # back normal (points behind the creature, +x when upright)
    def back_n(ang):
        d = v(ang)
        return np.array([-d[1], d[0]])

    # shoulders
    bn_chest = back_n(chest_dir)
    shoulder = {"n": sp[-1] - bn_chest * 2.0 + v(chest_dir, -6),
                "f": sp[-1] + bn_chest * 10.0 + np.array([0.0, -4.0]) + v(chest_dir, -6)}
    arms = {}
    for side in ("n", "f"):
        S = shoulder[side]
        ua = PI / 2 + ch[f"{side}_a"]
        fa = ua + ch[f"{side}_e"]
        ha = fa + ch[f"{side}_w"]
        w = ch[f"{side}_ik"]
        reach = 0.0
        if w > 0:
            tgt = np.array([ch["root_x"] + ch[f"{side}_ix"], ch[f"{side}_iy"]])
            a1, a2, _, _, reach = two_bone(S, tgt, UPPER, FORE, ELBOW_SIGN)
            ua = ua + wrap(a1 - ua) * w
            fa = fa + wrap(a2 - fa) * w
            ha = ha + wrap((PI * 0.86) - ha) * w
        el = S + v(ua, UPPER)
        wr = el + v(fa, FORE)
        kn = wr + v(ha, HAND)
        curl, spread = ch[f"{side}_c"], ch[f"{side}_s"]
        fingers = []
        for k, off in enumerate((-1.5, -0.5, 0.5, 1.5)):
            a0 = ha + off * 0.16 * spread + curl * 0.55
            pts = [kn]
            q = kn
            angs = [a0, a0 + curl * 0.95, a0 + curl * 1.6 + 0.18]
            for L, ang in zip(FING, angs):
                q = q + v(ang, L * (0.85 if k in (0, 3) else 1.0))
                pts.append(q)
            fingers.append(pts)
        arms[side] = dict(sh=S, el=el, wr=wr, kn=kn, fingers=fingers, reach=reach,
                          ua=ua, fa=fa, ha=ha)
        J[f"{side}_sh"], J[f"{side}_el"], J[f"{side}_wr"], J[f"{side}_kn"] = S, el, wr, kn
        for k, fp in enumerate(fingers):
            J[f"{side}_f{k}_tip"] = fp[-1]
    J["n_hand_tip"] = arms["n"]["fingers"][1][-1]

    # legs
    legs = {}
    for side in ("n", "f"):
        toe, swing, planted = foot(side, t)
        ma = META_ANG + 0.35 * swing
        ankle = toe + v(ma, META)
        hipj = root + HIP_OFF[side]
        a1, a2, knee, ank2, reach = two_bone(hipj, ankle, THIGH, SHIN, KNEE_SIGN)
        legs[side] = dict(hip=hipj, knee=knee, ankle=ank2, toe=toe, reach=reach,
                          planted=planted, ma=ma)
        J[f"{side}_hipj"], J[f"{side}_knee"], J[f"{side}_ankle"], J[f"{side}_toe"] = hipj, knee, ank2, toe

    # heart sits in the ribcage, toward the front
    heart = sp[2] * 0.45 + sp[3] * 0.55 - bn_chest * 8.0
    J["heart"] = heart

    # cloak attach points along the back
    attach = []
    for s in (1.0, 0.9, 0.78, 0.66, 0.54, 0.42, 0.3):
        f = s * 3.0
        i = min(int(f), 2)
        u = f - i
        pt = sp[i] * (1 - u) + sp[i + 1] * u
        bn = back_n(dirs[i])
        width = 14 + 10 * s
        attach.append(pt + bn * width + (v(chest_dir, 8) if s == 1.0 else 0))
    return dict(t=t, ch=ch, J=J, spine=sp, dirs=dirs, neck=[sp[-1], n1, n2], head_dir=hd,
                arms=arms, legs=legs, attach=np.array(attach), chest_dir=chest_dir)


# --------------------------------------------------------------------------
# Bones for QC (rigid lengths that must never change)
# --------------------------------------------------------------------------
BONES = [("spine0", "spine1", SPINE[0]), ("spine1", "spine2", SPINE[1]),
         ("spine2", "spine3", SPINE[2]), ("spine3", "neck1", NECK[0]),
         ("neck1", "neck2", NECK[1]), ("neck2", "head_tip", HEAD_LEN)]
for _s in ("n", "f"):
    BONES += [(f"{_s}_sh", f"{_s}_el", UPPER), (f"{_s}_el", f"{_s}_wr", FORE),
              (f"{_s}_wr", f"{_s}_kn", HAND),
              (f"{_s}_hipj", f"{_s}_knee", THIGH), (f"{_s}_knee", f"{_s}_ankle", SHIN),
              (f"{_s}_toe", f"{_s}_ankle", META)]

# --------------------------------------------------------------------------
# Cloak simulation
# --------------------------------------------------------------------------
CHAIN_LEN = [11, 12, 12, 11, 10, 8, 7]
SEG_L = [24.0, 25.0, 25.0, 24.0, 23.0, 22.0, 20.0]
NCH, PMAX = len(CHAIN_LEN), max(CHAIN_LEN)
SUBSTEPS = 4
GRAV = 1100.0


def wind(t):
    base = 55.0 + 30.0 * math.sin(t * 0.7) + 18.0 * math.sin(t * 1.9 + 1.3)
    blast = 1400.0 * bump(t, 16.5, 0.45) + 500.0 * bump(t, 21.3, 0.25)
    return np.array([base + blast, -0.15 * blast + 8.0 * math.sin(t * 2.3)])


class Cloak:
    def __init__(self, attach):
        self.pos = np.zeros((NCH, PMAX, 2))
        for c in range(NCH):
            for i in range(PMAX):
                self.pos[c, i] = attach[c] + np.array([4.0 * i, SEG_L[c] * i])
        self.prev = self.pos.copy()
        self.mask = np.zeros((NCH, PMAX), bool)
        for c in range(NCH):
            self.mask[c, :CHAIN_LEN[c]] = True

    def step(self, attach, t, dt):
        acc = np.array([0.0, GRAV]) + wind(t)
        # per-particle wind falloff (hem catches more wind)
        k = np.linspace(0.3, 1.0, PMAX)[None, :, None]
        vel = (self.pos - self.prev) * 0.986
        self.prev = self.pos.copy()
        self.pos = self.pos + vel + acc[None, None, :] * k * dt * dt
        for _ in range(10):
            self.pos[:, 0] = attach
            for c in range(NCH):
                n = CHAIN_LEN[c]
                for i in range(1, n):
                    d = self.pos[c, i] - self.pos[c, i - 1]
                    L = math.hypot(d[0], d[1]) + 1e-9
                    corr = d * (1 - SEG_L[c] / L)
                    if i == 1:
                        self.pos[c, i] -= corr
                    else:
                        self.pos[c, i] -= corr * 0.5
                        self.pos[c, i - 1] += corr * 0.5
            # keep neighbouring chains from drifting apart (cloth cohesion)
            for c in range(NCH - 1):
                n = min(CHAIN_LEN[c], CHAIN_LEN[c + 1])
                rest = np.hypot(*(attach[c] - attach[c + 1])) + 6.0
                for i in range(1, n):
                    d = self.pos[c + 1, i] - self.pos[c, i]
                    L = math.hypot(d[0], d[1]) + 1e-9
                    lim = rest * (1.0 + 0.22 * i)
                    if L > lim:
                        corr = d * (1 - lim / L) * 0.5
                        self.pos[c, i] += corr
                        self.pos[c + 1, i] -= corr
            np.minimum(self.pos[:, :, 1], GROUND - 1.0, out=self.pos[:, :, 1])
        self.pos[:, 0] = attach


def solve(path=RIG_FILE, verbose=True):
    names = sorted(build(0.0)["J"].keys())
    joints = np.zeros((NFRAMES, len(names), 2))
    cloak = np.zeros((NFRAMES, NCH, PMAX, 2))
    reach = np.zeros((NFRAMES, 4))
    planted = np.zeros((NFRAMES, 2), bool)
    scal_keys = ["heart", "eyes", "jaw", "zoom", "shake_x", "shake_y", "fog_front"]
    scal = np.zeros((NFRAMES, len(scal_keys)))
    dt = 1.0 / FPS
    r0 = build(0.0)
    cl = Cloak(r0["attach"])
    for _ in range(int(3.0 * FPS * SUBSTEPS)):       # pre-roll so frame 0 is settled
        cl.step(r0["attach"], 0.0, dt / SUBSTEPS)
    prev_attach = r0["attach"]
    for f in range(NFRAMES):
        t = f * dt
        r = build(t)
        for s in range(SUBSTEPS):
            u = (s + 1) / SUBSTEPS
            if f > 0:
                a = prev_attach * (1 - u) + r["attach"] * u
                ts = t - dt + u * dt
            else:
                a, ts = r["attach"], t
            cl.step(a, ts, dt / SUBSTEPS)
        prev_attach = r["attach"]
        cloak[f] = cl.pos
        joints[f] = [r["J"][n] for n in names]
        reach[f] = [r["arms"]["n"]["reach"] if r["ch"]["n_ik"] > 0 else 0,
                    r["arms"]["f"]["reach"] if r["ch"]["f_ik"] > 0 else 0,
                    r["legs"]["n"]["reach"], r["legs"]["f"]["reach"]]
        planted[f] = [r["legs"]["n"]["planted"], r["legs"]["f"]["planted"]]
        scal[f] = [r["ch"][k] for k in scal_keys]
    np.savez_compressed(path, names=np.array(names), joints=joints, cloak=cloak,
                        cloak_mask=cl.mask, reach=reach, planted=planted,
                        scal=scal, scal_keys=np.array(scal_keys), fps=FPS,
                        seg_l=np.array(SEG_L), chain_len=np.array(CHAIN_LEN))
    if verbose:
        print(f"solved {NFRAMES} frames -> {path}")
    return path


if __name__ == "__main__":
    import os
    os.makedirs(os.path.dirname(RIG_FILE), exist_ok=True)
    solve()
