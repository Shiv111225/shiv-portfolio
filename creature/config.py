"""Shared settings for the Ashen Cantor creature animation pipeline."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out")
FRAMES_DIR = os.path.join(OUT, "frames")      # PNG sequence (git-ignored)
QC_DIR = os.path.join(OUT, "qc")              # per-segment QC sheets + reports
RIG_FILE = os.path.join(OUT, "rig.npz")       # solved rig + cloth simulation
VIDEO = os.path.join(OUT, "ashen_cantor.mp4")

W, H = 1280, 720          # output resolution
SS = 2                    # supersampling factor for vector layers
FPS = 24
DURATION = 30.0
NFRAMES = int(round(FPS * DURATION))          # 720
SEG_SECONDS = 5.0
SEG_FRAMES = int(round(FPS * SEG_SECONDS))    # 120
NSEG = NFRAMES // SEG_FRAMES                  # 6

GROUND = 600.0            # world-space ground line (y grows downward)
LETTERBOX = 40            # px black bars top/bottom (2.0:1 framing)

SEGMENTS = [
    ("Dormant",   "Hunched in the courtyard mist; the ember heart barely breathes; eyes ignite."),
    ("Awakening", "Pushes off the stones, rises to full height and raises its arms like a choir-master."),
    ("Stalk",     "Three predatory steps toward camera; head low, a nervous twitch."),
    ("Roar",      "Rears back, then roars - heart flare, shockwave, ember burst, camera shake."),
    ("Strike",    "Lunging overhead claw slash with smear frames, then a far-arm backhand."),
    ("Recede",    "Steps back into the thickening fog; the eyes are the last thing to go. Title."),
]

# Intentional high-energy moments. QC still measures them, but spikes inside
# these windows are reported as "intentional" instead of failing the segment.
EVENTS = [
    # (time_s, label, half_window_s)
    (4.30, "eyes ignite (flicker)", 0.45),
    (16.55, "roar flare + shockwave + shake", 0.55),
    (21.30, "claw slash smear + impact shake", 0.45),
    (23.20, "backhand smear + shake", 0.40),
    (27.00, "title fade in", 0.05),
]


def frame_path(i):
    return os.path.join(FRAMES_DIR, f"{i:05d}.png")


def seg_range(s):
    return range(s * SEG_FRAMES, (s + 1) * SEG_FRAMES)


def in_event(t, pad=0.0):
    for et, label, hw in EVENTS:
        if abs(t - et) <= hw + pad:
            return label
    return None
