# The Ashen Cantor — 30 s 3D creature animation + render QC

Original creature concept: a gaunt gothic choir-beast with a crown of seven spire-horns, charred ash
skin with ember cracks, and a heart burning inside an open ribcage — in a moonlit cathedral courtyard.

```bash
(cd r3d && npm install)         # three.js (renders headless via the preinstalled Chromium)
python pipeline.py              # solve -> render + QC each 5 s segment -> encode -> verify
python pipeline.py --from-seg 3 # resume from a segment
python qc.py --segment 4        # re-check one segment
```

## How it is built
| file | role |
|---|---|
| `anim.py` | keyframes (monotone cubic), FK/IK rig, footstep planner, spring secondary motion, verlet cloak, auto-framing camera → `out/rig.npz`, `out/anim.bin` |
| `r3d/scene.js` | three.js renderer: PBR charred skin (bump + emissive cracks), hollow ribcage with a light-emitting heart, cloth drape, moon + brazier shadows, fog, sky dome, bloom, ACES, 3-sample motion blur |
| `r3d/render3d.js` | headless Chromium driver; writes PNG frames + per-frame screen positions of key parts |
| `qc.py` | per-segment checks + contact sheet + JSON report |
| `pipeline.py` | runs everything, stops at the first failing segment, verifies the encode |
| `render.py` | earlier 2D renderer (kept for quick previews) |

## Animation principles applied
* **Overlap / follow-through** – damped springs on spine, neck, head, elbows, wrists, fingers, jaw: each part trails its parent and settles; springs stiffen during hits so strikes stay sharp.
* **Weight** – during walks the pelvis is carried by the planted feet (surges on each step, no skating), hip rises over the passing leg, head counter-stabilises.
* **Anticipation** – dip before rising, rear-back before the roar, overhead wind-up before the slash.
* **Arcs / spacing** – checked on the QC motion-arc charts (one dot per frame).
* **Staging** – camera is keyed for intent, then clamped/zoomed per frame so the whole creature (horn crown and cloak included) stays inside the letterboxed frame.

## QC per 5 s segment
frames present/no duplicates · exposure + letterbox · temporal pops · frozen frames · bone-length drift ·
IK hyperextension · foot sliding/sinking · joint-acceleration spikes outside declared action events ·
cloak stability · segment seams · **framing (head, every horn tip, heart, hands, feet inside frame)** ·
re-render determinism. Encode: frame count, duration, PSNR vs. source frames.
