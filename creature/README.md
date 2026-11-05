# The Ashen Cantor — 30 s creature animation + render QC

Original creature concept: a gaunt gothic choir-beast with a crown of spire-horns and an ember heart
burning inside its ribcage, in a night cathedral courtyard (blue sorcery mist, inked black/crimson body).

```bash
python pipeline.py              # solve rig -> render + QC each 5 s segment -> encode -> verify
python pipeline.py --from-seg 3 # resume from a segment
python qc.py --segment 4        # re-check one segment
```

| file | role |
|---|---|
| `config.py` | resolution, 24 fps, 6 × 5 s segments, declared action events |
| `anim.py` | monotone-cubic keyframes, FK/IK rig, footstep planner, verlet cloak sim → `out/rig.npz` |
| `render.py` | deterministic renderer (parallax cathedral, fog, rim light, bloom, embers, smear frames, grade) |
| `qc.py` | per-segment checks + contact sheet (`out/qc/segN_*.jpg`) + JSON report |
| `pipeline.py` | runs it all, stops at the first failing segment, verifies the encode |

QC per segment: frame completeness/duplicates, exposure + letterbox, temporal pops, frozen frames,
bone-length drift, IK hyperextension, foot sliding/sinking, joint-acceleration spikes outside declared
events, cloak stability, segment seams, bit-exact re-render determinism. Encode check: frame count,
duration, PSNR vs. source PNGs.
