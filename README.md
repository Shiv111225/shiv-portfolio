# shiv-portfolio
3D portfolio website - symbiote x black panther theme

A living black liquid mass (custom GLSL noise-displaced shader) with pulsing violet
"kinetic energy" veins, orbiting metallic shards, floating spores and bloom — built
with Three.js, no build step.

- Move the pointer: tendrils reach toward it. Click / scroll fast: the mass agitates.
- Scrolling charges the energy veins and moves the mass beside each section.

## Run locally
```bash
python3 -m http.server 8000
```
Then open http://localhost:8000.

## Customize
Edit the text, skills, projects and links in `index.html`. Colors live in `:root` in `style.css`.

## Deploy
It's a static site — drag the folder into Netlify, or connect this repo with no build
command and publish directory `.`.
