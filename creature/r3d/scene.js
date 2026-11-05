// THE ASHEN CANTOR - deterministic three.js renderer.
// Animation data comes from ../out/anim.bin (solved in Python: rig, IK, springs, cloth, camera).
// window.renderFrame(f) draws frame f (with 3-sample motion blur) and returns screen-space
// positions of key parts so the QC can verify framing.
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

const W = 1280, H = 720, LB = 40, FOV = 30;
const V3 = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
const UP = V3(0, 1, 0);

// ------------------------------------------------------------------ deterministic rng
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ------------------------------------------------------------------ data
let META, DATA, G, JIDX = {}, SIDX = {}, OFF_CL, OFF_SC;
async function loadData() {
  META = await (await fetch('../out/anim.json')).json();
  DATA = new Float32Array(await (await fetch('../out/anim.bin')).arrayBuffer());
  G = META.ground;
  META.names.forEach((n, i) => (JIDX[n] = i));
  META.scal.forEach((n, i) => (SIDX[n] = i));
  OFF_CL = META.nj * 2;
  OFF_SC = OFF_CL + META.nch * META.pmax * 2;
}
const sample = (i) => DATA.subarray(Math.max(0, i) * META.stride, (Math.max(0, i) + 1) * META.stride);
const jp = (s, n, z = 0) => { const k = JIDX[n]; return V3(s[2 * k], G - s[2 * k + 1], z); };
const sc = (s, n) => s[OFF_SC + SIDX[n]];
const clp = (s, c, i) => { const k = OFF_CL + (c * META.pmax + i) * 2; return V3(s[k], G - s[k + 1], 0); };
const dir2 = (a) => V3(Math.cos(a), -Math.sin(a), 0);

// ------------------------------------------------------------------ procedural textures
function canvasTex(w, h, draw, srgb = true, repeat = null) {
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const g = c.getContext('2d', { willReadFrequently: true }); draw(g, w, h);
  const t = new THREE.CanvasTexture(c);
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  if (repeat) { t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(repeat[0], repeat[1]); }
  t.anisotropy = 4;
  return t;
}
function blurPass(g, w, h, px) {          // one blur over the whole canvas (cheap)
  const t = document.createElement('canvas'); t.width = w; t.height = h;
  t.getContext('2d', { willReadFrequently: true }).drawImage(g.canvas, 0, 0);
  g.save(); g.clearRect(0, 0, w, h); g.filter = `blur(${px}px)`;
  for (const ox of [-w, 0, w]) for (const oy of [-h, 0, h]) g.drawImage(t, ox, oy);   // keep it tileable
  g.restore();
}
function noiseFill(g, w, h, rnd, n, rmin, rmax, colFn, blur = 0) {
  if (blur) {                               // draw on a layer, blur once, composite
    const L = document.createElement('canvas'); L.width = w; L.height = h;
    const lg = L.getContext('2d', { willReadFrequently: true });
    noiseFill(lg, w, h, rnd, n, rmin, rmax, colFn, 0); blurPass(lg, w, h, blur);
    g.drawImage(L, 0, 0); return;
  }
  g.save();
  for (let i = 0; i < n; i++) {
    const x = rnd() * w, y = rnd() * h, r = rmin + rnd() * (rmax - rmin);
    g.fillStyle = colFn(rnd);
    for (const ox of [-w, 0, w]) for (const oy of [-h, 0, h]) { // tileable
      g.beginPath(); g.arc(x + ox, y + oy, r, 0, 7); g.fill();
    }
  }
  g.restore();
}
function cracks(g, w, h, rnd, n, width, col) {
  g.strokeStyle = col; g.lineCap = 'round';
  for (let i = 0; i < n; i++) {
    let x = rnd() * w, y = rnd() * h, a = rnd() * 6.28;
    g.lineWidth = width * (0.4 + rnd());
    g.beginPath(); g.moveTo(x, y);
    for (let k = 0; k < 18; k++) {
      a += (rnd() - 0.5) * 1.1; x += Math.cos(a) * 9; y += Math.sin(a) * 9; g.lineTo(x, y);
      if (rnd() < 0.12) { g.stroke(); g.beginPath(); g.moveTo(x, y); a += (rnd() - 0.5) * 2; }
    }
    g.stroke();
  }
}
function makeTextures() {
  const T = {};
  let r = mulberry32(11);
  T.skin = canvasTex(512, 512, (g, w, h) => {
    g.fillStyle = '#231616'; g.fillRect(0, 0, w, h);
    noiseFill(g, w, h, r, 900, 2, 14, (q) => `rgba(${50 + q() * 40 | 0},${24 + q() * 18 | 0},${22 + q() * 16 | 0},0.25)`, 2);
    noiseFill(g, w, h, r, 2500, 0.5, 2, (q) => `rgba(${q() * 20 | 0},${q() * 10 | 0},${q() * 10 | 0},0.6)`);
    cracks(g, w, h, r, 60, 1.2, 'rgba(10,5,5,0.7)');
  }, true, [2, 5]);
  r = mulberry32(12);
  T.skinBump = canvasTex(512, 512, (g, w, h) => {
    g.fillStyle = '#808080'; g.fillRect(0, 0, w, h);
    noiseFill(g, w, h, r, 1600, 1, 9, (q) => { const v = 90 + q() * 90 | 0; return `rgba(${v},${v},${v},0.35)`; }, 1.5);
    cracks(g, w, h, r, 80, 1.6, 'rgba(20,20,20,0.9)');
  }, false, [2, 5]);
  r = mulberry32(13);
  T.skinEmit = canvasTex(512, 512, (g, w, h) => {
    g.fillStyle = '#000'; g.fillRect(0, 0, w, h);
    cracks(g, w, h, r, 34, 2.2, 'rgb(255,120,50)');
    blurPass(g, w, h, 1.2);
    cracks(g, w, h, mulberry32(14), 34, 0.8, 'rgb(255,200,120)');
  }, true, [2, 5]);
  r = mulberry32(21);
  T.stone = canvasTex(1024, 1024, (g, w, h) => {
    g.fillStyle = '#0d0e13'; g.fillRect(0, 0, w, h);
    const n = 8, s = w / n;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const v = 26 + r() * 18 | 0;
      g.fillStyle = `rgb(${v},${v + 2},${v + 8})`;
      const off = (j % 2) * s / 2;
      g.fillRect((i * s + off) % w + 3, j * s + 3, s - 6, s - 6);
    }
    noiseFill(g, w, h, r, 3000, 1, 6, (q) => { const v = q() * 50 | 0; return `rgba(${v},${v},${v + 6},0.25)`; }, 1);
  }, true, [10, 5]);
  r = mulberry32(22);
  T.stoneRough = canvasTex(1024, 1024, (g, w, h) => {
    g.fillStyle = '#e0e0e0'; g.fillRect(0, 0, w, h);
    noiseFill(g, w, h, r, 26, 40, 140, () => 'rgba(90,90,90,0.35)', 40);   // puddles = glossy
  }, false, [10, 5]);
  r = mulberry32(23);
  T.fog = canvasTex(1024, 256, (g, w, h) => {
    g.clearRect(0, 0, w, h);
    for (let i = 0; i < 260; i++) {
      const x = r() * w, y = h * (0.35 + 0.5 * r()), rr = 20 + r() * 70;
      g.fillStyle = `rgba(255,255,255,${0.05 + r() * 0.12})`;
      for (const ox of [-w, 0, w]) { g.beginPath(); g.ellipse(x + ox, y, rr * 1.8, rr * 0.6, 0, 0, 7); g.fill(); }
    }
    blurPass(g, w, h, 18);
  }, true);
  T.fog.wrapS = THREE.RepeatWrapping;
  T.ember = canvasTex(64, 64, (g, w, h) => {
    const gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    gr.addColorStop(0, 'rgba(255,240,200,1)'); gr.addColorStop(0.25, 'rgba(255,140,50,0.8)');
    gr.addColorStop(1, 'rgba(255,60,10,0)'); g.fillStyle = gr; g.fillRect(0, 0, w, h);
  });
  T.glow = canvasTex(128, 128, (g, w, h) => {
    const gr = g.createRadialGradient(64, 64, 0, 64, 64, 64);
    gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(0.3, 'rgba(255,255,255,0.35)');
    gr.addColorStop(1, 'rgba(255,255,255,0)'); g.fillStyle = gr; g.fillRect(0, 0, w, h);
  });
  T.rose = canvasTex(256, 256, (g) => {
    const gr = g.createRadialGradient(128, 128, 10, 128, 128, 120);
    gr.addColorStop(0, '#e8f2ff'); gr.addColorStop(0.6, '#7fa8ff'); gr.addColorStop(1, '#2a4590');
    g.fillStyle = gr; g.beginPath(); g.arc(128, 128, 120, 0, 7); g.fill();
    g.strokeStyle = '#0a0e18'; g.lineWidth = 9;
    for (let k = 0; k < 12; k++) { const a = k * Math.PI / 6; g.beginPath(); g.moveTo(128, 128); g.lineTo(128 + 120 * Math.cos(a), 128 + 120 * Math.sin(a)); g.stroke(); }
    g.lineWidth = 7; for (const rr of [40, 82, 118]) { g.beginPath(); g.arc(128, 128, rr, 0, 7); g.stroke(); }
  });
  T.window = canvasTex(64, 128, (g, w, h) => {
    g.fillStyle = '#000'; g.fillRect(0, 0, w, h);
    const gr = g.createLinearGradient(0, 0, 0, h); gr.addColorStop(0, '#ffd59a'); gr.addColorStop(1, '#ff9a40');
    g.fillStyle = gr; g.beginPath(); g.moveTo(6, h - 4); g.lineTo(6, 34); g.quadraticCurveTo(w / 2, -6, w - 6, 34);
    g.lineTo(w - 6, h - 4); g.fill(); g.fillStyle = '#000'; g.fillRect(w / 2 - 2, 20, 4, h); g.fillRect(0, 70, w, 4);
  });
  r = mulberry32(31);
  T.cloakAlpha = canvasTex(256, 256, (g, w, h) => {
    g.fillStyle = '#fff'; g.fillRect(0, 0, w, h); g.fillStyle = '#000';
    // ragged hem + moth holes
    g.beginPath(); g.moveTo(0, h);
    for (let x = 0; x <= w; x += 6) g.lineTo(x, h - 10 - r() * 46 - (x % 36 < 6 ? 30 : 0));
    g.lineTo(w, h); g.fill();
    for (let i = 0; i < 26; i++) { g.beginPath(); g.ellipse(r() * w, h * (0.35 + r() * 0.5), 2 + r() * 7, 3 + r() * 12, 0, 0, 7); g.fill(); }
  }, false);
  r = mulberry32(32);
  T.cloth = canvasTex(256, 256, (g, w, h) => {
    g.fillStyle = '#1a1820'; g.fillRect(0, 0, w, h);
    for (let x = 0; x < w; x += 2) { g.fillStyle = `rgba(0,0,0,${r() * 0.35})`; g.fillRect(x, 0, 1, h); }
    noiseFill(g, w, h, r, 200, 4, 26, (q) => `rgba(${40 + q() * 30 | 0},${20 + q() * 20 | 0},${25 + q() * 20 | 0},0.12)`, 6);
  }, true);
  return T;
}

// ------------------------------------------------------------------ geometry helpers
function ringFrame(T) {
  const ref = Math.abs(T.z) < 0.9 ? V3(0, 0, 1) : V3(1, 0, 0);
  const N = V3().crossVectors(ref, T).normalize();
  const Z = V3().crossVectors(N, T).normalize();
  return [N, Z];
}
function taperTube(pts, radii, seg = 10, flat = 1) {
  const pos = [], uv = [], idx = [];
  const n = pts.length;
  for (let i = 0; i < n; i++) {
    const T = (i < n - 1 ? pts[i + 1].clone().sub(pts[i]) : pts[i].clone().sub(pts[i - 1])).normalize();
    const [N, Z] = ringFrame(T);
    for (let j = 0; j <= seg; j++) {
      const a = (j / seg) * Math.PI * 2;
      const p = pts[i].clone().addScaledVector(N, Math.cos(a) * radii[i]).addScaledVector(Z, Math.sin(a) * radii[i] * flat);
      pos.push(p.x, p.y, p.z); uv.push(j / seg, i / (n - 1));
    }
  }
  for (let i = 0; i < n - 1; i++) for (let j = 0; j < seg; j++) {
    const a = i * (seg + 1) + j, b = a + seg + 1, c = a + 1, d = b + 1;
    idx.push(a, b, c, b, d, c);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.setIndex(idx); g.computeVertexNormals();
  return g;
}

class DynTube {                       // torso/neck: rebuilt every frame along the spine
  constructor(mat, rings = 40, seg = 18) {
    this.rings = rings; this.seg = seg;
    const nv = rings * (seg + 1);
    this.geo = new THREE.BufferGeometry();
    this.pos = new Float32Array(nv * 3);
    const uv = [], idx = [];
    for (let i = 0; i < rings; i++) for (let j = 0; j <= seg; j++) uv.push(j / seg * 2, i / (rings - 1) * 3);
    for (let i = 0; i < rings - 1; i++) for (let j = 0; j < seg; j++) {
      const a = i * (seg + 1) + j, b = a + seg + 1, c = a + 1, d = b + 1; idx.push(a, b, c, b, d, c);
    }
    this.geo.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    this.geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
    this.geo.setIndex(idx);
    this.mesh = new THREE.Mesh(this.geo, mat);
    this.mesh.castShadow = this.mesh.receiveShadow = true;
    this.mesh.frustumCulled = false;
  }
  update(points, radiusFn) {
    const curve = new THREE.CatmullRomCurve3(points, false, 'centripetal');
    let k = 0;
    for (let i = 0; i < this.rings; i++) {
      const u = i / (this.rings - 1);
      const P = curve.getPointAt(u), T = curve.getTangentAt(u).normalize();
      const N = V3(T.y, -T.x, 0).normalize();      // back normal in the body plane
      const Z = V3(0, 0, 1);
      const [rb, rf, rz] = radiusFn(u);             // back radius, front radius, side radius
      for (let j = 0; j <= this.seg; j++) {
        const a = (j / this.seg) * Math.PI * 2, c = Math.cos(a), s = Math.sin(a);
        const rn = c > 0 ? rb : rf;
        this.pos[k++] = P.x + N.x * c * rn + Z.x * s * rz;
        this.pos[k++] = P.y + N.y * c * rn + Z.y * s * rz;
        this.pos[k++] = P.z + N.z * c * rn + Z.z * s * rz;
      }
    }
    this.geo.attributes.position.needsUpdate = true;
    this.geo.computeVertexNormals();
  }
}

class Bone {                          // tapered limb segment between two joints
  constructor(parent, ra, rb, mat, seg = 12) {
    const g = new THREE.CylinderGeometry(rb, ra, 1, seg, 1, false);
    g.translate(0, 0.5, 0);
    this.m = new THREE.Mesh(g, mat); this.m.castShadow = this.m.receiveShadow = true;
    parent.add(this.m);
  }
  set(a, b) {
    const d = b.clone().sub(a), L = d.length();
    this.m.position.copy(a);
    this.m.quaternion.setFromUnitVectors(UP, d.divideScalar(L || 1));
    this.m.scale.set(1, L, 1);
  }
}
function ball(parent, r, mat, sx = 1, sy = 1, sz = 1, seg = 16) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(r, seg, Math.max(8, seg * 0.7 | 0)), mat);
  m.scale.set(sx, sy, sz); m.castShadow = m.receiveShadow = true; parent.add(m); return m;
}
function cone(parent, r, h, mat, seg = 8) {
  const g = new THREE.ConeGeometry(r, h, seg); g.translate(0, h / 2, 0);
  const m = new THREE.Mesh(g, mat); m.castShadow = true; parent.add(m); return m;
}
function aim(m, a, b) {                 // point a +Y object from a toward b, scaled to length
  const d = b.clone().sub(a); const L = d.length();
  m.position.copy(a); m.quaternion.setFromUnitVectors(UP, d.divideScalar(L || 1));
  return L;
}

// ------------------------------------------------------------------ scene
let renderer, scene, camera, composer, bloom, T, C = {}, ENV = {};
const accum = {};

function buildEnvironment() {
  const stone = new THREE.MeshStandardMaterial({ color: 0x3a4152, roughness: 0.95, metalness: 0, map: T.stone });
  const stoneDark = new THREE.MeshStandardMaterial({ color: 0x1b1f2a, roughness: 0.95 });
  // ground
  const gmat = new THREE.MeshStandardMaterial({ map: T.stone, roughnessMap: T.stoneRough, roughness: 1, metalness: 0.0, color: 0x9aa3b8 });
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(9000, 4500), gmat);
  ground.rotation.x = -Math.PI / 2; ground.position.set(700, 0, -1200); ground.receiveShadow = true;
  scene.add(ground);

  const winMat = new THREE.MeshBasicMaterial({ map: T.window, color: 0xffffff, fog: true });
  const rnd = mulberry32(7);
  const addWindow = (x, y, z, w, h, lit) => {
    if (!lit) return;
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), winMat);
    m.position.set(x, y, z); scene.add(m);
  };
  const spire = (x, z, w, bodyH, spH, mat) => {
    const b = new THREE.Mesh(new THREE.BoxGeometry(w, bodyH, w), mat); b.position.set(x, bodyH / 2, z); scene.add(b);
    const c = new THREE.Mesh(new THREE.ConeGeometry(w * 0.72, spH, 4), mat);
    c.rotation.y = Math.PI / 4; c.position.set(x, bodyH + spH / 2, z); scene.add(c);
    for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
      const p = new THREE.Mesh(new THREE.ConeGeometry(w * 0.12, spH * 0.3, 4), mat);
      p.position.set(x + sx * w / 2, bodyH + spH * 0.15, z + sz * w / 2); scene.add(p);
    }
  };
  // --- far cathedral
  const CZ = -2600, CX = 900;
  const nave = new THREE.Mesh(new THREE.BoxGeometry(560, 1000, 500), stone); nave.position.set(CX, 500, CZ); scene.add(nave);
  const gable = new THREE.Mesh(new THREE.CylinderGeometry(0, 330, 320, 4, 1), stone);
  gable.scale.set(1, 1, 0.8); gable.rotation.y = Math.PI / 4; gable.position.set(CX, 1160, CZ); scene.add(gable);
  spire(CX - 380, CZ + 60, 190, 1500, 900, stone);
  spire(CX + 380, CZ + 60, 190, 1380, 820, stone);
  spire(CX, CZ - 100, 60, 1250, 480, stone);
  const rose = new THREE.Mesh(new THREE.CircleGeometry(130, 48), new THREE.MeshBasicMaterial({ map: T.rose, fog: false, color: 0xbfd4ff }));
  rose.position.set(CX, 820, CZ + 252); scene.add(rose); ENV.rose = rose;
  const roseGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: T.glow, color: 0x5577ff, transparent: true, opacity: 0.35, blending: THREE.AdditiveBlending, depthWrite: false, fog: false }));
  roseGlow.scale.set(900, 900, 1); roseGlow.position.set(CX, 820, CZ + 280); scene.add(roseGlow); ENV.roseGlow = roseGlow;
  for (const sx of [-1, 1]) for (let k = 0; k < 5; k++) addWindow(CX + sx * 380, 300 + k * 230, CZ + 157, 46, 110, rnd() < 0.6);
  for (let k = 0; k < 3; k++) addWindow(CX - 150 + k * 150, 380, CZ + 252, 60, 170, true);
  for (const side of [-1, 1]) for (let k = 0; k < 7; k++) {
    const x = CX + side * (700 + k * 260 + rnd() * 40), h = 650 + rnd() * 500;
    spire(x, CZ + 100 + rnd() * 200, 170, h, 250 + rnd() * 450, stone);
    for (let y = 200; y < h - 150; y += 200) addWindow(x, y, CZ + 100 + 86 + 200 * 0, 34, 80, rnd() < 0.4);
    const bt = new THREE.Mesh(new THREE.BoxGeometry(40, 700, 40), stone);
    bt.position.set(x - side * 150, 300, CZ + 160); bt.rotation.z = side * 0.45; scene.add(bt);
  }
  // --- courtyard ruins (mid ground): sparse broken columns for depth, cathedral stays visible
  for (let k = 0; k < 7; k++) {
    const x = -500 + k * 380 + rnd() * 120, z = -1100 - rnd() * 500, h = 160 + rnd() * 380;
    const col = new THREE.Mesh(new THREE.CylinderGeometry(26, 30, h, 14), stoneDark); col.position.set(x, h / 2, z);
    col.rotation.z = (rnd() - 0.5) * 0.08; col.castShadow = true; scene.add(col);
    const plinth = new THREE.Mesh(new THREE.BoxGeometry(84, 30, 84), stoneDark); plinth.position.set(x, 15, z); scene.add(plinth);
  }
  // --- foreground: balustrade + hooded statue (real parallax)
  const fgMat = new THREE.MeshStandardMaterial({ color: 0x20242e, roughness: 0.9, map: T.stone });
  const rail = new THREE.Mesh(new THREE.BoxGeometry(700, 26, 50), fgMat); rail.position.set(-120, 70, 860); rail.castShadow = true; scene.add(rail);
  const base = new THREE.Mesh(new THREE.BoxGeometry(700, 30, 60), fgMat); base.position.set(-120, 10, 860); scene.add(base);
  const balG = new THREE.LatheGeometry([[0, 0], [10, 0], [12, 12], [7, 30], [15, 70], [8, 108], [12, 122], [0, 135]].map(p => new THREE.Vector2(p[0], p[1])), 14);
  for (let k = 0; k < 9; k++) { const b = new THREE.Mesh(balG, fgMat); b.position.set(-440 + k * 70, 2, 860); b.scale.set(1, 0.5, 1); b.castShadow = true; scene.add(b); }
  const statue = new THREE.Group();
  const robe = new THREE.Mesh(new THREE.LatheGeometry([[0, 0], [95, 0], [80, 200], [62, 420], [48, 520], [0, 560]].map(p => new THREE.Vector2(p[0], p[1])), 24), fgMat);
  const hood = new THREE.Mesh(new THREE.SphereGeometry(55, 20, 14), fgMat); hood.scale.set(1, 1.25, 1); hood.position.y = 590;
  const plinth = new THREE.Mesh(new THREE.BoxGeometry(240, 120, 240), fgMat); plinth.position.y = -60;
  statue.add(robe, hood, plinth); statue.position.set(1500, 120, 300); statue.rotation.y = -0.5;
  statue.traverse(o => { if (o.isMesh) o.castShadow = true; }); scene.add(statue);
  // --- brazier (warm key light, flickers deterministically)
  const bz = new THREE.Group();
  const bowl = new THREE.Mesh(new THREE.CylinderGeometry(46, 18, 40, 16, 1, true), new THREE.MeshStandardMaterial({ color: 0x1a1512, roughness: 0.5, metalness: 0.7, side: THREE.DoubleSide }));
  bowl.position.y = 150; const stand = new THREE.Mesh(new THREE.CylinderGeometry(6, 14, 140, 8), bowl.material); stand.position.y = 70;
  bz.add(bowl, stand); bz.position.set(80, 0, 620); bz.traverse(o => { if (o.isMesh) o.castShadow = true; }); scene.add(bz);
  const fireMat = new THREE.SpriteMaterial({ map: T.glow, color: 0xff7a2a, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false });
  ENV.fire = []; for (let k = 0; k < 5; k++) { const sp = new THREE.Sprite(fireMat.clone()); bz.add(sp); ENV.fire.push(sp); }
  ENV.fireLight = new THREE.PointLight(0xff8a3a, 1, 2400, 1.2); ENV.fireLight.position.set(80, 230, 620);
  ENV.fireLight.castShadow = true; ENV.fireLight.shadow.mapSize.set(1024, 1024); ENV.fireLight.shadow.bias = -0.002;
  scene.add(ENV.fireLight);
  // --- sky dome: dark zenith, pale cold horizon glow behind the cathedral (silhouette staging)
  const sky = new THREE.Mesh(new THREE.SphereGeometry(9000, 32, 16), new THREE.ShaderMaterial({ side: THREE.BackSide, depthWrite: false, fog: false,
    vertexShader: 'varying vec3 vP; void main(){ vP = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.); }',
    fragmentShader: `varying vec3 vP; void main(){ float h = clamp(vP.y, -0.1, 1.0);
      vec3 c = mix(vec3(0.20,0.27,0.42), vec3(0.025,0.035,0.07), pow(clamp(h*2.2,0.,1.), 0.7));
      float m = pow(max(dot(vP, normalize(vec3(0.35,0.3,-1.0))), 0.0), 18.0);
      gl_FragColor = vec4(c + vec3(0.35,0.45,0.7)*m, 1.0); }` }));
  sky.position.set(700, 0, 0); scene.add(sky); ENV.sky = sky;
  // --- moon
  const moon = new THREE.Sprite(new THREE.SpriteMaterial({ map: T.glow, color: 0xcfe0ff, fog: false, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
  moon.scale.set(900, 900, 1); moon.position.set(2300, 2300, -5000); scene.add(moon);
  // --- fog sheets
  ENV.fog = [];
  for (const [z, y, h, op, sp] of [[-2200, 380, 900, 0.45, 6], [-1500, 260, 700, 0.32, 9], [-700, 160, 420, 0.24, 14],
                                    [-250, 90, 260, 0.16, 20], [120, 60, 200, 0.08, 30]]) {
    const tex = T.fog.clone(); tex.needsUpdate = true; tex.wrapS = THREE.RepeatWrapping;
    const m = new THREE.Mesh(new THREE.PlaneGeometry(8000, h), new THREE.MeshBasicMaterial({ map: tex, color: 0x6d86b8, transparent: true, opacity: op, depthWrite: false, fog: false }));
    tex.repeat.set(4, 1);
    m.position.set(700, y, z); m.renderOrder = 2; scene.add(m); ENV.fog.push({ m, tex, sp, op, front: z > 0 });
  }
  // --- blue sorcery ribbons
  ENV.ribbons = [];
  for (const [y, z, ph] of []) {
    const pts = []; for (let i = 0; i <= 60; i++) { const x = -600 + i * 50; pts.push(V3(x, y + 60 * Math.sin(i * 0.25 + ph), z + 80 * Math.sin(i * 0.13))); }
    const g = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 240, 9, 6);
    const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: 0x5c8cff, transparent: true, opacity: 0.32, blending: THREE.AdditiveBlending, depthWrite: false }));
    scene.add(m); ENV.ribbons.push({ m, y, ph });
  }
}

function buildCreature() {
  const skin = new THREE.MeshStandardMaterial({
    map: T.skin, bumpMap: T.skinBump, bumpScale: 4.0, roughness: 0.74, metalness: 0.0,
    emissiveMap: T.skinEmit, emissive: new THREE.Color(1.0, 0.25, 0.06), emissiveIntensity: 0.3, color: 0x4a4040
  });
  const horn = new THREE.MeshStandardMaterial({ color: 0x1c1714, roughness: 0.34, metalness: 0.1, bumpMap: T.skinBump, bumpScale: 1.0 });
  const claw = new THREE.MeshStandardMaterial({ color: 0x0c0a0a, roughness: 0.22, metalness: 0.2 });
  const bone = new THREE.MeshStandardMaterial({ color: 0x4a3c36, roughness: 0.5, bumpMap: T.skinBump, bumpScale: 0.8 });
  const cloth = new THREE.MeshStandardMaterial({ map: T.cloth, alphaMap: T.cloakAlpha, alphaTest: 0.5, side: THREE.DoubleSide, roughness: 0.96, color: 0x9a8f9e });
  C.mats = { skin, horn, claw, bone, cloth };
  const root = new THREE.Group(); scene.add(root); C.root = root;

  C.torso = new DynTube(skin); root.add(C.torso.mesh);
  C.pelvis = ball(root, 1, skin, 30, 22, 24);
  // ribcage
  C.ribs = []; for (let k = 0; k < 6; k++) {
    const m = new THREE.Mesh(new THREE.TorusGeometry(1, 0.1, 8, 32), bone); m.castShadow = true; root.add(m); C.ribs.push(m);
  }
  C.sternum = new Bone(root, 3, 2.2, bone);
  C.spikes = []; for (let k = 0; k < 7; k++) C.spikes.push(cone(root, 4.5, 24, horn));
  // heart
  C.heart = new THREE.Mesh(new THREE.SphereGeometry(12, 24, 16), new THREE.MeshBasicMaterial({ color: 0xffa060 }));
  root.add(C.heart);
  C.heartGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: T.glow, color: 0xff4a18, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
  root.add(C.heartGlow);
  C.heartLight = new THREE.PointLight(0xff4a1a, 1, 900, 1.6); root.add(C.heartLight);

  // head
  const head = new THREE.Group(); root.add(head); C.head = head;
  ball(head, 1, skin, 24, 19, 17).position.set(10, 10, 0);       // cranium
  ball(head, 1, skin, 36, 9.5, 10.5).position.set(42, 3, 0);     // snout
  ball(head, 1, skin, 20, 5, 19).position.set(30, 15, 0);        // brow ridge
  for (const z of [-12, 12]) ball(head, 1, skin, 14, 7, 6).position.set(22, -1, z);  // cheekbones
  const jaw = new THREE.Group(); jaw.position.set(8, -6, 0); head.add(jaw); C.jaw = jaw;
  ball(jaw, 1, skin, 34, 5.5, 9.5).position.set(28, -3, 0);
  const tooth = new THREE.MeshStandardMaterial({ color: 0xd8c8b0, roughness: 0.35 });
  for (let k = 0; k < 7; k++) for (const z of [-6.5, 6.5]) {
    const u = cone(head, 1.4, 7, tooth, 5); u.position.set(24 + k * 7, -4, z * (1 - k * 0.06)); u.rotation.z = Math.PI;
    const l = cone(jaw, 1.3, 6, tooth, 5); l.position.set(18 + k * 7, -1, z * (1 - k * 0.06));
  }
  C.throat = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 10), new THREE.MeshBasicMaterial({ color: 0xff5020 }));
  C.throat.scale.set(26, 4, 6); C.throat.position.set(36, -5, 0); head.add(C.throat);
  C.eyes = []; for (const z of [-10.5, 10.5]) {
    const e = new THREE.Mesh(new THREE.SphereGeometry(3.2, 12, 8), new THREE.MeshBasicMaterial({ color: 0xff7a2a }));
    e.scale.set(1.7, 0.6, 1); e.position.set(40, 9.5, z); head.add(e);
    const gs = new THREE.Sprite(new THREE.SpriteMaterial({ map: T.glow, color: 0xff5a1a, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
    gs.position.copy(e.position); gs.scale.set(26, 26, 1); head.add(gs);
    C.eyes.push({ e, gs });
  }
  // crown of horns (static tapered tubes in a lag pivot)
  const crown = new THREE.Group(); crown.position.set(4, 20, 0); head.add(crown); C.crown = crown;
  C.hornTips = [];
  for (let k = 0; k < 7; k++) {
    const c = k - 3, len = 112 - Math.abs(c) * 15, spread = c * 0.33;
    const pts = [], rad = [];
    let p = V3(Math.abs(c) * 3, -Math.abs(c) * 2, c * 6);
    let el = 0.55 + Math.abs(c) * 0.09;                         // elevation from 'back'
    for (let i = 0; i <= 8; i++) {
      pts.push(p.clone()); rad.push(4.6 * (1 - i / 8) ** 1.15 + 0.35);
      const d = V3(-Math.cos(el) * Math.cos(spread), Math.sin(el), Math.sin(spread) * 0.9).normalize();
      p = p.clone().addScaledVector(d, len / 8);
      el -= 0.07;                                               // curve back like a swept spire
    }
    const m = new THREE.Mesh(taperTube(pts, rad, 10), horn); m.castShadow = true; crown.add(m);
    const tip = new THREE.Object3D(); tip.position.copy(pts[8]); crown.add(tip); C.hornTips.push(tip);
  }
  // limbs
  C.limbs = {};
  for (const [side, z] of [['n', 22], ['f', -22]]) {
    const L = {};
    L.z = z;
    L.upper = new Bone(root, 9.5, 7, skin); L.fore = new Bone(root, 7, 4.6, skin); L.hand = new Bone(root, 4.6, 4.0, skin);
    L.sh = ball(root, 12, skin); L.el = ball(root, 7.6, skin); L.wr = ball(root, 5, skin); L.kn = ball(root, 4.4, skin);
    L.fingers = [];
    for (let k = 0; k < 4; k++) L.fingers.push({ a: new Bone(root, 3.0, 2.4, skin), b: new Bone(root, 2.4, 1.8, skin), j: ball(root, 2.5, skin, 1, 1, 1, 8), j2: ball(root, 2.0, skin, 1, 1, 1, 8), c: new Bone(root, 1.8, 0.15, claw, 8) });
    const lz = side === 'n' ? 13 : -13;
    L.lz = lz;
    L.thigh = new Bone(root, 14, 9, skin); L.shin = new Bone(root, 8.5, 5.5, skin); L.meta = new Bone(root, 5.5, 4.2, skin);
    L.hip = ball(root, 14, skin); L.knee = ball(root, 9.5, skin); L.ankle = ball(root, 6, skin);
    L.talons = []; for (let k = 0; k < 3; k++) L.talons.push(new Bone(root, 3.0, 0.2, claw, 8));
    C.limbs[side] = L;
  }
  // cloak: elliptical drape around the body between the shoulder chain (front) and the
  // rearmost chain (back), open at the front for the arms; rebuilt every frame from the cloth sim
  const pm = META.pmax, NC = 26;
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(NC * pm * 3), uv = [], idx = [];
  for (let k = 0; k < NC; k++) for (let i = 0; i < pm; i++) uv.push(k / (NC - 1) * 2.0, i / (pm - 1));
  for (let k = 0; k < NC - 1; k++) for (let i = 0; i < pm - 1; i++) {
    const a = k * pm + i, b = a + 1, c = a + pm, d = c + 1; idx.push(a, b, c, b, d, c);
  }
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  geo.setIndex(idx);
  C.cloak = new THREE.Mesh(geo, cloth); C.cloak.castShadow = C.cloak.receiveShadow = true; C.cloak.frustumCulled = false;
  C.cloakPos = pos; C.cloakNC = NC; root.add(C.cloak);
}

function poseCreature(s, t) {
  const J = (n, z = 0) => jp(s, n, z);
  const heart = sc(s, 'heart');
  C.mats.skin.emissiveIntensity = 0.02 + 1.4 * heart ** 3;     // cracks only flare with the heart
  // torso along spine + neck
  const sp = [J('spine0'), J('spine1'), J('spine2'), J('spine3'), J('neck1'), J('neck2')];
  C.torso.update(sp, (u) => {
    // [back radius, front radius, side radius]
    if (u < 0.12) return [26 - u * 40, 24 - u * 50, 24 - u * 30];
    if (u < 0.35) return [21, 15, 17];
    if (u < 0.62) return [20, 9, 11];               // hollow ribcage region (core only)
    return [13 - (u - 0.62) * 10, 11 - (u - 0.62) * 8, 11 - (u - 0.62) * 7];
  });
  C.pelvis.position.copy(sp[0]);
  // ribs
  const cdir = dir2(sc(s, 'chest_dir'));
  const N = V3(cdir.y, -cdir.x, 0);           // back normal
  for (let k = 0; k < C.ribs.length; k++) {
    const u = 0.05 + k * 0.17;
    const P = sp[2].clone().lerp(sp[3], u).addScaledVector(N, -9);
    const rn = 31 - k * 1.8, rz = 27 - k * 1.5;
    const m = new THREE.Matrix4().makeBasis(N, V3(0, 0, -1), cdir);
    const r = C.ribs[k]; r.position.copy(P); r.quaternion.setFromRotationMatrix(m); r.scale.set(rn, rz, rn);
  }
  const s0 = sp[2].clone().lerp(sp[3], 0.05).addScaledVector(N, -40), s1 = sp[2].clone().lerp(sp[3], 0.9).addScaledVector(N, -36);
  C.sternum.set(s0, s1);
  for (let k = 0; k < C.spikes.length; k++) {
    const u = k / (C.spikes.length - 1);
    const seg = Math.min(2, Math.floor(u * 3)), f = u * 3 - seg;
    const P = sp[seg].clone().lerp(sp[seg + 1], f);
    const d = sp[seg + 1].clone().sub(sp[seg]).normalize(), n = V3(d.y, -d.x, 0);
    const base = P.clone().addScaledVector(n, 18);
    aim(C.spikes[k], base, base.clone().addScaledVector(n, 1).addScaledVector(d, 0.45));
    C.spikes[k].scale.setScalar(0.7 + 0.5 * Math.sin(u * Math.PI));
  }
  // heart
  const hp = J('heart');
  C.heart.position.copy(hp); C.heart.scale.setScalar(0.8 + 0.25 * heart);
  C.heart.material.color.setRGB(1.0 * (0.4 + 2.6 * heart), 0.35 * (0.3 + 2.0 * heart), 0.12 * (0.3 + 1.5 * heart));
  C.heartGlow.position.copy(hp).add(V3(0, 0, 20)); C.heartGlow.scale.setScalar(50 + 70 * heart);
  C.heartGlow.material.opacity = 0.15 + 0.5 * heart;
  C.heartLight.position.copy(hp).add(V3(-10, 0, 0)); C.heartLight.intensity = 9000 * (0.15 + heart * heart * 1.6);
  // head
  const n2 = J('neck2'), f3 = dir2(sc(s, 'head_dir')), u3 = V3(f3.y, -f3.x, 0), s3 = V3().crossVectors(f3, u3);
  C.head.position.copy(n2).addScaledVector(f3, -4);
  C.head.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(f3, u3, s3));
  C.jaw.rotation.z = -sc(s, 'jaw') * 0.8;
  const jw = sc(s, 'jaw');
  C.throat.material.color.setRGB(3 * jw * (0.4 + heart), 0.7 * jw * (0.4 + heart), 0.15 * jw);
  const eyes = sc(s, 'eyes');
  for (const { e, gs } of C.eyes) { e.material.color.setRGB(3.2 * eyes, 1.0 * eyes, 0.25 * eyes); gs.material.opacity = 0.9 * eyes; }
  C.crown.rotation.z = sc(s, 'horn_lag');
  // arms
  for (const side of ['n', 'f']) {
    const L = C.limbs[side], z = L.z;
    const sh = J(side + '_sh', z), el = J(side + '_el', z), wr = J(side + '_wr', z), kn = J(side + '_kn', z);
    L.upper.set(sh, el); L.fore.set(el, wr); L.hand.set(wr, kn);
    L.sh.position.copy(sh); L.el.position.copy(el); L.wr.position.copy(wr); L.kn.position.copy(kn);
    for (let k = 0; k < 4; k++) {
      const fz = z + (k - 1.5) * 3.6;
      const k0 = kn.clone().setZ(fz), p1 = J(`${side}_f${k}_1`, fz), p2 = J(`${side}_f${k}_2`, fz + (k - 1.5) * 1.5), tip = J(`${side}_f${k}_tip`, fz + (k - 1.5) * 2.5);
      const F = L.fingers[k]; F.a.set(k0, p1); F.b.set(p1, p2); F.c.set(p2, tip); F.j.position.copy(p1); F.j2.position.copy(p2);
    }
    // legs
    const lz = L.lz;
    const hip = J(side + '_hipj', lz), knee = J(side + '_knee', lz), ank = J(side + '_ankle', lz), toe = J(side + '_toe', lz);
    L.thigh.set(hip, knee); L.shin.set(knee, ank); L.meta.set(ank, toe);
    L.hip.position.copy(hip); L.knee.position.copy(knee); L.ankle.position.copy(ank);
    for (let k = 0; k < 3; k++) {
      const a = (k - 1) * 0.5;
      const tip = toe.clone().add(V3(-Math.cos(a) * (17 - Math.abs(k - 1) * 3), -toe.y + 0.5, Math.sin(a) * 15));
      L.talons[k].set(toe.clone().add(V3(0, 1.5, 0)), tip);
    }
  }
  // cloak drape
  const pm = META.pmax, NC = C.cloakNC, P = C.cloakPos;
  const rows = [];
  for (let i = 0; i < pm; i++) {
    const F = clp(s, 0, Math.min(i, META.chain_len[0] - 1));
    let B = F, bx = -1e9;                       // rearmost chain point on this row
    for (let c = 1; c < META.nch; c++) { const p = clp(s, c, Math.min(i, META.chain_len[c] - 1)); if (p.x > bx) { bx = p.x; B = p; } }
    rows.push([F, B]);
  }
  let q = 0;
  for (let k = 0; k < NC; k++) {
    const phi = 0.55 + (k / (NC - 1)) * (2 * Math.PI - 1.1);   // leave the front open
    for (let i = 0; i < pm; i++) {
      const [F, B] = rows[i];
      const M = F.clone().add(B).multiplyScalar(0.5), A = B.clone().sub(F).multiplyScalar(0.5);
      const Wd = 30 + 2.4 * i;
      const fold = Math.sin(phi * 5 + i * 0.4) * (1.5 + i * 0.7) + Math.sin(phi * 9 - i * 0.9 + t * 0.7) * i * 0.25;
      const r = 1 + fold / Wd;
      P[q++] = M.x - A.x * Math.cos(phi) * r; P[q++] = M.y - A.y * Math.cos(phi) * r - Math.abs(Math.sin(phi)) * 2;
      P[q++] = Wd * Math.sin(phi) * r;
    }
  }
  C.cloak.geometry.attributes.position.needsUpdate = true;
  C.cloak.geometry.computeVertexNormals();
}

// ------------------------------------------------------------------ FX
let EMB, BURST, TRAILS = [], RING;
function buildFX() {
  const n = 260;
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  EMB = new THREE.Points(g, new THREE.PointsMaterial({ size: 9, map: T.ember, vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true }));
  EMB.frustumCulled = false; scene.add(EMB);
  const r = mulberry32(99);
  EMB.p = Array.from({ length: n }, () => ({ spawn: r() * 34 - 4, life: 3 + r() * 4, x: -400 + r() * 2400, z: -700 + r() * 1100, vx: 8 + r() * 25, vy: 25 + r() * 50, ph: r() * 6.28, sz: r() }));
  const ringTex = canvasTex(512, 512, (g) => { const gr = g.createRadialGradient(256, 256, 150, 256, 256, 256);
    gr.addColorStop(0, 'rgba(255,255,255,0)'); gr.addColorStop(0.7, 'rgba(255,255,255,0.5)'); gr.addColorStop(0.86, 'rgba(255,255,255,0.15)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = gr; g.fillRect(0, 0, 512, 512); });
  RING = new THREE.Sprite(new THREE.SpriteMaterial({ map: ringTex, color: 0xff5522, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
  scene.add(RING);
}
function updateFX(t, s, f) {
  const pos = EMB.geometry.attributes.position.array, col = EMB.geometry.attributes.color.array;
  let k = 0;
  for (const p of EMB.p) {
    const age = ((t - p.spawn) % 40 + 40) % 40;
    const a = age < p.life ? Math.sin((age / p.life) * Math.PI) : 0;
    pos[k] = p.x + p.vx * age + 14 * Math.sin(age * 1.3 + p.ph); pos[k + 1] = 10 + p.vy * age; pos[k + 2] = p.z;
    col[k] = a * 1.6; col[k + 1] = a * 0.6; col[k + 2] = a * 0.2; k += 3;
  }
  EMB.geometry.attributes.position.needsUpdate = EMB.geometry.attributes.color.needsUpdate = true;
  // bursts (roar + slash)
  if (!BURST) {
    BURST = [];
    for (const [t0, cnt, joint, spd, seed] of [[16.55, 140, 'heart', 420, 5], [21.32, 70, 'n_f1_tip', 520, 6], [23.28, 50, 'f_f1_tip', 450, 8]]) {
      const s0 = sample(Math.round(t0 * META.fps * META.sub)), o = jp(s0, joint, joint.startsWith('f') ? -22 : 22);
      const rr = mulberry32(seed);
      const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(cnt * 3), 3));
      g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(cnt * 3), 3));
      const pts = new THREE.Points(g, new THREE.PointsMaterial({ size: 7, map: T.ember, vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
      pts.frustumCulled = false; scene.add(pts);
      BURST.push({ t0, pts, o, p: Array.from({ length: cnt }, () => { const th = rr() * 6.28, ph = (rr() - 0.5) * 2.4; return { d: V3(Math.cos(th) * Math.cos(ph), Math.sin(th) * Math.cos(ph), Math.sin(ph)), sp: spd * (0.3 + rr()), life: 0.8 + rr() * 1.6 }; }) });
    }
  }
  for (const b of BURST) {
    const age = t - b.t0, P = b.pts.geometry.attributes.position.array, Cc = b.pts.geometry.attributes.color.array;
    b.pts.visible = age >= 0;
    if (age < 0) continue;
    let k = 0;
    for (const p of b.p) {
      const dist = p.sp * (1 - Math.exp(-2.2 * age)) / 2.2, a = Math.max(0, 1 - age / p.life) ** 1.5;
      P[k] = b.o.x + p.d.x * dist; P[k + 1] = b.o.y + p.d.y * dist + 30 * age * age; P[k + 2] = b.o.z + p.d.z * dist;
      Cc[k] = 2.2 * a; Cc[k + 1] = 0.8 * a; Cc[k + 2] = 0.25 * a; k += 3;
    }
    b.pts.geometry.attributes.position.needsUpdate = b.pts.geometry.attributes.color.needsUpdate = true;
  }
  // shockwave ring
  const age = t - 16.55;
  RING.visible = age >= 0 && age < 0.9;
  if (RING.visible) {
    const hp = jp(sample(Math.round(16.55 * 96)), 'heart');
    RING.position.copy(hp); RING.scale.setScalar(60 + age * 2400);
    RING.material.opacity = 0.8 * (1 - age / 0.9) ** 2; RING.material.color.setRGB(1.6, 0.45, 0.18);
  }
  // claw trails: real swept paths of the fingertips
  for (const tr of TRAILS) { scene.remove(tr); tr.geometry.dispose(); }
  TRAILS = [];
  for (const [t0, t1, side, fade] of [[21.12, 21.5, 'n', 0.9], [23.1, 23.48, 'f', 0.8]]) {
    if (t < t0 + 0.02 || t > t1 + fade) continue;
    const tEnd = Math.min(t, t1), op = t > t1 ? (1 - (t - t1) / fade) ** 3 : 1;
    const tStart = Math.max(t0, tEnd - 0.11);
    for (let k = 0; k < 4; k++) {
      const pts = [];
      for (let u = Math.round(tStart * 96); u <= Math.round(tEnd * 96); u++) pts.push(jp(sample(u), `${side}_f${k}_tip`, (side === 'n' ? 22 : -22) + (k - 1.5) * 6));
      if (pts.length < 3) continue;
      const cur = new THREE.CatmullRomCurve3(pts);
      const g = new THREE.TubeGeometry(cur, pts.length * 4, 1.2, 6);
      const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: new THREE.Color(3.0 * op, 0.9 * op, 0.4 * op), transparent: true, opacity: op, blending: THREE.AdditiveBlending, depthWrite: false }));
      scene.add(m); TRAILS.push(m);
    }
  }
}

// ------------------------------------------------------------------ camera
function poseCamera(s, t) {
  const cx = sc(s, 'cam_x'), cy = G - sc(s, 'cam_y'), z = sc(s, 'zoom');
  const d = (H / z / 2) / Math.tan(THREE.MathUtils.degToRad(FOV / 2));
  const yaw = 0.16 + 0.05 * Math.sin(t * 0.21), pitch = -0.045;   // low camera looking up: monumental staging
  const tgt = V3(cx, cy, 0);
  camera.position.set(cx - Math.sin(yaw) * d, cy + Math.sin(pitch) * d, Math.cos(yaw) * d);
  camera.lookAt(tgt);
  // shake in screen space (camera-local)
  const k = 1 / z;
  camera.translateX(sc(s, 'shake_x') * k); camera.translateY(-sc(s, 'shake_y') * k);
  // re-centre: yaw shifts the framing, so offset lookAt to keep the 2D framing contract
  camera.updateMatrixWorld();
}

// ------------------------------------------------------------------ init + frame
let finalMat, accumMat, quadScene, quadCam, titleTex, titleCtx, moonLight, fill;
async function init() {
  await loadData();
  renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(1); renderer.setSize(W, H);
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.4;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  document.body.appendChild(renderer.domElement);
  scene = new THREE.Scene();
  
  scene.fog = new THREE.FogExp2(0x1e2a46, 0.00019);
  camera = new THREE.PerspectiveCamera(FOV, W / H, 10, 12000);
  T = makeTextures();
  scene.add(new THREE.HemisphereLight(0x4a64a0, 0x120c0c, 1.3));
  moonLight = new THREE.DirectionalLight(0xb4caff, 5.0);
  moonLight.castShadow = true; moonLight.shadow.mapSize.set(2048, 2048);
  Object.assign(moonLight.shadow.camera, { left: -700, right: 700, top: 700, bottom: -300, near: 10, far: 4000 });
  moonLight.shadow.bias = -0.0006; moonLight.shadow.normalBias = 1.5; moonLight.shadow.radius = 4;
  scene.add(moonLight, moonLight.target);
  fill = new THREE.DirectionalLight(0x6a7cb0, 0.55); scene.add(fill, fill.target);
  ENV.rim = new THREE.DirectionalLight(0x9fc0ff, 6.0); scene.add(ENV.rim, ENV.rim.target);   // cold kicker from behind-left
  buildEnvironment(); buildCreature(); buildFX();

  composer = new EffectComposer(renderer, new THREE.WebGLRenderTarget(W, H, { type: THREE.HalfFloatType }));
  composer.addPass(new RenderPass(scene, camera));
  bloom = new UnrealBloomPass(new THREE.Vector2(W, H), 0.75, 0.55, 0.82); composer.addPass(bloom);
  composer.addPass(new OutputPass());
  composer.renderToScreen = false;

  accum.rt = new THREE.WebGLRenderTarget(W, H, { type: THREE.HalfFloatType });
  quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  quadScene = new THREE.Scene();
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2));
  quadScene.add(quad);
  accumMat = new THREE.ShaderMaterial({
    uniforms: { tex: { value: null }, w: { value: 1 } }, transparent: true, depthTest: false, depthWrite: false,
    blending: THREE.CustomBlending, blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor, blendEquation: THREE.AddEquation,
    vertexShader: 'varying vec2 vUv; void main(){vUv=uv; gl_Position=vec4(position.xy,0.,1.);}',
    fragmentShader: 'uniform sampler2D tex; uniform float w; varying vec2 vUv; void main(){gl_FragColor=vec4(texture2D(tex,vUv).rgb*w,1.0);}'
  });
  const tc = document.createElement('canvas'); tc.width = W; tc.height = H; titleCtx = tc.getContext('2d');
  titleTex = new THREE.CanvasTexture(tc);
  finalMat = new THREE.ShaderMaterial({
    uniforms: { tex: { value: accum.rt.texture }, title: { value: titleTex }, seed: { value: 0 }, lb: { value: LB / H } },
    depthTest: false, depthWrite: false,
    vertexShader: 'varying vec2 vUv; void main(){vUv=uv; gl_Position=vec4(position.xy,0.,1.);}',
    fragmentShader: `uniform sampler2D tex, title; uniform float seed, lb; varying vec2 vUv;
      float h(vec2 p){ return fract(sin(dot(p, vec2(12.9898,78.233)) + seed*17.13) * 43758.5453); }
      void main(){
        vec2 uv = vUv; vec2 c = uv - 0.5;
        vec3 col = texture2D(tex, uv).rgb;
        float v = 1.0 - 0.5 * pow(clamp(dot(c*vec2(1.1,1.35), c*vec2(1.1,1.35))*1.6, 0.0, 1.0), 1.3);
        col *= v;
        vec2 gp = floor(gl_FragCoord.xy / 1.5);
        float g = (h(gp) + h(gp + 7.3) - 1.0) * 0.035;
        col += g * (0.6 + 0.4 * (1.0 - dot(col, vec3(0.33))));
        vec4 tt = texture2D(title, uv);
        col = mix(col, tt.rgb, tt.a);
        if (uv.y < lb || uv.y > 1.0 - lb) col = vec3(0.0);
        gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
      }`
  });
  quad.material = finalMat;
  quadScene.userData.quad = quad;
  window.DBG = { setScene, get composer() { return composer; }, renderer, scene, camera, bloom };
  window.READY = true;
}

function drawTitle(a) {
  const g = titleCtx; g.clearRect(0, 0, W, H);
  if (a > 0.001) {
    g.textAlign = 'center';
    g.font = '46px "Liberation Serif", serif';
    g.fillStyle = `rgba(232,216,196,${a})`;
    g.letterSpacing = '14px';
    g.fillText('THE ASHEN CANTOR', W / 2 + 7, H / 2 - 6);
    g.letterSpacing = '3px';
    g.font = '17px "Liberation Serif", serif';
    g.fillStyle = `rgba(180,150,140,${Math.max(0, a * 1.4 - 0.4)})`;
    g.fillText('creature concept  —  30 s animation test', W / 2, H / 2 + 34);
    g.fillStyle = `rgba(190,50,30,${a})`; g.fillRect(W / 2 - 190 * a, H / 2 + 12, 380 * a, 1);
  }
  titleTex.needsUpdate = true;
}

function setScene(i) {
  const s = sample(i), t = i / (META.fps * META.sub);
  poseCreature(s, t);
  poseCamera(s, t);
  const hip = jp(s, 'hip');
  moonLight.position.copy(hip).add(V3(700, 1400, -1500)); moonLight.target.position.copy(hip);
  fill.position.copy(camera.position).add(V3(-600, 300, 0)); fill.target.position.copy(hip);
  ENV.rim.position.copy(hip).add(V3(-500, 700, -1200)); ENV.rim.target.position.copy(hip);
  ENV.sky.position.set(camera.position.x, 0, camera.position.z);
  for (const fg of ENV.fog) {
    fg.tex.offset.x = (t * fg.sp) / 2000;
    fg.m.material.opacity = fg.front ? fg.op + 0.55 * sc(s, 'fog_front') : fg.op * (0.8 + 0.5 * sc(s, 'fog_front'));
  }
  for (const rb of ENV.ribbons) { rb.m.position.x = Math.sin(t * 0.3 + rb.ph) * 120; rb.m.position.y = Math.sin(t * 0.5 + rb.ph) * 20; }
  ENV.roseGlow.material.opacity = 0.3 + 0.08 * Math.sin(t * 1.7);
  const fl = 0.82 + 0.1 * Math.sin(t * 13.1) + 0.06 * Math.sin(t * 23.7 + 1.3) + 0.04 * Math.sin(t * 41.9 + 0.4);
  ENV.fireLight.intensity = 38000 * fl;
  ENV.fire.forEach((sp, k) => { const ph = t * (2.5 + k * 0.7) + k * 1.7; sp.position.set(Math.sin(ph) * 10, 175 + k * 14 + 8 * Math.sin(ph * 1.3), 0);
    const sz = (70 - k * 9) * (0.85 + 0.15 * Math.sin(ph * 2.1)); sp.scale.set(sz * 0.8, sz, 1); sp.material.opacity = (0.95 - k * 0.15) * fl; });
  updateFX(t, s, i);
  return s;
}

function project(v) {
  const p = v.clone().project(camera);
  return [(p.x + 1) / 2 * W, (1 - p.y) / 2 * H];
}

window.renderFrame = function (f, nsub = 3) {
  const base = f * META.sub;
  const subs = [];
  for (let k = nsub - 1; k >= 0; k--) subs.push(base - k);   // 180-degree shutter
  renderer.setRenderTarget(accum.rt); renderer.setClearColor(0x000000, 1); renderer.clear();
  const quad = quadScene.userData.quad;
  for (const i of subs) {
    setScene(i);
    composer.render();
    quad.material = accumMat; accumMat.uniforms.tex.value = composer.readBuffer.texture; accumMat.uniforms.w.value = 1 / subs.length;
    renderer.setRenderTarget(accum.rt); renderer.render(quadScene, quadCam);
  }
  const s = setScene(base);  // projection for QC at the frame's own time
  camera.updateMatrixWorld();
  drawTitle(sc(s, 'title'));
  quad.material = finalMat; finalMat.uniforms.seed.value = f;
  renderer.setRenderTarget(null); renderer.render(quadScene, quadCam);
  const pts = {};
  for (const n of ['head_tip', 'heart', 'neck2', 'n_hand_tip', 'f_f1_tip', 'n_toe', 'f_toe', 'hip'])
    pts[n] = project(jp(s, n, n.startsWith('n_') ? 22 : n.startsWith('f_') ? -22 : 0));
  C.head.updateMatrixWorld(true);
  pts.horns = C.hornTips.map(o => project(o.getWorldPosition(V3())));
  return pts;
};

init();
