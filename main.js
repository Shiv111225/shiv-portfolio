import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const isMobile = window.innerWidth < 768;

// ---------- Renderer / scene ----------
const canvas = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x030305, 1);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x030305);
scene.fog = new THREE.FogExp2(0x030305, 0.08);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 0, 6);

// ---------- GLSL: 3D simplex noise (Ashima Arts / Stefan Gustavson, MIT) ----------
const noiseGLSL = /* glsl */ `
vec3 mod289(vec3 x){return x-floor(x*(1.0/289.0))*289.0;}
vec4 mod289(vec4 x){return x-floor(x*(1.0/289.0))*289.0;}
vec4 permute(vec4 x){return mod289(((x*34.0)+1.0)*x);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159-0.85373472095314*r;}
float snoise(vec3 v){
  const vec2 C=vec2(1.0/6.0,1.0/3.0);
  const vec4 D=vec4(0.0,0.5,1.0,2.0);
  vec3 i=floor(v+dot(v,C.yyy));
  vec3 x0=v-i+dot(i,C.xxx);
  vec3 g=step(x0.yzx,x0.xyz);
  vec3 l=1.0-g;
  vec3 i1=min(g.xyz,l.zxy);
  vec3 i2=max(g.xyz,l.zxy);
  vec3 x1=x0-i1+C.xxx;
  vec3 x2=x0-i2+C.yyy;
  vec3 x3=x0-D.yyy;
  i=mod289(i);
  vec4 p=permute(permute(permute(i.z+vec4(0.0,i1.z,i2.z,1.0))+i.y+vec4(0.0,i1.y,i2.y,1.0))+i.x+vec4(0.0,i1.x,i2.x,1.0));
  float n_=0.142857142857;
  vec3 ns=n_*D.wyz-D.xzx;
  vec4 j=p-49.0*floor(p*ns.z*ns.z);
  vec4 x_=floor(j*ns.z);
  vec4 y_=floor(j-7.0*x_);
  vec4 x=x_*ns.x+ns.yyyy;
  vec4 y=y_*ns.x+ns.yyyy;
  vec4 h=1.0-abs(x)-abs(y);
  vec4 b0=vec4(x.xy,y.xy);
  vec4 b1=vec4(x.zw,y.zw);
  vec4 s0=floor(b0)*2.0+1.0;
  vec4 s1=floor(b1)*2.0+1.0;
  vec4 sh=-step(h,vec4(0.0));
  vec4 a0=b0.xzyw+s0.xzyw*sh.xxyy;
  vec4 a1=b1.xzyw+s1.xzyw*sh.zzww;
  vec3 p0=vec3(a0.xy,h.x);
  vec3 p1=vec3(a0.zw,h.y);
  vec3 p2=vec3(a1.xy,h.z);
  vec3 p3=vec3(a1.zw,h.w);
  vec4 norm=taylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));
  p0*=norm.x;p1*=norm.y;p2*=norm.z;p3*=norm.w;
  vec4 m=max(0.6-vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)),0.0);
  m=m*m;
  return 42.0*dot(m*m,vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));
}
`;

// ---------- The symbiote: a living, displaced liquid-black mass ----------
const uniforms = {
  uTime: { value: 0 },
  uMouse: { value: new THREE.Vector3(10, 10, 10) }, // mouse point in object space
  uAgitation: { value: 0 },                          // spikes up with pointer speed / scroll
  uCharge: { value: 0 },                             // kinetic energy glow (grows on scroll)
};

const symbioteMat = new THREE.ShaderMaterial({
  uniforms,
  vertexShader: /* glsl */ `
    uniform float uTime;
    uniform vec3  uMouse;
    uniform float uAgitation;
    varying vec3  vNormal;
    varying vec3  vViewPos;
    varying vec3  vObjPos;
    varying float vDisp;
    ${noiseGLSL}

    float field(vec3 p){
      float t = uTime * 0.35;
      float n  = snoise(p * 1.1 + vec3(t, t * 0.6, -t)) * 0.32;
      n += snoise(p * 2.4 - vec3(t * 1.3)) * 0.08;
      // tendrils: sharp ridged noise that sharpens with agitation
      float ridge = 1.0 - abs(snoise(p * 3.2 + vec3(0.0, t * 2.0, 0.0)));
      n += pow(ridge, 8.0) * (0.06 + uAgitation * 0.35);
      // reach toward the pointer
      float d = distance(normalize(p), normalize(uMouse));
      n += smoothstep(0.9, 0.0, d) * (0.25 + uAgitation * 0.5);
      return n;
    }

    void main(){
      vec3 p = position;
      vec3 n = normalize(normal);
      float disp = field(p);
      vec3 displaced = p + n * disp;

      // recompute normal via finite differences on the tangent plane
      vec3 tangent = normalize(cross(n, abs(n.y) < 0.99 ? vec3(0.0,1.0,0.0) : vec3(1.0,0.0,0.0)));
      vec3 bitangent = normalize(cross(n, tangent));
      float e = 0.01;
      vec3 pa = p + tangent * e;  pa += normalize(pa) * field(pa);
      vec3 pb = p + bitangent * e; pb += normalize(pb) * field(pb);
      vec3 newNormal = normalize(cross(pa - displaced, pb - displaced));

      vDisp = disp;
      vObjPos = p;
      vNormal = normalize(normalMatrix * newNormal);
      vec4 mv = modelViewMatrix * vec4(displaced, 1.0);
      vViewPos = -mv.xyz;
      gl_Position = projectionMatrix * mv;
    }
  `,
  fragmentShader: /* glsl */ `
    uniform float uTime;
    uniform float uCharge;
    varying vec3  vNormal;
    varying vec3  vViewPos;
    varying vec3  vObjPos;
    varying float vDisp;
    ${noiseGLSL}

    void main(){
      vec3 N = normalize(vNormal);
      vec3 V = normalize(vViewPos);

      // oily black base with sharp wet speculars
      vec3 L1 = normalize(vec3(0.6, 0.8, 0.7));
      vec3 L2 = normalize(vec3(-0.7, -0.2, 0.5));
      float spec1 = pow(max(dot(reflect(-L1, N), V), 0.0), 90.0);
      float spec2 = pow(max(dot(reflect(-L2, N), V), 0.0), 40.0);
      float diff = max(dot(N, L1), 0.0) * 0.06;

      float fres = pow(1.0 - max(dot(N, V), 0.0), 3.0);

      vec3 base = vec3(0.012, 0.010, 0.018) + diff;
      vec3 silver = vec3(0.85, 0.85, 0.95);
      vec3 violet = vec3(0.62, 0.30, 1.0);

      // kinetic energy veins flowing across the surface
      float t = uTime * 0.5;
      float v = abs(snoise(vObjPos * 2.2 + vec3(0.0, t, t * 0.5)));
      float veins = smoothstep(0.03, 0.0, v);
      float v2 = abs(snoise(vObjPos * 4.5 - vec3(t * 0.8)));
      veins += smoothstep(0.02, 0.0, v2) * 0.5;
      float pulse = 0.55 + 0.45 * sin(uTime * 2.0 + vObjPos.y * 3.0);
      float veinStrength = (0.35 + uCharge * 0.9) * pulse;

      vec3 col = base;
      col += silver * spec1 * 0.7 + vec3(0.6, 0.5, 0.9) * spec2 * 0.12;
      col += violet * fres * (0.35 + uCharge * 0.35);
      col += violet * veins * veinStrength * 1.3;
      // crevices glow faintly where the mass folds in
      col += violet * smoothstep(0.0, -0.3, vDisp) * 0.15;

      gl_FragColor = vec4(col, 1.0);
    }
  `,
});

const detail = isMobile ? 48 : 96;
const symbiote = new THREE.Mesh(new THREE.IcosahedronGeometry(1.35, detail), symbioteMat);
scene.add(symbiote);

// ---------- Orbiting vibranium shards ----------
const shardGroup = new THREE.Group();
const shardMat = new THREE.MeshStandardMaterial({
  color: 0x1a1426, metalness: 1, roughness: 0.25, emissive: 0x6a2bd9, emissiveIntensity: 0.35,
});
const shardGeo = new THREE.OctahedronGeometry(0.08, 0);
const shardCount = isMobile ? 24 : 48;
const shards = [];
for (let i = 0; i < shardCount; i++) {
  const m = new THREE.Mesh(shardGeo, shardMat);
  const r = 2.3 + Math.random() * 1.6;
  const a = Math.random() * Math.PI * 2;
  const tilt = (Math.random() - 0.5) * 1.2;
  m.userData = { r, a, tilt, speed: 0.1 + Math.random() * 0.25, spin: Math.random() * 2 };
  m.scale.setScalar(0.5 + Math.random() * 1.4);
  shardGroup.add(m);
  shards.push(m);
}
scene.add(shardGroup);

scene.add(new THREE.AmbientLight(0x221133, 1.2));
const keyLight = new THREE.PointLight(0x9d4dff, 30, 20);
keyLight.position.set(3, 2, 4);
scene.add(keyLight);
const rimLight = new THREE.PointLight(0xc0c0ff, 15, 20);
rimLight.position.set(-4, -1, 2);
scene.add(rimLight);

// ---------- Floating spores (particles) ----------
const pCount = isMobile ? 600 : 1500;
const pPos = new Float32Array(pCount * 3);
const pSeed = new Float32Array(pCount);
for (let i = 0; i < pCount; i++) {
  pPos[i * 3] = (Math.random() - 0.5) * 22;
  pPos[i * 3 + 1] = (Math.random() - 0.5) * 22;
  pPos[i * 3 + 2] = (Math.random() - 0.5) * 14 - 2;
  pSeed[i] = Math.random();
}
const pGeo = new THREE.BufferGeometry();
pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
pGeo.setAttribute('aSeed', new THREE.BufferAttribute(pSeed, 1));
const pMat = new THREE.ShaderMaterial({
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  uniforms: { uTime: uniforms.uTime, uPixelRatio: { value: renderer.getPixelRatio() } },
  vertexShader: /* glsl */ `
    uniform float uTime; uniform float uPixelRatio;
    attribute float aSeed; varying float vA;
    void main(){
      vec3 p = position;
      p.y += mod(uTime * (0.05 + aSeed * 0.15) + aSeed * 22.0, 22.0) - 11.0 - position.y * 0.0;
      p.x += sin(uTime * 0.3 + aSeed * 40.0) * 0.3;
      vec4 mv = modelViewMatrix * vec4(p, 1.0);
      gl_Position = projectionMatrix * mv;
      gl_PointSize = (2.0 + aSeed * 4.0) * uPixelRatio * (6.0 / -mv.z);
      vA = 0.25 + aSeed * 0.6;
    }
  `,
  fragmentShader: /* glsl */ `
    varying float vA;
    void main(){
      float d = length(gl_PointCoord - 0.5);
      float a = smoothstep(0.5, 0.0, d) * vA;
      gl_FragColor = vec4(vec3(0.65, 0.4, 1.0), a);
    }
  `,
});
scene.add(new THREE.Points(pGeo, pMat));

// ---------- Post-processing ----------
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.7, 0.5, 0.55);
composer.addPass(bloom);
composer.addPass(new OutputPass());

// ---------- Interaction ----------
const pointer = new THREE.Vector2(0, 0);
const pointerTarget = new THREE.Vector2(0, 0);
let lastPointer = new THREE.Vector2(0, 0);
let agitationTarget = 0;
const raycaster = new THREE.Raycaster();
const hitSphere = new THREE.Sphere(new THREE.Vector3(), 1.6);
const tmpHit = new THREE.Vector3();

window.addEventListener('pointermove', (e) => {
  pointerTarget.set((e.clientX / window.innerWidth) * 2 - 1, -(e.clientY / window.innerHeight) * 2 + 1);
  const speed = pointerTarget.distanceTo(lastPointer);
  agitationTarget = Math.min(1, agitationTarget + speed * 4);
  lastPointer.copy(pointerTarget);
});
window.addEventListener('pointerdown', () => { agitationTarget = 1; });

let scrollProgress = 0;
const updateScroll = () => {
  const max = document.documentElement.scrollHeight - window.innerHeight;
  scrollProgress = max > 0 ? window.scrollY / max : 0;
};
window.addEventListener('scroll', updateScroll, { passive: true });
updateScroll();

// Blob drifts to alternate sides as sections change, so content stays readable
const layout = [
  { x: 2.0, y: 0, s: 1.0 },   // hero (text left)
  { x: -2.2, y: 0, s: 0.8 },  // about (card right)
  { x: 0, y: 0, s: 0.6 },     // skills (wide card – shrink behind)
  { x: 0, y: -0.2, s: 0.55 }, // projects (full width – shrink behind)
  { x: 2.2, y: 0, s: 0.85 },  // contact (card left)
];
if (isMobile) layout.forEach((l) => { l.x = 0; l.y = 0.8; l.s *= 0.7; });

function layoutAt(p) {
  const f = p * (layout.length - 1);
  const i = Math.min(Math.floor(f), layout.length - 2);
  const t = f - i;
  const e = t * t * (3 - 2 * t);
  const a = layout[i], b = layout[i + 1];
  return { x: a.x + (b.x - a.x) * e, y: a.y + (b.y - a.y) * e, s: a.s + (b.s - a.s) * e };
}

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  composer.setSize(window.innerWidth, window.innerHeight);
});

// ---------- Loop ----------
const clock = new THREE.Clock();
let lastScroll = 0;

function tick() {
  const dt = Math.min(clock.getDelta(), 0.05);
  const speedMul = reduceMotion ? 0.25 : 1;
  uniforms.uTime.value += dt * speedMul;
  const t = uniforms.uTime.value;

  pointer.lerp(pointerTarget, 0.08);

  // scroll speed also agitates the mass
  const scrollDelta = Math.abs(scrollProgress - lastScroll);
  lastScroll = scrollProgress;
  agitationTarget = Math.min(1, agitationTarget + scrollDelta * 25);
  agitationTarget *= 0.95;
  uniforms.uAgitation.value += (agitationTarget - uniforms.uAgitation.value) * 0.1;
  uniforms.uCharge.value += (scrollProgress - uniforms.uCharge.value) * 0.05;

  // position / scale from scroll
  const L = layoutAt(scrollProgress);
  symbiote.position.x += (L.x - symbiote.position.x) * 0.06;
  symbiote.position.y += (L.y - symbiote.position.y) * 0.06;
  const s = symbiote.scale.x + (L.s - symbiote.scale.x) * 0.06;
  symbiote.scale.setScalar(s);
  shardGroup.position.copy(symbiote.position);
  shardGroup.scale.setScalar(s);

  symbiote.rotation.y = t * 0.12 + pointer.x * 0.4;
  symbiote.rotation.x = pointer.y * 0.3;

  // pointer -> point on the blob (object space) so tendrils reach toward it
  raycaster.setFromCamera(pointer, camera);
  hitSphere.center.copy(symbiote.position);
  hitSphere.radius = 1.6 * s;
  if (raycaster.ray.intersectSphere(hitSphere, tmpHit)) {
    symbiote.worldToLocal(tmpHit);
    uniforms.uMouse.value.lerp(tmpHit, 0.15);
  } else {
    // aim at the nearest point on the ray to keep a gentle lean
    raycaster.ray.closestPointToPoint(symbiote.position, tmpHit);
    symbiote.worldToLocal(tmpHit);
    uniforms.uMouse.value.lerp(tmpHit.multiplyScalar(3), 0.05);
  }

  shards.forEach((m) => {
    const d = m.userData;
    const a = d.a + t * d.speed;
    m.position.set(Math.cos(a) * d.r, Math.sin(a * 1.3) * d.tilt + Math.sin(a) * d.tilt, Math.sin(a) * d.r * 0.6);
    m.rotation.x = t * d.spin;
    m.rotation.y = t * d.spin * 0.7;
  });
  shardGroup.rotation.z = 0.25;

  camera.position.x += (pointer.x * 0.3 - camera.position.x) * 0.05;
  camera.position.y += (pointer.y * 0.2 - camera.position.y) * 0.05;
  camera.lookAt(0, 0, 0);

  bloom.strength = 0.6 + uniforms.uCharge.value * 0.3 + uniforms.uAgitation.value * 0.3;

  composer.render();
  requestAnimationFrame(tick);
}
tick();

// ---------- DOM bits ----------
document.getElementById('year').textContent = new Date().getFullYear();

const io = new IntersectionObserver(
  (entries) => entries.forEach((e) => e.isIntersecting && e.target.classList.add('in')),
  { threshold: 0.2 }
);
document.querySelectorAll('.reveal').forEach((el) => io.observe(el));

requestAnimationFrame(() => document.getElementById('loader').classList.add('done'));
