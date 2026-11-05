// Drive the three.js renderer headlessly and write PNG frames + framing data.
//   node render3d.js --frames 0-119 [--workers 3] [--out ../out/frames] [--nsub 3]
const http = require('http');
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const ROOT = path.resolve(__dirname, '..');
const args = Object.fromEntries(process.argv.slice(2).reduce((a, v, i, arr) => (v.startsWith('--') ? a.concat([[v.slice(2), arr[i + 1]]]) : a), []));
const OUT = path.resolve(args.out || path.join(ROOT, 'out', 'frames'));
const PROJ = path.join(ROOT, 'out', 'proj');
const WORKERS = parseInt(args.workers || '3');
const NSUB = parseInt(args.nsub || '3');
function parseFrames(s) {
  const out = [];
  for (const part of s.split(',')) {
    const [a, b] = part.split('-').map(Number);
    for (let f = a; f <= (b ?? a); f++) out.push(f);
  }
  return out;
}
const frames = parseFrames(args.frames || '0');
fs.mkdirSync(OUT, { recursive: true }); fs.mkdirSync(PROJ, { recursive: true });

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.bin': 'application/octet-stream' };
const server = http.createServer((req, res) => {
  const p = path.join(ROOT, decodeURIComponent(req.url.split('?')[0]));
  if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { 'Content-Type': MIME[path.extname(p)] || 'application/octet-stream' });
  fs.createReadStream(p).pipe(res);
});

(async () => {
  await new Promise(r => server.listen(0, r));
  const port = server.address().port;
  const queue = frames.slice();
  let done = 0; const t0 = Date.now();
  const worker = async () => {
    const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
    page.on('console', m => { if (m.type() === 'error') console.error('[page]', m.text()); });
    page.on('pageerror', e => { console.error('[pageerror]', e.message); process.exitCode = 1; });
    await page.goto(`http://127.0.0.1:${port}/r3d/index.html`);
    await page.waitForFunction(() => window.READY === true, null, { timeout: 120000 });
    while (queue.length) {
      const f = queue.shift();
      const pts = await page.evaluate(([f, n]) => window.renderFrame(f, n), [f, NSUB]);
      const png = await page.locator('canvas').screenshot({ type: 'png' });
      fs.writeFileSync(path.join(OUT, String(f).padStart(5, '0') + '.png'), png);
      fs.writeFileSync(path.join(PROJ, String(f).padStart(5, '0') + '.json'), JSON.stringify(pts));
      done++;
      if (done % 20 === 0) console.log(`${done}/${frames.length} frames, ${((Date.now() - t0) / done / 1000 * WORKERS).toFixed(2)} s/frame/worker`);
    }
    await browser.close();
  };
  await Promise.all(Array.from({ length: Math.min(WORKERS, frames.length) }, worker));
  server.close();
  console.log(`rendered ${frames.length} frames in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
})();
