// coresample's page: a pan/zoom viewer over any multiscale source, click to search, hits stream
// into the side panel. Raw data only; nothing is precomputed.
import { PRESETS, type Preset } from "./datasets.ts";
import { Embedder } from "./embedder.ts";
import type { Dims, ScanInput } from "./scan.ts";
import { Search, coarseLevelFor, type State } from "./search.ts";
import { OmeZarrSource, legacySurvey, type Source, type Vec3 } from "./sources.ts";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>("view");
const ctx = canvas.getContext("2d")!;
const COARSE_BUDGET = 4e6;  // voxels of the coarse level scanned whole (the scan is on the path to first hits)

let src: Source | null = null;
let preset: Preset | null = null;
const view = { cx: 0, cy: 0, z: 0, scale: 1 };  // full-resolution pixels; scale = full-res px per screen px
let search: Search | null = null;
let hitEls: HTMLElement[] = [];

// ---------------------------------------------------------------------------------------
// model (warmed up on load, so the first click does not wait for 88 MB)
// ---------------------------------------------------------------------------------------
const emb = new Embedder(new URL("models/dinov2-small-112.onnx", document.baseURI).href, (l) => console.log("[onnx]", l));
const modelEl = $("model");
(async () => {
  try {
    modelEl.textContent = "model: downloading DINOv2-small (88 MB, once)";
    const t = performance.now();
    await emb.embed([{ data: new Float32Array(64 * 64), size: 64, channels: 1 }]);
    const n = 32, d = new Float32Array(64 * 64);
    const t2 = performance.now();
    await emb.embed(Array.from({ length: n }, () => ({ data: d, size: 64, channels: 1 })));
    const rate = n / ((performance.now() - t2) / 1000);
    modelEl.textContent = `model: ${emb.provider === "webgpu" ? "GPU (WebGPU)" : "CPU (WebAssembly, slow)"} ready, ${rate.toFixed(0)} patches/s (${((t2 - t) / 1000).toFixed(0)} s to load)`;
    if (emb.provider !== "webgpu")
      modelEl.title = window.isSecureContext
        ? "No WebGPU here: try Chrome on a machine with a GPU (on Linux, chrome://flags/#enable-unsafe-webgpu)."
        : "WebGPU needs https or localhost: open https://davidackerman.github.io/coresample/ instead.";
  } catch (e) {
    modelEl.textContent = `model failed: ${(e as Error).message}`;
  }
})();

// ---------------------------------------------------------------------------------------
// the cheap scan in a worker
// ---------------------------------------------------------------------------------------
const worker = new Worker(new URL("./scan.worker.ts", import.meta.url), { type: "module" });
let wid = 0;
const waiting = new Map<number, (r: { score: Float32Array; dims: Dims } | Error) => void>();
worker.onmessage = (e) => {
  const { id, score, dims, error } = e.data;
  waiting.get(id)?.(error ? new Error(error) : { score, dims });
  waiting.delete(id);
};
function scanInWorker(inp: ScanInput): Promise<{ score: Float32Array; dims: Dims }> {
  const id = ++wid;
  return new Promise((resolve, reject) => {
    waiting.set(id, (r) => (r instanceof Error ? reject(r) : resolve(r)));
    const vol = Float32Array.from(inp.vol), query = Float32Array.from(inp.query);
    worker.postMessage({ id, inp: { ...inp, vol, query } }, [vol.buffer, query.buffer]);
  });
}

// ---------------------------------------------------------------------------------------
// viewer
// ---------------------------------------------------------------------------------------
function resize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(canvas.clientWidth * dpr);
  canvas.height = Math.round(canvas.clientHeight * dpr);
  render();
}
window.addEventListener("resize", resize);

const displayLevel = () => (src ? Math.max(0, Math.min(src.levels.length - 1, Math.floor(Math.log2(Math.max(view.scale, 1))))) : 0);

let gen = 0;
const off = document.createElement("canvas");
async function render() {
  if (!src) return;
  const g = ++gen;
  const L = displayLevel();
  // a coarse level first (instant from cache when panning), then the one for this zoom
  for (const lv of L + 2 < src.levels.length ? [L + 2, L] : [L]) {
    await drawLevel(lv, g);
    if (g !== gen) return;
  }
}

async function drawLevel(L: number, g: number) {
  const s0 = src!;
  const W = canvas.width, H = canvas.height, f = 2 ** L;
  const x0 = Math.floor((view.cx - (W / 2) * view.scale) / f) - 1, y0 = Math.floor((view.cy - (H / 2) * view.scale) / f) - 1;
  const w = Math.ceil((W * view.scale) / f) + 3, h = Math.ceil((H * view.scale) / f) + 3;
  if (w * h > 12e6) return;
  const z = s0.dims === 3 ? Math.floor(view.z / f) : 0;
  const r = await s0.read(L, [z, y0, x0], [z + 1, y0 + h, x0 + w]);
  if (g !== gen) return;
  const img = new ImageData(w, h), C = r.channels;
  for (let i = 0, n = w * h; i < n; i++) {
    const v = r.data[i * C];
    img.data[4 * i] = v;
    img.data[4 * i + 1] = C === 3 ? r.data[i * C + 1] : v;
    img.data[4 * i + 2] = C === 3 ? r.data[i * C + 2] : v;
    img.data[4 * i + 3] = 255;
  }
  off.width = w; off.height = h;
  off.getContext("2d")!.putImageData(img, 0, 0);
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, W, H);
  ctx.imageSmoothingEnabled = view.scale > f * 0.75;
  const sx = (x0 * f - view.cx) / view.scale + W / 2, sy = (y0 * f - view.cy) / view.scale + H / 2;
  ctx.drawImage(off, sx, sy, (w * f) / view.scale, (h * f) / view.scale);
  overlays();
}

function toScreen(fx: number, fy: number) {
  return [(fx - view.cx) / view.scale + canvas.width / 2, (fy - view.cy) / view.scale + canvas.height / 2];
}

function overlays() {
  if (!search || !src) return;
  const st = search.state, Wl = st.working, f = 2 ** Wl, P = st.patch, side = (P * f) / view.scale;
  const d3 = src.dims === 3;
  const box = (c: Vec3, color: string, label: string, lw = 2) => {
    if (d3 && Math.abs(c[0] * f - view.z) > (P * f) / 2) return;
    const [x, y] = toScreen(c[2] * f, c[1] * f);
    ctx.strokeStyle = color; ctx.lineWidth = lw;
    ctx.strokeRect(x - side / 2, y - side / 2, side, side);
    if (label) { ctx.fillStyle = color; ctx.font = `${Math.max(11, 12 * (window.devicePixelRatio || 1))}px system-ui`; ctx.fillText(label, x - side / 2 + 2, y - side / 2 - 3); }
  };
  st.hits.slice(0, 30).forEach((h, k) => box(h.center, "#ffd24a", String(k + 1)));
  box(search.query, "#5ab0ff", "query", 3);
}

// pan / zoom / click
let drag: { x: number; y: number; cx: number; cy: number; moved: boolean } | null = null;
canvas.addEventListener("pointerdown", (e) => {
  canvas.setPointerCapture(e.pointerId);
  drag = { x: e.clientX, y: e.clientY, cx: view.cx, cy: view.cy, moved: false };
});
canvas.addEventListener("pointermove", (e) => {
  if (!drag) return;
  const dpr = window.devicePixelRatio || 1, dx = (e.clientX - drag.x) * dpr, dy = (e.clientY - drag.y) * dpr;
  if (Math.abs(dx) + Math.abs(dy) > 4) drag.moved = true;
  if (drag.moved) { view.cx = drag.cx - dx * view.scale; view.cy = drag.cy - dy * view.scale; render(); }
});
canvas.addEventListener("pointerup", (e) => {
  if (drag && !drag.moved) {
    const r = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    query((e.clientX - r.left) * dpr, (e.clientY - r.top) * dpr);
  }
  drag = null;
});
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  if (!src) return;
  const r = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  const px = (e.clientX - r.left) * dpr, py = (e.clientY - r.top) * dpr;
  const [fx, fy] = [view.cx + (px - canvas.width / 2) * view.scale, view.cy + (py - canvas.height / 2) * view.scale];
  const k = Math.exp(e.deltaY * 0.0015), max = 2 ** (src.levels.length - 1) * 2;
  view.scale = Math.min(max, Math.max(0.25, view.scale * k));
  view.cx = fx - (px - canvas.width / 2) * view.scale;
  view.cy = fy - (py - canvas.height / 2) * view.scale;
  render();
}, { passive: false });

const zIn = $<HTMLInputElement>("z");
zIn.addEventListener("input", () => { view.z = Number(zIn.value); $("zval").textContent = zIn.value; render(); });
window.addEventListener("keydown", (e) => {
  if (!src || src.dims !== 3 || (e.target as HTMLElement).tagName === "INPUT") return;
  const step = 2 ** displayLevel();
  if (e.key === "]" || e.key === "PageDown") view.z = Math.min(src.levels[0].shape[0] - 1, view.z + step);
  else if (e.key === "[" || e.key === "PageUp") view.z = Math.max(0, view.z - step);
  else return;
  zIn.value = String(view.z); $("zval").textContent = zIn.value;
  render();
});

// ---------------------------------------------------------------------------------------
// datasets
// ---------------------------------------------------------------------------------------
const sel = $<HTMLSelectElement>("dataset");
for (const p of PRESETS) sel.add(new Option(p.label, p.id));
sel.addEventListener("change", () => openPreset(PRESETS.find((p) => p.id === sel.value)!));
$("open").addEventListener("click", () => openCustom($<HTMLInputElement>("custom").value));

async function setSource(s: Source, v: Preset["view"], p: Preset | null) {
  clearSearch();
  src = s;
  preset = p;
  Object.assign(view, { cx: v.cx, cy: v.cy, z: v.z ?? Math.floor(s.levels[0].shape[0] / 2), scale: v.scale });
  $("attrib").textContent = s.attribution ? `Data: ${s.attribution}` : s.name;
  $("hint").textContent = (p?.note ? p.note + " " : "") + "Drag to pan, scroll to zoom, click to search.";
  const zb = $("zbar");
  zb.style.display = s.dims === 3 ? "flex" : "none";
  if (s.dims === 3) { zIn.max = String(s.levels[0].shape[0] - 1); zIn.value = String(view.z); $("zval").textContent = zIn.value; }
  if (p?.patch) $<HTMLSelectElement>("patch").value = String(p.patch);
  const u = new URL(location.href);
  u.searchParams.set("d", p ? p.id : "custom");
  history.replaceState(null, "", u);
  resize();
}

async function openPreset(p: Preset) {
  $("status").textContent = `opening ${p.label}...`;
  try { await setSource(await p.open(), p.view, p); $("status").textContent = "Click on the image to search."; }
  catch (e) { $("status").textContent = `could not open: ${(e as Error).message}`; }
}

async function openCustom(text: string) {
  text = text.trim();
  if (!text) return;
  try {
    let s: Source;
    const m = /^legacy:\s*([-\d.]+)\s*,\s*([-\d.]+)/i.exec(text);
    if (m) s = legacySurvey(Number(m[1]), Number(m[2]));
    else s = await OmeZarrSource.open(text);
    const L0 = s.levels[0].shape;
    await setSource(s, { cx: L0[2] / 2, cy: L0[1] / 2, z: Math.floor(L0[0] / 2), scale: Math.max(1, Math.max(L0[2], L0[1]) / 1600) }, null);
  } catch (e) {
    $("status").textContent = `could not open: ${(e as Error).message}`;
  }
}

// ---------------------------------------------------------------------------------------
// search
// ---------------------------------------------------------------------------------------
function clearSearch() {
  search?.stop();
  search = null;
  hitEls = [];
  $("hits").innerHTML = '<div class="empty">Hits appear here as tiles are read. Mark them good or bad to re-rank.</div>';
  for (const b of ["stop", "more", "clear"]) $<HTMLButtonElement>(b).disabled = true;
  ($("bar").firstElementChild as HTMLElement).style.width = "0";
  $("link").style.display = "none";
  render();
}
$("clear").addEventListener("click", () => { clearSearch(); $("status").textContent = "Click on the image to search."; });
$("stop").addEventListener("click", () => search?.stop());
$("more").addEventListener("click", () => {
  if (!search) return;
  const st = search.state;
  search.readUntil(Math.min(st.tilesTotal, Math.max(st.tilesBudget * 2, st.tilesBudget + 8)));
});

function regionFor(W: number, P: number): [Vec3, Vec3] {
  const s = src!, L = s.levels[W], f = 2 ** W, d3 = s.dims === 3;
  const whole: [Vec3, Vec3] = [[0, 0, 0], [...L.shape] as Vec3];
  const around = (k: number): [Vec3, Vec3] => {
    const hw = (canvas.width * view.scale * k) / 2 / f, hh = (canvas.height * view.scale * k) / 2 / f;
    const cx = view.cx / f, cy = view.cy / f;
    const lo: Vec3 = [0, Math.max(0, Math.floor(cy - hh)), Math.max(0, Math.floor(cx - hw))];
    const hi: Vec3 = [d3 ? L.shape[0] : 1, Math.min(L.shape[1], Math.ceil(cy + hh)), Math.min(L.shape[2], Math.ceil(cx + hw))];
    return [lo, hi];
  };
  const mode = $<HTMLSelectElement>("region").value;
  if (mode === "whole") return whole;
  if (mode !== "auto") return around(Number(mode));
  for (const r of [whole, around(16), around(8), around(4), around(2), around(1)])
    if (coarseLevelFor(s, W, P, r, COARSE_BUDGET) >= 0) return r;
  return around(1);
}

function query(px: number, py: number) {
  if (!src) return;
  const s = src, n = s.levels.length;
  const P = Number($<HTMLSelectElement>("patch").value);
  const W = Math.max(0, Math.min(n - 2, displayLevel()));   // search at the level being viewed
  const f = 2 ** W;
  const fx = view.cx + (px - canvas.width / 2) * view.scale, fy = view.cy + (py - canvas.height / 2) * view.scale;
  const c: Vec3 = [s.dims === 3 ? Math.floor(view.z / f) : 0, Math.floor(fy / f), Math.floor(fx / f)];
  const L = s.levels[W].shape;
  if (c[1] < P / 2 || c[2] < P / 2 || c[1] > L[1] - P / 2 || c[2] > L[2] - P / 2) { $("status").textContent = "too close to the edge"; return; }
  clearSearch();
  const region = regionFor(W, P);
  search = new Search(s, emb, c, { working: W, patch: P, region, budgetFrac: Number($<HTMLSelectElement>("budget").value),
                                   maxCoarseVoxels: COARSE_BUDGET, scanFn: scanInWorker, topN: 60, maxTiles: 48 });
  for (const b of ["stop", "clear"]) $<HTMLButtonElement>(b).disabled = false;
  let queued = false;
  const me = search;
  search.on(() => {  // at most one panel update per frame, always with the latest state
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; if (search === me) { panel(me.state); render(); } });
  });
  search.start();
}

function drawPatch(cv: HTMLCanvasElement, pix: Uint8Array, P: number, C: number) {
  cv.width = P; cv.height = P;
  const img = new ImageData(P, P);
  for (let i = 0; i < P * P; i++) {
    img.data[4 * i] = pix[i * C];
    img.data[4 * i + 1] = pix[i * C + (C === 3 ? 1 : 0)];
    img.data[4 * i + 2] = pix[i * C + (C === 3 ? 2 : 0)];
    img.data[4 * i + 3] = 255;
  }
  cv.getContext("2d")!.putImageData(img, 0, 0);
}

const fmt = (b: number) => (b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${(b / 1e3).toFixed(0)} KB`);

function panel(st: State) {
  if (!search || !src) return;
  const s = search, C = src.channels, P = st.patch;
  if (st.phase !== "query") drawPatch($<HTMLCanvasElement>("qcanvas"), s.patchPixels(-1), P, C);
  const pct = st.tilesTotal ? (100 * st.tilesRead) / st.tilesTotal : 0;
  const lines = [
    `<b>${st.phase}</b>: ${st.message}`,
    st.coarse >= 0 ? `scanned level ${st.coarse} in ${st.scanSeconds.toFixed(1)} s; searching level ${st.working}` : "",
    st.tilesTotal ? `tiles read <b>${st.tilesRead}</b> of ${st.tilesTotal} (${pct.toFixed(2)}%), ${st.patches} patches embedded` +
      (st.queued ? `, ${st.queued} queued` : "") : "",
    `${fmt(st.bytes)} downloaded, ${st.seconds.toFixed(1)} s` + (st.firstHitSeconds ? `, first hits at ${st.firstHitSeconds.toFixed(1)} s` : ""),
  ];
  $("status").innerHTML = lines.filter(Boolean).join("\n");
  ($("bar").firstElementChild as HTMLElement).style.width = `${st.tilesBudget ? (100 * st.tilesRead) / st.tilesBudget : 0}%`;
  $<HTMLButtonElement>("stop").disabled = st.phase !== "read";
  $<HTMLButtonElement>("more").disabled = !(st.phase === "paused");
  const box = $("hits");
  if (!st.hits.length) return;
  if (box.querySelector(".empty")) box.innerHTML = "";
  st.hits.forEach((h, k) => {
    let el = hitEls[k];
    if (!el) {
      el = document.createElement("div");
      el.className = "hit";
      el.innerHTML = `<canvas></canvas><div class="cap"><span></span><span><button data-g="1">good</button> <button data-g="0">bad</button></span></div>`;
      box.appendChild(el);
      hitEls[k] = el;
      el.querySelector("canvas")!.addEventListener("click", () => goTo(Number(el!.dataset.index)));
      el.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
        const i = Number(el!.dataset.index), good = b.dataset.g === "1";
        search?.mark(i, good);
      }));
    }
    if (el.dataset.index !== String(h.index)) {
      el.dataset.index = String(h.index);
      drawPatch(el.querySelector("canvas")!, s.patchPixels(h.index), P, C);
    }
    el.querySelector(".cap span")!.textContent = `#${k + 1} ${h.score.toFixed(3)}`;
  });
  for (let k = st.hits.length; k < hitEls.length; k++) hitEls[k].remove();
  hitEls.length = st.hits.length;
}

function goTo(i: number) {
  if (!search || !src) return;
  const c = search.center(i), f = 2 ** search.state.working;
  view.cx = c[2] * f; view.cy = c[1] * f;
  if (src.dims === 3) { view.z = c[0] * f; zIn.value = String(view.z); $("zval").textContent = zIn.value; }
  view.scale = f;
  const link = src.link?.(search.state.working, c);
  const a = $<HTMLAnchorElement>("link");
  if (link) { a.href = link; a.style.display = ""; } else a.style.display = "none";
  render();
}

// start
const want = new URL(location.href).searchParams.get("d");
const first = PRESETS.find((p) => p.id === want) ?? PRESETS[0];
sel.value = first.id;
openPreset(first);
