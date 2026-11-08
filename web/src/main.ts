// coresample's page: a pan/zoom viewer over any multiscale source, click to search, hits stream
// into the side panel. Raw data only; nothing is precomputed.
import { PRESETS, type Layer, type Preset } from "./datasets.ts";
import { Embedder, type Focus } from "./embedder.ts";
import type { Dims, ScanInput } from "./scan.ts";
import { Search, coarseLevelFor, type MatchOptions, type Method, type Spread, type State } from "./search.ts";
import { GlobeViewer } from "./globeviewer.ts";
import { NgViewer } from "./ngviewer.ts";
import { OlViewer } from "./olviewer.ts";
import { OmeZarrSource, TileSource, legacySurvey, openZoomify, type Source, type Vec3 } from "./sources.ts";
import { PrecomputedSource } from "./precomputed.ts";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>("view");
const ctx = canvas.getContext("2d")!;
const hover = $<HTMLCanvasElement>("hover");
const hctx = hover.getContext("2d")!;
let mouse: { x: number; y: number } | null = null;  // device pixels over the canvas
const COARSE_BUDGET = 4e6;  // voxels of the coarse level scanned whole (the scan is on the path to first hits)

let src: Source | null = null;
let preset: Preset | null = null;
const view = { cx: 0, cy: 0, z: 0, scale: 1 };  // full-resolution pixels; scale = full-res px per screen px
let search: Search | null = null;
let olv: OlViewer | null = null;  // OpenLayers for tile pyramids
let ngv: NgViewer | null = null;
let ngBoxesKey = "";
let gv: GlobeViewer | null = null;  // CesiumJS globe for planetary mosaics (toggled)
const globeDiv = document.createElement("div");
globeDiv.style.cssText = "position:absolute;inset:0;display:none;background:#000";  // Neuroglancer for OME-Zarr (2-D and 3-D); the canvas viewer only as a fallback
const olDiv = document.createElement("div");
olDiv.style.cssText = "position:absolute;inset:0;display:none;background:#000";
canvas.after(olDiv);
olDiv.after(globeDiv);
/** Full-resolution pixels per device pixel, whichever viewer is showing. */
const zoomScale = () => (gv ? gv.scale() : olv ? olv.scale() : ngv ? ngv.scale() : view.scale);
let hitEls: HTMLElement[] = [];

// ---------------------------------------------------------------------------------------
// model (warmed up on load, so the first click does not wait for 88 MB)
// ---------------------------------------------------------------------------------------
/** A model file from Cache Storage when there, else downloaded once into it (88 MB each). */
async function cachedBytes(url: string): Promise<ArrayBuffer> {
  try {
    const cache = await caches.open("coresample-models-v3");  // v3: the 224 model outputs its tokens
    const hit = await cache.match(url);
    if (hit) return hit.arrayBuffer();
    const r = await fetch(url);
    if (!r.ok) throw new Error(`model: HTTP ${r.status}`);
    await cache.put(url, r.clone());
    return r.arrayBuffer();
  } catch {
    return (await fetch(url)).arrayBuffer();
  }
}
try { caches.delete("coresample-models-v2").catch(() => {}); } catch { /* no Cache Storage here */ }  // the previous models
const emb = new Embedder((size) => new URL(`models/dinov2-small-${size}.onnx`, document.baseURI).href,
                         (l) => console.log("[onnx]", l), cachedBytes);
const modelEl = $("model");
function setModel(kind: "loading" | "gpu" | "cpu" | "err", text: string, title = "") {
  modelEl.className = `model ${kind}`;
  modelEl.querySelector("span")!.textContent = text;
  modelEl.title = title || text;
}

/** Load the model at the chosen input size and measure it. */
let warmed: Promise<void> | null = null;
async function warmUp() {
  try {
    setModel("loading", `loading the model (${emb.input} px)`, `DINOv2-small at ${emb.input} px input: 88 MB the first time, then cached by the browser`);
    const t = performance.now();
    await emb.embed([{ data: new Float32Array(64 * 64), size: 64, channels: 1 }]);
    const n = 32, d = new Float32Array(64 * 64);
    const t2 = performance.now();
    await emb.embed(Array.from({ length: n }, () => ({ data: d, size: 64, channels: 1 })));
    const rate = n / ((performance.now() - t2) / 1000);
    const gpu = emb.provider === "webgpu";
    const detail = `DINOv2-small, ${emb.input} px input, ${rate.toFixed(0)} patches/s, ${((t2 - t) / 1000).toFixed(0)} s to load.`;
    setModel(gpu ? "gpu" : "cpu", `${gpu ? "GPU" : "CPU, slow"}: ${rate.toFixed(0)} patches/s`,
      gpu ? detail : detail + (window.isSecureContext
        ? " No WebGPU here: try Chrome on a machine with a GPU (on Linux, chrome://flags/#enable-unsafe-webgpu)."
        : " WebGPU needs https or localhost: open https://davidackerman.github.io/coresample/ instead."));
  } catch (e) {
    setModel("err", "model failed", (e as Error).message);
  }
}
/** The model loads once the viewer has had the network to itself for a while (its first data
 * comes first), or at the first click, whichever is sooner. */
const warm = () => (warmed ??= warmUp());
setTimeout(warm, 8000);

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
  const stage = canvas.parentElement!;  // the canvas is hidden under OpenLayers or Neuroglancer
  hover.width = Math.round(stage.clientWidth * dpr); hover.height = Math.round(stage.clientHeight * dpr);
  render();
  drawHover();
}
window.addEventListener("resize", resize);

const displayLevel = () => (src ? Math.max(0, Math.min(src.levels.length - 1, Math.floor(Math.log2(Math.max(zoomScale(), 1))))) : 0);

let gen = 0;
const off = document.createElement("canvas");
async function render() {
  if (!src) return;
  saveView();
  if (olv || ngv || gv) { overlays(); return; }
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
  if (ngv) {  // the boxes as a Neuroglancer annotation layer
    if (!search) { if (ngBoxesKey !== "none") { ngBoxesKey = "none"; ngv.setBoxes(null, []); } return; }
    const st = search.state, f = 2 ** st.working, side = st.patch * f, d3 = src?.dims === 3;
    const box = (c: Vec3, lab?: string) => ({ fz: d3 ? c[0] * f : 0, fy: c[1] * f, fx: c[2] * f, side, label: lab });
    const key = `${st.hits.length}|${st.hits[0]?.index}|${st.hits[st.hits.length - 1]?.index}`;
    if (key === ngBoxesKey) return;  // a state restore per change, not per frame
    ngBoxesKey = key;
    ngv.setBoxes(box(search.query, "query"), st.hits.slice(0, 30).map((h, k) => box(h.center, `#${k + 1}`)));
    return;
  }
  if (gv) {  // the boxes as outlines on the globe
    if (!search) { gv.setBoxes(null, []); return; }
    const st = search.state, f = 2 ** st.working, side = st.patch * f;
    const box = (c: Vec3, lab?: string) => ({ fx: c[2] * f, fy: c[1] * f, side, label: lab });
    gv.setBoxes(box(search.query, "query"), st.hits.slice(0, 30).map((h, k) => box(h.center, String(k + 1))));
    return;
  }
  if (olv) {  // the boxes as OpenLayers features
    if (!search) { olv.setBoxes(null, []); return; }
    const st = search.state, f = 2 ** st.working, side = st.patch * f;
    const box = (c: Vec3, lab?: string) => ({ fx: c[2] * f, fy: c[1] * f, side, label: lab });
    olv.setBoxes(box(search.query, "query"), st.hits.slice(0, 30).map((h, k) => box(h.center, String(k + 1))));
    return;
  }
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

/** The level a click searches at (the one on screen, leaving room for coarser levels). */
const levelSel = $<HTMLSelectElement>("level");
const searchLevel = () => {
  if (!src) return 0;
  const chosen = levelSel.value === "auto" ? displayLevel() : Number(levelSel.value);
  return Math.max(0, Math.min(src.levels.length - 2, chosen));
};
levelSel.addEventListener("change", () => drawHover());
const patchSel = $<HTMLSelectElement>("patch");
const patchSeg = $("patchseg");
function showPatch() { for (const b of patchSeg.querySelectorAll("button")) b.classList.toggle("on", b.textContent === patchSel.value); }
patchSeg.addEventListener("click", (e) => {
  const b = (e.target as HTMLElement).closest("button");
  if (!b) return;
  patchSel.value = b.textContent!;
  showPatch();
  patchSel.dispatchEvent(new Event("change"));
});
showPatch();
/** Level menu for this source: each level with its voxel (pixel) size. */
function fillLevels(s: Source) {
  levelSel.innerHTML = '<option value="auto" selected>auto (as viewed)</option>';
  s.levels.slice(0, -1).forEach((L, i) => {
    const v = L.scale[2], u = (s.unit || "").toLowerCase();
    const nm = u.startsWith("nano") ? v : u.startsWith("micro") ? v * 1e3 : u === "m" ? v * 1e9 : NaN;
    const txt = !Number.isNaN(nm) ? (nm >= 1e9 ? `${(nm / 1e9).toFixed(nm >= 1e10 ? 0 : 1)} m` : nm >= 1e3 ? `${(nm / 1e3).toFixed(1)} \u00b5m` : `${nm.toFixed(1)} nm`)
              : u.startsWith("arcsec") ? `${v.toFixed(2)}\u2033` : `${2 ** i}x`;
    levelSel.add(new Option(`${i}: ${txt}/px`, String(i)));
  });
}

/** A length in the source's units (its voxel size's unit) as readable text; "" when unknown. */
function fmtLen(v: number): string {
  const u = (src?.unit || "").toLowerCase();
  const nm = u.startsWith("nano") ? 1 : u.startsWith("micro") ? 1e3 : u === "m" || u.startsWith("meter") || u.startsWith("metre") ? 1e9 : NaN;
  if (!Number.isNaN(nm)) {
    const x = v * nm;
    return x >= 1e12 ? `${(x / 1e12).toFixed(1)} km` : x >= 1e9 ? `${(x / 1e9).toFixed(x >= 1e10 ? 0 : 1)} m` : x >= 1e6 ? `${(x / 1e6).toFixed(1)} mm`
         : x >= 1e3 ? `${(x / 1e3).toFixed(x >= 1e4 ? 0 : 1)} \u00b5m` : `${x.toFixed(0)} nm`;
  }
  return u.startsWith("arcsec") ? (v >= 60 ? `${(v / 60).toFixed(1)}\u2032` : `${v.toFixed(0)}\u2033`) : "";
}
/** How far a hit is from the query, physically (or in voxels of the search level). */
function distance(c: Vec3, q: Vec3, W: number): { v: number; text: string } {
  const sc = src!.levels[W].scale, d3 = src!.dims === 3;
  const v = Math.hypot(d3 ? (c[0] - q[0]) * sc[0] : 0, (c[1] - q[1]) * sc[1], (c[2] - q[2]) * sc[2]);
  const t = fmtLen(v);
  return { v, text: t ? `${t} away` : `${Math.round(Math.hypot(d3 ? c[0] - q[0] : 0, c[1] - q[1], c[2] - q[2]))} px away` };
}

/** A patch's physical size at the search level, in readable units. */
function patchSize(): string {
  if (!src) return "";
  const P = Number($<HTMLSelectElement>("patch").value), W = searchLevel(), phys = fmtLen(P * src.levels[W].scale[2]);
  return `${P} px at level ${W}` + (phys ? ` = ${phys}` : "");
}

/** The dashed box under the cursor: exactly the region a click would search for. */
function drawHover() {
  scheduleWhere();
  $("psize").textContent = src ? `px at level ${searchLevel()}` + (patchSize().includes("= ") ? ` = ${patchSize().split("= ")[1]}` : "") : "";
  if (ngv) { ngv.drawHover(); return; }  // Neuroglancer's mouse is in its frame; it draws the box
  if (gv) return;  // the globe draws its own
  hctx.clearRect(0, 0, hover.width, hover.height);
  if (!src || !mouse || drag?.moved || olv) return;
  const P = Number($<HTMLSelectElement>("patch").value), side = (P * 2 ** searchLevel()) / view.scale;
  const dpr = window.devicePixelRatio || 1;
  hctx.setLineDash([6 * dpr, 4 * dpr]);
  hctx.lineWidth = 2 * dpr;
  hctx.strokeStyle = "#5ab0ff";
  hctx.strokeRect(mouse.x - side / 2, mouse.y - side / 2, side, side);
  hctx.setLineDash([]);
  const label = patchSize();
  hctx.font = `${12 * dpr}px system-ui`;
  const tw = hctx.measureText(label).width;
  const lx = mouse.x - side / 2, ly = mouse.y - side / 2 - 6 * dpr;
  hctx.fillStyle = "#000a";
  hctx.fillRect(lx - 3 * dpr, ly - 13 * dpr, tw + 6 * dpr, 17 * dpr);
  hctx.fillStyle = "#cfe6ff";
  hctx.fillText(label, lx, ly);
}

// ---------------------------------------------------------------------------------------
// coordinates: under the cursor, for the query, typed in to go somewhere
// ---------------------------------------------------------------------------------------
/** Longitude and latitude of a full-resolution pixel on a map of a body, or null. */
function lonLat(fx: number, fy: number): [number, number] | null {
  if (!src || !preset?.geo) return null;
  const W = src.levels[0].shape[2], H = src.levels[0].shape[1];
  if (preset.geo === "equirect") return [(fx / W) * 360 - 180, 90 - (fy / H) * 180];
  const lat = (Math.atan(Math.sinh(Math.PI * (1 - (2 * fy) / W))) * 180) / Math.PI;  // web mercator: a square world
  return [(fx / W) * 360 - 180, lat];
}
function pixelOf(lon: number, lat: number): [number, number] | null {
  if (!src || !preset?.geo) return null;
  const W = src.levels[0].shape[2], H = src.levels[0].shape[1];
  if (preset.geo === "equirect") return [((lon + 180) / 360) * W, ((90 - lat) / 180) * H];
  const r = (lat * Math.PI) / 180;
  return [((lon + 180) / 360) * W, ((1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2) * W];
}
/** "x 2,999,500  y 833,700  z 2,666" plus degrees on a body or the physical position. */
function coordText(fx: number, fy: number, fz?: number): string {
  if (!src) return "";
  const n = (v: number) => Math.round(v).toLocaleString("en-US");
  let t = `x ${n(fx)}  y ${n(fy)}` + (src.dims === 3 && fz !== undefined ? `  z ${n(fz)}` : "");
  const ll = lonLat(fx, fy);
  if (ll) t += `  (${Math.abs(ll[1]).toFixed(3)}\u00b0 ${ll[1] >= 0 ? "N" : "S"}, ${Math.abs(ll[0]).toFixed(3)}\u00b0 ${ll[0] >= 0 ? "E" : "W"})`;
  else {
    const sc = src.levels[0].scale, parts = [fx * sc[2], fy * sc[1], ...(src.dims === 3 && fz !== undefined ? [fz * sc[0]] : [])].map(fmtLen);
    if (parts.every(Boolean)) t += `  (${parts.join(", ")})`;
  }
  return t;
}
function showCursor(fx: number, fy: number, fz?: number) { $("cursor").textContent = coordText(fx, fy, fz); }
$("goto").addEventListener("keydown", (e) => {
  if ((e as KeyboardEvent).key !== "Enter" || !src) return;
  const v = ($<HTMLInputElement>("goto").value.match(/-?\d+(\.\d+)?/g) ?? []).map(Number);
  if (v.length < 2) return;
  let fx = v[0], fy = v[1], fz = v[2];
  // on a map of a body, two numbers that look like degrees are latitude, longitude
  if (preset?.geo && v.length === 2 && Math.abs(v[0]) <= 90 && Math.abs(v[1]) <= 360) {
    const p = pixelOf(v[1] > 180 ? v[1] - 360 : v[1], v[0]);
    if (p) [fx, fy] = p;
  }
  jumpTo(fx, fy, fz);
});
/** Centre every viewer on a full-resolution position, keeping the zoom. */
function jumpTo(fx: number, fy: number, fz?: number) {
  if (!src) return;
  const scale = zoomScale(), z = fz ?? (ngv ? ngv.center().z : view.z);
  if (gv) gv.goTo(fx, fy, scale);
  if (olv) olv.goTo(fx, fy, scale);
  if (ngv) ngv.goTo(fx, fy, z, null);
  view.cx = fx; view.cy = fy;
  if (src.dims === 3) { view.z = Math.round(z); zIn.value = String(view.z); $("zval").textContent = zIn.value; }
  render();
  saveView();
}
/** The view in the address bar (?v=x,y,z,scale), so a link opens this place. */
let viewTimer = 0;
function saveView() {
  clearTimeout(viewTimer);
  viewTimer = window.setTimeout(() => {
    const v = currentView();
    if (!v || !src) return;
    const u = new URL(location.href);
    u.searchParams.set("v", [(v.x0 + v.x1) / 2, (v.y0 + v.y1) / 2, v.z, zoomScale()].map((x) => +x.toFixed(2)).join(","));
    history.replaceState(null, "", u);
  }, 600);
}

// pan / zoom / click
let drag: { x: number; y: number; cx: number; cy: number; moved: boolean } | null = null;
canvas.addEventListener("pointerdown", (e) => {
  canvas.setPointerCapture(e.pointerId);
  drag = { x: e.clientX, y: e.clientY, cx: view.cx, cy: view.cy, moved: false };
});
canvas.addEventListener("pointermove", (e) => {
  const rr = canvas.getBoundingClientRect(), dd = window.devicePixelRatio || 1;
  mouse = { x: (e.clientX - rr.left) * dd, y: (e.clientY - rr.top) * dd };
  showCursor(view.cx + (mouse.x - canvas.width / 2) * view.scale, view.cy + (mouse.y - canvas.height / 2) * view.scale, view.z);
  drawHover();
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
canvas.addEventListener("pointerleave", () => { mouse = null; drawHover(); });
$("patch").addEventListener("change", drawHover);
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
  drawHover();
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
/** 8.8 T, 122 G, 268 M: a pixel or voxel count, short. */
const short = (v: number) => (v >= 1e12 ? `${(v / 1e12).toFixed(1)} T` : v >= 1e9 ? `${(v / 1e9).toFixed(v >= 1e11 ? 0 : 1)} G` : v >= 1e6 ? `${(v / 1e6).toFixed(0)} M` : `${(v / 1e3).toFixed(0)} k`);
const dsBtn = $<HTMLButtonElement>("dsbtn"), dsMenu = $("dsmenu"), dsList = $("dslist"), sortSeg = $("sort");
let sortKey = "size";
try { sortKey = localStorage.getItem("coresample-sort") ?? "size"; } catch { /* no storage here */ }
function fillDatasets() {
  for (const b of sortSeg.querySelectorAll("button")) b.classList.toggle("on", b.dataset.k === sortKey);
  const list = [...PRESETS].sort(sortKey === "name" ? (a, b) => a.label.localeCompare(b.label)
    : sortKey === "kind" ? (a, b) => a.kind.localeCompare(b.kind) || b.count - a.count : (a, b) => b.count - a.count);
  const lmax = Math.log10(Math.max(...PRESETS.map((p) => p.count))), lmin = Math.log10(Math.min(...PRESETS.map((p) => p.count))) - 0.5;
  dsList.innerHTML = "";
  for (const p of list) {
    const b = document.createElement("button");
    b.className = "dsrow" + (preset?.id === p.id ? " on" : "");
    b.setAttribute("role", "option");
    const w = (100 * (Math.log10(p.count) - lmin)) / (lmax - lmin);
    b.innerHTML = `<span class="nm"></span><span class="sz">${short(p.count)}<small>${p.volume ? "vox" : "px"}</small></span>`
                + `<span class="meta"></span><span class="bar-size"><i style="width:${w.toFixed(0)}%"></i></span>`;
    b.querySelector(".nm")!.textContent = p.label;
    b.querySelector(".meta")!.textContent = `${p.kind}, ${p.detail}`;
    b.addEventListener("click", () => { dsMenu.hidden = true; openPreset(p); });
    dsList.appendChild(b);
  }
}
sortSeg.addEventListener("click", (e) => {
  const k = (e.target as HTMLElement).closest("button")?.dataset.k;
  if (!k) return;
  sortKey = k;
  try { localStorage.setItem("coresample-sort", k); } catch { /* no storage here */ }
  fillDatasets();
});
dsBtn.addEventListener("click", () => { dsMenu.hidden = !dsMenu.hidden; if (!dsMenu.hidden) fillDatasets(); });
document.addEventListener("pointerdown", (e) => { if (!dsMenu.hidden && !(e.target as HTMLElement).closest(".dpick")) dsMenu.hidden = true; });
window.addEventListener("keydown", (e) => { if (e.key === "Escape") dsMenu.hidden = true; });
$("open").addEventListener("click", () => { dsMenu.hidden = true; openCustom($<HTMLInputElement>("custom").value); });
$("custom").addEventListener("keydown", (e) => { if ((e as KeyboardEvent).key === "Enter") $("open").click(); });

let ngAvailable: Promise<boolean> | null = null;
/** Whether this site serves Neuroglancer at ./ng/ (the deployed build, or the dev relay). */
function hasNeuroglancer() {
  return (ngAvailable ??= fetch("ng/index.html", { method: "HEAD" }).then((r) => r.ok && (r.headers.get("content-type") ?? "").includes("html")).catch(() => false));
}

/** "W x H px (8.8 Tpx), 13 levels" or the voxel count for a volume. */
function sizeText(s: Source): string {
  const [z, y, x] = s.levels[0].shape, n = s.dims === 3 ? x * y * z : x * y;
  const si = (v: number) => (v >= 1e12 ? `${(v / 1e12).toFixed(1)} T` : v >= 1e9 ? `${(v / 1e9).toFixed(1)} G` : v >= 1e6 ? `${(v / 1e6).toFixed(0)} M` : `${(v / 1e3).toFixed(0)} k`);
  const f = (v: number) => v.toLocaleString("en-US");
  return s.dims === 3 ? `${f(x)} x ${f(y)} x ${f(z)} voxels (${si(n)}vox), ${s.levels.length} levels`
                      : `${f(x)} x ${f(y)} px (${si(n)}px), ${s.levels.length} levels`;
}

async function setSource(s: Source, v: Preset["view"], p: Preset | null) {
  clearSearch();
  src = s;
  preset = p;
  Object.assign(view, { cx: v.cx, cy: v.cy, z: v.z ?? Math.floor(s.levels[0].shape[0] / 2), scale: v.scale });
  gv?.destroy();
  gv = null;
  globeDiv.style.display = "none";
  globeBtn.textContent = "globe";
  globeBtn.style.display = p?.globe ? "" : "none";
  olv?.destroy();
  olv = null;
  ngv?.destroy();
  ngv = null;
  ngBoxesKey = "";
  const tiled = s instanceof TileSource, volume = (s instanceof OmeZarrSource || s instanceof PrecomputedSource) && (await hasNeuroglancer());
  olDiv.style.display = tiled ? "block" : "none";
  canvas.style.display = tiled || volume ? "none" : "block";
  hover.style.display = tiled ? "none" : "block";
  if (volume) {
    ngv = new NgViewer(canvas.parentElement!, s as OmeZarrSource | PrecomputedSource, { cx: v.cx, cy: v.cy, z: v.z ?? view.z, scale: v.scale }, hover, {
      click: (fz, fy, fx) => queryAt(fx, fy, fz),
      hoverBox: () => (src ? { side: Number($<HTMLSelectElement>("patch").value) * 2 ** searchLevel(), text: patchSize() } : null),
      moved: () => { drawHover(); saveView(); },
      pointer: (fx, fy, fz) => showCursor(fx, fy, fz),
    }, p?.layers ?? []);
    hover.parentElement!.appendChild(hover);  // the hover canvas above the viewer's frame
    $("zbar").style.display = "none";  // Neuroglancer moves through z itself
  }
  if (tiled) {
    olv = new OlViewer(olDiv, s as TileSource, v, {
      click: (fx, fy) => queryAt(fx, fy),
      hover: () => (src ? { side: Number($<HTMLSelectElement>("patch").value) * 2 ** searchLevel(), text: patchSize() } : null),
      moved: () => { drawHover(); saveView(); },
      pointer: (fx, fy) => showCursor(fx, fy),
    });
  }
  $("attrib").textContent = `${s.name}: ${sizeText(s)}` + (s.attribution ? `. Data: ${s.attribution}` : "");
  $("hint").textContent = (p?.note ? p.note + " " : "") + "Drag to pan, scroll to zoom, click to search.";
  $("hint").classList.remove("gone");
  $("dsname").textContent = p ? p.label : s.name;
  const [z0, y0, x0] = s.levels[0].shape;
  $("dssize").textContent = `${short(s.dims === 3 ? x0 * y0 * z0 : x0 * y0)}${s.dims === 3 ? "vox" : "px"}`;
  const zb = $("zbar");
  zb.style.display = s.dims === 3 && !ngv ? "flex" : "none";
  if (s.dims === 3) { zIn.max = String(s.levels[0].shape[0] - 1); zIn.value = String(view.z); $("zval").textContent = zIn.value; }
  if (p?.patch) { patchSel.value = String(p.patch); showPatch(); }
  fillLayers(ngv ? p?.layers ?? [] : []);
  fillLevels(s);
  const u = new URL(location.href);
  u.searchParams.set("d", p ? p.id : "custom");
  u.searchParams.delete("v");  // a view belongs to its dataset
  history.replaceState(null, "", u);
  resize();
}

/** A view from the address bar (?v=x,y,z,scale), once, for the first dataset opened. */
let startView: Preset["view"] | null = (() => {
  const v = (new URL(location.href).searchParams.get("v") ?? "").split(",").map(Number);
  return v.length === 4 && v.every(Number.isFinite) ? { cx: v[0], cy: v[1], z: v[2], scale: v[3] } : null;
})();
function takeStartView() { const v = startView; startView = null; return v; }

async function openPreset(p: Preset) {
  $("phase").textContent = `Opening ${p.label}`;
  try { await setSource(await p.open(), takeStartView() ?? p.view, p); idle(); }
  catch (e) { $("phase").textContent = "Could not open the data"; $("status").textContent = (e as Error).message; }
}
function idle() {
  $("phase").textContent = "Click on the image to search";
  $("status").textContent = "It reads the best-ranked tiles first and keeps going until it has read the whole region or you pause it.";
}

async function openCustom(text: string) {
  text = text.trim();
  if (!text) return;
  const openBtn = $<HTMLButtonElement>("open");
  $("phase").textContent = "Opening your data";
  $("status").textContent = text;
  openBtn.disabled = true; openBtn.textContent = "Opening...";
  try {
    let s: Source;
    const m = /^legacy:\s*([-\d.]+)\s*,\s*([-\d.]+)/i.exec(text);
    if (m) s = legacySurvey(Number(m[1]), Number(m[2]));
    else if (/^precomputed:\/\//i.test(text)) s = await PrecomputedSource.open(text);
    else if (/\{z\}/.test(text) && /\{x\}/.test(text) && /\{y\}/.test(text)) {
      // a web-map tile template: square world, 256 px tiles; "#maxZoom=N" sets the finest zoom
      const [tpl, frag] = text.split("#"), mz = Number(new URLSearchParams(frag ?? "").get("maxZoom") ?? 18);
      s = new TileSource({ name: tpl.replace(/^https?:\/\//, "").split("/")[0], tileSize: 256, maxZoom: mz, channels: 3,
                           url: (z, x, y) => tpl.replace("{z}", String(z)).replace("{x}", String(x)).replace("{y}", String(y)) });
    } else if (/ImageProperties\.xml$/i.test(text) || /^zoomify:/i.test(text)) {
      const base = text.replace(/^zoomify:/i, "").replace(/\/?ImageProperties\.xml$/i, "");
      s = await openZoomify(base, base.split("/").filter(Boolean).pop() ?? "Zoomify image");
    } else {
      // OME-Zarr, or else a precomputed volume (an info file at the URL)
      try { s = await OmeZarrSource.open(text); }
      catch (ez) { try { s = await PrecomputedSource.open(text); } catch { throw ez; } }
    }
    const L0 = s.levels[0].shape;
    await setSource(s, takeStartView() ?? { cx: L0[2] / 2, cy: L0[1] / 2, z: Math.floor(L0[0] / 2), scale: Math.max(1, Math.max(L0[2], L0[1]) / 1600) }, null);
    const u = new URL(location.href);  // a link that opens this data again
    u.searchParams.delete("d"); u.searchParams.set("url", text);
    history.replaceState(null, "", u);
    idle();
  } catch (e) {
    const msg = (e as Error).message || String(e);
    $("phase").textContent = "Could not open the data";
    $("status").textContent = (/fetch/i.test(msg) ? "The host did not answer, or does not allow reads from web pages (CORS). " : "") + msg
      + "\nReadable: OME-Zarr, Neuroglancer precomputed (precomputed://...), web-map tiles, Zoomify, legacy:RA,Dec.";
  } finally {
    openBtn.disabled = false; openBtn.textContent = "Open";
  }
}

/** The extra layers (segmentations...) as toggles; they show in Neuroglancer. */
function fillLayers(layers: Layer[]) {
  const box = $("layers");
  box.innerHTML = "";
  $("layersec").hidden = !layers.length;
  for (const l of layers) {
    const lab = document.createElement("label");
    lab.title = `${l.kind}: ${l.url}`;
    lab.innerHTML = `<input type="checkbox"><i></i><span></span>`;
    const cb = lab.querySelector("input")!;
    cb.checked = !!l.visible;
    (lab.querySelector("i") as HTMLElement).style.background = l.color;
    lab.querySelector("span")!.textContent = l.name;
    cb.addEventListener("change", () => ngv?.setLayerVisible(l.name, cb.checked));
    box.appendChild(lab);
  }
}

// ---------------------------------------------------------------------------------------
// search
// ---------------------------------------------------------------------------------------
function clearSearch() {
  search?.stop();
  search = null;
  hitEls = [];
  $("hits").innerHTML = '<div class="empty">Hits appear here as tiles are read, best first. Hover one to mark it good or bad: the list re-ranks.</div>';
  $("hitcount").textContent = "";
  const stopBtn = $<HTMLButtonElement>("stop");
  stopBtn.disabled = true; stopBtn.textContent = "Pause";
  $<HTMLButtonElement>("clear").disabled = true;
  ($("bar").firstElementChild as HTMLElement).style.width = "0";
  $("qcanvas").classList.add("none");
  $<HTMLCanvasElement>("qcanvas").getContext("2d")?.clearRect(0, 0, 999, 999);
  $("link").style.display = "none";
  $("wheresec").hidden = true;
  render();
}
$("clear").addEventListener("click", () => { clearSearch(); idle(); });
$("stop").addEventListener("click", () => {  // pause, or carry on from where it paused
  if (!search) return;
  if (search.state.phase === "read") search.stop();
  else if (search.state.phase === "paused") search.resume();
});

function regionFor(W: number, P: number): [Vec3, Vec3] {
  const s = src!, L = s.levels[W], f = 2 ** W, d3 = s.dims === 3;
  const whole: [Vec3, Vec3] = [[0, 0, 0], [...L.shape] as Vec3];
  const vis = gv ? gv.extentPx() : olv ? olv.extentPx() : ngv ? ngv.extentPx() : { x0: view.cx - (canvas.width * view.scale) / 2, x1: view.cx + (canvas.width * view.scale) / 2,
                                      y0: view.cy - (canvas.height * view.scale) / 2, y1: view.cy + (canvas.height * view.scale) / 2 };
  const around = (k: number): [Vec3, Vec3] => {
    const hw = ((vis.x1 - vis.x0) * k) / 2 / f, hh = ((vis.y1 - vis.y0) * k) / 2 / f;
    const cx = (vis.x0 + vis.x1) / 2 / f, cy = (vis.y0 + vis.y1) / 2 / f;
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

function query(px: number, py: number) {  // canvas viewer: device pixels to full resolution
  queryAt(view.cx + (px - canvas.width / 2) * view.scale, view.cy + (py - canvas.height / 2) * view.scale);
}

function queryAt(fx: number, fy: number, fz?: number) {
  if (!src) return;
  const s = src, n = s.levels.length;
  const P = Number($<HTMLSelectElement>("patch").value);
  const W = searchLevel();   // search at the level being viewed (as the hover box shows)
  const f = 2 ** W;
  const c: Vec3 = [s.dims === 3 ? Math.floor((fz ?? view.z) / f) : 0, Math.floor(fy / f), Math.floor(fx / f)];
  const L = s.levels[W].shape;
  if (c[1] < P / 2 || c[2] < P / 2 || c[1] > L[1] - P / 2 || c[2] > L[2] - P / 2) { $("status").textContent = "too close to the edge"; return; }
  clearSearch();
  warm();
  $("hint").classList.add("gone");
  const region = regionFor(W, P);
  search = new Search(s, emb, c, { working: W, patch: P, region, maxCoarseVoxels: COARSE_BUDGET, scanFn: scanInWorker, topN: 60,
                                   match: matchOptions(), method: $<HTMLSelectElement>("method").value as Method, spread: spread() });
  $<HTMLButtonElement>("clear").disabled = false;
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

const PHASE: Record<State["phase"], string> = { query: "Reading the query", scan: "Quick pass over a coarse level", read: "Searching",
                                                paused: "Paused", done: "Done", error: "Something went wrong" };
const num = (v: number) => v.toLocaleString("en-US");

function panel(st: State) {
  if (!search || !src) return;
  const s = search, C = src.channels, P = st.patch;
  if (st.phase !== "query") { drawPatch($<HTMLCanvasElement>("qcanvas"), s.patchPixels(-1), P, C); $("qcanvas").classList.remove("none"); }
  // tiles read, discounted by the patches still waiting to be compared
  const frac = st.tilesTotal ? (st.tilesRead / st.tilesTotal) * (st.patches / Math.max(1, st.patches + st.queued)) : 0;
  const pct = frac >= 0.1 ? (100 * frac).toFixed(0) : frac >= 0.01 ? (100 * frac).toFixed(1) : (100 * frac).toFixed(2);
  const [rlo, rhi] = s.regionBox, Ls = src.levels[st.working].shape;
  const whole = rlo.every((v) => v === 0) && rhi[1] >= Ls[1] && rhi[2] >= Ls[2] && (src.dims !== 3 || rhi[0] >= Ls[0]);
  const what = whole ? (src.dims === 3 ? "the volume" : "the image") : "the search area";
  $("phase").textContent = st.phase === "read" || st.phase === "paused" ? `${PHASE[st.phase]}: ${pct}% of ${what} read`
                         : st.phase === "error" ? PHASE.error : PHASE[st.phase];
  $("phase").title = `How much of ${what} has been read at full detail (level ${st.working}) and compared, best-looking parts first; `
                   + "the map below shows where. The area is set under Options \u25b8 region (auto: the whole dataset when the quick first pass can cover it).";
  const lines = st.phase === "error" ? [st.message] : [
    st.tilesTotal ? `${num(st.patches)} ${s.opt.method === "tokens" ? "windows" : "patches"} compared in ${num(st.tilesRead)} of ${num(st.tilesTotal)} tiles` : st.message,
    `${fmt(st.bytes)} read in ${st.seconds.toFixed(0)} s` + (st.firstHitSeconds ? `, first hits after ${st.firstHitSeconds.toFixed(1)} s` : ""),
    st.coarse >= 0 ? `level ${st.working}, ranked by a quick pass over level ${st.coarse} (${st.scanSeconds.toFixed(1)} s)` : "",
    `query at ${coordText(s.query[2] * 2 ** st.working, s.query[1] * 2 ** st.working, s.query[0] * 2 ** st.working)}`,
  ];
  $("status").textContent = lines.filter(Boolean).join("\n");
  ($("bar").firstElementChild as HTMLElement).style.width = `${(100 * frac).toFixed(2)}%`;
  const stopBtn = $<HTMLButtonElement>("stop");
  stopBtn.disabled = !(st.phase === "read" || st.phase === "paused");
  stopBtn.textContent = st.phase === "paused" ? "Resume" : "Pause";
  drawWhere(st);
  const box = $("hits");
  const sp = spread();
  $("hitcount").textContent = !st.hits.length ? "" : `${st.hits.length} best` + (sp.between > 1 ? `, ${sp.between} boxes apart` : ", no overlaps")
    + (sp.fromQuery > 0 ? `, ${sp.fromQuery}+ boxes from the query` : "");
  if (!st.hits.length) return;
  if (box.querySelector(".empty")) box.innerHTML = "";
  const q = s.query;
  const shown = st.hits.map((h, rank) => ({ h, rank, d: distance(h.center, q, st.working) }));
  if (hitSort !== "best") shown.sort((a, b) => (hitSort === "near" ? a.d.v - b.d.v : b.d.v - a.d.v));
  shown.forEach(({ h, rank, d }, k) => {
    let el = hitEls[k];
    if (!el) {
      el = document.createElement("div");
      el.className = "hit";
      el.innerHTML = `<canvas title="go there"></canvas><div class="fb"><button data-g="1" title="good: more like this">&#10003;</button><button data-g="0" title="bad: fewer like this">&#10005;</button></div><div class="cap"><b></b><span></span></div><div class="z"></div>`;
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
    el.querySelector(".cap b")!.textContent = `#${rank + 1}`;
    el.querySelector(".cap span")!.textContent = h.score.toFixed(3);
    el.querySelector(".z")!.textContent = d.text + (src?.dims === 3 ? `, z ${num(h.center[0] * 2 ** st.working)}` : "");
  });
  for (let k = st.hits.length; k < hitEls.length; k++) hitEls[k].remove();
  hitEls.length = st.hits.length;
}

// ---------------------------------------------------------------------------------------
// where it has looked: the search region from above (x-y) and, for a volume, from the side
// (x-z), with the tiles read, the hits, the query and the view
// ---------------------------------------------------------------------------------------
const whereCv = $<HTMLCanvasElement>("where");
let wherePanels: { a: number; b: number; x0: number; y0: number; sa: number; sb: number; w: number; h: number }[] = [];
function drawWhere(st: State) {
  if (!search || !src || !st.tilesTotal) { $("wheresec").hidden = true; return; }
  $("wheresec").hidden = false;
  const dpr = window.devicePixelRatio || 1, W = st.working, f = 2 ** W, ch = src.levels[W].chunk, d3 = src.dims === 3;
  const [rlo, rhi] = search.regionBox, ext = [rhi[0] - rlo[0], rhi[1] - rlo[1], rhi[2] - rlo[2]];
  // side by side, sharing the vertical (y) axis: from above (x across) and from the side (z across);
  // one scale across, y stretched to a readable height if the volume is thin in y
  const cw = Math.max(200, whereCv.clientWidth) * dpr, maxH = 150 * dpr, minH = 56 * dpr, gap = 10 * dpr, top = 14 * dpr;
  const specs = d3 ? [{ a: 2, name: "from above (x, y)" }, { a: 0, name: "from the side (z, y)" }] : [{ a: 2, name: "" }];
  const across = specs.reduce((t, sp) => t + ext[sp.a], 0);
  const sa = Math.min((cw - 2 - gap * (specs.length - 1)) / across, maxH / Math.max(1, ext[1]));
  const sb = Math.max(sa, minH / Math.max(1, ext[1])), h = ext[1] * sb, y0 = d3 ? top : 0;
  let x = (cw - (across * sa + gap * (specs.length - 1))) / 2;
  wherePanels = specs.map((sp) => { const w = ext[sp.a] * sa, p = { a: sp.a, b: 1, x0: x, y0, sa, sb, w, h }; x += w + gap; return p; });
  whereCv.width = cw; whereCv.height = y0 + h + 2;
  whereCv.style.height = `${(y0 + h + 2) / dpr}px`;
  const g = whereCv.getContext("2d")!;
  g.clearRect(0, 0, whereCv.width, whereCv.height);
  const tiles = search.readTiles(), view = currentView();
  wherePanels.forEach((p, k) => {
    const X = (v: number) => p.x0 + (v - rlo[p.a]) * p.sa, Y = (v: number) => p.y0 + (v - rlo[p.b]) * p.sb;
    if (specs[k].name) { g.fillStyle = "#5f6977"; g.font = `${10.5 * dpr}px system-ui`; g.fillText(specs[k].name, p.x0, p.y0 - 4 * dpr); }
    g.fillStyle = "#0b0e12"; g.fillRect(p.x0, p.y0, p.w, p.h);
    g.strokeStyle = "#313946"; g.lineWidth = dpr; g.strokeRect(p.x0, p.y0, p.w, p.h);
    g.save(); g.beginPath(); g.rect(p.x0, p.y0, p.w, p.h); g.clip();  // everything inside the region's outline
    g.fillStyle = "#5ab0ff"; g.globalAlpha = 0.16;  // read tiles: overlapping columns add up
    for (const t of tiles) g.fillRect(X(t[p.a] * ch[p.a]), Y(t[p.b] * ch[p.b]), Math.max(1, ch[p.a] * p.sa), Math.max(1, ch[p.b] * p.sb));
    g.globalAlpha = 1;
    if (view) {  // what is on screen: a rectangle from above, the slice from the side
      g.strokeStyle = "#ffffffcc"; g.lineWidth = dpr;
      if (p.a === 0) { g.beginPath(); g.moveTo(X(view.z / f), Y(view.y0 / f)); g.lineTo(X(view.z / f), Y(view.y1 / f)); g.stroke(); }
      else g.strokeRect(X(view.x0 / f), Y(view.y0 / f), ((view.x1 - view.x0) / f) * p.sa, ((view.y1 - view.y0) / f) * p.sb);
    }
    st.hits.slice(0, 30).forEach((hh, i) => {
      g.fillStyle = "#ffd24a";
      g.beginPath(); g.arc(X(hh.center[p.a]), Y(hh.center[p.b]), (i < 3 ? 3.2 : 2.2) * dpr, 0, 7); g.fill();
    });
    const q = search!.query;
    g.strokeStyle = "#5ab0ff"; g.lineWidth = 2 * dpr;
    g.beginPath(); g.arc(X(q[p.a]), Y(q[p.b]), 4.5 * dpr, 0, 7); g.stroke();
    g.restore();
  });
}
let whereQueued = false;
/** Redraw the map at the next frame (the view moved). */
function scheduleWhere() {
  if (whereQueued || !search) return;
  whereQueued = true;
  requestAnimationFrame(() => { whereQueued = false; if (search) drawWhere(search.state); });
}
/** The view in full-resolution pixels (x and y extent, z slice), whichever viewer shows it. */
function currentView(): { x0: number; x1: number; y0: number; y1: number; z: number } | null {
  if (!src) return null;
  const e = gv ? gv.extentPx() : olv ? olv.extentPx() : ngv ? ngv.extentPx()
    : { x0: view.cx - (canvas.width * view.scale) / 2, x1: view.cx + (canvas.width * view.scale) / 2,
        y0: view.cy - (canvas.height * view.scale) / 2, y1: view.cy + (canvas.height * view.scale) / 2 };
  return { ...e, z: ngv ? ngv.center().z : view.z };
}
whereCv.addEventListener("click", (e) => {  // go to the place clicked, keeping the other coordinate
  if (!search || !src) return;
  const r = whereCv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  const px = (e.clientX - r.left) * dpr, py = (e.clientY - r.top) * dpr, f = 2 ** search.state.working;
  const p = wherePanels.find((q) => px >= q.x0 && px <= q.x0 + q.w && py >= q.y0 && py <= q.y0 + q.h);
  if (!p) return;
  const [rlo] = search.regionBox, va = rlo[p.a] + (px - p.x0) / p.sa, vb = rlo[p.b] + (py - p.y0) / p.sb;
  const v = currentView()!, cur = { x: (v.x0 + v.x1) / 2, z: v.z };
  // from above: x and y; from the side: z and y (x stays)
  const fx = p.a === 2 ? va * f : cur.x, fy = vb * f, fz = p.a === 0 ? va * f : cur.z, scale = zoomScale();
  if (gv) gv.goTo(fx, fy, scale);
  if (olv) olv.goTo(fx, fy, scale);
  if (ngv) ngv.goTo(fx, fy, fz, null);
  view.cx = fx; view.cy = fy;
  if (src.dims === 3) { view.z = Math.round(fz); zIn.value = String(view.z); $("zval").textContent = zIn.value; }
  render();
  drawWhere(search.state);
});

function goTo(i: number) {
  if (!search || !src) return;
  const c = search.center(i), f = 2 ** search.state.working;
  if (gv) gv.goTo(c[2] * f, c[1] * f, f);
  if (olv) olv.goTo(c[2] * f, c[1] * f, f);
  if (ngv) ngv.goTo(c[2] * f, c[1] * f, c[0] * f, f);
  view.cx = c[2] * f; view.cy = c[1] * f;
  if (src.dims === 3) { view.z = c[0] * f; zIn.value = String(view.z); $("zval").textContent = zIn.value; }
  view.scale = f;
  const link = src.link?.(search.state.working, c);
  const a = $<HTMLAnchorElement>("link");
  if (link) { a.href = link; a.style.display = ""; } else a.style.display = "none";
  render();
}

// match options (test 5 compared them on labelled EM; see README "How matching works")
function matchOptions(): MatchOptions {
  return { focus: $<HTMLSelectElement>("focus").value as Focus, rotations: $<HTMLInputElement>("rot").checked,
           scales: $<HTMLInputElement>("scl").checked, adapt: $<HTMLInputElement>("adapt").checked };
}
for (const id of ["focus", "rot", "scl", "adapt"]) $(id).addEventListener("change", () => search?.setMatch(matchOptions()));
function spread(): Spread { return { fromQuery: Number($<HTMLSelectElement>("far").value), between: Number($<HTMLSelectElement>("between").value) }; }
for (const id of ["far", "between"]) $(id).addEventListener("change", () => search?.setSpread(spread()));
let hitSort = "best";
$("hitsort").addEventListener("click", (e) => {
  const k = (e.target as HTMLElement).closest("button")?.dataset.k;
  if (!k) return;
  hitSort = k;
  for (const b of $("hitsort").querySelectorAll("button")) b.classList.toggle("on", b.dataset.k === k);
  if (search) panel(search.state);
});
$("method").addEventListener("change", () => {  // candidates are embedded differently: search again
  if (search) { const q = search.query, f = 2 ** search.state.working; queryAt(q[2] * f, q[1] * f, q[0] * f); }
});
$("detail").addEventListener("change", async () => {
  const det = $<HTMLSelectElement>("detail");
  det.dataset.touched = "1";
  emb.input = Number(det.value);
  await warmUp();
  if (search) { const q = search.query, f = 2 ** search.state.working; queryAt(q[2] * f, q[1] * f, q[0] * f); }  // embeddings differ: search again
});

// the map / globe toggle (planetary mosaics)
const globeBtn = $<HTMLButtonElement>("globe");
globeBtn.addEventListener("click", async () => {
  if (!src || !preset?.globe || !(src instanceof TileSource)) return;
  if (gv) {  // back to the map, at the globe's place
    const e = gv.extentPx(), s = gv.scale();
    gv.destroy(); gv = null;
    globeDiv.style.display = "none"; olDiv.style.display = "block";
    olv?.goTo((e.x0 + e.x1) / 2, (e.y0 + e.y1) / 2, s);
    globeBtn.textContent = "globe";
  } else {
    const at = olv ? { ...olv.center(), scale: olv.scale() } : { cx: view.cx, cy: view.cy, scale: view.scale };
    globeBtn.textContent = "map";
    olDiv.style.display = "none"; globeDiv.style.display = "block";
    gv = new GlobeViewer(globeDiv, src, preset.globe, at, {
      click: (fx, fy) => queryAt(fx, fy),
      hover: () => (src ? { side: Number($<HTMLSelectElement>("patch").value) * 2 ** searchLevel(), text: patchSize() } : null),
      moved: () => { drawHover(); saveView(); },
      pointer: (fx, fy) => showCursor(fx, fy),
    });
    try { await gv.ready; overlays(); } catch (err) { $("status").textContent = String((err as Error).message ?? err); }
  }
});

// for debugging from the console
(window as any).coresample = { get ngv() { return ngv; }, get olv() { return olv; }, get gv() { return gv; }, get search() { return search; } };

// start
const params = new URL(location.href).searchParams, want = params.get("d"), own = params.get("url");
if (own) { $<HTMLInputElement>("custom").value = own; openCustom(own); }
else {
  const first = PRESETS.find((p) => p.id === want) ?? PRESETS[0];
  openPreset(first);
}
