// coresample's page: a pan/zoom viewer over any multiscale source, click to search, hits stream
// into the side panel. Raw data only; nothing is precomputed.
import { PRESETS, type Layer, type Preset } from "./datasets.ts";
import { Embedder, type Focus } from "./embedder.ts";
import { scanInWorkers } from "./scanpool.ts";
import { DETAIL, NOWHERE, Search, coarseLevelFor, type MatchOptions, type Method, type Spread, type State } from "./search.ts";
import type { Patch, TokenGrid } from "./embedder.ts";
import { SketchPad, tonesOf, type Tone } from "./sketchpad.ts";
import { BrushPad } from "./brushpad.ts";
import { BRUSH } from "./embedder.ts";
import { GlobeViewer } from "./globeviewer.ts";
import { NgViewer } from "./ngviewer.ts";
import { OlViewer } from "./olviewer.ts";
import { OmeZarrSource, TileSource, legacySurvey, openZoomify, type Source, type Vec3 } from "./sources.ts";
import { countEvent, countOpen } from "./count.ts";
import { PrecomputedSource } from "./precomputed.ts";
import { canPick, entryFolder, findImages, handleFolder, pickFolder, serveFolder, type Folder } from "./local.ts";
import { IMAGE_FILE, imageFolder } from "./imagefile.ts";
import { HEAT_GRADIENT, heatCanvas, heatGrid, heatRange } from "./heat.ts";

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
let boxesKey = "", boxesAt = 0, boxesTimer = 0, boxesNow = false;
/** Whether the viewer's boxes should be redrawn: when they changed; while the search reads, at
 * most once a second (each new hit renumbers the ones below it, and redrawing on every one made
 * the boxes flicker), at once for a new example, a mark, or when the search stops. */
function boxesDue(key: string): boolean {
  if (key === boxesKey) return false;
  const ph = search?.state.phase, reading = ph === "read" || ph === "scan";
  const wait = 1000 - (performance.now() - boxesAt), sameQuery = key.split("|")[0] === boxesKey.split("|")[0];
  if (reading && sameQuery && !boxesNow && wait > 0) {
    clearTimeout(boxesTimer);
    boxesTimer = window.setTimeout(overlays, wait);
    return false;
  }
  clearTimeout(boxesTimer);
  boxesKey = key; boxesAt = performance.now(); boxesNow = false;
  return true;
}
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
/** What the model is doing while it is not ready yet (shown by the model chip and the panel). */
let modelNote: string | null = null;
function noteModel(text: string | null) {
  modelNote = text;
  if (text) setModel("loading", text, "DINOv2-small: 88 MB the first time, then cached by the browser");
  if (search?.state.phase === "query") panel(search.state);
}
/** The model file, from the browser's cache or downloaded (with progress) and cached. */
async function cachedBytes(url: string): Promise<ArrayBuffer> {
  let cache: Cache | null = null;
  try {
    cache = await caches.open("coresample-models-v3");  // v3: the 224 model outputs its tokens
    const hit = await cache.match(url);
    if (hit) { noteModel("starting the model"); return hit.arrayBuffer(); }
  } catch { cache = null; /* no Cache Storage here */ }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`model: HTTP ${r.status}`);
  const total = Number(r.headers.get("content-length")) || 0, parts: Uint8Array[] = [];
  let got = 0, shown = 0;
  const reader = r.body!.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    got += value.length;
    if (got - shown > 2e6) { shown = got; noteModel(`downloading the model: ${(got / 1e6).toFixed(0)}${total ? ` of ${(total / 1e6).toFixed(0)}` : ""} MB`); }
  }
  const bytes = new Uint8Array(got);
  let o = 0;
  for (const p of parts) { bytes.set(p, o); o += p.length; }
  noteModel("starting the model");
  try { await cache?.put(url, new Response(bytes, { headers: { "content-type": "application/octet-stream" } })); } catch { /* not cached: downloaded again next time */ }
  return bytes.buffer;
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
    noteModel(`loading the model (${emb.input} px)`);
    const t = performance.now();
    await emb.embed([{ data: new Float32Array(64 * 64), size: 64, channels: 1 }]);
    const n = 32, d = new Float32Array(64 * 64);
    const t2 = performance.now();
    await emb.embed(Array.from({ length: n }, () => ({ data: d, size: 64, channels: 1 })));
    const rate = n / ((performance.now() - t2) / 1000);
    modelNote = null;
    loadNote = `${((t2 - t) / 1000).toFixed(0)} s to load`;
    showRate(rate, "measured on loading");
    rateAt = { n: emb.count, s: emb.seconds, t: performance.now() };
  } catch (e) {
    modelNote = null;
    setModel("err", "model failed", (e as Error).message);
  }
}
let loadNote = "";
/** The model chip: its speed (patches through the model per second of its own time). */
function showRate(rate: number, how: string) {
  const gpu = emb.provider === "webgpu";
  const detail = `DINOv2-small, ${emb.input} px input, ${rate.toFixed(0)} patches/s (${how}), ${loadNote}.`;
  setModel(gpu ? "gpu" : "cpu", `DINOv2-small \u00b7 ${gpu ? "GPU" : "CPU, slow"}: ${rate.toFixed(0)} patches/s`,
    gpu ? detail : detail + (window.isSecureContext
      ? " No WebGPU here: try Chrome on a machine with a GPU (on Linux, chrome://flags/#enable-unsafe-webgpu)."
      : " WebGPU needs https or localhost: open https://davidackerman.github.io/coresample/ instead."));
}
/** While searching, the speed over the last few seconds of model work. */
let rateAt = { n: 0, s: 0, t: 0 };
function liveRate() {
  if (modelNote || !emb.provider) return;
  const now = performance.now(), dn = emb.count - rateAt.n, ds = emb.seconds - rateAt.s;
  if (now - rateAt.t < 2000 || dn < 64 || ds <= 0) return;
  showRate(dn / ds, "while searching");
  rateAt = { n: emb.count, s: emb.seconds, t: now };
}
/** The model loads once the viewer has had the network to itself for a while (its first data
 * comes first), or at the first click, whichever is sooner. */
const warm = () => (warmed ??= warmUp());
setTimeout(warm, 8000);

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
  if (ngv || gv || olv) {
    // what the boxes show: the examples, then the 30 best hits (a viewer redraws only on change)
    const key = search ? `${search.state.examples}|${search.state.hits.slice(0, 30).map((h) => h.index).join(",")}` : "none";
    if (!boxesDue(key)) return;
  }
  if (ngv) {  // the boxes as a Neuroglancer annotation layer
    if (!search) { ngv.setBoxes([], []); return; }
    const st = search.state, f = 2 ** st.working, side = st.patch * f, d3 = src?.dims === 3;
    const box = (c: Vec3, lab?: string) => ({ fz: d3 ? c[0] * f : 0, fy: c[1] * f, fx: c[2] * f, side, label: lab });
    ngv.setBoxes(queryBoxes(box), st.hits.slice(0, 30).map((h, k) => box(h.center, `#${k + 1}`)));
    return;
  }
  if (gv) {  // the boxes as outlines on the globe
    if (!search) { gv.setBoxes([], []); return; }
    const st = search.state, f = 2 ** st.working, side = st.patch * f;
    const box = (c: Vec3, lab?: string) => ({ fx: c[2] * f, fy: c[1] * f, side, label: lab });
    gv.setBoxes(queryBoxes(box), st.hits.slice(0, 30).map((h, k) => box(h.center, String(k + 1))));
    return;
  }
  if (olv) {  // the boxes as OpenLayers features
    if (!search) { olv.setBoxes([], []); return; }
    const st = search.state, f = 2 ** st.working, side = st.patch * f;
    const box = (c: Vec3, lab?: string) => ({ fx: c[2] * f, fy: c[1] * f, side, label: lab });
    olv.setBoxes(queryBoxes(box), st.hits.slice(0, 30).map((h, k) => box(h.center, String(k + 1))));
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
  search.exampleCenters().forEach((c, k) => { if (c.every(Number.isFinite)) box(c, "#5ab0ff", k ? `example ${k + 1}` : "query", 3); });
}

/** The query's boxes, one per example ("query", then "example 2", ...). */
function queryBoxes<B>(box: (c: Vec3, label?: string) => B): B[] {
  return search ? search.exampleCenters().flatMap((c, k) => (c.every(Number.isFinite) ? [box(c, k ? `example ${k + 1}` : "query")] : [])) : [];
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
    return x >= 1e12 ? `${(x / 1e12).toFixed(x >= 1e14 ? 0 : 1)} km` : x >= 1e9 ? `${(x / 1e9).toFixed(x >= 1e10 ? 0 : 1)} m` : x >= 1e6 ? `${(x / 1e6).toFixed(1)} mm`
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
  return preset.geo === "sky" ? [360 - (fx / W) * 360, lat] : [(fx / W) * 360 - 180, lat];  // sky: right ascension, 360 at the left
}
function pixelOf(lon: number, lat: number): [number, number] | null {
  if (!src || !preset?.geo) return null;
  const W = src.levels[0].shape[2], H = src.levels[0].shape[1];
  if (preset.geo === "equirect") return [((lon + 180) / 360) * W, ((90 - lat) / 180) * H];
  const r = (lat * Math.PI) / 180;
  return [((preset.geo === "sky" ? 360 - lon : lon + 180) / 360) * W, ((1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2) * W];
}
/** "x 2,999,500  y 833,700  z 2,666" plus degrees on a body or the physical position. */
function coordText(fx: number, fy: number, fz?: number): string {
  if (!src) return "";
  const n = (v: number) => Math.round(v).toLocaleString("en-US");
  let t = `x ${n(fx)}  y ${n(fy)}` + (src.dims === 3 && fz !== undefined ? `  z ${n(fz)}` : "");
  const ll = lonLat(fx, fy);
  if (ll && preset?.geo === "sky") t += `  (RA ${ll[0].toFixed(4)}\u00b0, Dec ${ll[1] >= 0 ? "+" : "\u2212"}${Math.abs(ll[1]).toFixed(4)}\u00b0)`;
  else if (ll) t += `  (${Math.abs(ll[1]).toFixed(3)}\u00b0 ${ll[1] >= 0 ? "N" : "S"}, ${Math.abs(ll[0]).toFixed(3)}\u00b0 ${ll[0] >= 0 ? "E" : "W"})`;
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
  // on the sky, two numbers that look like degrees are right ascension, declination
  if (preset?.geo === "sky" && v.length === 2 && v[0] >= 0 && v[0] <= 360 && Math.abs(v[1]) <= 90) {
    const p = pixelOf(v[0], v[1]);
    if (p) [fx, fy] = p;
  }
  // on a map of a body, two numbers that look like degrees are latitude, longitude
  else if (preset?.geo && v.length === 2 && Math.abs(v[0]) <= 90 && Math.abs(v[1]) <= 360) {
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
    query((e.clientX - r.left) * dpr, (e.clientY - r.top) * dpr, e.shiftKey);
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
/** The menu as the dataset list (under the datasets button) or as the box for your own data
 * (under "+ Your data"); its button again closes it. */
function showMenu(mode: "list" | "own") {
  if (!dsMenu.hidden && dsMenu.dataset.mode === mode) { dsMenu.hidden = true; return; }
  dsMenu.dataset.mode = mode;
  dsMenu.hidden = false;
  const own = $("ownbtn"), room = window.innerWidth - 8 - dsMenu.offsetWidth - dsMenu.parentElement!.getBoundingClientRect().left;
  dsMenu.style.left = mode === "own" ? `${Math.max(0, Math.min(own.offsetLeft, room))}px` : "0";
  if (mode === "list") fillDatasets();
  else requestAnimationFrame(() => $("custom").focus());  // after the click that opened it settles
}
dsBtn.addEventListener("click", () => showMenu("list"));
document.addEventListener("pointerdown", (e) => { if (!dsMenu.hidden && !(e.target as HTMLElement).closest(".dpick")) dsMenu.hidden = true; });
window.addEventListener("keydown", (e) => { if (e.key === "Escape") dsMenu.hidden = true; });
$("open").addEventListener("click", () => { dsMenu.hidden = true; openCustom($<HTMLInputElement>("custom").value); });
/** The box for your own data, open. */
function openOwnBox() { dsMenu.hidden = true; showMenu("own"); }
$("ownbtn").addEventListener("click", () => showMenu("own"));
$("ownlink").addEventListener("click", openOwnBox);
$("pickimg").addEventListener("click", () => $<HTMLInputElement>("imgfile").click());
$("imgfile").addEventListener("change", () => {
  const f = $<HTMLInputElement>("imgfile").files?.[0];
  dsMenu.hidden = true;
  if (f) void openImage(f);
  $<HTMLInputElement>("imgfile").value = "";
});
$("pick").hidden = !canPick;
$("pickhelp").textContent = "or drop either anywhere on the page";
$("pick").addEventListener("click", async () => {
  dsMenu.hidden = true;
  try { const f = await pickFolder(); if (f) await openLocal(f); }
  catch (e) { $("phase").textContent = "Could not open the folder"; $("status").textContent = (e as Error).message; }
});
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
  boxesKey = "";
  const tiled = s instanceof TileSource, volume = (s instanceof OmeZarrSource || s instanceof PrecomputedSource) && (await hasNeuroglancer());
  olDiv.style.display = tiled ? "block" : "none";
  canvas.style.display = tiled || volume ? "none" : "block";
  hover.style.display = tiled ? "none" : "block";
  if (volume) {
    ngv = new NgViewer(canvas.parentElement!, s as OmeZarrSource | PrecomputedSource, { cx: v.cx, cy: v.cy, z: v.z ?? view.z, scale: v.scale }, hover, {
      click: (fz, fy, fx, add) => queryAt(fx, fy, fz, add),
      hoverBox: () => (src ? { side: Number($<HTMLSelectElement>("patch").value) * 2 ** searchLevel(), text: patchSize() } : null),
      moved: () => { drawHover(); saveView(); },
      pointer: (fx, fy, fz) => showCursor(fx, fy, fz),
    }, p?.layers ?? []);
    hover.parentElement!.appendChild(hover);  // the hover canvas above the viewer's frame
    const frame = ngv.iframe;
    frame.addEventListener("load", () => { if (frame.contentWindow) watchDrags(frame.contentWindow); }, { once: true });
    $("zbar").style.display = "none";  // Neuroglancer moves through z itself
  }
  if (tiled) {
    olv = new OlViewer(olDiv, s as TileSource, v, {
      click: (fx, fy, add) => queryAt(fx, fy, undefined, add),
      hover: () => (src ? { side: Number($<HTMLSelectElement>("patch").value) * 2 ** searchLevel(), text: patchSize() } : null),
      moved: () => { drawHover(); saveView(); },
      pointer: (fx, fy) => showCursor(fx, fy),
    });
  }
  $("attrib").textContent = `${s.name}: ${sizeText(s)}` + (s.attribution ? `. Data: ${s.attribution}` : "");
  $("hint").textContent = (p?.note ? p.note + " " : "") + "Drag to pan, scroll to zoom, click to search; shift-click adds another example.";
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
  if (tiled && p?.globe && wantGlobe()) void toggleGlobe();
}

/** A view from the address bar (?v=x,y,z,scale), once, for the first dataset opened. */
let startView: Preset["view"] | null = (() => {
  const v = (new URL(location.href).searchParams.get("v") ?? "").split(",").map(Number);
  return v.length === 4 && v.every(Number.isFinite) ? { cx: v[0], cy: v[1], z: v[2], scale: v[3] } : null;
})();
function takeStartView() { const v = startView; startView = null; return v; }

/** What the usage counts call the open dataset: a preset's id, or "custom/<format>". */
let countAs = "";
async function openPreset(p: Preset) {
  $("phase").textContent = `Opening ${p.label}`;
  try { await setSource(await p.open(), takeStartView() ?? p.view, p); idle(); countAs = p.id; countOpen(p.id, p.label); }
  catch (e) { $("phase").textContent = "Could not open the data"; $("status").textContent = (e as Error).message; }
}
function idle() {
  $("phase").textContent = "Click on the image to search";
  const own = document.createElement("button");
  own.className = "link"; own.textContent = "paste a link";
  own.addEventListener("click", openOwnBox);
  $("status").replaceChildren("It reads the most promising tiles first and keeps going until it has covered the region or you pause it.\n"
    + "Your own data: drop a folder on the page, or ", own, ".");
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
    const fmt = m ? "legacy" : s instanceof PrecomputedSource ? "precomputed" : s instanceof OmeZarrSource ? "ome-zarr" : /\{z\}/.test(text) ? "tiles" : "zoomify";
    await showOwn(s, text, fmt);
  } catch (e) {
    const msg = (e as Error).message || String(e);
    $("phase").textContent = "Could not open the data";
    $("status").textContent = (/fetch/i.test(msg) ? "The host did not answer, or does not allow reads from web pages (CORS). " : "") + msg
      + "\nReadable: OME-Zarr, Neuroglancer precomputed (precomputed://...), web-map tiles, Zoomify, legacy:RA,Dec.";
  } finally {
    openBtn.disabled = false; openBtn.textContent = "Open";
  }
}

/** Your own data, opened: shown at its middle, the address bar a link to it (`link`; none for a
 * folder on this computer), counted as its format. */
async function showOwn(s: Source, link: string | null, fmt: string) {
  const L0 = s.levels[0].shape;
  await setSource(s, takeStartView() ?? { cx: L0[2] / 2, cy: L0[1] / 2, z: Math.floor(L0[0] / 2), scale: Math.max(1, Math.max(L0[2], L0[1]) / 1600) }, null);
  const u = new URL(location.href);
  u.searchParams.delete("d");
  if (link) u.searchParams.set("url", link); else u.searchParams.delete("url");
  history.replaceState(null, "", u);
  idle();
  countAs = `custom/${fmt}`; countOpen(countAs, `your own data (${fmt})`);
}

/** An image file on this computer (PNG, JPEG, TIFF...), read whole and searched as an OME-Zarr. */
async function openImage(file: File) {
  $("phase").textContent = `Opening ${file.name}`;
  $("status").textContent = "Reading the image into memory";
  try { await openLocal(await imageFolder(file), "image"); }
  catch (e) { $("phase").textContent = "Could not open the image"; $("status").textContent = (e as Error).message || String(e); }
}

/** A folder on this computer: the image in it (the folder itself, or the first inside). */
async function openLocal(f: Folder, fmt?: string) {
  $("phase").textContent = `Opening ${f.name}`;
  $("status").textContent = "Looking for an image in the folder";
  try {
    const found = await findImages(f), ok = found.filter((x) => x.kind !== "n5");
    if (!ok.length) throw new Error(found.length ? `${f.name} holds N5, which is not read here: OME-Zarr or Neuroglancer precomputed.`
                                                 : `no OME-Zarr image or precomputed volume in ${f.name}.`);
    const { path, kind } = ok[0], base = await serveFolder(f), url = path ? `${base}/${path}` : base;
    const name = path ? `${f.name}/${path}` : f.name;
    const s = kind === "precomputed" ? await PrecomputedSource.open(url, name) : await OmeZarrSource.open(url, name, { mirrors: false });
    await showOwn(s, null, `local-${fmt ?? kind}`);
    if (ok.length > 1) $("status").textContent = `Opened ${name}. Also in the folder: ${ok.slice(1).map((x) => x.path).join(", ")} (drop one of those to open it).`;
  } catch (e) {
    $("phase").textContent = "Could not open the folder";
    $("status").textContent = (e as Error).message || String(e);
  }
}

// a folder dropped anywhere on the page (or on the viewer's frame, which has its own document)
const dropEl = $("drop");
const withFiles = (e: DragEvent) => [...(e.dataTransfer?.types ?? [])].includes("Files");
function watchDrags(w: Window) {
  w.addEventListener("dragenter", (e) => { if (withFiles(e)) { e.preventDefault(); dropEl.hidden = false; } }, true);
  w.addEventListener("dragover", (e) => { if (withFiles(e)) e.preventDefault(); }, true);
  w.addEventListener("drop", (e) => { if (withFiles(e)) { e.preventDefault(); e.stopPropagation(); dropped(e); } }, true);
}
watchDrags(window);
dropEl.addEventListener("dragover", (e) => { e.preventDefault(); e.dataTransfer!.dropEffect = "copy"; });
dropEl.addEventListener("dragleave", (e) => { if (e.target === dropEl) dropEl.hidden = true; });
function dropped(e: DragEvent) {
  dropEl.hidden = true;
  const items = [...(e.dataTransfer?.items ?? [])].filter((i) => i.kind === "file");
  const entry = items[0]?.webkitGetAsEntry?.(), file = items[0]?.getAsFile();  // only during the event
  if (entry?.isDirectory) { void openLocal(entryFolder(entry as FileSystemDirectoryEntry)); return; }
  if (file && IMAGE_FILE.test(file.name)) { void openImage(file); return; }
  $("phase").textContent = "Drop a folder or an image";
  $("status").textContent = `${file?.name ?? "That"} is not read here: drop an OME-Zarr or precomputed folder, or a PNG, JPEG, WebP or TIFF.`;
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
  $("marks").hidden = true;
  $("hits").innerHTML = '<div class="empty">Hits appear here as tiles are read, best first. Hover one: ✓ more like this, ✗ fewer like this.</div>';
  $("hitcount").textContent = "";
  const stopBtn = $<HTMLButtonElement>("stop");
  stopBtn.disabled = true; stopBtn.textContent = "Pause";
  $<HTMLButtonElement>("clear").disabled = true;
  $("stages").replaceChildren();
  $("qcanvas").classList.add("none");
  $<HTMLCanvasElement>("qcanvas").getContext("2d")?.clearRect(0, 0, 999, 999);
  $("link").style.display = "none";
  $("wheresec").hidden = true;
  olv?.setHeat(null);
  ngv?.setHeat(null);
  heatCache = null;
  qDrawn = "";
  maps.clear(); toks.clear(); mapFor = null; pca = null; qgrids = null; qmaps = [];
  $<HTMLCanvasElement>("qmap").width = 128;
  setAdding(false);
  $<HTMLButtonElement>("addex").disabled = true;
  $<HTMLButtonElement>("brushbtn").disabled = true;
  $("bpad").hidden = true;
  render();
}
$("clear").addEventListener("click", () => { brushW = null; clearSearch(); idle(); });
$("stop").addEventListener("click", () => {  // pause, or carry on from where it paused
  if (!search) return;
  if (search.state.phase === "read") search.stop();
  else if (search.state.phase === "paused") search.resume();
  else if (search.state.phase === "done" && search.state.ruledOut) search.readRest();
});

/** The first area to search, at level W: as Options > region says; auto takes the largest area
 * the quick pass can scan whole, smaller than the view if need be (a tilted globe shows a huge
 * area; the search goes on outward from it anyway), then around `at` (level W, if given). */
function regionFor(W: number, P: number, at?: Vec3): [Vec3, Vec3] {
  const s = src!, L = s.levels[W], f = 2 ** W, d3 = s.dims === 3;
  const whole: [Vec3, Vec3] = [[0, 0, 0], [...L.shape] as Vec3];
  const vis = gv ? gv.extentPx() : olv ? olv.extentPx() : ngv ? ngv.extentPx() : { x0: view.cx - (canvas.width * view.scale) / 2, x1: view.cx + (canvas.width * view.scale) / 2,
                                      y0: view.cy - (canvas.height * view.scale) / 2, y1: view.cy + (canvas.height * view.scale) / 2 };
  const around = (k: number): [Vec3, Vec3] => {
    const hw = ((vis.x1 - vis.x0) * k) / 2 / f, hh = ((vis.y1 - vis.y0) * k) / 2 / f;
    const mid = k < 1 && at?.every(Number.isFinite);
    const cx = mid ? at![2] : (vis.x0 + vis.x1) / 2 / f, cy = mid ? at![1] : (vis.y0 + vis.y1) / 2 / f;
    const lo: Vec3 = [0, Math.max(0, Math.floor(cy - hh)), Math.max(0, Math.floor(cx - hw))];
    const hi: Vec3 = [d3 ? L.shape[0] : 1, Math.min(L.shape[1], Math.ceil(cy + hh)), Math.min(L.shape[2], Math.ceil(cx + hw))];
    return [lo, hi];
  };
  const mode = $<HTMLSelectElement>("region").value;
  if (mode === "whole") return whole;
  if (mode !== "auto") return around(Number(mode));
  if (coarseLevelFor(s, W, P, whole, COARSE_BUDGET) >= 0) return whole;
  for (let k = 16; k >= 1 / 256; k /= 2) if (coarseLevelFor(s, W, P, around(k), COARSE_BUDGET) >= 0) return around(k);
  return around(1);
}

function query(px: number, py: number, add = false) {  // canvas viewer: device pixels to full resolution
  queryAt(view.cx + (px - canvas.width / 2) * view.scale, view.cy + (py - canvas.height / 2) * view.scale, undefined, add);
}

const combineText = () => ($<HTMLSelectElement>("combine").value === "any" ? "any one of them" : "their mean");
/** A short message over the image for a few seconds. */
let toastTimer = 0;
function toast(msg: string) {
  const h = $("hint");
  h.textContent = msg;
  h.classList.remove("gone");
  clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => h.classList.add("gone"), Math.max(3500, 45 * msg.length));  // long enough to read
}

// more examples of the same thing: shift-click, or "+ example" and then a click (one example)
let adding = false;
function setAdding(on: boolean) {
  adding = on;
  $("addex").classList.toggle("on", on);
  $("stage").classList.toggle("adding", on);
}
$("addex").addEventListener("click", () => setAdding(!adding));
window.addEventListener("keydown", (e) => { if (e.key === "Escape") setAdding(false); });

function addExampleAt(fx: number, fy: number, fz?: number) {
  const s = search!, W = s.state.working, P = s.state.patch, f = 2 ** W, L = src!.levels[W].shape;
  const c: Vec3 = [src!.dims === 3 ? Math.floor((fz ?? view.z) / f) : 0, Math.floor(fy / f), Math.floor(fx / f)];
  if (c[1] < P / 2 || c[2] < P / 2 || c[1] > L[1] - P / 2 || c[2] > L[2] - P / 2) { toast("too close to the edge"); return; }
  toast(`adding example ${s.state.examples + 1}\u2026`);
  s.addExample(c).then((err) => toast(err ?? `${s.state.examples} examples: hits now match ${combineText()}`));
}

function queryAt(fx: number, fy: number, fz?: number, add = false) {
  if (!src) return;
  if ((add || adding) && search && search.state.phase !== "error") { setAdding(false); addExampleAt(fx, fy, fz); return; }
  const s = src;
  const P = Number($<HTMLSelectElement>("patch").value);
  const W = searchLevel();   // search at the level being viewed (as the hover box shows)
  const f = 2 ** W;
  const c: Vec3 = [s.dims === 3 ? Math.floor((fz ?? view.z) / f) : 0, Math.floor(fy / f), Math.floor(fx / f)];
  const L = s.levels[W].shape;
  if (c[1] < P / 2 || c[2] < P / 2 || c[1] > L[1] - P / 2 || c[2] > L[2] - P / 2) { $("status").textContent = "too close to the edge"; return; }
  countEvent(`search/${countAs || "unknown"}`);
  brushW = null;  // marks belong to the box they were made on
  begin(c, W, P);
}

/** The search sees a multi-channel image's channels as the viewer shows them (which are on,
 * their colours and ranges), as they are when a search starts or the pad opens. */
function takeChannels() {
  if (src instanceof OmeZarrSource && src.chans.length) src.setMix(ngv?.channelMix() ?? null);
}
const mixKey = (m: { index: number; color: number[]; range: number[] }[]) => JSON.stringify(m.map((c) => [c.index, c.color, c.range]));
setInterval(() => {  // channels changed in the viewer while a search runs: say a new one is needed
  const m = search && src instanceof OmeZarrSource && src.chans.length ? ngv?.channelMix() : null;
  const on = m?.filter((c) => c.visible);
  $("chnote").hidden = !on || mixKey(on.length ? on : m!.slice(0, 1)) === mixKey((src as OmeZarrSource).mixed());
}, 1000);

/** Which of the query box's 8 x 8 parts matter (the brush), or null: every part alike. */
let brushW: Float32Array | null = null;
let started: { c: Vec3; W: number; P: number; sketch?: Patch } | null = null;  // the search's start, to start it again with a brush

/** A new search for the box at `c` (level W, side P), or for a drawing. */
function begin(c: Vec3, W: number, P: number, sketch?: Patch) {
  const s = src!;
  clearSearch();
  started = { c, W, P, sketch };
  if (!sketch) takeChannels();  // a drawing took them when the pad opened
  warm();
  $("hint").classList.add("gone");
  const region = regionFor(W, P, c);
  search = new Search(s, emb, c, { working: W, patch: P, region, maxCoarseVoxels: COARSE_BUDGET, scanFn: scanInWorkers, topN: 60,
                                   match: matchOptions(), method: $<HTMLSelectElement>("method").value as Method, spread: spread(), sketch,
                                   expand: $<HTMLSelectElement>("region").value === "auto", brush: brushW ?? undefined,
                                   roughPass: ({ "1": true, "0": false } as Record<string, boolean>)[params.get("rough") ?? ""] });
  $<HTMLButtonElement>("clear").disabled = false;
  $<HTMLButtonElement>("addex").disabled = false;
  const tok = $<HTMLSelectElement>("method").value === "tokens", bb = $<HTMLButtonElement>("brushbtn");
  bb.disabled = tok;
  bb.classList.toggle("on", !!brushW);
  bb.textContent = brushW ? `what matters: ${brushW.reduce((n, v) => n + (v > 0 ? 1 : 0), 0)} parts` : "what matters";
  if (tok) bb.title = "Not with Options > tiles + tokens";
  let queued = false;
  const me = search;
  search.on(() => {  // at most one panel update per frame, always with the latest state
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; if (search === me) { panel(me.state); render(); } });
  });
  search.start();
  showWhyHelp();
}

function patchImage(pix: Uint8Array, P: number, C: number): ImageData {
  const img = new ImageData(P, P);
  for (let i = 0; i < P * P; i++) {
    img.data[4 * i] = pix[i * C];
    img.data[4 * i + 1] = pix[i * C + (C === 3 ? 1 : 0)];
    img.data[4 * i + 2] = pix[i * C + (C === 3 ? 2 : 0)];
    img.data[4 * i + 3] = 255;
  }
  return img;
}
function drawPatch(cv: HTMLCanvasElement, pix: Uint8Array, P: number, C: number) {
  cv.width = P; cv.height = P;
  cv.getContext("2d")!.putImageData(patchImage(pix, P, C), 0, 0);
}
/** The query's examples side by side in a square (one: the patch itself). */
let qDrawn = "";
/** Cells that fill a square of side S with n pictures, row by row, the last row's cells
 * widened so no part of the square is left empty (2: side by side, 3: two above one...). */
function mosaic(n: number, S: number): { x: number; y: number; w: number; h: number }[] {
  const cols = Math.ceil(Math.sqrt(n)), rows = Math.ceil(n / cols), out: { x: number; y: number; w: number; h: number }[] = [];
  for (let k = 0; k < n; k++) {
    const r = Math.floor(k / cols), inRow = r < rows - 1 ? cols : n - (rows - 1) * cols, j = k - r * cols;
    const x = Math.round((j * S) / inRow), y = Math.round((r * S) / rows);
    out.push({ x, y, w: Math.round(((j + 1) * S) / inRow) - x, h: Math.round(((r + 1) * S) / rows) - y });
  }
  return out;
}
/** The middle of a square picture of side `size` with the cell's shape: [sx, sy, sw, sh]. */
function cropTo(size: number, w: number, h: number): [number, number, number, number] {
  const sw = w < h ? (size * w) / h : size, sh = h < w ? (size * h) / w : size;
  return [(size - sw) / 2, (size - sh) / 2, sw, sh];
}
function drawExamples(cv: HTMLCanvasElement, pixs: Uint8Array[], P: number, C: number) {
  const n = pixs.length, S = Math.ceil(Math.sqrt(n)) * P;
  cv.width = S; cv.height = S;
  const ctx2 = cv.getContext("2d")!, one = document.createElement("canvas");
  one.width = P; one.height = P;
  ctx2.fillStyle = "#0b0e12"; ctx2.fillRect(0, 0, S, S);
  const gap = n > 1 ? Math.max(1, Math.round(S / 48)) : 0;  // a thin line between examples
  mosaic(n, S).forEach((r, i) => {
    one.getContext("2d")!.putImageData(patchImage(pixs[i], P, C), 0, 0);
    ctx2.drawImage(one, ...cropTo(P, r.w, r.h), r.x, r.y, r.w - (r.x + r.w < S ? gap : 0), r.h - (r.y + r.h < S ? gap : 0));
  });
  cv.title = n > 1 ? `${n} examples: hits match ${combineText()}` : "the query (shift-click on the image to add another example)";
  if (brushW && n === 1) {  // the parts that do not matter, dimmed
    const c = S / BRUSH;
    ctx2.fillStyle = "#000000a0";
    brushW.forEach((v, i) => { if (!v) ctx2.fillRect((i % BRUSH) * c, Math.floor(i / BRUSH) * c, c, c); });
  }
}

/** A duration: seconds, minutes, hours, days or years. */
const dur = (t: number) => (t < 90 ? `${Math.round(t)} s` : t < 5400 ? `${Math.round(t / 60)} min` : t < 129600 ? `${Math.round(t / 3600)} hours`
                            : t < 6.3e7 ? `${Math.round(t / 86400)} days` : `${Math.round(t / 3.15e7)} years`);
/** "~N for all of it" at this pace while searching outward (and a hint to zoom out if that is long). */
function finishIn(st: State, total: number, done: number): string {
  if (st.phase !== "read" || st.seconds < 10 || !done || done >= total) return "";
  const t = ((total - done) * st.seconds) / done;
  return `~${dur(t)} for all of it` + (t > 3600 ? " (zoom out: faster)" : "");
}
/** Whether the first search area is already the whole level (nothing to search outward). */
function firstIsWhole(s: Search): boolean {
  const [lo, hi] = s.opt.region!, L = src!.levels[s.state.working].shape;
  return lo.every((v) => v === 0) && hi[1] >= L[1] && hi[2] >= L[2] && (src!.dims !== 3 || hi[0] >= L[0]);
}
/** The size of one search area (the first region), in physical units or pixels. */
function fmtArea(s: Search): string {
  const o = s.opt.region!, sc = src!.levels[s.state.working].scale, h = o[1][1] - o[0][1], w = o[1][2] - o[0][2];
  const a = fmtLen(w * sc[2]), b = fmtLen(h * sc[1]);
  return a && b ? `${a} x ${b}` : `${num(w)} x ${num(h)} px`;
}
const fmt = (b: number) => (b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${(b / 1e3).toFixed(0)} KB`);

const PHASE: Record<State["phase"], string> = { query: "Reading the query", scan: "Quick pass", read: "Searching",
                                                paused: "Paused", done: "Done", error: "Something went wrong" };
const num = (v: number) => v.toLocaleString("en-US");
/** A count in few characters: 4,800 as 4.8k, 216,161,406 as 216M. */
const abbr = (v: number) => (v < 1e4 ? num(v) : v < 1e6 ? `${+(v / 1e3).toPrecision(3)}k` : v < 1e9 ? `${+(v / 1e6).toPrecision(3)}M` : `${+(v / 1e9).toPrecision(3)}G`);
/** The stage rows under the query: a name, a bar of how much it has covered, the share, a tooltip. */
function stages(rows: [string, number, string][]) {
  const box = $("stages");
  while (box.children.length > rows.length) box.lastElementChild!.remove();
  rows.forEach(([name, f, tip], k) => {
    let el = box.children[k] as HTMLElement | undefined;
    if (!el) { el = document.createElement("div"); el.className = "stage"; el.innerHTML = "<span></span><i><b></b></i><span></span>"; box.appendChild(el); }
    el.title = tip;
    el.children[0].textContent = name;
    (el.children[1].firstElementChild as HTMLElement).style.width = f > 0 ? `max(2px, ${(100 * Math.min(1, f)).toFixed(2)}%)` : "0";
    el.children[2].textContent = `${pctOf(f)}%`;
  });
}
/** A share as a percentage, with as many decimals as a small one needs. */
const pctOf = (f: number) => (f >= 0.1 ? (100 * f).toFixed(0) : f >= 0.01 ? (100 * f).toFixed(1) : f >= 0.0001 || f === 0 ? (100 * f).toFixed(2) : "<0.01");

function panel(st: State) {
  if (!search || !src) return;
  const s = search, C = src.channels, P = st.patch;
  liveRate();
  if (st.phase !== "query" && qDrawn !== `${st.examples}`) {
    drawExamples($<HTMLCanvasElement>("qcanvas"), s.examplePixels(), P, C);
    $("qcanvas").classList.remove("none");
    qDrawn = `${st.examples}`;
  }
  // each stage's share of the data (searching outward: of the whole level, else of the area);
  // the tiles read are discounted by the patches still waiting to be compared
  const expand = !!s.opt.expand, total = expand ? st.levelTiles : st.tilesTotal, W = st.working;
  const [rlo, rhi] = s.regionBox, Ls = src.levels[W].shape;
  const whole = expand || (rlo.every((v) => v === 0) && rhi[1] >= Ls[1] && rhi[2] >= Ls[2] && (src.dims !== 3 || rhi[0] >= Ls[0]));
  const what = whole ? (src.dims === 3 ? "the volume" : "the image") : "the search area";
  const scanning = st.phase === "scan", unit = src instanceof TileSource ? "images" : "chunks";
  const tok = s.opt.method === "tokens", items = tok ? "windows" : "patches", outward = expand && !firstIsWhole(s);
  $("phase").textContent = st.phase === "done" ? (st.ruledOut ? "Done: the model's picks read" : "Done: every tile read") : PHASE[st.phase];
  const rows: [string, number, string][] = [];
  if (st.coarse >= 0) rows.push(["quick pass", scanning ? (st.scanChunks ? st.scanRead / st.scanChunks : 0) : expand ? st.scanned : 1,
    `Cheap statistics (brightness and texture) at level ${st.coarse}, ${2 ** (st.coarse - W)}x coarser, rank where to look first. `
    + (scanning ? (st.scanChunks ? `${num(st.scanRead)} of the ${num(st.scanChunks)} ${unit} it needs read.` : "")
       : `It took ${st.scanSeconds.toFixed(1)} s` + (outward ? ` and has covered ${pctOf(st.scanned)}% of ${what}: ${num(st.areas)} areas of ${fmtArea(s)} `
         + "so far, nearest first, more while the search goes on; the reads go to the best places of all of them." : "."))]);
  if (!scanning && st.tilesTotal) {
    if (st.rough) rows.push(["model", total ? st.looked / total : 0,
      `${modelName()} looks over the quick pass's best places at level ${st.rough.level}, ${2 ** (st.rough.level - W)}x coarser: `
      + `${num(st.rough.places)} places in ${num(st.looked)} of ${num(total)} tiles. The best ${100 * DETAIL}% of them are read in detail.`]);
    rows.push(["detail", total ? (st.tilesRead / total) * (st.patches / Math.max(1, st.patches + st.queued)) : 0,
      `Read at level ${W} (the level on screen) and compared by ${modelName()}${tok ? ", windows of its tokens" : ""}: `
      + `${num(st.patches)} ${items} in ${num(st.tilesRead)} of ${num(total)} tiles.`]);
  }
  stages(st.phase === "query" || st.phase === "error" ? [] : rows);
  const waiting = scanning && src instanceof TileSource ? src.waiting : 0;
  const lines = st.phase === "error" ? [st.message]
    : st.phase === "query" ? [modelNote ? `waiting for the model (${modelNote.replace(/ the model:?/, "")})` + (modelNote.startsWith("downloading") ? ": only the first time" : "") : st.message]
    : scanning ? [(st.scanChunks && st.scanRead >= st.scanChunks ? `comparing every place at level ${st.coarse} with the query`
                   : `reading level ${st.coarse}` + (st.scanChunks ? `: ${num(st.scanRead)} of ${num(st.scanChunks)} ${unit}` : ""))
                  + (waiting ? `, ${num(waiting)} waiting` : ""),
                  `${fmt(st.bytes)} in ${dur(st.seconds)}`]
    : [`${abbr(st.patches)} ${items} in ${abbr(st.tilesRead)} tiles` + (st.phase === "done" && st.ruledOut ? `, ${abbr(st.ruledOut)} ruled out` : ""),
       `${fmt(st.bytes)} in ${dur(st.seconds)}` + (st.firstHitSeconds ? `, first hits ${st.firstHitSeconds.toFixed(1)} s` : ""),
       outward ? finishIn(st, total, st.rough && st.looked ? st.looked : st.tilesRead) : ""];
  lines.push(st.examples > 1 ? `${st.examples} examples${s.opt.sketch ? " with the drawing" : ""}: hits match ${combineText()}`
             : s.opt.sketch ? "your drawing" : "");
  $("status").textContent = lines.filter(Boolean).join("\n");
  $("status").title = outward ? "Searching outward from the query at the level on screen. Zooming out before clicking searches a coarser level: faster, for bigger things." : "";
  if (st.examples === 1 && !s.opt.sketch)
    $("qcanvas").title = `the query, at ${coordText(s.query[2] * 2 ** W, s.query[1] * 2 ** W, s.query[0] * 2 ** W)} (shift-click on the image adds another example)`;
  const stopBtn = $<HTMLButtonElement>("stop");
  const rest = st.phase === "done" && st.ruledOut > 0;  // the tiles the model ruled out can still be read
  stopBtn.disabled = !(st.phase === "read" || st.phase === "paused" || rest);
  stopBtn.textContent = rest ? "Read the rest" : st.phase === "paused" ? "Resume" : "Pause";
  stopBtn.title = rest ? `Also read the ${num(st.ruledOut)} tiles the model ruled out, most promising first` : "";
  drawWhere(st);
  const box = $("hits");
  const sp = spread();
  $("hitcount").textContent = !st.hits.length ? "" : `${st.hits.length} best` + (sp.between > 1 ? `, ${sp.between} boxes apart` : ", no overlaps")
    + (sp.fromQuery > 0 ? `, ${sp.fromQuery}+ boxes from the ${st.examples > 1 ? "examples" : "query"}` : "");
  if (!st.hits.length) return;
  if (box.querySelector(".empty")) box.innerHTML = "";
  const shown = st.hits.map((h, rank) => {
    const near = s.nearestExample(h.center);
    return { h, rank, d: near ? distance(h.center, near, st.working) : { v: 0, text: "" } };
  });
  if (hitSort !== "best") shown.sort((a, b) => (hitSort === "near" ? a.d.v - b.d.v : b.d.v - a.d.v));
  shown.forEach(({ h, rank, d }, k) => {
    let el = hitEls[k];
    if (!el) {
      el = document.createElement("div");
      el.className = "hit";
      el.innerHTML = `<canvas title="go there"></canvas><canvas class="map"></canvas><div class="fb"><button data-g="1" title="More like this: the query moves toward this hit, every hit is scored again, and the tiles still to read favour places like it (click again to undo)">&#10003;</button><button data-g="0" title="Fewer like this: hidden, and the query moves away from it (undo under Hits)">&#10005;</button></div><div class="cap"><b></b><span></span></div><div class="z"></div><div class="why"></div>`;
      box.appendChild(el);
      hitEls[k] = el;
      el.querySelector("canvas")!.addEventListener("click", () => goTo(Number(el!.dataset.index)));
      el.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
        const i = Number(el!.dataset.index), good = b.dataset.g === "1";
        if (!search) return;
        boxesNow = true;  // the boxes follow the re-ranked list at once
        if (good && search.markOf(i) === "good") { search.unmark(i); toast("Mark undone."); return; }
        search.mark(i, good);
        toast(good ? "✓ More like this: hits re-ranked and reads steered toward it. Click ✓ again to undo."
                   : "✗ Fewer like this: hidden, hits re-ranked away from it. Undo under Hits.");
      }));
    }
    if (el.dataset.index !== String(h.index)) {
      el.dataset.index = String(h.index);
      drawPatch(el.querySelector("canvas")!, s.patchPixels(h.index), P, C);
      drawMap(el, h.index);
    }
    el.classList.toggle("good", s.markOf(h.index) === "good");
    el.querySelector(".cap b")!.textContent = `#${rank + 1}`;
    el.querySelector(".cap span")!.textContent = h.score.toFixed(3);
    el.querySelector(".z")!.textContent = [d.text, src?.dims === 3 ? `z ${num(h.center[0] * 2 ** st.working)}` : ""].filter(Boolean).join(", ");
    if (whyOn) el.querySelector(".why")!.textContent = whyText(s, h.index);
  });
  for (let k = st.hits.length; k < hitEls.length; k++) hitEls[k].remove();
  const mk = s.marks();
  $("marks").hidden = !(mk.good + mk.bad);
  $("markstext").textContent = [mk.good ? `${mk.good} ✓ more like this` : "", mk.bad ? `${mk.bad} ✗ fewer like this (hidden)` : ""].filter(Boolean).join(", ");
  hitEls.length = st.hits.length;
  if (whyOn || matchOn) fillMaps();
}

// ---------------------------------------------------------------------------------------
// why: how each hit was matched. Under each hit, the parts of its score; over it, a map of
// which parts of it look like the query (each of its 16 x 16 tokens from the 224 px model,
// by cosine with the query's average token: the "parts" term of the score, in finer detail)
// ---------------------------------------------------------------------------------------
const modelName = () => `DINOv2-small (${emb.input === 224 ? "detailed, 224" : "fast, 112"} px)`;
let whyOn = false;
/** match: a spotlight on each hit where it matches what was clicked (the rest dimmed), and on
 * the query, the part being matched. Each piece (16 x 16 per box, 224 px model) is scored by
 * its cosine with the query's centre-weighted piece after both lose the typical piece of the
 * query and the hits (their mean): what they share everywhere, such as background, drops out
 * and what is distinctive about the clicked thing stays. Each thumbnail on its own scale: its
 * best-matching parts lit (how good the hit is overall is the score under it). */
let matchOn = false;
const maps = new Map<number, Float32Array>();  // hit index -> how each of its pieces matches the clicked thing (g x g, softened)
const toks = new Map<number, Float32Array>();  // hit index -> its unit tokens (g x g x D)
let mapFor: Search | null = null, mapKey = "", qgrids: TokenGrid[] | null = null, qtok: Float32Array | null = null, qcen: Float32Array | null = null;
let filling = false;
let qmaps: Float32Array[] = [];  // the same for each example's own pieces
let pca: { mean: Float32Array; comps: Float32Array[]; lo: number[]; hi: number[] } | null = null;
const VARIANT: Record<string, string> = { "s-1": "at half size", "s+1": "at double size" };
function whyText(s: Search, i: number): string {
  const e = s.explain(i);
  if (!e) return "";
  const name = { look: "look", parts: "parts", centre: "centre", brushed: "marked", tokens: "tokens" };
  const bits = [e.parts.map((p) => `${name[p.name]} ${p.value.toFixed(2)}`).join(" · ")];
  if (e.variant !== "id") bits.push(VARIANT[e.variant] ?? `turned (${e.variant.slice(1)})`);
  if (e.example >= 0) bits.push(`closest: example ${e.example + 1}`);
  if (Math.abs(e.feedback) >= 0.005) bits.push(`marks ${e.feedback > 0 ? "+" : ""}${e.feedback.toFixed(2)}`);
  return bits.join(", ");
}
function showWhyHelp() {
  $("whyhelp").hidden = !whyOn;
  document.body.classList.toggle("parts", whyOn && !matchOn);
  $("matchhelp").hidden = !matchOn;
  const marked = !!search?.opt.brush;
  $("matchhelp").textContent = `Bright: the parts of each hit that look most like ${marked ? "the parts you marked" : "the middle of your box"}; the dimmer, `
    + `the less alike. What every box has (plain background, the usual texture) does not count. On your box: ${marked ? "the parts you marked" : "the middle part it looks for"}.`;
  if (!whyOn || !search) return;
  const m = search.opt.match, tok = search.opt.method === "tokens";
  $("whytext").innerHTML = `<b>${modelName()}</b>: cosine similarity to the query (1 = the same), `
    + (tok ? "of the window's average token. "
       : m?.focus === "brush" ? "of <b>marked</b>: the mean of the pieces you marked as mattering, at the same places in the box. "
       : `the mean of <b>look</b> (its summary of the box) and <b>${m?.focus === "center" ? "centre" : "parts"}</b> `
         + `(${m?.focus === "center" ? "its pieces, weighted to the centre" : "its 8 x 8 pieces"}). `)
    + (matchOn ? "" : "Colours: pieces the model sees alike share a colour, in the query and every hit. ")
    + (emb.input === 224 ? "" : "(224 px model: 88 MB the first time.)");
}
$("why").addEventListener("click", () => {
  whyOn = !whyOn;
  $("why").classList.toggle("on", whyOn);
  document.body.classList.toggle("why", whyOn);
  showWhyHelp();
  if (search) panel(search.state);
  drawQueryMap();
});
$("match").addEventListener("click", () => {
  matchOn = !matchOn;
  $("match").classList.toggle("on", matchOn);
  document.body.classList.toggle("match", matchOn);
  showWhyHelp();
  if (search) panel(search.state);
  for (const el of hitEls) drawMap(el, Number(el.dataset.index));
  drawQueryMap();
});

/** A g x g map (of `ch` channels) softened 3 x 3 (weights 1-2-1): single tokens are noisy. */
function soften(v: Float32Array, n: number, ch = 1): Float32Array {
  const b = new Float32Array(v.length);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) for (let c = 0; c < ch; c++) {
    let sum = 0, w = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const yy = y + dy, xx = x + dx;
      if (yy < 0 || xx < 0 || yy >= n || xx >= n) continue;
      const ww = (2 - Math.abs(dy)) * (2 - Math.abs(dx));
      sum += ww * v[(yy * n + xx) * ch + c]; w += ww;
    }
    b[(y * n + x) * ch + c] = sum / w;
  }
  return b;
}
/** The three main directions of a set of tokens (D each): a sample of at most 3,000, centred,
 * block power iteration; and each direction's 2nd-98th percentile, for the colour scale. */
function fitPca(sets: Float32Array[], D: number) {
  const all = sets.reduce((n, a) => n + a.length / D, 0), step = Math.max(1, Math.floor(all / 3000)), rows: Float32Array[] = [];
  let k = 0;
  for (const a of sets) for (let t = 0; t < a.length / D; t++, k++) if (k % step === 0) rows.push(a.subarray(t * D, (t + 1) * D));
  const mean = new Float32Array(D);
  for (const r of rows) for (let d = 0; d < D; d++) mean[d] += r[d] / rows.length;
  const X = rows.map((r) => r.map((v, d) => v - mean[d]));
  let rnd = 1;
  const comps = [0, 1, 2].map(() => Float32Array.from({ length: D }, () => ((rnd = (rnd * 16807) % 2147483647) / 2147483647) - 0.5));
  for (let it = 0; it < 30; it++) {
    const next = comps.map(() => new Float32Array(D));
    for (const x of X) for (let c = 0; c < 3; c++) {
      let p = 0;
      for (let d = 0; d < D; d++) p += x[d] * comps[c][d];
      for (let d = 0; d < D; d++) next[c][d] += p * x[d];
    }
    for (let c = 0; c < 3; c++) {  // orthonormal again (Gram-Schmidt)
      for (let j = 0; j < c; j++) { let p = 0; for (let d = 0; d < D; d++) p += next[c][d] * next[j][d]; for (let d = 0; d < D; d++) next[c][d] -= p * next[j][d]; }
      const n = Math.hypot(...next[c]) || 1;
      for (let d = 0; d < D; d++) next[c][d] /= n;
      comps[c] = next[c];
    }
  }
  const lo: number[] = [], hi: number[] = [];
  for (let c = 0; c < 3; c++) {
    const pr = X.map((x) => { let p = 0; for (let d = 0; d < D; d++) p += x[d] * comps[c][d]; return p; }).sort((a, b) => a - b);
    lo.push(pr[Math.floor(pr.length * 0.02)]); hi.push(pr[Math.floor(pr.length * 0.98)]);
  }
  return { mean, comps, lo, hi };
}
/** Tokens (g x g x D) as colours: each projected on the three directions, scaled to 0-1. */
function pcaColours(tok: Float32Array, g: number, D: number): Float32Array {
  const p = pca!, out = new Float32Array(g * g * 3);
  for (let t = 0; t < g * g; t++) for (let c = 0; c < 3; c++) {
    let v = 0;
    for (let d = 0; d < D; d++) v += (tok[t * D + d] - p.mean[d]) * p.comps[c][d];
    out[t * 3 + c] = Math.max(0, Math.min(1, (v - p.lo[c]) / Math.max(1e-6, p.hi[c] - p.lo[c])));
  }
  return soften(out, g, 3);
}
/** Draw a g x g colour map (RGBA from `px`) into `cv`, smoothly enlarged. */
function paintMap(cv: HTMLCanvasElement, g: number, px: (t: number) => [number, number, number, number], cell?: { x: number; y: number; w: number; h: number }) {
  const img = new ImageData(g, g);
  for (let t = 0; t < g * g; t++) img.data.set(px(t), 4 * t);
  const tmp = document.createElement("canvas");
  tmp.width = g; tmp.height = g;
  tmp.getContext("2d")!.putImageData(img, 0, 0);
  const S = 128, c2 = cv.getContext("2d")!, r = cell ?? { x: 0, y: 0, w: S, h: S };
  c2.imageSmoothingEnabled = true; c2.imageSmoothingQuality = "high";
  c2.drawImage(tmp, ...cropTo(g, r.w, r.h), r.x, r.y, r.w, r.h);
}
/** The query's tokens (each example's) from the 224 px model, and their average token. */
async function queryTokens(s: Search) {
  const P = s.state.patch, C = src!.channels;
  qgrids = await emb.tokens(s.examplePixels().map((pix) => ({ data: Float32Array.from(pix), size: P, channels: C })));
  const D = qgrids[0].D, q = new Float32Array(D);
  for (const g of qgrids) {
    const m = new Float32Array(D);
    for (let t = 0; t < g.g * g.g; t++) for (let d = 0; d < D; d++) m[d] += g.tokens[t * D + d];
    const n = Math.hypot(...m) || 1;
    for (let d = 0; d < D; d++) q[d] += m[d] / n;
  }
  const n = Math.hypot(...q) || 1;
  qtok = q.map((v) => v / n);
  // the centre-weighted average piece (Gaussian, sigma a quarter of the grid, as the "centre"
  // embedding), or the marked pieces' when the search has a brush
  const c = new Float32Array(D), bw = s.opt.brush;
  for (const g of qgrids) {
    const m = new Float32Array(D), sg = g.g / 4, mid = (g.g - 1) / 2;
    for (let y = 0; y < g.g; y++) for (let x = 0; x < g.g; x++) {
      const w = bw ? bw[Math.floor((y * BRUSH) / g.g) * BRUSH + Math.floor((x * BRUSH) / g.g)]
                   : Math.exp(-((y - mid) ** 2 + (x - mid) ** 2) / (2 * sg * sg)), t = y * g.g + x;
      for (let d = 0; d < D; d++) m[d] += w * g.tokens[t * D + d];
    }
    const k = Math.hypot(...m) || 1;
    for (let d = 0; d < D; d++) c[d] += m[d] / k;
  }
  const kc = Math.hypot(...c) || 1;
  qcen = c.map((v) => v / kc);
}
/** Maps for the hits shown that have none yet, a few at a time; the colours' directions are
 * fitted once, on the query and the first hits, so colours stay put as more arrive. */
async function fillMaps() {
  const s = search;
  if (!s || filling) return;
  const key = `${s.state.examples}|${s.state.patch}|${emb.input}`;
  if (mapFor !== s || mapKey !== key) { maps.clear(); toks.clear(); mapFor = s; mapKey = key; qgrids = null; qtok = null; qcen = null; qmaps = []; pca = null; }
  filling = true;
  try {
    if (!qgrids) { await queryTokens(s); drawQueryMap(); }
    for (;;) {
      if (search !== s || !(whyOn || matchOn)) return;
      const todo = hitEls.map((el) => Number(el.dataset.index)).filter((i) => !toks.has(i)).slice(0, 8);
      if (!todo.length) return;
      const P = s.state.patch, C = src!.channels;
      const grids = await emb.tokens(todo.map((i) => ({ data: Float32Array.from(s.patchPixels(i)), size: P, channels: C })));
      grids.forEach((g, k) => toks.set(todo[k], g.tokens));
      if (!pca) { pca = fitPca([...qgrids!.map((g) => g.tokens), ...toks.values()], grids[0].D); drawQueryMap(); }
      matchMaps();
      for (const el of hitEls) drawMap(el, Number(el.dataset.index));
      drawQueryMap();
    }
  } catch (e) { console.warn("why maps:", e); } finally { filling = false; }
}
/** Every map again (the typical piece changes as hits arrive; cheap: one cosine per piece). */
function matchMaps() {
  if (!qgrids || !qcen) return;
  const D = qgrids[0].D, mu = new Float32Array(D);
  let n = 0;
  for (const tk of [...qgrids.map((g) => g.tokens), ...toks.values()]) {
    for (let i = 0; i < tk.length; i++) mu[i % D] += tk[i];
    n += tk.length / D;
  }
  for (let d = 0; d < D; d++) mu[d] /= Math.max(1, n);
  const c = qcen.map((v, d) => v - mu[d]), cn = Math.hypot(...c) || 1;
  const score = (tk: Float32Array, g: number) => {
    const v = new Float32Array(g * g);
    for (let t = 0; t < v.length; t++) {
      let dot = 0, nn = 0;
      for (let d = 0; d < D; d++) { const x = tk[t * D + d] - mu[d]; dot += x * c[d]; nn += x * x; }
      v[t] = dot / (Math.sqrt(nn) * cn || 1);
    }
    return soften(v, g);
  };
  for (const [i, tk] of toks) maps.set(i, score(tk, Math.round(Math.sqrt(tk.length / D))));
  qmaps = qgrids.map((g) => score(g.tokens, g.g));
}
/** The spotlight over one map, on its own scale: clear on its best-matching parts (from its
 * 95th percentile up), darker the less a part matches, darkest below its median. */
function spotlight(m: Float32Array): (t: number) => [number, number, number, number] {
  const s = Array.from(m).sort((a, b) => a - b), lo = s[Math.floor(s.length * 0.5)], hi = s[Math.floor(s.length * 0.95)];
  return (t) => [0, 0, 0, Math.round(210 * (1 - Math.max(0, Math.min(1, (m[t] - lo) / Math.max(1e-6, hi - lo)))))];
}
function drawMap(el: HTMLElement, i: number) {
  const cv = el.querySelector<HTMLCanvasElement>(".map"), m = maps.get(i), tk = toks.get(i);
  if (!cv) return;
  cv.width = 128; cv.height = 128;  // (cleared)
  if (!m || mapFor !== search) return;
  const g = Math.round(Math.sqrt(m.length));
  if (!matchOn) {
    if (!pca || !tk) return;
    const col = pcaColours(tk, g, tk.length / (g * g));
    paintMap(cv, g, (t) => [255 * col[3 * t], 255 * col[3 * t + 1], 255 * col[3 * t + 2], 255]);
    return;
  }
  paintMap(cv, g, spotlight(m));
}
/** Over the query's thumbnail (each example in its cell): with match on, the spotlight on the part
 * being matched; with why, its pieces in the hits' colours. */
function drawQueryMap() {
  const cv = $<HTMLCanvasElement>("qmap");
  cv.width = 128; cv.height = 128;
  if (!qgrids || mapFor !== search) return;
  const cells = mosaic(qgrids.length, 128);  // laid out as the examples are (drawExamples)
  if (matchOn) {
    qmaps.forEach((v, k) => paintMap(cv, qgrids![k].g, spotlight(v), cells[k]));
    return;
  }
  if (!whyOn || !pca) return;
  qgrids.forEach((g, k) => {
    const col = pcaColours(g.tokens, g.g, g.D);
    paintMap(cv, g.g, (t) => [255 * col[3 * t], 255 * col[3 * t + 1], 255 * col[3 * t + 2], 255], cells[k]);
  });
}

// ---------------------------------------------------------------------------------------
// where it has looked: the search region from above (x-y) and, for a volume, from the side
// (x-z), with the tiles read, the hits, the query and the view
// ---------------------------------------------------------------------------------------
const whereCv = $<HTMLCanvasElement>("where");
/** What the map spans: searching outward, the whole dataset (the area searched so far outlined);
 * else the search region. */
function whereFrame(s: Search): [Vec3, Vec3] {
  if (!s.opt.expand) return s.regionBox;
  const L = src!.levels[s.state.working].shape;
  return [[0, 0, 0], [L[0], L[1], L[2]]];
}
/** A small picture of a 2-D dataset (its finest level at most 512 px across, if one is at most
 * 1024) to lay under the map; null until it has loaded. */
let backdrop: { src: Source; level: number; cv: HTMLCanvasElement | null } | null = null;
function whereBackdrop(): { level: number; cv: HTMLCanvasElement } | null {
  if (!src || src.dims === 3) return null;
  if (backdrop?.src === src) return backdrop.cv ? { level: backdrop.level, cv: backdrop.cv } : null;
  const s0 = src, n = s0.levels.length;
  let L = n - 1;
  while (L > 0 && Math.max(s0.levels[L - 1].shape[1], s0.levels[L - 1].shape[2]) <= 512) L--;
  const sh = s0.levels[L].shape;
  backdrop = { src: s0, level: L, cv: null };
  if (Math.max(sh[1], sh[2]) > 1024) return null;  // e.g. the sky, whose coarsest level is 8192 px
  s0.read(L, [0, 0, 0], [1, sh[1], sh[2]]).then((r) => {
    if (backdrop?.src !== s0) return;
    const cv = document.createElement("canvas"), img = new ImageData(sh[2], sh[1]), C = r.channels;
    cv.width = sh[2]; cv.height = sh[1];
    for (let i = 0; i < sh[1] * sh[2]; i++) {
      for (let c = 0; c < 3; c++) { const v = r.data[i * C + (C === 3 ? c : 0)]; img.data[4 * i + c] = v === v ? v : 0; }
      img.data[4 * i + 3] = 255;
    }
    cv.getContext("2d")!.putImageData(img, 0, 0);
    backdrop.cv = cv;
    scheduleWhere();
  }).catch(() => {});
  return null;
}
let wherePanels: { a: number; b: number; x0: number; y0: number; sa: number; sb: number; w: number; h: number }[] = [];
function drawWhere(st: State) {
  if (!search || !src || !st.tilesTotal) { $("wheresec").hidden = true; return; }
  $("wheresec").hidden = false;
  $("heatimg").hidden = !olv && !ngv;  // the heat over the image: 2-D maps, and Neuroglancer's slice
  const dpr = window.devicePixelRatio || 1, W = st.working, f = 2 ** W, ch = src.levels[W].chunk, d3 = src.dims === 3;
  const [rlo, rhi] = whereFrame(search), ext = [rhi[0] - rlo[0], rhi[1] - rlo[1], rhi[2] - rlo[2]];
  const [alo, ahi] = search.regionBox, outlined = !!search.opt.expand && !firstIsWhole(search), bd = whereBackdrop();
  $("lsearched").hidden = !outlined;
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
  const tiles = search.readTiles(), view = currentView(), heat = heatImages(st);
  wherePanels.forEach((p, k) => {
    const X = (v: number) => p.x0 + (v - rlo[p.a]) * p.sa, Y = (v: number) => p.y0 + (v - rlo[p.b]) * p.sb;
    if (specs[k].name) {  // the long label if it fits over the panel, else the axes, else none
      g.fillStyle = "#5f6977"; g.font = `${10.5 * dpr}px system-ui`;
      const room = p.w + (k < wherePanels.length - 1 ? gap * 0.8 : cw - p.x0 - p.w), short = specs[k].name.replace(/^.*\((.*)\)$/, "$1");
      const text = g.measureText(specs[k].name).width <= room ? specs[k].name : g.measureText(short).width <= room ? short : "";
      if (text) g.fillText(text, p.x0, p.y0 - 4 * dpr);
    }
    g.fillStyle = "#0b0e12"; g.fillRect(p.x0, p.y0, p.w, p.h);
    g.strokeStyle = "#313946"; g.lineWidth = dpr; g.strokeRect(p.x0, p.y0, p.w, p.h);
    g.save(); g.beginPath(); g.rect(p.x0, p.y0, p.w, p.h); g.clip();  // everything inside the region's outline
    const Lw = src!.levels[W].shape;
    if (bd && p.a === 2 && ext[2] / 2 ** (bd.level - W) >= 24) {  // the dataset, dimmed, if not too coarse for the frame
      g.globalAlpha = 0.45; g.imageSmoothingEnabled = true;
      g.drawImage(bd.cv, X(0), Y(0), Lw[2] * p.sa, Lw[1] * p.sb);
      g.globalAlpha = 1;
    }
    g.fillStyle = "#5ab0ff"; g.globalAlpha = 0.16;  // read tiles: overlapping columns add up
    for (const t of tiles) g.fillRect(X(t[p.a] * ch[p.a]), Y(t[p.b] * ch[p.b]), Math.max(1, ch[p.a] * p.sa), Math.max(1, ch[p.b] * p.sb));
    g.globalAlpha = 1;
    if (heat?.[k]) { g.imageSmoothingEnabled = false; g.drawImage(heat[k], Math.round(p.x0), Math.round(p.y0)); }
    if (view) {  // what is on screen: a rectangle from above, the slice from the side
      g.strokeStyle = "#ffffffcc"; g.lineWidth = dpr;
      if (p.a === 0) { g.beginPath(); g.moveTo(X(view.z / f), Y(view.y0 / f)); g.lineTo(X(view.z / f), Y(view.y1 / f)); g.stroke(); }
      else g.strokeRect(X(view.x0 / f), Y(view.y0 / f), ((view.x1 - view.x0) / f) * p.sa, ((view.y1 - view.y0) / f) * p.sb);
    }
    st.hits.slice(0, 30).forEach((hh, i) => {
      g.fillStyle = "#ffd24a";
      g.beginPath(); g.arc(X(hh.center[p.a]), Y(hh.center[p.b]), (i < 3 ? 3.2 : 2.2) * dpr, 0, 7); g.fill();
    });
    g.strokeStyle = "#5ab0ff"; g.lineWidth = 2 * dpr;
    for (const q of search!.exampleCenters()) if (q.every(Number.isFinite)) { g.beginPath(); g.arc(X(q[p.a]), Y(q[p.b]), 4.5 * dpr, 0, 7); g.stroke(); }
    if (outlined) {  // the area searched so far, at least a few pixels so it shows on a big map
      const cx = (X(alo[p.a]) + X(ahi[p.a])) / 2, cy = (Y(alo[p.b]) + Y(ahi[p.b])) / 2;
      const w = Math.max(10 * dpr, (ahi[p.a] - alo[p.a]) * p.sa), hh = Math.max(10 * dpr, (ahi[p.b] - alo[p.b]) * p.sb);
      g.strokeStyle = "#7ee0a0"; g.lineWidth = 1.5 * dpr; g.setLineDash([3 * dpr, 2 * dpr]);
      g.strokeRect(cx - w / 2, cy - hh / 2, w, hh);
      g.setLineDash([]);
    }
    g.restore();
  });
}

// ---------------------------------------------------------------------------------------
// the similarity heat: every place compared, coloured by its score (the best `top` share of
// scores, faint to solid), in the map above and, on 2-D maps, over the image
// ---------------------------------------------------------------------------------------
const heatTop = $<HTMLInputElement>("heattop"), heatOn = $<HTMLInputElement>("heaton"), heatOp = $<HTMLInputElement>("heatop");
$("heatgrad").style.background = HEAT_GRADIENT;
let heatCache: { key: string; ver: number; n: number; at: number; imgs: HTMLCanvasElement[] } | null = null;
let heatTimer = 0;
/** The heat pictures for the map's panels, rebuilt at most about once a second while the
 * search runs (and at once when the settings or the map's size change). */
function heatImages(st: State): HTMLCanvasElement[] | null {
  const s = search!, h = s.heat();
  if (!h.n) return null;
  const top = Number(heatTop.value) / 100, onImg = (!!olv || !!ngv) && heatOn.checked;
  const key = `${top}|${onImg}|${heatOp.value}|${wherePanels.map((p) => `${Math.round(p.w)}x${Math.round(p.h)}`).join()}`;
  const now = performance.now(), c = heatCache;
  if (c && c.key === key && ((c.ver === h.version && c.n === h.n) || now - c.at < 1000)) {
    if (!(c.ver === h.version && c.n === h.n) && !heatTimer)  // changed meanwhile: once more later
      heatTimer = window.setTimeout(() => { heatTimer = 0; scheduleWhere(); }, 1000 - (now - c.at));
    return c.imgs;
  }
  const [lo, hi] = heatRange(h, top), [rlo, rhi] = s.regionBox, [flo] = whereFrame(s), P = st.patch, f = 2 ** st.working;
  const foot = src!.dims === 3 || s.opt.method === "tokens" ? P / 2 : P;  // the spacing of the places compared
  const imgs = wherePanels.map((p, k) => {
    const w = Math.max(1, Math.round(p.w)), ht = Math.max(1, Math.round(p.h));
    return heatCanvas(heatGrid(h, w, ht, p.a, p.b, flo, p.sa, p.sb, foot), w, ht, lo, hi, c?.imgs[k]);
  });
  if (olv) {  // over the image: one cell per place, coarser if the region is huge
    if (!onImg) olv.setHeat(null);
    else {
      const ext = [rhi[1] - rlo[1], rhi[2] - rlo[2]], cell = foot * Math.max(1, Math.ceil(Math.sqrt((ext[0] / foot) * (ext[1] / foot) / 4e6)));
      const gw = Math.ceil(ext[1] / cell), gh = Math.ceil(ext[0] / cell);
      const cv = heatCanvas(heatGrid(h, gw, gh, 2, 1, rlo, 1 / cell, 1 / cell, foot), gw, gh, lo, hi);
      olv.setHeat({ canvas: cv, x0: rlo[2] * f, y0: rlo[1] * f, x1: (rlo[2] + gw * cell) * f, y1: (rlo[1] + gh * cell) * f }, Number(heatOp.value) / 100);
    }
  }
  // over Neuroglancer's slice: the places at the slice's depth (the viewer redraws it as the view moves)
  ngv?.setHeat(onImg ? { h, lo, hi, f, foot, opacity: Number(heatOp.value) / 100 } : null);
  $("heatlo").textContent = lo.toFixed(2);
  $("heathi").textContent = hi.toFixed(2);
  $("heatval").textContent = `${heatTop.value}%`;
  heatCache = { key, ver: h.version, n: h.n, at: now, imgs };
  return imgs;
}
for (const el of [heatTop, heatOn, heatOp]) el.addEventListener("input", () => { $("heatval").textContent = `${heatTop.value}%`; if (search) drawWhere(search.state); });
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
  const [rlo] = whereFrame(search), va = rlo[p.a] + (px - p.x0) / p.sa, vb = rlo[p.b] + (py - p.y0) / p.sb;
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
  return { focus: brushW ? "brush" : $<HTMLSelectElement>("focus").value as Focus, rotations: $<HTMLInputElement>("rot").checked,
           scales: $<HTMLInputElement>("scl").checked, adapt: $<HTMLInputElement>("adapt").checked,
           combine: $<HTMLSelectElement>("combine").value as "mean" | "any" };
}
for (const id of ["focus", "rot", "scl", "adapt", "combine"]) $(id).addEventListener("change", () => search?.setMatch(matchOptions()));
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
/** The same query again (same place or drawing, level and box), as a new search. */
function again() { if (search) begin(search.query, search.state.working, search.state.patch, search.opt.sketch); }
$("method").addEventListener("change", again);  // candidates are embedded differently: search again
$("detail").addEventListener("change", async () => {
  const det = $<HTMLSelectElement>("detail");
  det.dataset.touched = "1";
  emb.input = Number(det.value);
  await warmUp();
  again();  // embeddings differ: search again
});

// the marks' undo
$("undomark").addEventListener("click", () => { boxesNow = true; search?.unmark(); });
$("clearmarks").addEventListener("click", () => { boxesNow = true; search?.clearMarks(); });

// the map / globe toggle (planetary mosaics): they open on the globe unless the address says
// globe=0, which choosing the map writes there (so a shared link keeps the choice)
const globeBtn = $<HTMLButtonElement>("globe");
const wantGlobe = () => new URL(location.href).searchParams.get("globe") !== "0";
globeBtn.addEventListener("click", () => {
  const u = new URL(location.href);
  if (gv) u.searchParams.set("globe", "0"); else u.searchParams.delete("globe");
  history.replaceState(null, "", u);
  toggleGlobe();
});
async function toggleGlobe() {
  if (!src || !preset?.globe || !(src instanceof TileSource)) return;
  if (gv) {  // back to the map, at the globe's place
    const e = gv.extentPx(), s = gv.scale();
    gv.destroy(); gv = null;
    globeDiv.style.display = "none"; olDiv.style.display = "block";
    olv?.goTo((e.x0 + e.x1) / 2, (e.y0 + e.y1) / 2, s);
    globeBtn.textContent = "globe";
    boxesKey = ""; overlays();  // the map's boxes as they are now
  } else {
    const at = olv ? { ...olv.center(), scale: olv.scale() } : { cx: view.cx, cy: view.cy, scale: view.scale };
    globeBtn.textContent = "map";
    olDiv.style.display = "none"; globeDiv.style.display = "block";
    gv = new GlobeViewer(globeDiv, src, preset.globe, at, {
      click: (fx, fy, add) => queryAt(fx, fy, undefined, add),
      hover: () => (src ? { side: Number($<HTMLSelectElement>("patch").value) * 2 ** searchLevel(), text: patchSize() } : null),
      moved: () => { drawHover(); saveView(); },
      pointer: (fx, fy) => showCursor(fx, fy),
    });
    try { await gv.ready; boxesKey = ""; overlays(); } catch (err) { $("status").textContent = String((err as Error).message ?? err); }
  }
}

// ---------------------------------------------------------------------------------------
// drawing what to find: a box at the search level, drawn in the view's own greys
// ---------------------------------------------------------------------------------------
const pad = new SketchPad($<HTMLCanvasElement>("padcv"));
let padAt: { W: number; P: number } | null = null;
/** The view's centre at level W (z, y, x). */
function viewCentre(W: number): Vec3 {
  const v = currentView()!, f = 2 ** W;
  return [src!.dims === 3 ? Math.floor(v.z / f) : 0, Math.floor((v.y0 + v.y1) / 2 / f), Math.floor((v.x0 + v.x1) / 2 / f)];
}
/** The pixels (channels interleaved) of the box of side `side` at the view's centre, level W. */
async function viewPixels(W: number, side: number): Promise<{ data: Float32Array; channels: number }> {
  const c = viewCentre(W), h = side / 2;
  const r = await src!.read(W, [c[0], c[1] - h, c[2] - h], [c[0] + 1, c[1] + h, c[2] + h]);
  return { data: r.data, channels: r.channels };
}
/** The pen buttons in the pad's colours. */
function showPens() {
  for (const b of $("padtone").querySelectorAll<HTMLButtonElement>("button")) {
    const t = b.dataset.t as Tone;
    b.style.boxShadow = `inset 0 -3px 0 ${pad.colour(t)}`;
  }
}
async function openPad() {
  if (!src) return;
  const W = searchLevel(), P = Number(patchSel.value);
  padAt = { W, P };
  if (!search) takeChannels();
  $("pad").hidden = false;
  $("padhelp").textContent = `One box of ${patchSize()}. Any colour works: "look like the data" turns it into this data's own `
    + `${src.channels === 3 ? "colours" : "greys"} by brightness and adds its grain; untick it to search the drawing as drawn.`;
  pad.reset(P, { dark: [40, 40, 40], mid: [128, 128, 128], light: [215, 215, 215], noise: 0 });
  try {  // the tones (greys or colours) and grain of what is on screen
    const side = Math.min(4 * P, 512), v = await viewPixels(W, side);
    pad.reset(P, tonesOf(v.data, v.channels, side, side));
  } catch { /* the defaults */ }
  showPens();
}
$("drawbtn").addEventListener("click", () => ($("pad").hidden ? openPad() : ($("pad").hidden = true)));
$("padx").addEventListener("click", () => { $("pad").hidden = true; });
function choosePen(t: Tone) {
  pad.tone = t;
  for (const b of $("padtone").querySelectorAll<HTMLButtonElement>("button")) b.classList.toggle("on", b.dataset.t === t);
}
$("padtone").addEventListener("click", (e) => {
  const t = (e.target as HTMLElement).closest("button")?.dataset.t as Tone | undefined;
  if (t) choosePen(t);
});
const padCol = $<HTMLInputElement>("padcol");
function pickColour() {
  const h = padCol.value;
  pad.any = [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)) as [number, number, number];
  choosePen("any");
  showPens();
}
padCol.addEventListener("input", pickColour);
padCol.addEventListener("click", pickColour);
$("padfill").addEventListener("click", () => pad.fill());
$("padsize").addEventListener("click", (e) => {
  const r = (e.target as HTMLElement).closest("button")?.dataset.r;
  if (!r) return;
  pad.radius = Number(r);
  for (const b of $("padsize").querySelectorAll("button")) b.classList.toggle("on", b.dataset.r === r);
});
$("padblank").addEventListener("click", () => pad.reset(pad.P, pad.tones));
$("padview").addEventListener("click", async () => { if (padAt) { const v = await viewPixels(padAt.W, padAt.P); pad.load(v.data, v.channels); } });
$("padgo").addEventListener("click", () => {
  if (!src || !padAt) return;
  const sk = pad.patch(src.channels, $<HTMLInputElement>("padtex").checked);
  $("pad").hidden = true;
  countEvent(`draw/${countAs || "unknown"}`);
  brushW = null;
  begin(NOWHERE, padAt.W, padAt.P, sk);
});

// the brush: which parts of the query box matter
const bpad = new BrushPad($<HTMLCanvasElement>("bpadcv"), () => { $<HTMLButtonElement>("bpadgo").disabled = !bpad.marked; });
$("brushbtn").addEventListener("click", () => {
  if (!search || !src) return;
  const box = $("bpad");
  if (!box.hidden) { box.hidden = true; return; }
  bpad.load(search.examplePixels()[0], search.state.patch, src.channels, brushW);
  $<HTMLButtonElement>("bpadgo").disabled = !bpad.marked;
  $("pad").hidden = true;
  box.hidden = false;
});
$("bpadx").addEventListener("click", () => { $("bpad").hidden = true; });
$("bpadall").addEventListener("click", () => bpad.fill(1));
$("bpadnone").addEventListener("click", () => bpad.fill(0));
/** The search again from its start, with the brush as it is now. */
function againWithBrush(w: Float32Array | null) {
  brushW = w;
  $("bpad").hidden = true;
  if (started && src) begin(started.c, started.W, started.P, started.sketch);
}
$("bpadgo").addEventListener("click", () => againWithBrush(bpad.marked ? Float32Array.from(bpad.w) : null));
$("bpadoff").addEventListener("click", () => againWithBrush(null));

// for debugging from the console
(window as any).coresample = { get src() { return src; }, get ngv() { return ngv; }, get olv() { return olv; }, get gv() { return gv; }, get search() { return search; },
  local: { open: openLocal, handleFolder, entryFolder } };

// start
const params = new URL(location.href).searchParams, want = params.get("d"), own = params.get("url");
if (own) { $<HTMLInputElement>("custom").value = own; openCustom(own); }
else {
  const first = PRESETS.find((p) => p.id === want) ?? PRESETS[0];
  openPreset(first);
}
