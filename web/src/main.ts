// coresample's page: a pan/zoom viewer over any multiscale source, click to search, hits stream
// into the side panel. Raw data only; nothing is precomputed.
import { PRESETS, type Preset } from "./datasets.ts";
import { Embedder, type Focus } from "./embedder.ts";
import type { Dims, ScanInput } from "./scan.ts";
import { Search, coarseLevelFor, type MatchOptions, type State } from "./search.ts";
import { GlobeViewer } from "./globeviewer.ts";
import { NgViewer } from "./ngviewer.ts";
import { OlViewer } from "./olviewer.ts";
import { OmeZarrSource, TileSource, legacySurvey, openZoomify, type Source, type Vec3 } from "./sources.ts";

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
    const cache = await caches.open("coresample-models-v2");  // v2: the [CLS | mean | centre] export
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
const emb = new Embedder((size) => new URL(`models/dinov2-small-${size}.onnx`, document.baseURI).href,
                         (l) => console.log("[onnx]", l), cachedBytes);
const modelEl = $("model");

/** Load the model at the chosen input size and measure it. */
async function warmUp() {
  try {
    modelEl.textContent = `model: loading DINOv2-small at ${emb.input} px (88 MB the first time, then cached)`;
    const t = performance.now();
    await emb.embed([{ data: new Float32Array(64 * 64), size: 64, channels: 1 }]);
    const n = 32, d = new Float32Array(64 * 64);
    const t2 = performance.now();
    await emb.embed(Array.from({ length: n }, () => ({ data: d, size: 64, channels: 1 })));
    const rate = n / ((performance.now() - t2) / 1000);
    modelEl.textContent = `model: ${emb.provider === "webgpu" ? "GPU (WebGPU)" : "CPU (WebAssembly, slow)"}, ${emb.input} px input, ${rate.toFixed(0)} patches/s (${((t2 - t) / 1000).toFixed(0)} s to load)`;
    if (emb.provider !== "webgpu")
      modelEl.title = window.isSecureContext
        ? "No WebGPU here: try Chrome on a machine with a GPU (on Linux, chrome://flags/#enable-unsafe-webgpu)."
        : "WebGPU needs https or localhost: open https://davidackerman.github.io/coresample/ instead.";
  } catch (e) {
    modelEl.textContent = `model failed: ${(e as Error).message}`;
  }
}
setTimeout(warmUp, 2500);  // after the viewer and first data have had the network

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

/** A patch's physical size at the search level, in readable units. */
function patchSize(): string {
  if (!src) return "";
  const P = Number($<HTMLSelectElement>("patch").value), W = searchLevel();
  const v = P * src.levels[W].scale[2], u = (src.unit || "").toLowerCase();
  const nm = u.startsWith("nano") ? 1 : u.startsWith("micro") ? 1e3 : u === "m" || u.startsWith("meter") || u.startsWith("metre") ? 1e9 : NaN;
  let phys = "";
  if (!Number.isNaN(nm)) {
    const x = v * nm;  // nanometres
    phys = x >= 1e12 ? `${(x / 1e12).toFixed(1)} km` : x >= 1e9 ? `${(x / 1e9).toFixed(x >= 1e10 ? 0 : 1)} m` : x >= 1e6 ? `${(x / 1e6).toFixed(1)} mm`
         : x >= 1e3 ? `${(x / 1e3).toFixed(x >= 1e4 ? 0 : 1)} \u00b5m` : `${x.toFixed(0)} nm`;
  } else if (u.startsWith("arcsec")) phys = v >= 60 ? `${(v / 60).toFixed(1)}\u2032` : `${v.toFixed(0)}\u2033`;
  return `${P} px at level ${W}` + (phys ? ` = ${phys}` : "");
}

/** The dashed box under the cursor: exactly the region a click would search for. */
function drawHover() {
  $("psize").textContent = src ? `= ${patchSize().split("= ")[1] ?? patchSize()}` : "";
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

// pan / zoom / click
let drag: { x: number; y: number; cx: number; cy: number; moved: boolean } | null = null;
canvas.addEventListener("pointerdown", (e) => {
  canvas.setPointerCapture(e.pointerId);
  drag = { x: e.clientX, y: e.clientY, cx: view.cx, cy: view.cy, moved: false };
});
canvas.addEventListener("pointermove", (e) => {
  const rr = canvas.getBoundingClientRect(), dd = window.devicePixelRatio || 1;
  mouse = { x: (e.clientX - rr.left) * dd, y: (e.clientY - rr.top) * dd };
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
const sel = $<HTMLSelectElement>("dataset");
for (const p of PRESETS) sel.add(new Option(p.label, p.id));
sel.addEventListener("change", () => openPreset(PRESETS.find((p) => p.id === sel.value)!));
$("open").addEventListener("click", () => openCustom($<HTMLInputElement>("custom").value));

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
  const tiled = s instanceof TileSource, volume = s instanceof OmeZarrSource && (await hasNeuroglancer());
  olDiv.style.display = tiled ? "block" : "none";
  canvas.style.display = tiled || volume ? "none" : "block";
  hover.style.display = tiled ? "none" : "block";
  if (volume) {
    ngv = new NgViewer(canvas.parentElement!, s as OmeZarrSource, { cx: v.cx, cy: v.cy, z: v.z ?? view.z, scale: v.scale }, hover, {
      click: (fz, fy, fx) => queryAt(fx, fy, fz),
      hoverBox: () => (src ? { side: Number($<HTMLSelectElement>("patch").value) * 2 ** searchLevel(), text: patchSize() } : null),
      moved: () => drawHover(),
    });
    hover.parentElement!.appendChild(hover);  // the hover canvas above the viewer's frame
    $("zbar").style.display = "none";  // Neuroglancer moves through z itself
  }
  if (tiled) {
    olv = new OlViewer(olDiv, s as TileSource, v, {
      click: (fx, fy) => queryAt(fx, fy),
      hover: () => (src ? { side: Number($<HTMLSelectElement>("patch").value) * 2 ** searchLevel(), text: patchSize() } : null),
      moved: () => drawHover(),
    });
  }
  $("attrib").textContent = `${s.name}: ${sizeText(s)}` + (s.attribution ? `. Data: ${s.attribution}` : "");
  $("hint").textContent = (p?.note ? p.note + " " : "") + "Drag to pan, scroll to zoom, click to search.";
  const zb = $("zbar");
  zb.style.display = s.dims === 3 && !ngv ? "flex" : "none";
  if (s.dims === 3) { zIn.max = String(s.levels[0].shape[0] - 1); zIn.value = String(view.z); $("zval").textContent = zIn.value; }
  if (p?.patch) $<HTMLSelectElement>("patch").value = String(p.patch);
  fillLevels(s);
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
    else if (/\{z\}/.test(text) && /\{x\}/.test(text) && /\{y\}/.test(text)) {
      // a web-map tile template: square world, 256 px tiles; "#maxZoom=N" sets the finest zoom
      const [tpl, frag] = text.split("#"), mz = Number(new URLSearchParams(frag ?? "").get("maxZoom") ?? 18);
      s = new TileSource({ name: tpl.replace(/^https?:\/\//, "").split("/")[0], tileSize: 256, maxZoom: mz, channels: 3,
                           url: (z, x, y) => tpl.replace("{z}", String(z)).replace("{x}", String(x)).replace("{y}", String(y)) });
    } else if (/ImageProperties\.xml$/i.test(text) || /^zoomify:/i.test(text)) {
      const base = text.replace(/^zoomify:/i, "").replace(/\/?ImageProperties\.xml$/i, "");
      s = await openZoomify(base, base.split("/").filter(Boolean).pop() ?? "Zoomify image");
    } else s = await OmeZarrSource.open(text);
    const L0 = s.levels[0].shape;
    await setSource(s, { cx: L0[2] / 2, cy: L0[1] / 2, z: Math.floor(L0[0] / 2), scale: Math.max(1, Math.max(L0[2], L0[1]) / 1600) }, null);
    const u = new URL(location.href);  // a link that opens this data again
    u.searchParams.delete("d"); u.searchParams.set("url", text);
    history.replaceState(null, "", u);
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
  const region = regionFor(W, P);
  search = new Search(s, emb, c, { working: W, patch: P, region, budgetFrac: Number($<HTMLSelectElement>("budget").value),
                                   maxCoarseVoxels: COARSE_BUDGET, scanFn: scanInWorker, topN: 60, maxTiles: 48,
                                   match: matchOptions() });
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
      (st.queued ? `, ${st.queued} queued` : "") + (search.skipped ? `, ${search.skipped} blank/no-data skipped` : "") : "",
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
    el.querySelector(".cap span")!.textContent = `#${k + 1} ${h.score.toFixed(3)}` + (src?.dims === 3 ? ` z${h.center[0] * 2 ** st.working}` : "");
  });
  for (let k = st.hits.length; k < hitEls.length; k++) hitEls[k].remove();
  hitEls.length = st.hits.length;
}

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
      moved: () => drawHover(),
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
  sel.value = first.id;
  openPreset(first);
}
