// Multiscale sources for coresample's page. Every source presents the same thing: a pyramid
// of levels, finest first, each 2x coarser than the one before in y and x (and z for a
// volume), read as regions of 0-255 floats laid out z, y, x, channel. A 2-D image is a volume
// one voxel deep (z = 1), so the search code has one path for both.
//
// OME-Zarr (zarr v2 and v3, any host with CORS) is read with zarrita; tile pyramids (XYZ / TMS
// web maps, WMTS-style templates, Deep Zoom) by fetching and decoding the image tiles.
import * as zarr from "zarrita";

export type Vec3 = [number, number, number];
export interface Level { shape: Vec3; chunk: Vec3; scale: Vec3 }
export interface Region { data: Float32Array; shape: Vec3; channels: number }

export interface Source {
  name: string;
  dims: 2 | 3;
  channels: 1 | 3;
  levels: Level[];
  unit: string;
  attribution?: string;
  /** Bytes fetched from the network so far (compressed, as stored). */
  bytesRead: number;
  read(level: number, lo: Vec3, hi: Vec3): Promise<Region>;
  /** A link that opens the original data at a place, when there is a viewer for it. */
  link?(level: number, center: Vec3): string | null;
}

export function normalizeUrl(url: string): string {
  url = url.trim().replace(/^(zarr[23]?|n5):\/\//, "").replace(/\/+$/, "");
  if (url.startsWith("s3://")) {
    const [bucket, ...key] = url.slice(5).split("/");
    return `https://${bucket}.s3.amazonaws.com/${key.join("/")}`;
  }
  if (url.startsWith("gs://")) return "https://storage.googleapis.com/" + url.slice(5);
  return url;
}

// ---------------------------------------------------------------------------------------
// A small byte-bounded LRU, shared by the readers so display and search reuse fetches
// ---------------------------------------------------------------------------------------

class Lru<V> {
  private map = new Map<string, { v: V; size: number }>();
  private total = 0;
  private limit: number;
  constructor(limit: number) { this.limit = limit; }
  get(k: string): V | undefined {
    const e = this.map.get(k);
    if (!e) return undefined;
    this.map.delete(k);
    this.map.set(k, e);
    return e.v;
  }
  set(k: string, v: V, size: number) {
    const old = this.map.get(k);
    if (old) { this.total -= old.size; this.map.delete(k); }
    this.map.set(k, { v, size });
    this.total += size;
    for (const [key, e] of this.map) {
      if (this.total <= this.limit) break;
      this.map.delete(key);
      this.total -= e.size;
    }
  }
}

// ---------------------------------------------------------------------------------------
// OME-Zarr
// ---------------------------------------------------------------------------------------

/** A fetch store that counts bytes and keeps recent chunks (display and search share them). */
class CountingStore {
  bytes = 0;
  private inner: zarr.FetchStore;
  private cache = new Lru<Uint8Array | null>(768 << 20);
  private pending = new Map<string, Promise<Uint8Array | undefined>>();
  constructor(url: string) { this.inner = new zarr.FetchStore(url); }
  async get(key: zarr.AbsolutePath, opts?: RequestInit): Promise<Uint8Array | undefined> {
    const hit = this.cache.get(key);
    if (hit !== undefined) return hit ?? undefined;
    let p = this.pending.get(key);
    if (!p) {
      p = (async () => {
        let err: unknown;
        for (let attempt = 0; attempt < 4; attempt++) {  // transient network failures: retry
          try {
            const v = await this.inner.get(key, opts);
            if (v) this.bytes += v.byteLength;
            this.cache.set(key, v ?? null, v ? v.byteLength : 64);
            return v;
          } catch (e) {
            err = e;
            await new Promise((r) => setTimeout(r, 250 * 2 ** attempt));
          }
        }
        throw err;
      })().finally(() => this.pending.delete(key));
      this.pending.set(key, p);
    }
    return p;
  }
}

interface OmeAxis { name: string; type?: string }

export class OmeZarrSource implements Source {
  name: string;
  dims: 2 | 3 = 3;
  channels: 1 | 3 = 1;
  levels: Level[] = [];
  unit = "";
  attribution?: string;
  private store: CountingStore;
  private arrays: zarr.Array<zarr.DataType, CountingStore>[] = [];
  private axes: { z: number; y: number; x: number; c: number; t: number; n: number } = { z: -1, y: -1, x: -1, c: -1, t: -1, n: 0 };
  private window: [number, number] | null = null;  // intensity window for non-uint8 data
  private ngUrl: string;

  url: string;
  private constructor(url: string, name?: string) {
    this.url = url;
    this.store = new CountingStore(url);
    this.name = name ?? url.split("/").slice(-3).join("/");
    this.ngUrl = url;
  }
  get bytesRead() { return this.store.bytes; }

  static async open(url: string, name?: string): Promise<OmeZarrSource> {
    const s = new OmeZarrSource(normalizeUrl(url), name);
    const root = zarr.root(s.store);
    const grp = await zarr.open(root, { kind: "group" });
    let attrs = grp.attrs as Record<string, any>;
    if (attrs.ome) attrs = attrs.ome;
    const ms = attrs.multiscales?.[0];
    if (!ms) throw new Error(`no OME multiscales at ${s.url}`);
    const axes: OmeAxis[] = (ms.axes ?? []).map((a: any) => (typeof a === "string" ? { name: a } : a));
    const names = axes.map((a) => a.name.toLowerCase());
    s.axes = { z: names.indexOf("z"), y: names.indexOf("y"), x: names.indexOf("x"),
               c: names.findIndex((n, i) => n === "c" || axes[i].type === "channel"), t: names.indexOf("t"), n: names.length };
    if (s.axes.y < 0 || s.axes.x < 0) throw new Error("need y and x axes");
    s.dims = 2;  // a z axis of extent 1 (as IDR's 2-D images carry) is still an image
    s.unit = (axes[s.axes.x] as any).unit ?? "";
    for (const d of ms.datasets) {
      const arr = await zarr.open(root.resolve(d.path), { kind: "array" });
      s.arrays.push(arr as any);
      const sc = (d.coordinateTransformations ?? []).find((t: any) => t.type === "scale")?.scale ?? names.map(() => 1);
      const pick = (v: number[], i: number, dflt: number) => (i >= 0 ? v[i] : dflt);
      s.levels.push({
        shape: [pick(arr.shape, s.axes.z, 1), arr.shape[s.axes.y], arr.shape[s.axes.x]],
        chunk: [pick(arr.chunks, s.axes.z, 1), arr.chunks[s.axes.y], arr.chunks[s.axes.x]],
        scale: [pick(sc, s.axes.z, 1), sc[s.axes.y], sc[s.axes.x]],
      });
    }
    if (s.axes.z >= 0 && s.arrays[0].shape[s.axes.z] > 1) s.dims = 3;
    if (s.axes.c >= 0 && s.arrays[0].shape[s.axes.c] >= 3) s.channels = 3;
    const dt = String(s.arrays[0].dtype);
    if (dt !== "uint8") {  // window from the coarsest level, then every read maps to 0-255
      const L = s.levels.length - 1, sh = s.levels[L].shape;
      s.window = [0, 1];
      const r = await s.readRaw(L, [0, 0, 0], sh);
      const v = Float64Array.from(r.data.filter((x) => x !== 0)).sort();
      s.window = v.length ? [v[Math.floor(v.length * 0.005)], v[Math.floor(v.length * 0.995)]] : [0, 1];
    }
    return s;
  }

  private async readRaw(level: number, lo: Vec3, hi: Vec3): Promise<Region> {
    const arr = this.arrays[level];
    const lv = this.levels[level];
    const out = new Float32Array((hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]) * this.channels);
    const a = lo.map((v, i) => Math.max(0, Math.min(v, lv.shape[i]))) as Vec3;
    const b = hi.map((v, i) => Math.max(0, Math.min(v, lv.shape[i]))) as Vec3;
    const shape: Vec3 = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    if (a.some((v, i) => b[i] <= v)) return { data: out, shape, channels: this.channels };
    const sel: (number | zarr.Slice | null)[] = new Array(this.axes.n).fill(0);
    const ax = this.axes;
    if (ax.z >= 0) sel[ax.z] = this.dims === 3 ? zarr.slice(a[0], b[0]) : 0;
    sel[ax.y] = zarr.slice(a[1], b[1]);
    sel[ax.x] = zarr.slice(a[2], b[2]);
    if (ax.c >= 0) sel[ax.c] = this.channels === 3 ? zarr.slice(0, 3) : 0;
    const res = await zarr.get(arr, sel as any);
    const data = res.data as ArrayLike<number>;
    // result axes are the non-integer selections in array order
    const kept = sel.map((s, i) => (typeof s === "number" ? -1 : i)).filter((i) => i >= 0);
    const strideOf = (axis: number) => (axis < 0 || !kept.includes(axis) ? 0 : res.stride[kept.indexOf(axis)]);
    const sz = strideOf(ax.z), sy = strideOf(ax.y), sx = strideOf(ax.x), sc = this.channels === 3 ? strideOf(ax.c) : 0;
    const C = this.channels;
    for (let z = a[0]; z < b[0]; z++)
      for (let y = a[1]; y < b[1]; y++)
        for (let x = a[2]; x < b[2]; x++) {
          const src = (z - a[0]) * sz + (y - a[1]) * sy + (x - a[2]) * sx;
          const dst = (((z - lo[0]) * shape[1] + (y - lo[1])) * shape[2] + (x - lo[2])) * C;
          for (let c = 0; c < C; c++) out[dst + c] = Number(data[src + c * sc]);
        }
    return { data: out, shape, channels: C };
  }

  async read(level: number, lo: Vec3, hi: Vec3): Promise<Region> {
    const r = await this.readRaw(level, lo, hi);
    if (this.window) {
      const [w0, w1] = this.window, k = 255 / Math.max(w1 - w0, 1e-9);
      for (let i = 0; i < r.data.length; i++) r.data[i] = Math.min(255, Math.max(0, (r.data[i] - w0) * k));
    }
    return r;
  }

  link(level: number, center: Vec3): string | null {
    const s0 = this.levels[0].scale, f = 2 ** level;
    const pos = [center[0] * (this.dims === 3 ? f : 1) + 0.5, center[1] * f + 0.5, center[2] * f + 0.5];
    const unit = this.unit.startsWith("micro") ? 1e-6 : this.unit.startsWith("nano") || this.unit === "" ? 1e-9 : 1;
    const names = this.dims === 3 ? ["z", "y", "x"] : ["y", "x"];
    const dims: Record<string, [number, string]> = {};
    names.forEach((n, i) => (dims[n] = [s0[this.dims === 3 ? i : i + 1] * unit, "m"]));
    const state = {
      dimensions: dims, position: this.dims === 3 ? pos : pos.slice(1), crossSectionScale: 1, layout: "xy",
      layers: [{ type: "image", source: `zarr://${this.ngUrl}`, name: this.name }],
    };
    return "https://neuroglancer-demo.appspot.com/#!" + encodeURIComponent(JSON.stringify(state));
  }
}

// ---------------------------------------------------------------------------------------
// Tile pyramids (XYZ / TMS / WMTS templates / Deep Zoom)
// ---------------------------------------------------------------------------------------

export interface TileSpec {
  name: string;
  /** URL of tile (col x, row y) at zoom z, where zoom `maxZoom` is the finest. */
  url: (z: number, x: number, y: number) => string;
  tileSize: number;
  maxZoom: number;
  minZoom?: number;
  /** Full-resolution size in pixels (default: a square web-map world, tileSize * 2^maxZoom). */
  width?: number;
  height?: number;
  /** Pixels each tile overlaps its neighbours by (Deep Zoom). */
  overlap?: number;
  channels?: 1 | 3;
  unit?: string;
  pixelSize?: number;  // physical size of a full-resolution pixel, in `unit`
  attribution?: string;
  link?: (zoom: number, x: number, y: number) => string | null;
  /** Requests in flight at once (a cutout service rate-limits; plain tile CDNs do not). */
  maxConcurrent?: number;
}

/** Decodes an image to RGBA pixels; the browser's decoder by default (tests swap it). */
export let decodeImage = async (blob: Blob): Promise<{ data: Uint8ClampedArray; width: number; height: number }> => {
  const bmp = await createImageBitmap(blob);
  const cv = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = cv.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(bmp, 0, 0);
  const img = ctx.getImageData(0, 0, bmp.width, bmp.height);
  bmp.close();
  return { data: img.data, width: img.width, height: img.height };
};
export function setDecoder(f: typeof decodeImage) { decodeImage = f; }

export class TileSource implements Source {
  name: string;
  dims: 2 | 3 = 2;
  channels: 1 | 3;
  levels: Level[] = [];
  unit: string;
  attribution?: string;
  bytesRead = 0;
  private cache = new Lru<Promise<{ data: Uint8ClampedArray; width: number; height: number } | null>>(4000);
  private nLevels: number;

  spec: TileSpec;
  constructor(spec: TileSpec) {
    this.spec = spec;
    this.name = spec.name;
    this.channels = spec.channels ?? 3;
    this.unit = spec.unit ?? "px";
    this.attribution = spec.attribution;
    const ts = spec.tileSize;
    const W = spec.width ?? ts * 2 ** spec.maxZoom, H = spec.height ?? ts * 2 ** spec.maxZoom;
    this.nLevels = spec.maxZoom - (spec.minZoom ?? 0) + 1;
    for (let i = 0; i < this.nLevels; i++) {
      const f = 2 ** i, px = (spec.pixelSize ?? 1) * f;
      this.levels.push({ shape: [1, Math.ceil(H / f), Math.ceil(W / f)], chunk: [1, ts, ts], scale: [px, px, px] });
    }
  }

  private active = 0;
  private queue: (() => void)[] = [];
  private async slot<T>(f: () => Promise<T>): Promise<T> {
    const max = this.spec.maxConcurrent ?? 12;
    if (this.active >= max) await new Promise<void>((r) => this.queue.push(r));
    this.active++;
    try { return await f(); } finally { this.active--; this.queue.shift()?.(); }
  }

  private blobs = new Lru<Promise<Blob | null>>(4000);

  /** The tile's image file (cached), as the viewer draws it and the search decodes it. */
  blob(zoom: number, x: number, y: number): Promise<Blob | null> {
    const key = `${zoom}/${x}/${y}`;
    let b = this.blobs.get(key);
    if (!b) {
      b = (async () => {
        for (let attempt = 0; attempt < 5; attempt++) {
          let wait = 300 * 2 ** attempt;
          try {
            const r = await this.slot(() => fetch(this.spec.url(zoom, x, y)));
            if (r.status === 404 || r.status === 400) return null;  // no tile there
            if (r.status >= 500 && r.status !== 503 && attempt >= 1) return null;  // a broken tile (some caches have gaps)
            if (r.status === 429 || r.status === 503) wait = 1500 * 2 ** attempt;  // asked to slow down
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            const blob = await r.blob();
            this.bytesRead += blob.size;
            return blob;
          } catch (e) {
            // a server error without CORS headers reaches the page as a failed fetch (TypeError):
            // treat it like the 5xx it most likely is, and give up after two tries
            if (e instanceof TypeError && attempt >= 1) return null;
            await new Promise((res) => setTimeout(res, wait));
          }
        }
        return null;
      })();
      this.blobs.set(key, b, 1);
    }
    return b;
  }

  private tile(zoom: number, x: number, y: number) {
    const key = `${zoom}/${x}/${y}`;
    let t = this.cache.get(key);
    if (!t) {
      t = this.blob(zoom, x, y).then((b) => (b ? decodeImage(b).catch(() => null) : null));
      this.cache.set(key, t, 1);
    }
    return t;
  }

  async read(level: number, lo: Vec3, hi: Vec3): Promise<Region> {
    const ts = this.spec.tileSize, ov = this.spec.overlap ?? 0;
    const zoom = this.spec.maxZoom - level;
    const shape: Vec3 = [1, hi[1] - lo[1], hi[2] - lo[2]];
    const C = this.channels;
    const out = new Float32Array(shape[1] * shape[2] * C).fill(NaN);  // NaN = no data (outside, or a tile that failed)
    const L = this.levels[level];
    const y0 = Math.max(0, lo[1]), y1 = Math.min(L.shape[1], hi[1]);
    const x0 = Math.max(0, lo[2]), x1 = Math.min(L.shape[2], hi[2]);
    if (y1 <= y0 || x1 <= x0) return { data: out, shape, channels: C };
    const jobs: Promise<void>[] = [];
    for (let ty = Math.floor(y0 / ts); ty <= Math.floor((y1 - 1) / ts); ty++)
      for (let tx = Math.floor(x0 / ts); tx <= Math.floor((x1 - 1) / ts); tx++)
        jobs.push(this.tile(zoom, tx, ty).then((img) => {
          if (!img) return;  // stays NaN
          const ox = tx > 0 ? ov : 0, oy = ty > 0 ? ov : 0;  // Deep Zoom tiles carry their overlap
          const ya = Math.max(y0, ty * ts), yb = Math.min(y1, (ty + 1) * ts);
          const xa = Math.max(x0, tx * ts), xb = Math.min(x1, (tx + 1) * ts);
          for (let y = ya; y < yb; y++) {
            const sy = y - ty * ts + oy;
            if (sy >= img.height) break;
            for (let x = xa; x < xb; x++) {
              const sx = x - tx * ts + ox;
              if (sx >= img.width) break;
              const s = (sy * img.width + sx) * 4, d = ((y - lo[1]) * shape[2] + (x - lo[2])) * C;
              if (C === 3) { out[d] = img.data[s]; out[d + 1] = img.data[s + 1]; out[d + 2] = img.data[s + 2]; }
              else out[d] = 0.299 * img.data[s] + 0.587 * img.data[s + 1] + 0.114 * img.data[s + 2];
            }
          }
        }));
    await Promise.all(jobs);
    return { data: out, shape, channels: C };
  }

  link(level: number, center: Vec3): string | null {
    if (!this.spec.link) return null;
    const zoom = this.spec.maxZoom - level;
    return this.spec.link(zoom, center[2], center[1]);
  }
}

/** Luminance of a region (z, y, x), the channel the cheap scores use. */
export function luminance(r: Region): Float32Array {
  if (r.channels === 1) return r.data;
  const n = r.data.length / 3, out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = 0.299 * r.data[3 * i] + 0.587 * r.data[3 * i + 1] + 0.114 * r.data[3 * i + 2];
  return out;
}

/** A Zoomify pyramid (ImageProperties.xml + TileGroupN/z-x-y.jpg), as ESA/Hubble and ESA/Webb
 * publish their large images. Tier t counts ceil(W / (tile * 2^(maxTier - t))) tiles across
 * (OpenLayers' "default" tier sizes); a tile's group is its index over all tiers / 256. */
export async function openZoomify(base: string, name: string, attribution?: string, pixelSize?: number, unit?: string): Promise<TileSource> {
  base = base.replace(/\/+$/, "");
  const xml = await (await fetch(`${base}/ImageProperties.xml`)).text();
  const attr = (k: string) => Number(new RegExp(`${k}\\s*=\\s*["'](\\d+)["']`, "i").exec(xml)![1]);
  const W = attr("WIDTH"), H = attr("HEIGHT"), ts = attr("TILESIZE"), total = attr("NUMTILES");
  const tiers: [number, number][] = [];
  for (let f = 1; ; f *= 2) {
    tiers.unshift([Math.ceil(W / (ts * f)), Math.ceil(H / (ts * f))]);
    if (W / f <= ts && H / f <= ts) break;
  }
  const cum = [0];
  for (const [nx, ny] of tiers) cum.push(cum[cum.length - 1] + nx * ny);
  if (cum[cum.length - 1] !== total) console.warn(`zoomify ${name}: computed ${cum[cum.length - 1]} tiles, file says ${total}`);
  const maxZoom = tiers.length - 1;
  return new TileSource({
    name, tileSize: ts, maxZoom, width: W, height: H, channels: 3, attribution, pixelSize, unit,
    url: (z, x, y) => `${base}/TileGroup${Math.floor((cum[z] + y * tiers[z][0] + x) / 256)}/${z}-${x}-${y}.jpg`,
  });
}

/** A field of the DESI Legacy Imaging Surveys as a tile pyramid built from the survey's cutout
 * service (its map tiles send no CORS header; cutout.jpg does). The field is `size` pixels of
 * `pixscale` arcsec around (ra0, dec0), north up and east left; each tile is a cutout centred on
 * the tile at that level's pixel scale (a flat approximation, fine over a degree or two). */
export function legacySurvey(ra0: number, dec0: number, layer = "ls-dr9", size = 16384, pixscale = 0.262, name?: string): TileSource {
  const ts = 512, maxZoom = Math.round(Math.log2(size / ts));
  const cosd = Math.cos((dec0 * Math.PI) / 180);
  return new TileSource({
    name: name ?? `Legacy Surveys ${layer} at RA ${ra0.toFixed(3)}, Dec ${dec0.toFixed(3)}`,
    tileSize: ts, maxZoom, width: size, height: size, channels: 3, unit: "arcsec", pixelSize: pixscale,
    attribution: "DESI Legacy Imaging Surveys (legacysurvey.org), cutout service",
    maxConcurrent: 3,
    url: (z, x, y) => {
      const level = maxZoom - z, ps = pixscale * 2 ** level, n = size / 2 ** level;
      const dx = ((x + 0.5) * ts - n / 2) * ps, dy = ((y + 0.5) * ts - n / 2) * ps;  // arcsec from the field centre
      const ra = ra0 - dx / 3600 / cosd, dec = dec0 - dy / 3600;
      return `https://www.legacysurvey.org/viewer/cutout.jpg?ra=${ra.toFixed(6)}&dec=${dec.toFixed(6)}&layer=${layer}&pixscale=${ps.toFixed(4)}&size=${ts}`;
    },
    link: (zoom, x, y) => {
      const level = maxZoom - zoom, ps = pixscale * 2 ** level, n = size / 2 ** level;
      const ra = ra0 - ((x - n / 2) * ps) / 3600 / cosd, dec = dec0 - ((y - n / 2) * ps) / 3600;
      return `https://www.legacysurvey.org/viewer?ra=${ra.toFixed(5)}&dec=${dec.toFixed(5)}&layer=${layer}&zoom=14`;
    },
  });
}

/** Fraction of a region's values that are no data (NaN). */
export function missingFraction(a: Float32Array): number {
  let n = 0;
  for (let i = 0; i < a.length; i++) if (a[i] !== a[i]) n++;
  return a.length ? n / a.length : 1;
}

/** Copy with no-data (NaN) replaced by the median of the valid values (neutral for the scan). */
export function fillMissing(a: Float32Array): Float32Array {
  const out = Float32Array.from(a), valid: number[] = [];
  const step = Math.max(1, Math.floor(a.length / 100000));
  for (let i = 0; i < a.length; i += step) if (a[i] === a[i]) valid.push(a[i]);
  valid.sort((x, y) => x - y);
  const med = valid.length ? valid[valid.length >> 1] : 0;
  for (let i = 0; i < out.length; i++) if (out[i] !== out[i]) out[i] = med;
  return out;
}

/** A patch worth embedding: all data present, not blank (no-data zeros or a flat area). */
export function informative(a: Float32Array): boolean {
  let n = 0, zeros = 0, sum = 0, sq = 0;
  for (let i = 0; i < a.length; i++) {
    const v = a[i];
    if (v !== v) return false;
    n++; sum += v; sq += v * v;
    if (v === 0) zeros++;
  }
  if (!n || zeros > 0.25 * n) return false;
  const m = sum / n;
  return sq / n - m * m > 1.5 * 1.5;
}
