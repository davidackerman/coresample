// A TIFF on the web (an address ending in .tif or .tiff), read by byte ranges and served as an
// OME-Zarr made on the fly, as an image file from this computer is (src/imagefile.ts,
// src/local.ts): Neuroglancer and the search read it by URL. Its levels are the file's own
// pyramid where it has one (a COG's overviews) and, below the coarsest it has, levels made from
// every 2^k-th row (rows 0, 2^k, 2*2^k...: a coarser level's rows are a finer one's), each row
// averaged along x. An uncompressed row costs only its own bytes, so
// a level 2^k coarser than the file reads one row in 2^k: a zoomed-out view of a TIFF with no
// pyramid costs 2^k times what a pyramid's level would, not the whole file. A compressed tile or
// strip is read whole (and kept) when a sampled row falls in it.
import { serveFolder, type Folder } from "./local.ts";
import { OmeZarrSource, mirrorsOf } from "./sources.ts";

/** Addresses read as a TIFF. */
export const TIFF_URL = /^https?:\/\/[^?#]+\.(tiff?|btf|tf8)([?#].*)?$/i;

type Arr = Uint8Array | Int8Array | Uint16Array | Int16Array | Uint32Array | Int32Array | Float32Array | Float64Array;
type Ctor = { new (n: number): Arr; new (b: ArrayBufferLike, o: number, n: number): Arr };

/** One image of the file (an IFD): the full resolution, or one of its overviews. */
interface Ifd {
  index: number; im: any; w: number; h: number; tiled: boolean; tw: number; th: number;
  comp: number; spp: number; planar: 1 | 2; b: number; fmt: number; le: boolean; across: number; down: number;
  ctor: Ctor; photo: number; table?: Promise<{ off: Float64Array; cnt: Float64Array }>; decoder?: Promise<any>;
  /** Set by decoder(): the decoded block's byte order is little-endian whatever the file's (geotiff's
   * floating-point predictor writes it so), and the horizontal predictor is undone here, after
   * the bytes are swapped to this machine's order (geotiff's would add them in the wrong order). */
  decodedLE?: boolean; ownPredictor?: boolean;
}
/** A level served: rows and columns of `ifd`, every `s`-th row read, `s` columns averaged. */
interface Lv { ifd: Ifd; s: number; y: number; x: number; chunk: [number, number]; scale: [number, number] }

const MAX_COARSEST = 1.5e9;  // bytes the whole coarsest level may cost to read
const VIEW_BYTES = 64e6;     // what the first view may cost (a 1600 x 1000 px screen)
const EXTENT = 8192;         // image pixels a chunk of a made level spans each way
const BLOCK = 8192;          // columns of an uncompressed row read and kept together
const side = (n: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, 2 ** Math.round(Math.log2(n))));

const CTORS: Record<string, Ctor> = { "1:1": Uint8Array, "1:2": Uint16Array, "1:4": Uint32Array, "2:1": Int8Array, "2:2": Int16Array,
                                      "2:4": Int32Array, "3:4": Float32Array, "3:8": Float64Array } as any;
const DTYPE = new Map<Ctor, string>([[Uint8Array as any, "|u1"], [Int8Array as any, "|i1"], [Uint16Array as any, "<u2"], [Int16Array as any, "<i2"],
                                     [Uint32Array as any, "<u4"], [Int32Array as any, "<i4"], [Float32Array as any, "<f4"], [Float64Array as any, "<f4"]]);

/** Byte ranges of one file, spread over the addresses it has (S3 answers 6 at a time per
 * address), a few retries each; counts what it downloads. */
class Ranges {
  read = 0;
  private hosts: string[];
  private n = 0;
  private active = 0;
  private waiting: (() => void)[] = [];
  constructor(url: string) { this.hosts = [url]; }
  /** Other addresses of the file, each used only if it gives the same size and first bytes. */
  async addMirrors(size: number, head: Uint8Array) {
    const ok = await Promise.all(mirrorsOf(this.hosts[0]).map(async (u) => {
      try {
        const r = await fetch(u, { headers: { Range: `bytes=0-${head.length - 1}` } });
        const b = new Uint8Array(await r.arrayBuffer());
        return r.status === 206 && Number(r.headers.get("content-range")?.split("/")[1]) === size && b.every((v, i) => v === head[i]) ? u : null;
      } catch { return null; }
    }));
    for (const u of ok) if (u) this.hosts.push(u);
  }
  async get(off: number, len: number): Promise<Uint8Array> {
    if (this.active >= 8 * this.hosts.length) await new Promise<void>((r) => this.waiting.push(r));
    this.active++;
    try {
      for (let attempt = 0; ; attempt++) {
        const u = attempt ? this.hosts[0] : this.hosts[this.n++ % this.hosts.length];
        try {
          const r = await fetch(u, { headers: { Range: `bytes=${off}-${off + len - 1}` } });
          if (r.status !== 206) { void r.body?.cancel(); throw new Error(r.status === 200 ? "the host does not read parts of files (byte ranges)" : `HTTP ${r.status}`); }
          const b = new Uint8Array(await r.arrayBuffer());
          if (b.length !== len) throw new Error(`asked for ${len} bytes, got ${b.length}`);
          this.read += len;
          return b;
        } catch (e) {
          if (attempt >= 3 || /byte ranges/.test(String(e))) throw e;
          await new Promise((r) => setTimeout(r, 300 * 2 ** attempt));
        }
      }
    } finally {
      this.active--;
      this.waiting.shift()?.();
    }
  }
}

/** Values of `u8` (bytes in the file's order, or little-endian: `le`) as numbers of the file's
 * type, in this machine's order. */
function typed(u8: Uint8Array, I: Ifd, le = I.le): Arr {
  let u = u8.byteOffset % I.b ? u8.slice() : u8;
  if (!le && I.b > 1) {
    if (u === u8) u = u8.slice();
    for (let i = 0; i < u.length; i += I.b) u.subarray(i, i + I.b).reverse();
  }
  return new I.ctor(u.buffer, u.byteOffset, u.byteLength / I.b);
}

async function describe(im: any, index: number): Promise<Ifd> {
  const fd = im.getFileDirectory(), spp = im.getSamplesPerPixel(), bits = im.getBitsPerSample(), fmt = im.getSampleFormat() || 1;
  for (let i = 1; i < spp; i++) if (im.getBitsPerSample(i) !== bits || (im.getSampleFormat(i) || 1) !== fmt) throw new Error("samples of different types are not read here");
  const ctor = bits % 8 ? null : CTORS[`${fmt}:${bits / 8}`] ?? null;
  if (!ctor) throw new Error(`${bits}-bit ${fmt === 3 ? "float" : fmt === 2 ? "signed" : "unsigned"} samples are not read here`);
  const w = im.getWidth(), h = im.getHeight(), tw = im.getTileWidth(), th = im.isTiled ? im.getTileHeight() : Math.min(h, im.getTileHeight());  // tiles are padded, strips not
  const comp = Number(fd.getValue("Compression") ?? 1) || 1, photo = Number(fd.getValue("PhotometricInterpretation") ?? 1);
  // YCbCr outside JPEG stores its colour subsampled (a layout of its own); JPEG's decoder undoes
  // the subsampling, and block() turns its Y, Cb, Cr into red, green, blue
  if (photo === 6 && comp !== 7) throw new Error("YCbCr colour is read here only in JPEG-compressed TIFFs");
  return { index, im, w, h, tiled: im.isTiled, tw, th, comp, spp, planar: im.planarConfiguration === 2 ? 2 : 1,
           b: bits / 8, fmt, le: !!im.littleEndian, across: Math.ceil(w / tw), down: Math.ceil(h / th), ctor, photo };
}

/** A promise kept for next time, forgotten if it fails (so a later read tries again). */
function once<T>(I: Ifd, key: "table" | "decoder", make: () => Promise<T>): Promise<T> {
  if (!I[key]) {
    const p = make();
    (I as any)[key] = p;
    p.catch(() => { if (I[key] === (p as any)) I[key] = undefined; });
  }
  return I[key] as Promise<T>;
}

/** JPEG's Y, Cb, Cr (8 bits, interleaved) to red, green, blue in place (JFIF, as geotiff's readRGB). */
function fromYCbCr(u: Uint8Array, spp: number) {
  const v = new Uint8ClampedArray(u.buffer, u.byteOffset, u.length);  // rounds and clamps to 0-255
  for (let i = 0; i + 2 < v.length; i += spp) {
    const y = v[i], cb = v[i + 1] - 128, cr = v[i + 2] - 128;
    v[i] = y + 1.402 * cr;
    v[i + 1] = y - 0.344136 * cb - 0.714136 * cr;
    v[i + 2] = y + 1.772 * cb;
  }
}

/** The tile or strip offsets and byte counts of an image, read once. */
function table(I: Ifd) {
  return once(I, "table", async () => {
    const fd = I.im.getFileDirectory(), [o, c] = I.tiled ? ["TileOffsets", "TileByteCounts"] : ["StripOffsets", "StripByteCounts"];
    let off: unknown, cnt: unknown;
    try { [off, cnt] = await Promise.all([fd.loadValue(o), fd.loadValue(c)]); }
    catch (e) {  // geotiff (3.0.5) keeps a failed load of a long tag for good: forgotten, so the next read fetches it again
      for (const code of I.tiled ? [324, 325] : [273, 279]) { const a = fd.deferredArrays?.get(code); if (a) a.fullFetchPromise = null; }
      throw e;
    }
    return { off: Float64Array.from(off as ArrayLike<number | bigint>, Number), cnt: Float64Array.from(cnt as ArrayLike<number | bigint>, Number) };
  });
}
function decoder(I: Ifd) {
  return once(I, "decoder", async () => {
    const fd = I.im.getFileDirectory(), get = async (t: string) => { try { return await fd.loadValue(t); } catch { return undefined; } };
    const predictor = Number(await get("Predictor")) || 1, swapped = !I.le && I.b > 1;
    I.decodedLE = swapped && predictor === 3;
    I.ownPredictor = swapped && predictor === 2;
    const p: Record<string, unknown> = { tileWidth: I.tw, tileHeight: I.th, planarConfiguration: I.planar, bitsPerSample: await get("BitsPerSample"),
                                         predictor: I.ownPredictor ? 1 : predictor, samplesPerPixel: I.spp };
    if (I.comp === 7) p.JPEGTables = await get("JPEGTables");
    if (I.comp === 34887) p.LercParameters = await get("LercParameters");
    return (await import("geotiff")).getDecoder(I.comp, p as any);
  });
}
/** The horizontal predictor undone: each sample plus the one before it in its row, wrapping as
 * the type does (`v` in this machine's order, rows of `w` pixels of `n` samples). */
function accumulate(v: Arr, w: number, n: number) {
  const row = w * n;
  for (let r = 0; r + row <= v.length; r += row) for (let i = r + n; i < r + row; i++) v[i] += v[i - n];
}

export interface RemoteTiff { folder: Folder; name: string; bytes: () => number; startScale: number; note: string }

/** A remote TIFF as an OME-Zarr folder (for serveFolder), how much it has downloaded, and the
 * scale to open at (all of it, unless that view would cost more than VIEW_BYTES). */
export async function openTiff(url: string): Promise<RemoteTiff> {
  const name = decodeURIComponent(url.split(/[?#]/)[0].split("/").pop() ?? "TIFF");
  const tif = await (await import("geotiff")).fromUrl(url, { allowFullFile: false });  // loaded when first needed
  // the full resolution, then its overviews (smaller images of the same kind; masks left out)
  const ifds: Ifd[] = [];
  for (let i = 0; i < 32; i++) {
    let im: any;
    try { im = await tif.getImage(i); } catch { break; }
    let sub = 0;
    try { sub = Number(im.getFileDirectory().getValue("NewSubfileType") ?? 0); } catch { /* deferred: not a mask */ }
    if (i && sub & 4) continue;
    const I = await describe(im, i), p = ifds.at(-1);
    if (p && !(I.w < p.w && I.h <= p.h && I.spp === p.spp && I.b === p.b && I.fmt === p.fmt)) break;  // a z plane or another image: not read here
    ifds.push(I);
  }
  const I0 = ifds[0], net = new Ranges(url);
  const pxb = (I: Ifd) => I.b * (I.planar === 1 ? I.spp : 1);
  // the file's size: from the range answer, else (a host that does not let pages see
  // Content-Range) a HEAD's length, else what the full resolution would take uncompressed (more
  // than a compressed file is: the cost guard below stays on, erring on the safe side)
  const size = await (async () => {
    const r = await fetch(url, { headers: { Range: "bytes=0-15" } });
    if (r.status !== 206) { void r.body?.cancel(); throw new Error(`${name}: the host does not read parts of files (byte ranges)`); }
    const head = new Uint8Array(await r.arrayBuffer());
    let n = Number(r.headers.get("content-range")?.split("/")[1]);
    if (n > 0) { await net.addMirrors(n, head); return n; }
    try { n = Number((await fetch(url, { method: "HEAD" })).headers.get("content-length")); } catch { n = NaN; }
    return n > 0 ? n : ifds.reduce((t, I) => t + I.w * I.h * pxb(I), 0);
  })();
  // a projected GeoTIFF's pixel size, in metres (else pixels, unitless)
  let unit: string | undefined, px: [number, number] = [1, 1];
  try {
    const g = I0.im.getGeoKeys(), r = I0.im.getResolution();
    if (g?.GTModelTypeGeoKey === 1 && (g.ProjLinearUnitsGeoKey ?? 9001) === 9001 && Math.abs(r[0]) > 0 && Math.abs(r[1]) > 0) { unit = "meter"; px = [Math.abs(r[1]), Math.abs(r[0])]; }
  } catch { /* no geo keys */ }
  // levels at 2^k: the file's image nearest below, every s-th row of it
  const levels: Lv[] = [];
  for (let k = 0; k < 40; k++) {
    const want = 2 ** k;
    let I = ifds[0];
    for (const J of ifds) if (I0.w / J.w <= want * 1.01) I = J;
    const f = I0.w / I.w, s = Math.max(1, Math.round(want / f)), y = Math.ceil(I.h / s), x = Math.ceil(I.w / s);
    // a chunk spans about EXTENT pixels of the image each way, so that the coarser levels the
    // viewer loads under a view (to show something while the finer one comes) span little more
    levels.push({ ifd: I, s, y, x, chunk: s === 1 && I.tiled ? [I.th, I.tw] : [side(EXTENT / s, 64, 512), side(EXTENT / s, 64, 2048)],
                  scale: [px[0] * (I0.h / I.h) * s, px[1] * f * s] });
    if (Math.max(y, x) <= 256) break;
  }
  // what reading a level costs, per pixel of it: an uncompressed row, the row's bytes; a
  // compressed one, every tile or strip a sampled row falls in
  const perPx = (L: Lv) => {
    const I = L.ifd;
    if (L.s === 1) return pxb(I);
    if (I.comp === 1) return L.s * pxb(I);
    return (size / (I0.w * I0.h)) * L.s * L.s * Math.min(1, I.th / L.s);  // the file's bytes per pixel, s x s pixels a pixel, the share of tile rows read
  };
  const coarsest = levels.at(-1)!, cost = perPx(coarsest) * coarsest.y * coarsest.x;
  if (cost > MAX_COARSEST) {
    const I = coarsest.ifd, what = ifds.length > 1 ? `its smallest copy is ${I.w} x ${I.h} px` : "it has no pyramid (smaller copies)";
    throw new Error(`${name}: ${what} and is compressed in ${I.tiled ? "tiles" : "strips"} of ${I.th} rows, read whole, so even zoomed all the way out a view would read ${(cost / 1e9).toFixed(1)} GB of it. Add overviews (gdaladdo, or make it a COG) or convert it to OME-Zarr.`);
  }
  // the first view: all of it, unless that would read more than VIEW_BYTES. At a scale between
  // 2^k and 2^(k+1) the viewer shows level k, (scale / 2^k)^2 of its pixels to a screen pixel.
  const L0 = levels[0], want = Math.max(1, Math.max(L0.y, L0.x) / 1600);
  const viewCost = (sc: number) => { const k = Math.min(levels.length - 1, Math.floor(Math.log2(sc))); return 1.6e6 * (sc / 2 ** k) ** 2 * perPx(levels[k]); };
  let startScale = want;
  while (startScale > 1 && viewCost(startScale) > VIEW_BYTES) startScale = 2 ** (Math.ceil(Math.log2(startScale)) - 1);
  const own = ifds.length > 1 ? `its own ${ifds.length} levels` : "no pyramid";
  const note = `${name}: ${(size / 1e9).toFixed(1)} GB, ${own}${coarsest.ifd !== ifds.at(-1) || coarsest.s > 1 ? `; coarser levels from every 2^k-th row` : ""}.`;

  // decoded tiles and strips, kept while they fit
  const kept = new Map<string, { p: Promise<Arr>; n: number }>();
  let keptBytes = 0;
  const block = (I: Ifd, idx: number): Promise<Arr> => {
    const key = `${I.index}:${idx}`, hit = kept.get(key);
    if (hit) { kept.delete(key); kept.set(key, hit); return hit.p; }
    const vals = I.tw * I.th * (I.planar === 1 ? I.spp : 1), n = vals * I.b;
    const p = (async () => {
      const { off, cnt } = await table(I);
      if (!cnt[idx]) return new I.ctor(vals);
      const raw = await net.get(off[idx], cnt[idx]);
      const out = new Uint8Array(await (await decoder(I)).decode(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength)));
      if (I.photo === 6 && I.spp >= 3 && I.planar === 1) fromYCbCr(out, I.spp);
      const v = typed(out, I, I.decodedLE || I.le);
      if (I.ownPredictor) accumulate(v, I.tw, I.planar === 1 ? I.spp : 1);
      return v;
    })();
    const e = { p, n };
    kept.set(key, e);
    keptBytes += n;
    p.catch(() => { if (kept.get(key) === e) { kept.delete(key); keptBytes -= n; } });
    for (const [k, v] of kept) { if (keptBytes <= 384 << 20) break; kept.delete(k); keptBytes -= v.n; }
    return p;
  };

  // uncompressed rows, in blocks of BLOCK columns (a tile's width, tiled), kept while they fit:
  // every level samples rows 0, s, 2s..., so a coarser level's rows are a finer level's, read
  // once. Blocks asked for within a few milliseconds are read together: the adjacent blocks of a
  // row (and rows next to each other in the file) in one request.
  const bw = (I: Ifd) => (I.tiled ? I.tw : Math.min(I.w, BLOCK));
  const rowsKept = new Map<string, { p: Promise<Uint8Array>; n: number }>();
  let rowBytes = 0, timer: ReturnType<typeof setTimeout> | null = null;
  let batch: { I: Ifd; plane: number; Y: number; j: number; n: number; ok: (b: Uint8Array) => void; no: (e: unknown) => void }[] = [];
  const rowBlock = (I: Ifd, plane: number, Y: number, j: number): Promise<Uint8Array> => {
    const key = `${I.index}:${plane}:${Y}:${j}`, hit = rowsKept.get(key);
    if (hit) { rowsKept.delete(key); rowsKept.set(key, hit); return hit.p; }
    const B = bw(I), n = (I.tiled ? B : Math.min(I.w, (j + 1) * B) - j * B) * pxb(I);
    let ok!: (b: Uint8Array) => void, no!: (e: unknown) => void;
    const e = { p: new Promise<Uint8Array>((res, rej) => { ok = res; no = rej; }), n };
    rowsKept.set(key, e);
    rowBytes += n;
    e.p.catch(() => { if (rowsKept.get(key) === e) { rowsKept.delete(key); rowBytes -= n; } });
    for (const [k, v] of rowsKept) { if (rowBytes <= 256 << 20) break; rowsKept.delete(k); rowBytes -= v.n; }
    batch.push({ I, plane, Y, j, n, ok, no });
    timer ??= setTimeout(flush, 4);
    return e.p;
  };
  async function flush() {
    timer = null;
    const todo = batch;
    batch = [];
    let segs: { o: number; t: (typeof todo)[number] }[];
    try {
      segs = await Promise.all(todo.map(async (t) => {
        const { off } = await table(t.I), by = Math.floor(t.Y / t.I.th), blk = t.plane + by * t.I.across + (t.I.tiled ? t.j : 0);
        return { o: off[blk] + ((t.Y - by * t.I.th) * t.I.tw + (t.I.tiled ? 0 : t.j * bw(t.I))) * pxb(t.I), t };
      }));
    } catch (err) {  // the offsets did not load: these reads fail (and are tried again when next asked for)
      for (const t of todo) t.no(err);
      return;
    }
    segs.sort((p, q) => p.o - q.o);
    const groups: (typeof segs)[] = [];
    for (const g of segs) {
      const last = groups.at(-1), end = last ? last.at(-1)!.o + last.at(-1)!.t.n : 0;
      // one request for ranges close together, so long as the gap read with them is less than they are
      if (last && g.o >= end && (g.o === end || g.o - end < Math.min(32768, g.t.n)) && g.o + g.t.n - last[0].o <= 16 << 20) last.push(g); else groups.push([g]);
    }
    for (const g of groups) {
      const o = g[0].o;
      net.get(o, g.at(-1)!.o + g.at(-1)!.t.n - o).then((bytes) => { for (const x of g) x.t.ok(bytes.slice(x.o - o, x.o - o + x.t.n)); },
                                                       (err) => { for (const x of g) x.t.no(err); });
    }
  }
  /** Rows `Ys` of image I, columns [X0, X1), channel c, as stored (uncompressed images). */
  async function rawRows(I: Ifd, c: number, Ys: number[], X0: number, X1: number): Promise<Float64Array[]> {
    const cs = I.planar === 1 ? I.spp : 1, plane = I.planar === 2 ? c * I.across * I.down : 0, B = bw(I);
    return Promise.all(Ys.map(async (Y) => {
      const dst = new Float64Array(X1 - X0), j0 = Math.floor(X0 / B);  // 64-bit: 32-bit integers stay exact
      await Promise.all(Array.from({ length: Math.floor((X1 - 1) / B) - j0 + 1 }, async (_, k) => {
        const j = j0 + k, v = typed(await rowBlock(I, plane, Y, j), I), a = Math.max(X0, j * B), e = Math.min(X1, (j + 1) * B, I.w);
        for (let X = a; X < e; X++) dst[X - X0] = v[(X - j * B) * cs + (c % cs)];
      }));
      return dst;
    }));
  }
  /** Rows [Ya, Yb) of image I, columns [X0, X1), channel c, averaged: only those in the tile or
   * strip row that holds row Yc, which is read whole (compressed images). */
  async function codedRows(I: Ifd, c: number, spans: [number, number, number][], X0: number, X1: number): Promise<Float64Array[]> {
    const cs = I.planar === 1 ? I.spp : 1, plane = I.planar === 2 ? c * I.across * I.down : 0;
    return Promise.all(spans.map(async ([Ya, Yb, Yc]) => {
      const by = Math.floor(Yc / I.th), r0 = Math.max(Ya, by * I.th), r1 = Math.min(Yb, (by + 1) * I.th, I.h), dst = new Float64Array(X1 - X0);
      await Promise.all(Array.from({ length: Math.floor((X1 - 1) / I.tw) - Math.floor(X0 / I.tw) + 1 }, async (_, k) => {
        const bx = Math.floor(X0 / I.tw) + k, v = await block(I, plane + by * I.across + bx);
        const a = Math.max(X0, bx * I.tw), e = Math.min(X1, (bx + 1) * I.tw, I.w);
        for (let Y = r0; Y < r1; Y++) {
          const base = (Y - by * I.th) * I.tw - bx * I.tw;
          for (let X = a; X < e; X++) dst[X - X0] += v[(base + X) * cs + (c % cs)];
        }
        if (r1 - r0 > 1) for (let X = a; X < e; X++) dst[X - X0] /= r1 - r0;
      }));
      return dst;
    }));
  }

  const outCtor = (I: Ifd): Ctor => (I.ctor === (Float64Array as any) ? (Float32Array as any) : I.ctor);
  async function chunk(L: Lv, c: number, yi: number, xi: number): Promise<Arr> {
    const I = L.ifd, s = L.s, [cy, cx] = L.chunk, out = new (outCtor(I))(cy * cx);
    const y0 = yi * cy, y1 = Math.min(L.y, y0 + cy), x0 = xi * cx, x1 = Math.min(L.x, x0 + cx);
    if (y0 >= y1 || x0 >= x1) return out;
    const X0 = x0 * s, X1 = Math.min(I.w, x1 * s), ys = Array.from({ length: y1 - y0 }, (_, j) => y0 + j);
    const rows = I.comp === 1
      ? await rawRows(I, c, ys.map((y) => y * s), X0, X1)
      : await codedRows(I, c, ys.map((y) => [y * s, Math.min(I.h, (y + 1) * s), y * s]), X0, X1);
    const round = !(out instanceof Float32Array);
    rows.forEach((v, j) => {
      for (let i = 0; i < x1 - x0; i++) {
        const a = i * s, e = Math.min(v.length, a + s);
        let t = 0;
        for (let q = a; q < e; q++) t += v[q];
        out[j * cx + i] = round ? Math.round(t / (e - a)) : t / (e - a);
      }
    });
    return out;
  }

  // the OME-Zarr (v2, uncompressed): channels for a colour or multi-sample image
  const C = I0.spp, multi = C > 1, rgb = C >= 3 && (I0.photo === 2 || I0.photo === 6);  // YCbCr is read as RGB (block)
  const space = (n: string) => ({ name: n, type: "space", ...(unit ? { unit } : {}) });
  const attrs = {
    multiscales: [{ version: "0.4", name, axes: [...(multi ? [{ name: "c", type: "channel" }] : []), space("y"), space("x")],
      datasets: levels.map((L, k) => ({ path: String(k), coordinateTransformations: [{ type: "scale", scale: [...(multi ? [1] : []), ...L.scale] }] })) }],
    ...(multi ? { omero: { channels: Array.from({ length: C }, (_, k) => ({
      label: rgb && k < 3 ? ["red", "green", "blue"][k] : String(k), active: k < 3,
      color: ["FF0000", "00FF00", "0000FF", "FFFFFF", "00FFFF", "FF00FF", "FFFF00"][k % 7],
      ...(I0.ctor === (Uint8Array as any) ? { window: { start: 0, end: 255, min: 0, max: 255 } } : {}) })) } } : {}),
  };
  const text = (v: unknown, p: string) => new File([JSON.stringify(v)], p);
  const folder: Folder = {
    name,
    async list() { return []; },
    async file(path) {
      if (path === ".zgroup") return text({ zarr_format: 2 }, path);
      if (path === ".zattrs") return text(attrs, path);
      const parts = path.split("/"), L = /^\d+$/.test(parts[0]) ? levels[Number(parts[0])] : undefined;
      if (!L) return null;
      if (parts[1] === ".zarray" && parts.length === 2)
        return text({ zarr_format: 2, shape: [...(multi ? [C] : []), L.y, L.x], chunks: [...(multi ? [1] : []), ...L.chunk], dtype: DTYPE.get(L.ifd.ctor),
                      compressor: null, fill_value: 0, order: "C", filters: null, dimension_separator: "/" }, path);
      const idx = parts.slice(1).map(Number);
      if (idx.length !== (multi ? 3 : 2) || idx.some((v) => !Number.isInteger(v) || v < 0)) return null;
      const [c, yi, xi] = multi ? idx : [0, ...idx];
      if (c >= C || yi * L.chunk[0] >= L.y || xi * L.chunk[1] >= L.x) return null;
      const a = await chunk(L, c, yi, xi);
      return new File([new Uint8Array(a.buffer as ArrayBuffer, a.byteOffset, a.byteLength)], path);
    },
  };
  return { folder, name, bytes: () => net.read, startScale, note };
}

/** A remote TIFF opened for the page: served at ./local/<id>/ (src/local.ts) and read there as
 * an OME-Zarr, its downloads counted as the file's own bytes. */
export async function tiffSource(url: string, name?: string): Promise<{ s: OmeZarrSource; startScale: number; note: string }> {
  const t = await openTiff(url);
  const s = await OmeZarrSource.open(await serveFolder(t.folder), name ?? t.name, { mirrors: false, bytes: t.bytes });
  return { s, startScale: t.startScale, note: t.note };
}
