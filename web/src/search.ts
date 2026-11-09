// The progressive search (test 3's chunk-wise strategy): scan a coarse level with cheap scores,
// rank the working level's tiles by the best window inside each, then read tiles in that order,
// embed every patch in each tile read, and keep an updated, non-overlapping hit list. It keeps
// reading, best-ranked tiles first, until every tile of the region is read or it is stopped;
// results stream after every batch. Memory stays bounded: only the best few thousand candidates
// (and any marked good or bad) are kept, so feedback (Rocchio) re-ranks among those. More
// examples can join the query at any time (it becomes their mean), and every place compared
// leaves its score in a compact list for the similarity heat map.
import { GRID, PART, similarity, type Embedder, type Focus, type Patch, type TokenGrid } from "./embedder.ts";
import { scan as scanSync, type Dims, type ScanInput, type ScanNorm } from "./scan.ts";
import { fillMissing, informative, luminance, missingFraction, type Region, type Source, type Vec3 } from "./sources.ts";

export interface Hit { center: Vec3; score: number; index: number }
/** A marked hit's embedding at the rough level (by the hit's item id, so a mark can be undone). */
interface Mark { id: number; e: Float32Array }
/** How a candidate scored (Search.explain): the score is the mean of its parts' cosines with the
 * query variant that matched best. */
export interface Explain {
  score: number;
  parts: { name: "look" | "parts" | "centre" | "tokens"; value: number }[];
  variant: string;   // "id", a rotation "r1".."r7", a scale "s-1"/"s+1"
  example: number;   // with "match any": the example (0-based) that matched best, else -1
  feedback: number;  // what the good/bad marks added to the score
  adapted: boolean;  // compared after removing the candidates' mean ("adapt to data")
}
export interface State {
  phase: "query" | "scan" | "read" | "paused" | "done" | "error";
  message: string;
  working: number; coarse: number; patch: number;
  tilesRead: number; tilesBudget: number; tilesTotal: number;
  patches: number; queued: number; hits: Hit[];
  bytes: number; seconds: number; scanSeconds: number; firstHitSeconds: number | null;
  rough: { level: number; places: number } | null;  // the model's coarse pass that orders the reads
  looked: number;    // tiles the rough pass has looked at (0 without one)
  ruledOut: number;  // of those, not read in detail: below the best DETAIL share (until readRest)
  examples: number;  // examples the query is made of (shift-click adds one)
  areas: number;     // areas scanned so far (with `expand`, the first and those around it)
  levelTiles: number;  // tiles in the whole working level
  scanRead: number; scanChunks: number;  // the first quick pass: chunks of the coarse level read, of all it needs
  scanned: number;   // share of the working level the quick pass has covered (the areas scanned)
}

export interface Options {
  working: number;
  patch: number;
  region?: [Vec3, Vec3];      // working-level voxels; default the whole level
  maxCoarseVoxels?: number;
  topN?: number;
  keep?: number;              // candidates kept for re-ranking (the rest are dropped as read)
  spread?: Spread;
  roughPass?: boolean;        // the model on a coarser level picks the tiles read in detail (GPU only; default on)
  method?: Method;
  match?: MatchOptions;
  stride?: number;            // patch grid step (default P/2 in 3-D, P in 2-D)
  sketch?: Patch;             // a drawing (P x P at the working level) to search for instead of a place
  expand?: boolean;           // once the region is read, go on to the areas around it, nearest first, until the whole level is read
  scanFn?: (inp: ScanInput) => Promise<{ score: Float32Array; dims: Dims; norm?: ScanNorm }>;
}

const MARGIN = 4;
const KEEP = 3000;
/** The "place" of a drawn query: none (every comparison with it is false). */
export const NOWHERE: Vec3 = [NaN, NaN, NaN];
const placed = (c: Vec3) => c.every(Number.isFinite);

/** The model's pass over a coarser level that decides which tiles are read in detail (test 8:
 * two levels up, the same physical box, it finds the right tiles sooner than the cheap
 * statistics at the same cost). It runs in rounds, a few hundred places at a time: the level-L
 * chunks under the next tiles of the cheap scan's order, each downloaded once and every tile
 * under it scored from it (a chunk comes whole, so all its places are used); the next round's
 * chunks download while the model scores this one. Tiles are read best rough score first, and
 * only the best DETAIL share of the tiles looked at: the rest are ruled out (readRest reads
 * them too). */
interface Rough {
  L: number; p: number; stride: number; k: number;
  q0: Float32Array; q: Float32Array;             // the query at level L (and after feedback)
  goods: Mark[]; bads: Mark[];                   // marked hits seen at level L
  pool: { t: Vec3; s: number; embs: Float32Array[] }[];  // scored tiles not read yet (best first when !dirty)
  dirty: boolean;                                // tiles joined the pool or were scored again since it was sorted
  withEmbs: number;                              // pool tiles that kept their embeddings (for feedback)
  scores: number[];                              // every rough score so far (informative tiles)
  ahead: Promise<RoughBatch> | null;             // the next round, downloading
  cursor: number;                                // next tile of the cheap order to look at
  cells: number;                                 // places scored
  scored: Set<number>;                           // tiles scored already (tileKey)
  ranked: Set<number>;                           // every tile of the order (tileKey)
}
/** One round of the rough pass downloaded: its tiles and their places at level L. */
interface RoughBatch { tiles: Vec3[]; cells: { ti: number; c: Vec3 }[]; patches: (Patch | null)[] }
/** The share of the tiles looked at that is read in detail (test 8: after the model two levels
 * up has ordered them, reading 5% of the tiles reached the precision of reading all). */
export const DETAIL = 0.1;
/** Pool tiles that keep their rough embeddings, so feedback can score them again (each tile
 * holds a few, 3 KB each); tiles further down keep only their score. */
const POOL_EMBS = 20000;
/** A working-level tile as one number (each index below 2^20, z below 2^13). */
/** How long the outward scanning rests after each area, in multiples of the time it took. */
const SPREAD_REST = 4;
const tileKey = (t: Vec3) => t[0] * 2 ** 40 + t[1] * 2 ** 20 + t[2];

/** A candidate kept for ranking: its place, embedding, pixels and current score. */
interface Item { c: Vec3; e: Float32Array; pix: Uint8Array; s: number; h: number }

/** One example of what to find: its place, its embeddings (keyed by variant: "id", the
 * rotations "r1".."r7", the scales "s-1", "s+1"), its pixels and its view at the rough level. */
interface Example { c: Vec3; vars: Map<string, Float32Array>; pix: Uint8Array; rough: Float32Array | null }

/** Every place compared, with its score (for the heat map): centres (z, y, x, working level)
 * and scores. Scores are set when a place is compared; the kept candidates' scores follow
 * re-ranking (feedback, more examples), the rest keep the score they had when dropped. */
export interface Heat { c: Int32Array; s: Float32Array; n: number; version: number }
const HEAT_MAX = 6_000_000;  // places remembered (16 bytes each)

/** The mean of unit vectors, each part (PART dims) made unit length again. */
function unitMean(vs: Float32Array[]): Float32Array {
  const q = new Float32Array(vs[0].length);
  for (const v of vs) for (let d = 0; d < q.length; d++) q[d] += v[d];
  return unitParts(q);
}
function unitParts(q: Float32Array): Float32Array {
  for (let o = 0; o < q.length; o += PART) {
    let n = 0;
    for (let d = o; d < Math.min(q.length, o + PART); d++) n += q[d] * q[d];
    n = Math.sqrt(n) || 1;
    for (let d = o; d < Math.min(q.length, o + PART); d++) q[d] /= n;
  }
  return q;
}

/** How candidates are compared with the query (test 5 measured each on labelled EM). */
export interface MatchOptions {
  focus: Focus;        // "whole" patch or the "center" object
  rotations: boolean;  // also the query rotated/flipped 8 ways, best match counts
  scales: boolean;     // also the query at half and double the field of view
  adapt: boolean;      // subtract the candidates' mean embedding before comparing
  combine?: "mean" | "any";  // several examples: their mean (default), or the best match among them
}
/** How far apart hits must be, in boxes (patch sides): from the query, and from each other
 * (1 = no overlap). */
export interface Spread { fromQuery: number; between: number }
export const DEFAULT_SPREAD: Spread = { fromQuery: 0, between: 1 };

/** How candidates are embedded. "patches": every patch of the stride grid through the model on
 * its own (test 5's baseline). "tokens": the Mars CTX explorer's way (test 7): each tile is cut
 * into crops twice the patch size, each crop goes through the model once, and every window of
 * the patch's size, at every token (1/8 of the patch), is the mean of its tokens. */
export type Method = "patches" | "tokens";
const WIN = GRID / 2;     // a patch spans 8 x 8 tokens of a crop twice its size
const PER_PLANE = 8;      // windows kept per tile plane (best, non-overlapping)

// the measured baseline: no variant beat it consistently across the three runs of test 5
export const DEFAULT_MATCH: MatchOptions = { focus: "whole", rotations: false, scales: false, adapt: false };

/** The finest level at least 2 levels above `W` (1 if need be) that can be scanned whole over
 * `region` within `budget` voxels while the query window stays >= 4 (>= 2) voxels; -1 if none. */
export function coarseLevelFor(src: Source, W: number, P: number, region: [Vec3, Vec3], budget: number): number {
  const n = src.levels.length, d3 = src.dims === 3, [lo, hi] = region;
  const vox = (L: number) => {
    const f = 2 ** (L - W);
    return (d3 ? Math.ceil((hi[0] - lo[0]) / f) : 1) * Math.ceil((hi[1] - lo[1]) / f) * Math.ceil((hi[2] - lo[2]) / f);
  };
  for (let k = 2; W + k < n; k++) if (P / 2 ** k >= 4 && vox(W + k) <= budget) return W + k;
  for (let k = 1; W + k < n; k++) if (P / 2 ** k >= 2 && vox(W + k) <= budget) return W + k;
  return -1;
}

export class Search {
  state: State;
  query: Vec3;
  private qvars: Float32Array[] = [];  // the query's embeddings (variants), feedback applied
  private q0vars: Float32Array[] = []; // the same before feedback: the mean of the examples'
  private examples: Example[] = [];
  private qlabels: { ex: number; key: string }[] = [];  // which example and variant each query vector is
  private sum: Float64Array | null = null;  // running sum of candidate embeddings (for "adapt")
  private good: number[] = [];       // item ids
  private bad: number[] = [];
  private items = new Map<number, Item>();
  private nextId = 0;
  private embedded = 0;              // patches embedded so far (kept or not)
  private order: Vec3[] = [];       // tiles (working-level tile indices) in read order
  private orderS: number[] = [];    // their cheap scores (the part not taken yet stays sorted by them)
  private queue: { c: Vec3; pri: number; patch: Patch }[] = [];
  skipped = 0;  // patches not embedded: no data or blank
  private scans: { score: Float32Array | null; dims: Dims; clo: Vec3; f: number; w: Dims; lo: Vec3; hi: Vec3 }[] = [];
  private tileS = new Map<number, number>();  // each ranked tile's cheap score (its best window)
  private norm: ScanNorm | null = null;        // the first area's score scale, for the areas after it
  private qscan: { C: number; f: number; w: Dims; m: Dims; query: Float32Array; qdims: Dims } | null = null;
  private allTiles = new Set<number>();  // every tile ranked so far (tileKey)
  private area: [Vec3, Vec3];            // the areas scanned so far (their bounding box)
  private limits: [Vec3, Vec3];          // where patches may be: the region, or with `expand` the whole level
  private ring = 0;
  private ringQ: [Vec3, Vec3][] = [];
  private next = 0;
  private stopFlag = false;
  private t0 = performance.now();
  private bytes0: number;
  private region: [Vec3, Vec3];
  private listeners: ((s: State) => void)[] = [];

  src: Source;
  emb: Embedder;
  opt: Options;
  constructor(src: Source, emb: Embedder, query: Vec3, opt: Options) {
    this.src = src;
    this.emb = emb;
    this.opt = opt;
    this.query = query;
    const L = src.levels[opt.working];
    this.region = opt.region ?? [[0, 0, 0], [...L.shape] as Vec3];
    this.area = [[...this.region[0]] as Vec3, [...this.region[1]] as Vec3];
    this.limits = opt.expand ? [[0, 0, 0], [...L.shape] as Vec3] : this.region;
    this.bytes0 = src.bytesRead;
    this.state = { phase: "query", message: "embedding the query", working: opt.working, coarse: -1, patch: opt.patch,
                   tilesRead: 0, tilesBudget: 0, tilesTotal: 0, patches: 0, queued: 0, hits: [], bytes: 0, seconds: 0,
                   scanSeconds: 0, firstHitSeconds: null, rough: null, looked: 0, ruledOut: 0, examples: 1, areas: 1,
                   levelTiles: L.shape.reduce((n, v, a) => n * Math.ceil(v / L.chunk[a]), 1), scanRead: 0, scanChunks: 0, scanned: 0 };
    this.ready = new Promise((r) => (this.setReady = r));
  }
  private ready: Promise<void>;
  private setReady!: () => void;

  on(f: (s: State) => void) { this.listeners.push(f); }
  /** The searched region (working-level voxels) and the tiles read so far (tile indices). */
  get regionBox(): [Vec3, Vec3] { return this.area; }
  readTiles(): Vec3[] { return this.readList; }
  /** The examples' places and pixels (the first is the query clicked). */
  exampleCenters(): Vec3[] { return this.examples.map((e) => e.c); }
  examplePixels(): Uint8Array[] { return this.examples.map((e) => e.pix); }
  /** Every place compared so far and its score (see Heat). */
  heat(): Heat { return { c: this.hc, s: this.hs, n: this.hn, version: this.hv }; }
  private hc = new Int32Array(3 * 4096);
  private hs = new Float32Array(4096);
  private hn = 0;
  private hv = 0;
  private heatAdd(c: Vec3, s: number): number {
    if (this.hn >= HEAT_MAX) return -1;
    if (this.hn === this.hs.length) {
      const n = Math.min(HEAT_MAX, 2 * this.hs.length), hc = new Int32Array(3 * n), hs = new Float32Array(n);
      hc.set(this.hc); hs.set(this.hs); this.hc = hc; this.hs = hs;
    }
    this.hc.set(c, 3 * this.hn);
    this.hs[this.hn] = s;
    return this.hn++;
  }
  /** Whether a patch centred at `c` overlaps an example (those are not hits). */
  private atExample(c: Vec3): boolean {
    const h = this.opt.patch / 2, dz = this.src.dims === 3 ? h : 1;
    return this.examples.some((e) => Math.abs(c[0] - e.c[0]) < dz && Math.abs(c[1] - e.c[1]) < h && Math.abs(c[2] - e.c[2]) < h);
  }
  /** The example nearest `c` (null if every example is a drawing). */
  nearestExample(c: Vec3): Vec3 | null {
    const d3 = this.src.dims === 3, dist = (e: Vec3) => Math.hypot(d3 ? c[0] - e[0] : 0, c[1] - e[1], c[2] - e[2]);
    const at = this.examples.map((e) => e.c).filter(placed);
    return at.length ? at.reduce((b, e) => (dist(e) < dist(b) ? e : b)) : null;
  }
  private readList: Vec3[] = [];
  private rough: Rough | null = null;
  private emit(patch: Partial<State>) {
    this.state = { ...this.state, ...patch, bytes: this.src.bytesRead - this.bytes0, seconds: (performance.now() - this.t0) / 1000 };
    for (const f of this.listeners) f(this.state);
  }

  /** Pixels of the patch (y, x, c as bytes) of hit or query `i` (-1 = the query). */
  patchPixels(i: number): Uint8Array { return i < 0 ? this.examples[0].pix : this.items.get(i)!.pix; }
  center(i: number): Vec3 { return this.items.get(i)!.c; }

  private get tokenMode() { return this.opt.method === "tokens"; }

  /** What the query embeds from: its patch, or (token mode) the crop twice its size around it,
   * whose central window of tokens is the query. */
  private async queryBase(c: Vec3): Promise<Patch> {
    if (!placed(c)) return this.sketchBase();
    if (!this.tokenMode) return (await this.readPatches([c])).patches[0];
    return (await this.readPatches([c], 2 * this.opt.patch)).patches[0];
  }

  /** The drawing as the model sees it: itself, or (token mode) in the middle of a crop twice its
   * size filled with its median. */
  private sketchBase(): Patch {
    const sk = this.opt.sketch!, P = sk.size, C = sk.channels;
    if (!this.tokenMode) return sk;
    const S = 2 * P, med = median(sk.data), d = new Float32Array(S * S * C).fill(med);
    for (let y = 0; y < P; y++) d.set(sk.data.subarray(y * P * C, (y + 1) * P * C), ((y + P / 2) * S + P / 2) * C);
    return { data: d, size: S, channels: C };
  }

  private async readPatches(centers: Vec3[], side = this.opt.patch): Promise<{ patches: Patch[]; bytes: Uint8Array[] }> {
    const P = side, h = P / 2, W = this.opt.working;
    const patches: Patch[] = [], bytes: Uint8Array[] = [];
    await Promise.all(centers.map(async (c, i) => {
      const r = await this.src.read(W, [c[0], c[1] - h, c[2] - h], [c[0] + 1, c[1] + h, c[2] + h]);
      patches[i] = { data: r.data, size: P, channels: r.channels };
      bytes[i] = Uint8Array.from(r.data);
    }));
    return { patches, bytes };
  }

  private chooseCoarse(): number {
    const C = coarseLevelFor(this.src, this.opt.working, this.opt.patch, this.region, this.opt.maxCoarseVoxels ?? 8e6);
    if (C < 0) throw new Error("no coarse level is small enough to scan whole and still shows the query; zoom in or search the view");
    return C;
  }

  async start() {
    try {
      const P = this.opt.patch, W = this.opt.working, d3 = this.src.dims === 3, sk = this.opt.sketch;
      let pix: Uint8Array;
      if (sk) pix = Uint8Array.from(sk.data);
      else {
        const q = await this.readPatches([this.query]);
        if (missingFraction(q.patches[0].data) > 0.05)
          throw new Error("no data under the query (outside the image, or the server has a gap there): click somewhere else or change the zoom");
        pix = q.bytes[0];
      }
      this.examples = [{ c: sk ? NOWHERE : this.query, vars: await this.queryVariants(await this.queryBase(sk ? NOWHERE : this.query), this.query), pix, rough: null }];
      this.combine();
      this.qvars = this.q0vars.map((v) => Float32Array.from(v));
      // 1. the cheap scan's query: its window at the coarse level
      const C = this.chooseCoarse(), f = 2 ** (C - W);
      const sw = Math.max(2, Math.round(P / f));
      const w: Dims = [d3 ? sw : 1, sw, sw], m: Dims = [d3 ? MARGIN : 0, MARGIN, MARGIN];
      this.emit({ phase: "scan", coarse: C, message: `the quick pass: level ${C}, ${2 ** (C - W)}x coarser, cheap scores` });
      let qr: Region;
      if (sk) qr = sketchWindow(sk, w, m);  // the drawing shrunk to the coarse level, its median around it
      else {
        const qc: Vec3 = [d3 ? Math.floor(this.query[0] / f) : 0, Math.floor(this.query[1] / f), Math.floor(this.query[2] / f)];
        const qlo: Vec3 = [qc[0] - Math.floor(w[0] / 2) - m[0], qc[1] - Math.floor(w[1] / 2) - m[1], qc[2] - Math.floor(w[2] / 2) - m[2]];
        qr = await this.src.read(C, qlo, [qlo[0] + w[0] + 2 * m[0], qlo[1] + w[1] + 2 * m[1], qlo[2] + w[2] + 2 * m[2]]);
      }
      this.qscan = { C, f, w, m, query: fillMissing(luminance(qr)), qdims: qr.shape };
      // 2. the region scanned, its tiles ranked
      // time, bytes and chunks keep counting on a slow server
      const ts = performance.now(), n0 = this.src.chunksRead, need = this.chunksIn(C, this.region);
      if (n0 !== undefined) this.emit({ scanRead: 0, scanChunks: need });
      const tick = setInterval(() => this.emit(n0 === undefined ? {} : { scanRead: Math.min(need, this.src.chunksRead! - n0) }), 500);
      try { ({ tiles: this.order, scores: this.orderS } = await this.scanRegion(this.region)); } finally { clearInterval(tick); }
      this.emit({ scanSeconds: (performance.now() - ts) / 1000, scanRead: need, scanChunks: need });
      await this.setupRough();
      this.setReady();
      this.emit({ phase: "read", tilesTotal: this.order.length, tilesBudget: this.order.length, message: "reading tiles, best-ranked first" });
      await this.resume();
    } catch (e) {
      this.emit({ phase: "error", message: String((e as Error).message ?? e) });
      this.setReady();
    }
  }

  /** Chunks of coarse level C (with the scan's margin) under an area of the working level. */
  private chunksIn(C: number, [rlo, rhi]: [Vec3, Vec3]): number {
    const { f, m } = this.qscan!, L = this.src.levels[C], d3 = this.src.dims === 3;
    return [0, 1, 2].reduce((n, a) => {
      if (a === 0 && !d3) return n;
      const lo = Math.max(0, Math.floor(rlo[a] / f) - m[a]), hi = Math.min(L.shape[a], Math.ceil(rhi[a] / f) + m[a]);
      return n * Math.max(1, Math.floor((hi - 1) / L.chunk[a]) - Math.floor(lo / L.chunk[a]) + 1);
    }, 1);
  }

  /** The cheap scan of one area (working-level voxels [lo, hi)) at the coarse level, and the
   * working level's tiles in it not ranked before, ranked by their best window. */
  private async scanRegion([rlo, rhi]: [Vec3, Vec3]): Promise<{ tiles: Vec3[]; scores: number[] }> {
    const q = this.qscan!, { C, f, w, m } = q, d3 = this.src.dims === 3, W = this.opt.working;
    const clo: Vec3 = [d3 ? Math.floor(rlo[0] / f) : 0, Math.floor(rlo[1] / f), Math.floor(rlo[2] / f)];
    const chi: Vec3 = [d3 ? Math.ceil(rhi[0] / f) : 1, Math.ceil(rhi[1] / f), Math.ceil(rhi[2] / f)];
    const vol = await this.src.read(C, [clo[0] - m[0], clo[1] - m[1], clo[2] - m[2]], [chi[0] + m[0], chi[1] + m[1], chi[2] + m[2]]);
    if (this.state.phase === "scan") this.emit({ scanRead: this.state.scanChunks, message: `the quick pass: level ${C} read, comparing every place in it with the query` });
    const inp: ScanInput = { vol: fillMissing(luminance(vol)), dims: vol.shape, query: Float32Array.from(q.query), qdims: q.qdims, window: w, margin: m,
                             norm: this.norm ?? undefined };
    const res = this.opt.scanFn ? await this.opt.scanFn(inp) : scanSync(inp);
    this.norm ??= res.norm ?? null;
    // the first area keeps every window's score (each patch's priority); the areas after it, so
    // memory stays small however far the search spreads, only each tile's best (priorAt)
    this.scans.push({ score: this.scans.length ? null : res.score, dims: res.dims, clo, f, w, lo: rlo, hi: rhi });
    const Ls = this.src.levels[W].shape, cov = (rhi[0] - rlo[0]) * (rhi[1] - rlo[1]) * (rhi[2] - rlo[2]) / (Ls[0] * Ls[1] * Ls[2]);
    this.emit({ scanned: Math.min(1, this.state.scanned + cov) });  // areas do not overlap
    const ch = this.src.levels[W].chunk;
    const prior = new Map<number, number>();
    const od = res.dims;
    let i = 0;
    for (let z = 0; z < od[0]; z++)
      for (let y = 0; y < od[1]; y++)
        for (let x = 0; x < od[2]; x++, i++) {
          const c = [clo[0] + z + Math.floor(w[0] / 2), clo[1] + y + Math.floor(w[1] / 2), clo[2] + x + Math.floor(w[2] / 2)];
          const wc = [d3 ? c[0] * f + (f >> 1) : 0, c[1] * f + (f >> 1), c[2] * f + (f >> 1)];
          if (wc[1] < rlo[1] || wc[1] >= rhi[1] || wc[2] < rlo[2] || wc[2] >= rhi[2] || (d3 && (wc[0] < rlo[0] || wc[0] >= rhi[0]))) continue;
          const key = tileKey([Math.floor(wc[0] / ch[0]), Math.floor(wc[1] / ch[1]), Math.floor(wc[2] / ch[2])]);
          if (this.allTiles.has(key)) continue;  // ranked with an earlier area
          const v = res.score[i];
          if (!(prior.get(key)! >= v)) prior.set(key, v);
        }
    const best = [...prior.entries()].sort((a, b) => b[1] - a[1]);
    const tiles = best.map(([k, v]) => {
      this.allTiles.add(k);
      this.tileS.set(k, v);
      return [Math.floor(k / 2 ** 40), Math.floor(k / 2 ** 20) % 2 ** 20, k % 2 ** 20] as Vec3;
    });
    return { tiles, scores: best.map(([, v]) => v) };
  }

  /** With `expand`: the next area around the first (rings of areas its size, each ring nearest
   * first), or null when the whole level has been covered. */
  private nextArea(): [Vec3, Vec3] | null {
    const shape = this.src.levels[this.opt.working].shape, [rlo, rhi] = this.region;
    const B = [0, 1, 2].map((a) => rhi[a] - rlo[a]);
    // rings far enough to cover the level along every axis the region does not span
    const maxRing = Math.max(...[0, 1, 2].map((a) => Math.ceil(Math.max(rlo[a], shape[a] - rhi[a]) / Math.max(1, B[a]))));
    while (!this.ringQ.length) {
      if (++this.ring > maxRing) return null;
      const R = this.ring, span = [0, 1, 2].map((a) => (B[a] >= shape[a] ? 0 : R)), out: [Vec3, Vec3][] = [];
      for (let dz = -span[0]; dz <= span[0]; dz++) for (let dy = -span[1]; dy <= span[1]; dy++) for (let dx = -span[2]; dx <= span[2]; dx++) {
        if (Math.max(Math.abs(dz), Math.abs(dy), Math.abs(dx)) !== R) continue;
        const d = [dz, dy, dx];
        const lo = [0, 1, 2].map((a) => Math.max(0, rlo[a] + d[a] * B[a])) as Vec3;
        const hi = [0, 1, 2].map((a) => Math.min(shape[a], rhi[a] + d[a] * B[a])) as Vec3;
        if (lo.some((v, a) => hi[a] <= v)) continue;
        out.push([lo, hi]);
      }
      const q = this.query.every(Number.isFinite) ? this.query : [(rlo[0] + rhi[0]) / 2, (rlo[1] + rhi[1]) / 2, (rlo[2] + rhi[2]) / 2];
      const dist = ([lo, hi]: [Vec3, Vec3]) => Math.hypot(...[0, 1, 2].map((a) => (lo[a] + hi[a]) / 2 - q[a]));
      this.ringQ = out.sort((a, b) => dist(a) - dist(b));
    }
    return this.ringQ.shift()!;
  }

  /** Scan the next area and merge its tiles into the order; false when no area is left. One
   * scan at a time (the reader and the spreading share it). */
  private scanOne(): Promise<boolean> {
    return (this.scanBusy ??= (async () => {
      const b = this.nextArea();
      if (!b) return false;
      const { tiles, scores } = await this.scanRegion(b);
      this.area = [[0, 1, 2].map((a) => Math.min(this.area[0][a], b[0][a])) as Vec3, [0, 1, 2].map((a) => Math.max(this.area[1][a], b[1][a])) as Vec3];
      if (tiles.length) { this.mergeOrder(tiles, scores); this.dry = false; }
      this.emit({ areas: this.state.areas + 1, tilesTotal: this.order.length });
      return true;
    })().finally(() => (this.scanBusy = null)));
  }
  private scanBusy: Promise<boolean> | null = null;

  /** A new area's tiles into the part of the order not taken yet, by cheap score: every area is
   * scored on the first one's scale, so the reads go to the best places anywhere scanned so far. */
  private mergeOrder(tiles: Vec3[], scores: number[]) {
    const h = this.untakenFrom(), ot = this.order.splice(h), os = this.orderS.splice(h);
    let i = 0, j = 0;
    while (i < ot.length || j < tiles.length)
      if (j >= tiles.length || (i < ot.length && os[i] >= scores[j])) { this.order.push(ot[i]); this.orderS.push(os[i++]); }
      else { this.order.push(tiles[j]); this.orderS.push(scores[j++]); }
  }
  /** Where the order's untaken part starts: tiles not read yet (with the rough pass: not looked at yet). */
  private untakenFrom(): number { return this.rough ? this.rough.cursor : this.next; }

  /** With `expand`, when the reader has run out: scan areas until one adds tiles. False when the
   * whole level has been covered. */
  private async expandOnce(): Promise<boolean> {
    if (!this.opt.expand) return false;
    while (!this.stopFlag) {
      if (this.untakenFrom() < this.order.length) { this.dry = false; return true; }  // the spreading added some
      if (!(await this.scanOne())) return this.untakenFrom() < this.order.length;
    }
    return false;
  }

  /** With `expand`: keep scanning outward while the reads go on, so they can go to the best
   * places anywhere scanned instead of finishing the area around the click first. One area at a
   * time, resting SPREAD_REST times as long as each took: the scans compete with the reads for
   * the CPU and the network (hemibrain at 8 nm, 90 s: resting as long cost 25% of the tiles read). */
  private async spread() {
    if (!this.opt.expand || this.spreading) return;
    this.spreading = true;
    try {
      while (!this.stopFlag) {
        const t = performance.now();
        if (!(await this.scanOne())) break;
        await new Promise((r) => setTimeout(r, Math.max(500, SPREAD_REST * (performance.now() - t))));
      }
    } catch { /* an area that fails: the reader's own expansion carries on */ }
    finally { this.spreading = false; }
  }
  private spreading = false;

  /** Cheap-scan score at a working-level position (the window centred nearest it, in the scan
   * of the area it is in). */
  private priorAt(c: Vec3): number {
    let sc = this.scans[0];
    for (let k = this.scans.length - 1; k >= 0; k--) {
      const b = this.scans[k];
      if ([0, 1, 2].every((a) => c[a] >= b.lo[a] && c[a] < b.hi[a])) { sc = b; break; }
    }
    if (!sc) return 0;
    if (!sc.score) {  // an area after the first: its tile's best
      const ch = this.src.levels[this.opt.working].chunk;
      return this.tileS.get(tileKey([Math.floor(c[0] / ch[0]), Math.floor(c[1] / ch[1]), Math.floor(c[2] / ch[2])])) ?? 0;
    }
    const idx = [0, 1, 2].map((a) => {
      const v = Math.floor(c[a] / sc.f) - sc.clo[a] - Math.floor(sc.w[a] / 2);
      return Math.max(0, Math.min(sc.dims[a] - 1, v));
    });
    return sc.score[(idx[0] * sc.dims[1] + idx[1]) * sc.dims[2] + idx[2]];
  }

  /** Read a tile and queue its patches (the stride grid inside it), each with its prior. */
  private async readTile(t: Vec3) {
    const P = this.opt.patch, h = P / 2, W = this.opt.working, d3 = this.src.dims === 3;
    const step = this.opt.stride ?? (d3 ? P / 2 : P);
    // the grid starts at 0, on the chunk grid: in 2-D (stride P) every patch then lies inside one
    // tile, so a tile read downloads that tile alone; in 3-D (stride P/2) half the patches along
    // an axis reach into the chunk before, so a tile read takes the chunks before it (4, not 9)
    const ch = this.src.levels[W].chunk, o = [0, 0, 0], [llo, lhi] = this.limits;
    const lo = [t[0] * ch[0], t[1] * ch[1], t[2] * ch[2]], hi = [lo[0] + ch[0], lo[1] + ch[1], lo[2] + ch[2]];
    const grid = (a: number) => {  // stride grid anchored at the region's corner, patch inside the limits
      const r: number[] = [];
      const flat = a === 0 && !d3, st = flat ? 1 : step;
      const first = o[a] + (flat ? 0 : h), from = Math.max(lo[a], flat ? llo[a] : llo[a] + h), lastOk = flat ? lhi[a] - 1 : lhi[a] - h;
      for (let v = first + Math.ceil((from - first) / st) * st; v < hi[a] && v <= lastOk; v += st) r.push(v);
      return r;
    };
    const cs: Vec3[] = [];
    for (const z of grid(0)) for (const y of grid(1)) for (const x of grid(2)) {
      const c: Vec3 = [z, y, x];
      if (!this.atExample(c)) cs.push(c);
    }
    if (!cs.length) return;
    const ext = (a: number) => cs.reduce(([m, M], c) => [Math.min(m, c[a]), Math.max(M, c[a])], [Infinity, -Infinity]);
    const [z0, zl] = ext(0), [y0, y1] = ext(1), [x0, x1] = ext(2), z1 = zl + 1;
    const blo: Vec3 = [z0, y0 - h, x0 - h], bhi: Vec3 = [z1, y1 + h, x1 + h];  // just the patches
    let r: Region;
    try { r = await this.src.read(W, blo, bhi); } catch { return; }  // a tile that fails is skipped
    const [, by, bx] = r.shape, C = r.channels;
    for (const c of cs) {
      const d = new Float32Array(P * P * C);
      for (let y = 0; y < P; y++) {
        const row = ((c[0] - z0) * by + (c[1] - h + y - blo[1])) * bx + (c[2] - h - blo[2]);
        d.set(r.data.subarray(row * C, (row + P) * C), y * P * C);
      }
      if (!informative(d)) { this.skipped++; continue; }  // no data or blank: nothing to match
      this.queue.push({ c, pri: this.priorAt(c), patch: { data: d, size: P, channels: C } });
    }
  }

  /** Read tiles until every ranked tile is read (with the rough pass: every tile it picks) or
   * `stop()` (also continues after a stop). Tile reads run in the background, several at a
   * time, so downloads overlap the model; their patches join one queue ranked by the cheap
   * score at each patch; the best queued patches are embedded in small batches and the hits
   * update after every batch, so results stream and the likeliest come first. */
  async resume() { await this.readUntil(Infinity); }

  /** Also read the tiles the rough pass ruled out (best first), then carry on. */
  async readRest() {
    if (!this.rough || this.readAll) return;
    this.readAll = true;
    this.dry = false;
    await this.resume();
  }
  private readAll = false;

  /** Read tiles up to `budget` in total (tests use it to read a fixed number). */
  async readUntil(budget: number) {
    if (this.reading) return;
    this.reading = true;
    this.stopFlag = false;
    this.dry = false;
    this.emit({ phase: "read", tilesBudget: Math.min(budget, this.order.length), message: "reading tiles, best-ranked first" });
    void this.spread();
    const W = this.opt.working, P = this.opt.patch, gpu = this.emb.provider === "webgpu";
    const limit = () => Math.min(budget, this.order.length);
    while (!this.stopFlag) {
      if (this.tokenMode) {  // a few tiles at a time, every crop of them through the model in one go
        const tiles = this.next < limit() ? await this.takeTiles(gpu ? 4 : 1, limit()) : [];
        if (!tiles.length) { if (this.next < budget && (await this.expandOnce())) continue; break; }
        const preps = (await Promise.all(tiles.map((t) => this.prepTokens(t)))).flat();
        const grids = await this.emb.tokens(preps.flatMap((p) => p.crops));
        const fresh: number[] = [];
        let k = 0;
        for (const p of preps) { fresh.push(...this.scoreWindows(p, grids.slice(k, k + p.crops.length))); k += p.crops.length; }
        this.rank(fresh);
        this.prune();
        await new Promise((r) => setTimeout(r, 0));
        this.emit({ tilesRead: this.next, patches: this.embedded, queued: 0, ...this.roughCounts(),
                    firstHitSeconds: this.state.firstHitSeconds ?? (this.state.hits.length ? (performance.now() - this.t0) / 1000 : null) });
        continue;
      }
      const B = gpu ? 64 : 16;
      this.feed(limit, B);
      const coming = [...this.inflight, ...(this.taking ? [this.taking] : [])];
      if (this.queue.length < B && coming.length) { await Promise.race(coming); continue; }  // wait for a full batch
      if (!this.queue.length) {  // nothing queued and nothing coming: the next area, or the end
        if (this.next < budget && (await this.expandOnce())) continue;
        break;
      }
      if (this.queueDirty) { this.queue.sort((x, y) => y.pri - x.pri); this.queueDirty = false; }
      const batch = this.queue.splice(0, B);
      const e = await this.emb.embed(batch.map((q) => q.patch), batch.map((q) => `${W}|${P}|${q.c.join(",")}`));
      const fresh: number[] = [];
      batch.forEach((q, k) => {
        this.sum ??= new Float64Array(e[k].length);
        for (let d = 0; d < e[k].length; d++) this.sum[d] += e[k][d];
        const id = this.nextId++;
        this.items.set(id, { c: q.c, e: e[k], pix: Uint8Array.from(q.patch.data), s: -Infinity, h: this.heatAdd(q.c, -Infinity) });
        fresh.push(id);
      });
      this.embedded += batch.length;
      this.rank(fresh);
      this.prune();
      await new Promise((r) => setTimeout(r, 0));  // let the page draw
      this.emit({ tilesRead: this.next, patches: this.embedded, queued: this.queue.length, ...this.roughCounts(),
                  firstHitSeconds: this.state.firstHitSeconds ?? (this.state.hits.length ? (performance.now() - this.t0) / 1000 : null) });
    }
    this.reading = false;
    const all = !this.stopFlag && (this.next < budget || this.next >= this.order.length);
    const ruled = this.roughCounts().ruledOut;
    this.emit({ phase: all ? "done" : "paused", tilesRead: this.next, queued: this.queue.length, ...this.roughCounts(),
                message: !all ? "stopped"
                  : ruled ? `read the tiles the model picked; it ruled out ${ruled}`
                  : this.opt.expand ? "read every tile of the level" : "read every tile of the region" });
  }
  private reading = false;
  private inflight = new Set<Promise<void>>();  // tile reads under way
  private taking: Promise<void> | null = null;  // picking the next tiles (may run a rough round)
  private dry = false;                          // the last pick found nothing more (until the order grows)
  private queueDirty = false;                   // patches joined the queue since it was sorted

  /** Keep a few tile reads going in the background (their patches join the queue as they land),
   * so the downloads overlap the model instead of taking turns with it. */
  private feed(limit: () => number, B: number) {
    const ahead = 32;
    const room = () => !this.stopFlag && !this.dry && this.inflight.size < ahead && this.queue.length < 4 * B;
    if (this.taking || !room()) return;
    this.taking = (async () => {
      while (room()) {
        const tiles = this.next < limit() ? await this.takeTiles(Math.min(4, ahead - this.inflight.size), limit()) : [];
        if (!tiles.length) { this.dry = true; break; }
        for (const t of tiles) {
          const p: Promise<void> = this.readTile(t).finally(() => { this.inflight.delete(p); this.queueDirty = true; });
          this.inflight.add(p);
        }
      }
    })().finally(() => (this.taking = null));
  }

  /** The rough pass's counts for the panel. */
  private roughCounts(): { looked: number; ruledOut: number } {
    const r = this.rough;
    return r ? { looked: r.scored.size, ruledOut: this.readAll ? 0 : r.pool.length } : { looked: 0, ruledOut: 0 };
  }

  /** The next `n` tiles to read: in the cheap scan's order, or, with the rough pass, the best
   * rough scores first, up to the DETAIL share of the tiles it has looked at (more rounds
   * scored as needed; none when this area is used up). */
  private async takeTiles(n: number, limit: number): Promise<Vec3[]> {
    n = Math.min(n, limit - this.next);
    let out: Vec3[];
    const r = this.rough;
    if (!r) out = this.order.slice(this.next, this.next + n);
    else {
      const allowed = () => (this.readAll ? Infinity : Math.max(8, Math.ceil(DETAIL * r.scores.length)));
      const more = () => r.cursor < this.order.length || !!r.ahead;
      while (!this.stopFlag && (r.pool.length < n || this.next + n > allowed()) && (this.roughBusy || more())) await this.roughOnce();
      if (r.dirty) { r.pool.sort((a, b) => b.s - a.s); r.dirty = false; }
      out = r.pool.splice(0, Math.max(0, Math.min(n, allowed() - this.next))).map((t) => t.t);
      // the next round under way while these are read, so the reads never wait for it
      if (!this.stopFlag && more() && allowed() - this.next - out.length < 48) void this.roughOnce();
    }
    this.next += out.length;
    this.readList.push(...out);
    return out;
  }

  /** Decide whether there is a rough pass and set it up: the model two levels coarser (one, for
   * small boxes), the same physical box (at least 16 px there), on the GPU only. */
  private async setupRough() {
    // on by default with a GPU; `roughPass: true` forces it (tests, slow CPUs), false turns it off
    if (this.opt.roughPass === false || (this.opt.roughPass !== true && this.emb.provider !== "webgpu")) return;
    const P = this.opt.patch, W = this.opt.working, d3 = this.src.dims === 3;
    const k = P / 4 >= 16 && W + 2 < this.src.levels.length ? 2 : P / 2 >= 16 && W + 1 < this.src.levels.length ? 1 : 0;
    if (!k) return;
    const L = W + k, p = P / 2 ** k, f = 2 ** k;
    const q = this.query, c: Vec3 = [d3 ? Math.floor(q[0] / f) : 0, Math.floor(q[1] / f), Math.floor(q[2] / f)];
    const base = this.opt.sketch ? shrink(this.opt.sketch, p) : (await this.readAt(L, [c], p))[0];
    if (!base || !informative(base.data)) return;
    const q0 = (await this.emb.embed([base]))[0];
    this.examples[0].rough = q0;
    // 2-D grids are not overlapping (stride P at the working level): sample every other place
    this.rough = { L, p, stride: d3 ? p : 2 * p, k, q0, q: Float32Array.from(q0), goods: [], bads: [], pool: [], dirty: false, withEmbs: 0,
                   scores: [], ahead: null, cursor: 0, cells: 0, scored: new Set(), ranked: this.allTiles };
  }

  /** p x p patches at level L around `centers` (z, y, x of that level). */
  private async readAt(L: number, centers: Vec3[], p: number): Promise<(Patch | null)[]> {
    const h = p / 2;
    return Promise.all(centers.map(async (c) => {
      try {
        const r = await this.src.read(L, [c[0], c[1] - h, c[2] - h], [c[0] + 1, c[1] + h, c[2] + h]);
        return { data: r.data, size: p, channels: r.channels };
      } catch { return null; }
    }));
  }

  /** A rough round: the one under way, or a new one (one at a time). */
  private roughOnce(): Promise<void> {
    return (this.roughBusy ??= this.roughRound().finally(() => (this.roughBusy = null)));
  }
  private roughBusy: Promise<void> | null = null;

  /** One round of the rough pass: the level-L chunks under the next tiles of the cheap order,
   * every place in them through the model, each tile under them scored by its best place. The
   * next round's chunks start downloading before this one goes through the model. */
  private async roughRound() {
    const r = this.rough!;
    const cur = await (r.ahead ?? this.roughFetch());
    r.ahead = r.cursor < this.order.length ? this.roughFetch() : null;
    const { tiles, cells, patches } = cur;
    const keep = cells.map((_, i) => i).filter((i) => patches[i] && informative(patches[i]!.data));
    const embs = keep.length ? await this.emb.embed(keep.map((i) => patches[i]!)) : [];
    const per = tiles.map(() => ({ s: -Infinity, embs: [] as Float32Array[] }));
    keep.forEach((i, j) => {
      const t = per[cells[i].ti];
      t.embs.push(embs[j]);
      t.s = Math.max(t.s, similarity(embs[j], r.q, "whole"));
    });
    tiles.forEach((t, i) => {
      // a tile with nothing informative at level L comes last (ruled out unless readRest)
      r.pool.push({ t, s: per[i].s, embs: per[i].embs });
      if (per[i].embs.length) r.withEmbs++;
      if (per[i].s > -Infinity) r.scores.push(per[i].s);
    });
    r.dirty = true;
    if (r.withEmbs > 1.25 * POOL_EMBS) {  // memory: only the best tiles keep their embeddings
      r.pool.sort((a, b) => b.s - a.s);
      r.dirty = false;
      r.withEmbs = 0;
      for (const t of r.pool) if (t.embs.length && ++r.withEmbs > POOL_EMBS) { t.embs = []; r.withEmbs--; }
    }
    r.cells += keep.length;
    this.emit({ rough: { level: r.L, places: r.cells }, ...this.roughCounts() });
  }

  /** The next round's tiles (from the cheap order) and their places at level L, downloaded. */
  private async roughFetch(): Promise<RoughBatch> {
    const r = this.rough!, W = this.opt.working, d3 = this.src.dims === 3, ch = this.src.levels[W].chunk, f = 2 ** r.k;
    const chL = this.src.levels[r.L].chunk;
    const [rlo, rhi] = this.limits, h = r.p / 2;
    const perTile = (d3 ? Math.max(1, ch[0] / f / r.stride) : 1) * Math.max(1, ch[1] / f / r.stride) * Math.max(1, ch[2] / f / r.stride);
    const budget = this.emb.provider === "webgpu" ? 768 : 48;  // places per round
    const tiles: Vec3[] = [], seen = new Set<string>();
    while (r.cursor < this.order.length && tiles.length * perTile < budget) {
      const t = this.order[r.cursor++];
      if (r.scored.has(tileKey(t))) continue;
      // the level-L chunk holding this tile's corner, and every ranked tile under that chunk
      const cl = [0, 1, 2].map((a) => Math.floor((t[a] * ch[a]) / f / chL[a]));
      const ck = cl.join(",");
      if (seen.has(ck)) continue;
      seen.add(ck);
      const lo = [0, 1, 2].map((a) => Math.floor((cl[a] * chL[a] * f) / ch[a]));
      const hi = [0, 1, 2].map((a) => Math.max(lo[a] + 1, Math.ceil(((cl[a] + 1) * chL[a] * f) / ch[a])));
      for (let z = lo[0]; z < hi[0]; z++) for (let y = lo[1]; y < hi[1]; y++) for (let x = lo[2]; x < hi[2]; x++) {
        const u: Vec3 = [z, y, x], k = tileKey(u);
        if (r.ranked.has(k) && !r.scored.has(k)) { r.scored.add(k); tiles.push(u); }
      }
      if (!r.scored.has(tileKey(t))) { r.scored.add(tileKey(t)); tiles.push(t); }  // (chunk sizes that do not line up)
    }
    const cells: { ti: number; c: Vec3 }[] = [];
    // places at (k + 1/2) strides, so each p-box lies inside one level-L chunk (one download)
    const off = r.stride / 2;
    const axis = (a: number, t: Vec3) => {  // places of this tile along axis a, at level L
      const lo = Math.max(t[a] * ch[a], rlo[a]) / f, hi = Math.min((t[a] + 1) * ch[a], rhi[a]) / f, out: number[] = [];
      if (a === 0 && !d3) return [0];
      for (let v = Math.ceil((lo + h - off) / r.stride) * r.stride + off; v + h <= hi; v += r.stride) out.push(v);
      if (!out.length && hi - lo >= r.p) out.push(Math.floor((lo + hi) / 2));
      return out;
    };
    tiles.forEach((t, ti) => { for (const z of axis(0, t)) for (const y of axis(1, t)) for (const x of axis(2, t)) cells.push({ ti, c: [z, y, x] }); });
    return { tiles, cells, patches: await this.readAt(r.L, cells.map((c) => c.c), r.p) };
  }

  /** Feedback reaches the rough pass too: the marked hit seen at level L moves the rough query,
   * and the tiles not read yet are scored again, so the reads follow the marks. */
  private async roughFeedback(id: number, good: boolean) {
    const r = this.rough, it = this.items.get(id);
    if (!r || !it) return;
    const f = 2 ** r.k, c: Vec3 = [this.src.dims === 3 ? Math.floor(it.c[0] / f) : 0, Math.floor(it.c[1] / f), Math.floor(it.c[2] / f)];
    const p = (await this.readAt(r.L, [c], r.p))[0];
    if (!p || !informative(p.data)) return;
    const e = (await this.emb.embed([p]))[0];
    if (!(good ? this.good : this.bad).includes(id)) return;  // undone meanwhile
    (good ? r.goods : r.bads).push({ id, e });
    this.roughRequery();
  }

  /** The rough query from the examples and the marks (as applyFeedback), and the tiles not
   * read yet scored again with it. */
  private roughRequery() {
    const r = this.rough!;
    const q = Float32Array.from(r.q0);
    const add = (vs: Mark[], w: number) => { for (const { e } of vs) for (let d = 0; d < q.length; d++) q[d] += (w / vs.length) * e[d]; };
    add(r.goods, 0.75);
    add(r.bads, -0.25);
    r.q = unitParts(q);
    for (const t of r.pool) if (t.embs.length) t.s = t.embs.reduce((m, e) => Math.max(m, similarity(e, r.q, "whole")), -Infinity);
    r.dirty = true;
  }

  /** Another example of what to find. The query becomes the mean of the examples' embeddings
   * (each part unit length again), as Earth Engine's similarity search and similar.earth
   * combine several points; the rough pass's query too. The coarse scan that ranked the tiles
   * first stays the first example's. Resolves to an error message, or null when added. */
  async addExample(c: Vec3): Promise<string | null> {
    await this.ready;
    if (this.state.phase === "error") return this.state.message;
    if (this.atExample(c)) return "that place is already an example";
    const q = await this.readPatches([c]);
    if (missingFraction(q.patches[0].data) > 0.05) return "no data there";
    const ex: Example = { c, vars: await this.queryVariants(await this.queryBase(c), c), pix: q.bytes[0], rough: null };
    this.examples.push(ex);
    this.combine();
    this.applyFeedback();
    const r = this.rough;
    if (r) {
      const f = 2 ** r.k, rc: Vec3 = [this.src.dims === 3 ? Math.floor(c[0] / f) : 0, Math.floor(c[1] / f), Math.floor(c[2] / f)];
      const p = (await this.readAt(r.L, [rc], r.p))[0];
      if (p && informative(p.data)) ex.rough = (await this.emb.embed([p]))[0];
      r.q0 = unitMean(this.examples.flatMap((e) => (e.rough ? [e.rough] : [])));
      this.roughRequery();
    }
    this.rank();
    this.emit({ examples: this.examples.length });
    return null;
  }

  /** The query's variants as the mean of the examples' (a variant an example lacks, such as a
   * scale off the pyramid, is the mean of those that have it), or with "any" every example's
   * variants side by side (a candidate scores its best match among them). Test 9: the mean
   * gains with more examples on vesicles, "any" did better on mito. */
  private combine() {
    if ((this.opt.match ?? DEFAULT_MATCH).combine === "any") {
      this.q0vars = this.examples.flatMap((e) => [...e.vars.values()]);
      this.qlabels = this.examples.flatMap((e, ex) => [...e.vars.keys()].map((key) => ({ ex, key })));
      return;
    }
    const keys: string[] = [];
    for (const e of this.examples) for (const k of e.vars.keys()) if (!keys.includes(k)) keys.push(k);
    this.q0vars = keys.map((k) => unitMean(this.examples.flatMap((e) => (e.vars.has(k) ? [e.vars.get(k)!] : []))));
    this.qlabels = keys.map((key) => ({ ex: -1, key }));
  }

  /** Drop all but the best `keep` candidates (marked ones stay), so memory stays bounded however
   * long the search reads. */
  private prune() {
    const keep = this.opt.keep ?? KEEP;
    if (this.items.size <= keep * 1.25) return;
    const marked = new Set([...this.good, ...this.bad]);
    const ids = [...this.items.keys()].filter((i) => !marked.has(i)).sort((a, b) => this.items.get(b)!.s - this.items.get(a)!.s);
    for (const i of ids.slice(keep)) this.items.delete(i);
  }

  /** Token mode: the crops (twice the patch, side by side) covering one tile's window centres,
   * per plane of the stride grid in z. */
  private async prepTokens(t: Vec3): Promise<TokPlane[]> {
    const P = this.opt.patch, h = P / 2, S = 2 * P, W = this.opt.working, d3 = this.src.dims === 3;
    const ch = this.src.levels[W].chunk, [rlo, rhi] = this.limits;
    const lo = [t[0] * ch[0], t[1] * ch[1], t[2] * ch[2]], hi = [lo[0] + ch[0], lo[1] + ch[1], lo[2] + ch[2]];
    // window centres this tile owns: inside the tile, at least half a patch inside the region
    const ylo = Math.max(lo[1], rlo[1] + h), yhi = Math.min(hi[1], rhi[1] - h + 1);
    const xlo = Math.max(lo[2], rlo[2] + h), xhi = Math.min(hi[2], rhi[2] - h + 1);
    if (yhi <= ylo || xhi <= xlo) return [];
    const zs: number[] = [];
    if (!d3) zs.push(0);
    else { const st = P / 2, first = rlo[0] + h; for (let z = first + Math.max(0, Math.ceil((lo[0] - first) / st)) * st; z < hi[0] && z <= rhi[0] - h; z += st) zs.push(z); }
    if (!zs.length) return [];
    const ny = Math.ceil((yhi - ylo + P) / S), nx = Math.ceil((xhi - xlo + P) / S);
    const oy = ylo - h, ox = xlo - h;
    let r: Region;
    try { r = await this.src.read(W, [zs[0], oy, ox], [zs[zs.length - 1] + 1, oy + ny * S, ox + nx * S]); } catch { return []; }
    const [, by, bx] = r.shape, C = r.channels, out: TokPlane[] = [];
    for (const z of zs) {
      const crops: Patch[] = [], ok: boolean[] = [];
      for (let cy = 0; cy < ny; cy++) for (let cx = 0; cx < nx; cx++) {
        const d = new Float32Array(S * S * C);
        for (let y = 0; y < S; y++) {
          const row = ((z - zs[0]) * by + cy * S + y) * bx + cx * S;
          d.set(r.data.subarray(row * C, (row + S) * C), y * S * C);
        }
        const good = missingFraction(d) < 0.5;
        ok.push(good);
        crops.push({ data: good ? fillMissing(d) : d.fill(0), size: S, channels: C });
      }
      out.push({ z, oy, ox, ny, nx, ylo, yhi, xlo, xhi, crops, ok, region: r, z0: zs[0] });
    }
    return out;
  }

  /** Token mode: every window of WIN x WIN tokens in a plane's stitched grid, as the unit mean
   * of its tokens, scored against the query; the best few (not overlapping) are kept. */
  private scoreWindows(p: TokPlane, grids: TokenGrid[]): number[] {
    const P = this.opt.patch, h = P / 2, tp = (2 * P) / GRID, D = grids[0]?.D ?? PART, d3 = this.src.dims === 3;
    const Gy = p.ny * GRID, Gx = p.nx * GRID, m = this.opt.match ?? DEFAULT_MATCH;
    // integral image of the stitched unit tokens: (Gy + 1) x (Gx + 1) x D
    const I = new Float32Array((Gy + 1) * (Gx + 1) * D);
    const bad = new Int32Array((Gy + 1) * (Gx + 1));  // integral count of tokens from missing crops
    for (let y = 0; y < Gy; y++) {
      const cy = Math.floor(y / GRID);
      for (let x = 0; x < Gx; x++) {
        const cx = Math.floor(x / GRID), k = cy * p.nx + cx, g = grids[k];
        const src = g.tokens, so = ((y % GRID) * GRID + (x % GRID)) * D;
        const o = ((y + 1) * (Gx + 1) + (x + 1)) * D, up = (y * (Gx + 1) + (x + 1)) * D, left = ((y + 1) * (Gx + 1) + x) * D, diag = (y * (Gx + 1) + x) * D;
        for (let d = 0; d < D; d++) I[o + d] = src[so + d] + I[up + d] + I[left + d] - I[diag + d];
        const bo = (y + 1) * (Gx + 1) + (x + 1);
        bad[bo] = (p.ok[k] ? 0 : 1) + bad[y * (Gx + 1) + x + 1] + bad[(y + 1) * (Gx + 1) + x] - bad[y * (Gx + 1) + x];
      }
    }
    const cand: { r: number; c: number; s: number; e: Float32Array }[] = [];
    const v = new Float32Array(D);
    for (let r = 0; r + WIN <= Gy; r++) for (let c = 0; c + WIN <= Gx; c++) {
      const cyN = p.oy + (r + WIN / 2) * tp, cxN = p.ox + (c + WIN / 2) * tp;
      if (cyN < p.ylo || cyN >= p.yhi || cxN < p.xlo || cxN >= p.xhi) continue;
      const a = r * (Gx + 1) + c, b = r * (Gx + 1) + c + WIN, cc = (r + WIN) * (Gx + 1) + c, dd = (r + WIN) * (Gx + 1) + c + WIN;
      if (bad[dd] - bad[b] - bad[cc] + bad[a] > 0) continue;  // touches a crop with no data
      if (this.atExample([p.z, cyN, cxN])) continue;  // an example itself
      let n = 0;
      for (let d = 0; d < D; d++) { const x = I[dd * D + d] - I[b * D + d] - I[cc * D + d] + I[a * D + d]; v[d] = x; n += x * x; }
      n = Math.sqrt(n) || 1;
      const e = new Float32Array(D);
      for (let d = 0; d < D; d++) e[d] = v[d] / n;
      this.sum ??= new Float64Array(D);
      for (let d = 0; d < D; d++) this.sum[d] += e[d];
      this.embedded++;
      let s = -Infinity;
      for (const qv of this.qvars) s = Math.max(s, similarity(e, qv, m.focus));
      cand.push({ r, c, s, e });
      if (r % (WIN / 2) === 0 && c % (WIN / 2) === 0) this.heatAdd([p.z, Math.round(cyN), Math.round(cxN)], s);  // the heat map at half a box
    }
    // the best windows of this plane, not overlapping, become candidates
    cand.sort((x, y) => y.s - x.s);
    const kept: typeof cand = [], fresh: number[] = [];
    for (const w of cand) {
      if (kept.every((k) => Math.abs(k.r - w.r) >= WIN || Math.abs(k.c - w.c) >= WIN)) kept.push(w);
      if (kept.length >= PER_PLANE) break;
    }
    const [, by, bx] = p.region.shape, C = p.region.channels;
    for (const w of kept) {
      const cy = Math.round(p.oy + (w.r + WIN / 2) * tp), cx = Math.round(p.ox + (w.c + WIN / 2) * tp);
      const pix = new Uint8Array(P * P * C);
      for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) {
        const src = (((p.z - p.z0) * by + (cy - h + y - p.oy)) * bx + (cx - h + x - p.ox)) * C;
        for (let k = 0; k < C; k++) { const val = p.region.data[src + k]; pix[(y * P + x) * C + k] = val === val ? val : 0; }
      }
      const id = this.nextId++;
      this.items.set(id, { c: [p.z, cy, cx], e: w.e, pix, s: w.s, h: this.heatAdd([p.z, cy, cx], w.s) });
      fresh.push(id);
    }
    return fresh;
  }

  stop() { this.stopFlag = true; }

  /** The query's embeddings: the patch itself, plus its 7 other rotations/flips and its view at
   * half and double the field (one level finer and coarser), as the match options ask. */
  private async queryVariants(base: Patch, at: Vec3): Promise<Map<string, Float32Array>> {
    const m = this.opt.match ?? DEFAULT_MATCH, P = base.size, C = base.channels;
    const patches: Patch[] = [base], keys = ["id"];
    if (m.rotations) {
      const rot = (d: Float32Array, k: number, flip: boolean) => {  // k quarter turns, optional mirror
        const o = new Float32Array(d.length);
        for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) {
          let sx = flip ? P - 1 - x : x, sy = y;
          for (let t = 0; t < k; t++) [sx, sy] = [sy, P - 1 - sx];
          for (let c = 0; c < C; c++) o[(y * P + x) * C + c] = d[(sy * P + sx) * C + c];
        }
        return o;
      };
      for (const flip of [false, true]) for (let k = 0; k < 4; k++) if (k || flip) { patches.push({ data: rot(base.data, k, flip), size: P, channels: C }); keys.push(`r${patches.length - 1}`); }
    }
    if (m.scales && placed(at)) {  // a drawing has no other levels
      const W = this.opt.working, h = P / 2, q = at, d3 = this.src.dims === 3;  // P: the base's side
      for (const L of [W - 1, W + 1]) {
        if (L < 0 || L >= this.src.levels.length) continue;
        const f = 2 ** (W - L), c = [d3 ? Math.floor(q[0] * f) : 0, Math.floor(q[1] * f), Math.floor(q[2] * f)];
        const r = await this.src.read(L, [c[0], c[1] - h, c[2] - h], [c[0] + 1, c[1] + h, c[2] + h]);
        if (informative(r.data)) { patches.push({ data: r.data, size: P, channels: r.channels }); keys.push(L < W ? "s-1" : "s+1"); }
      }
    }
    const keyed = (vs: Float32Array[]) => new Map(vs.map((v, i) => [keys[i], v]));
    if (!this.tokenMode) return keyed(await this.emb.embed(patches));
    // token mode: each variant's crop through the model, its central window of tokens averaged
    return keyed((await this.emb.tokens(patches)).map(({ tokens, g, D }) => {
      const v = new Float32Array(D), o = (g - WIN) / 2;
      for (let y = o; y < o + WIN; y++) for (let x = o; x < o + WIN; x++)
        for (let d = 0; d < D; d++) v[d] += tokens[(y * g + x) * D + d];
      let n = 0;
      for (let d = 0; d < D; d++) n += v[d] * v[d];
      n = Math.sqrt(n) || 1;
      for (let d = 0; d < D; d++) v[d] /= n;
      return v;
    }));
  }

  /** New match options: re-rank everything embedded so far (variants re-embedded if needed). */
  async setMatch(m: MatchOptions) {
    const old = this.opt.match ?? DEFAULT_MATCH;
    this.opt.match = m;
    if (m.rotations !== old.rotations || m.scales !== old.scales) {
      for (const e of this.examples) e.vars = await this.queryVariants(await this.queryBase(e.c), e.c);
      this.combine();
      this.applyFeedback();
    } else if (m.combine !== old.combine) {
      this.combine();
      this.applyFeedback();
    }
    this.rank();
    this.emit({});
  }

  /** New spacing rules: re-rank at once (nothing is re-embedded). */
  setSpread(sp: Spread) {
    this.opt.spread = sp;
    this.rank();
    this.emit({});
  }

  /** Mark hit (patch index) good or bad and re-rank everything embedded so far. */
  /** Mark a hit good (the query moves toward it) or bad (hidden, the query moves away from it);
   * everything kept is scored again, and the tiles not read yet too. A mark replaces any other
   * on the same hit. */
  mark(index: number, good: boolean) {
    this.dropMark(index);
    (good ? this.good : this.bad).push(index);
    this.markOrder.push(index);
    this.roughFeedback(index, good);
    this.afterMarks();
  }
  /** A hit's mark, if any. */
  markOf(index: number): "good" | "bad" | null { return this.good.includes(index) ? "good" : this.bad.includes(index) ? "bad" : null; }
  /** How many hits are marked good and bad. */
  marks(): { good: number; bad: number } { return { good: this.good.length, bad: this.bad.length }; }
  /** Undo a hit's mark (the last mark when no hit is given). */
  unmark(index?: number) {
    const i = index ?? this.markOrder[this.markOrder.length - 1];
    if (i === undefined || !this.markOf(i)) return;
    this.dropMark(i);
    this.afterMarks();
  }
  private markOrder: number[] = [];
  /** Undo every mark. */
  clearMarks() { for (const i of [...this.markOrder]) this.dropMark(i); this.afterMarks(); }
  private dropMark(i: number) {
    const out = (a: number[]) => { const k = a.indexOf(i); if (k >= 0) a.splice(k, 1); };
    out(this.good); out(this.bad); out(this.markOrder);
    if (this.rough) {
      this.rough.goods = this.rough.goods.filter((m) => m.id !== i);
      this.rough.bads = this.rough.bads.filter((m) => m.id !== i);
    }
  }
  private afterMarks() {
    if (this.rough) this.roughRequery();
    this.applyFeedback();
    this.rank();
    this.emit({});
  }

  /** Rocchio on every query variant: toward the good hits, away from the bad ones, each part
   * kept unit length as the embeddings are. */
  private applyFeedback() {
    this.qvars = this.q0vars.map((q0) => {
      const q = Float32Array.from(q0);
      const add = (ids: number[], wgt: number) => {
        if (!ids.length) return;
        for (const i of ids) { const e = this.items.get(i)!.e; for (let d = 0; d < q.length; d++) q[d] += (wgt / ids.length) * e[d]; }
      };
      add(this.good, 0.75);
      add(this.bad, -0.25);
      return unitParts(q);
    });
  }

  /** The candidates' mean embedding, when "adapt to data" subtracts it. */
  private adaptMean(): Float32Array | undefined {
    if (!(this.opt.match ?? DEFAULT_MATCH).adapt || !this.sum || this.embedded <= 16) return undefined;
    const mu = new Float32Array(this.sum.length);
    for (let d = 0; d < mu.length; d++) mu[d] = this.sum[d] / this.embedded;
    return mu;
  }

  /** How candidate `i` scored: each compared part's cosine with the query variant that matched
   * best ("look": the CLS token, the patch as a whole; "parts": the mean of its patch tokens,
   * what is in it; "centre": the centre-weighted tokens; "tokens": a window's mean token), which
   * variant and example that was, and what feedback added. */
  explain(i: number): Explain | null {
    const it = this.items.get(i);
    if (!it || !this.qvars.length) return null;
    const m = this.opt.match ?? DEFAULT_MATCH, mu = this.adaptMean();
    let best = 0, bs = -Infinity;
    this.qvars.forEach((q, j) => { const v = similarity(it.e, q, m.focus, mu); if (v > bs) { bs = v; best = j; } });
    const q = this.qvars[best], tok = it.e.length === PART;
    const ps: [Explain["parts"][number]["name"], number][] = tok ? [["tokens", 0]] : m.focus === "center" ? [["look", 0], ["centre", 2]] : [["look", 0], ["parts", 1]];
    const part = (p: number) => { const o = p * PART, sl = (a: Float32Array) => a.subarray(o, o + PART); return similarity(sl(it.e), sl(q), "whole", mu && sl(mu)); };
    let s0 = -Infinity;
    for (const q0 of this.q0vars) s0 = Math.max(s0, similarity(it.e, q0, m.focus, mu));
    return { score: bs, parts: ps.map(([name, p]) => ({ name, value: part(p) })), variant: this.qlabels[best]?.key ?? "id",
             example: this.qlabels[best]?.ex ?? -1, feedback: bs - s0, adapted: !!mu };
  }

  /** Score the candidates `only` (all of them when not given, or when "adapt" moves the mean)
   * and rebuild the non-overlapping hit list. */
  private rank(only?: number[]) {
    const P = this.opt.patch, d3 = this.src.dims === 3, N = this.opt.topN ?? 60, m = this.opt.match ?? DEFAULT_MATCH;
    const mu = this.adaptMean();
    for (const i of mu || !only ? this.items.keys() : only) {
      const it = this.items.get(i)!;
      let s = -Infinity;
      for (const q of this.qvars) s = Math.max(s, similarity(it.e, q, m.focus, mu));
      it.s = s;
      if (it.h >= 0) this.hs[it.h] = s;
    }
    this.hv++;
    const scored = [...this.items.entries()].sort((a, b) => b[1].s - a[1].s);
    const hits: Hit[] = [];
    const bad = new Set(this.bad);
    const sp = this.opt.spread ?? DEFAULT_SPREAD, gap = sp.between * P, far = sp.fromQuery * P;
    for (const [i, { s, c }] of scored) {
      if (bad.has(i) || (this.examples.length > 1 && this.atExample(c))) continue;
      // at least `fromQuery` boxes from every example (straight-line, in voxels of the search level)
      if (far > 0 && this.examples.some((e) => Math.hypot(d3 ? c[0] - e.c[0] : 0, c[1] - e.c[1], c[2] - e.c[2]) < far)) continue;
      if (hits.every((h) => Math.abs(h.center[1] - c[1]) >= gap || Math.abs(h.center[2] - c[2]) >= gap || (d3 && Math.abs(h.center[0] - c[0]) >= gap))) {
        hits.push({ center: c, score: s, index: i });
        if (hits.length >= N) break;
      }
    }
    this.state.hits = hits;
  }
}

/** The median of the defined values of `a`. */
function median(a: Float32Array): number {
  const v = Array.from(a).filter((x) => x === x).sort((x, y) => x - y);
  return v.length ? v[v.length >> 1] : 0;
}

/** A patch shrunk to `p` x `p` (area average). */
function shrink(pt: Patch, p: number): Patch {
  const P = pt.size, C = pt.channels, k = P / p, d = new Float32Array(p * p * C);
  for (let y = 0; y < p; y++) for (let x = 0; x < p; x++) for (let c = 0; c < C; c++) {
    let s = 0, n = 0;
    for (let yy = Math.floor(y * k); yy < Math.ceil((y + 1) * k); yy++)
      for (let xx = Math.floor(x * k); xx < Math.ceil((x + 1) * k); xx++) { s += pt.data[(yy * P + xx) * C + c]; n++; }
    d[(y * p + x) * C + c] = s / n;
  }
  return { data: d, size: p, channels: C };
}

/** A drawing as the cheap scan's query window: shrunk to the window, repeated through its
 * depth, with `margin` of its median around it. */
function sketchWindow(sk: Patch, w: Dims, m: Dims): Region {
  const lum = luminance({ data: sk.data, shape: [1, sk.size, sk.size], channels: sk.channels });
  const s = shrink({ data: lum, size: sk.size, channels: 1 }, w[1]), med = median(lum);
  const shape: Vec3 = [w[0] + 2 * m[0], w[1] + 2 * m[1], w[2] + 2 * m[2]], d = new Float32Array(shape[0] * shape[1] * shape[2]).fill(med);
  for (let z = m[0]; z < m[0] + w[0]; z++) for (let y = 0; y < w[1]; y++)
    d.set(s.data.subarray(y * w[2], (y + 1) * w[2]), (z * shape[1] + y + m[1]) * shape[2] + m[2]);
  return { data: d, shape, channels: 1 };
}

/** Token mode: one plane of a tile, its crops and where its windows may be centred. */
interface TokPlane {
  z: number; z0: number; oy: number; ox: number; ny: number; nx: number;
  ylo: number; yhi: number; xlo: number; xhi: number;
  crops: Patch[]; ok: boolean[]; region: Region;
}

export type { Region };
