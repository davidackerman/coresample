// The progressive search (test 3's chunk-wise strategy): scan a coarse level with cheap scores,
// rank the working level's tiles by the best window inside each, then read tiles in that order,
// embed every patch in each tile read, and keep an updated, non-overlapping hit list. It keeps
// reading, best-ranked tiles first, until every tile of the region is read or it is stopped;
// results stream after every batch. Memory stays bounded: only the best few thousand candidates
// (and any marked good or bad) are kept, so feedback (Rocchio) re-ranks among those.
import { PART, similarity, type Embedder, type Focus, type Patch } from "./embedder.ts";
import { scan as scanSync, type Dims, type ScanInput } from "./scan.ts";
import { fillMissing, informative, luminance, missingFraction, type Region, type Source, type Vec3 } from "./sources.ts";

export interface Hit { center: Vec3; score: number; index: number }
export interface State {
  phase: "query" | "scan" | "read" | "paused" | "done" | "error";
  message: string;
  working: number; coarse: number; patch: number;
  tilesRead: number; tilesBudget: number; tilesTotal: number;
  patches: number; queued: number; hits: Hit[];
  bytes: number; seconds: number; scanSeconds: number; firstHitSeconds: number | null;
}

export interface Options {
  working: number;
  patch: number;
  region?: [Vec3, Vec3];      // working-level voxels; default the whole level
  maxCoarseVoxels?: number;
  topN?: number;
  keep?: number;              // candidates kept for re-ranking (the rest are dropped as read)
  match?: MatchOptions;
  stride?: number;            // patch grid step (default P/2 in 3-D, P in 2-D)
  scanFn?: (inp: ScanInput) => Promise<{ score: Float32Array; dims: Dims }>;
}

const MARGIN = 4;
const KEEP = 3000;

/** A candidate kept for ranking: its place, embedding, pixels and current score. */
interface Item { c: Vec3; e: Float32Array; pix: Uint8Array; s: number }

/** How candidates are compared with the query (test 5 measured each on labelled EM). */
export interface MatchOptions {
  focus: Focus;        // "whole" patch or the "center" object
  rotations: boolean;  // also the query rotated/flipped 8 ways, best match counts
  scales: boolean;     // also the query at half and double the field of view
  adapt: boolean;      // subtract the candidates' mean embedding before comparing
}
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
  private q0vars: Float32Array[] = []; // the same before feedback
  private sum: Float64Array | null = null;  // running sum of candidate embeddings (for "adapt")
  private good: number[] = [];       // item ids
  private bad: number[] = [];
  private items = new Map<number, Item>();
  private nextId = 0;
  private embedded = 0;              // patches embedded so far (kept or not)
  private order: Vec3[] = [];       // tiles (working-level tile indices) in read order
  private queue: { c: Vec3; pri: number; patch: Patch }[] = [];
  skipped = 0;  // patches not embedded: no data or blank
  private scanInfo: { score: Float32Array; dims: Dims; clo: Vec3; f: number; w: Dims } | null = null;
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
    this.bytes0 = src.bytesRead;
    this.state = { phase: "query", message: "embedding the query", working: opt.working, coarse: -1, patch: opt.patch,
                   tilesRead: 0, tilesBudget: 0, tilesTotal: 0, patches: 0, queued: 0, hits: [], bytes: 0, seconds: 0,
                   scanSeconds: 0, firstHitSeconds: null };
  }

  on(f: (s: State) => void) { this.listeners.push(f); }
  private emit(patch: Partial<State>) {
    this.state = { ...this.state, ...patch, bytes: this.src.bytesRead - this.bytes0, seconds: (performance.now() - this.t0) / 1000 };
    for (const f of this.listeners) f(this.state);
  }

  /** Pixels of the patch (y, x, c as bytes) of hit or query `i` (-1 = the query). */
  patchPixels(i: number): Uint8Array { return i < 0 ? this.qpix : this.items.get(i)!.pix; }
  center(i: number): Vec3 { return this.items.get(i)!.c; }
  private qpix!: Uint8Array;

  private async readPatches(centers: Vec3[]): Promise<{ patches: Patch[]; bytes: Uint8Array[] }> {
    const P = this.opt.patch, h = P / 2, W = this.opt.working;
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
      const P = this.opt.patch, W = this.opt.working, d3 = this.src.dims === 3;
      const q = await this.readPatches([this.query]);
      if (missingFraction(q.patches[0].data) > 0.05)
        throw new Error("no data under the query (outside the image, or the server has a gap there): click somewhere else or change the zoom");
      this.qpix = q.bytes[0];
      this.q0vars = await this.queryVariants(q.patches[0]);
      this.qvars = this.q0vars.map((v) => Float32Array.from(v));
      // 1. cheap scan of the coarse level
      const C = this.chooseCoarse(), f = 2 ** (C - W);
      const s = Math.max(2, Math.round(P / f));
      const w: Dims = [d3 ? s : 1, s, s], m: Dims = [d3 ? MARGIN : 0, MARGIN, MARGIN];
      this.emit({ phase: "scan", coarse: C, message: `scanning level ${C} with cheap scores` });
      const ts = performance.now();
      const [rlo, rhi] = this.region;
      const clo: Vec3 = [d3 ? Math.floor(rlo[0] / f) : 0, Math.floor(rlo[1] / f), Math.floor(rlo[2] / f)];
      const chi: Vec3 = [d3 ? Math.ceil(rhi[0] / f) : 1, Math.ceil(rhi[1] / f), Math.ceil(rhi[2] / f)];
      const vol = await this.src.read(C, [clo[0] - m[0], clo[1] - m[1], clo[2] - m[2]], [chi[0] + m[0], chi[1] + m[1], chi[2] + m[2]]);
      const qc: Vec3 = [d3 ? Math.floor(this.query[0] / f) : 0, Math.floor(this.query[1] / f), Math.floor(this.query[2] / f)];
      const qlo: Vec3 = [qc[0] - Math.floor(w[0] / 2) - m[0], qc[1] - Math.floor(w[1] / 2) - m[1], qc[2] - Math.floor(w[2] / 2) - m[2]];
      const qr = await this.src.read(C, qlo, [qlo[0] + w[0] + 2 * m[0], qlo[1] + w[1] + 2 * m[1], qlo[2] + w[2] + 2 * m[2]]);
      const inp: ScanInput = { vol: fillMissing(luminance(vol)), dims: vol.shape, query: fillMissing(luminance(qr)), qdims: qr.shape, window: w, margin: m };
      const res = this.opt.scanFn ? await this.opt.scanFn(inp) : scanSync(inp);
      this.scanInfo = { score: res.score, dims: res.dims, clo, f, w };
      this.emit({ scanSeconds: (performance.now() - ts) / 1000 });
      // 2. rank the working level's tiles by their best window
      const ch = this.src.levels[W].chunk;
      const prior = new Map<string, number>();
      const od = res.dims;
      let i = 0;
      for (let z = 0; z < od[0]; z++)
        for (let y = 0; y < od[1]; y++)
          for (let x = 0; x < od[2]; x++, i++) {
            const c = [clo[0] + z + Math.floor(w[0] / 2), clo[1] + y + Math.floor(w[1] / 2), clo[2] + x + Math.floor(w[2] / 2)];
            const wc = [d3 ? c[0] * f + (f >> 1) : 0, c[1] * f + (f >> 1), c[2] * f + (f >> 1)];
            if (wc[1] < rlo[1] || wc[1] >= rhi[1] || wc[2] < rlo[2] || wc[2] >= rhi[2] || (d3 && (wc[0] < rlo[0] || wc[0] >= rhi[0]))) continue;
            const key = `${Math.floor(wc[0] / ch[0])},${Math.floor(wc[1] / ch[1])},${Math.floor(wc[2] / ch[2])}`;
            const v = res.score[i];
            if (!(prior.get(key)! >= v)) prior.set(key, v);
          }
      this.order = [...prior.entries()].sort((a, b) => b[1] - a[1]).map(([k]) => k.split(",").map(Number) as Vec3);
      this.emit({ phase: "read", tilesTotal: this.order.length, tilesBudget: this.order.length, message: "reading tiles, best-ranked first" });
      await this.resume();
    } catch (e) {
      this.emit({ phase: "error", message: String((e as Error).message ?? e) });
    }
  }

  /** Cheap-scan score at a working-level position (the window centred nearest it). */
  private priorAt(c: Vec3): number {
    const sc = this.scanInfo;
    if (!sc) return 0;
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
    const ch = this.src.levels[W].chunk, [rlo, rhi] = this.region;
    const lo = [t[0] * ch[0], t[1] * ch[1], t[2] * ch[2]], hi = [lo[0] + ch[0], lo[1] + ch[1], lo[2] + ch[2]];
    const grid = (a: number) => {  // stride grid from the region corner, patch inside the region
      const r: number[] = [];
      const flat = a === 0 && !d3;
      const first = rlo[a] + (flat ? 0 : h), lastOk = flat ? rhi[a] - 1 : rhi[a] - h, st = flat ? 1 : step;
      for (let v = first + Math.max(0, Math.ceil((lo[a] - first) / st)) * st; v < hi[a] && v <= lastOk; v += st) r.push(v);
      return r;
    };
    const cs: Vec3[] = [];
    for (const z of grid(0)) for (const y of grid(1)) for (const x of grid(2)) {
      if (Math.abs(z - this.query[0]) < (d3 ? h : 1) && Math.abs(y - this.query[1]) < h && Math.abs(x - this.query[2]) < h) continue;
      cs.push([z, y, x]);
    }
    if (!cs.length) return;
    const zs = cs.map((c) => c[0]), z0 = Math.min(...zs), z1 = Math.max(...zs) + 1;
    const blo: Vec3 = [z0, lo[1] - h, lo[2] - h], bhi: Vec3 = [z1, hi[1] + h, hi[2] + h];
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

  /** Read tiles until every ranked tile is read or `stop()` (also continues after a stop).
   * Tiles are read a few at a time in coarse-scan order; their patches join one queue ranked by
   * the cheap score at each patch; the best queued patches are embedded in small batches and the
   * hits update after every batch, so results stream and the likeliest come first. */
  async resume() { await this.readUntil(this.order.length); }

  /** Read tiles up to `budget` in total (tests use it to read a fixed number). */
  async readUntil(budget: number) {
    if (this.reading) return;
    this.reading = true;
    this.stopFlag = false;
    this.emit({ phase: "read", tilesBudget: budget, message: "reading tiles, best-ranked first" });
    const W = this.opt.working, P = this.opt.patch, d3 = this.src.dims === 3;
    const ahead = d3 ? 6 : 3;
    while (!this.stopFlag) {
      const B = this.emb.provider === "webgpu" ? 64 : 16;
      const limit = Math.min(budget, this.order.length);
      if (this.queue.length < 2 * B && this.next < limit) {
        const tiles = this.order.slice(this.next, Math.min(limit, this.next + ahead));
        this.next += tiles.length;
        await Promise.all(tiles.map((t) => this.readTile(t)));
        this.queue.sort((x, y) => y.pri - x.pri);
      }
      if (!this.queue.length) {
        if (this.next >= limit) break;
        continue;
      }
      const batch = this.queue.splice(0, B);
      const e = await this.emb.embed(batch.map((q) => q.patch), batch.map((q) => `${W}|${P}|${q.c.join(",")}`));
      const fresh: number[] = [];
      batch.forEach((q, k) => {
        this.sum ??= new Float64Array(e[k].length);
        for (let d = 0; d < e[k].length; d++) this.sum[d] += e[k][d];
        const id = this.nextId++;
        this.items.set(id, { c: q.c, e: e[k], pix: Uint8Array.from(q.patch.data), s: -Infinity });
        fresh.push(id);
      });
      this.embedded += batch.length;
      this.rank(fresh);
      this.prune();
      await new Promise((r) => setTimeout(r, 0));  // let the page draw
      this.emit({ tilesRead: this.next, patches: this.embedded, queued: this.queue.length,
                  firstHitSeconds: this.state.firstHitSeconds ?? (this.state.hits.length ? (performance.now() - this.t0) / 1000 : null) });
    }
    this.reading = false;
    const all = this.next >= this.order.length && !this.queue.length;
    this.emit({ phase: all ? "done" : "paused", tilesRead: this.next, queued: this.queue.length,
                message: all ? "read every tile of the region" : "stopped" });
  }
  private reading = false;

  /** Drop all but the best `keep` candidates (marked ones stay), so memory stays bounded however
   * long the search reads. */
  private prune() {
    const keep = this.opt.keep ?? KEEP;
    if (this.items.size <= keep * 1.25) return;
    const marked = new Set([...this.good, ...this.bad]);
    const ids = [...this.items.keys()].filter((i) => !marked.has(i)).sort((a, b) => this.items.get(b)!.s - this.items.get(a)!.s);
    for (const i of ids.slice(keep)) this.items.delete(i);
  }

  stop() { this.stopFlag = true; }

  /** The query's embeddings: the patch itself, plus its 7 other rotations/flips and its view at
   * half and double the field (one level finer and coarser), as the match options ask. */
  private async queryVariants(base: Patch): Promise<Float32Array[]> {
    const m = this.opt.match ?? DEFAULT_MATCH, P = base.size, C = base.channels;
    const patches: Patch[] = [base];
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
      for (const flip of [false, true]) for (let k = 0; k < 4; k++) if (k || flip) patches.push({ data: rot(base.data, k, flip), size: P, channels: C });
    }
    if (m.scales) {
      const W = this.opt.working, h = P / 2, q = this.query, d3 = this.src.dims === 3;
      for (const L of [W - 1, W + 1]) {
        if (L < 0 || L >= this.src.levels.length) continue;
        const f = 2 ** (W - L), c = [d3 ? Math.floor(q[0] * f) : 0, Math.floor(q[1] * f), Math.floor(q[2] * f)];
        const r = await this.src.read(L, [c[0], c[1] - h, c[2] - h], [c[0] + 1, c[1] + h, c[2] + h]);
        if (informative(r.data)) patches.push({ data: r.data, size: P, channels: r.channels });
      }
    }
    return this.emb.embed(patches);
  }

  /** New match options: re-rank everything embedded so far (variants re-embedded if needed). */
  async setMatch(m: MatchOptions) {
    const old = this.opt.match ?? DEFAULT_MATCH;
    this.opt.match = m;
    if (m.rotations !== old.rotations || m.scales !== old.scales) {
      const q = await this.readPatches([this.query]);
      this.q0vars = await this.queryVariants(q.patches[0]);
      this.applyFeedback();
    }
    this.rank();
    this.emit({});
  }

  /** Mark hit (patch index) good or bad and re-rank everything embedded so far. */
  mark(index: number, good: boolean) {
    (good ? this.good : this.bad).push(index);
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
      for (let o = 0; o < q.length; o += PART) {
        let n = 0;
        for (let d = o; d < o + PART; d++) n += q[d] * q[d];
        n = Math.sqrt(n) || 1;
        for (let d = o; d < o + PART; d++) q[d] /= n;
      }
      return q;
    });
  }

  /** Score the candidates `only` (all of them when not given, or when "adapt" moves the mean)
   * and rebuild the non-overlapping hit list. */
  private rank(only?: number[]) {
    const P = this.opt.patch, d3 = this.src.dims === 3, N = this.opt.topN ?? 60, m = this.opt.match ?? DEFAULT_MATCH;
    let mu: Float32Array | undefined;
    if (m.adapt && this.sum && this.embedded > 16) {
      mu = new Float32Array(this.sum.length);
      for (let d = 0; d < mu.length; d++) mu[d] = this.sum[d] / this.embedded;
    }
    for (const i of mu || !only ? this.items.keys() : only) {
      const it = this.items.get(i)!;
      let s = -Infinity;
      for (const q of this.qvars) s = Math.max(s, similarity(it.e, q, m.focus, mu));
      it.s = s;
    }
    const scored = [...this.items.entries()].sort((a, b) => b[1].s - a[1].s);
    const hits: Hit[] = [];
    const bad = new Set(this.bad);
    for (const [i, { s, c }] of scored) {
      if (bad.has(i)) continue;
      if (hits.every((h) => Math.abs(h.center[1] - c[1]) >= P || Math.abs(h.center[2] - c[2]) >= P || (d3 && Math.abs(h.center[0] - c[0]) >= P))) {
        hits.push({ center: c, score: s, index: i });
        if (hits.length >= N) break;
      }
    }
    this.state.hits = hits;
  }
}

export type { Region };
