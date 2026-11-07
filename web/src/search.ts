// The progressive search (test 3's chunk-wise strategy): scan a coarse level with cheap scores,
// rank the working level's tiles by the best window inside each, then read tiles in that order,
// embed every patch in each tile read, and keep an updated, non-overlapping hit list. Results
// stream: the caller gets a new state after every batch of tiles and can stop, continue, or mark
// hits good/bad (Rocchio feedback re-ranks everything embedded so far).
import { cosine, type Embedder, type Patch } from "./embedder.ts";
import { scan as scanSync, type Dims, type ScanInput } from "./scan.ts";
import { luminance, type Region, type Source, type Vec3 } from "./sources.ts";

export interface Hit { center: Vec3; score: number; index: number }
export interface State {
  phase: "query" | "scan" | "read" | "paused" | "done" | "error";
  message: string;
  working: number; coarse: number; patch: number;
  tilesRead: number; tilesBudget: number; tilesTotal: number;
  patches: number; hits: Hit[];
  bytes: number; seconds: number; scanSeconds: number; firstHitSeconds: number | null;
}

export interface Options {
  working: number;
  patch: number;
  region?: [Vec3, Vec3];      // working-level voxels; default the whole level
  budgetFrac?: number;        // fraction of the region's tiles read before pausing
  minTiles?: number;
  maxCoarseVoxels?: number;
  topN?: number;
  scanFn?: (inp: ScanInput) => Promise<{ score: Float32Array; dims: Dims }>;
}

const MARGIN = 4;

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
  private qvec!: Float32Array;
  private q0!: Float32Array;
  private good: number[] = [];
  private bad: number[] = [];
  private centers: Vec3[] = [];
  private embs: Float32Array[] = [];
  private pixels: Uint8Array[] = [];
  private order: Vec3[] = [];       // tiles (working-level tile indices) in read order
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
                   tilesRead: 0, tilesBudget: 0, tilesTotal: 0, patches: 0, hits: [], bytes: 0, seconds: 0,
                   scanSeconds: 0, firstHitSeconds: null };
  }

  on(f: (s: State) => void) { this.listeners.push(f); }
  private emit(patch: Partial<State>) {
    this.state = { ...this.state, ...patch, bytes: this.src.bytesRead - this.bytes0, seconds: (performance.now() - this.t0) / 1000 };
    for (const f of this.listeners) f(this.state);
  }

  /** Pixels of the patch (y, x, c as bytes) of hit or query `i` (-1 = the query). */
  patchPixels(i: number): Uint8Array { return i < 0 ? this.qpix : this.pixels[i]; }
  center(i: number): Vec3 { return this.centers[i]; }
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
      this.qpix = q.bytes[0];
      this.q0 = (await this.emb.embed(q.patches))[0];
      this.qvec = this.q0;
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
      const inp: ScanInput = { vol: luminance(vol), dims: vol.shape, query: luminance(qr), qdims: qr.shape, window: w, margin: m };
      const res = this.opt.scanFn ? await this.opt.scanFn(inp) : scanSync(inp);
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
      const total = (d3 ? Math.ceil((rhi[0] - rlo[0]) / ch[0]) : 1) * Math.ceil((rhi[1] - rlo[1]) / ch[1]) * Math.ceil((rhi[2] - rlo[2]) / ch[2]);
      const budget = Math.min(this.order.length, Math.max(this.opt.minTiles ?? 12, Math.round((this.opt.budgetFrac ?? 0.03) * total)));
      this.emit({ phase: "read", tilesTotal: total, tilesBudget: budget, message: "reading the best-ranked tiles" });
      await this.readUntil(budget);
    } catch (e) {
      this.emit({ phase: "error", message: String((e as Error).message ?? e) });
    }
  }

  /** Read more tiles, up to `budget` in total (continue after a pause). */
  async readUntil(budget: number) {
    this.stopFlag = false;
    this.emit({ phase: "read", tilesBudget: budget });
    const P = this.opt.patch, h = P / 2, step = P / 2, W = this.opt.working, d3 = this.src.dims === 3;
    const ch = this.src.levels[W].chunk, [rlo, rhi] = this.region;
    const batchTiles = d3 ? 6 : 1;
    while (this.next < Math.min(budget, this.order.length) && !this.stopFlag) {
      const tiles = this.order.slice(this.next, Math.min(budget, this.next + batchTiles));
      this.next += tiles.length;
      const centers: Vec3[] = [], patches: Patch[] = [], pix: Uint8Array[] = [];
      await Promise.all(tiles.map(async (t) => {
        const lo = [t[0] * ch[0], t[1] * ch[1], t[2] * ch[2]], hi = [lo[0] + ch[0], lo[1] + ch[1], lo[2] + ch[2]];
        const grid = (a: number) => {  // grid positions (stride P/2 from the region corner, patch inside region)
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
        // one read per tile: its z range, grown by half a patch in y and x
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
          centers.push(c);
          patches.push({ data: d, size: P, channels: C });
          pix.push(Uint8Array.from(d));
        }
      }));
      if (centers.length) {
        const keys = centers.map((c) => `${W}|${P}|${c.join(",")}`);
        const e = await this.emb.embed(patches, keys);
        for (let k = 0; k < centers.length; k++) {
          this.centers.push(centers[k]);
          this.embs.push(e[k]);
          this.pixels.push(pix[k]);
        }
      }
      this.rank();
      await new Promise((r) => setTimeout(r, 0));  // let the page draw between tiles
      this.emit({ tilesRead: this.next, patches: this.centers.length,
                  firstHitSeconds: this.state.firstHitSeconds ?? (this.state.hits.length ? (performance.now() - this.t0) / 1000 : null) });
    }
    this.emit({ phase: this.next >= this.order.length ? "done" : "paused",
                message: this.next >= this.order.length ? "read every ranked tile" : "paused: continue to read more" });
  }

  stop() { this.stopFlag = true; }

  /** Mark hit (patch index) good or bad and re-rank everything embedded so far. */
  mark(index: number, good: boolean) {
    (good ? this.good : this.bad).push(index);
    const q = Float32Array.from(this.q0);
    const add = (ids: number[], wgt: number) => {
      if (!ids.length) return;
      for (const i of ids) for (let d = 0; d < q.length; d++) q[d] += (wgt / ids.length) * this.embs[i][d];
    };
    add(this.good, 0.75);
    add(this.bad, -0.25);
    // keep each half (CLS, mean) unit length, as the embeddings are
    for (const [a, b] of [[0, q.length / 2], [q.length / 2, q.length]]) {
      let n = 0;
      for (let d = a; d < b; d++) n += q[d] * q[d];
      n = Math.sqrt(n) || 1;
      for (let d = a; d < b; d++) q[d] /= n;
    }
    this.qvec = q;
    this.rank();
    this.emit({});
  }

  private rank() {
    const P = this.opt.patch, d3 = this.src.dims === 3, N = this.opt.topN ?? 60;
    const scored = this.embs.map((e, i) => ({ i, s: cosine(e, this.qvec) }));
    scored.sort((a, b) => b.s - a.s);
    const hits: Hit[] = [];
    const bad = new Set(this.bad);
    for (const { i, s } of scored) {
      if (bad.has(i)) continue;
      const c = this.centers[i];
      if (hits.every((h) => Math.abs(h.center[1] - c[1]) >= P || Math.abs(h.center[2] - c[2]) >= P || (d3 && Math.abs(h.center[0] - c[0]) >= P))) {
        hits.push({ center: c, score: s, index: i });
        if (hits.length >= N) break;
      }
    }
    this.state.hits = hits;
  }
}

export type { Region };
