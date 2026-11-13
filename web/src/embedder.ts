// The page's image models, through chunkmirage's ONNX Runtime Web wrapper (WebGPU when the
// browser has it, else WebAssembly). The model files are the exports of
// web/scripts/export_model.py. DINOv2-small is the default: at 112 px input it outputs (n, 1152)
// = [CLS | mean token | centre-weighted mean token], each 384-d part L2-normalized; matching picks
// the parts. At 224 px it outputs its raw tokens (n, 257, 384): the same 1152-d embedding is
// pooled from them here ("detailed" matching), and `tokens` hands out the 16 x 16 token grid for
// the tile-token mode. The other models (MorphEm, PE-Core-T) output their raw tokens at 112 px
// (n, 50, 384), pooled here the same way. A brush (which parts of the box matter) adds a fourth
// part, the brushed tokens' weighted mean: DINOv2 at 112 px from a second export that outputs its
// raw tokens (n, 65, 384), fetched only then.
import { Onnx, type OnnxModel } from "../../../chunkmirage/web/src/onnx.ts";

export const DIM = 1152;
export const PART = 384;
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];

export interface Patch { data: Float32Array; size: number; channels: number }  // y, x, c; 0-255

/** How a model takes its pixels: ImageNet's mean and std (DINOv2), 0.5 and 0.5 (PE-Core), or
 * each patch to zero mean and unit std (MorphEm); `grey`: one input channel (of colour, the mean
 * of the three). */
export type Norm = "imagenet" | "half" | "standardize";
export interface Prep { grey?: boolean; norm?: Norm }

/** A model the page can run. Test 6 (experiments/RESULTS.md) measured them on labelled HeLa EM. */
export interface ModelSpec extends Prep {
  id: string;
  name: string;        // short, for the model chip and "why"
  label: string;       // the Options > model choice
  input: number;       // px
  grid: number;        // its token grid: grid x grid tokens per patch
  file: string;        // models/<file>.onnx
  brushFile?: string;  // a tokens export for a brush, when `file` outputs the pooled embedding
  ownTokens?: boolean; // why / match maps from its own tokens (else DINOv2's 16 x 16 at 224 px)
  adapt?: boolean;     // "adapt to data" on when chosen: its CLS token is nearly the same for
                       // every patch (cosine 0.999), so raw scores all sit near 1 (test 6: no fewer found)
  mb: number;          // its download, the first time
}
export const MODELS: ModelSpec[] = [
  { id: "112", name: "DINOv2-small", label: "DINOv2-small, fast (112 px)", input: 112, grid: 8, file: "dinov2-small-112",
    brushFile: "dinov2-small-112-tokens", norm: "imagenet", mb: 44 },
  { id: "224", name: "DINOv2-small 224", label: "DINOv2-small, detailed (224 px)", input: 224, grid: 16, file: "dinov2-small-224", norm: "imagenet", mb: 44 },
  { id: "morphem", name: "MorphEm", label: "MorphEm: grey, for EM (112 px)", input: 112, grid: 7, file: "morphem-112-tokens",
    grey: true, norm: "standardize", ownTokens: true, mb: 43 },
  { id: "pecore-t", name: "PE-Core-T", label: "PE-Core-T: fastest on a CPU (112 px)", input: 112, grid: 7, file: "pecore-t16-112-tokens",
    norm: "half", ownTokens: true, adapt: true, mb: 12 },
];

/** Where each output pixel samples along one axis (bilinear, half-pixel centres): the two source
 * indices and the weight of the second, per (source size, output size). */
const axes = new Map<string, { i0: Int32Array; i1: Int32Array; w: Float32Array }>();
function axisTable(s: number, n: number) {
  const key = `${s}>${n}`;
  let t = axes.get(key);
  if (!t) {
    const k = s / n, i0 = new Int32Array(n), i1 = new Int32Array(n), w = new Float32Array(n);
    for (let i = 0; i < n; i++) { const f = Math.min(s - 1, Math.max(0, (i + 0.5) * k - 0.5)), a = Math.floor(f); i0[i] = a; i1[i] = Math.min(s - 1, a + 1); w[i] = f - a; }
    axes.set(key, (t = { i0, i1, w }));
  }
  return t;
}

/** Bilinear resize of a square patch to INPUT x INPUT, normalized as the model takes it, written
 * channel-first (3 channels, or 1 when `grey`). One channel plane at a time, the sampling from a
 * table and the normalization folded into one multiply-add (1.2-1.7x the per-pixel version). */
export function prepare(p: Patch, out: Float32Array, offset: number, INPUT = 112, how: Prep = {}) {
  const s = p.size, C = p.channels, plane = INPUT * INPUT, norm = how.norm ?? "imagenet", nc = how.grey ? 1 : 3, ax = axisTable(s, INPUT);
  // each output channel's source plane: the patch's one channel, its mean of three (grey), or its own
  const srcs: Float32Array[] = [];
  if (C === 1) srcs.push(p.data);
  else if (how.grey) { const g = new Float32Array(s * s); for (let i = 0; i < s * s; i++) g[i] = (p.data[3 * i] + p.data[3 * i + 1] + p.data[3 * i + 2]) / 3; srcs.push(g); }
  else for (let c = 0; c < 3; c++) { const g = new Float32Array(s * s); for (let i = 0; i < s * s; i++) g[i] = p.data[3 * i + c]; srcs.push(g); }
  for (let c = 0; c < nc; c++) {
    const src = srcs[Math.min(c, srcs.length - 1)], o = offset + c * plane;
    const a = norm === "imagenet" ? 1 / (255 * STD[c]) : norm === "half" ? 2 / 255 : 1 / 255;
    const b = norm === "imagenet" ? -MEAN[c] / STD[c] : norm === "half" ? -1 : 0;
    for (let y = 0; y < INPUT; y++) {
      const r0 = ax.i0[y] * s, r1 = ax.i1[y] * s, wy = ax.w[y], oy = o + y * INPUT;
      for (let x = 0; x < INPUT; x++) {
        const x0 = ax.i0[x], x1 = ax.i1[x], wx = ax.w[x];
        const top = src[r0 + x0] + wx * (src[r0 + x1] - src[r0 + x0]), bot = src[r1 + x0] + wx * (src[r1 + x1] - src[r1 + x0]);
        out[oy + x] = (top + wy * (bot - top)) * a + b;
      }
    }
  }
  if (norm === "standardize") {  // as test 6 fed MorphEm: torch's x.std (n - 1), its 1e-6 on (x - 0.5) / 0.5
    let m = 0, q = 0;
    for (let i = 0; i < plane; i++) m += out[offset + i];
    m /= plane;
    for (let i = 0; i < plane; i++) q += (out[offset + i] - m) ** 2;
    const sd = Math.sqrt(q / (plane - 1)) + 5e-7;
    for (let i = 0; i < plane; i++) out[offset + i] = (out[offset + i] - m) / sd;
  }
}

/** The brush's grid over the box: the 112 px model's 8 x 8 tokens (row-major weights, 0-1). */
export const BRUSH = 8;
/** A brush's weights on a grid of g x g tokens (each cell of the brush's grid covers (g/8)^2). */
function brushOn(brush: Float32Array, g: number): Float32Array {
  const w = new Float32Array(g * g);
  for (let y = 0; y < g; y++) for (let x = 0; x < g; x++) w[y * g + x] = brush[Math.floor((y * BRUSH) / g) * BRUSH + Math.floor((x * BRUSH) / g)];
  return w;
}
/** A short key for a brush (cache keys). */
export function brushKey(brush: Float32Array): string {
  let h = 2166136261;
  for (const v of brush) h = Math.imul(h ^ Math.round(v * 100), 16777619);
  return (h >>> 0).toString(36);
}

/** The 112 model's [CLS | mean | centre-weighted] embedding, pooled from item `j` of a raw
 * token output (T = 1 + g*g tokens of D): the same pooling as export_model.py's `Pooled`; with
 * a brush, a fourth part, the tokens weighted by it (none brushed: all alike). */
const weights = new Map<number, Float32Array>();
export function poolTokens(data: Float32Array, j: number, T: number, D: number, brush?: Float32Array | null): Float32Array {
  const g = Math.round(Math.sqrt(T - 1));
  let w = weights.get(g);
  if (!w) {
    w = new Float32Array(g * g);
    let sum = 0;
    for (let y = 0; y < g; y++) for (let x = 0; x < g; x++) {
      const yy = y - (g - 1) / 2, xx = x - (g - 1) / 2;
      sum += w[y * g + x] = Math.exp(-(yy * yy + xx * xx) / (2 * (g / 4) ** 2));
    }
    for (let i = 0; i < w.length; i++) w[i] /= sum;
    weights.set(g, w);
  }
  const np = brush ? 4 : 3, out = new Float32Array(np * D), base = j * T * D;
  let b = brush ? brushOn(brush, g) : null;
  if (b && !b.some((v) => v > 0)) b = b.fill(1);
  for (let d = 0; d < D; d++) out[d] = data[base + d];
  for (let t = 1; t < T; t++) {
    const o = base + t * D, wt = w[t - 1], wb = b ? b[t - 1] : 0;
    for (let d = 0; d < D; d++) { out[D + d] += data[o + d] / (T - 1); out[2 * D + d] += wt * data[o + d]; }
    if (wb) for (let d = 0; d < D; d++) out[3 * D + d] += wb * data[o + d];
  }
  for (let p = 0; p < np; p++) unitPart(out, p * D, D);
  return out;
}
function unitPart(v: Float32Array, o: number, n: number) {
  let s = 0;
  for (let i = o; i < o + n; i++) s += v[i] * v[i];
  s = Math.sqrt(s) || 1;
  for (let i = o; i < o + n; i++) v[i] /= s;
}

export class Embedder {
  onnx: Onnx;
  /** The model in use (Options > model): MODELS' first, DINOv2-small at 112 px, unless chosen. */
  spec: ModelSpec = MODELS[0];
  provider: string | null = null;
  batch = 32;
  seconds = 0;
  count = 0;
  private cache = new Map<string, Float32Array>();
  private files = new Map<string, OnnxModel>();
  private url: (file: string) => string;
  private get: (url: string) => Promise<ArrayBuffer>;

  /** `modelUrl(file)` gives each model file's address; bytes are fetched when first used. */
  constructor(modelUrl: (file: string) => string, log?: (s: string) => void, fetchBytes?: (url: string) => Promise<ArrayBuffer>) {
    this.onnx = new Onnx({ log });
    this.url = modelUrl;
    this.get = fetchBytes ?? ((u: string) => fetch(u).then((r) => {
      if (!r.ok) throw new Error(`model ${u}: HTTP ${r.status}`);
      return r.arrayBuffer();
    }));
  }
  private file(f: string): OnnxModel {
    let m = this.files.get(f);
    if (!m) this.files.set(f, m = { key: this.url(f), bytes: () => this.get(this.url(f)) });
    return m;
  }
  /** Model input size (px). */
  get input(): number { return this.spec.input; }
  /** Use the model with this id (MODELS); an unknown one leaves DINOv2-small at 112 px. */
  use(id: string) { this.spec = MODELS.find((m) => m.id === id) ?? MODELS[0]; }

  /** Embeddings of patches; `keys` (optional) cache them for the session. With a `brush` (which
   * of the box's 8 x 8 parts matter, 0-1), a fourth part: the brushed tokens' weighted mean.
   * `sp`: the model (a search keeps the one it began with; the page's choice is for the next). */
  async embed(patches: Patch[], keys?: string[], brush?: Float32Array | null, sp: ModelSpec = this.spec): Promise<Float32Array[]> {
    const size = sp.input, ch = sp.grey ? 1 : 3, n = ch * size * size, bk = brush ? `|b${brushKey(brush)}` : "";
    const model = this.file(brush && sp.brushFile ? sp.brushFile : sp.file);  // the others output their tokens already
    keys = keys?.map((k) => `${k}|${sp.id}${bk}`);
    const out: (Float32Array | null)[] = patches.map((_, i) => (keys ? this.cache.get(keys[i]) ?? null : null));
    const todo = out.map((v, i) => (v ? -1 : i)).filter((i) => i >= 0);
    if (!todo.length) return out as Float32Array[];
    // the batch size from where the model runs, before the first pass (a first batch of 32 on the
    // CPU held the page for seconds): building the session tells
    if (!this.provider && typeof this.onnx.session === "function") { await this.onnx.session(model); this.provider = this.onnx.provider; }
    if (this.provider === "wasm") this.batch = 8;
    const prep = (idx: number[]) => {
      const x = new Float32Array(idx.length * n);
      idx.forEach((i, j) => prepare(patches[i], x, j * n, size, sp));
      return x;
    };
    let idx = todo.slice(0, this.batch), x = prep(idx);
    for (let s = 0; s < todo.length;) {
      // give the page a turn between batches: on the CPU (WebAssembly) a pass blocks the thread
      await new Promise((r) => setTimeout(r, 0));
      const t0 = performance.now(), pass = this.onnx.run(model, x, [idx.length, ch, size, size]);
      // the next batch's pixels made while the GPU runs this one (on the CPU it changes nothing)
      const nextS = s + idx.length, nextIdx = todo.slice(nextS, nextS + this.batch), nextX = nextIdx.length ? prep(nextIdx) : null;
      const r = await pass;
      this.seconds += (r.ms ?? performance.now() - t0) / 1000;  // the pass alone, not its wait for a turn
      this.count += idx.length;
      this.provider = this.onnx.provider;
      const data = r.data as Float32Array, raw = r.dims.length === 3;  // raw tokens: pool them here
      idx.forEach((i, j) => {
        const e = raw ? poolTokens(data, j, r.dims[1], r.dims[2], brush) : Float32Array.from(data.subarray(j * DIM, (j + 1) * DIM));
        out[i] = e;
        if (keys) this.cache.set(keys[i], e);
      });
      // a bounded cache (oldest out first): a search can read without end
      for (const k of this.cache.keys()) { if (this.cache.size <= 20000) break; this.cache.delete(k); }
      s = nextS;
      if (!nextX) break;
      idx = nextIdx; x = nextX;
    }
    return out as Float32Array[];
  }
  /** Each crop's grid of unit-length patch tokens: DINOv2-small's 16 x 16 at 224 px (the
   * tile-token mode, whatever the model), or with `own` the model in use's when it has its own
   * (MorphEm's, PE-Core-T's 7 x 7 at 112 px: the why / match maps); `model`, as embed's. */
  async tokens(crops: Patch[], own = false, model: ModelSpec = this.spec): Promise<TokenGrid[]> {
    const sp = own && model.ownTokens ? model : TOKENS_MODEL;
    return this.runTokens(crops, sp, sp.file);
  }
  /** Each crop's grid of unit-length patch tokens from model `sp` itself at its input size
   * (DINOv2-small's 8 x 8 at 112 px from its tokens export, MorphEm's and PE-Core-T's 7 x 7,
   * DINOv2 224's 16 x 16): the rough pass's tokens (Options.roughTokens). */
  async gridTokens(crops: Patch[], sp: ModelSpec = this.spec): Promise<TokenGrid[]> {
    return this.runTokens(crops, sp, sp.brushFile ?? sp.file);
  }
  /** Drop the embeddings kept for the session (the bench: each run starts cold). */
  clearCache() { this.cache.clear(); }
  private async runTokens(crops: Patch[], sp: ModelSpec, file: string): Promise<TokenGrid[]> {
    const size = sp.input, ch = sp.grey ? 1 : 3, n = ch * size * size;
    const out: TokenGrid[] = [];
    for (let s = 0; s < crops.length;) {
      await new Promise((r) => setTimeout(r, 0));
      const B = this.provider === "wasm" ? 2 : 16, part = crops.slice(s, s + B);
      s += part.length;
      const x = new Float32Array(part.length * n);
      part.forEach((p, j) => prepare(p, x, j * n, size, sp));
      const t0 = performance.now();
      const r = await this.onnx.run(this.file(file), x, [part.length, ch, size, size]);
      this.seconds += (r.ms ?? performance.now() - t0) / 1000;  // the pass alone, not its wait for a turn
      this.count += part.length;
      this.provider = this.onnx.provider;
      const data = r.data as Float32Array, T = r.dims[1], D = r.dims[2], g = Math.round(Math.sqrt(T - 1));
      part.forEach((_, j) => {
        const tok = data.slice((j * T + 1) * D, (j + 1) * T * D);  // drop CLS
        for (let t = 0; t < g * g; t++) unitPart(tok, t * D, D);
        out.push({ tokens: tok, g, D });
      });
    }
    return out;
  }
}

const TOKENS_MODEL = MODELS.find((m) => m.id === "224")!;  // the model whose output is its tokens
export const TOKENS_INPUT = TOKENS_MODEL.input;
export const GRID = TOKENS_INPUT / 14;  // 16 x 16 tokens per crop

export interface TokenGrid { tokens: Float32Array; g: number; D: number }  // unit tokens, row-major g x g x D

export function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** Which parts of the embedding a match compares: the whole patch (CLS + mean of tokens), the
 * object at its centre (CLS + centre-weighted tokens), or only what the brush marked (the
 * brushed tokens' mean; embeddings made with a brush): with the CLS token as well it found no
 * more on labelled EM, alone +0.04 p@10 (test 12). */
export type Focus = "whole" | "center" | "brush";
const PARTS: Record<Focus, number[]> = { whole: [0, 1], center: [0, 2], brush: [3] };

/** Similarity of embedding `a` to query `q`: the mean over the chosen parts of their cosine,
 * after subtracting `mu` (the candidates' mean embedding: "adapt to the data") when given. */
export function similarity(a: Float32Array, q: Float32Array, focus: Focus, mu?: Float32Array): number {
  let total = 0;
  const parts = a.length === PART ? [0] : PARTS[focus];  // a window of tokens is one 384-d part
  for (const p of parts) {
    const o = p * PART;
    if (!mu) { let s = 0; for (let i = o; i < o + PART; i++) s += a[i] * q[i]; total += s; continue; }
    let s = 0, na = 0, nq = 0;
    for (let i = o; i < o + PART; i++) { const x = a[i] - mu[i], y = q[i] - mu[i]; s += x * y; na += x * x; nq += y * y; }
    total += s / (Math.sqrt(na * nq) + 1e-9);
  }
  return total / parts.length;
}

/** Back-compat for the whole-patch match without adaptation. */
export function cosine(a: Float32Array, b: Float32Array): number {
  return similarity(a, b, "whole");
}
