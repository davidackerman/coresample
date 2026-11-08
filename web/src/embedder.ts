// DINOv2-small patch embeddings in the page, through chunkmirage's ONNX Runtime Web wrapper
// (WebGPU when the browser has it, else WebAssembly). The model files are the exports of
// web/scripts/export_model.py. At 112 px input the model outputs (n, 1152) = [CLS | mean token |
// centre-weighted mean token], each 384-d part L2-normalized; matching picks the parts. At 224 px
// it outputs its raw tokens (n, 257, 384): the same 1152-d embedding is pooled from them here
// ("detailed" matching), and `tokens` hands out the 16 x 16 token grid for the tile-token mode.
import { Onnx, type OnnxModel } from "../../../chunkmirage/web/src/onnx.ts";

export const DIM = 1152;
export const PART = 384;
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];

export interface Patch { data: Float32Array; size: number; channels: number }  // y, x, c; 0-255

/** Bilinear resize of a square patch to INPUT x INPUT, normalized, written channel-first. */
export function prepare(p: Patch, out: Float32Array, offset: number, INPUT = 112) {
  const s = p.size, C = p.channels, k = s / INPUT, plane = INPUT * INPUT;
  for (let y = 0; y < INPUT; y++) {
    const fy = Math.min(s - 1, Math.max(0, (y + 0.5) * k - 0.5)), y0 = Math.floor(fy), y1 = Math.min(s - 1, y0 + 1), wy = fy - y0;
    for (let x = 0; x < INPUT; x++) {
      const fx = Math.min(s - 1, Math.max(0, (x + 0.5) * k - 0.5)), x0 = Math.floor(fx), x1 = Math.min(s - 1, x0 + 1), wx = fx - x0;
      for (let c = 0; c < 3; c++) {
        const cc = C === 3 ? c : 0;
        const v = (1 - wy) * ((1 - wx) * p.data[(y0 * s + x0) * C + cc] + wx * p.data[(y0 * s + x1) * C + cc])
                + wy * ((1 - wx) * p.data[(y1 * s + x0) * C + cc] + wx * p.data[(y1 * s + x1) * C + cc]);
        out[offset + c * plane + y * INPUT + x] = (v / 255 - MEAN[c]) / STD[c];
      }
    }
  }
}

/** The 112 model's [CLS | mean | centre-weighted] embedding, pooled from item `j` of a raw
 * token output (T = 1 + g*g tokens of D): the same pooling as export_model.py's `Pooled`. */
const weights = new Map<number, Float32Array>();
export function poolTokens(data: Float32Array, j: number, T: number, D: number): Float32Array {
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
  const out = new Float32Array(3 * D), base = j * T * D;
  for (let d = 0; d < D; d++) out[d] = data[base + d];
  for (let t = 1; t < T; t++) {
    const o = base + t * D, wt = w[t - 1];
    for (let d = 0; d < D; d++) { out[D + d] += data[o + d] / (T - 1); out[2 * D + d] += wt * data[o + d]; }
  }
  for (let p = 0; p < 3; p++) unitPart(out, p * D, D);
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
  models: Record<number, OnnxModel> = {};
  /** Model input size: 112 (fast) or 224 (4x the compute, more detail; test 5: about +30% p@10). */
  input = 112;
  provider: string | null = null;
  batch = 32;
  seconds = 0;
  count = 0;
  private cache = new Map<string, Float32Array>();

  /** `modelUrl(size)` gives each input size's model file; bytes are fetched when first used. */
  constructor(modelUrl: (size: number) => string, log?: (s: string) => void, fetchBytes?: (url: string) => Promise<ArrayBuffer>) {
    this.onnx = new Onnx({ log });
    const get = fetchBytes ?? ((u: string) => fetch(u).then((r) => {
      if (!r.ok) throw new Error(`model ${u}: HTTP ${r.status}`);
      return r.arrayBuffer();
    }));
    for (const size of [112, 224]) this.models[size] = { key: modelUrl(size), bytes: () => get(modelUrl(size)) };
  }
  get model(): OnnxModel { return this.models[this.input]; }

  /** Embeddings of patches; `keys` (optional) cache them for the session. */
  async embed(patches: Patch[], keys?: string[]): Promise<Float32Array[]> {
    const size = this.input;
    keys = keys?.map((k) => `${k}|${size}`);
    const out: (Float32Array | null)[] = patches.map((_, i) => (keys ? this.cache.get(keys[i]) ?? null : null));
    const todo = out.map((v, i) => (v ? -1 : i)).filter((i) => i >= 0);
    for (let s = 0; s < todo.length; s += this.batch) {
      // give the page a turn between batches: on the CPU (WebAssembly) a pass blocks the thread
      await new Promise((r) => setTimeout(r, 0));
      if (this.provider === "wasm") this.batch = 8;
      const idx = todo.slice(s, s + this.batch);
      const x = new Float32Array(idx.length * 3 * size * size);
      idx.forEach((i, j) => prepare(patches[i], x, j * 3 * size * size, size));
      const t0 = performance.now();
      const r = await this.onnx.run(this.models[size], x, [idx.length, 3, size, size]);
      this.seconds += (r.ms ?? performance.now() - t0) / 1000;  // the pass alone, not its wait for a turn
      this.count += idx.length;
      this.provider = this.onnx.provider;
      const data = r.data as Float32Array, raw = r.dims.length === 3;  // raw tokens (the 224 model): pool them here
      idx.forEach((i, j) => {
        const e = raw ? poolTokens(data, j, r.dims[1], r.dims[2]) : Float32Array.from(data.subarray(j * DIM, (j + 1) * DIM));
        out[i] = e;
        if (keys) this.cache.set(keys[i], e);
      });
      // a bounded cache (oldest out first): a search can read without end
      for (const k of this.cache.keys()) { if (this.cache.size <= 20000) break; this.cache.delete(k); }
    }
    return out as Float32Array[];
  }
  /** Each crop's 16 x 16 grid of unit-length patch tokens (the 224 model, whatever `input` is). */
  async tokens(crops: Patch[]): Promise<TokenGrid[]> {
    const size = TOKENS_INPUT, out: TokenGrid[] = [];
    for (let s = 0; s < crops.length;) {
      await new Promise((r) => setTimeout(r, 0));
      const B = this.provider === "wasm" ? 2 : 16, part = crops.slice(s, s + B);
      s += part.length;
      const x = new Float32Array(part.length * 3 * size * size);
      part.forEach((p, j) => prepare(p, x, j * 3 * size * size, size));
      const t0 = performance.now();
      const r = await this.onnx.run(this.models[size], x, [part.length, 3, size, size]);
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

export const TOKENS_INPUT = 224;  // the model whose output is its tokens
export const GRID = TOKENS_INPUT / 14;  // 16 x 16 tokens per crop

export interface TokenGrid { tokens: Float32Array; g: number; D: number }  // unit tokens, row-major g x g x D

export function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** Which parts of the embedding a match compares: the whole patch (CLS + mean of tokens) or the
 * object at its centre (CLS + centre-weighted tokens). */
export type Focus = "whole" | "center";
const PARTS: Record<Focus, number[]> = { whole: [0, 1], center: [0, 2] };

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
