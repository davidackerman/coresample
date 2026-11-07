// DINOv2-small patch embeddings in the page, through chunkmirage's ONNX Runtime Web wrapper
// (WebGPU when the browser has it, else WebAssembly). The model file is the export of
// web/scripts/export_model.py: input (n, 3, 112, 112) ImageNet-normalized, output (n, 768) =
// [CLS, mean token], each L2-normalized, the same embedding the Python tests used.
import { Onnx, type OnnxModel } from "../../../chunkmirage/web/src/onnx.ts";

export const INPUT = 112;
export const DIM = 768;
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];

export interface Patch { data: Float32Array; size: number; channels: number }  // y, x, c; 0-255

/** Bilinear resize of a square patch to INPUT x INPUT, normalized, written channel-first. */
export function prepare(p: Patch, out: Float32Array, offset: number) {
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

export class Embedder {
  onnx: Onnx;
  model: OnnxModel;
  provider: string | null = null;
  batch = 32;
  seconds = 0;
  count = 0;
  private cache = new Map<string, Float32Array>();

  constructor(modelUrl: string, log?: (s: string) => void) {
    this.onnx = new Onnx({ log });
    this.model = { key: modelUrl, bytes: () => fetch(modelUrl).then((r) => {
      if (!r.ok) throw new Error(`model ${modelUrl}: HTTP ${r.status}`);
      return r.arrayBuffer();
    }) };
  }

  /** Embeddings of patches; `keys` (optional) cache them for the session. */
  async embed(patches: Patch[], keys?: string[]): Promise<Float32Array[]> {
    const out: (Float32Array | null)[] = patches.map((_, i) => (keys ? this.cache.get(keys[i]) ?? null : null));
    const todo = out.map((v, i) => (v ? -1 : i)).filter((i) => i >= 0);
    for (let s = 0; s < todo.length; s += this.batch) {
      // give the page a turn between batches: on the CPU (WebAssembly) a pass blocks the thread
      await new Promise((r) => setTimeout(r, 0));
      if (this.provider === "wasm") this.batch = 8;
      const idx = todo.slice(s, s + this.batch);
      const x = new Float32Array(idx.length * 3 * INPUT * INPUT);
      idx.forEach((i, j) => prepare(patches[i], x, j * 3 * INPUT * INPUT));
      const t0 = performance.now();
      const r = await this.onnx.run(this.model, x, [idx.length, 3, INPUT, INPUT]);
      this.seconds += (performance.now() - t0) / 1000;
      this.count += idx.length;
      this.provider = this.onnx.provider;
      const data = r.data as Float32Array;
      idx.forEach((i, j) => {
        const e = Float32Array.from(data.subarray(j * DIM, (j + 1) * DIM));
        out[i] = e;
        if (keys) this.cache.set(keys[i], e);
      });
    }
    return out as Float32Array[];
  }
}

export function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** Cosine similarity for the [CLS | mean] embeddings (each half unit length). */
export function cosine(a: Float32Array, b: Float32Array): number {
  return dot(a, b) / 2;
}
