// The search's bookkeeping, on made-up 2-D sources with a stand-in model (no network, no ONNX):
// embeddings cached per source, and a search cancelled while it waits for the model reads no
// further. Run: npm test
import { test } from "node:test";
import assert from "node:assert/strict";
import { DIM, Embedder } from "../src/embedder.ts";
import { Search } from "../src/search.ts";
import type { Level, Region, Source, Vec3 } from "../src/sources.ts";

/** A 1024 x 1024 image of `f(y, x)` (0-255) with 5 levels, 256 px tiles; counts its reads. */
function source(name: string, f: (y: number, x: number) => number): Source & { reads: number } {
  const levels: Level[] = [];
  for (let k = 0; k < 5; k++) levels.push({ shape: [1, 1024 >> k, 1024 >> k], chunk: [1, 256, 256], scale: [1, 2 ** k, 2 ** k] });
  return {
    name, dims: 2, channels: 1, levels, unit: "px", bytesRead: 0, reads: 0,
    async read(level: number, lo: Vec3, hi: Vec3): Promise<Region> {
      this.reads++;
      const h = hi[1] - lo[1], w = hi[2] - lo[2], s = 2 ** level, out = new Float32Array(h * w);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = f((lo[1] + y) * s, (lo[2] + x) * s);
      return { data: out, shape: [1, h, w], channels: 1 };
    },
  };
}
const stripes = (y: number, x: number) => 128 + 100 * Math.sin(x / 9) * Math.cos(y / 23);
const rings = (y: number, x: number) => 128 + 100 * Math.cos(Math.hypot(y - 500, x - 500) / 7);

/** The real Embedder (and its cache) with the model swapped for a stand-in: each part of the
 * embedding is the input's first values, so different pixels give different vectors. `wait`
 * delays every pass. Counts the patches passed through it. */
function embedder(wait = 0) {
  const emb = new Embedder(() => "stand-in");
  const counter = { patches: 0 };
  (emb as any).onnx = {
    provider: "wasm",
    async run(_m: unknown, x: Float32Array, dims: number[]) {
      if (wait) await new Promise((r) => setTimeout(r, wait));
      const n = dims[0], per = x.length / n, out = new Float32Array(n * DIM);
      for (let i = 0; i < n; i++) for (let d = 0; d < DIM; d++) out[i * DIM + d] = x[i * per + ((d * 97) % per)];
      counter.patches += n;
      return { data: out, dims: [n, DIM] };
    },
  };
  return { emb, counter };
}
const options = () => ({ working: 0, patch: 64, region: [[0, 0, 0], [1, 1024, 1024]] as [Vec3, Vec3], roughPass: false, topN: 10 });

test("another dataset at the same places gets its own embeddings, not the first's", async () => {
  const { emb, counter } = embedder();
  const a = new Search(source("a", stripes), emb, [0, 300, 300], options());
  await a.start();
  assert.equal(a.state.phase, "done");
  const first = counter.patches;
  assert.ok(first > 200, `the first search embedded ${first} patches`);
  const b = new Search(source("b", rings), emb, [0, 300, 300], options());
  await b.start();
  const second = counter.patches - first;
  assert.ok(second >= first - 2, `the second dataset embedded ${second} patches of ${first}: the rest came from the first's cache`);
  // the same source again: from the cache
  const again = new Search(a.src, emb, [0, 300, 300], options());
  await again.start();
  assert.ok(counter.patches - first - second < 10, "the first dataset again should come from the cache");
});

test("a search cancelled while the model runs on its query reads nothing after", async () => {
  const { emb } = embedder(200);
  const src = source("slow model", stripes), s = new Search(src, emb, [0, 300, 300], options());
  const done = s.start();
  await new Promise((r) => setTimeout(r, 50));  // the query's pass under way
  s.cancel();
  await done;
  const reads = src.reads;
  await new Promise((r) => setTimeout(r, 600));
  assert.equal(src.reads, reads, "it read on after it was cancelled");
  assert.ok(reads <= 2, `${reads} reads: the query's patch alone`);
  assert.equal(s.state.phase, "query");
  await s.resume();  // nothing resumes a cancelled search
  assert.equal(src.reads, reads);
});
