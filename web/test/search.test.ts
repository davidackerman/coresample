// The search's bookkeeping, on made-up 2-D sources with a stand-in model (no network, no ONNX):
// embeddings cached per source, and a search cancelled while it waits for the model reads no
// further. Run: npm test
import { test } from "node:test";
import assert from "node:assert/strict";
import { DIM, Embedder } from "../src/embedder.ts";
import { DEFAULT_MATCH, NOWHERE, resized, Search, swapTones } from "../src/search.ts";
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

test("light and dark swapped: a patch mirrored in its own range, its colour kept", () => {
  const grey = swapTones({ data: Float32Array.from([10, 50, NaN, 200]), size: 2, channels: 1 });
  assert.deepEqual([...grey.data].map((v) => (v === v ? v : -1)), [200, 160, -1, 10]);
  const col = swapTones({ data: Float32Array.from([40, 40, 40, 140, 120, 100]), size: 1, channels: 3 }).data;
  assert.ok(col[0] > col[3], "the dark pixel is now the brighter one");
  assert.ok(Math.abs(col[3] - col[4] - 20) < 1e-3 && Math.abs(col[4] - col[5] - 20) < 1e-3, "the channels keep their differences");
});

test("a drawing with light and dark the other way round finds its place only when that is allowed", async () => {
  const blobs = (y: number, x: number) => 128 + 60 * Math.sin(x / 9) * Math.cos(y / 23) + 40 * Math.sin((x + 2 * y) / 31);
  const src = source("blobs", blobs), { emb } = embedder();
  const first = new Search(src, emb, [0, 300, 300], options());
  await first.start();
  const c = first.state.hits[3].center;
  const r = await src.read(0, [0, c[1] - 32, c[2] - 32], [1, c[1] + 32, c[2] + 32]);
  const sketch = swapTones({ data: r.data, size: 64, channels: 1 });
  const at = (s: Search) => s.state.hits.find((h) => h.center[1] === c[1] && h.center[2] === c[2]);
  const plain = new Search(src, emb, NOWHERE, { ...options(), sketch });
  await plain.start();
  assert.ok(!at(plain) || at(plain)!.score < 0.5, `as drawn, its place is no hit (${at(plain)?.score})`);
  const either = new Search(src, emb, NOWHERE, { ...options(), sketch, match: { ...DEFAULT_MATCH, tones: true } });
  await either.start();
  const h = at(either);
  assert.ok(h && h.score > 0.99, `light or dark: its place scores ${h?.score}`);
  assert.equal(either.explain(h!.index)?.variant, "iid");
  // and switched on during a search: the hits re-ranked
  await plain.setMatch({ ...DEFAULT_MATCH, tones: true });
  assert.ok((at(plain)?.score ?? 0) > 0.99, "switched on afterwards, its place is found");
});

test("any size, for a drawing: drawn at half the size it is in the data, its place is found at double", async () => {
  const blobs = (y: number, x: number) => 128 + 60 * Math.sin(x / 19) * Math.cos(y / 37) + 40 * Math.sin((x + 2 * y) / 53);
  const src = source("blobs2", blobs), { emb } = embedder();
  const first = new Search(src, emb, [0, 300, 300], options());
  await first.start();
  const c = first.state.hits[3].center;
  // the same place one level up: everything in it half the size, in the box's middle half
  const r = await src.read(1, [0, c[1] / 2 - 32, c[2] / 2 - 32], [1, c[1] / 2 + 32, c[2] / 2 + 32]);
  const sketch = { data: r.data, size: 64, channels: 1 };
  const at = (s: Search) => s.state.hits.find((h) => h.center[1] === c[1] && h.center[2] === c[2]);
  const s = new Search(src, emb, NOWHERE, { ...options(), sketch, match: { ...DEFAULT_MATCH, scales: true } });
  await s.start();
  const h = at(s);
  assert.ok(h && h.score > 0.95, `any size: its place scores ${h?.score}`);
  assert.equal(s.explain(h!.index)?.variant, "s-1");
  const small = resized(sketch, false);
  assert.equal(small.data[0], small.data[5], "the shrunk drawing has one tone around it");
});
