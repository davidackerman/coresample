// The two read fixes on trial (experiment/fast-reads) on a small OME-Zarr volume served over
// HTTP: lazy neighbour chunks in 3-D (Options.lazyNeighbours) and the rough pass from the model's
// tokens (Options.roughTokens), with a stand-in model. Run: npm test
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { createServer, type Server } from "node:http";
import { DIM, Embedder } from "../src/embedder.ts";
import { Search, type Options } from "../src/search.ts";
import { OmeZarrSource, type Vec3 } from "../src/sources.ts";

const Z = 72, Y = 160, X = 160, CH = [32, 32, 32];  // 32 deep: room for the rough pass's 16 px box a level up
const value = (z: number, y: number, x: number) => (z * 7 + Math.floor(40 * Math.sin(y / 5) * Math.cos(x / 7)) + 128) & 255;
function file(path: string): Buffer | null {
  const shape = (k: number) => [Z, Y, X].map((v) => Math.ceil(v / 2 ** k));
  if (path === ".zgroup") return Buffer.from(JSON.stringify({ zarr_format: 2 }));
  if (path === ".zattrs") return Buffer.from(JSON.stringify({ multiscales: [{ version: "0.4", axes: ["z", "y", "x"].map((name) => ({ name, type: "space" })),
    datasets: [0, 1, 2].map((k) => ({ path: String(k), coordinateTransformations: [{ type: "scale", scale: [2 ** k, 2 ** k, 2 ** k] }] })) }] }));
  const za = /^(\d)\/\.zarray$/.exec(path);
  if (za) return Buffer.from(JSON.stringify({ zarr_format: 2, shape: shape(Number(za[1])), chunks: CH, dtype: "|u1", compressor: null,
    fill_value: 0, order: "C", filters: null }));
  const m = /^(\d)\/(\d+)\.(\d+)\.(\d+)$/.exec(path);
  if (!m) return null;
  const [k, cz, cy, cx] = m.slice(1).map(Number), f = 2 ** k, out = Buffer.alloc(CH[0] * CH[1] * CH[2]);
  for (let z = 0; z < CH[0]; z++) for (let y = 0; y < CH[1]; y++) for (let x = 0; x < CH[2]; x++)
    out[(z * CH[1] + y) * CH[2] + x] = value((cz * CH[0] + z) * f, (cy * CH[1] + y) * f, (cx * CH[2] + x) * f);
  return out;
}
let server: Server, base = "";
before(async () => {
  server = createServer((req, res) => {
    const f = file((req.url ?? "").slice(1));
    if (!f) { res.writeHead(404).end(); return; }
    res.writeHead(200, { "content-length": f.length }).end(f);
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  base = `http://127.0.0.1:${(server.address() as { port: number }).port}`;
});
after(() => server.close());

/** A stand-in model: the pooled export gives 1152 numbers picked from the pixels; a tokens
 * export (its file name says so) gives CLS + 8 x 8 tokens, each from its own 14 x 14 pixels. */
function stand(provider = "wasm") {
  const emb = new Embedder((f) => f);
  let runs = 0;
  (emb as any).onnx = { provider, async run(m: { key: string }, x: Float32Array, dims: number[]) {
    runs++;
    const n = dims[0], S = dims[2], per = x.length / n;
    if (!m.key.includes("tokens")) {
      const out = new Float32Array(n * DIM);
      for (let i = 0; i < n; i++) for (let d = 0; d < DIM; d++) out[i * DIM + d] = x[i * per + ((d * 97) % per)];
      return { data: out, dims: [n, DIM] };
    }
    const g = 8, tp = S / g, T = 1 + g * g, D = 384, out = new Float32Array(n * T * D);
    for (let i = 0; i < n; i++) for (let t = 0; t < g * g; t++) {
      const ty = Math.floor(t / g), tx = t % g, o = (i * T + 1 + t) * D;
      for (let d = 0; d < D; d++) out[o + d] = x[i * per + (ty * tp + ((d * 7) % tp)) * S + tx * tp + ((d * 13) % tp)];
    }
    return { data: out, dims: [n, T, D] };
  } };
  emb.provider = provider;
  return { emb, runs: () => runs };
}
const opt = (more: Partial<Options>): Options => ({ working: 0, patch: 32, region: [[0, 0, 0], [Z, Y, X]] as [Vec3, Vec3], roughPass: false, topN: 20,
                                                   maxCoarseVoxels: 1e9, ...more });

test("lazy neighbours with a policy that always reads them find the same hits as reading every tile whole", async () => {
  const a = new Search(await OmeZarrSource.open(base, "a", { mirrors: false }), stand().emb, [36, 80, 80], opt({}));
  const b = new Search(await OmeZarrSource.open(base, "b", { mirrors: false }), stand().emb, [36, 80, 80], opt({ lazyNeighbours: { top: 1e9 } }));
  await a.start();
  await b.start();
  assert.equal(b.state.phase, "done", b.state.message);
  assert.equal(b.state.patches, a.state.patches);
  assert.deepEqual(new Set(b.state.hits.map((h) => h.center.join(","))), new Set(a.state.hits.map((h) => h.center.join(","))));
  assert.ok(b.lazy.tiles > 0 && b.lazy.parked === 0, JSON.stringify(b.lazy));
});

test("lazy neighbours that never look promising still read every crossing patch once its chunks are all there", async () => {
  // the whole region is read, so every neighbour comes down as a tile: nothing is left parked
  const a = new Search(await OmeZarrSource.open(base, "a", { mirrors: false }), stand().emb, [36, 80, 80], opt({}));
  const b = new Search(await OmeZarrSource.open(base, "b", { mirrors: false }), stand().emb, [36, 80, 80], opt({ lazyNeighbours: { quantile: 2 } }));
  await a.start();
  await b.start();
  assert.equal(b.state.phase, "done", b.state.message);
  assert.equal(b.state.patches, a.state.patches, JSON.stringify(b.lazy));
  assert.equal(b.lazy.parked, b.lazy.unparked, JSON.stringify(b.lazy));
});

test("the rough pass from tokens ranks the tiles with a model run per crop, not per place", async () => {
  const pa = stand("webgpu"), pb = stand("webgpu");
  const a = new Search(await OmeZarrSource.open(base, "a", { mirrors: false }), pa.emb, [36, 80, 80], opt({ roughPass: true }));
  const b = new Search(await OmeZarrSource.open(base, "b", { mirrors: false }), pb.emb, [36, 80, 80], opt({ roughPass: true, roughTokens: true }));
  await a.start();
  await b.start();
  for (const s of [a, b]) {
    assert.equal(s.state.phase, "done", s.state.message);
    assert.ok(s.state.hits.length > 0);
    assert.ok((s.state.rough?.places ?? 0) > 20, `${s.state.rough?.places} places`);
  }
  assert.equal(b.state.rough!.places, a.state.rough!.places);
  // here a level-1 chunk plane (32 px) holds 2 x 2 places: half the runs or fewer (hemibrain's
  // 64 px planes hold 16); the query's own runs count in both
  assert.ok(pb.emb.count - b.state.patches <= (pa.emb.count - a.state.patches) / 1.8,
            `rough runs: tokens ${pb.emb.count - b.state.patches}, per place ${pa.emb.count - a.state.patches}`);
});
