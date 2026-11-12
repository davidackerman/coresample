// Reads of every zStep-th plane (Source.read's zStep, used by the search's 3-D tile reads): the
// same values as the full read's planes, from a small OME-Zarr volume served over HTTP; and a
// 3-D search finds the same hits with them as without. Run: npm test
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { createServer, type Server } from "node:http";
import { DIM, Embedder } from "../src/embedder.ts";
import { Search } from "../src/search.ts";
import { OmeZarrSource, type Source, type Vec3 } from "../src/sources.ts";

const Z = 72, Y = 160, X = 160, CH = [16, 32, 32];
const value = (z: number, y: number, x: number) => (z * 7 + Math.floor(40 * Math.sin(y / 5) * Math.cos(x / 7)) + 128) & 255;
/** The files of a 3-level OME-Zarr (v2, uncompressed uint8): level k samples `value` every 2^k voxels. */
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

test("every 4th plane, from inside and from before the volume, equals those planes of the full read", async () => {
  const s = await OmeZarrSource.open(base, "planes", { mirrors: false });
  for (const [lo, hi] of [[[3, 10, 20], [50, 60, 70]], [[-6, 0, 0], [21, 40, 40]], [[60, 100, 100], [80, 140, 150]]] as [Vec3, Vec3][]) {
    const all = await s.read(0, lo, hi), some = await s.read(0, lo, hi, 4);
    const n = Math.ceil((hi[0] - lo[0]) / 4), plane = (hi[1] - lo[1]) * (hi[2] - lo[2]);
    assert.deepEqual(some.shape, [n, hi[1] - lo[1], hi[2] - lo[2]]);
    for (let k = 0; k < n; k++)
      assert.deepEqual(some.data.subarray(k * plane, (k + 1) * plane), all.data.subarray(4 * k * plane, (4 * k + 1) * plane), `plane ${lo[0] + 4 * k}`);
  }
});

test("a 3-D search finds the same hits reading the patches' planes alone", async () => {
  const stand = () => {
    const emb = new Embedder(() => "stand-in");
    (emb as any).onnx = { provider: "wasm", async run(_m: unknown, x: Float32Array, dims: number[]) {
      const n = dims[0], per = x.length / n, out = new Float32Array(n * DIM);
      for (let i = 0; i < n; i++) for (let d = 0; d < DIM; d++) out[i * DIM + d] = x[i * per + ((d * 97) % per)];
      return { data: out, dims: [n, DIM] };
    } };
    return emb;
  };
  const s = await OmeZarrSource.open(base, "planes", { mirrors: false });
  // the same source, every plane read (zStep ignored, as a source may)
  const every: Source = { ...s, name: s.name, levels: s.levels, dims: 3, channels: 1, unit: "", bytesRead: 0, read: (L, lo, hi) => s.read(L, lo, hi) };
  const opt = () => ({ working: 0, patch: 32, region: [[0, 0, 0], [Z, Y, X]] as [Vec3, Vec3], roughPass: false, topN: 20, maxCoarseVoxels: 1e9 });
  const a = new Search(s, stand(), [36, 80, 80], opt()), b = new Search(every, stand(), [36, 80, 80], opt());
  await a.start();
  await b.start();
  assert.equal(a.state.phase, "done", a.state.message);
  assert.ok(a.state.patches > 100, `${a.state.patches} patches`);
  assert.deepEqual(a.state.hits.map((h) => h.center), b.state.hits.map((h) => h.center));
  assert.deepEqual(a.state.hits.map((h) => h.score), b.state.hits.map((h) => h.score));
});
