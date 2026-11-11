// A remote TIFF served as an OME-Zarr (src/remotetiff.ts): every level's values, from small
// TIFFs in test/fixtures (made by formulas recomputed here) read over HTTP by byte ranges.
// Levels the file has are its own; levels below come from every 2^k-th row, each averaged
// along x (a compressed strip: the rows of it that fall in the output row, averaged too); such
// a level reads 1/2^k of an uncompressed file, and a finer level then reads only the rows the
// coarser did not. Run: npm test
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { createServer, type Server } from "node:http";
import { readFileSync } from "node:fs";
import { openTiff, type RemoteTiff } from "../src/remotetiff.ts";

const dir = new URL("./fixtures/", import.meta.url);
let server: Server, base = "";
before(async () => {
  server = createServer((req, res) => {
    let buf: Buffer;
    try { buf = readFileSync(new URL("." + (req.url ?? ""), dir)); } catch { res.writeHead(404).end(); return; }
    const m = /^bytes=(\d+)-(\d+)$/.exec(req.headers.range ?? "");
    if (!m) { res.writeHead(200, { "content-length": buf.length }).end(buf); return; }
    const a = Number(m[1]), b = Math.min(buf.length - 1, Number(m[2]));
    res.writeHead(206, { "content-range": `bytes ${a}-${b}/${buf.length}`, "content-length": b - a + 1 }).end(buf.subarray(a, b + 1));
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  base = `http://127.0.0.1:${(server.address() as { port: number }).port}/`;
});
after(() => server.close());

const CTORS: Record<string, any> = { "|u1": Uint8Array, "<u2": Uint16Array, "<i2": Int16Array };
/** Level k of the served OME-Zarr, channel c, as one array (y, x). */
async function level(t: RemoteTiff, k: number, c = 0) {
  const za = JSON.parse(await (await t.folder.file(`${k}/.zarray`))!.text());
  const multi = za.shape.length === 3, [H, W] = za.shape.slice(-2), [cy, cx] = za.chunks.slice(-2), out = new Float64Array(H * W);
  for (let yi = 0; yi * cy < H; yi++)
    for (let xi = 0; xi * cx < W; xi++) {
      const f = await t.folder.file(multi ? `${k}/${c}/${yi}/${xi}` : `${k}/${yi}/${xi}`);
      const a = new CTORS[za.dtype](await f!.arrayBuffer());
      for (let y = 0; y < cy && yi * cy + y < H; y++)
        for (let x = 0; x < cx && xi * cx + x < W; x++) out[(yi * cy + y) * W + xi * cx + x] = a[y * cx + x];
    }
  return { H, W, chunk: [cy, cx], dtype: za.dtype as string, data: out };
}
/** Every 2^k-th row (rows 0, 2^k, 2*2^k...), each 2^k columns averaged; `rows` > 1: all the rows
 * of each 2^k averaged (a compressed strip holds them all). */
function sampled(v: (x: number, y: number) => number, H0: number, W0: number, k: number, rows = 1) {
  const s = 2 ** k, H = Math.ceil(H0 / s), W = Math.ceil(W0 / s), out = new Float64Array(H * W);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const Ys = rows === 1 ? [y * s] : Array.from({ length: Math.min(s, H0 - y * s) }, (_, i) => y * s + i);
      let t = 0, n = 0;
      for (const Y of Ys) for (let X = x * s; X < Math.min(W0, (x + 1) * s); X++) { t += v(X, Y); n++; }
      out[y * W + x] = Math.round(t / n);
    }
  return out;
}
const same = (a: Float64Array, b: Float64Array, what: string) => {
  assert.equal(a.length, b.length, `${what}: size`);
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) assert.fail(`${what}: value ${i} is ${a[i]}, not ${b[i]}`);
};

test("uncompressed strips, no pyramid: levels from every 2^k-th row, reading 1/2^k of the file", async () => {
  const v = (x: number, y: number) => (x * x + 3 * y) % 251;
  const t = await openTiff(base + "strips_u8.tif");
  const attrs = JSON.parse(await (await t.folder.file(".zattrs"))!.text());
  assert.equal(attrs.multiscales[0].datasets.length, 3);  // 700, 350, 175 px
  assert.deepEqual(attrs.multiscales[0].datasets[2].coordinateTransformations[0].scale, [4, 4]);
  for (let k = 0; k < 3; k++) same((await level(t, k)).data, sampled(v, 500, 700, k), `level ${k}`);
  const fresh = await openTiff(base + "strips_u8.tif"), size = readFileSync(new URL("strips_u8.tif", dir)).length;
  await level(fresh, 2);
  const share = fresh.bytes() / size;
  assert.ok(share > 0.2 && share < 0.3, `level 2 read ${share.toFixed(3)} of the file, not about 1/4`);
  await level(fresh, 1);  // rows 0, 2, 4...: 0, 4, 8... are kept from level 2
  const more = fresh.bytes() / size - share;
  assert.ok(more > 0.2 && more < 0.3, `level 1 after level 2 read ${more.toFixed(3)} more of the file, not about 1/4`);
});

test("a tiled, compressed TIFF with overviews: its own levels, one tile a chunk", async () => {
  const t = await openTiff(base + "cog_u16.tif");
  for (const [k, H, W] of [[0, 480, 640], [1, 240, 320], [2, 120, 160]]) {
    const L = await level(t, k);
    assert.deepEqual([L.H, L.W, L.dtype], [H, W, "<u2"]);
    assert.deepEqual(L.chunk, [128, 128]);
    same(L.data, sampled((x, y) => (x * 37 + y * 11 + 1000 * k) % 4000, H, W, 0), `level ${k}`);
  }
});

test("compressed RGB strips, no pyramid: channels, and the rows of a strip averaged", async () => {
  const v = [(x: number, y: number) => (x + y) % 256, (x: number) => (2 * x) % 256, (x: number, y: number) => (3 * y + Math.floor(x / 7)) % 256];
  const t = await openTiff(base + "deflate_rgb.tif");
  const attrs = JSON.parse(await (await t.folder.file(".zattrs"))!.text());
  assert.deepEqual(attrs.omero.channels.map((c: any) => c.label), ["red", "green", "blue"]);
  for (let c = 0; c < 3; c++) {
    same((await level(t, 0, c)).data, sampled(v[c], 200, 300, 0), `channel ${c}, level 0`);
    same((await level(t, 1, c)).data, sampled(v[c], 200, 300, 1, 2), `channel ${c}, level 1`);  // strips of 8 rows: both rows of each pair
  }
});

test("big-endian, planar, signed 16-bit", async () => {
  const v = [(x: number, y: number) => x * 5 - y * 3, (x: number, y: number) => ((x * y) % 3001) - 1500];
  const t = await openTiff(base + "planar_be.tif");
  for (let c = 0; c < 2; c++)
    for (let k = 0; k < 2; k++) {
      const L = await level(t, k, c);
      assert.equal(L.dtype, "<i2");
      same(L.data, sampled(v[c], 300, 260, k), `channel ${c}, level ${k}`);
    }
});
