// A raw precomputed volume read across chunk borders, with and without a z step. Run: npm test
import { test } from "node:test";
import assert from "node:assert/strict";
import { PrecomputedSource } from "../src/precomputed.ts";

const SIZE = [50, 40, 24], CH = [16, 16, 8];  // x, y, z
const value = (z: number, y: number, x: number) => (z * 31 + y * 7 + x * 3) % 251;

test("a raw precomputed volume reads the right voxels across chunks, NaN outside, planes skipped by a z step", async () => {
  const real = globalThis.fetch;
  globalThis.fetch = (async (u: string) => {
    const url = String(u);
    if (url.endsWith("/info")) return new Response(JSON.stringify({ "@type": "neuroglancer_multiscale_volume", data_type: "uint8", num_channels: 1, type: "image",
      scales: [{ key: "s0", size: SIZE, resolution: [8, 8, 8], chunk_sizes: [CH], encoding: "raw", voxel_offset: [0, 0, 0] }] }));
    const m = /s0\/(\d+)-(\d+)_(\d+)-(\d+)_(\d+)-(\d+)$/.exec(url);
    if (!m) return new Response(null, { status: 404 });
    const [x0, x1, y0, y1, z0, z1] = m.slice(1).map(Number), out = new Uint8Array((x1 - x0) * (y1 - y0) * (z1 - z0));
    let i = 0;
    for (let z = z0; z < z1; z++) for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) out[i++] = value(z, y, x);
    return new Response(out);
  }) as typeof fetch;
  try {
    const s = await PrecomputedSource.open("http://volume.test/v");
    for (const st of [1, 3]) {
      const lo = [2, -3, 10] as [number, number, number], hi = [23, 38, 53] as [number, number, number];
      const r = await s.read(0, lo, hi, st);
      assert.deepEqual(r.shape, [Math.ceil(21 / st), 41, 43]);
      let i = 0;
      for (let z = lo[0]; z < hi[0]; z += st) for (let y = lo[1]; y < hi[1]; y++) for (let x = lo[2]; x < hi[2]; x++, i++) {
        const inside = y >= 0 && x < SIZE[0];
        if (inside ? r.data[i] !== value(z, y, x) : r.data[i] === r.data[i]) assert.fail(`step ${st}: (${z}, ${y}, ${x}) is ${r.data[i]}`);
      }
    }
  } finally { globalThis.fetch = real; }
});
