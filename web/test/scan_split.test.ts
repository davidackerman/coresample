// The cheap scan cut into slabs (as the page runs it in several workers) gives the scores of the
// scan in one piece, and the one-piece scan gives the scores of the scan before it was cut up
// (kept here verbatim as the reference). Run: npm test
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  scan, scanPrep, scanSlabs, slabVolume, scanBlock, scanMerge, percentile, textureStats, histMap, type Dims, type ScanInput,
} from "../src/scan.ts";

// ---- the scan as it was (816aa0b), the reference -------------------------------------------
function percentileOld(a: Float32Array, q: number[]): number[] {
  const step = Math.max(1, Math.floor(a.length / 2000000));
  const s: number[] = [];
  for (let i = 0; i < a.length; i += step) s.push(a[i]);
  s.sort((x, y) => x - y);
  return q.map((p) => s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))]);
}
function zscore(a: Float32Array): Float32Array {
  let m = 0, v = 0;
  for (let i = 0; i < a.length; i++) m += a[i];
  m /= a.length;
  for (let i = 0; i < a.length; i++) v += (a[i] - m) ** 2;
  const s = Math.sqrt(v / a.length) + 1e-9, out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = (a[i] - m) / s;
  return out;
}
function scanOld(inp: ScanInput): { score: Float32Array; dims: Dims } {
  const { vol, dims, query, qdims, window: w, margin: m } = inp;
  const inner = new Float32Array((dims[0] - 2 * m[0]) * (dims[1] - 2 * m[1]) * (dims[2] - 2 * m[2]));
  {
    let k = 0;
    for (let z = m[0]; z < dims[0] - m[0]; z++)
      for (let y = m[1]; y < dims[1] - m[1]; y++) {
        const row = (z * dims[1] + y) * dims[2];
        inner.set(vol.subarray(row + m[2], row + dims[2] - m[2]), k);
        k += dims[2] - 2 * m[2];
      }
  }
  const edges = (() => {
    const [lo, hi] = percentileOld(inner, [0.5, 99.5]);
    return Array.from({ length: 17 }, (_, i) => lo + ((hi - lo) * i) / 16);
  })();
  const qc: number[] = new Array(16).fill(0);
  let nq = 0;
  for (let z = m[0]; z < m[0] + w[0]; z++)
    for (let y = m[1]; y < m[1] + w[1]; y++)
      for (let x = m[2]; x < m[2] + w[2]; x++) {
        const v = query[(z * qdims[1] + y) * qdims[2] + x];
        let b = 0;
        while (b < 15 && v >= edges[b + 1]) b++;
        qc[b]++;
        nq++;
      }
  const qcdf = new Float32Array(16);
  let acc = 0;
  for (let b = 0; b < 16; b++) { acc += qc[b] / nq; qcdf[b] = acc; }
  const qt = textureStats(query, qdims, w);
  const qi = (m[0] * qt.dims[1] + m[1]) * qt.dims[2] + m[2];
  const qstats = qt.maps.map((mp) => mp[qi]);
  const tex = textureStats(vol, dims, w);
  const hist = histMap(vol, dims, w, qcdf, edges);
  const region: Dims = [dims[0] - 2 * m[0], dims[1] - 2 * m[1], dims[2] - 2 * m[2]];
  const od: Dims = [Math.max(1, region[0] - w[0] + 1), Math.max(1, region[1] - w[1] + 1), Math.max(1, region[2] - w[2] + 1)];
  const n = od[0] * od[1] * od[2];
  const td = tex.dims;
  const spread = tex.maps.map((mp) => {
    let mu = 0, v = 0, cnt = 0;
    for (let z = m[0]; z + w[0] <= dims[0] - m[0]; z++)
      for (let y = m[1]; y + w[1] <= dims[1] - m[1]; y++)
        for (let x = m[2]; x + w[2] <= dims[2] - m[2]; x++) { mu += mp[(z * td[1] + y) * td[2] + x]; cnt++; }
    mu /= Math.max(cnt, 1);
    for (let z = m[0]; z + w[0] <= dims[0] - m[0]; z++)
      for (let y = m[1]; y + w[1] <= dims[1] - m[1]; y++)
        for (let x = m[2]; x + w[2] <= dims[2] - m[2]; x++) v += (mp[(z * td[1] + y) * td[2] + x] - mu) ** 2;
    return Math.sqrt(v / Math.max(cnt, 1)) + 1e-9;
  });
  const t = new Float32Array(n), h = new Float32Array(n);
  let i = 0;
  for (let z = 0; z < od[0]; z++)
    for (let y = 0; y < od[1]; y++)
      for (let x = 0; x < od[2]; x++, i++) {
        const j = ((z + m[0]) * td[1] + (y + m[1])) * td[2] + (x + m[2]);
        let s = 0;
        for (let f = 0; f < 4; f++) s += ((tex.maps[f][j] - qstats[f]) / spread[f]) ** 2;
        t[i] = -Math.sqrt(s);
        h[i] = hist.data[j];
      }
  const zt = zscore(t), zh = zscore(h), score = new Float32Array(n);
  for (let k = 0; k < n; k++) score[k] = (zt[k] + zh[k]) / 2;
  return { score, dims: od };
}
// ---------------------------------------------------------------------------------------------

/** Textured test data: blobs, a gradient and noise, from a fixed seed. */
function data(d: Dims, seed: number): Float32Array {
  let s = seed;
  const rnd = () => ((s = (s * 1103515245 + 12345) % 2147483648) / 2147483648);
  const a = new Float32Array(d[0] * d[1] * d[2]);
  let i = 0;
  for (let z = 0; z < d[0]; z++)
    for (let y = 0; y < d[1]; y++)
      for (let x = 0; x < d[2]; x++, i++)
        a[i] = 120 + 60 * Math.sin(x / 7 + z / 5) * Math.cos(y / 9) + 0.3 * x + 30 * rnd();
  return a;
}
function input(region: Dims, w: Dims, m: Dims, seed = 1): ScanInput {
  const dims: Dims = [region[0] + 2 * m[0], region[1] + 2 * m[1], region[2] + 2 * m[2]];
  const qdims: Dims = [w[0] + 2 * m[0], w[1] + 2 * m[1], w[2] + 2 * m[2]];
  return { vol: data(dims, seed), dims, query: data(qdims, seed + 11), qdims, window: w, margin: m };
}
function split(inp: ScanInput, k: number) {
  const { vol, dims, window: w, margin: m } = inp, prep = scanPrep(inp), slabs = scanSlabs(dims, w, m, k);
  const parts = slabs.map((s) => { const sv = slabVolume(vol, dims, s); return scanBlock(sv.vol, sv.dims, s, dims, w, m, prep); });
  return { slabs, ...scanMerge(parts, dims, w, m, prep) };
}
const maxDiff = (a: Float32Array, b: Float32Array) => { let d = 0; for (let i = 0; i < a.length; i++) d = Math.max(d, Math.abs(a[i] - b[i])); return d; };

const CASES: [string, Dims, Dims, Dims][] = [
  ["3-D volume", [40, 30, 50], [8, 8, 8], [4, 4, 4]],
  ["2-D image", [1, 180, 150], [1, 8, 8], [0, 4, 4]],
  ["3-D, region thinner than a window in z", [6, 30, 40], [8, 8, 8], [4, 4, 4]],
  ["2-D, small window", [1, 90, 70], [1, 2, 2], [0, 4, 4]],
];

test("typed-array percentile equals the old comparator sort", () => {
  for (const seed of [1, 2, 3]) {
    const a = data([1, 300, 300], seed);
    for (let i = 0; i < a.length; i += 17) a[i] = Math.round(a[i]);  // ties
    assert.deepEqual(percentile(a, [0.5, 50, 99.5]), percentileOld(a, [0.5, 50, 99.5]));
  }
});

for (const [name, region, w, m] of CASES) {
  test(`one-piece scan equals the scan before the change: ${name}`, () => {
    const inp = input(region, w, m), a = scan(inp), b = scanOld(inp);
    assert.deepEqual(a.dims, b.dims);
    assert.equal(maxDiff(a.score, b.score), 0);
  });
  test(`scan in 2, 3 and 4 slabs equals the scan in one piece: ${name}`, () => {
    const inp = input(region, w, m, 5), whole = scan(inp);
    for (const k of [2, 3, 4]) {
      const r = split(inp, k);
      assert.deepEqual(r.dims, whole.dims);
      // window sums along the cut axis start at a different voxel, so rounding may differ slightly
      assert.ok(maxDiff(r.score, whole.score) < 1e-4, `k=${k}: ${r.slabs.length} slabs, max diff ${maxDiff(r.score, whole.score)}`);
    }
  });
}

test("slabs cover every window once, in order, with the filters' border inside the volume", () => {
  for (const [, region, w, m] of CASES) {
    const dims: Dims = [region[0] + 2 * m[0], region[1] + 2 * m[1], region[2] + 2 * m[2]];
    const slabs = scanSlabs(dims, w, m, 4);
    let next = 0;
    for (const s of slabs) {
      assert.equal(s.c0, next);
      assert.ok(s.c1 > s.c0 && s.lo >= 0 && s.hi <= dims[s.axis]);
      next = s.c1;
    }
    const od = [region[0] - w[0] + 1, region[1] - w[1] + 1, region[2] - w[2] + 1].map((v) => Math.max(1, v));
    assert.equal(next, od[slabs[0].axis]);
  }
});
