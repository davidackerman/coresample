// The cheap scan: score every window of a coarse level against the query with model-free
// statistics (16-bin intensity histogram EMD and window texture stats), as search.py does for
// the feasibility tests (methods hist + tex; NCC was unreliable there). Volumes are Float32Arrays
// laid out z, y, x; a 2-D image has z = 1 and windows one voxel deep. Pure TypeScript, no DOM:
// it runs in a worker, and in Node for tests.

export type Dims = [number, number, number];

/** Sum over every window of shape w (valid mode), one separable running sum per axis. */
export function boxSum(a: Float32Array, d: Dims, w: Dims): { data: Float32Array; dims: Dims } {
  let cur = a, cd: Dims = [...d] as Dims;
  for (let axis = 2; axis >= 0; axis--) {
    const k = w[axis];
    if (k === 1) continue;
    const nd = [...cd] as Dims;
    nd[axis] = cd[axis] - k + 1;
    if (nd[axis] <= 0) return { data: new Float32Array(0), dims: [0, 0, 0] };
    const out = new Float32Array(nd[0] * nd[1] * nd[2]);
    const stride = axis === 2 ? 1 : axis === 1 ? cd[2] : cd[1] * cd[2];
    const ostride = axis === 2 ? 1 : axis === 1 ? nd[2] : nd[1] * nd[2];
    const [A, B] = axis === 2 ? [cd[0] * cd[1], 1] : axis === 1 ? [cd[0], cd[2]] : [1, cd[1] * cd[2]];
    const len = cd[axis];
    for (let p = 0; p < A; p++)
      for (let q = 0; q < B; q++) {
        const base = axis === 2 ? p * cd[2] : axis === 1 ? p * cd[1] * cd[2] + q : q;
        const obase = axis === 2 ? p * nd[2] : axis === 1 ? p * nd[1] * nd[2] + q : q;
        let s = 0;
        for (let i = 0; i < k; i++) s += cur[base + i * stride];
        out[obase] = s;
        for (let i = 1; i <= len - k; i++) {
          s += cur[base + (i + k - 1) * stride] - cur[base + (i - 1) * stride];
          out[obase + i * ostride] = s;
        }
      }
    cur = out;
    cd = nd;
  }
  return { data: cur === a ? Float32Array.from(a) : cur, dims: cd };
}

/** scipy.ndimage.correlate1d along `axis` with mode "reflect" (d c b a | a b c d | d c b a),
 * line by line. Axes of extent 1 are skipped (the caller standardizes features anyway). */
export function correlate1d(a: Float32Array, d: Dims, axis: number, k: Float32Array, out?: Float32Array): Float32Array {
  const n = d[axis], r = (k.length - 1) >> 1;
  const res = out ?? new Float32Array(a.length);
  if (n === 1) { res.set(a); return res; }
  const stride = axis === 2 ? 1 : axis === 1 ? d[2] : d[1] * d[2];
  const lines = a.length / n, line = new Float32Array(n + 2 * r);
  for (let l = 0; l < lines; l++) {
    const base = axis === 2 ? l * n : axis === 1 ? Math.floor(l / d[2]) * d[1] * d[2] + (l % d[2]) : l;
    for (let i = 0; i < n + 2 * r; i++) {
      let j = i - r;
      while (j < 0 || j >= n) j = j < 0 ? -j - 1 : 2 * n - j - 1;
      line[i] = a[base + j * stride];
    }
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let t = 0; t <= 2 * r; t++) s += k[t] * line[i + t];
      res[base + i * stride] = s;
    }
  }
  return res;
}

function gauss1d(sigma: number): Float32Array {
  const r = Math.floor(4 * sigma + 0.5), k = new Float32Array(2 * r + 1);  // scipy's truncate=4
  let s = 0;
  for (let i = -r; i <= r; i++) s += k[i + r] = Math.exp(-(i * i) / (2 * sigma * sigma));
  for (let i = 0; i < k.length; i++) k[i] /= s;
  return k;
}

/** Separable Gaussian blur (scipy.ndimage.gaussian_filter, mode reflect, truncate 4). */
export function blur(a: Float32Array, d: Dims, sigma: number): Float32Array {
  const k = gauss1d(sigma);
  let cur = a;
  for (let axis = 0; axis < 3; axis++) if (d[axis] > 1) cur = correlate1d(cur, d, axis, k);
  return cur === a ? Float32Array.from(a) : cur;
}

const DERIV = new Float32Array([-1, 0, 1]), SMOOTH = new Float32Array([1, 2, 1]);

/** Per-voxel features whose window means describe texture: value, value^2, Sobel gradient
 * magnitude, |DoG(1, 2)| (as search.texture_features). */
export function features(a: Float32Array, d: Dims): Float32Array[] {
  const n = a.length, sq = new Float32Array(n), g = new Float32Array(n);
  for (let i = 0; i < n; i++) sq[i] = a[i] * a[i];
  const tmp = new Float32Array(n), tmp2 = new Float32Array(n);
  for (let axis = 0; axis < 3; axis++) {
    if (d[axis] === 1) continue;
    let cur = correlate1d(a, d, axis, DERIV, tmp);
    for (let o = 0; o < 3; o++) {
      if (o === axis || d[o] === 1) continue;
      const next = correlate1d(cur, d, o, SMOOTH, cur === tmp ? tmp2 : tmp);
      cur = next;
    }
    for (let i = 0; i < n; i++) g[i] += cur[i] * cur[i];
  }
  for (let i = 0; i < n; i++) g[i] = Math.sqrt(g[i]);
  const b1 = blur(a, d, 1), b2 = blur(a, d, 2), dog = new Float32Array(n);
  for (let i = 0; i < n; i++) dog[i] = Math.abs(b1[i] - b2[i]);
  return [a, sq, g, dog];
}

/** (4 maps) window mean, std, mean gradient, mean |DoG|. */
export function textureStats(a: Float32Array, d: Dims, w: Dims): { maps: Float32Array[]; dims: Dims } {
  const nw = w[0] * w[1] * w[2];
  const f = features(a, d).map((x) => boxSum(x, d, w));
  const dims = f[0].dims, n = f[0].data.length;
  const mean = new Float32Array(n), std = new Float32Array(n), gm = new Float32Array(n), dg = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const m = f[0].data[i] / nw;
    mean[i] = m;
    std[i] = Math.sqrt(Math.max(f[1].data[i] / nw - m * m, 0));
    gm[i] = f[2].data[i] / nw;
    dg[i] = f[3].data[i] / nw;
  }
  return { maps: [mean, std, gm, dg], dims };
}

export function percentile(a: Float32Array, q: number[]): number[] {
  const step = Math.max(1, Math.floor(a.length / 2000000));  // a sample is plenty
  const s = new Float32Array(Math.ceil(a.length / step));
  for (let i = 0, k = 0; i < a.length; i += step, k++) s[k] = a[i];
  s.sort();  // a typed array sorts by value, the same order as (x, y) => x - y, about 10x faster
  return q.map((p) => s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))]);
}

/** Negative histogram EMD of every window against the query's histogram. */
export function histMap(a: Float32Array, d: Dims, w: Dims, qcdf: Float32Array, edges: number[]): { data: Float32Array; dims: Dims } {
  const nb = edges.length - 1, nw = w[0] * w[1] * w[2];
  let emd: Float32Array | null = null, dims: Dims = [0, 0, 0];
  const ind = new Float32Array(a.length);
  for (let b = 0; b < nb - 1; b++) {
    const t = edges[b + 1];
    for (let i = 0; i < a.length; i++) ind[i] = a[i] < t ? 1 : 0;  // cumulative: bins <= b
    const s = boxSum(ind, d, w);
    if (!emd) { emd = new Float32Array(s.data.length); dims = s.dims; }
    for (let i = 0; i < s.data.length; i++) emd[i] += Math.abs(s.data[i] / nw - qcdf[b]);
  }
  for (let i = 0; i < emd!.length; i++) emd![i] = -emd![i];
  return { data: emd!, dims };
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

export interface ScanInput {
  vol: Float32Array; dims: Dims;   // the coarse level over the region, with `margin` voxels of context
  query: Float32Array; qdims: Dims;  // the query window with `margin` voxels of context
  window: Dims;
  margin: Dims;
}

/** What the scan needs from the whole volume before its windows are scored: the 16 histogram
 * bins (from percentiles of the region itself), the query's cumulative histogram and its four
 * texture statistics. */
export interface ScanPrep { edges: number[]; qcdf: Float32Array; qstats: number[] }

/** The windows scored: those whose corner lies in the region (the volume minus its margin),
 * `od` of them per axis; the result's index i is the window whose corner is at margin + i. */
export function scanDims(dims: Dims, w: Dims, m: Dims): Dims {
  const region: Dims = [dims[0] - 2 * m[0], dims[1] - 2 * m[1], dims[2] - 2 * m[2]];
  return [Math.max(1, region[0] - w[0] + 1), Math.max(1, region[1] - w[1] + 1), Math.max(1, region[2] - w[2] + 1)];
}

export function scanPrep(inp: ScanInput): ScanPrep {
  const { vol, dims, query, qdims, window: w, margin: m } = inp;
  // statistics of the region itself (not its margin, which is zero past the array's edge),
  // as search.py takes them from the inner volume
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
    const [lo, hi] = percentile(inner, [0.5, 99.5]);
    return Array.from({ length: 17 }, (_, i) => lo + ((hi - lo) * i) / 16);
  })();
  // query histogram (centre window of the query context) and texture stats
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
  return { edges, qcdf, qstats };
}

/** One slab of the scan: the volume's voxels `lo`..`hi` along `axis` (0, or 1 for a 2-D image),
 * holding the windows whose corners are `c0`..`c1` of the `scanDims` along that axis, plus HALO
 * voxels each side where the volume has them, so every filter sees inside the slab what it would
 * see in the whole volume. Slabs cut the slowest axis that has more than one voxel, so each is one
 * contiguous run of the volume and its windows one run of the result. */
export interface ScanSlab { axis: 0 | 1; c0: number; c1: number; lo: number; hi: number }
/** The filters' reach: the Gaussian of sigma 2 at scipy's truncate 4 (the others reach less). */
export const HALO = 8;

/** Cut the scan into at most `k` slabs, each at least 4 windows thick. */
export function scanSlabs(dims: Dims, w: Dims, m: Dims, k: number): ScanSlab[] {
  const od = scanDims(dims, w, m), axis: 0 | 1 = dims[0] > 1 ? 0 : 1, n = od[axis];
  const parts = Math.max(1, Math.min(k, Math.floor(n / 4)));
  const out: ScanSlab[] = [];
  for (let p = 0; p < parts; p++) {
    const c0 = Math.floor((p * n) / parts), c1 = Math.floor(((p + 1) * n) / parts);
    out.push({ axis, c0, c1, lo: Math.max(0, m[axis] + c0 - HALO), hi: Math.min(dims[axis], m[axis] + c1 - 1 + w[axis] + HALO) });
  }
  return out;
}

/** A slab's own voxels (a copy) and their shape. */
export function slabVolume(vol: Float32Array, dims: Dims, s: ScanSlab): { vol: Float32Array; dims: Dims } {
  const plane = s.axis === 0 ? dims[1] * dims[2] : dims[2];
  const d: Dims = s.axis === 0 ? [s.hi - s.lo, dims[1], dims[2]] : [dims[0], s.hi - s.lo, dims[2]];
  return { vol: vol.slice(s.lo * plane, s.hi * plane), dims: d };
}

/** The raw maps of one slab's windows: texture mean, std, mean gradient and mean |DoG|, then the
 * histogram distance, each shaped like the result with `c1 - c0` along the slab's axis. */
export function scanBlock(vol: Float32Array, sd: Dims, s: ScanSlab, dims: Dims, w: Dims, m: Dims, prep: ScanPrep): Float32Array[] {
  const tex = textureStats(vol, sd, w);
  const hist = histMap(vol, sd, w, prep.qcdf, prep.edges);
  const td = tex.dims, maps = [...tex.maps, hist.data];
  const bd = scanDims(dims, w, m);
  bd[s.axis] = s.c1 - s.c0;
  const off: Dims = [m[0], m[1], m[2]];  // the slab's first window corner, in slab voxels
  off[s.axis] = m[s.axis] + s.c0 - s.lo;
  const out = maps.map(() => new Float32Array(bd[0] * bd[1] * bd[2]));
  let i = 0;
  for (let z = 0; z < bd[0]; z++)
    for (let y = 0; y < bd[1]; y++)
      for (let x = 0; x < bd[2]; x++, i++) {
        const j = ((z + off[0]) * td[1] + (y + off[1])) * td[2] + (x + off[2]);
        for (let f = 0; f < 5; f++) out[f][i] = maps[f][j];
      }
  return out;
}

/** The slabs' maps joined (in slab order) and scored: each texture distance scaled by its spread
 * over the windows, then the mean z-score of -EMD and -texture distance. */
export function scanMerge(parts: Float32Array[][], dims: Dims, w: Dims, m: Dims, prep: ScanPrep): { score: Float32Array; dims: Dims } {
  const od = scanDims(dims, w, m), n = od[0] * od[1] * od[2];
  const maps = [0, 1, 2, 3, 4].map(() => new Float32Array(n));
  let at = 0;
  for (const p of parts) {
    for (let f = 0; f < 5; f++) maps[f].set(p[f], at);
    at += p[0].length;
  }
  if (at !== n) throw new Error(`scan slabs cover ${at} of ${n} windows`);
  // per-feature spread over the windows inside the region (search.py: texture_stats(inner).std());
  // none fit when the region is smaller than a window on some axis
  const inside = [0, 1, 2].every((a) => dims[a] - 2 * m[a] >= w[a]);
  const spread = maps.slice(0, 4).map((mp) => {
    if (!inside) return 1e-9;
    let mu = 0, v = 0;
    for (let i = 0; i < n; i++) mu += mp[i];
    mu /= n;
    for (let i = 0; i < n; i++) v += (mp[i] - mu) ** 2;
    return Math.sqrt(v / n) + 1e-9;
  });
  const t = new Float32Array(n), { qstats } = prep;
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let f = 0; f < 4; f++) s += ((maps[f][i] - qstats[f]) / spread[f]) ** 2;
    t[i] = -Math.sqrt(s);
  }
  const zt = zscore(t), zh = zscore(maps[4]), score = new Float32Array(n);
  for (let k = 0; k < n; k++) score[k] = (zt[k] + zh[k]) / 2;
  return { score, dims: od };
}

/** Combined score (mean z-score of -EMD and -texture distance) of every window whose corner
 * lies in the region (the volume minus its margin); index i of the result is the window
 * whose corner is at margin + i. In one piece; the page runs the same steps on slabs in
 * several workers (scanpool.ts). */
export function scan(inp: ScanInput): { score: Float32Array; dims: Dims } {
  const { vol, dims, window: w, margin: m } = inp, prep = scanPrep(inp);
  const parts = scanSlabs(dims, w, m, 1).map((s) => {
    const sv = slabVolume(vol, dims, s);
    return scanBlock(sv.vol, sv.dims, s, dims, w, m, prep);
  });
  return scanMerge(parts, dims, w, m, prep);
}
