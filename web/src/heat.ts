// The similarity heat map: every place the search has compared, coloured by how well it
// matched, as similar.earth and Earth Engine's similarity search show it. It is drawn from the
// search's compact list of places and scores (Search.heat()): into the map of where it has
// looked (largest score along the hidden axis), and over a 2-D image itself.
import type { Heat } from "./search.ts";
import type { Vec3 } from "./sources.ts";

/** One hue (magenta), dark for the lowest score shown and light for the best: its own
 * sequential scale, apart from the page's blue (query, tiles read) and yellow (hits). */
const STOPS: [number, number, number, number][] = [[0, 0x4a, 0x15, 0x30], [0.5, 0xd5, 0x51, 0x81], [1, 0xfb, 0xd3, 0xe3]];
const LUT = (() => {
  const t = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const v = i / 255, k = v < STOPS[1][0] ? 0 : 1, [a, b] = [STOPS[k], STOPS[k + 1]], f = (v - a[0]) / (b[0] - a[0]);
    for (let c = 0; c < 3; c++) t[3 * i + c] = a[c + 1] + f * (b[c + 1] - a[c + 1]);
  }
  return t;
})();
/** The ramp's colour at `t` (0-1). */
export function heatColor(t: number): [number, number, number] {
  const j = Math.round(Math.max(0, Math.min(1, t)) * 255) * 3;
  return [LUT[j], LUT[j + 1], LUT[j + 2]];
}
/** The ramp as a CSS gradient (for the legend). */
export const HEAT_GRADIENT = `linear-gradient(90deg, ${STOPS.map(([p, r, g, b]) => `rgb(${r},${g},${b}) ${p * 100}%`).join(", ")})`;

/** The scores shown: from the score that leaves the best `top` fraction above it, to the best. */
export function heatRange(h: Heat, top: number): [number, number] {
  const step = Math.max(1, Math.floor(h.n / 20000)), v: number[] = [];
  for (let i = 0; i < h.n; i += step) if (h.s[i] > -Infinity) v.push(h.s[i]);
  let hi = -Infinity;
  for (let i = 0; i < h.n; i++) if (h.s[i] > hi) hi = h.s[i];
  if (!v.length) return [0, 1];
  v.sort((a, b) => a - b);
  const lo = v[Math.min(v.length - 1, Math.floor((1 - top) * (v.length - 1)))];
  return [Math.min(lo, hi - 1e-6), hi];
}

/** The best score in each cell of a grid laid over two axes (`a` across, `b` down, in working
 * voxels from `origin`, `scaleA`/`scaleB` cells per voxel); a place covers `foot` voxels along
 * each axis around its centre. -Infinity where nothing was compared. */
export function heatGrid(h: Heat, w: number, ht: number, a: number, b: number, origin: Vec3,
                         scaleA: number, scaleB: number, foot: number): Float32Array {
  const g = new Float32Array(w * ht).fill(-Infinity), half = foot / 2;
  for (let i = 0; i < h.n; i++) {
    const s = h.s[i];
    if (!(s > -Infinity)) continue;
    const va = h.c[3 * i + a] - origin[a], vb = h.c[3 * i + b] - origin[b];
    const x0 = Math.max(0, Math.floor((va - half) * scaleA)), x1 = Math.min(w, Math.max(x0 + 1, Math.ceil((va + half) * scaleA)));
    const y0 = Math.max(0, Math.floor((vb - half) * scaleB)), y1 = Math.min(ht, Math.max(y0 + 1, Math.ceil((vb + half) * scaleB)));
    for (let y = y0; y < y1; y++) for (let x = x0, o = y * w; x < x1; x++) if (s > g[o + x]) g[o + x] = s;
  }
  return g;
}

/** A canvas of the grid's cells coloured by score (below `lo` left clear): faint for low
 * scores, solid for the best. */
export function heatCanvas(g: Float32Array, w: number, ht: number, lo: number, hi: number, cv = document.createElement("canvas")): HTMLCanvasElement {
  cv.width = w; cv.height = ht;
  const img = new ImageData(w, ht), d = img.data, k = 1 / Math.max(1e-9, hi - lo);
  for (let i = 0; i < w * ht; i++) {
    const s = g[i];
    if (!(s >= lo)) continue;
    const t = Math.min(1, (s - lo) * k), j = Math.round(t * 255) * 3;
    d[4 * i] = LUT[j]; d[4 * i + 1] = LUT[j + 1]; d[4 * i + 2] = LUT[j + 2]; d[4 * i + 3] = Math.round(255 * (0.35 + 0.65 * t));
  }
  cv.getContext("2d")!.putImageData(img, 0, 0);
  return cv;
}
