// The drawing pad: draw what to look for, as one box at the search level, in the data's own
// tones (its dark, middle and light greys, or colours, taken from the view). The drawing becomes the query
// (Search's `sketch`): the model embeds it as it would a patch of the data, so it can be
// softened and given the data's grain first ("look like the data").
import type { Patch } from "./embedder.ts";

export type RGB = [number, number, number];
export interface Tones { dark: RGB; mid: RGB; light: RGB; noise: number }
export type Tone = "dark" | "mid" | "light";

/** The data's tones in a 2-D image (w x h, `channels` interleaved): the mean colour of the
 * pixels around the 5th, 50th and 95th percentiles of brightness, and the grain (standard
 * deviation of the brightness minus a 3 x 3 blur of it). */
export function tonesOf(px: Float32Array, channels: number, w: number, h: number): Tones {
  const n = w * h, lum = new Float32Array(n);
  for (let i = 0; i < n; i++) lum[i] = channels === 3 ? 0.299 * px[3 * i] + 0.587 * px[3 * i + 1] + 0.114 * px[3 * i + 2] : px[i];
  const idx = Array.from({ length: n }, (_, i) => i).filter((i) => lum[i] === lum[i]).sort((a, b) => lum[a] - lum[b]);
  const q = (f: number): RGB => {  // mean colour of the 4% of pixels around this percentile
    if (!idx.length) return [128, 128, 128];
    const a = Math.max(0, Math.floor((f - 0.02) * idx.length)), b = Math.min(idx.length, Math.max(a + 1, Math.ceil((f + 0.02) * idx.length)));
    const m: RGB = [0, 0, 0];
    for (let k = a; k < b; k++) for (let c = 0; c < 3; c++) m[c] += px[idx[k] * channels + (channels === 3 ? c : 0)] / (b - a);
    return m;
  };
  let s = 0, s2 = 0, m = 0;
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    let b = 0, ok = true;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const t = lum[(y + dy) * w + x + dx]; if (t !== t) ok = false; b += t; }
    if (!ok) continue;
    const r = lum[y * w + x] - b / 9;
    s += r; s2 += r * r; m++;
  }
  const noise = m > 1 ? Math.sqrt(Math.max(0, s2 / m - (s / m) ** 2)) : 0;
  return { dark: q(0.05), mid: q(0.5), light: q(0.95), noise };
}

export class SketchPad {
  P = 64;
  tone: Tone = "dark";
  radius = 2;  // brush radius in pixels of a 64 px box (scaled with the box)
  tones: Tones = { dark: [40, 40, 40], mid: [128, 128, 128], light: [215, 215, 215], noise: 0 };
  private g: CanvasRenderingContext2D;
  private last: [number, number] | null = null;

  constructor(private cv: HTMLCanvasElement) {
    this.g = cv.getContext("2d", { willReadFrequently: true })!;
    cv.addEventListener("pointerdown", (e) => { cv.setPointerCapture(e.pointerId); this.last = this.at(e); this.stroke(this.last, this.last); });
    cv.addEventListener("pointermove", (e) => { if (!this.last) return; const p = this.at(e); this.stroke(this.last, p); this.last = p; });
    const end = () => { this.last = null; };
    cv.addEventListener("pointerup", end);
    cv.addEventListener("pointercancel", end);
  }

  private at(e: PointerEvent): [number, number] {
    const r = this.cv.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * this.P, ((e.clientY - r.top) / r.height) * this.P];
  }
  private grey(t: Tone) { const [r, g, b] = this.tones[t].map(Math.round); return `rgb(${r},${g},${b})`; }
  /** The pens' colours (for the buttons). */
  colour(t: Tone) { return this.grey(t); }
  private stroke(a: [number, number], b: [number, number]) {
    const g = this.g, w = Math.max(1, (2 * this.radius * this.P) / 64);
    g.strokeStyle = g.fillStyle = this.grey(this.tone);
    g.lineWidth = w; g.lineCap = "round"; g.lineJoin = "round";
    g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke();
    g.beginPath(); g.arc(b[0], b[1], w / 2, 0, 7); g.fill();
  }

  /** A blank box of side P in the middle tone. */
  reset(P: number, tones: Tones) {
    this.P = P; this.tones = tones;
    this.cv.width = P; this.cv.height = P;
    this.g.fillStyle = this.grey("mid");
    this.g.fillRect(0, 0, P, P);
  }
  /** A P x P image (`channels` interleaved) to draw over. */
  load(px: Float32Array, channels: number) {
    const P = this.P, img = new ImageData(P, P);
    for (let i = 0; i < P * P; i++) for (let c = 0; c < 3; c++) {
      const v = px[i * channels + (channels === 3 ? c : 0)];
      img.data[4 * i + c] = v === v ? v : this.tones.mid[c];
      img.data[4 * i + 3] = 255;
    }
    this.g.putImageData(img, 0, 0);
  }

  /** The drawing as a patch of `channels` channels; with `texture`, softened (a 3 x 3 blur)
   * and given the data's grain, as the data looks. */
  patch(channels: number, texture: boolean): Patch {
    const P = this.P, rgba = this.g.getImageData(0, 0, P, P).data, d = new Float32Array(P * P * channels);
    for (let c = 0; c < channels; c++) {
      let v = new Float32Array(P * P);
      for (let i = 0; i < P * P; i++) v[i] = rgba[4 * i + c];
      if (texture) {
        const k = [1, 2, 1], b = new Float32Array(P * P);
        for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) {
          let s = 0, n = 0;
          for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
            const yy = y + dy, xx = x + dx;
            if (yy < 0 || xx < 0 || yy >= P || xx >= P) continue;
            const w = k[dy + 1] * k[dx + 1];
            s += w * v[yy * P + xx]; n += w;
          }
          // grain: Gaussian noise (Box-Muller) at the data's level
          const u = 1 - Math.random(), r = Math.random();
          b[y * P + x] = s / n + this.tones.noise * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r);
        }
        v = b;
      }
      for (let i = 0; i < P * P; i++) d[i * channels + c] = Math.max(0, Math.min(255, v[i]));
    }
    return { data: d, size: P, channels };
  }
}
