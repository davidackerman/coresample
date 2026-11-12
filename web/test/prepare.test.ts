// The model's pixels (embedder.ts prepare): the table-driven resize gives what the per-pixel
// bilinear resize it replaced gave, for every way a model takes its input. Run: npm test
import { test } from "node:test";
import assert from "node:assert/strict";
import { prepare, type Patch, type Prep } from "../src/embedder.ts";

const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
/** The resize as it was: each output pixel sampled and normalized on its own. */
function reference(p: Patch, out: Float32Array, INPUT: number, how: Prep) {
  const s = p.size, C = p.channels, k = s / INPUT, plane = INPUT * INPUT, norm = how.norm ?? "imagenet";
  let src = p.data, stride = C;
  if (how.grey && C === 3) { src = new Float32Array(s * s); for (let i = 0; i < s * s; i++) src[i] = (p.data[3 * i] + p.data[3 * i + 1] + p.data[3 * i + 2]) / 3; stride = 1; }
  const nc = how.grey ? 1 : 3;
  for (let y = 0; y < INPUT; y++) {
    const fy = Math.min(s - 1, Math.max(0, (y + 0.5) * k - 0.5)), y0 = Math.floor(fy), y1 = Math.min(s - 1, y0 + 1), wy = fy - y0;
    for (let x = 0; x < INPUT; x++) {
      const fx = Math.min(s - 1, Math.max(0, (x + 0.5) * k - 0.5)), x0 = Math.floor(fx), x1 = Math.min(s - 1, x0 + 1), wx = fx - x0;
      for (let c = 0; c < nc; c++) {
        const cc = stride === 3 ? c : 0;
        const v = (1 - wy) * ((1 - wx) * src[(y0 * s + x0) * stride + cc] + wx * src[(y0 * s + x1) * stride + cc])
                + wy * ((1 - wx) * src[(y1 * s + x0) * stride + cc] + wx * src[(y1 * s + x1) * stride + cc]);
        out[c * plane + y * INPUT + x] = norm === "imagenet" ? (v / 255 - MEAN[c]) / STD[c] : norm === "half" ? (v / 255 - 0.5) / 0.5 : v / 255;
      }
    }
  }
  if (norm === "standardize") {
    let m = 0, q = 0;
    for (let i = 0; i < plane; i++) m += out[i];
    m /= plane;
    for (let i = 0; i < plane; i++) q += (out[i] - m) ** 2;
    const sd = Math.sqrt(q / (plane - 1)) + 5e-7;
    for (let i = 0; i < plane; i++) out[i] = (out[i] - m) / sd;
  }
}

test("the same pixels as the per-pixel resize: grey and colour, every normalization, up and down", () => {
  for (const C of [1, 3]) for (const size of [32, 64, 128, 300]) for (const INPUT of [112, 224])
    for (const how of [{ norm: "imagenet" }, { norm: "half" }, { norm: "standardize", grey: true }] as Prep[]) {
      const d = new Float32Array(size * size * C);
      for (let i = 0; i < d.length; i++) d[i] = (i * 31 + (i >> 7) * 17) % 256;
      const p = { data: d, size, channels: C as 1 | 3 }, n = (how.grey ? 1 : 3) * INPUT * INPUT;
      const a = new Float32Array(n), b = new Float32Array(n + 5);
      reference(p, a, INPUT, how);
      prepare(p, b, 5, INPUT, how);  // at an offset, as in a batch
      let worst = 0;
      for (let i = 0; i < n; i++) worst = Math.max(worst, Math.abs(a[i] - b[5 + i]));
      assert.ok(worst < 1e-5, `C=${C} ${size}->${INPUT} ${JSON.stringify(how)}: off by ${worst}`);
    }
});
