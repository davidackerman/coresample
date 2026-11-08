// The 224 px model now outputs raw tokens and the page pools them (embedder.ts poolTokens): the
// pooled embedding must equal what the old pooled export gave. usage:
//   node --experimental-strip-types test/token_pool_parity.ts <old pooled 224 onnx>
import { readFileSync } from "node:fs";
import { Onnx } from "../../../chunkmirage/web/src/onnx.ts";
import { DIM, poolTokens } from "../src/embedder.ts";

const oldPath = process.argv[2];
const onnx = new Onnx({});
const model = (path: string) => ({ key: path, bytes: async () => readFileSync(path).buffer as ArrayBuffer });
const n = 3, x = new Float32Array(n * 3 * 224 * 224).map(() => Math.random() * 4 - 2);
const tok = await onnx.run(model(new URL("../public/models/dinov2-small-224.onnx", import.meta.url).pathname), x, [n, 3, 224, 224]);
const old = await onnx.run(model(oldPath), x, [n, 3, 224, 224]);
let worst = 0;
for (let j = 0; j < n; j++) {
  const e = poolTokens(tok.data as Float32Array, j, tok.dims[1], tok.dims[2]);
  for (let d = 0; d < DIM; d++) worst = Math.max(worst, Math.abs(e[d] - (old.data as Float32Array)[j * DIM + d]));
}
console.log(`tokens ${tok.dims}, pooled vs old export: max |diff| = ${worst.toExponential(2)}`);
if (worst > 1e-4) process.exit(1);
