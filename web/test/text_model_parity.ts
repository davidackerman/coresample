// Manual: the exported PE-Core-S text and image models in ONNX Runtime Web (WebAssembly, as the
// page runs them without a GPU) against PyTorch, the text through the page's tokenizer.
//   node --experimental-strip-types test/text_model_parity.ts <reference.json>
// reference.json: { texts, text: [[512]...], image: [512] } from open_clip (the image: one
// 3 x 112 x 112 input of torch.linspace(-1, 1, n)). Needs the models exported (npm run model).
import { readFileSync } from "node:fs";
import * as ort from "onnxruntime-web";
import { CONTEXT, ClipTokenizer } from "../src/cliptok.ts";

const ref = JSON.parse(readFileSync(process.argv[2], "utf8")) as { texts: string[]; text: number[][]; image: number[] };
const file = (f: string) => new Uint8Array(readFileSync(new URL(`../public/models/${f}`, import.meta.url)));
const joined = (fs: string[]) => new Uint8Array(Buffer.concat(fs.map((f) => file(f))));  // as the page joins the parts
const tok = new ClipTokenizer(readFileSync(new URL("../public/models/clip-bpe-merges.txt", import.meta.url), "utf8"));
const cos = (a: ArrayLike<number>, b: ArrayLike<number>) => { let s = 0, na = 0, nb = 0; for (let i = 0; i < a.length; i++) { s += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; } return s / Math.sqrt(na * nb); };
const text = await ort.InferenceSession.create(joined(["pecore-s16-text.part0.onnx", "pecore-s16-text.part1.onnx"]), { executionProviders: ["wasm"] });
let worst = 1;
for (const [i, t] of ref.texts.entries()) {
  const out = await text.run({ [text.inputNames[0]]: new ort.Tensor("float32", tok.encode(t), [1, CONTEXT]) });
  const c = cos(out[text.outputNames[0]].data as Float32Array, ref.text[i]);
  worst = Math.min(worst, c);
  console.log(`text ${JSON.stringify(t)}: cosine to PyTorch ${c.toFixed(6)}`);
}
const image = await ort.InferenceSession.create(file("pecore-s16-112-clip.onnx"), { executionProviders: ["wasm"] });
const n = 3 * 112 * 112, x = Float32Array.from({ length: n }, (_, i) => -1 + (2 * i) / (n - 1));
const io = await image.run({ [image.inputNames[0]]: new ort.Tensor("float32", x, [1, 3, 112, 112]) });
const ci = cos(io[image.outputNames[0]].data as Float32Array, ref.image);
console.log(`image: cosine to PyTorch ${ci.toFixed(6)}`);
if (Math.min(worst, ci) < 0.999) { console.error("FAILED: below 0.999"); process.exit(1); }
console.log("ok");
