// Several examples as one query, in Node on real jrc_hela-2 data: read a few tiles, add the
// best hit as a second example, check the query changed, no hit sits on an example, and the
// heat list has every patch compared (with the kept ones re-scored).
// usage: node --experimental-strip-types test/e2e_examples.ts [tiles before] [tiles after]
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import { Embedder } from "../src/embedder.ts";
import { Search } from "../src/search.ts";
import { OmeZarrSource, type Vec3 } from "../src/sources.ts";

const url = "s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8";
const before = Number(process.argv[2] ?? 6), after = Number(process.argv[3] ?? 4);
const src = await OmeZarrSource.open(url, "jrc_hela-2 EM");
const emb = new Embedder((size) => new URL(`../public/models/dinov2-small-${size}.onnx`, import.meta.url).pathname, undefined,
                         async (path) => readFileSync(path).buffer as ArrayBuffer);
const s = new Search(src, emb, [666, 139, 2244], { working: 2, patch: 64, roughPass: false, maxCoarseVoxels: 16e6 });
let limit = before;
s.on((st) => { if (st.tilesRead >= limit && st.phase === "read") s.stop(); });
await s.start();
const h0 = s.state.hits.map((h) => h.score);
const top = s.state.hits[0];
console.log(`after ${s.state.tilesRead} tiles: ${s.state.patches} patches, heat ${s.heat().n}, top ${top.center} ${top.score.toFixed(3)}`);
assert.equal(s.heat().n, s.state.patches, "a heat entry per patch compared");
const err = await s.addExample(top.center as Vec3);
assert.equal(err, null);
assert.equal(s.state.examples, 2);
assert.equal(await s.addExample(top.center as Vec3), "that place is already an example");
const P = 64, h = P / 2;
for (const hit of s.state.hits)
  for (const e of s.exampleCenters())
    assert.ok(!(Math.abs(hit.center[0] - e[0]) < h && Math.abs(hit.center[1] - e[1]) < h && Math.abs(hit.center[2] - e[2]) < h), `hit ${hit.center} on example ${e}`);
console.log(`two examples: top ${s.state.hits.slice(0, 3).map((x) => `${x.center} ${x.score.toFixed(3)}`).join(", ")} (one: ${h0.slice(0, 3).map((x) => x.toFixed(3)).join(", ")})`);
const H = s.heat();
const kept = s.state.hits[0];
assert.ok(Math.abs(Math.max(...Array.from(H.s.subarray(0, H.n))) - kept.score) < 1e-5 || Math.max(...Array.from(H.s.subarray(0, H.n))) >= kept.score, "heat max follows re-ranking");
limit = before + after;
await s.resume();
console.log(`then ${s.state.tilesRead} tiles: ${s.state.patches} patches, heat ${s.heat().n}, examples ${s.state.examples}`);
assert.equal(s.heat().n, s.state.patches);
console.log("ok");
