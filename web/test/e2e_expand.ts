// Searching outward, in Node on real jrc_hela-2 data: a small first area, `expand` on; read
// until three areas have been scanned, then check every tile was read once and the reading
// went past the first area (with the rough pass: its picks there). usage: node --experimental-strip-types test/e2e_expand.ts [rough|norough]
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import { Embedder } from "../src/embedder.ts";
import { Search } from "../src/search.ts";
import { OmeZarrSource, type Vec3 } from "../src/sources.ts";

const url = "s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8";
const src = await OmeZarrSource.open(url, "jrc_hela-2 EM");
const emb = new Embedder((size) => new URL(`../public/models/dinov2-small-${size}.onnx`, import.meta.url).pathname, undefined,
                         async (path) => readFileSync(path).buffer as ArrayBuffer);
const q: Vec3 = [666, 139, 2244];
const region: [Vec3, Vec3] = [[602, 64, 2116], [730, 256, 2372]];  // a small box (2 x 3 x 4 tiles): the next areas are around it
const s = new Search(src, emb, q, { working: 2, patch: 64, region, expand: true, roughPass: process.argv[2] === "rough", maxCoarseVoxels: 16e6 });
const ch = src.levels[2].chunk;
const outsideFirst = (t: Vec3) => [0, 1, 2].some((a) => (t[a] + 1) * ch[a] <= region[0][a] || t[a] * ch[a] >= region[1][a]);
let last = 0;
s.on((st) => {
  if (st.areas !== last) { console.log(`areas ${st.areas}: tiles ${st.tilesRead}/${st.tilesTotal} of ${st.levelTiles}, area x ${s.regionBox[0][2]}-${s.regionBox[1][2]}`); last = st.areas; }
  // without the rough pass every tile is read; with it only the best tenth of those it looked over
  if (st.areas >= 3 && st.phase === "read" && (st.rough ? s.readTiles().some(outsideFirst) : st.tilesRead >= st.tilesTotal - 2)) s.stop();
});
await s.start();
const read = s.readTiles().map((t) => t.join(","));
assert.equal(new Set(read).size, read.length, "a tile read twice");
const [alo, ahi] = s.regionBox;
console.log(`read ${read.length} tiles; areas scanned cover ${alo} to ${ahi} (the first ${region[0]} to ${region[1]}); phase ${s.state.phase}`);
assert.ok([0, 1, 2].some((a) => alo[a] < region[0][a] || ahi[a] > region[1][a]), "never went past the first area");
const outside = s.readTiles().filter(outsideFirst);
console.log(`${outside.length} tiles read wholly outside the first area`);
assert.ok(outside.length > 0, "no tile read outside the first area");
assert.ok(s.state.areas >= 3);
console.log("hits from x", s.state.hits.slice(0, 8).map((h) => h.center[2]).join(" "));
console.log("ok");
