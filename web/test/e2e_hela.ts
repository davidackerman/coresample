// End-to-end check of the page's search core in Node (no DOM): real jrc_hela-2 data from S3,
// DINOv2 through chunkmirage's ONNX wrapper (WASM in Node). Writes the hits for grading.
// usage: node --experimental-strip-types test/e2e_hela.ts [query z,y,x] [tiles] [coarse voxels] [patches|tokens]
import { readFileSync, writeFileSync } from "node:fs";
import { Embedder } from "../src/embedder.ts";
import { Search } from "../src/search.ts";
import { OmeZarrSource, type Vec3 } from "../src/sources.ts";

const url = "s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8";
const query: Vec3 = (process.argv[2] ?? "666,139,2244").split(",").map(Number) as Vec3;  // working-level (s2) voxel
const tiles = Number(process.argv[3] ?? 24);  // tiles read before stopping (the page reads until paused)
const t = performance.now();
const src = await OmeZarrSource.open(url, "jrc_hela-2 EM");
console.log(`opened ${src.levels.length} levels in ${((performance.now() - t) / 1000).toFixed(1)}s; s2 shape ${src.levels[2].shape} chunk ${src.levels[2].chunk}`);
const method = (process.argv[5] ?? "patches") as "patches" | "tokens";
const roughPass = process.argv[6] === "rough" ? true : process.argv[6] === "norough" ? false : undefined;
const emb = new Embedder((file) => new URL(`../public/models/${file}.onnx`, import.meta.url).pathname, undefined,
                         async (path) => readFileSync(path).buffer as ArrayBuffer);
const s = new Search(src, emb, query, { working: 2, patch: 64, method, roughPass, maxCoarseVoxels: Number(process.argv[4] ?? 16e6) });
let last = "";
s.on((st) => {
  const line = `${st.phase} coarse=s${st.coarse}${st.rough ? ` rough=s${st.rough.level}:${st.rough.places}` : ""} tiles ${st.tilesRead}/${st.tilesBudget} of ${st.tilesTotal} patches ${st.patches} hits ${st.hits.length} ` +
    `MB ${(st.bytes / 1e6).toFixed(1)} t=${st.seconds.toFixed(1)}s scan=${st.scanSeconds.toFixed(1)}s`;
  if (line.split(" t=")[0] !== last) { console.log(line); last = line.split(" t=")[0]; }
  if (st.tilesRead >= tiles) s.stop();
});
await s.start();
console.log(`provider ${emb.provider}, ${emb.count} patches embedded in ${emb.seconds.toFixed(1)}s`);
writeFileSync(`test/e2e_hela_hits${method === "tokens" ? "_tokens" : ""}.json`, JSON.stringify({ query, working: 2, patch: 64, method, state: s.state }));
console.log("top 5", s.state.hits.slice(0, 5).map((h) => `${h.center} ${h.score.toFixed(3)}`));
