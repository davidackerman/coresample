// Writes the TS scan of the s4 whole volume for the e2e query, for comparison with search.py.
import { writeFileSync } from "node:fs";
import { scan } from "../src/scan.ts";
import { luminance, OmeZarrSource, type Vec3 } from "../src/sources.ts";
const src = await OmeZarrSource.open("s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8");
const C = 4, f = 4, m = 4, w = 16, q: Vec3 = [666, 139, 2244];
const sh = src.levels[C].shape;
const vol = await src.read(C, [-m, -m, -m], [sh[0] + m, sh[1] + m, sh[2] + m]);
const qc = q.map((v) => Math.floor(v / f));
const qlo = qc.map((v) => v - w / 2 - m) as Vec3;
const qr = await src.read(C, qlo, [qlo[0] + w + 2 * m, qlo[1] + w + 2 * m, qlo[2] + w + 2 * m]);
const t = performance.now();
const r = scan({ vol: luminance(vol), dims: vol.shape, query: luminance(qr), qdims: qr.shape, window: [w, w, w], margin: [m, m, m] });
console.log("scan", r.dims, ((performance.now() - t) / 1000).toFixed(1) + "s");
writeFileSync("test/scan_ts.bin", Buffer.from(r.score.buffer));
writeFileSync("test/scan_ts.json", JSON.stringify({ dims: r.dims }));
