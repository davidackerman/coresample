// The hits as CSV. Run: npm test
import { test } from "node:test";
import assert from "node:assert/strict";
import { hitsCsv } from "../src/exporthits.ts";

test("a volume's hits: voxels, physical positions, marks and links; only the columns in use", () => {
  const csv = hitsCsv([
    { rank: 1, score: 0.91234, x: 100.4, y: 200, z: 30, link: "https://v/#!{\"a\":1,\"b\":2}", mark: "more" },
    { rank: 2, score: 0.5, x: 8, y: 16, z: 24 },
  ], 64, { scale: [8, 8, 4], unit: "nanometer" });
  assert.deepEqual(csv.split("\n"), [
    "rank,score,x,y,z,box,x_nanometer,y_nanometer,z_nanometer,mark,link",
    '1,0.9123,100,200,30,64,803.2,1600,120,more,"https://v/#!{""a"":1,""b"":2}"',
    "2,0.5000,8,16,24,64,64,128,96,,",
    "",
  ]);
});

test("a map's hits: no z, degrees", () => {
  const csv = hitsCsv([{ rank: 1, score: 0.7, x: 5, y: 6, lon: -77.03653, lat: 38.8977 }], 256);
  assert.deepEqual(csv.split("\n"), ["rank,score,x,y,box,lon,lat", "1,0.7000,5,6,256,-77.036530,38.897700", ""]);
});
