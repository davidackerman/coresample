// The heat map's grid (src/heat.ts): each place paints the cells it covers, and only those.
// Run: npm test
import { test } from "node:test";
import assert from "node:assert/strict";
import { heatGrid } from "../src/heat.ts";

test("places off the grid's left or top edge paint nothing (no stripe along the edges)", () => {
  // three places of footprint 4: one inside, one left of the grid, one above it
  const c = new Int32Array([0, 10, 10, 0, 10, -20, 0, -20, 10]), s = new Float32Array([0.9, 0.5, 0.7]);
  const g = heatGrid({ c, s, n: 3, version: 0 }, 32, 32, 2, 1, [0, 0, 0], 1, 1, 4);
  const painted = [...g.keys()].filter((i) => g[i] > -Infinity);
  assert.deepEqual(painted.map((i) => [Math.floor(i / 32), i % 32]), [[8, 8], [8, 9], [8, 10], [8, 11], [9, 8], [9, 9], [9, 10], [9, 11],
    [10, 8], [10, 9], [10, 10], [10, 11], [11, 8], [11, 9], [11, 10], [11, 11]]);
});
