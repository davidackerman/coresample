// The hits as a file to take elsewhere: a CSV, one row a hit, best first.

export interface HitRow {
  rank: number; score: number;
  x: number; y: number; z?: number;   // the box's centre, in voxels (pixels) of the finest level, as the viewer counts them
  lon?: number; lat?: number;         // on a body or the sky (degrees; the sky: right ascension, declination)
  link?: string | null;               // the data's own viewer there
  mark?: string;                      // "more" / "fewer": marked by hand
}

const cell = (v: unknown) => {
  const s = v === undefined || v === null ? "" : String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

/** The rows as CSV. `box`: the box's side in the same voxels; `phys`: the size of one voxel
 * (x, y, z) and its unit, for the physical columns; columns no row has are left out. */
export function hitsCsv(rows: HitRow[], box: number, phys?: { scale: [number, number, number]; unit: string }): string {
  const has = (k: keyof HitRow) => rows.some((r) => r[k] !== undefined && r[k] !== null && r[k] !== "");
  const d3 = has("z"), u = phys?.unit ? phys.unit.replace(/[^\w]/g, "") : "";
  const cols: [string, (r: HitRow) => unknown][] = [["rank", (r) => r.rank], ["score", (r) => r.score.toFixed(4)], ["x", (r) => Math.round(r.x)], ["y", (r) => Math.round(r.y)]];
  if (d3) cols.push(["z", (r) => (r.z === undefined ? "" : Math.round(r.z))]);
  cols.push(["box", () => box]);
  if (phys && u) {
    const p = (v: number) => Number(v.toPrecision(8));
    cols.push([`x_${u}`, (r) => p(r.x * phys.scale[0])], [`y_${u}`, (r) => p(r.y * phys.scale[1])]);
    if (d3) cols.push([`z_${u}`, (r) => (r.z === undefined ? "" : p(r.z * phys.scale[2]))]);
  }
  if (has("lon")) cols.push(["lon", (r) => r.lon?.toFixed(6)], ["lat", (r) => r.lat?.toFixed(6)]);
  if (has("mark")) cols.push(["mark", (r) => r.mark]);
  if (has("link")) cols.push(["link", (r) => r.link]);
  return [cols.map((c) => c[0]).join(","), ...rows.map((r) => cols.map((c) => cell(c[1](r))).join(","))].join("\n") + "\n";
}
