// The cheap scan off the page's main thread, in several workers (one per spare core, up to 4,
// as decode.ts): the coarse volume is cut into slabs along its slowest axis (scan.ts's
// scanSlabs, each with the filters' border), each worker scores its slab's windows, and one of
// them joins the slabs. In one worker the scan was the largest single part of the wait for the
// first hit (4.5-5.7 s on a 4-million-voxel coarse level, RTX 2080 Ti workstation).
import { scanSlabs, slabVolume, type Dims, type ScanInput, type ScanNorm, type ScanPrep } from "./scan.ts";

type Reply = { id: number; error?: string; [k: string]: unknown };
let pool: Worker[] | null = null, ids = 0;
const jobs = new Map<number, { resolve: (r: Reply) => void; reject: (e: Error) => void }>();

function workers(): Worker[] {
  if (pool) return pool;
  const n = Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 2) - 1));
  pool = Array.from({ length: n }, () => {
    const w = new Worker(new URL("./scan.worker.ts", import.meta.url), { type: "module" });
    w.onmessage = (e: MessageEvent<Reply>) => {
      const j = jobs.get(e.data.id);
      jobs.delete(e.data.id);
      if (j) e.data.error ? j.reject(new Error(e.data.error)) : j.resolve(e.data);
    };
    return w;
  });
  return pool;
}

function call(w: Worker, msg: object, transfer: Transferable[] = []): Promise<Reply> {
  return new Promise((resolve, reject) => {
    const id = ++ids;
    jobs.set(id, { resolve, reject });
    w.postMessage({ id, ...msg }, transfer);
  });
}

/** Scores of every window of the coarse volume against the query (scan.ts's `scan`), computed
 * in slabs across the worker pool. */
export async function scanInWorkers(inp: ScanInput): Promise<{ score: Float32Array; dims: Dims; norm: ScanNorm }> {
  const ws = workers(), { vol, dims, window: w, margin: m } = inp;
  const slabs = scanSlabs(dims, w, m, ws.length);
  if (slabs.length === 1) {  // too thin to cut: the whole scan in one worker
    const v = Float32Array.from(vol), q = Float32Array.from(inp.query);
    const r = await call(ws[0], { kind: "scan", inp: { ...inp, vol: v, query: q } }, [v.buffer, q.buffer]);
    return { score: r.score as Float32Array, dims: r.dims as Dims, norm: r.norm as ScanNorm };
  }
  // the histogram's bins come from the whole region, so they are found first (a copy of the
  // volume), unless an earlier area's scale is given
  const prep = inp.norm?.prep ?? ((await call(ws[0], { kind: "prep", inp })).prep as ScanPrep);
  const parts = await Promise.all(slabs.map(async (s, i) => {
    const sv = slabVolume(vol, dims, s);
    const r = await call(ws[i % ws.length], { kind: "block", vol: sv.vol, sd: sv.dims, slab: s, dims, w, m, prep }, [sv.vol.buffer]);
    return r.maps as Float32Array[];
  }));
  const r = await call(ws[0], { kind: "merge", parts, dims, w, m, prep, norm: inp.norm }, parts.flat().map((a) => a.buffer));
  return { score: r.score as Float32Array, dims: r.dims as Dims, norm: r.norm as ScanNorm };
}

// started with the page, as the one scan worker was, so the first search does not wait for them
if (typeof Worker !== "undefined" && typeof navigator !== "undefined") workers();
