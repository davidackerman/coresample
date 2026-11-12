// Image decoding off the page's main thread: a few workers (one per spare core, up to 4) turn
// image files (map tiles, JPEG volume chunks) into pixels with the browser's own decoder. On a
// big search the main thread was 78% busy and a third of that was this.
export interface Decoded { data: Uint8ClampedArray | Uint8Array; width: number; height: number }

let pool: Worker[] | null | undefined, turn = 0, ids = 0;
const jobs = new Map<number, { resolve: (d: Decoded) => void; reject: (e: Error) => void; w: Worker }>();

function workers(): Worker[] | null {
  if (pool !== undefined) return pool;
  if (typeof Worker === "undefined" || typeof OffscreenCanvas === "undefined" || typeof navigator === "undefined") return (pool = null);
  try {
    const n = Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 2) - 1));
    pool = Array.from({ length: n }, () => {
      const w = new Worker(new URL("./decode.worker.ts", import.meta.url), { type: "module" });
      w.onmessage = (e) => {
        const j = jobs.get(e.data.id);
        jobs.delete(e.data.id);
        if (j) e.data.error ? j.reject(new Error(e.data.error)) : j.resolve(e.data);
      };
      // a worker that could not start (its script failed to load) would never answer: its images
      // fail now, and the ones after decode on the page instead
      w.onerror = (e) => {
        e.preventDefault();
        pool = null;
        for (const [id, j] of jobs) if (j.w === w) { jobs.delete(id); j.reject(new Error("the image decoder stopped")); }
      };
      return w;
    });
  } catch { pool = null; }
  return pool;
}

/** The pixels of an image file, decoded in a worker: RGBA, or with `planes` that many channels
 * as separate planes, or with `grey` one plane of luminance; `raw` skips colour management
 * (data, not photos). Null where there are no workers (Node): decode on the spot instead. An
 * ArrayBuffer is handed over (detached here). */
export function decodeOff(bytes: ArrayBuffer | Blob, type: string, opts: { planes?: number; raw?: boolean; grey?: boolean } = {}): Promise<Decoded> | null {
  const ws = workers();
  if (!ws) return null;
  return new Promise((resolve, reject) => {
    const id = ids++, w = ws[turn++ % ws.length];
    jobs.set(id, { resolve, reject, w });
    w.postMessage({ id, bytes, type, planes: opts.planes, raw: !!opts.raw, grey: !!opts.grey }, bytes instanceof ArrayBuffer ? [bytes] : []);
  });
}
