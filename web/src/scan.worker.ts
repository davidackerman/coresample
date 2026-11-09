// The cheap scan off the page's main thread: the whole scan, or one step of it cut into slabs
// (scanpool.ts): the histogram bins from the whole region, one slab's raw maps, or the slabs joined.
import { scan, scanPrep, scanBlock, scanMerge } from "./scan.ts";

const post = (msg: object, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(msg, transfer);

self.onmessage = (e: MessageEvent) => {
  const { id, kind } = e.data;
  try {
    if (kind === "prep") post({ id, prep: scanPrep(e.data.inp) });
    else if (kind === "block") {
      const { vol, sd, slab, dims, w, m, prep } = e.data;
      const maps = scanBlock(vol, sd, slab, dims, w, m, prep);
      post({ id, maps }, maps.map((a) => a.buffer));
    } else if (kind === "merge") {
      const { parts, dims, w, m, prep } = e.data;
      const r = scanMerge(parts, dims, w, m, prep);
      post({ id, score: r.score, dims: r.dims }, [r.score.buffer]);
    } else {
      const r = scan(e.data.inp);
      post({ id, score: r.score, dims: r.dims }, [r.score.buffer]);
    }
  } catch (err) {
    post({ id, error: String(err) });
  }
};
