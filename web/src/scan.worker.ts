// The cheap scan off the page's main thread.
import { scan, type ScanInput } from "./scan.ts";

self.onmessage = (e: MessageEvent<{ id: number; inp: ScanInput }>) => {
  const { id, inp } = e.data;
  try {
    const r = scan(inp);
    (self as unknown as Worker).postMessage({ id, score: r.score, dims: r.dims }, [r.score.buffer]);
  } catch (err) {
    (self as unknown as Worker).postMessage({ id, error: String(err) });
  }
};
