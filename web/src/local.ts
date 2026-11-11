// Data on this computer: a folder (OME-Zarr or Neuroglancer precomputed) dropped on the page or
// picked, read where it is; nothing is uploaded. public/local-sw.js serves it at
// ./local/<id>/<path> and passes each request here, so the readers and Neuroglancer (in its own
// frame and worker) read it by URL as they read any host.

/** A folder's files by path ("0/.zarray") and its entries (to find the image inside). */
export interface Folder {
  name: string;
  file(path: string): Promise<File | null>;
  list(path: string): Promise<{ name: string; dir: boolean }[]>;
}

/** Directories by path, each looked up once (`child` opens one inside another, or null). */
function dirCache<D>(root: D, child: (d: D, name: string) => Promise<D | null>) {
  const dirs = new Map<string, Promise<D | null>>([["", Promise.resolve(root)]]);
  const dir = (path: string): Promise<D | null> => {
    let p = dirs.get(path);
    if (!p) {
      const k = path.lastIndexOf("/");
      p = dir(k < 0 ? "" : path.slice(0, k)).then((d) => (d ? child(d, path.slice(k + 1)) : null));
      dirs.set(path, p);
    }
    return p;
  };
  return dir;
}
const split = (path: string) => { const k = path.lastIndexOf("/"); return [k < 0 ? "" : path.slice(0, k), path.slice(k + 1)]; };

/** A folder dropped on the page (the drop's entry: every browser has it). */
export function entryFolder(root: FileSystemDirectoryEntry): Folder {
  const dir = dirCache(root, (d, name) => new Promise((res) => d.getDirectory(name, {}, (e) => res(e as FileSystemDirectoryEntry), () => res(null))));
  return {
    name: root.name,
    async file(path) {
      const [p, name] = split(path), d = await dir(p);
      if (!d || !name) return null;
      return new Promise((res) => d.getFile(name, {}, (e) => (e as FileSystemFileEntry).file(res, () => res(null)), () => res(null)));
    },
    async list(path) {
      const d = await dir(path);
      if (!d) return [];
      const r = d.createReader(), out: { name: string; dir: boolean }[] = [];
      for (;;) {  // entries come in batches
        const batch = await new Promise<FileSystemEntry[]>((res) => r.readEntries(res, () => res([])));
        if (!batch.length) return out;
        for (const e of batch) out.push({ name: e.name, dir: e.isDirectory });
      }
    },
  };
}

/** A folder picked with the browser's folder picker (Chrome and Edge). */
export function handleFolder(root: FileSystemDirectoryHandle): Folder {
  const dir = dirCache<FileSystemDirectoryHandle>(root, (d, name) => d.getDirectoryHandle(name).catch(() => null));
  return {
    name: root.name,
    async file(path) {
      const [p, name] = split(path), d = await dir(p);
      if (!d || !name) return null;
      try { return await (await d.getFileHandle(name)).getFile(); } catch { return null; }
    },
    async list(path) {
      const d = await dir(path), out: { name: string; dir: boolean }[] = [];
      if (d) for await (const h of (d as any).values() as AsyncIterable<FileSystemHandle>) out.push({ name: h.name, dir: h.kind === "directory" });
      return out;
    },
  };
}

export const canPick = typeof window !== "undefined" && typeof (window as any).showDirectoryPicker === "function";
export async function pickFolder(): Promise<Folder | null> {
  try { return handleFolder(await (window as any).showDirectoryPicker({ mode: "read" })); }
  catch (e) { if ((e as Error).name === "AbortError") return null; throw e; }
}

// ---------------------------------------------------------------------------------------
// serving: the service worker asks, this page answers from the folder
// ---------------------------------------------------------------------------------------
const folders = new Map<string, Folder>();
let ready: Promise<ServiceWorkerContainer> | null = null;

function serviceWorker(): Promise<ServiceWorkerContainer> {
  return (ready ??= (async () => {
    const c = navigator.serviceWorker;
    if (!c) throw new Error("this browser cannot read folders here (no service workers: a private window?)");
    c.addEventListener("message", answer);
    c.startMessages();
    await c.register("./local-sw.js", { scope: "./" });
    await c.ready;
    if (!c.controller) {  // the first time: the worker takes over this page as it starts
      await new Promise<void>((res, rej) => {
        c.addEventListener("controllerchange", () => res(), { once: true });
        setTimeout(() => (c.controller ? res() : rej(new Error("the folder reader did not start: reload the page and drop it again"))), 5000);
      });
    }
    return c;
  })().catch((e) => { ready = null; throw e; }));
}

interface Ask { coresampleLocal: string; path: string; range: { start: number; end: number | null } | { suffix: number } | null; head: boolean }
async function answer(e: MessageEvent) {
  const m = e.data as Ask, port = e.ports[0];
  if (!m?.coresampleLocal || !port) return;
  const f = folders.get(m.coresampleLocal);
  if (!f) { port.postMessage(null); return; }
  try {
    const file = await f.file(m.path);
    if (!file) { port.postMessage({ status: 404 }); return; }
    const size = file.size, r = m.range;
    let a = 0, b = size;
    if (r && "suffix" in r) a = Math.max(0, size - r.suffix);
    else if (r) { a = r.start; b = Math.min(size, r.end ?? size); }
    if (r && a >= size && size > 0) { port.postMessage({ status: 416, size }); return; }
    const body = m.head ? null : await file.slice(a, b).arrayBuffer();
    port.postMessage({ status: r ? 206 : 200, size, start: a, end: b, body }, body ? [body] : []);
  } catch (err) {
    port.postMessage({ status: 500, error: String(err) });
  }
}

/** The address the readers and the viewer read `f` at (no trailing slash). */
export async function serveFolder(f: Folder): Promise<string> {
  const c = await serviceWorker();
  const id = crypto.randomUUID().slice(0, 8);
  folders.set(id, f);
  c.controller?.postMessage({ coresampleLocal: id });  // so the worker knows which window has it
  return new URL(`local/${id}`, document.baseURI).href;
}

// ---------------------------------------------------------------------------------------
// what is in the folder
// ---------------------------------------------------------------------------------------
export type Kind = "ome-zarr" | "precomputed" | "n5";
export interface Found { path: string; kind: Kind }

/** The images in `f`, nearest the top first and labels last: the folder itself if it is one,
 * else what its groups and plain folders hold (a container, a plate, a folder of .zarr). */
export async function findImages(f: Folder, max = 12): Promise<Found[]> {
  const json = async (p: string) => { try { const x = await f.file(p); return x ? JSON.parse(await x.text()) : null; } catch { return null; } };
  const kindOf = async (dir: string): Promise<Kind | "group" | "array" | null> => {
    const pre = dir ? dir + "/" : "";
    const v3 = await json(pre + "zarr.json");
    if (v3) return v3.node_type === "array" ? "array" : (v3.attributes?.ome ?? v3.attributes ?? {}).multiscales ? "ome-zarr" : "group";
    const za = await json(pre + ".zattrs");
    if (za?.multiscales) return "ome-zarr";
    if (await f.file(pre + ".zarray")) return "array";
    if ((await json(pre + "info"))?.scales) return "precomputed";
    if (za || (await f.file(pre + ".zgroup"))) return "group";
    if (await f.file(pre + "attributes.json")) return "n5";
    return null;
  };
  const out: Found[] = [];
  let queue = [""], looked = 0;
  for (let depth = 0; depth < 5 && queue.length && out.length < max; depth++) {
    const next: string[] = [];
    for (const d of queue) {
      if (++looked > 300 || out.length >= max) break;
      const k = await kindOf(d);
      if (k === "ome-zarr" || k === "precomputed" || k === "n5") { out.push({ path: d, kind: k }); continue; }
      if (k === "array") continue;
      for (const e of await f.list(d)) if (e.dir && !e.name.startsWith(".")) next.push(d ? `${d}/${e.name}` : e.name);
    }
    queue = next;
  }
  const label = (p: string) => p.split("/").includes("labels");
  return out.sort((a, b) => Number(a.kind === "n5") - Number(b.kind === "n5") || Number(label(a.path)) - Number(label(b.path)));
}
