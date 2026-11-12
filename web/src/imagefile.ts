// An image file from this computer (dropped on the page or picked): PNG, JPEG, WebP, GIF, BMP and
// AVIF decoded by the browser; TIFF by geotiff.js, a multi-page one as z planes and channels (as
// its OME-XML or ImageJ description says; else every full-size page a z plane). The image is read
// whole into memory and served as an OME-Zarr made on the fly (levels halved down to 256 px, each
// chunk cut when asked for), so the readers and Neuroglancer read it the way they read a folder
// (src/local.ts).
import type { Folder } from "./local.ts";

type Arr = Uint8Array | Uint16Array | Float32Array;
interface Img {
  name: string; c: number; z: number; y: number; x: number;
  data: Arr[];  // per channel, z * y * x
  dtype: "|u1" | "<u2" | "<f4";
  unit?: string; size?: [number, number, number];  // physical size of a voxel (z, y, x) in `unit`
  labels?: string[]; colors?: string[]; window?: [number, number];
}

/** The files read here, by name. */
export const IMAGE_FILE = /\.(png|jpe?g|webp|gif|bmp|avif|tiff?)$/i;
const MAX_SAMPLES = 1.5e9;  // values held in memory (channels x z x y x x)

export async function imageFolder(file: File): Promise<Folder> {
  const img = /\.tiff?$/i.test(file.name) || file.type === "image/tiff" ? await readTiff(file) : await readPicture(file);
  return virtualZarr(img);
}

async function readPicture(file: File): Promise<Img> {
  let bmp: ImageBitmap;
  // the file's own values: no colour management (a PNG's gamma or a JPEG's profile would change them)
  try { bmp = await createImageBitmap(file, { colorSpaceConversion: "none", premultiplyAlpha: "none" }); } catch { throw new Error(`${file.name}: this browser cannot decode it`); }
  const { width: x, height: y } = bmp, n = x * y;
  const cv = new OffscreenCanvas(x, y), g = cv.getContext("2d", { willReadFrequently: true })!;
  g.drawImage(bmp, 0, 0);
  bmp.close();
  let px: Uint8ClampedArray;
  try { px = g.getImageData(0, 0, x, y).data; } catch { throw new Error(`${file.name}: ${x} x ${y} px is more than this browser can decode at once; convert it to OME-Zarr`); }
  let grey = true;
  for (let i = 0; i < n && grey; i++) grey = px[4 * i] === px[4 * i + 1] && px[4 * i] === px[4 * i + 2];
  const C = grey ? 1 : 3;
  const data = Array.from({ length: C }, (_, k) => { const a = new Uint8Array(n); for (let i = 0; i < n; i++) a[i] = px[4 * i + k]; return a; });
  return { name: file.name, c: C, z: 1, y, x, data, dtype: "|u1", window: [0, 255],
           ...(C === 3 ? { labels: ["red", "green", "blue"], colors: ["FF0000", "00FF00", "0000FF"] } : {}) };
}

const UNITS: Record<string, string> = { "µm": "micrometer", "um": "micrometer", "micron": "micrometer", "\\u00B5m": "micrometer", "nm": "nanometer", "mm": "millimeter", "Å": "angstrom" };

async function readTiff(file: File): Promise<Img> {
  const { fromBlob } = await import("geotiff");
  const tif = await fromBlob(file);
  const first = await tif.getImage(0), x = first.getWidth(), y = first.getHeight(), spp = first.getSamplesPerPixel();
  const n = await tif.getImageCount(), pages: number[] = [];
  for (let i = 0; i < n; i++) {  // full-size pages: not a pyramid's smaller levels, nor masks (NewSubfileType bit 4)
    const im = i ? await tif.getImage(i) : first;
    let sub = 0;
    try { sub = Number(im.getFileDirectory().getValue("NewSubfileType") ?? 0); } catch { /* deferred: not a mask */ }
    if (im.getWidth() === x && im.getHeight() === y && !(i && sub & 4)) pages.push(i);
  }
  const fd = first.getFileDirectory(), tag = async (t: string) => { try { return await fd.loadValue(t as any); } catch { return undefined; } };
  const desc = String((await tag("ImageDescription")) ?? "");
  // how the pages are laid out: OME-XML (DimensionOrder), ImageJ (channels, then slices), else z planes
  let C = 1, Z = pages.length, cFirst = true, unit: string | undefined, size: [number, number, number] | undefined, labels: string[] | undefined;
  const ome = /<Pixels\b([^>]*)>/.exec(desc);
  if (ome) {
    const at = (k: string) => new RegExp(`\\b${k}="([^"]*)"`).exec(ome[1])?.[1];
    const sc = Number(at("SizeC") ?? 1) / (spp > 1 ? spp : 1), sz = Number(at("SizeZ") ?? 1), order = at("DimensionOrder") ?? "XYZCT";
    if (sc * sz <= pages.length) { C = sc; Z = sz; cFirst = order.indexOf("C") < order.indexOf("Z"); }
    const u = at("PhysicalSizeXUnit") ?? "µm", px = Number(at("PhysicalSizeX")), pz = Number(at("PhysicalSizeZ") ?? at("PhysicalSizeX"));
    if (px > 0) { unit = UNITS[u] ?? "micrometer"; size = [pz > 0 ? pz : px, Number(at("PhysicalSizeY")) || px, px]; }
    const names = [...desc.matchAll(/<Channel\b[^>]*\bName="([^"]*)"/g)].map((m) => m[1]);
    if (names.length === C) labels = names;
  } else if (/^ImageJ=/m.test(desc)) {
    const at = (k: string) => Number(new RegExp(`^${k}=(\\S+)`, "m").exec(desc)?.[1]);
    const ch = at("channels") || 1, sl = at("slices") || Math.floor(pages.length / ch);
    if (ch * sl <= pages.length) { C = ch; Z = sl; cFirst = true; }
    const u = new RegExp("^unit=(\\S+)", "m").exec(desc)?.[1];
    const res = await tag("XResolution"), sp = at("spacing");
    if (u && UNITS[u]) {
      unit = UNITS[u];
      const pxs = typeof res === "number" ? 1 / res : res && (res as ArrayLike<number>).length >= 2 && (res as ArrayLike<number>)[0] ? (res as ArrayLike<number>)[1] / (res as ArrayLike<number>)[0] : 0;  // pixels per unit, as a rational
      if (pxs > 0) size = [sp > 0 ? sp : pxs, pxs, pxs];
    }
  }
  const planes = C * Z * (spp > 1 ? spp : 1);
  if (planes * x * y > MAX_SAMPLES) throw new Error(`${file.name}: ${x} x ${y} px x ${planes} planes is too much to hold in memory here; convert it to OME-Zarr (bioformats2raw)`);
  const bits = first.getBitsPerSample(), fmt = (first as any).getSampleFormat?.() ?? 1;
  const dtype: Img["dtype"] = fmt === 1 && bits === 8 ? "|u1" : fmt === 1 && bits === 16 ? "<u2" : "<f4";
  const make = () => (dtype === "|u1" ? new Uint8Array(Z * y * x) : dtype === "<u2" ? new Uint16Array(Z * y * x) : new Float32Array(Z * y * x));
  const outC = spp > 1 ? spp : C, data = Array.from({ length: outC }, make);
  for (let zi = 0; zi < Z; zi++)
    for (let ci = 0; ci < (spp > 1 ? 1 : C); ci++) {
      const page = pages[cFirst ? zi * C + ci : ci * Z + zi];
      const r = (await (await tif.getImage(page)).readRasters()) as unknown as ArrayLike<number>[];
      r.forEach((plane, s) => { const ch = spp > 1 ? s : ci; if (ch < outC) (data[ch] as any).set(plane, zi * y * x); });
    }
  return { name: file.name, c: outC, z: Z, y, x, data, dtype, unit, size, labels,
           ...(spp === 3 ? { labels: ["red", "green", "blue"], colors: ["FF0000", "00FF00", "0000FF"] } : {}),
           ...(dtype === "|u1" && (spp === 3 || outC === 1) ? { window: [0, 255] as [number, number] } : {}) };
}

interface Lv { z: number; y: number; x: number; fz: number; data: Arr[] }

/** Half the size in y and x (and z, in a volume with more than one plane): the mean of each block. */
function halve(L: Lv, d3: boolean): Lv {
  const nz = d3 && L.z > 1 ? Math.ceil(L.z / 2) : L.z, ny = Math.ceil(L.y / 2), nx = Math.ceil(L.x / 2), kz = nz < L.z ? 2 : 1;
  const data = L.data.map((a) => {
    const out = new (a.constructor as any)(nz * ny * nx) as Arr, int = !(a instanceof Float32Array);
    for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      let s = 0, m = 0;
      for (let dz = 0; dz < kz; dz++) { const zz = z * kz + dz; if (zz >= L.z) continue;
        for (let dy = 0; dy < 2; dy++) { const yy = 2 * y + dy; if (yy >= L.y) continue;
          for (let dx = 0; dx < 2; dx++) { const xx = 2 * x + dx; if (xx >= L.x) continue; s += a[(zz * L.y + yy) * L.x + xx]; m++; } } }
      out[(z * ny + y) * nx + x] = int ? Math.round(s / m) : s / m;
    }
    return out;
  });
  return { z: nz, y: ny, x: nx, fz: L.fz * kz, data };
}

/** ImageJ's composite colours (3 channels: RGB), as the reader gives channels without any. */
const COLOURS = ["FF0000", "00FF00", "0000FF", "FFFFFF", "00FFFF", "FF00FF", "FFFF00"];
/** A channel's values shown black to full colour: the given window, else 0.5-99.5% of its
 * values at the coarsest level, zeros left out. */
function windowOf(a: Arr, given?: [number, number]): [number, number] {
  if (given) return given;
  const v = Float64Array.from(Array.prototype.filter.call(a, (x: number) => x !== 0) as number[]).sort();
  return v.length && v[v.length - 1] > v[0] ? [v[Math.floor(v.length * 0.005)], v[Math.floor(v.length * 0.995)]] : [0, 1];
}

/** The image as an OME-Zarr (v2, uncompressed): metadata and chunks made when asked for. Every
 * channel gets a colour and a window (Neuroglancer rejects channel metadata without colours). */
function virtualZarr(img: Img): Folder {
  const d3 = img.z > 1, multi = img.c > 1;
  const levels: Lv[] = [{ z: img.z, y: img.y, x: img.x, fz: 1, data: img.data }];
  while (Math.max(levels.at(-1)!.y, levels.at(-1)!.x) > 256) levels.push(halve(levels.at(-1)!, d3));
  const ch = d3 ? [64, 64, 64] : [512, 512], [sz, sy, sx] = img.size ?? [1, 1, 1];
  const space = (n: string) => ({ name: n, type: "space", ...(img.unit ? { unit: img.unit } : {}) });
  const axes = [...(multi ? [{ name: "c", type: "channel" }] : []), ...(d3 ? [space("z")] : []), space("y"), space("x")];
  const attrs = {
    multiscales: [{ version: "0.4", name: img.name, axes, datasets: levels.map((L, k) => ({ path: String(k), coordinateTransformations: [{ type: "scale",
      scale: [...(multi ? [1] : []), ...(d3 ? [sz * L.fz] : []), sy * 2 ** k, sx * 2 ** k] }] })) }],
    ...(multi ? { omero: { channels: Array.from({ length: img.c }, (_, k) => ({
      label: img.labels?.[k] ?? String(k), active: k < 3, color: img.colors?.[k] ?? COLOURS[k % COLOURS.length],
      window: (([a, b]) => ({ start: a, end: b, min: a, max: b }))(windowOf(levels.at(-1)!.data[k], img.window)) })) } } : {}),
  };
  const text = (s: unknown, name: string) => new File([JSON.stringify(s)], name);
  return {
    name: img.name,
    async list() { return []; },
    async file(path) {
      if (path === ".zgroup") return text({ zarr_format: 2 }, path);
      if (path === ".zattrs") return text(attrs, path);
      const parts = path.split("/"), L = levels[Number(parts[0])];
      if (!L || !/^\d+$/.test(parts[0])) return null;
      if (parts[1] === ".zarray" && parts.length === 2)
        return text({ zarr_format: 2, shape: [...(multi ? [img.c] : []), ...(d3 ? [L.z] : []), L.y, L.x], chunks: [...(multi ? [1] : []), ...ch],
                      dtype: img.dtype, compressor: null, fill_value: 0, order: "C", filters: null, dimension_separator: "/" }, path);
      const idx = parts.slice(1).map(Number);
      if (idx.length !== (multi ? 1 : 0) + (d3 ? 3 : 2) || idx.some((v) => !Number.isInteger(v) || v < 0)) return null;
      const c = multi ? idx[0] : 0, [zi, yi, xi] = d3 ? idx.slice(multi ? 1 : 0) : [0, ...idx.slice(multi ? 1 : 0)];
      const [cz, cy, cx] = d3 ? ch : [1, ...ch], src = L.data[c];
      if (!src || zi * cz >= L.z || yi * cy >= L.y || xi * cx >= L.x) return null;
      const out = new (src.constructor as any)(cz * cy * cx) as Arr, x0 = xi * cx, w = Math.min(cx, L.x - x0);
      for (let z = 0; z < cz && zi * cz + z < L.z; z++)
        for (let y = 0; y < cy && yi * cy + y < L.y; y++) {
          const o = ((zi * cz + z) * L.y + yi * cy + y) * L.x + x0;
          (out as any).set(src.subarray(o, o + w), (z * cy + y) * cx);
        }
      return new File([out as Uint8Array<ArrayBuffer>], path);
    },
  };
}
