// Neuroglancer precomputed volumes (precomputed://..., as Neuroglancer and CloudVolume write
// them): the info's scales are the levels, chunks unsharded (<key>/x0-x1_y0-y1_z0-z1) or sharded
// (neuroglancer_uint64_sharded_v1, identity hash), encoded raw, as JPEG (a chunk's z planes as
// one tall image, decoded by the browser) or compressed_segmentation. The sharding and the
// segmentation decoding are chunkmirage's (web/src/precomputed.ts). A chunk not stored is zeros.
import {
  chunkName, compressedMorton, compressedSegmentation, Shards, type Fetch, type PrecomputedInfo, type PrecomputedScale,
} from "../../../chunkmirage/web/src/precomputed.ts";
import { decodeOff } from "./decode.ts";
import { Lru, normalizeUrl, type Level, type Region, type Source, type Vec3 } from "./sources.ts";

type Typed = Uint8Array | Uint16Array | Uint32Array | Int8Array | Int16Array | Int32Array | Float32Array | Float64Array | BigUint64Array;
const TYPED: Record<string, any> = {
  uint8: Uint8Array, uint16: Uint16Array, uint32: Uint32Array, int8: Int8Array, int16: Int16Array, int32: Int32Array,
  float32: Float32Array, float64: Float64Array, uint64: BigUint64Array,
};

interface Scale { s: PrecomputedScale; encoding: string; chunk: number[]; off: number[]; dir: string; shards: Shards | null }

export class PrecomputedSource implements Source {
  name: string;
  dims: 2 | 3 = 3;
  channels: 1 | 3 = 1;
  levels: Level[] = [];
  unit = "nanometer";
  bytesRead = 0;
  url: string;
  info!: PrecomputedInfo;
  private scales: Scale[] = [];
  private cache = new Lru<{ data: Typed; size: number[] }>(768 << 20);
  private pending = new Map<string, Promise<{ data: Typed; size: number[] }>>();
  private window: [number, number] | null = null;  // intensity window for data that is not uint8

  private constructor(url: string, name?: string) {
    this.url = url;
    this.name = name ?? url.split("/").slice(-2).join("/");
  }

  /** Neuroglancer's source string for this volume. */
  get ngSource() { return `precomputed://${this.url}`; }
  /** The voxel offset of the finest scale (z, y, x): Neuroglancer's coordinates are offset by it. */
  ngOffset: Vec3 = [0, 0, 0];

  static async open(url: string, name?: string): Promise<PrecomputedSource> {
    const s = new PrecomputedSource(normalizeUrl(url.replace(/^precomputed:\/\//, "")), name);
    const r = await fetch(`${s.url}/info`);
    if (!r.ok) throw new Error(`${s.url}/info: HTTP ${r.status}`);
    const info = (s.info = await r.json()) as PrecomputedInfo;
    if (!info?.scales?.length || (info["@type"] ?? "neuroglancer_multiscale_volume") !== "neuroglancer_multiscale_volume")
      throw new Error(`${s.url}: not a precomputed volume`);
    if (!TYPED[info.data_type]) throw new Error(`${s.url}: data type ${info.data_type} is not read here`);
    s.channels = (info.num_channels ?? 1) >= 3 ? 3 : 1;
    const bytesAt: Fetch = (u, range) => s.fetchBytes(u, range);
    s.scales = info.scales.map((sc) => {
      const encoding = sc.encoding ?? "raw";
      if (!["raw", "jpeg", "compressed_segmentation"].includes(encoding)) throw new Error(`${sc.key}: ${encoding} encoding is not read here`);
      const dir = `${s.url}/${sc.key.replace(/^\/+|\/+$/g, "")}`;
      return { s: sc, encoding, chunk: sc.chunk_sizes[0], off: sc.voxel_offset ?? [0, 0, 0], dir,
               shards: sc.sharding ? new Shards(dir, sc.sharding, bytesAt) : null };
    });
    s.ngOffset = [...(s.scales[0].off)].reverse() as Vec3;
    // the pyramid as the page wants it: finest first, z, y, x; a 2-D image is one plane deep
    s.levels = s.scales.map(({ s: sc, chunk }) => ({
      shape: [sc.size[2], sc.size[1], sc.size[0]] as Vec3, chunk: [chunk[2], chunk[1], chunk[0]] as Vec3,
      scale: [sc.resolution[2], sc.resolution[1], sc.resolution[0]] as Vec3,
    }));
    s.dims = s.levels[0].shape[0] > 1 ? 3 : 2;
    if (info.data_type !== "uint8") {  // window from the coarsest level, then every read maps to 0-255
      const L = s.levels.length - 1, sh = s.levels[L].shape;
      s.window = [0, 1];
      const raw = await s.readRaw(L, [0, 0, 0], sh);
      const v = Float64Array.from(raw.filter((x) => x !== 0 && x === x)).sort();
      s.window = v.length ? [v[Math.floor(v.length * 0.005)], v[Math.floor(v.length * 0.995)]] : [0, 1];
    }
    return s;
  }

  /** Bytes [start, end) of `url`, or all of it; null if not there (404). Retried, counted. */
  private async fetchBytes(url: string, range?: [number, number]): Promise<ArrayBuffer | null> {
    let err: unknown;
    for (let attempt = 0; attempt < 4; attempt++) {
      try {
        const r = await fetch(url, range ? { headers: { range: `bytes=${range[0]}-${range[1] - 1}` } } : {});
        if (r.status === 404 || r.status === 403 && !range) return null;
        if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
        const buf = await r.arrayBuffer();
        this.bytesRead += buf.byteLength;
        return range && r.status === 200 ? buf.slice(range[0], range[1]) : buf;
      } catch (e) {
        err = e;
        await new Promise((res) => setTimeout(res, 250 * 2 ** attempt));
      }
    }
    throw err;
  }

  /** Chunk `g` (x, y, z grid position) of scale `li`: its voxels (c, z, y, x) and size (x, y, z). */
  private chunkAt(li: number, g: number[]): Promise<{ data: Typed; size: number[] }> {
    const key = `${li}/${g.join(",")}`;
    const hit = this.cache.get(key);
    if (hit) return Promise.resolve(hit);
    let p = this.pending.get(key);
    if (p) return p;
    p = (async () => {
      const { s, encoding, chunk, off, dir, shards } = this.scales[li], T = TYPED[this.info.data_type], C = this.info.num_channels ?? 1;
      const lo = g.map((v, a) => v * chunk[a]), hi = lo.map((v, a) => Math.min(v + chunk[a], s.size[a]));
      const size = hi.map((v, a) => v - lo[a]), n = size[0] * size[1] * size[2];
      const bytes = shards
        ? await shards.chunk(compressedMorton(g, s.size.map((v, a) => Math.ceil(v / chunk[a]))))
        : await this.fetchBytes(`${dir}/${chunkName(lo.map((v, a) => v + off[a]), hi.map((v, a) => v + off[a]))}`);
      let data: Typed;
      if (!bytes) data = new T(n * C);
      else if (encoding === "jpeg") data = await jpeg(bytes, size[0], size[1] * size[2], C);
      else if (encoding === "compressed_segmentation")
        data = compressedSegmentation(bytes, size, C, s.compressed_segmentation_block_size ?? [8, 8, 8], this.info.data_type === "uint64");
      else data = new T(bytes);
      const out = { data, size };
      this.cache.set(key, out, data.byteLength);
      return out;
    })().finally(() => this.pending.delete(key));
    this.pending.set(key, p);
    return p;
  }

  /** Voxels [lo, hi) (z, y, x) of level `li`, channel-interleaved numbers; outside the volume NaN. */
  private async readRaw(li: number, lo: Vec3, hi: Vec3): Promise<Float32Array> {
    const { s, chunk } = this.scales[li], C = this.channels, Cin = this.info.num_channels ?? 1;
    const shape = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    const out = new Float32Array(shape[0] * shape[1] * shape[2] * C).fill(NaN);
    const size = [s.size[2], s.size[1], s.size[0]], ch = [chunk[2], chunk[1], chunk[0]];  // z, y, x
    const a = lo.map((v, i) => Math.max(0, v)), b = hi.map((v, i) => Math.min(size[i], v));
    if (a.some((v, i) => b[i] <= v)) return out;
    const jobs: Promise<void>[] = [];
    for (let cz = Math.floor(a[0] / ch[0]); cz * ch[0] < b[0]; cz++)
      for (let cy = Math.floor(a[1] / ch[1]); cy * ch[1] < b[1]; cy++)
        for (let cx = Math.floor(a[2] / ch[2]); cx * ch[2] < b[2]; cx++) {
          jobs.push(this.chunkAt(li, [cx, cy, cz]).then(({ data, size: cs }) => {
            const [sx, sy, sz] = cs, n = sx * sy * sz, z0 = cz * ch[0], y0 = cy * ch[1], x0 = cx * ch[2];
            const zlo = Math.max(a[0], z0), zhi = Math.min(b[0], z0 + sz), ylo = Math.max(a[1], y0), yhi = Math.min(b[1], y0 + sy);
            const xlo = Math.max(a[2], x0), xhi = Math.min(b[2], x0 + sx);
            for (let z = zlo; z < zhi; z++) for (let y = ylo; y < yhi; y++) {
              const src = ((z - z0) * sy + (y - y0)) * sx, dst = ((z - lo[0]) * shape[1] + (y - lo[1])) * shape[2];
              for (let x = xlo; x < xhi; x++)
                for (let c = 0; c < C; c++) out[(dst + x - lo[2]) * C + c] = Number(data[Math.min(c, Cin - 1) * n + src + x - x0]);
            }
          }));
        }
    await Promise.all(jobs);
    return out;
  }

  async read(level: number, lo: Vec3, hi: Vec3): Promise<Region> {
    const data = await this.readRaw(level, lo, hi);
    if (this.window) {
      const [w0, w1] = this.window, k = 255 / Math.max(1e-12, w1 - w0);
      for (let i = 0; i < data.length; i++) { const v = data[i]; if (v === v) data[i] = Math.max(0, Math.min(255, (v - w0) * k)); }
    }
    return { data, shape: [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]], channels: this.channels };
  }

  /** Neuroglancer's global dimensions for the finest scale (x, y, z in metres). */
  ngDimensions(): Record<string, [number, string]> {
    const r = this.scales[0].s.resolution;
    return { x: [r[0] * 1e-9, "m"], y: [r[1] * 1e-9, "m"], z: [r[2] * 1e-9, "m"] };
  }
}

/** A JPEG `width` x `height` (a chunk's x by its y * z) as `channels` planes of uint8, decoded
 * by the browser without colour management. */
async function jpeg(bytes: ArrayBuffer, width: number, height: number, channels: number): Promise<Uint8Array> {
  const off = decodeOff(bytes, "image/jpeg", { planes: channels, raw: true });  // in a worker
  if (off) {
    const d = await off;
    if (d.width !== width || d.height !== height) throw new Error(`a JPEG chunk ${d.width} x ${d.height}, not ${width} x ${height}`);
    return d.data as Uint8Array;
  }
  const image = await createImageBitmap(new Blob([bytes], { type: "image/jpeg" }), { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  if (image.width !== width || image.height !== height) throw new Error(`a JPEG chunk ${image.width} x ${image.height}, not ${width} x ${height}`);
  const g = new OffscreenCanvas(width, height).getContext("2d", { willReadFrequently: true })!;
  g.drawImage(image, 0, 0); image.close();
  const rgba = g.getImageData(0, 0, width, height).data, n = width * height, out = new Uint8Array(n * channels);
  for (let c = 0; c < channels; c++) for (let i = 0; i < n; i++) out[c * n + i] = rgba[4 * i + c];
  return out;
}
