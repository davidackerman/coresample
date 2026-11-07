#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib", "neuroglancer"]
# ///
"""chunkmap: visualize a remote zarr array using only compressed chunk sizes.

No pixel data is decompressed. Sizes come from a bucket listing (S3 or GCS,
one request per 1000 chunks), or from per-chunk HEAD requests on other hosts.
For zarr v3 sharded arrays, --inner reads each shard's index to get the size
of every inner chunk.

Usage (uv installs the dependencies on first run, nothing else needed):
  uv run chunkmap.py URL [--level PATH] [--max-chunks N] [--inner] [--out map.png]
  uv run chunkmap.py URL --ng [--ng-bind 0.0.0.0 --ng-port 8080]

URL may be https://, s3://, gs://, or Neuroglancer-style (zarr://s3://..., precomputed://gs://...).

URL can be an OME-Zarr multiscale group (v0.4 or v0.5), a single zarr array, or a
Neuroglancer precomputed volume (unsharded, or sharded: sizes come from the shard indexes).
Every level from the finest one under --max-chunks up to the coarsest is mapped.
The PNG shows the finest; with --ng all of them are served as one multiscale
Neuroglancer layer over the original data, so zooming out shows each coarser
level's own chunk sizes.
"""
import argparse, functools, gzip, http.server, json, os, re, socket, sys, tempfile, threading, time, warnings
import urllib.error, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import numpy as np

UA = {"User-Agent": "chunkmap/0.1"}
META = {".zarray", ".zattrs", ".zgroup", ".zmetadata", "zarr.json"}


class Progress:
    """Prints 'what: n/total (elapsed)' to stderr at most every few seconds."""

    def __init__(self, what, total):
        self.what, self.total, self.n, self.t0, self.last = what, total, 0, time.time(), 0.0

    def add(self, k=1):
        self.n += k
        now = time.time()
        if now - self.last > 3 or self.n >= self.total:
            self.last = now
            print(f"\r  {self.what}: {self.n}/{self.total} ({now - self.t0:.0f}s)   ",
                  end="", file=sys.stderr, flush=True)

    def done(self):
        if self.last:
            print(file=sys.stderr)


def get(url, headers=None, method="GET"):
    req = urllib.request.Request(url, headers={**UA, **(headers or {})}, method=method)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, r.headers, (b"" if method == "HEAD" else r.read())


def get_json(url):
    try:
        return json.loads(get(url)[2])
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):
            return None
        raise


def normalize_url(url):
    """Accept Neuroglancer-style URLs: zarr://, zarr2://, zarr3://, s3://, gs://."""
    url = re.sub(r"^(zarr[23]?|precomputed)://", "", url.strip()).rstrip("/")
    if url.startswith("s3://"):
        bucket, _, key = url[5:].partition("/")
        return f"https://{bucket}.s3.amazonaws.com/{key}".rstrip("/")
    if url.startswith("gs://"):
        return "https://storage.googleapis.com/" + url[5:]
    return url


# ---------- metadata ----------
def array_meta(url):
    m = get_json(url + "/zarr.json")
    if m is not None and m.get("node_type") == "array":
        return 3, m
    m = get_json(url + "/.zarray")
    return (2, m) if m is not None else (None, None)


def group_attrs(url):
    g = get_json(url + "/zarr.json")
    if g is not None:
        a = g.get("attributes", {})
        return a.get("ome", a)
    return get_json(url + "/.zattrs") or {}


def shape_chunks(v, m):
    if v == 2:
        return m["shape"], m["chunks"]
    return m["shape"], m["chunk_grid"]["configuration"]["chunk_shape"]


def n_chunks(v, m):
    s, c = shape_chunks(v, m)
    return int(np.prod([-(-a // b) for a, b in zip(s, c)]))


def plain_geometry(m, ndim):
    names = m.get("dimension_names") or []
    names = [n or f"d{i}" for i, n in enumerate(names)] or [f"d{i}" for i in range(ndim)]
    return dict(names=names, types=[None] * ndim, units=[None] * ndim,
                scale=np.ones(ndim), corner=np.zeros(ndim))


def level_geometry(ms, i, ndim):
    """Physical scale and translation per axis for multiscale level i (OME-Zarr)."""
    axes = ms.get("axes") or []
    names = [a["name"] if isinstance(a, dict) else a for a in axes] or [f"d{k}" for k in range(ndim)]
    types = [a.get("type") if isinstance(a, dict) else None for a in axes] or [None] * ndim
    units = [a.get("unit") if isinstance(a, dict) else None for a in axes] or [None] * ndim

    def st(transforms):
        sc, tr = np.ones(ndim), np.zeros(ndim)
        for t in transforms or []:
            if t["type"] == "scale":
                sc = np.array(t["scale"], float)
            elif t["type"] == "translation":
                tr = np.array(t["translation"], float)
        return sc, tr

    ds, dt = st(ms["datasets"][i].get("coordinateTransformations"))
    gs, gt = st(ms.get("coordinateTransformations"))
    scale, trans = gs * ds, gs * dt + gt
    # OME translations locate voxel centers; Neuroglancer works with lower corners.
    return dict(names=names, types=types, units=units, scale=scale, corner=trans - 0.5 * scale)


def key_funcs(v, m):
    if v == 2:
        sep, pre = m.get("dimension_separator") or ".", ""
    else:
        enc = m.get("chunk_key_encoding", {"name": "default"})
        cfg = enc.get("configuration", {})
        if enc["name"] == "v2":
            sep, pre = cfg.get("separator", "."), ""
        else:
            sep = cfg.get("separator", "/")
            pre = "c" + sep

    def key_of(idx):
        return pre + sep.join(map(str, idx))

    def parse(rel):
        if pre and not rel.startswith(pre):
            return None
        try:
            return tuple(int(t) for t in rel[len(pre):].split(sep))
        except ValueError:
            return None

    return key_of, parse


# ---------- sizes: listing ----------
def s3_parts(url):
    p = urllib.parse.urlparse(url)
    path = urllib.parse.unquote(p.path).strip("/")
    if re.match(r"^.+\.s3[.-]([a-z0-9-]+\.)?amazonaws\.com$", p.netloc):
        return f"{p.scheme}://{p.netloc}/", path
    if re.match(r"^s3[.-]([a-z0-9-]+\.)?amazonaws\.com$", p.netloc):
        bucket, _, key = path.partition("/")
        return f"{p.scheme}://{p.netloc}/{bucket}", key
    return None


def list_s3(endpoint, prefix):
    token = None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        root = ET.fromstring(get(endpoint + "?" + urllib.parse.urlencode(q))[2])
        tag = lambda e: e.tag.rsplit("}", 1)[-1]
        token, truncated = None, False
        for el in root:
            if tag(el) == "Contents":
                f = {tag(c): c.text for c in el}
                yield f["Key"], int(f["Size"])
            elif tag(el) == "IsTruncated":
                truncated = el.text == "true"
            elif tag(el) == "NextContinuationToken":
                token = el.text
        if not (truncated and token):
            return


def gcs_parts(url):
    p = urllib.parse.urlparse(url)
    path = urllib.parse.unquote(p.path).strip("/")
    if p.netloc == "storage.googleapis.com":
        bucket, _, key = path.partition("/")
        return bucket, key
    if p.netloc.endswith(".storage.googleapis.com"):
        return p.netloc[: -len(".storage.googleapis.com")], path
    return None


def list_gcs(bucket, prefix):
    token = None
    while True:
        q = {"prefix": prefix, "fields": "items(name,size),nextPageToken", "maxResults": "1000"}
        if token:
            q["pageToken"] = token
        d = json.loads(get(f"https://storage.googleapis.com/storage/v1/b/"
                           f"{urllib.parse.quote(bucket)}/o?" + urllib.parse.urlencode(q))[2])
        for it in d.get("items", []):
            yield it["name"], int(it["size"])
        token = d.get("nextPageToken")
        if not token:
            return


def try_list(arr_url, prefixes=None, threads=32):
    """List objects under arr_url with sizes. With prefixes (e.g. one per index along the
    first axis), the slabs are listed in parallel instead of one long paginated stream."""
    s3, gcs = s3_parts(arr_url), gcs_parts(arr_url)
    if not (s3 or gcs):
        return None
    key = (s3 or gcs)[1]

    def one(prefix):
        full = f"{key}/{prefix}"
        it = list_s3(s3[0], full) if s3 else list_gcs(gcs[0], full)
        return [(k[len(key) + 1:], n) for k, n in it]

    try:
        if not prefixes:
            return one("")
        out, prog = one(prefixes[0]), Progress("listing slabs", len(prefixes))  # fails fast if refused
        prog.add()
        with ThreadPoolExecutor(threads) as ex:
            for part in ex.map(one, prefixes[1:]):
                out.extend(part)
                prog.add()
        prog.done()
        return out
    except urllib.error.HTTPError as e:
        print(f"Bucket listing refused (HTTP {e.code}); falling back to HEAD requests", file=sys.stderr)
    return None


# ---------- sizes: per-object requests ----------
def remote_size(url):
    try:
        _, h, _ = get(url, method="HEAD")
        if h.get("Content-Length") is not None:
            return float(h["Content-Length"])
        _, h, _ = get(url, headers={"Range": "bytes=0-0"})
        cr = h.get("Content-Range")
        return float(cr.split("/")[-1]) if cr else np.nan
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):  # absent chunk = fill value (S3 gives 403 without list rights)
            return np.nan
        raise


def shard_inner_sizes(arr_url, key_of, outer, m, threads):
    cfg = next((c["configuration"] for c in m.get("codecs", []) if c["name"] == "sharding_indexed"), None)
    if cfg is None:
        sys.exit("--inner needs a zarr v3 array with the sharding_indexed codec")
    shard_shape = m["chunk_grid"]["configuration"]["chunk_shape"]
    per = [s // c for s, c in zip(shard_shape, cfg["chunk_shape"])]
    n = int(np.prod(per))
    crc = any(c.get("name") == "crc32c" for c in cfg.get("index_codecs", []))
    isz = 16 * n + (4 if crc else 0)
    at_end = cfg.get("index_location", "end") == "end"
    grid = np.full([g * p for g, p in zip(outer.shape, per)], np.nan)
    idxs = [i for i in np.ndindex(outer.shape) if np.isfinite(outer[i])]

    def fetch(idx):
        rng = f"bytes=-{isz}" if at_end else f"bytes=0-{isz - 1}"
        b = get(f"{arr_url}/{key_of(idx)}", headers={"Range": rng})[2]
        if len(b) > isz:  # server ignored the Range header
            b = b[-isz:] if at_end else b[:isz]
        tab = np.frombuffer(b[:16 * n], "<u8").reshape(per + [2])
        s = tab[..., 1].astype(float)
        s[tab[..., 1] == np.iinfo(np.uint64).max] = np.nan
        return idx, s

    prog = Progress("reading shard indexes", len(idxs))
    with ThreadPoolExecutor(threads) as ex:
        for idx, s in ex.map(fetch, idxs):
            grid[tuple(slice(i * p, (i + 1) * p) for i, p in zip(idx, per))] = s
            prog.add()
    prog.done()
    return grid, list(cfg["chunk_shape"])


# ---------- neuroglancer precomputed ----------
def morton_decode(ids, bits):
    """Compressed Morton code -> (x, y, z) chunk grid positions."""
    out = np.zeros((3, len(ids)), dtype=np.uint64)
    j = 0
    for i in range(max(bits)):
        for d in range(3):
            if i < bits[d]:
                out[d] |= ((ids >> np.uint64(j)) & np.uint64(1)) << np.uint64(i)
                j += 1
    return out.astype(np.int64)


def precomputed_shard_sizes(base, sc, grid, sizes, threads):
    sh = sc["sharding"]
    nmini = 1 << sh["minishard_bits"]
    isz = 16 * nmini
    gz = sh.get("minishard_index_encoding", "raw") == "gzip"
    listed = try_list(base)
    if listed is not None:
        shards = [rel for rel, _ in listed if rel.endswith(".shard")]
    else:
        sbits = sh["shard_bits"]
        if (1 << sbits) > 200000:
            sys.exit("Too many possible shards to probe without a bucket listing; pick a coarser --level")
        shards = [f"{i:0{-(-sbits // 4)}x}.shard" for i in range(1 << sbits)]
    bits = [(n - 1).bit_length() for n in grid]
    slow_shards = []

    def rng(url, start, stop):
        b = get(url, headers={"Range": f"bytes={start}-{stop - 1}"})[2]
        return b[start:stop] if len(b) > stop - start else b  # server ignored Range

    def header(name):
        url = f"{base}/{name}"
        try:
            head = rng(url, 0, isz)
        except urllib.error.HTTPError as e:
            if e.code in (403, 404, 416):
                return url, None
            raise
        if len(head) < isz:
            return url, None
        tab = np.frombuffer(head, "<u8").reshape(nmini, 2).astype(np.int64)
        return url, tab[tab[:, 1] > tab[:, 0]]

    # Pass 1: every shard's header, in parallel. Each lists where its minishard indexes are.
    tasks, prog = [], Progress("reading shard headers", len(shards))
    with ThreadPoolExecutor(threads) as ex:
        for url, tab in ex.map(header, shards):
            prog.add()
            if tab is None or not len(tab):
                continue
            lo, hi = int(tab[:, 0].min()), int(tab[:, 1].max())
            if hi - lo <= 4 * int((tab[:, 1] - tab[:, 0]).sum()) + 65536:
                tasks.append((url, lo, hi, [(int(x), int(y)) for x, y in tab]))  # stored together
            else:  # interleaved with chunk data: one small request per minishard
                slow_shards.append(url)
                tasks.extend((url, int(x), int(y), [(int(x), int(y))]) for x, y in tab)
    prog.done()

    def fetch(task):
        url, lo, hi, segs = task
        blob = rng(url, isz + lo, isz + hi)
        return [blob[x - lo:y - lo] for x, y in segs]

    # Pass 2: all minishard indexes across all shards, in parallel. They hold each chunk's size.
    prog = Progress("reading minishard indexes", len(tasks))
    with ThreadPoolExecutor(threads) as ex:
        for pieces in ex.map(fetch, tasks):
            prog.add()
            for pc in pieces:
                arr = np.frombuffer(gzip.decompress(pc) if gz else pc, "<u8").reshape(3, -1)
                x, y, z = morton_decode(np.cumsum(arr[0], dtype=np.uint64), bits)
                ok = (x < grid[0]) & (y < grid[1]) & (z < grid[2])
                sizes[x[ok], y[ok], z[ok]] = arr[2][ok]
    prog.done()
    extra = (f"; {len(slow_shards)} shards keep minishard indexes apart, "
             f"{len(tasks)} index requests") if slow_shards else ""
    return f"shard indexes ({len(shards)} shards{extra})"


def precomputed_level(url, sc, a):
    key, cs, size = sc["key"], sc["chunk_sizes"][0], sc["size"]
    off, res = sc.get("voxel_offset", [0, 0, 0]), sc["resolution"]
    grid = [-(-n // c) for n, c in zip(size, cs)]
    sizes = np.full(grid, np.nan)  # x, y, z
    base = f"{url}/{key}"
    if "sharding" in sc:
        how = precomputed_shard_sizes(base, sc, grid, sizes, a.threads)
    else:
        prefixes = [f"{off[0] + i * cs[0]}-{min(off[0] + (i + 1) * cs[0], off[0] + size[0])}_"
                    for i in range(grid[0])]
        listed = try_list(base, prefixes, a.threads)
        if listed is not None:
            how = "bucket listing"
            for rel, n in listed:
                mt = re.fullmatch(r"(-?\d+)-(-?\d+)_(-?\d+)-(-?\d+)_(-?\d+)-(-?\d+)", rel)
                if not mt:
                    continue
                idx = tuple((int(mt.group(k)) - o) // c for k, o, c in zip((1, 3, 5), off, cs))
                if all(0 <= i < g for i, g in zip(idx, grid)):
                    sizes[idx] = n
        else:
            how = "HEAD requests"

            def key_of(idx):
                return "_".join(f"{o + i * c}-{min(o + (i + 1) * c, o + n)}"
                                for i, o, c, n in zip(idx, off, cs, size))

            idxs = list(np.ndindex(*grid))
            prog = Progress("HEAD requests", len(idxs))
            with ThreadPoolExecutor(a.threads) as ex:
                for idx, v in zip(idxs, ex.map(lambda i: remote_size(f"{base}/{key_of(i)}"), idxs)):
                    sizes[idx] = v
                    prog.add()
            prog.done()
    r = np.array(res[::-1], float)
    geom = dict(names=["z", "y", "x"], types=["space"] * 3, units=["nanometer"] * 3,
                scale=r, corner=np.array(off[::-1], float) * r)  # precomputed is corner-based
    return dict(sizes=sizes.transpose(2, 1, 0), chunks=list(cs[::-1]), geom=geom,
                label=f"{url} (precomputed, scale {key})", shape=list(size[::-1]), how=how)


def precomputed_levels(url, info, a):
    scales = info["scales"]  # finest first
    if a.level is not None:
        sc = next((s for s in scales if s["key"] == a.level), None)
        if sc is None:
            sys.exit(f"Scale {a.level!r} not found; scales are {[s['key'] for s in scales]}")
        return [precomputed_level(url, sc, a)]
    n = lambda s: int(np.prod([-(-a_ // c) for a_, c in zip(s["size"], s["chunk_sizes"][0])]))
    keep = [s for s in scales if n(s) <= a.max_chunks] or [scales[-1]]
    skipped = [s["key"] for s in scales if s not in keep]
    if skipped:
        print(f"Skipping scales over --max-chunks: {skipped}", file=sys.stderr)
    out = []
    for sc in keep:
        print(f"Scale {sc['key']}: {n(sc)} chunks", file=sys.stderr)
        t0 = time.time()
        out.append(precomputed_level(url, sc, a))
        out[-1]["seconds"] = time.time() - t0
    return out


# ---------- neuroglancer ----------
SHADER = """#uicontrol invlerp size(range=[%.3f, %.3f])
#uicontrol float opacity slider(min=0, max=1, default=0.6)
#uicontrol bool show_missing checkbox(default=false)
void main() {
  float v = getDataValue();
  if (v < 0.0) {
    if (show_missing) emitRGBA(vec4(0.23, 0.43, 0.65, 0.5 * opacity));
    else emitTransparent();
    return;
  }
  emitRGBA(vec4(colormapJet(size()), opacity));
}
"""


def spatial_axes(geom, nd):
    return ([i for i, t in enumerate(geom["types"]) if t == "space"]
            or [i for i, n in enumerate(geom["names"]) if n in ("x", "y", "z")]
            or list(range(max(0, nd - 3), nd)))


def map_levels(levels):
    """Per level: log10 sizes over the spatial axes, with the physical size and corner of a map voxel."""
    out = []
    for L in levels:
        g, nd = L["geom"], L["sizes"].ndim
        sp = spatial_axes(g, nd)
        other = tuple(i for i in range(nd) if i not in sp)
        with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            a = np.nanmax(L["sizes"], axis=other) if other else L["sizes"]
            d = np.log10(np.where(a > 0, a, np.nan)).astype("<f4")
        out.append(dict(data=d, names=[g["names"][i] for i in sp], units=[g["units"][i] for i in sp],
                        voxel=[float(g["scale"][i] * L["chunks"][i]) for i in sp],
                        corner=[float(g["corner"][i]) for i in sp]))
    return out


def write_zarr2(path, arr, chunk=128):
    os.makedirs(path)
    ch = [min(n, chunk) for n in arr.shape]
    json.dump({"zarr_format": 2, "shape": list(arr.shape), "chunks": ch, "dtype": "<f4",
               "compressor": None, "fill_value": -1.0, "order": "C", "filters": None,
               "dimension_separator": "/"}, open(f"{path}/.zarray", "w"))
    arr = np.nan_to_num(arr, nan=-1.0)  # -1 marks missing chunks
    for idx in np.ndindex(*[-(-n // c) for n, c in zip(arr.shape, ch)]):
        part = arr[tuple(slice(i * c, (i + 1) * c) for i, c in zip(idx, ch))]
        blk = np.full(ch, -1.0, "<f4")
        blk[tuple(slice(0, n) for n in part.shape)] = part
        f = f"{path}/" + "/".join(map(str, idx))
        os.makedirs(os.path.dirname(f), exist_ok=True)
        open(f, "wb").write(blk.tobytes())


def write_ome_map(mlev, outdir):
    """One OME-Zarr 0.4 pyramid, a level per mapped source level, finest first."""
    axes = [{"name": n, "type": "space", **({"unit": u} if u else {})}
            for n, u in zip(mlev[0]["names"], mlev[0]["units"])]
    datasets = []
    for k, L in enumerate(mlev):
        # OME translations are voxel centers: center of map voxel 0 = corner + half a map voxel.
        center = [c + 0.5 * v for c, v in zip(L["corner"], L["voxel"])]
        datasets.append({"path": str(k), "coordinateTransformations": [
            {"type": "scale", "scale": L["voxel"]}, {"type": "translation", "translation": center}]})
        write_zarr2(f"{outdir}/{k}", L["data"])
    json.dump({"zarr_format": 2}, open(f"{outdir}/.zgroup", "w"))
    json.dump({"multiscales": [{"version": "0.4", "name": "chunk sizes", "axes": axes,
                                "datasets": datasets}]}, open(f"{outdir}/.zattrs", "w"))


def serve_dir(directory, bind, port):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            super().end_headers()

        def log_message(self, *args):
            pass

    srv = http.server.ThreadingHTTPServer((bind, port), functools.partial(Handler, directory=directory))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    host = socket.gethostname() if bind in ("0.0.0.0", "") else bind
    return f"http://{host}:{srv.server_address[1]}"


def neuroglancer_viewer(levels, data_source, bind, port, map_port):
    try:
        import neuroglancer
    except ImportError:
        sys.exit("--ng needs the neuroglancer package: pip install neuroglancer")
    mlev = map_levels(levels)
    outdir = tempfile.mkdtemp(prefix="chunkmap_")
    write_ome_map(mlev, outdir)
    map_url = serve_dir(outdir, bind, map_port)
    fin = mlev[0]["data"]
    lo, hi = np.nanpercentile(fin, [1, 99]) if np.isfinite(fin).any() else (0.0, 1.0)
    neuroglancer.set_server_bind_address(bind, port)
    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        s.layers["data"] = neuroglancer.ImageLayer(source=data_source)
        s.layers["chunk size"] = neuroglancer.ImageLayer(source="zarr://" + map_url, shader=SHADER % (lo, hi))
        s.selected_layer.layer = "chunk size"
        s.selected_layer.visible = True
    return viewer, map_url


# ---------- output ----------
def render(sizes, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    a = sizes.copy()
    while a.ndim > 3:  # collapse leading t/c axes
        a = np.nanmax(a, axis=0)
    while a.ndim < 3:
        a = a[None]
    pos = np.where(a > 0, a, np.nan)
    la = np.log10(pos)
    lo, hi = np.nanpercentile(la, [1, 99]) if np.isfinite(la).any() else (0.0, 1.0)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#3a6ea5")  # missing chunks
    z, y, x = la.shape
    views = [(f"XY slice, z chunk {z // 2}", la[z // 2]),
             (f"XZ slice, y chunk {y // 2}", la[:, y // 2, :]),
             (f"YZ slice, x chunk {x // 2}", la[:, :, x // 2]),
             ("XY, mean size over z", np.log10(np.nanmean(pos, axis=0)))]
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (t, img) in zip(axs.flat, views):
        im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(t)
    axs.flat[4].imshow(np.isnan(a).mean(axis=0), cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    axs.flat[4].set_title("XY, fraction of missing chunks over z")
    v = pos[np.isfinite(pos)]
    if v.size:
        axs.flat[5].hist(np.log10(v), bins=60)
    axs.flat[5].set_title("Compressed chunk sizes")
    axs.flat[5].set_xlabel("log10 bytes")
    fig.colorbar(im, ax=axs[0, :].tolist(), shrink=0.8, label="log10 compressed bytes (blue = missing)")
    fig.suptitle(title)
    fig.savefig(out, dpi=120, bbox_inches="tight")


def zarr_level(arr_url, v, m, a):
    shape, chunks = shape_chunks(v, m)
    grid = tuple(-(-n // c) for n, c in zip(shape, chunks))
    key_of, parse = key_funcs(v, m)
    sizes = np.full(grid, np.nan)
    sep = key_of((0, 0))[len(key_of((0,))):-1] if len(grid) > 1 else None
    prefixes = [key_of((i,)) + sep for i in range(grid[0])] if sep else None
    listed = try_list(arr_url, prefixes, a.threads)
    if listed is not None:
        how = "bucket listing"
        for rel, sz in listed:
            if rel in META:
                continue
            idx = parse(rel)
            if idx and len(idx) == len(grid) and all(0 <= i < g for i, g in zip(idx, grid)):
                sizes[idx] = sz
    else:
        how = "HEAD requests"
        idxs = list(np.ndindex(grid))
        prog = Progress("HEAD requests", len(idxs))
        with ThreadPoolExecutor(a.threads) as ex:
            for idx, sz in zip(idxs, ex.map(lambda i: remote_size(f"{arr_url}/{key_of(i)}"), idxs)):
                sizes[idx] = sz
                prog.add()
        prog.done()
    if a.inner:
        sizes, chunks = shard_inner_sizes(arr_url, key_of, sizes, m, a.threads)
        how += " + shard indexes"
    return sizes, list(chunks), shape, how


def zarr_levels(a):
    url = a.url
    v, m = array_meta(url)
    if m is not None:
        sizes, chunks, shape, how = zarr_level(url, v, m, a)
        return [dict(sizes=sizes, chunks=chunks, geom=plain_geometry(m, len(shape)),
                     label=f"{url}  (zarr v{v})", shape=shape, how=how)]
    ms = group_attrs(url).get("multiscales")
    if not ms:
        sys.exit(f"No zarr array, OME multiscales or precomputed info found at {url}")
    ms = ms[0]
    paths = [d["path"] for d in ms["datasets"]]  # finest first, per OME-Zarr
    if a.level is not None and a.level not in paths:
        sys.exit(f"Level {a.level!r} not found; levels are {paths}")
    metas = [(i, p, *array_meta(f"{url}/{p}")) for i, p in enumerate(paths)
             if a.level is None or p == a.level]
    metas = [t for t in metas if t[3] is not None]
    keep = [t for t in metas if a.level is not None or n_chunks(t[2], t[3]) <= a.max_chunks] or metas[-1:]
    skipped = [t[1] for t in metas if t not in keep]
    if skipped:
        print(f"Skipping levels over --max-chunks: {skipped}", file=sys.stderr)
    out = []
    for i, p, v, m in keep:
        print(f"Level {p}: {n_chunks(v, m)} chunks", file=sys.stderr)
        t0 = time.time()
        sizes, chunks, shape, how = zarr_level(f"{url}/{p}", v, m, a)
        out.append(dict(sizes=sizes, chunks=chunks, geom=level_geometry(ms, i, len(shape)),
                        label=f"{url}/{p}  (zarr v{v}, level {p})", shape=shape, how=how,
                        seconds=time.time() - t0))
    return out


# ---------- N5 ----------
def n5_level(arr_url, m, geom, a, label):
    """One N5 dataset. N5 lists dimensions fastest-first (x, y, z) and keys blocks as
    'i0/i1/i2' in that order; the map is returned slowest-first like zarr (z, y, x)."""
    dims, block = m["dimensions"], m["blockSize"]
    grid = [-(-n // c) for n, c in zip(dims, block)]
    sizes = np.full(grid, np.nan)
    listed = try_list(arr_url, [f"{i}/" for i in range(grid[0])], a.threads)
    if listed is not None:
        how = "bucket listing"
        for rel, sz in listed:
            try:
                idx = tuple(int(t) for t in rel.split("/"))
            except ValueError:
                continue
            if len(idx) == len(grid) and all(0 <= i < g for i, g in zip(idx, grid)):
                sizes[idx] = sz
    else:
        how = "HEAD requests"
        idxs = list(np.ndindex(*grid))
        prog = Progress("HEAD requests", len(idxs))
        with ThreadPoolExecutor(a.threads) as ex:
            for idx, sz in zip(idxs, ex.map(lambda i: remote_size(f"{arr_url}/" + "/".join(map(str, i))), idxs)):
                sizes[idx] = sz
                prog.add()
        prog.done()
    return dict(sizes=sizes.transpose(), chunks=block[::-1], geom=geom, label=label,
                shape=dims[::-1], how=how)


def n5_geometry(attrs, nd, group=None, level=0):
    """Physical scale and lower corner (slowest-first) of an N5 dataset.

    Understands the COSEM 'transform' (scale, translate = voxel centre, axes listed
    slowest-first), 'pixelResolution' / 'resolution' (fastest-first), and n5-viewer
    'downsamplingFactors' (corner kept at the origin at every level)."""
    t = attrs.get("transform")
    if t and "scale" in t:
        sc = np.array(t["scale"], float)
        tr = np.array(t.get("translate", [0.0] * nd), float)
        names = t.get("axes") or ["z", "y", "x"][-nd:]
        units = t.get("units") or [None] * nd
        return dict(names=list(names), types=["space"] * nd, units=list(units), scale=sc, corner=tr - 0.5 * sc)
    res = attrs.get("pixelResolution")
    unit = None
    if isinstance(res, dict):
        unit, res = res.get("unit"), res.get("dimensions")
    res = res or attrs.get("resolution") or (group or {}).get("resolution") or [1.0] * nd
    f = attrs.get("downsamplingFactors")
    if f is None and group and "scales" in group:
        f = group["scales"][level]
    sc = np.array(res, float)[::-1]
    if f is not None and not attrs.get("pixelResolution"):
        sc = sc * np.array(f, float)[::-1]
    names = (attrs.get("axes") or (group or {}).get("axes") or ["x", "y", "z"][:nd])[::-1]
    return dict(names=list(names), types=["space"] * nd, units=[unit] * nd, scale=sc, corner=np.zeros(nd))


def n5_levels(a, attrs):
    url = a.url
    if "dimensions" in attrs:
        nd = len(attrs["dimensions"])
        return [n5_level(url, attrs, n5_geometry(attrs, nd), a, f"{url}  (n5)")]
    ms = attrs.get("multiscales")
    if ms:
        paths = [d["path"] for d in ms[0]["datasets"]]
    else:  # n5-viewer style: s0, s1, ... with optional 'scales'
        paths = [f"s{i}" for i in range(len(attrs.get("scales", []) or range(64)))]
    if a.level is not None:
        paths = [p for p in paths if p == a.level] or sys.exit(f"Level {a.level!r} not found")
    metas = []
    for i, p in enumerate(paths):
        m = get_json(f"{url}/{p}/attributes.json")
        if m is None or "dimensions" not in m:
            if ms:
                continue
            break
        metas.append((i, p, m))
    if not metas:
        sys.exit(f"No N5 dataset or multiscale group found at {url}")
    nch = lambda m: int(np.prod([-(-n // c) for n, c in zip(m["dimensions"], m["blockSize"])]))
    keep = [t for t in metas if a.level is not None or nch(t[2]) <= a.max_chunks] or metas[-1:]
    skipped = [t[1] for t in metas if t not in keep]
    if skipped:
        print(f"Skipping levels over --max-chunks: {skipped}", file=sys.stderr)
    out = []
    for i, p, m in keep:
        print(f"Level {p}: {nch(m)} chunks", file=sys.stderr)
        t0 = time.time()
        nd = len(m["dimensions"])
        geom = n5_geometry(ms[0]["datasets"][i] if ms else m, nd, attrs, i)
        out.append(n5_level(f"{url}/{p}", m, geom, a, f"{url}/{p}  (n5, level {p})"))
        out[-1]["seconds"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("--level", help="map only this level path or precomputed scale key")
    ap.add_argument("--max-chunks", type=int, default=50000,
                    help="skip levels with more chunks than this (default 50000)")
    ap.add_argument("--inner", action="store_true", help="zarr v3 sharded: map inner chunks via shard indexes")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--out", default="chunkmap.png")
    ap.add_argument("--ng", action="store_true", help="serve the map as a Neuroglancer layer over the data")
    ap.add_argument("--ng-bind", default="127.0.0.1", help="use 0.0.0.0 to reach it from another machine")
    ap.add_argument("--ng-port", type=int, default=0, help="viewer port (default: any free port)")
    ap.add_argument("--ng-map-port", type=int, default=0, help="port serving the chunk-size pyramid")
    a = ap.parse_args()
    a.url = normalize_url(a.url)

    info = get_json(a.url + "/info")
    n5 = None if info else get_json(a.url + "/attributes.json")
    if info and "scales" in info:
        levels = precomputed_levels(a.url, info, a)
        data_source = "precomputed://" + a.url
    elif n5 and ("dimensions" in n5 or "multiscales" in n5 or "scales" in n5
                 or get_json(a.url + "/s0/attributes.json")):
        levels = n5_levels(a, n5)
        data_source = "n5://" + a.url
    else:
        levels = zarr_levels(a)
        data_source = "zarr://" + a.url

    print(f"{'level':<44} {'chunk grid':>18} {'present':>9} {'compressed GB':>14} {'time':>7}   via")
    for L in levels:
        sz = L["sizes"]
        print(f"{L['label'][-44:]:<44} {'x'.join(map(str, sz.shape)):>18} "
              f"{100 * np.isfinite(sz).mean():>8.1f}% {np.nansum(sz) / 1e9:>14.3f} "
              f"{L.get('seconds', 0):>6.0f}s   {L['how']}")

    F = levels[0]
    sizes = F["sizes"]
    present = np.isfinite(sizes)
    print(f"\nFinest mapped: {F['label']}")
    print(f"Shape {F['shape']}, chunk shape {F['chunks']}, grid {sizes.shape}, "
          f"{present.sum()} of {sizes.size} chunks present")
    if present.any():
        flat = sizes.ravel()
        order = np.argsort(np.where(np.isnan(flat), -1, flat))
        print("Largest chunks (index: bytes):")
        for k in order[::-1][:5]:
            print(f"  {tuple(int(i) for i in np.unravel_index(k, sizes.shape))}: {int(flat[k])}")
        print(f"Median {np.nanmedian(flat):.0f} bytes; min present {np.nanmin(flat):.0f}")
    np.save(a.out.rsplit(".", 1)[0] + "_sizes.npy", sizes)
    render(sizes, a.out, f"{F['label']}\ncompressed chunk sizes ({F['how']})")
    print(f"Wrote {a.out}")
    if a.ng:
        viewer, map_url = neuroglancer_viewer(levels, data_source, a.ng_bind, a.ng_port, a.ng_map_port)
        print(f"Chunk-size pyramid ({len(levels)} levels) served at {map_url}")
        print(f"Neuroglancer viewer: {viewer}\nLeave this running; Ctrl-C to stop.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    main()
