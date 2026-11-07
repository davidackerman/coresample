"""Search any multiscale volume for regions like a clicked point, using raw data only.

No labels, no precomputation: the pyramid already in the store is the index.
Reads go straight to the store with tensorstore (zarr v2, zarr v3, N5 over
http(s)/s3/gs/file), cheap scores rank the coarsest level, the top-K candidates are
refined level by level, and DINOv2-small re-ranks the survivors at the working level.

    uv run python query.py zarr://s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8 \\
        --point 2500,800,4000 --working 2 --coarsest 4 --patch 64 --k 100 --top 20

--point is a voxel of the finest level (s0), in the array's stored axis order.
Writes results/query_<name>.json and prints a Neuroglancer link with the hits.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
from pathlib import Path

import numpy as np
import tensorstore as ts

import search

ROOT = Path(__file__).resolve().parent


def normalize(url: str) -> str:
    url = re.sub(r"^(zarr[23]?|n5)://", "", url.strip()).rstrip("/")
    if url.startswith("s3://"):
        b, _, k = url[5:].partition("/")
        return f"https://{b}.s3.amazonaws.com/{k}"
    if url.startswith("gs://"):
        return "https://storage.googleapis.com/" + url[5:]
    return url


def kvstore(url: str):
    if url.startswith("http"):
        return {"driver": "http", "base_url": url + "/"}
    return {"driver": "file", "path": url.replace("file://", "") + "/"}


def _json(url):
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.loads(r.read())
    except Exception:
        return None


def discover_levels(url: str):
    """[(path, driver, scale)] finest first, from OME-Zarr or N5 multiscale metadata."""
    for name, driver in ((".zattrs", "zarr"), ("zarr.json", "zarr3"), ("attributes.json", "n5")):
        m = _json(f"{url}/{name}") if url.startswith("http") else (
            json.loads(Path(url, name).read_text()) if Path(url, name).exists() else None)
        if not m:
            continue
        attrs = m.get("attributes", m)
        attrs = attrs.get("ome", attrs)
        ms = attrs.get("multiscales")
        if not ms:
            continue
        out = []
        for d in ms[0]["datasets"]:
            sc = None
            for t in d.get("coordinateTransformations", []):
                if t["type"] == "scale":
                    sc = t["scale"]
            if sc is None and "transform" in d:
                sc = d["transform"]["scale"]
            out.append((d["path"], driver, np.array(sc or [1, 1, 1], float)))
        return out
    raise SystemExit(f"no multiscale metadata found at {url}")


class Reader:
    def __init__(self, url: str):
        self.url = url
        self.levels = discover_levels(url)
        self.ctx = ts.Context({"cache_pool": {"total_bytes_limit": 2 << 30}})
        self.stores = {}
        self.bytes_voxels = 0
        self.window = None
        f = [self.levels[i + 1][2] / self.levels[i][2] for i in range(len(self.levels) - 1)]
        for i, r in enumerate(f):
            if not np.allclose(r[-3:], 2):
                raise SystemExit(f"level {i + 1} is not a 2x downsampling of level {i} (ratio {r}); "
                                 "coarse_to_fine assumes factor-2 pyramids")

    def store(self, level):
        if level not in self.stores:
            path, driver, _ = self.levels[level]
            spec = {"driver": driver, "kvstore": kvstore(f"{self.url}/{path}")}
            st = ts.open(spec, read=True, context=self.ctx).result()
            if driver == "n5":  # tensorstore lists N5 dims fastest-first
                st = st[ts.d[::-1].transpose[:]]
            self.stores[level] = st
        return self.stores[level]

    def read(self, level, lo, hi):
        st = self.store(level)
        shape = np.array(st.shape[-3:])
        lo, hi = np.asarray(lo, int), np.asarray(hi, int)
        out = np.zeros(tuple(hi - lo), dtype=st.dtype.numpy_dtype)
        a, b = np.maximum(lo, 0), np.minimum(hi, shape)
        if np.all(b > a):
            sub = st[..., a[0]:b[0], a[1]:b[1], a[2]:b[2]].read().result()
            while sub.ndim > 3:
                sub = sub[0]
            out[tuple(slice(x - l, y - l) for x, y, l in zip(a, b, lo))] = sub
            self.bytes_voxels += int(np.prod(b - a))
        if out.dtype != np.uint8:  # one global uint8 window (from the coarsest level) for every read
            if self.window is None:
                c = self.store(len(self.levels) - 1)
                v = c.read().result().astype(np.float32)
                self.window = np.percentile(v[v != 0] if (v != 0).any() else v, [0.5, 99.5])
            lo_, hi_ = self.window
            out = np.clip((out.astype(np.float32) - lo_) / max(hi_ - lo_, 1e-6) * 255, 0, 255).astype(np.uint8)
        return out


def neuroglancer_link(url, levels, query_s0, hits_s0, scores):
    sc = levels[0][2]
    names = ["z", "y", "x"]
    dims = {n: [float(s) * 1e-9, "m"] for n, s in zip(names, sc)}
    anns = [{"type": "point", "id": "query", "point": [float(v) + 0.5 for v in query_s0],
             "description": "query"}]
    for i, (h, s) in enumerate(zip(hits_s0, scores)):
        anns.append({"type": "point", "id": f"hit{i}", "point": [float(v) + 0.5 for v in h],
                     "description": f"#{i + 1} cos={s:.3f}"})
    src = ("zarr://" if levels[0][1] == "zarr" else "zarr3://" if levels[0][1] == "zarr3" else "n5://") + url
    state = {
        "dimensions": dims,
        "position": [float(v) + 0.5 for v in query_s0],
        "crossSectionScale": 2,
        "layers": [
            {"type": "image", "source": src, "name": "data"},
            {"type": "annotation", "name": "hits", "source": {"url": "local://annotations",
                                                              "transform": {"outputDimensions": dims}},
             "annotations": anns},
        ],
        "selectedLayer": {"layer": "hits", "visible": True},
        "layout": "4panel",
    }
    return "https://neuroglancer-demo.appspot.com/#!" + urllib.parse.quote(json.dumps(state, separators=(",", ":")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("--point", required=True, help="query voxel at s0, z,y,x")
    ap.add_argument("--working", type=int, default=2, help="level where DINOv2 ranks candidates")
    ap.add_argument("--coarsest", type=int, default=None, help="level scanned whole (default working+2)")
    ap.add_argument("--patch", type=int, default=64, help="query box edge in working-level voxels")
    ap.add_argument("--k", type=int, default=100, help="candidates kept per level")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--region", default=None, help="s0 box zlo,ylo,xlo,zhi,yhi,xhi (default: whole array)")
    ap.add_argument("--methods", default="ncc,hist,tex")
    ap.add_argument("--no-embed", action="store_true", help="stop after cheap scores (no model)")
    a = ap.parse_args()

    url = normalize(a.url)
    rd = Reader(url)
    W, P = a.working, a.patch
    coarsest = a.coarsest if a.coarsest is not None else min(W + 2, len(rd.levels) - 1)
    f = 2**W
    q0 = np.array([int(v) for v in a.point.split(",")])
    qc = q0 // f
    shape0 = np.array(rd.store(0).shape[-3:])
    if a.region:
        r = np.array([int(v) for v in a.region.split(",")])
        region = (r[:3] // f, -(-r[3:] // f))
    else:
        region = (np.zeros(3, int), -(-shape0 // f))
    t0 = time.perf_counter()
    stages = search.coarse_to_fine(rd.read, (qc - P // 2, qc + P // 2), region, W, coarsest, P, a.k,
                                   tuple(a.methods.split(",")))
    for st in stages:
        print(f"level s{st.level}: {len(st.centers)} candidates, {st.seconds:.1f}s")
    cand = search.to_working(stages[-1], W)
    scores = stages[-1].scores
    if not a.no_embed and len(cand):
        from embed import Embedder

        emb = Embedder()
        h = P // 2
        get = lambda cs: np.stack([rd.read(W, c - [0, h, h], c + [1, h, h])[0] for c in cs])
        qe = emb.embed_at(url, W, qc[None], P, get)[0]
        E = emb.embed_at(url, W, cand, P, get)
        scores = search.rank_by_embedding(qe, E)
    notself = ~np.all(np.abs(cand - qc) < P / 2, 1)
    order = np.argsort(-scores[notself], kind="stable")
    hits = search.nms(cand[notself][order], np.array([P] * 3), a.top)
    lut = {tuple(c): s for c, s in zip(cand[notself], scores[notself])}
    hs = [lut[tuple(h)] for h in hits]
    hits_s0 = hits * f + f // 2
    dt = time.perf_counter() - t0
    vox_w = np.prod(np.asarray(region[1]) - np.asarray(region[0]))
    print(f"\n{len(hits)} hits in {dt:.1f}s; voxels read {rd.bytes_voxels / 1e6:.1f}M "
          f"({rd.bytes_voxels / vox_w:.1%} of the working level over the region)")
    for i, (h, s) in enumerate(zip(hits_s0, hs)):
        print(f"  #{i + 1:<3} s0 voxel {tuple(int(v) for v in h)}  score {s:.3f}")
    name = re.sub(r"[^A-Za-z0-9]+", "_", url.split("//")[-1])[-60:]
    out = ROOT / "results" / f"query_{name}.json"
    out.parent.mkdir(exist_ok=True)
    link = neuroglancer_link(url, rd.levels, q0, hits_s0, hs)
    out.write_text(json.dumps({"url": url, "point_s0": q0.tolist(), "working": W, "coarsest": coarsest, "patch": P,
                               "k": a.k, "seconds": dt, "hits_s0": hits_s0.tolist(), "scores": [float(s) for s in hs],
                               "neuroglancer": link}, indent=1))
    print(f"\nwrote {out}\nNeuroglancer: {link[:120]}... (full link in the json)")


if __name__ == "__main__":
    main()
