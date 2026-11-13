"""Test 15: read a tile's neighbour chunks only when its own patches look promising.

In 3-D the page's patches lie half a patch apart, so 3 in 4 cross into the chunk before (in y,
x or both) and a tile read downloads up to 4 chunks; on hemibrain 2/3 of the detail bytes were
neighbours never read as tiles themselves. Here, on labelled HeLa, tiles are read in the rough
pass's order (the model two levels up, as the page) and each policy decides which patches are
compared:

  full         every patch of a tile read (today: its neighbours downloaded)
  inside       only the patches inside the tile (one chunk per tile)
  top-K        inside first; the crossing ones too when the tile's best inside score reaches the
               K-th best score compared so far (neighbours downloaded then)
  quantile-X   the same when the tile's best is in the top 1-X of the tiles read so far

A crossing patch whose chunks all came down anyway (its neighbours read or fetched for another
tile) is compared too, free. Reported: p@10/p@25 against the share of tiles read, and the bytes
(every chunk downloaded, neighbours included) as a share of reading every tile once.
"""
from __future__ import annotations

import heapq
import json
import os
import time

import numpy as np

import search
import test2_coarse2fine as t2
from common import CACHE, RESULTS, Mirror
from test3_combined import working_chunk_grid
from test8_levels import REGIONS, coarse_grid, unit
import test6_models as t6

FRACS = [0.01, 0.02, 0.05, 0.1, 0.2]
POLICIES = ["full", "inside", "top-60", "top-200", "top-600", "quantile-0.5", "quantile-0.75"]


def run(target, m, data, log):
    name, full = ("mito", True) if target == "mito_full" else (target, False)
    cfg = t2.TARGETS[name]
    W, P = cfg["working"], cfg["patch"]
    h = P // 2
    data.working = W
    region_s0 = REGIONS["full" if full else "sub"]
    t2.REGION_S0 = region_s0
    rlo, rhi = region_s0[0] // 2 ** W, region_s0[1] // 2 ** W
    base = json.loads((RESULTS / f"test5_{target}_n30.json").read_text())
    queries = [np.asarray(q, int) for q in base["queries"]]
    gt = t2.instances(data.mirror, cfg["label"], W, rlo, rhi, cfg["min_voxels"])
    cents = gt["centroids"]
    gc = t2.grid_centers(rlo, rhi, P)
    z = np.load(CACHE / f"test5_{target}_112.npz")
    E = np.concatenate([unit(z["cls"].astype(np.float32)), unit(z["mean"].astype(np.float32))], 1)
    keys, inv, ch = working_chunk_grid(data, W, gc)
    n = len(keys)
    sizes = data.chunk_sizes(W)
    # every chunk a patch touches (y, x around its plane), as ids; the tiles are the first n
    cid = {tuple(k): i for i, k in enumerate(keys)}
    extra: dict[tuple, int] = {}

    def chunk_id(k):
        k = tuple(int(v) for v in k)
        if k in cid:
            return cid[k]
        return extra.setdefault(k, n + len(extra))

    need = []      # per patch: the chunk ids it needs besides its own
    crossing = np.zeros(len(gc), bool)
    for i, c in enumerate(gc):
        ks = {(c[0] // ch[0], y // ch[1], x // ch[2]) for y in (c[1] - h, c[1] + h - 1) for x in (c[2] - h, c[2] + h - 1)}
        own = tuple(int(v) for v in c // ch)
        others = [chunk_id(k) for k in ks if tuple(int(v) for v in k) != own]
        need.append(others)
        crossing[i] = bool(others)
    all_keys = list(keys) + list(extra)
    def size_of(k):
        k = tuple(int(v) for v in k)
        return float(sizes[k]) if all(0 <= v < d for v, d in zip(k, sizes.shape)) else 0.0
    cbytes = np.array([size_of(k) for k in all_keys], float)
    total = cbytes[:n].sum()
    by_tile = [[] for _ in range(n)]
    for i, j in enumerate(inv):
        by_tile[j].append(i)
    log(f"[{target}] {n} tiles, {len(gc)} patches, {crossing.mean():.0%} cross into a neighbour; "
        f"reading every tile with its neighbours costs {cbytes[np.unique(np.concatenate([[j for j in range(n)]] + [np.array(v, int) for v in need if v]))].sum() / total:.2f}x the tiles alone")
    Q = m(np.stack([data.read(W, q - [0, h, h], q + [1, h, h])[0] for q in queries]))

    # the rough pass's order: the model two levels up, one patch per cell, best cell per tile
    up = 2
    L, p = W + up, P // 2 ** up
    lo, hi = rlo // 2 ** up, -(-rhi // 2 ** up)
    cc = coarse_grid(lo, hi, p)
    t = time.time()
    vol = data.read(L, lo, hi)
    Ec = m(t2.patches_from(vol, lo, cc, p))
    Qc = m(np.stack([data.read(L, c - [0, p // 2, p // 2], c + [1, p // 2, p // 2])[0] for c in (q // 2 ** up for q in queries)]))
    owner = np.array([cid.get(tuple(k), -1) for k in (cc * 2 ** up + 2 ** up // 2) // ch])
    log(f"[{target}] rough pass at s{L}: {len(cc)} cells ({time.time() - t:.0f}s)")
    del vol

    res = {pol: {f: {"p@10": [], "p@25": [], "bytes": []} for f in FRACS} for pol in POLICIES}
    for qi, q in enumerate(queries):
        sims = E @ Q[qi]
        notself = ~np.all(np.abs(gc - q) < P / 2, 1)
        s = Ec @ Qc[qi]
        s[np.all(np.abs(cc - q // 2 ** up) < P / 2 ** (up + 1), 1)] = -np.inf
        pr = np.full(n, -np.inf)
        ok = owner >= 0
        np.maximum.at(pr, owner[ok], s[ok])
        rng = np.random.default_rng(qi)
        order = np.argsort(-(pr + rng.random(n) * 1e-9), kind="stable")
        marks = {max(1, int(round(f * n))): f for f in FRACS}
        last = max(marks)
        for pol in POLICIES:
            inc = np.zeros(len(gc), bool)
            down = np.zeros(len(cbytes), bool)
            deferred: list[int] = []
            heap: list[float] = []
            bests: list[float] = []
            K = int(pol.split("-")[1]) if pol.startswith("top-") else 0
            X = float(pol.split("-")[1]) if pol.startswith("quantile-") else 0.0
            for r, j in enumerate(order[:last], 1):
                down[j] = True
                ins = [i for i in by_tile[j] if not crossing[i] or pol == "full"]
                crs = [i for i in by_tile[j] if crossing[i] and pol != "full"]
                for i in ins:
                    inc[i] = True
                    for c2 in need[i]:
                        down[c2] = True
                sc = [sims[i] for i in ins if notself[i]]
                best = max(sc) if sc else -np.inf
                if pol in ("full", "inside"):
                    fetch = False
                elif K:
                    fetch = len(heap) < K or best >= heap[0]
                else:
                    fetch = not bests or best >= np.quantile(bests, X)
                bests.append(best)
                newsc = sc
                if fetch and crs:
                    for i in crs:
                        inc[i] = True
                        for c2 in need[i]:
                            down[c2] = True
                    newsc = sc + [sims[i] for i in crs if notself[i]]
                elif crs:
                    deferred.extend(crs)
                if K:
                    for v in newsc:
                        if len(heap) < K:
                            heapq.heappush(heap, v)
                        elif v > heap[0]:
                            heapq.heapreplace(heap, v)
                if r in marks:
                    f = marks[r]
                    free = [i for i in deferred if all(down[c2] for c2 in need[i])]
                    mm = inc.copy()
                    mm[free] = True
                    mm &= notself
                    e = t2.evaluate(t2.rank_nms(gc[mm], sims[mm], P), cents, P)
                    res[pol][f]["p@10"].append(e["p@10"])
                    res[pol][f]["p@25"].append(e["p@25"])
                    res[pol][f]["bytes"].append(cbytes[down].sum() / total)
    log(f"[{target}] p@10/p@25 over {len(queries)} queries after reading a share of the tiles in the rough order "
        f"(bytes, neighbours included, as a share of every tile once):")
    out = {"target": target, "n_tiles": n, "fracs": FRACS, "crossing": float(crossing.mean()), "policies": {}}
    for pol in POLICIES:
        row = []
        out["policies"][pol] = {}
        for f in FRACS:
            a = res[pol][f]
            p10, p25, b = np.mean(a["p@10"]), np.mean(a["p@25"]), np.mean(a["bytes"])
            out["policies"][pol][str(f)] = {"p@10": p10, "p@25": p25, "bytes": b,
                                            "p@10_each": a["p@10"], "p@25_each": a["p@25"]}
            row.append(f"{f:.0%}: {p10:.2f}/{p25:.2f} ({b:.1%})")
        log(f"   {pol:14s} " + "  ".join(row))
    (RESULTS / f"test15_{target}.json").write_text(json.dumps(out, indent=1))


def main():
    logf = open(RESULTS / "test15.log", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    os.environ.setdefault("TORCH_THREADS", "8")
    m, data = t6.Matcher("dinov2-small"), t2.Data(Mirror())
    for target in ("mito", "vesicle", "mito_full"):
        run(target, m, data, log)


if __name__ == "__main__":
    main()
