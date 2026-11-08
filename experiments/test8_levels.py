"""Test 8: what should decide which tiles are read first: cheap statistics, or the model one or two
levels coarser?

The page scans a coarse level with cheap statistics (histogram + texture, no model) and reads the
working level's chunks in that order, never cutting any (test 3). Test 2's cascade (cut at every
level) lost mito because the cheap scores cannot see them. The question here is whether the
model itself, on a coarse level, is the better guide: embed one patch per working-chunk-sized
cell at level W+1 (or W+2), the same physical box as the working patch (P/2 or P/4 px there),
rank the working chunks by their best coarse match, and read them in that order.

Every order reads whole working-level chunks and uses the working level's brute-force patch
embeddings for the chunks it has read (test 5's cache, CLS + mean, 112 px), so the orders differ
only in which chunks come first. Same 30 queries as tests 5-7. Reported: p@10/p@25 of the
hits among the chunks read after a fraction of them, and the bytes, including what the order
itself needed (the coarse level for the cheap scan; level W+1 or W+2 for the model).

    uv run python experiments/test8_levels.py [--targets mito vesicle mito_full]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import search
import test2_coarse2fine as t2
import test6_models as t6
from common import CACHE, RESULTS, Mirror
from test3_combined import working_chunk_grid

FRACS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
REGIONS = {"sub": (np.array([640, 0, 512]), np.array([2688, 1600, 4608])),
           "full": (np.zeros(3, int), np.array([6368, 1600, 12000]))}


def unit(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)


def coarse_grid(lo, hi, p):
    """Centres one patch apart (a working chunk is about one cell at W+1 for these targets)."""
    axes = [np.arange(a + p // 2, b - p // 2 + 1, p) for a, b in zip(lo, hi)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


def run(target, m, data, log):
    name, full = ("mito", True) if target == "mito_full" else (target, False)
    cfg = t2.TARGETS[name]
    W, P = cfg["working"], cfg["patch"]
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
    assert len(E) == len(gc), (len(E), len(gc))
    keys, inv, ch = working_chunk_grid(data, W, gc)
    n = len(keys)
    sizes_w = data.chunk_sizes(W)
    chunk_bytes = np.array([sizes_w[tuple(k)] for k in keys], float)
    total = chunk_bytes.sum()
    wbytes = data.cost([(data.path(W), rlo, rhi)], region_s0)["bytes"]
    cid = {tuple(k): i for i, k in enumerate(keys)}
    Q = m(np.stack([data.read(W, q - [0, P // 2, P // 2], q + [1, P // 2, P // 2])[0] for q in queries]))

    # the model one and two levels up: one patch of the same physical box per cell
    coarse = {}
    for up in (1, 2):
        L, p = W + up, P // 2 ** up
        lo, hi = rlo // 2 ** up, -(-rhi // 2 ** up)
        cc = coarse_grid(lo, hi, p)
        t = time.time()
        vol = data.read(L, lo, hi)
        Ec = m(t2.patches_from(vol, lo, cc, p))
        qc = [q // 2 ** up for q in queries]
        Qc = m(np.stack([data.read(L, c - [0, p // 2, p // 2], c + [1, p // 2, p // 2])[0] for c in qc]))
        owner = np.array([cid.get(tuple(k), -1) for k in (cc * 2 ** up + 2 ** up // 2) // ch])
        cost = data.cost([(data.path(L), lo, hi)], region_s0)["bytes"] / wbytes
        coarse[up] = (cc, Ec, Qc, owner, cost)
        log(f"[{target}] model at s{L}: {len(cc)} patches of {p} px ({time.time() - t:.0f}s), level costs {cost:.1%} of s{W}")
        del vol

    Lc = cfg["default_coarsest"]
    fc = 2 ** (Lc - W)
    scan_cost = data.cost([(data.path(Lc), rlo // fc - 4, -(-rhi // fc) + 4)], region_s0)["bytes"] / wbytes
    names = ["random", "cheap scan", "model W+1", "model W+2"]
    curves = {k: [] for k in names}
    for qi, q in enumerate(queries):
        sims = E @ Q[qi]
        notself = ~np.all(np.abs(gc - q) < P / 2, 1)
        prio = {}
        comb, centers = search.coarse_scan(data.read, (q - P // 2, q + P // 2), (rlo, rhi), W, Lc, P)
        pr = np.full(n, -np.inf)
        fl_c, fl_s = centers.reshape(-1, 3), comb.ravel()
        inside = np.all((fl_c >= rlo) & (fl_c < rhi), 1)
        for c, s in zip(fl_c[inside] // ch, fl_s[inside]):
            j = cid.get(tuple(c))
            if j is not None and s > pr[j]:
                pr[j] = s
        prio["cheap scan"] = pr
        for up in (1, 2):
            cc, Ec, Qc, owner, _ = coarse[up]
            s = Ec @ Qc[qi]
            s[np.all(np.abs(cc - q // 2 ** up) < P / 2 ** (up + 1), 1)] = -np.inf  # the query's own cell
            pr = np.full(n, -np.inf)
            ok = owner >= 0
            np.maximum.at(pr, owner[ok], s[ok])
            prio[f"model W+{up}"] = pr
        rng = np.random.default_rng(qi)
        prio["random"] = rng.random(n)
        for k in names:
            p_ = prio[k] + rng.random(n) * 1e-9  # ties broken at random
            order = np.argsort(-p_, kind="stable")
            curve = []
            for f in FRACS:
                sel = np.zeros(n, bool)
                sel[order[: max(1, int(round(f * n)))]] = True
                mm = sel[inv] & notself
                e = t2.evaluate(t2.rank_nms(gc[mm], sims[mm], P), cents, P)
                extra = {"cheap scan": scan_cost, "model W+1": coarse[1][4], "model W+2": coarse[2][4]}.get(k, 0.0)
                curve.append({"frac": f, "frac_bytes": float(chunk_bytes[sel].sum() / total + extra),
                              "p@10": e["p@10"], "p@25": e["p@25"]})
            curves[k].append(curve)
    allread = t2.evaluate(t2.rank_nms(gc, E @ Q[0], P), cents, P)  # sanity: q0 with everything read
    res = {"target": target, "n_chunks": int(n), "fracs": FRACS, "scan_cost": scan_cost,
           "model_cost": {f"W+{u}": coarse[u][4] for u in (1, 2)}, "curves": curves}
    log(f"[{target}] {n} chunks; p@10/p@25 averaged over {len(queries)} queries (bytes incl. the order's own reads):")
    for k in names:
        row = []
        for i, f in enumerate(FRACS):
            p10 = np.mean([c[i]["p@10"] for c in curves[k]])
            p25 = np.mean([c[i]["p@25"] for c in curves[k]])
            fb = np.mean([c[i]["frac_bytes"] for c in curves[k]])
            row.append(f"{f:.1%}: {p10:.2f}/{p25:.2f} ({fb:.1%})")
        log(f"   {k:<11} " + "  ".join(row))
    log(f"   (q0 with every chunk read: p@10 {allread['p@10']:.2f}; the 30-query brute force is p@10 "
        f"{next(r for r in base['rows'] if (r['pool'], r['query'], r['sim'], r['input']) == ('cls+mean', 'single', 'cosine', 112))['p@10']:.2f})")
    (RESULTS / f"test8_{target}.json").write_text(json.dumps(res, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=["mito", "vesicle", "mito_full"])
    args = ap.parse_args()
    logf = open(RESULTS / "test8.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    os.environ.setdefault("TORCH_THREADS", "6")
    m, data = t6.Matcher("dinov2-small"), t2.Data(Mirror())
    for t in args.targets:
        run(t, m, data, log)


if __name__ == "__main__":
    main()
