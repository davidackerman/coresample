"""Test 3: combine the cheap signals instead of choosing one.

Strategies, all reading whole working-level chunks and embedding every grid patch in
each chunk they read (the same patches and embeddings brute force uses, from the
cache), so they differ only in which chunks they read and in what order:

  random          chunks in random order
  adaptive        1% random seeds, then best-first expansion around the best hits
  adaptive+skip   adaptive, never reading chunks the free size map calls empty
  coarse          chunks in order of the cheap coarse scan (hist+tex), no feedback
  combined        skip empty chunks, seed with the top 1% by coarse scan (+0.5% random),
                  expand best-first on coarse prior + neighbours' similarity

Empty chunks come from the s0 chunk sizes alone (Otsu on log size, no labels). The cost
of a strategy is the bytes of the working-level chunks it read (from the listing) plus,
for coarse-using strategies, the whole coarsest level over the region.

    uv run python experiments/test3_combined.py [--targets mito vesicle] [--region full --suffix _full]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import search
import test2_coarse2fine as t2
from common import RESULTS, Mirror
from embed import Embedder

FRACS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5]
STRATEGIES = ["random", "adaptive", "adaptive+skip", "coarse", "combined"]


def working_chunk_grid(data, W, gc):
    c = np.array(data.mirror.info(data.path(W)).chunks)
    keys, inv = np.unique(gc // c, axis=0, return_inverse=True)
    return keys, inv.ravel(), c


def empty_mask(data, W, keys, c):
    """Per working chunk: mean log10 s0 chunk size over the s0 chunks it covers; empty =
    lower Otsu class."""
    s0 = data.chunk_sizes(0).astype(float)
    c0 = np.array(data.mirror.info(data.path(0)).chunks)
    span = c * 2**W  # s0 voxels per working chunk
    logs = np.log10(np.maximum(s0, 1))
    out = np.zeros(len(keys))
    for i, k in enumerate(keys):
        a = k * span // c0
        b = np.maximum(a + 1, -(-(k + 1) * span // c0))
        blk = logs[a[0]:b[0], a[1]:b[1], a[2]:b[2]]
        out[i] = blk.mean() if blk.size else 0.0
    return search.empty_chunks(out), out


def run(name, cfg, args, data, emb, log):
    W, P = cfg["working"], cfg["patch"]
    data.working = W
    region_s0 = t2.REGION_S0
    rlo, rhi = region_s0[0] // 2**W, region_s0[1] // 2**W
    prev = json.loads((RESULTS / f"test2_{name}{args.suffix}.json").read_text())
    queries = [np.array(q["center_working"]) for q in prev["queries"]][: args.queries]
    gt = t2.instances(data.mirror, cfg["label"], W, rlo, rhi, cfg["min_voxels"])
    cents = gt["centroids"]
    gc = t2.grid_centers(rlo, rhi, P)
    E = emb.embed_at(data.path(W), W, gc, P, lambda cs: t2.patches_read(data, W, cs, P))
    keys, inv, c = working_chunk_grid(data, W, gc)
    n = len(keys)
    empty, logsz = empty_mask(data, W, keys, c)
    sizes_w = data.chunk_sizes(W)
    chunk_bytes = np.array([sizes_w[tuple(k)] for k in keys], float)
    total = chunk_bytes.sum()
    Lc = cfg["default_coarsest"]
    fc = 2 ** (Lc - W)
    coarse_box = [(data.path(Lc), rlo // fc - 4, -(-rhi // fc) + 4)]
    overhead = data.cost(coarse_box, region_s0)["bytes"] / data.cost([(data.path(W), rlo, rhi)], region_s0)["bytes"]
    log(f"[{name}{args.suffix}] {n} working chunks, {empty.sum()} called empty ({empty.mean():.0%}), "
        f"coarse s{Lc} overhead {overhead:.2%} of working bytes, {len(cents)} instances")

    out = {"target": name, "region_s0": [r.tolist() for r in region_s0], "n_chunks": int(n),
           "empty_fraction": float(empty.mean()), "coarse_level": Lc, "coarse_overhead": overhead,
           "queries": [], "curves": {s: [] for s in STRATEGIES}}
    budget = int(round(max(FRACS) * n))
    for qi, qc in enumerate(queries):
        qbox = (qc - P // 2, qc + P // 2)
        qe = emb.embed_at(data.path(W), W, qc[None], P, lambda cs: t2.patches_read(data, W, cs, P))[0]
        sims = search.rank_by_embedding(qe, E)
        notself = ~np.all(np.abs(gc - qc) < P / 2, 1)
        best = np.full(n, -np.inf)
        np.maximum.at(best, inv, np.where(notself, sims, -np.inf))
        t = time.perf_counter()
        comb, centers = search.coarse_scan(data.read, qbox, (rlo, rhi), W, Lc, P)
        t_coarse = time.perf_counter() - t
        prior = np.full(n, -np.inf)
        cid = {tuple(k): i for i, k in enumerate(keys)}
        flat_c, flat_s = centers.reshape(-1, 3), comb.ravel()
        inside = np.all((flat_c >= rlo) & (flat_c < rhi), 1)
        for cc, s in zip(flat_c[inside] // c, flat_s[inside]):
            j = cid.get(tuple(cc))
            if j is not None and s > prior[j]:
                prior[j] = s
        prior[~np.isfinite(prior)] = np.nanmin(prior[np.isfinite(prior)])
        rng = np.random.default_rng(args.seed + qi)
        ev = lambda i: best[i]
        orders = {
            "random": list(rng.permutation(n)[:budget]),
            "adaptive": search.best_first_chunks(keys, np.zeros(n), ev, budget, seed=0.01, rng=rng),
            "adaptive+skip": search.best_first_chunks(keys, np.where(empty, -np.inf, 0.0), ev, budget, seed=0.01, rng=rng),
            "coarse": search.best_first_chunks(keys, prior, ev, budget, seed=1.0, beta=0.0, rng=rng),
            "combined": search.best_first_chunks(keys, np.where(empty, -np.inf, prior), ev, budget, seed=0.01,
                                                 explore=0.005, rng=rng),
        }
        out["queries"].append({"center_working": qc.tolist(), "coarse_seconds": t_coarse})
        for sname, order in orders.items():
            curve = []
            for f in FRACS:
                k = int(round(f * n))
                sel = np.zeros(n, bool)
                sel[order[:k]] = True
                m = sel[inv] & notself
                top = t2.rank_nms(gc[m], sims[m], P)
                e = t2.evaluate(top, cents, P)
                fb = chunk_bytes[sel].sum() / total + (overhead if sname in ("coarse", "combined") else 0.0)
                curve.append({"frac_chunks": f, "chunks_read": int(sel.sum()), "frac_bytes": float(fb),
                              **{kk: v for kk, v in e.items() if kk[:2] in ("p@", "r@")}})
            out["curves"][sname].append(curve)
        c5 = {s: out["curves"][s][-1][FRACS.index(0.05)] for s in STRATEGIES}
        log(f"[{name}{args.suffix} q{qi}] at 5% of chunks p@10/p@25: " +
            " ".join(f"{s}={c5[s]['p@10']:.2f}/{c5[s]['p@25']:.2f}" for s in STRATEGIES) + f" (coarse scan {t_coarse:.1f}s)")
    (RESULTS / f"test3_{name}{args.suffix}.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=["mito", "vesicle"])
    ap.add_argument("--queries", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--region", default=None)
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    if args.region == "full":
        t2.REGION_S0 = (np.zeros(3, int), np.array([6368, 1600, 12000]))
    logf = open(RESULTS / "test3.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    data = t2.Data(Mirror())
    emb = Embedder()
    for name in args.targets:
        run(name, t2.TARGETS[name], args, data, emb, log)


if __name__ == "__main__":
    main()
