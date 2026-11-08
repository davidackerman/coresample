"""Test 9: do several examples find more than one, and how should they be combined?

The page lets you add examples to a query (shift-click). Earth Engine's similarity search and
similar.earth average the examples' embeddings into one query; the alternative is to keep them
apart and score each candidate by its best match among them. Same 30 queries and brute-force
embeddings as tests 5-8 (CLS + mean, 112 px): for each query, its k - 1 extra examples are other
queries of the list (instances elsewhere in the region, drawn at random, nested: the 3 include
the 2). Hits exclude every example's own patch. Reported: p@10 / p@25, and the paired
difference from one example (mean +/- standard error over the 30 queries).

    uv run python test9_examples.py [--targets mito vesicle mito_full]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import test2_coarse2fine as t2
import test6_models as t6
from common import CACHE, RESULTS, Mirror

KS = [1, 2, 3, 5, 8]
REGIONS = {"sub": (np.array([640, 0, 512]), np.array([2688, 1600, 4608])),
           "full": (np.zeros(3, int), np.array([6368, 1600, 12000]))}
PART = 384


def unit(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)


def unit_parts(q):
    return np.concatenate([unit(q[..., i:i + PART]) for i in range(0, q.shape[-1], PART)], -1)


def run(target, m, data, log):
    name, full = ("mito", True) if target == "mito_full" else (target, False)
    cfg = t2.TARGETS[name]
    W, P = cfg["working"], cfg["patch"]
    data.working = W
    t2.REGION_S0 = REGIONS["full" if full else "sub"]
    rlo, rhi = t2.REGION_S0[0] // 2 ** W, t2.REGION_S0[1] // 2 ** W
    base = json.loads((RESULTS / f"test5_{target}_n30.json").read_text())
    queries = [np.asarray(q, int) for q in base["queries"]]
    cents = t2.instances(data.mirror, cfg["label"], W, rlo, rhi, cfg["min_voxels"])["centroids"]
    gc = t2.grid_centers(rlo, rhi, P)
    z = np.load(CACHE / f"test5_{target}_112.npz")
    E = np.concatenate([unit(z["cls"].astype(np.float32)), unit(z["mean"].astype(np.float32))], 1)
    Q = m(np.stack([data.read(W, q - [0, P // 2, P // 2], q + [1, P // 2, P // 2])[0] for q in queries]))
    Q = unit_parts(np.asarray(Q, np.float32))
    n = len(queries)
    rows = {f"{how} k={k}": [] for k in KS for how in ("mean", "max") if not (k == 1 and how == "max")}
    for i in range(n):
        others = np.random.default_rng(i).permutation([j for j in range(n) if j != i])
        for k in KS:
            ex = [i, *others[: k - 1]]
            mask = np.ones(len(gc), bool)
            for j in ex:
                mask &= ~np.all(np.abs(gc - queries[j]) < P / 2, 1)
            for how in ("mean", "max"):
                if k == 1 and how == "max":
                    continue
                s = E @ unit_parts(Q[ex].mean(0)) if how == "mean" else (E @ Q[ex].T).max(1)
                e = t2.evaluate(t2.rank_nms(gc[mask], s[mask], P), cents, P)
                rows[f"{how} k={k}"].append({"p@10": e["p@10"], "p@25": e["p@25"]})
    one = rows["mean k=1"]
    log(f"[{target}] {len(cents)} instances, {len(gc)} candidates, {n} queries; p@10, p@25 and the change from one example (+/- SE):")
    out = {}
    for key, r in rows.items():
        p10 = np.array([x["p@10"] for x in r]); p25 = np.array([x["p@25"] for x in r])
        d10 = p10 - np.array([x["p@10"] for x in one]); d25 = p25 - np.array([x["p@25"] for x in one])
        se = lambda d: d.std(ddof=1) / np.sqrt(len(d))  # noqa: E731
        out[key] = {"p@10": p10.mean(), "p@25": p25.mean(), "d10": d10.mean(), "se10": se(d10), "d25": d25.mean(), "se25": se(d25), "per_query": r}
        log(f"   {key:<10} p@10 {p10.mean():.2f}  p@25 {p25.mean():.2f}   change {d10.mean():+.3f}+/-{se(d10):.3f}  {d25.mean():+.3f}+/-{se(d25):.3f}")
    (RESULTS / f"test9_{target}.json").write_text(json.dumps({"target": target, "ks": KS, "queries": [q.tolist() for q in queries], "rows": out}, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=["mito", "vesicle", "mito_full"])
    args = ap.parse_args()
    logf = open(RESULTS / "test9.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    os.environ.setdefault("TORCH_THREADS", "4")
    m, data = t6.Matcher("dinov2-small"), t2.Data(Mirror())
    for t in args.targets:
        run(t, m, data, log)


if __name__ == "__main__":
    main()
