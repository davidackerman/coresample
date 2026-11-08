"""Test 4: relevance feedback. Does marking a few hits good/bad improve the next ones?

Simulates a person paging through results 10 at a time and marking each shown region
good or bad (graded with the reference labels, as a person would judge by eye). After each
page the ranking of everything not yet shown is recomputed:

  none      cosine similarity to the query (no feedback): just the next 10
  rocchio   query moved toward the good results and away from the bad ones
  logreg    a logistic regression on the DINOv2 embeddings of everything marked so far
            (plus the query as a positive), re-scoring all patches

Patches and embeddings are the brute-force set (cache). Two pools: all patches (feedback
on its own), and only the patches inside the coarse scan's top 2% of chunks (feedback on top
of the test 3 search, a few percent of the data read).

    uv run python experiments/test4_feedback.py [--targets mito vesicle] [--region full --suffix _full]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression

import search
import test2_coarse2fine as t2
import test3_combined as t3
from common import RESULTS, Mirror
from embed import Embedder

PAGES = 5
PAGE = 10
METHODS = ["none", "rocchio", "logreg"]


def contains(centers, cents, P):
    """For each centre: the set of instance indices whose centroid its box contains."""
    from scipy.spatial import cKDTree

    tree = cKDTree(cents)
    return [set(tree.query_ball_point(c, r=P / 2 - 1e-6, p=np.inf)) for c in centers]


def next_page(scores, centers, shown_centers, P, k=PAGE):
    """Best k not-yet-shown, non-overlapping boxes (Chebyshev distance >= P from each other
    and from everything already shown)."""
    order = np.argsort(-scores, kind="stable")
    picked = []
    sc = list(shown_centers)
    for i in order:
        c = centers[i]
        if all(np.any(np.abs(c - s) >= P) for s in sc):
            picked.append(i)
            sc.append(c)
            if len(picked) >= k:
                break
    return picked


def session(method, qe, E, centers, hit_sets, P, rng):
    """One simulated session; returns per-page precision and distinct instances found."""
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    q = qe / np.linalg.norm(qe)
    shown, labels = [], []
    pages = []
    found = set()
    def rocchio():
        pos = [i for i, l in zip(shown, labels) if l]
        neg = [i for i, l in zip(shown, labels) if not l]
        qq = q + (0.75 * En[pos].mean(0) if pos else 0) - (0.25 * En[neg].mean(0) if neg else 0)
        return En @ (qq / (np.linalg.norm(qq) + 1e-9))

    for p in range(PAGES):
        if method == "none" or not shown:
            s = En @ q
        elif method == "rocchio":
            s = rocchio()
        else:  # logreg; with only one class marked so far, rocchio for this page
            y = np.r_[1, np.array(labels, int)]
            if y.min() == y.max():
                s = rocchio()
            else:
                X = np.vstack([q[None], En[shown]])
                s = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(X, y).decision_function(En)
        page = next_page(s, centers, [centers[i] for i in shown], P)
        lab = [len(hit_sets[i]) > 0 for i in page]
        for i in page:
            found |= hit_sets[i]
        shown += page
        labels += lab
        pages.append({"precision": float(np.mean(lab)) if lab else 0.0, "found": len(found)})
    return pages


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
    hit_sets = contains(gc, cents, P)
    keys, inv, c = t3.working_chunk_grid(data, W, gc)
    n = len(keys)
    out = {"target": name, "region_s0": [r.tolist() for r in region_s0], "pools": {}}
    for pool in ("all", "coarse_top2pct"):
        res = {m: [] for m in METHODS}
        for qi, qc in enumerate(queries):
            qe = emb.embed_at(data.path(W), W, qc[None], P, lambda cs: t2.patches_read(data, W, cs, P))[0]
            notself = ~np.all(np.abs(gc - qc) < P / 2, 1)
            keep = notself.copy()
            if pool != "all":
                comb, centers = search.coarse_scan(data.read, (qc - P // 2, qc + P // 2), (rlo, rhi), W,
                                                   cfg["default_coarsest"], P)
                prior = np.full(n, -np.inf)
                cid = {tuple(k): i for i, k in enumerate(keys)}
                fc, fs = centers.reshape(-1, 3), comb.ravel()
                ok = np.all((fc >= rlo) & (fc < rhi), 1)
                for cc, sv in zip(fc[ok] // c, fs[ok]):
                    j = cid.get(tuple(cc))
                    if j is not None and sv > prior[j]:
                        prior[j] = sv
                top = np.argsort(-prior)[: max(1, int(round(0.02 * n)))]
                sel = np.zeros(n, bool)
                sel[top] = True
                keep &= sel[inv]
            idx = np.nonzero(keep)[0]
            rng = np.random.default_rng(qi)
            for m in METHODS:
                pages = session(m, qe, E[idx], gc[idx], [hit_sets[i] for i in idx], P, rng)
                res[m].append(pages)
            line = " | ".join(f"{m}: " + " ".join(f"{pg['precision']:.1f}" for pg in res[m][-1]) +
                              f" ({res[m][-1][-1]['found']} found)" for m in METHODS)
            log(f"[{name}{args.suffix} {pool} q{qi}] page precision {line}")
        out["pools"][pool] = res
    (RESULTS / f"test4_{name}{args.suffix}.json").write_text(json.dumps(out, indent=1))
    for pool, res in out["pools"].items():
        msg = []
        for m in METHODS:
            pr = np.mean([[pg["precision"] for pg in s] for s in res[m]], axis=0)
            fd = np.mean([s[-1]["found"] for s in res[m]])
            msg.append(f"{m}: pages {' '.join(f'{v:.2f}' for v in pr)}, {fd:.1f} instances found in {PAGES * PAGE}")
        log(f"== {name}{args.suffix} pool={pool}: " + " | ".join(msg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=["mito", "vesicle"])
    ap.add_argument("--queries", type=int, default=5)
    ap.add_argument("--region", default=None)
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    if args.region == "full":
        t2.REGION_S0 = (np.zeros(3, int), np.array([6368, 1600, 12000]))
    logf = open(RESULTS / "test4.log", "a")

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
