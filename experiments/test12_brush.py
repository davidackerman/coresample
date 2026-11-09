"""Test 12: does saying which part of the box matters (a brush over the query) find more?

The page's brush weights the query box's 8 x 8 token grid: brushed parts count, the rest less or
not at all. Here the brush is the query object's own label in the box (what a person would
paint over: the instance under the click), at token resolution, either exactly (the share of
each token's pixels inside it) or roughly (those tokens over a quarter inside, grown by one
token, as a quick stroke would). Test 5's candidates (every patch of the region), its 30
queries per target, DINOv2-small at 112 px; only the comparison changes:

  cls+mean            the page's default
  cls+center          Options > object at the centre (a fixed Gaussian)
  cls+brush           the brushed tokens' weighted mean, the same weights on every candidate
                      (the thing must sit where it sits in the query box)
  brush               that alone, without the CLS token
  cls+brush max-sim   each brushed query token's best match anywhere in the candidate,
                      weighted (the thing may sit anywhere in the box)

Differences are paired per query against cls+mean (+/- standard error).

    uv run python experiments/test12_brush.py [--targets mito vesicle mito_full]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import test2_coarse2fine as t2
import test5_matching as t5
from common import CACHE, RESULTS, Mirror

VARIANTS = ["cls+mean", "cls+center", "cls+brush exact", "cls+brush rough", "brush exact", "brush rough",
            "cls+brush max-sim exact", "cls+brush max-sim rough"]


def brush(data, cfg, W, P, qc):
    """The query object's label in the query box (the instance under the click), as 8 x 8
    token weights: exact (share of each token inside) and rough (over a quarter, grown by one)."""
    h = P // 2
    lab = data.mirror.read(f"{cfg['label']}/s{W}", qc - [0, h, h], qc + [1, h, h])[0]
    inst = lab[h, h]
    m = (lab == inst).astype(np.float32) if inst else np.zeros_like(lab, np.float32)
    g = 8
    exact = m.reshape(g, P // g, g, P // g).mean((1, 3))
    rough = exact > 0.25
    grown = rough.copy()
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        grown |= np.roll(np.roll(rough, dy, 0), dx, 1) & _inside(g, dy, dx)
    return exact.ravel(), grown.astype(np.float32).ravel()


def _inside(g, dy, dx):  # where np.roll did not wrap around
    ok = np.ones((g, g), bool)
    if dy > 0: ok[:dy] = False
    if dy < 0: ok[dy:] = False
    if dx > 0: ok[:, :dx] = False
    if dx < 0: ok[:, dx:] = False
    return ok


def run(name, args, data, log):
    full = name == "mito_full"
    cfg = t2.TARGETS["mito" if full else name]
    W, P = cfg["working"], cfg["patch"]
    data.working = W
    t2.REGION_S0 = (np.zeros(3, int), np.array([6368, 1600, 12000])) if full else (np.array([640, 0, 512]), np.array([2688, 1600, 4608]))
    rlo, rhi = t2.REGION_S0[0] // 2 ** W, t2.REGION_S0[1] // 2 ** W
    cents = t2.instances(data.mirror, cfg["label"], W, rlo, rhi, cfg["min_voxels"])["centroids"]
    queries = [np.array(q) for q in json.loads((RESULTS / f"test5_{name}_n30.json").read_text())["queries"]]
    gc = t2.grid_centers(rlo, rhi, P)
    model = t5.Model(112, keep_tokens=True)
    f = CACHE / f"test5_{name}_112tok.npz"  # test 5's (mito subvolume); the others made here, kept in memory
    if f.exists():
        C = dict(np.load(f))
    else:
        t = time.time()
        vol = data.read(W, rlo, rhi)
        parts = [model(t2.patches_from(vol, rlo, gc[i:i + 2048], P)) for i in range(0, len(gc), 2048)]
        C = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        log(f"[{name}] embedded {len(gc)} patches with tokens in {time.time() - t:.0f}s")
    T = C["tokens"]  # (n, 64, 384) unit tokens, float16
    E = {k: t5.unit(C[k].astype(np.float32)) for k in ("cls", "mean", "center")}
    log(f"[{name}] {len(cents)} instances, {len(gc)} candidates, {len(queries)} queries")
    per = {v: [] for v in VARIANTS}
    for qi, qc in enumerate(queries):
        Q = model(t5.query_patches(data, W, P, qc)[0][:1])
        q = {k: t5.unit(Q[k][0].astype(np.float32)) for k in ("cls", "mean", "center")}
        qt = Q["tokens"][0].astype(np.float32)  # (64, 384)
        notself = ~np.all(np.abs(gc - qc) < P / 2, 1)
        cls = E["cls"] @ q["cls"]
        s = {"cls+mean": (cls + E["mean"] @ q["mean"]) / 2, "cls+center": (cls + E["center"] @ q["center"]) / 2}
        for kind, w in zip(("exact", "rough"), brush(data, cfg, W, P, qc)):
            if w.sum() == 0:
                w = np.ones_like(w)
            qb = t5.unit(w @ qt)
            pooled, maxsim, on = np.empty(len(gc), np.float32), np.empty(len(gc), np.float32), np.nonzero(w)[0]
            for i in range(0, len(gc), 4096):
                t = T[i:i + 4096].astype(np.float32)
                pooled[i:i + 4096] = t5.unit(np.matmul(w, t)) @ qb
                best = np.matmul(t, qt[on].T).max(1)  # each brushed query token's best match among the candidate's
                maxsim[i:i + 4096] = best @ w[on] / w[on].sum()
            s[f"cls+brush {kind}"] = (cls + pooled) / 2
            s[f"brush {kind}"] = pooled
            s[f"cls+brush max-sim {kind}"] = (cls + maxsim) / 2
        for v in VARIANTS:
            per[v].append(t5.grade(gc, s[v], notself, cents, P))
        log(f"[{name} q{qi}] p@10 " + " ".join(f"{v}={per[v][-1]['p@10']:.1f}" for v in VARIANTS))
    out = {"n_candidates": int(len(gc)), "rows": []}
    base = {k: np.array([g[k] for g in per["cls+mean"]]) for k in ("p@10", "p@25")}
    for v in VARIANTS:
        row = {"variant": v}
        for k in ("p@10", "p@25"):
            a = np.array([g[k] for g in per[v]])
            d = a - base[k]
            row |= {k: float(a.mean()), f"{k}_diff": float(d.mean()), f"{k}_diff_se": float(d.std(ddof=1) / np.sqrt(len(d))),
                    f"{k}_per_query": a.tolist()}
        out["rows"].append(row)
        log(f"== {name} {v:<26} p@10 {row['p@10']:.2f} ({row['p@10_diff']:+.3f} +/- {row['p@10_diff_se']:.3f})"
            f"  p@25 {row['p@25']:.2f} ({row['p@25_diff']:+.3f} +/- {row['p@25_diff_se']:.3f})")
    (RESULTS / f"test12_{name}.json").write_text(json.dumps(out, indent=1))
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=["mito", "vesicle", "mito_full"])
    args = ap.parse_args()
    logf = open(RESULTS / "test12.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    data = t2.Data(Mirror())
    pooled = {}
    for name in args.targets:
        for v, g in run(name, args, data, log).items():
            pooled.setdefault(v, []).extend(g)
    base = np.array([g["p@10"] for g in pooled["cls+mean"]])
    for v in VARIANTS:  # all targets' queries together
        d = np.array([g["p@10"] for g in pooled[v]]) - base
        log(f"== pooled ({len(d)} queries) {v:<26} p@10 {d.mean():+.3f} +/- {d.std(ddof=1) / np.sqrt(len(d)):.3f}")


if __name__ == "__main__":
    main()
