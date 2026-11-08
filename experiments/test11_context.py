"""Test 11: can coarser levels rule out whole regions by their surroundings?

The page's rough pass (test 8) looks at every place with the model two levels up (the same
physical box, P/4 px there) and reads the best 10% in detail. On a big volume that pass itself is
the cost: one model run per working tile. Two levels further up the thing is a few pixels, so a
coarser level cannot see it; it can only see the surroundings (cytoplasm or nucleus, neuropil or
cell bodies). Here a box of P px at a coarser level, centred like the query's (so 4x or 6-8x its
width: the query with its surroundings), scores every region by how alike its surroundings are,
and the regions below the best `keep` share are never looked at by the rough pass.

Orders, all reading whole working tiles with test 5's brute-force embeddings (so only which tiles
are read differs), at the same share of tiles read in detail:
  rough W+2         the page now: tiles by their best rough place
  ctx4 keep k       surroundings 4x the box (level W+2, P px) keep the best k of the tiles; rough within
  ctx8 keep k       surroundings 6-8x the box (level W+3; P px, 48 where the volume is thinner)
  ctx4 alone        tiles by surroundings only (how much they say on their own)
  adaptive          read the first 1% by rough; if the tiles of its top 10 hits sit in the best
                    quarter by surroundings (mean percentile >= 0.75), keep the best quarter by the
                    surroundings that predicted best, else prune nothing
Costs counted: model runs (detail patches + rough places in the tiles not ruled out + the
surroundings' boxes) and bytes (detail tiles + the levels the guides read).

    uv run python experiments/test11_context.py [--targets mito vesicle mito_full]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import test2_coarse2fine as t2
import test6_models as t6
from common import CACHE, RESULTS, Mirror
from test3_combined import working_chunk_grid
from test8_levels import REGIONS, coarse_grid, unit

FRACS = [0.01, 0.02, 0.05, 0.1]
KEEPS = [0.5, 0.25]
EARLY = 0.01


def grid_axes(lo, hi, p, step):
    """Centres `step` apart with a p-box inside [lo, hi) on each axis (one centre if it barely fits)."""
    axes = []
    for a, b in zip(lo, hi):
        ax = np.arange(a + p // 2, b - p // 2 + 1, step)
        axes.append(ax if len(ax) else np.array([(a + b) // 2]))
    return axes


def box_at(c, p, lo, hi):
    """A p-box (one slice) centred near c, shifted inside [lo, hi)."""
    c = np.array(c)
    for a in (1, 2):
        c[a] = min(max(c[a], lo[a] + p // 2), hi[a] - p // 2)
    c[0] = min(max(c[0], lo[0]), hi[0] - 1)
    return c


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
    per_tile = np.bincount(inv, minlength=n)  # detail patches in each tile
    sizes_w = data.chunk_sizes(W)
    chunk_bytes = np.array([sizes_w[tuple(k)] for k in keys], float)
    total = chunk_bytes.sum()
    wbytes = data.cost([(data.path(W), rlo, rhi)], region_s0)["bytes"]
    cid = {tuple(k): i for i, k in enumerate(keys)}
    Q = m(np.stack([data.read(W, q - [0, P // 2, P // 2], q + [1, P // 2, P // 2])[0] for q in queries]))
    tile_c = keys * ch + ch // 2  # tile centres, working level

    # the rough pass: the model two levels up, one patch of the same physical box per cell (test 8)
    up, L2, p2 = 2, W + 2, P // 4
    lo2, hi2 = rlo // 4, -(-rhi // 4)
    vol2 = data.read(L2, lo2, hi2)
    cc = coarse_grid(lo2, hi2, p2)
    t = time.time()
    Ec = m(t2.patches_from(vol2, lo2, cc, p2))
    Qc = m(np.stack([data.read(L2, c - [0, p2 // 2, p2 // 2], c + [1, p2 // 2, p2 // 2])[0] for c in [q // 4 for q in queries]]))
    owner = np.array([cid.get(tuple(k), -1) for k in (cc * 4 + 2) // ch])
    rough_cost = data.cost([(data.path(L2), lo2, hi2)], region_s0)["bytes"] / wbytes
    log(f"[{target}] rough at s{L2}: {len(cc)} places of {p2} px ({time.time() - t:.0f}s), its level {rough_cost:.1%} of s{W}")

    # the surroundings: P px boxes (4x the box) at W+2, and 6-8x at W+3
    ctx = {}
    for key, Lc in (("ctx4", W + 2), ("ctx8", W + 3)):
        f = 2 ** (Lc - W)
        lo, hi = rlo // f, -(-rhi // f)
        pc = int(min(P, (min(hi[1] - lo[1], hi[2] - lo[2]) // 16) * 16))
        axes = grid_axes(lo, hi, pc, max(pc // 2, 8))
        g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        vol = vol2 if Lc == L2 else data.read(Lc, lo, hi)
        t = time.time()
        Ectx = m(t2.patches_from(vol, lo, g, pc))
        Qctx = m(np.stack([t2.patches_from(vol, lo, box_at(q // f, pc, lo, hi)[None], pc)[0] for q in queries]))
        # each tile takes the nearest box's score
        idx = [np.abs(axes[a][None, :] - (tile_c[:, a] / f)[:, None]).argmin(1) for a in range(3)]
        flat = np.ravel_multi_index(idx, [len(a) for a in axes])
        cost = 0.0 if Lc == L2 else data.cost([(data.path(Lc), lo, hi)], region_s0)["bytes"] / wbytes
        ctx[key] = (Ectx, Qctx, flat, len(g), cost)
        log(f"[{target}] {key} at s{Lc}: {len(g)} boxes of {pc} px = {pc * f / P:.0f}x the box ({time.time() - t:.0f}s), level {cost:.2%} of s{W}")
        if Lc != L2:
            del vol
    del vol2

    names = ["rough W+2"] + [f"{c} keep {k:.0%}" for c in ctx for k in KEEPS] + ["ctx4 alone", "adaptive"]
    curves = {k: [] for k in names}
    chose = []
    for qi, q in enumerate(queries):
        sims = E @ Q[qi]
        notself = ~np.all(np.abs(gc - q) < P / 2, 1)
        rng = np.random.default_rng(qi)
        s = Ec @ Qc[qi]
        s[np.all(np.abs(cc - q // 4) < P / 8, 1)] = -np.inf  # the query's own cell
        rough = np.full(n, -np.inf)
        ok = owner >= 0
        np.maximum.at(rough, owner[ok], s[ok])
        rough += rng.random(n) * 1e-9
        cscore = {}
        for key, (Ectx, Qctx, flat, _, _) in ctx.items():
            cscore[key] = (Ectx @ Qctx[qi])[flat] + rng.random(n) * 1e-9
        pct = {key: np.argsort(np.argsort(v)) / (n - 1) for key, v in cscore.items()}  # percentile per tile

        def kept(key, k):
            return pct[key] >= 1 - k

        def evaluate(sel, rough_frac, ctx_used):
            mm = sel[inv] & notself
            e = t2.evaluate(t2.rank_nms(gc[mm], sims[mm], P), cents, P)
            runs = per_tile[sel].sum() + rough_frac * ok.sum() + sum(ctx[c][3] for c in ctx_used)
            byts = chunk_bytes[sel].sum() / total + rough_frac * rough_cost + sum(ctx[c][4] for c in ctx_used)
            return {"p@10": e["p@10"], "p@25": e["p@25"], "runs": float(runs), "bytes": float(byts)}

        def take(prio, budget, start=None):
            sel = np.zeros(n, bool) if start is None else start.copy()
            order = np.argsort(-prio, kind="stable")
            order = order[~sel[order] & np.isfinite(prio[order])]
            sel[order[: max(0, budget - sel.sum())]] = True
            return sel

        # the first EARLY share by rough, and whether the surroundings predicted its best hits
        early = take(rough, max(1, int(round(EARLY * n))))
        mm = early[inv] & notself
        top = t2.rank_nms(gc[mm], sims[mm], P)[:10]
        tiles_hit = np.array([cid[tuple(k)] for k in (np.asarray(top) // ch)], int) if len(top) else np.array([], int)
        pred = {key: float(pct[key][tiles_hit].mean()) if len(tiles_hit) else 0.0 for key in ctx}
        best = max(pred, key=pred.get)
        prune = pred[best] >= 0.75
        chose.append((best if prune else None, pred))

        rows = {k: [] for k in names}
        for f in FRACS:
            b = max(1, int(round(f * n)))
            rows["rough W+2"].append(evaluate(take(rough, b), 1.0, []))
            for key in ctx:
                for k in KEEPS:
                    pr = np.where(kept(key, k), rough, -np.inf)
                    rows[f"{key} keep {k:.0%}"].append(evaluate(take(pr, b), k, [key]))
            rows["ctx4 alone"].append(evaluate(take(cscore["ctx4"], b), 0.0, ["ctx4"]))
            if prune:
                pr = np.where(kept(best, 0.25), rough, -np.inf)
                rows["adaptive"].append(evaluate(take(pr, b, start=early if b >= early.sum() else None), 0.1 + 0.9 * 0.25, list(ctx)))
            else:
                rows["adaptive"].append(evaluate(take(rough, b), 1.0, list(ctx)))
        for k in names:
            curves[k].append(rows[k])

    log(f"[{target}] {n} tiles, {len(queries)} queries; p@10 / p@25 after reading a share of the tiles in detail; "
        f"model runs and bytes at 5% relative to the rough pass now:")
    i5 = FRACS.index(0.05)
    ref_runs = np.mean([c[i5]["runs"] for c in curves["rough W+2"]])
    ref_bytes = np.mean([c[i5]["bytes"] for c in curves["rough W+2"]])
    for k in names:
        cells = []
        for i, f in enumerate(FRACS):
            cells.append(f"{f:.0%}: {np.mean([c[i]['p@10'] for c in curves[k]]):.2f}/{np.mean([c[i]['p@25'] for c in curves[k]]):.2f}")
        rr = np.mean([c[i5]["runs"] for c in curves[k]]) / ref_runs
        rb = np.mean([c[i5]["bytes"] for c in curves[k]]) / ref_bytes
        log(f"   {k:<15} " + "  ".join(cells) + f"   runs x{rr:.2f} bytes x{rb:.2f}")
    npr = sum(1 for b, _ in chose if b)
    log(f"   adaptive pruned for {npr} of {len(chose)} queries; mean percentile of the early top hits' tiles: "
        + ", ".join(f"{key} {np.mean([p[key] for _, p in chose]):.2f}" for key in ctx))
    res = {"target": target, "n_tiles": int(n), "fracs": FRACS, "keeps": KEEPS, "curves": curves,
           "adaptive": [{"pruned_by": b, "pred": p} for b, p in chose], "rough_cost": rough_cost,
           "ctx": {k: {"boxes": v[3], "level_cost": v[4]} for k, v in ctx.items()}}
    (RESULTS / f"test11_{target}.json").write_text(json.dumps(res, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=["mito", "vesicle", "mito_full"])
    args = ap.parse_args()
    logf = open(RESULTS / "test11.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    os.environ.setdefault("TORCH_THREADS", "8")
    m, data = t6.Matcher("dinov2-small"), t2.Data(Mirror())
    for t in args.targets:
        run(t, m, data, log)


if __name__ == "__main__":
    main()
