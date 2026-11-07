"""Test 2: coarse-to-fine recall vs fraction of data read, against brute force.

Search logic (search.py) sees only raw EM. Labels (COSEM whole-volume segmentations)
are used here only to choose query instances and to grade results.

For each target class and each query instance:
  brute force   embed every patch on a grid over the subvolume at the working level
  coarse2fine   cheap scores on the whole coarsest level, descend through top-K
                candidates, embed only survivors at the working level
  sampling      random chunk order and best-first chunk expansion (simulated from the
                brute-force embeddings), the no-pyramid fallback
Hit: a returned region (patch^3 box at the working level) contains an instance
centroid. Data read is costed from the bucket listing's real chunk sizes.

    uv run python test2_coarse2fine.py [--targets mito vesicle] [--queries 5]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import search
from common import CACHE, EM_ZARR, LABELS_N5, RESULTS, Mirror, chunk_size_grid, level_path
from embed import Embedder

# Subvolume in s0 voxels (z, y, x): 10.7 x 6.4 x 16.4 um of cytoplasm left of the nucleus,
# with a nucleus edge and some resin at high y (results/overview_s4.png shows it).
REGION_S0 = (np.array([640, 0, 512]), np.array([2688, 1600, 4608]))

TARGETS = {
    "mito": dict(label=f"{LABELS_N5}/mito_seg", working=2, patch=64, coarsest=[5, 4, 3], default_coarsest=4,
                 min_voxels=500),
    "vesicle": dict(label=f"{LABELS_N5}/vesicle_seg", working=1, patch=64, coarsest=[5, 4, 3],
                    default_coarsest=3, min_voxels=30),
}
KS = [25, 50, 100, 200, 400]
DEFAULT_K = 100
METHOD_SETS = {"ncc": ("ncc",), "hist": ("hist",), "tex": ("tex",), "all": ("ncc", "hist", "tex"),
               "hist+tex": ("hist", "tex")}
NS = [10, 25, 50, 100, 200]
BANDWIDTH = 50e6  # bytes/s used to turn bytes into a modelled transfer time


# --------------------------------------------------------------------------------------
# Data access and costing
# --------------------------------------------------------------------------------------


class Data:
    def __init__(self, mirror: Mirror):
        self.mirror = mirror
        self.sizes = {}

    def path(self, level):
        return level_path(EM_ZARR, level)

    def read(self, level, lo, hi):
        return self.mirror.read(self.path(level), lo, hi)

    def chunk_sizes(self, level):
        if level not in self.sizes:
            f = CACHE / f"sizes_s{level}.npy"
            if f.exists():
                self.sizes[level] = np.load(f)
            else:
                self.sizes[level] = chunk_size_grid(self.mirror.info(self.path(level)))
                np.save(f, self.sizes[level])
        return self.sizes[level]

    def cost(self, boxes, region_s0, inner: int | None = None):
        """Bytes / chunks / voxels needed by a list of (path, lo, hi) reads.

        Stored chunks are charged whole, pro-rated by the part that lies inside the
        search region (so a coarse chunk larger than the region is not charged in full).
        inner=c instead charges c^3 sub-blocks at the stored chunk's byte density, the
        cost if the same data were sharded with c^3 inner chunks."""
        out = {"bytes": 0.0, "chunks": 0, "voxels": 0.0}
        by_path = {}
        for p, lo, hi in boxes:
            by_path.setdefault(p, []).append((lo, hi))
        for p, bl in by_path.items():
            level = int(p.rsplit("s", 1)[1])
            info = self.mirror.info(p)
            sizes = self.chunk_sizes(level)
            c = np.array(info.chunks)
            unit = np.array([inner] * 3) if inner else c
            rlo = region_s0[0] // 2**level
            rhi = -(-region_s0[1] // 2**level)
            shape = np.array(info.shape)
            touched = set()
            g = np.array([1, 4, 4])  # voxel accounting granularity (slices stay 1 thick in z)
            mask_lo = rlo // g
            mask = np.zeros(tuple(-(-(rhi - rlo) // g) + 2), bool)
            for lo, hi in bl:
                lo = np.maximum(lo, 0)
                hi = np.minimum(hi, shape)
                if np.any(hi <= lo):
                    continue
                a, b = lo // unit, (hi - 1) // unit
                for z in range(a[0], b[0] + 1):
                    for y in range(a[1], b[1] + 1):
                        for x in range(a[2], b[2] + 1):
                            touched.add((z, y, x))
                ml = np.clip(lo // g - mask_lo, 0, np.array(mask.shape))
                mh = np.clip(-(-hi // g) - mask_lo, 0, np.array(mask.shape))
                mask[ml[0]:mh[0], ml[1]:mh[1], ml[2]:mh[2]] = True
            out["voxels"] += mask.sum() * np.prod(g)
            for t in touched:
                t = np.array(t)
                blo, bhi = t * unit, np.minimum((t + 1) * unit, shape)
                ilo, ihi = np.maximum(blo, rlo), np.minimum(bhi, rhi)
                inside = np.prod(np.maximum(ihi - ilo, 0)) / np.prod(bhi - blo)
                if inside <= 0:
                    continue
                parent = tuple(t * unit // c)
                density = sizes[parent] / np.prod(np.minimum((np.array(parent) + 1) * c, shape) - np.array(parent) * c)
                out["bytes"] += density * np.prod(bhi - blo) * inside
                out["chunks"] += inside * np.prod(bhi - blo) / np.prod(unit) * (np.prod(unit) / np.prod(c))
        return out


# --------------------------------------------------------------------------------------
# Ground truth (grading only)
# --------------------------------------------------------------------------------------


def instances(mirror: Mirror, label: str, level: int, rlo, rhi, min_voxels: int, slab: int = 32):
    """Instance centroids (level voxels, absolute) from a label array, streamed in z slabs."""
    sums = cnt = None
    for z0 in range(rlo[0], rhi[0], slab):
        z1 = min(rhi[0], z0 + slab)
        lab = mirror.read(f"{label}/s{level}", [z0, rlo[1], rlo[2]], [z1, rhi[1], rhi[2]]).astype(np.int64)
        n = int(lab.max()) + 1
        if cnt is None or n > len(cnt):
            grow = lambda a, k: np.pad(a, [(0, k - a.shape[0])] + [(0, 0)] * (a.ndim - 1)) if a is not None else None
            cnt = grow(cnt, n) if cnt is not None else np.zeros(n)
            sums = grow(sums, n) if sums is not None else np.zeros((n, 3))
        flat = lab.ravel()
        m = len(cnt)
        cnt += np.bincount(flat, minlength=m)
        zz, yy, xx = np.meshgrid(np.arange(z0, z1), np.arange(rlo[1], rhi[1]), np.arange(rlo[2], rhi[2]),
                                 indexing="ij", sparse=True)
        for ax, coord in enumerate((zz, yy, xx)):
            sums[:, ax] += np.bincount(flat, weights=np.broadcast_to(coord, lab.shape).ravel(), minlength=m)
    ids = np.nonzero(cnt)[0]
    ids = ids[ids != 0]
    return dict(ids=ids, centroids=sums[ids] / cnt[ids, None], voxels=cnt[ids], big=cnt[ids] >= min_voxels)


def inside_point(mirror: Mirror, label: str, level: int, inst_id: int, centroid, radius: int):
    """The voxel of instance inst_id nearest its centroid (searched in a box of radius)."""
    c = np.round(centroid).astype(int)
    lo, hi = c - radius, c + radius + 1
    lab = mirror.read(f"{label}/s{level}", lo, hi)
    pts = np.argwhere(lab == inst_id)
    if len(pts) == 0:
        return None
    pts = pts + lo
    return pts[np.argmin(((pts - centroid) ** 2).sum(1))]


def evaluate(centers, centroids, patch, ns=NS):
    """precision@N and recall@N for ranked region centres (working voxels)."""
    half = patch / 2
    res = {}
    if len(centers) == 0:
        return {f"p@{n}": 0.0 for n in ns} | {f"r@{n}": 0.0 for n in ns}
    d = np.abs(centers[:, None, :] - centroids[None, :, :])  # (k, m, 3)
    contains = np.all(d < half, axis=2)
    for n in ns:
        c = contains[:n]
        res[f"p@{n}"] = float(c.any(1).mean()) if len(c) else 0.0
        res[f"r@{n}"] = float(c.any(0).mean())
        res[f"n@{n}"] = int(len(c))
    return res


def rank_nms(centers, scores, patch, k=max(NS)):
    order = np.argsort(-scores, kind="stable")
    return search.nms(centers[order], np.array([patch] * 3), k)


# --------------------------------------------------------------------------------------
# Main experiment
# --------------------------------------------------------------------------------------


def grid_centers(rlo, rhi, patch):
    s = patch // 2
    axes = [np.arange(lo + patch // 2, hi - patch // 2 + 1, s) for lo, hi in zip(rlo, rhi)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


def patches_from(vol, vlo, centers, patch):
    h = patch // 2
    out = np.empty((len(centers), patch, patch), np.uint8)
    for i, (z, y, x) in enumerate(centers - vlo):
        out[i] = vol[z, y - h : y + h, x - h : x + h]
    return out


def patches_read(data, level, centers, patch):
    h = patch // 2
    return np.stack([data.read(level, c - [0, h, h], c + [1, h, h])[0] for c in centers]) if len(centers) else \
        np.zeros((0, patch, patch), np.uint8)


def run_target(name, cfg, args, data, emb, log):
    W, P = cfg["working"], cfg["patch"]
    data.working = W
    rlo, rhi = REGION_S0[0] // 2**W, REGION_S0[1] // 2**W
    region = (rlo, rhi)
    t = time.time()
    gt = instances(data.mirror, cfg["label"], W, rlo, rhi, cfg["min_voxels"])
    cents = gt["centroids"]
    log(f"[{name}] {len(cents)} instances in region ({gt['big'].sum()} >= {cfg['min_voxels']} voxels), {time.time() - t:.0f}s")

    # query instances: big enough, with an object voxel near the centroid, well inside
    rng = np.random.default_rng(args.seed)
    margin = P
    ok = gt["big"] & np.all((cents >= rlo + margin) & (cents < rhi - margin), axis=1)
    qidx, qcenters = [], []
    for j in rng.permutation(np.nonzero(ok)[0]):
        pt = inside_point(data.mirror, cfg["label"], W, int(gt["ids"][j]), cents[j], P // 2)
        if pt is not None and np.all((pt >= rlo + margin) & (pt < rhi - margin)):
            qidx.append(j)
            qcenters.append(pt)
        if len(qidx) >= args.queries:
            break

    # ---- brute force ----
    data.mirror.reset_touched()
    t = time.time()
    vol = data.read(W, rlo, rhi)
    t_read = time.time() - t
    gc = grid_centers(rlo, rhi, P)
    t = time.time()
    e0 = dict(emb.stats)
    E = emb.embed_at(data.path(W), W, gc, P, lambda c: patches_from(vol, rlo, c, P))
    t_bf_embed = time.time() - t
    bf_cost = data.cost(data.mirror.boxes, REGION_S0)
    bf_cost32 = data.cost(data.mirror.boxes, REGION_S0, inner=32)
    log(f"[{name}] brute force: {len(gc)} patches, read {t_read:.0f}s, embed {t_bf_embed:.0f}s "
        f"({emb.stats['computed'] - e0['computed']} computed), {bf_cost['bytes'] / 1e9:.2f} GB")
    # chance: fraction of grid boxes containing a centroid
    from scipy.spatial import cKDTree
    dmin, _ = cKDTree(cents).query(gc, p=np.inf)
    chance = float(np.mean(dmin < P / 2))

    results = {"target": name, "working": W, "patch": P, "region_s0": [r.tolist() for r in REGION_S0],
               "n_instances": int(len(cents)), "chance_precision": chance, "brute": {}, "c2f": [], "sampling": {},
               "brute_cost": bf_cost, "brute_cost_inner32": bf_cost32, "brute_patches": int(len(gc)),
               "brute_embed_seconds_per_patch": t_bf_embed / max(1, emb.stats["computed"] - e0["computed"]) if
               emb.stats["computed"] > e0["computed"] else None, "queries": []}

    memo = {}
    for qi, (j, qc) in enumerate(zip(qidx, qcenters)):
        qc = np.asarray(qc, int)
        qbox = (qc - P // 2, qc + P // 2)
        qe = emb.embed_at(data.path(W), W, qc[None], P, lambda c: patches_read(data, W, c, P))[0]
        results["queries"].append({"instance": int(gt["ids"][j]), "center_working": qc.tolist(),
                                   "voxels": int(gt["voxels"][j])})
        # brute force ranking
        sims = search.rank_by_embedding(qe, E)
        # exclude the query's own box from results (it is trivially a hit)
        notself = ~np.all(np.abs(gc - qc) < P / 2, 1)
        top = rank_nms(gc[notself], sims[notself], P)
        ev = evaluate(top, cents, P)
        results["brute"][qi] = ev
        log(f"[{name} q{qi}] brute: " + " ".join(f"p@{n}={ev[f'p@{n}']:.2f}" for n in (10, 25, 50)) +
            f" r@100={ev['r@100']:.3f} (chance {chance:.2f})")

        # sampling baselines from the brute-force embeddings
        results["sampling"][qi] = sampling_curves(data, W, P, gc, sims, notself, cents, rlo, rhi, rng)

        # coarse-to-fine sweeps (one factor at a time around the defaults)
        dm, dc = args.methods, cfg["default_coarsest"]
        coarsests = args.coarsest or cfg["coarsest"]
        configs = [(dc, DEFAULT_K, dm)]
        if "k" in args.sweep:
            configs += [(dc, k, dm) for k in KS if k != DEFAULT_K]
        if "methods" in args.sweep:
            configs += [(dc, DEFAULT_K, m) for m in args.method_sweep if m != dm]
        if "coarsest" in args.sweep:
            configs += [(c, DEFAULT_K, dm) for c in coarsests if c != dc]
        if args.quick:
            configs = configs[:1]
        for coarsest, k, mset in configs:
            if coarsest <= W:
                continue
            data.mirror.reset_touched()
            c0 = dict(emb.stats)
            t = time.perf_counter()
            stages = search.coarse_to_fine(data.read, qbox, region, W, coarsest, P, k, METHOD_SETS[mset],
                                           memo=memo)
            t_cheap = time.perf_counter() - t
            stage_eval = []
            for st in stages:
                cw = search.to_working(st, W)
                keep = ~np.all(np.abs(cw - qc) < P / 2, 1)
                stage_eval.append({"level": st.level, "n": int(len(cw)), "seconds": st.seconds,
                                   **evaluate(rank_nms(cw[keep], st.scores[keep], P), cents, P)})
            cand = search.to_working(stages[-1], W)
            cand = cand[np.all((cand >= rlo + P // 2) & (cand <= rhi - P // 2), 1)]
            t = time.perf_counter()
            Ec = emb.embed_at(data.path(W), W, cand, P, lambda c: patches_read(data, W, c, P))
            # charge reads of cached patches too (a cold search would have read them)
            if len(cand):
                h = P // 2
                for c in cand:
                    data.mirror.boxes.append((data.path(W), c - [0, h, h], c + [1, h, h]))
            t_embed = time.perf_counter() - t
            s = search.rank_by_embedding(qe, Ec) if len(cand) else np.zeros(0)
            keep = ~np.all(np.abs(cand - qc) < P / 2, 1) if len(cand) else np.zeros(0, bool)
            top = rank_nms(cand[keep], s[keep], P)
            ev = evaluate(top, cents, P)
            cost = data.cost(data.mirror.boxes, REGION_S0)
            cost32 = data.cost(data.mirror.boxes, REGION_S0, inner=32)
            n_new = emb.stats["computed"] - c0["computed"]
            row = {"query": qi, "coarsest": coarsest, "k": k, "methods": mset, "final": ev, "stages": stage_eval,
                   "cost": cost, "cost_inner32": cost32,
                   "frac_bytes": cost["bytes"] / bf_cost["bytes"], "frac_bytes_inner32": cost32["bytes"] / bf_cost32["bytes"],
                   "frac_voxels": cost["voxels"] / bf_cost["voxels"], "frac_chunks": cost["chunks"] / bf_cost["chunks"],
                   "seconds_cheap": t_cheap, "seconds_embed": t_embed, "embedded": int(len(cand)), "embedded_new": int(n_new),
                   "seconds_per_level": [st.seconds for st in stages]}
            results["c2f"].append(row)
            log(f"[{name} q{qi}] c2f s{coarsest} K={k} {mset}: " + " ".join(f"p@{n}={ev[f'p@{n}']:.2f}" for n in (10, 25, 50))
                + f" r@100={ev['r@100']:.3f} | bytes {row['frac_bytes']:.1%} (32^3 inner {row['frac_bytes_inner32']:.1%}) "
                f"voxels {row['frac_voxels']:.2%} | cheap {t_cheap:.1f}s embed {t_embed:.1f}s ({n_new} new)")
        (RESULTS / f"test2_{name}{args.suffix}.json").write_text(json.dumps(results, indent=1, default=float))
    return results


def sampling_curves(data, W, P, gc, sims, notself, cents, rlo, rhi, rng, fracs=(0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)):
    """Recall/precision vs fraction of working-level chunks read, for random chunk order
    and best-first expansion seeded by 1% random chunks."""
    info = data.mirror.info(data.path(W))
    c = np.array(info.chunks)
    cid = gc // c
    keys, inv = np.unique(cid, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = len(keys)
    best = np.full(n, -np.inf)
    np.maximum.at(best, inv, np.where(notself, sims, -np.inf))
    out = {"random": [], "adaptive": []}

    def score(selected):
        m = selected[inv] & notself
        top = rank_nms(gc[m], sims[m], P)
        return evaluate(top, cents, P)

    order = rng.permutation(n)
    for f in fracs:
        sel = np.zeros(n, bool)
        sel[order[: max(1, int(round(f * n)))]] = True
        out["random"].append({"frac_chunks": float(sel.mean()), **score(sel)})

    # best-first: seed 1%, then repeatedly read the unread neighbour of the best chunk so far
    seed = order[: max(1, int(round(0.01 * n)))]
    sel = np.zeros(n, bool)
    sel[seed] = True
    pos = {tuple(k): i for i, k in enumerate(keys)}
    prio = np.full(n, -np.inf)

    def push(i):
        for d in np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij")).reshape(3, -1).T:
            j = pos.get(tuple(keys[i] + d))
            if j is not None and not sel[j]:
                prio[j] = max(prio[j], best[i])

    for i in seed:
        push(i)
    targets = sorted(int(round(f * n)) for f in fracs if f > 0.01)
    ti = 0
    while ti < len(targets):
        while sel.sum() >= targets[ti]:
            out["adaptive"].append({"frac_chunks": float(sel.mean()), **score(sel)})
            ti += 1
            if ti >= len(targets):
                break
        if ti >= len(targets):
            break
        cand = np.where(~sel, prio, -np.inf)
        j = int(np.argmax(cand)) if np.isfinite(cand.max()) else int(rng.choice(np.nonzero(~sel)[0]))
        sel[j] = True
        push(j)
    out["adaptive"].insert(0, {"frac_chunks": float(np.mean(np.isin(np.arange(n), seed))), **score(np.isin(np.arange(n), seed))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=list(TARGETS))
    ap.add_argument("--queries", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="default config only (smoke test)")
    ap.add_argument("--region", default=None, help="s0 box zlo,ylo,xlo,zhi,yhi,xhi, or 'full'")
    ap.add_argument("--suffix", default="", help="appended to result file names")
    ap.add_argument("--methods", default="all", choices=list(METHOD_SETS), help="default cheap-score set")
    ap.add_argument("--sweep", nargs="*", default=["k", "methods", "coarsest"], help="one-factor sweeps to run")
    ap.add_argument("--coarsest", nargs="*", type=int, default=None, help="coarsest levels for the sweep")
    ap.add_argument("--method-sweep", nargs="*", default=list(METHOD_SETS), help="method sets in the methods sweep")
    args = ap.parse_args()
    global REGION_S0
    if args.region == "full":
        REGION_S0 = (np.zeros(3, int), np.array([6368, 1600, 12000]))
    elif args.region:
        v = np.array([int(x) for x in args.region.split(",")])
        REGION_S0 = (v[:3], v[3:])
    RESULTS.mkdir(exist_ok=True)
    logf = open(RESULTS / "test2.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    data = Data(Mirror())
    emb = Embedder()
    for name in args.targets:
        run_target(name, TARGETS[name], args, data, emb, log)
    log(f"embedder stats {emb.stats}")


if __name__ == "__main__":
    main()
