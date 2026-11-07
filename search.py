"""Portable search logic: cheap scoring, peak picking, coarse-to-fine descent.

Plain numpy/scipy, no I/O, no labels. Data comes in through a `read(level, lo, hi)`
callable and embeddings through an `embed(level, centers)` callable, so the same
functions run on raw data alone and port to the browser (every operation here is a
prefix sum, an FFT, a separable filter or a sort).

Conventions: volumes are (z, y, x). A "window" is the query box at one level, shape w.
Score maps are in "valid" mode: map[i] scores the window whose corner is i, i.e.
centred at i + w // 2. Higher scores are better for every method.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import fft as sfft
from scipy import ndimage as ndi

METHODS = ("ncc", "hist", "tex")


# --------------------------------------------------------------------------------------
# Box sums (integral volume)
# --------------------------------------------------------------------------------------


def box_sum(x: np.ndarray, w) -> np.ndarray:
    """Sum of x over every window of shape w, valid mode. Exact (float64 prefix sums)."""
    s = x.astype(np.float64)
    for ax, k in enumerate(w):
        c = np.cumsum(s, axis=ax)
        pad = [(0, 0)] * s.ndim
        pad[ax] = (1, 0)
        c = np.pad(c, pad)
        n = c.shape[ax]
        s = np.take(c, np.arange(k, n), axis=ax) - np.take(c, np.arange(0, n - k), axis=ax)
    return s


# --------------------------------------------------------------------------------------
# Cheap methods
# --------------------------------------------------------------------------------------


def _d4(q: np.ndarray):
    """The 8 in-plane (y, x) rotations/flips of a (z, y, x) query."""
    out = []
    for f in (q, q[:, :, ::-1]):
        for k in range(4):
            r = np.rot90(f, k, axes=(1, 2))
            if r.shape == q.shape:
                out.append(r)
    return out


def ncc_map(vol: np.ndarray, query: np.ndarray, rotations: bool = True) -> np.ndarray:
    """Normalized cross-correlation of query with every window (FFT). Max over the
    in-plane dihedral group when rotations=True (shapes permitting)."""
    v = vol.astype(np.float64)
    w = query.shape
    n = float(np.prod(w))
    s1 = box_sum(v, w)
    s2 = box_sum(v * v, w)
    var_v = np.maximum(s2 - s1 * s1 / n, 0.0)
    shape = [a + b - 1 for a, b in zip(v.shape, w)]
    fshape = [sfft.next_fast_len(s, real=True) for s in shape]
    V = sfft.rfftn(v.astype(np.float32), fshape, workers=-1)
    valid = tuple(slice(k - 1, a) for k, a in zip(w, v.shape))
    best = None
    for q in (_d4(query) if rotations else [query]):
        q0 = q.astype(np.float64) - q.mean()
        qn = np.sqrt((q0 * q0).sum())
        # correlation = convolution with the flipped kernel
        Q = sfft.rfftn(q0[::-1, ::-1, ::-1].astype(np.float32), fshape, workers=-1)
        num = sfft.irfftn(V * Q, fshape, workers=-1)[valid]
        if qn < 1e-6:  # flat query: correlation undefined
            r = np.zeros_like(num)
        else:
            # flat windows (resin, padding, saturated areas): the numerator is FFT round-off
            # and dividing it by a ~0 variance gives huge spurious scores, so define NCC = 0
            flat = var_v <= 1e-2 * n
            r = np.where(flat, 0.0, num / np.maximum(np.sqrt(var_v) * qn, 1e-9))
            r = np.clip(r, -1.0, 1.0)
        best = r if best is None else np.maximum(best, r)
    return best


def intensity_edges(vol: np.ndarray, bins: int = 16, lo_hi=None) -> np.ndarray:
    lo, hi = lo_hi if lo_hi is not None else np.percentile(vol, [0.5, 99.5])
    return np.linspace(lo, hi, bins + 1)


def hist_map(vol: np.ndarray, query: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Negative earth mover's distance between each window's intensity histogram and
    the query's (1-D EMD = L1 distance between CDFs, in bin units)."""
    w = query.shape
    n = float(np.prod(w))
    nb = len(edges) - 1
    vb = np.clip(np.digitize(vol, edges[1:-1]), 0, nb - 1)
    qb = np.clip(np.digitize(query, edges[1:-1]), 0, nb - 1)
    qcdf = np.cumsum(np.bincount(qb.ravel(), minlength=nb)) / n
    out_shape = tuple(a - k + 1 for a, k in zip(vol.shape, w))
    emd = np.zeros(out_shape)
    cum = np.zeros(out_shape)
    for b in range(nb - 1):  # last CDF bin is 1 for both
        cum += box_sum(vb == b, w) / n
        emd += np.abs(cum - qcdf[b])
    return -emd


def texture_features(x: np.ndarray) -> list[np.ndarray]:
    """Per-voxel features whose window means describe texture."""
    x = x.astype(np.float32)
    g = np.sqrt(sum(ndi.sobel(x, axis=a) ** 2 for a in range(x.ndim)))
    dog = np.abs(ndi.gaussian_filter(x, 1.0) - ndi.gaussian_filter(x, 2.0))
    return [x, x * x, g, dog]


def texture_stats(x: np.ndarray, w) -> np.ndarray:
    """(4, *valid) window stats: mean, std, mean gradient magnitude, mean |DoG|."""
    n = float(np.prod(w))
    f, f2, g, dog = (box_sum(a, w) / n for a in texture_features(x))
    std = np.sqrt(np.maximum(f2 - f * f, 0))
    return np.stack([f, std, g, dog])


def texture_map(vol: np.ndarray, w, tq: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Negative standardized L2 distance between window texture stats and the query's
    stats tq (4,). `scale` (4,) are per-feature spreads (from the coarsest scan)."""
    t = texture_stats(vol, w)
    d = ((t - tq[:, None, None, None]) / scale[:, None, None, None]) ** 2
    return -np.sqrt(d.sum(0))


def query_texture(q_ctx: np.ndarray, w, margin: int) -> np.ndarray:
    """Texture stats of the query window, computed with `margin` voxels of context so
    filters see the same neighbourhood they see inside the searched volume."""
    t = texture_stats(q_ctx, w)
    return t[(slice(None),) + (margin,) * 3]


def score_maps(vol, q, methods=METHODS, edges=None, tq=None, tex_scale=None, rotations=True, margin=0):
    """All cheap maps for window q over vol. vol carries `margin` extra voxels of
    context on every side; the returned maps cover only the inner positions."""
    w = q.shape
    m = margin
    inner = vol[m : vol.shape[0] - m, m : vol.shape[1] - m, m : vol.shape[2] - m] if m else vol
    out = {}
    if "ncc" in methods:
        out["ncc"] = ncc_map(inner, q, rotations=rotations)
    if "hist" in methods:
        out["hist"] = hist_map(inner, q, edges if edges is not None else intensity_edges(inner))
    if "tex" in methods:
        t = texture_map(vol, w, tq, tex_scale)
        out["tex"] = t[m : t.shape[0] - m, m : t.shape[1] - m, m : t.shape[2] - m] if m else t
    return out


def zscore_stats(maps_list: list[dict]) -> dict:
    """Pooled (mean, std) per method over a list of map dicts."""
    stats = {}
    for m in maps_list[0]:
        allv = np.concatenate([d[m].ravel() for d in maps_list])
        stats[m] = (allv.mean(), allv.std() + 1e-9)
    return stats


def combine(maps: dict, stats: dict, methods) -> np.ndarray:
    """Mean z-score of the chosen methods (pooled stats so boxes are comparable)."""
    return sum((maps[m] - stats[m][0]) / stats[m][1] for m in methods) / len(methods)


# --------------------------------------------------------------------------------------
# Peaks
# --------------------------------------------------------------------------------------


def peaks(score: np.ndarray, k: int, min_dist) -> np.ndarray:
    """Top-k local maxima with greedy non-maximum suppression (Chebyshev distance
    < min_dist per axis suppresses). Returns (k, ndim) indices into score."""
    md = np.broadcast_to(np.asarray(min_dist), (score.ndim,))
    size = tuple(int(max(1, 2 * (m // 2) + 1)) for m in md)
    is_max = score == ndi.maximum_filter(score, size=size, mode="nearest")
    idx = np.argwhere(is_max)
    order = np.argsort(-score[tuple(idx.T)], kind="stable")
    return nms(idx[order], md, k)


def nms(sorted_points: np.ndarray, min_dist, k: int) -> np.ndarray:
    """Greedy NMS over points already sorted best-first."""
    md = np.asarray(min_dist)
    keep: list[np.ndarray] = []
    for p in sorted_points:
        if all(np.any(np.abs(p - q) >= md) for q in keep):
            keep.append(p)
            if len(keep) >= k:
                break
    return np.array(keep, dtype=int).reshape(-1, sorted_points.shape[1] if len(sorted_points) else 3)


# --------------------------------------------------------------------------------------
# Coarse-to-fine descent
# --------------------------------------------------------------------------------------


@dataclass
class Stage:
    """One level of the descent: candidate centres (level voxel coords) and scores."""

    level: int
    centers: np.ndarray  # (k, 3)
    scores: np.ndarray  # (k,)
    seconds: float = 0.0
    methods_scores: dict = field(default_factory=dict)


def window_at(level: int, working: int, patch: int) -> tuple[int, int, int]:
    s = max(2, patch // (2 ** (level - working)))
    return (s, s, s)


def coarse_to_fine(
    read,
    query_box,
    region,
    working: int,
    coarsest: int,
    patch: int,
    k: int | list[int],
    methods=METHODS,
    radius: int | None = None,
    cheap_at_working: bool = False,
    rotations: bool = True,
    margin: int = 4,
    clock=None,
    memo: dict | None = None,
):
    """Coarse-to-fine search with cheap scores only (embedding happens after).

    read(level, lo, hi) -> array for voxel box [lo, hi) at `level` (z, y, x).
    query_box / region are (lo, hi) boxes in WORKING-level voxels.
    Returns a list of Stage, coarsest first. The last stage is at the working level
    (if cheap_at_working) or one level above it, with centres mapped to working voxels.

    Search region at the coarsest level is read whole; every finer level reads only a
    (window + 2*radius) box around each surviving candidate. `memo` (a dict) keeps the
    coarsest scan's maps for all methods so parameter sweeps do not recompute them; the
    data is still read through `read` every time so read accounting stays honest.
    """
    import time

    clock = clock or time.perf_counter
    levels = list(range(coarsest, working - (1 if cheap_at_working else 0), -1))
    ks = k if isinstance(k, (list, tuple)) else [k] * len(levels)
    stages: list[Stage] = []
    qlo, qhi = (np.asarray(a) for a in query_box)
    rlo, rhi = (np.asarray(a) for a in region)

    m = margin
    edges = tex_scale = None
    prev = None
    for li, L in enumerate(levels):
        t0 = clock()
        f = 2 ** (L - working)
        w = np.array(window_at(L, working, patch))
        qc = (qlo + qhi) // 2 // f
        q_ctx = read(L, qc - w // 2 - m, qc - w // 2 + w + m).astype(np.float32)
        q = q_ctx[m:-m, m:-m, m:-m]
        tq = query_texture(q_ctx, w, m)
        if prev is None:
            lo, hi = rlo // f, -(-rhi // f)
            vol = read(L, lo - m, hi + m).astype(np.float32)
            mkey = (L, tuple(qc), tuple(lo), tuple(hi), rotations)
            if memo is not None and mkey in memo:
                edges, tex_scale, all_maps, t_scan = memo[mkey]
                t0 -= t_scan  # report the scan's compute time as if it ran now
            else:
                ts0 = clock()
                inner = vol[m:-m, m:-m, m:-m]
                edges = intensity_edges(inner)
                tex_scale = texture_stats(inner, w).reshape(4, -1).std(axis=1) + 1e-9
                all_maps = score_maps(vol, q, METHODS if memo is not None else methods, edges, tq,
                                      tex_scale, rotations, margin=m)
                if memo is not None:
                    memo[mkey] = (edges, tex_scale, all_maps, clock() - ts0)
            maps = {name: all_maps[name] for name in methods}
            stats = zscore_stats([maps])
            comb = combine(maps, stats, methods)
            pk = peaks(comb, ks[li], w // 2)
            centers = pk + lo + w // 2
            scores = comb[tuple(pk.T)]
            ms = {name: maps[name][tuple(pk.T)] for name in methods}
        else:
            r = radius if radius is not None else max(2, int(w[0]) // 8)
            boxes, all_maps = [], []
            for c in prev.centers * 2 + 1:  # parent voxel centre at the finer level
                lo = c - r - w // 2
                hi = c + r + (w - w // 2)
                vol = read(L, lo - m, hi + m).astype(np.float32)
                all_maps.append(score_maps(vol, q, methods, edges, tq, tex_scale, rotations, margin=m))
                boxes.append(lo)
            if not boxes:
                stages.append(Stage(L, np.zeros((0, 3), int), np.zeros(0), clock() - t0, {}))
                break
            stats = zscore_stats(all_maps)
            cand, cs, cm = [], [], []
            for lo, maps in zip(boxes, all_maps):
                comb = combine(maps, stats, methods)
                i = np.unravel_index(np.argmax(comb), comb.shape)
                cand.append(np.asarray(i) + lo + w // 2)
                cs.append(comb[i])
                cm.append({name: maps[name][i] for name in methods})
            cand, cs = np.array(cand), np.array(cs)
            order = np.argsort(-cs, kind="stable")
            keep = nms(cand[order], w // 2, ks[li])
            lut = {tuple(p): j for j, p in zip(order, cand[order])}
            idx = [lut[tuple(p)] for p in keep]
            centers, scores = keep, cs[idx]
            ms = {name: np.array([cm[j][name] for j in idx]) for name in methods}
        inside = np.all((centers >= rlo // f) & (centers < -(-rhi // f)), axis=1)
        centers, scores = centers[inside], scores[inside]
        ms = {name: v[inside] for name, v in ms.items()}
        prev = Stage(L, centers, scores, clock() - t0, ms)
        stages.append(prev)
    return stages


def to_working(stage: Stage, working: int) -> np.ndarray:
    """Candidate centres of a stage in working-level voxel coordinates."""
    f = 2 ** (stage.level - working)
    return stage.centers * f + f // 2


def rank_by_embedding(query_emb: np.ndarray, embs: np.ndarray) -> np.ndarray:
    """Cosine similarity of each embedding to the query embedding."""
    q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    e = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    return e @ q


# --------------------------------------------------------------------------------------
# Chunk-level best-first search (no pyramid needed; priors optional)
# --------------------------------------------------------------------------------------


def otsu(x: np.ndarray, bins: int = 256) -> tuple[float, float]:
    """Otsu's threshold of a 1-D sample and its effectiveness (between-class variance /
    total variance, 0..1; near 1 = clearly bimodal)."""
    x = np.asarray(x, float).ravel()
    x = x[np.isfinite(x)]
    if x.size < 2 or x.std() == 0:
        return float(x.mean()) if x.size else 0.0, 0.0
    hist, edges = np.histogram(x, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    mu0 = m0 / np.maximum(w0, 1)
    mu1 = (m0[-1] - m0) / np.maximum(w1, 1)
    between = w0 * w1 * (mu0 - mu1) ** 2 / w0[-1] ** 2
    i = int(np.argmax(between))
    return float(centers[i]), float(between[i] / x.var())


def otsu_threshold(x: np.ndarray, bins: int = 256) -> float:
    return otsu(x, bins)[0]


def empty_chunks(log_sizes: np.ndarray, min_separation: float = 0.6) -> np.ndarray:
    """Label-free 'empty' mask from compressed chunk sizes. Near-constant chunks (under 5%
    of the median size: padding, fill) are always empty; among the rest, the lower Otsu
    class is empty only if the split is clearly bimodal (effectiveness >= min_separation),
    because an Otsu split of one population would discard half the content."""
    x = np.asarray(log_sizes, float)
    pad = x < np.median(x) + np.log10(0.05)
    t, eta = otsu(x[~pad])
    return pad | ((x < t) if eta >= min_separation else np.zeros_like(pad))


def best_first_chunks(keys: np.ndarray, prior: np.ndarray, evaluate, budget: int, seed: float = 0.01,
                      explore: float = 0.0, beta: float = 1.0, rng=None) -> list[int]:
    """Order in which to read chunks, deciding as results come in.

    keys (n, 3): chunk grid positions. prior (n,): cheap prior, higher = look first,
    -inf = never read (e.g. empty). evaluate(i) -> best similarity found in chunk i
    (reads it and embeds its patches). Seeds with the top `seed` fraction by prior plus
    `explore` random chunks, then repeatedly reads the unread chunk maximizing
    z(prior) + beta * z(best similarity among its read neighbours).
    Returns the indices read, in order (at most `budget`)."""
    rng = rng or np.random.default_rng(0)
    n = len(keys)
    ok = np.isfinite(prior)
    if not ok.any():
        return []
    pz = np.full(n, -np.inf)
    p = prior[ok]
    pz[ok] = (p - p.mean()) / (p.std() + 1e-9) + 1e-6 * rng.random(ok.sum())  # random tie-break
    pos = {tuple(k): i for i, k in enumerate(keys)}
    offsets = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij")).reshape(3, -1).T
    offsets = offsets[np.any(offsets != 0, axis=1)]
    read = np.zeros(n, bool)
    nb_best = np.full(n, -np.inf)
    sims: list[float] = []
    order: list[int] = []

    def visit(i):
        read[i] = True
        order.append(i)
        s = evaluate(i)
        if s is None or not np.isfinite(s):
            return
        sims.append(s)
        for d in offsets:
            j = pos.get(tuple(keys[i] + d))
            if j is not None and not read[j]:
                nb_best[j] = max(nb_best[j], s)

    n_seed = max(1, int(round(seed * n)))
    cand = np.nonzero(ok)[0]
    for i in cand[np.argsort(-pz[cand], kind="stable")][:n_seed]:
        if len(order) < budget:
            visit(i)
    n_exp = int(round(explore * n))
    rest = cand[~read[cand]]
    for i in rng.permutation(rest)[:n_exp]:
        if len(order) < budget:
            visit(i)
    while len(order) < min(budget, ok.sum()):
        mu = np.mean(sims) if sims else 0.0
        sd = np.std(sims) + 1e-9 if len(sims) > 1 else 1.0
        nz = np.where(np.isfinite(nb_best), (nb_best - mu) / sd, 0.0)
        score = np.where(read | ~ok, -np.inf, pz + beta * nz)
        i = int(np.argmax(score))
        if not np.isfinite(score[i]):
            break
        visit(i)
    return order


def coarse_scan(read, query_box, region, working: int, level: int, patch: int, methods=("hist", "tex"),
                margin: int = 4, rotations: bool = True):
    """Cheap scores of every window position over the whole region at `level`.
    Returns (combined z-score map, centres of its windows in working-level voxels as a
    (*map.shape, 3) array)."""
    f = 2 ** (level - working)
    w = np.array(window_at(level, working, patch))
    qlo, qhi = (np.asarray(a) for a in query_box)
    rlo, rhi = (np.asarray(a) for a in region)
    m = margin
    qc = (qlo + qhi) // 2 // f
    q_ctx = read(level, qc - w // 2 - m, qc - w // 2 + w + m).astype(np.float32)
    q = q_ctx[m:-m, m:-m, m:-m]
    tq = query_texture(q_ctx, w, m)
    lo, hi = rlo // f, -(-rhi // f)
    vol = read(level, lo - m, hi + m).astype(np.float32)
    inner = vol[m:-m, m:-m, m:-m]
    edges = intensity_edges(inner)
    tex_scale = texture_stats(inner, w).reshape(4, -1).std(axis=1) + 1e-9
    maps = score_maps(vol, q, methods, edges, tq, tex_scale, rotations, margin=m)
    comb = combine(maps, zscore_stats([maps]), methods)
    idx = np.stack(np.meshgrid(*[np.arange(s) for s in comb.shape], indexing="ij"), -1)
    centers = (idx + lo + w // 2) * f + f // 2
    return comb, centers
