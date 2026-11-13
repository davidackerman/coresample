"""Test 14: the rough pass from the tokens of whole 112 px crops, instead of one blown-up patch per run.

The page's rough pass (test 8's "model at W+2") puts one place per model run through DINOv2-small:
the box at level W+2 is P/4 px (16 px for a 64 px box), resized up to the model's 112 px input.
Here the same level is cut into plain 112 px crops (nothing resized), each run gives an 8 x 8 grid
of 14 px tokens, and every window of `nt` x `nt` tokens is a place: its mean token against the
query's (the same window around the query, from its own plane, in the same crop grid). A tile's
priority is its best window, as test 8's is its best place; tiles are read in that order and
graded as test 8 does (the working level's brute-force embeddings for the tiles read, test 5's
cache), so only the order differs. Also at W+1 (the box 2.3 tokens wide there).

Speed: the model runs and the model's seconds (this machine's CPU, TORCH_THREADS) each order
needed to rank the whole region, and its download (as in test 8). Accuracy: p@10 / p@25 after
reading a share of the tiles. Same 30 queries as tests 5-8.

    .venv/bin/python experiments/test14_rough_tokens.py [--targets mito vesicle mito_full]
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
from test8_levels import FRACS, REGIONS, coarse_grid, unit

import torch  # noqa: E402  (after test6_models, which sets HF_HOME)
import torch.nn.functional as F  # noqa: E402

CROP, TOK = 112, 14
# per level up: (enlargement r, token windows) - the box (P / 2^up px) spans P * r / 2^up / 14 tokens
TOKEN_ORDERS = {1: [(1, (1, 2))], 2: [(1, (1,)), (2, (2,)), (4, (4, 5))]}
MEAN, STD = torch.from_numpy(t6.IMNET_MEAN), torch.from_numpy(t6.IMNET_STD)


def token_grids(m, vol, planes, r=1):
    """Planes `planes` of `vol` (level L, z y x, uint8), enlarged `r` times (bilinear; 1: as they
    are), cut into 112 px crops from the origin (the far edges padded by replication): per plane a
    stitched grid of unit tokens (Gy, Gx, D), how many of its rows and columns are real data, the
    model runs and seconds. A token spans TOK / r px of level L."""
    Y, X = vol.shape[1] * r, vol.shape[2] * r
    py, px = -Y % CROP, -X % CROP
    ny, nx = (Y + py) // CROP, (X + px) // CROP
    crops = []
    for z in planes:
        x = torch.from_numpy(vol[z].astype(np.float32) / 255.0)[None, None]
        if r != 1:
            x = F.interpolate(x, size=(Y, X), mode="bilinear", align_corners=False)
        x = F.pad(x, (0, px, 0, py), mode="replicate")[0, 0]
        crops.append(x.reshape(ny, CROP, nx, CROP).permute(0, 2, 1, 3).reshape(-1, CROP, CROP))
    crops = torch.cat(crops)
    out, t = [], time.time()
    with torch.inference_mode():
        for i in range(0, len(crops), 64):
            b = (crops[i:i + 64, None].repeat(1, 3, 1, 1) - MEAN) / STD
            h = m.m(pixel_values=b).last_hidden_state[:, m.skip:]
            out.append(F.normalize(h, dim=2))
    secs = time.time() - t
    g = CROP // TOK
    G = torch.cat(out).reshape(len(planes), ny, nx, g, g, -1).permute(0, 1, 3, 2, 4, 5).reshape(len(planes), ny * g, nx * g, -1)
    return G, (Y // TOK, X // TOK), len(crops), secs


def windows(G, real, nt, r=1):
    """Every nt x nt window of each plane's tokens (stride one token) that lies on real data: its
    unit mean token (n, D), and its plane index and centre in level-L px from the grid origin."""
    pooled = F.normalize(F.avg_pool2d(G.permute(0, 3, 1, 2), nt, stride=1), dim=1)  # (nz, D, oh, ow)
    nz, D, oh, ow = pooled.shape
    rr, c = np.meshgrid(np.arange(oh), np.arange(ow), indexing="ij")
    ok = (rr + nt <= real[0]) & (c + nt <= real[1])
    feats = pooled.permute(0, 2, 3, 1)[:, torch.from_numpy(ok)]  # (nz, k, D)
    cy, cx = (rr[ok] + nt / 2) * TOK / r, (c[ok] + nt / 2) * TOK / r
    zi = np.repeat(np.arange(nz), ok.sum())
    return feats.reshape(-1, D).numpy(), zi, np.tile(cy, nz), np.tile(cx, nz)


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

    def owners(zw, yw, xw):
        return np.array([cid.get(tuple(k), -1) for k in np.stack([zw, yw, xw], 1).astype(int) // ch])

    orders = {}  # name -> (per-query priority function, {runs, seconds, bytes})
    for up in (1, 2):
        L, p, f = W + up, P // 2 ** up, 2 ** up
        lo, hi = rlo // f, -(-rhi // f)
        vol = data.read(L, lo, hi)
        cost = data.cost([(data.path(L), lo, hi)], region_s0)["bytes"] / wbytes
        cc = coarse_grid(lo, hi, p)
        qc = [q // f for q in queries]

        # the page's rough pass: one place per run, P/4 (or P/2) px resized up to 112
        t = time.time()
        Ec = m(t2.patches_from(vol, lo, cc, p))
        secs = time.time() - t
        Qc = m(np.stack([data.read(L, c - [0, p // 2, p // 2], c + [1, p // 2, p // 2])[0] for c in qc]))
        own = owners(*(cc * f + f // 2).T)  # as test 8

        def patch_prio(qi, cc=cc, Ec=Ec, Qc=Qc, own=own, p=p, f=f):
            s = Ec @ Qc[qi]
            s[np.all(np.abs(cc - queries[qi] // f) < p / 2, 1)] = -np.inf  # the query's own cell
            pr = np.full(n, -np.inf)
            ok = own >= 0
            np.maximum.at(pr, own[ok], s[ok])
            return pr

        orders[f"model W+{up}"] = (patch_prio, {"runs": len(cc), "seconds": secs, "bytes": cost})
        log(f"[{target}] s{L} patches: {len(cc)} runs of {p} px resized to 112 ({secs:.1f}s)")

        # tokens of 112 px crops (of the level as it is, or enlarged r times), on the same planes
        planes = np.unique(cc[:, 0]) - lo[0]
        qplanes = np.unique([c[0] - lo[0] for c in qc])
        qrow = {zq: i for i, zq in enumerate(qplanes)}
        for r, nts in TOKEN_ORDERS[up]:
          G, real, runs, secs = token_grids(m, vol, planes, r)
          Gq, _, _, _ = token_grids(m, vol, qplanes, r)  # the queries' own planes (not counted: one crop each in the page)
          log(f"[{target}] s{L} tokens x{r}: {runs} runs of 112 px crops ({secs:.1f}s), {G.shape[1]} x {G.shape[2]} tokens per plane")
          for nt in nts:
            feats, zi, cy, cx = windows(G, real, nt, r)
            zl = planes[zi] + lo[0]
            own = owners(zl * f + f // 2, (cy + lo[1]) * f, (cx + lo[2]) * f)
            qf = []
            for c in qc:
                g = Gq[qrow[c[0] - lo[0]]]
                ty = int(round((c[1] - lo[1]) * r / TOK - nt / 2)); tx = int(round((c[2] - lo[2]) * r / TOK - nt / 2))
                ty = min(max(ty, 0), g.shape[0] - nt); tx = min(max(tx, 0), g.shape[1] - nt)
                qf.append(F.normalize(g[ty:ty + nt, tx:tx + nt].reshape(-1, g.shape[-1]).mean(0), dim=0).numpy())
            qf = np.stack(qf)

            def tok_prio(qi, feats=feats, zl=zl, cy=cy, cx=cx, own=own, qf=qf, p=p, lo=lo):
                c = qc[qi]
                s = feats @ qf[qi]
                s[(np.abs(zl - c[0]) < p / 2) & (np.abs(cy + lo[1] - c[1]) < p / 2) & (np.abs(cx + lo[2] - c[2]) < p / 2)] = -np.inf
                pr = np.full(n, -np.inf)
                ok = own >= 0
                np.maximum.at(pr, own[ok], s[ok])
                return pr

            orders[f"tokens W+{up} x{r} {nt}x{nt}"] = (tok_prio, {"runs": runs, "seconds": secs, "bytes": cost})
          del G, Gq
        del vol

    Lc = cfg["default_coarsest"]
    fc = 2 ** (Lc - W)
    scan_cost = data.cost([(data.path(Lc), rlo // fc - 4, -(-rhi // fc) + 4)], region_s0)["bytes"] / wbytes
    names = ["random", "cheap scan"] + list(orders)
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
        for k, (fn, _) in orders.items():
            prio[k] = fn(qi)
        rng = np.random.default_rng(qi)
        prio["random"] = rng.random(n)
        for k in names:
            p_ = prio[k] + rng.random(n) * 1e-9  # ties broken at random
            order = np.argsort(-p_, kind="stable")
            curve = []
            for fr in FRACS:
                sel = np.zeros(n, bool)
                sel[order[: max(1, int(round(fr * n)))]] = True
                mm = sel[inv] & notself
                e = t2.evaluate(t2.rank_nms(gc[mm], sims[mm], P), cents, P)
                extra = scan_cost if k == "cheap scan" else orders[k][1]["bytes"] if k in orders else 0.0
                curve.append({"frac": fr, "frac_bytes": float(chunk_bytes[sel].sum() / total + extra),
                              "p@10": e["p@10"], "p@25": e["p@25"]})
            curves[k].append(curve)
    full_read = [t2.evaluate(t2.rank_nms(gc[~np.all(np.abs(gc - q) < P / 2, 1)], (E @ Q[qi])[~np.all(np.abs(gc - q) < P / 2, 1)], P), cents, P)
                 for qi, q in enumerate(queries)]
    ceil10, ceil25 = np.mean([e["p@10"] for e in full_read]), np.mean([e["p@25"] for e in full_read])
    res = {"target": target, "n_chunks": int(n), "fracs": FRACS, "scan_cost": scan_cost, "ceiling": {"p@10": ceil10, "p@25": ceil25},
           "speed": {k: v[1] for k, v in orders.items()}, "curves": curves}
    log(f"[{target}] {n} tiles; p@10/p@25 over {len(queries)} queries after reading a share of the tiles "
        f"(every tile read: {ceil10:.2f}/{ceil25:.2f}); bytes incl. the order's own:")
    for k in names:
        row = []
        for i, fr in enumerate(FRACS):
            p10 = np.mean([c[i]["p@10"] for c in curves[k]])
            p25 = np.mean([c[i]["p@25"] for c in curves[k]])
            fb = np.mean([c[i]["frac_bytes"] for c in curves[k]])
            row.append(f"{fr:.1%}: {p10:.2f}/{p25:.2f} ({fb:.1%})")
        sp = orders[k][1] if k in orders else None
        log(f"   {k:<16} " + "  ".join(row) + (f"   [{sp['runs']} runs, {sp['seconds']:.1f}s]" if sp else ""))
    (RESULTS / f"test14_{target}.json").write_text(json.dumps(res, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=["mito", "vesicle", "mito_full"])
    args = ap.parse_args()
    logf = open(RESULTS / "test14.log", "a")

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
