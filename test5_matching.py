"""Test 5: what makes the best match? Matching variants on the same candidates, graded by labels.

The search decides which patches get embedded; this test asks, given the same patches, which way
of embedding and comparing them ranks the right ones first. Every variant re-ranks the
brute-force grid of the region (all patches), so differences are the matcher's alone.

Variants (all from one pass of DINOv2-small over the patches, keeping its token outputs):
  pooling     cls | mean | center (Gaussian-weighted tokens around the patch centre) |
              cls+mean (the current) | cls+center
  query       single | rot-max (best of 8 rotations/flips) | rot-mean (their average) |
              scale-max (best of the query at half, same and double field) | rot+scale-max
  similarity  cosine | centered (minus the candidates' mean embedding) | whitened (PCA of the
              candidates, 128 dims) | qe (alpha query expansion with the top 5)
plus, for mito: input 224 instead of 112, and token max-sim (mean over query tokens of the best
matching candidate token); and feedback: Rocchio vs the good hits as extra queries (max).

    uv run python test5_matching.py [--targets mito vesicle] [--queries 10] [--full-mito]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np
import torch

import test2_coarse2fine as t2
from common import CACHE, RESULTS, Mirror

os.environ.setdefault("HF_HOME", str(t2.RESULTS.parent / ".hf"))
MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
POOLS = ["cls", "mean", "center", "cls+mean", "cls+center"]
QMODES = ["single", "rot-max", "rot-mean", "scale-max", "rot+scale-max"]
SIMS = ["cosine", "centered", "whitened", "qe"]


class Model:
    def __init__(self, size=112, keep_tokens=False):
        from transformers import AutoModel

        torch.set_num_threads(os.cpu_count())
        self.m = AutoModel.from_pretrained("facebook/dinov2-small").eval()
        self.size = size
        self.keep_tokens = keep_tokens
        g = size // 14
        yy, xx = np.mgrid[0:g, 0:g] - (g - 1) / 2
        w = np.exp(-(yy ** 2 + xx ** 2) / (2 * (g / 4) ** 2)).ravel()
        self.center_w = torch.from_numpy((w / w.sum()).astype(np.float32))

    def __call__(self, patches: np.ndarray, batch=128):
        """(n, P, P) uint8 -> dict of pooled (n, d) float32 arrays (+ tokens if kept)."""
        out = {k: [] for k in ["cls", "mean", "center"] + (["tokens"] if self.keep_tokens else [])}
        with torch.inference_mode():
            for i in range(0, len(patches), batch):
                x = torch.from_numpy(patches[i:i + batch].astype(np.float32) / 255.0)[:, None]
                x = torch.nn.functional.interpolate(x, size=(self.size,) * 2, mode="bilinear", align_corners=False)
                x = (x.repeat(1, 3, 1, 1) - torch.from_numpy(MEAN)) / torch.from_numpy(STD)
                h = self.m(pixel_values=x).last_hidden_state
                tok = h[:, 1:]
                out["cls"].append(h[:, 0].numpy())
                out["mean"].append(tok.mean(1).numpy())
                out["center"].append((tok * self.center_w[None, :, None]).sum(1).numpy())
                if self.keep_tokens:
                    out["tokens"].append(torch.nn.functional.normalize(tok, dim=2).half().numpy())
        return {k: np.concatenate(v) for k, v in out.items()}


def unit(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)


def pooled(d, pool):
    if "+" in pool:
        a, b = pool.split("+")
        return np.concatenate([unit(d[a]), unit(d[b])], -1) / np.sqrt(2)
    return unit(d[pool])


def query_patches(data, W, P, qc):
    """The query's variants: 8 rotations/flips at the working level, and the same centre at
    half (level W-1) and double (level W+1) field of view."""
    h = P // 2
    base = data.read(W, qc - [0, h, h], qc + [1, h, h])[0]
    rots = [np.rot90(f, k) for f in (base, base[:, ::-1]) for k in range(4)]
    scales = [base]
    for dl in (-1, 1):
        L = W + dl
        if L < 0:
            continue
        c = np.array([qc[0] * 2 ** W // 2 ** L, qc[1] * 2 ** W // 2 ** L, qc[2] * 2 ** W // 2 ** L])
        scales.append(data.read(L, c - [0, h, h], c + [1, h, h])[0])
    return np.stack(rots), np.stack(scales)


def scores(E, Q, sim, mean=None, white=None, qe_k=5, qe_a=3.0):
    """Candidate scores for query embeddings Q (m, d): max over the m query variants."""
    if sim == "centered":
        E, Q = unit(E - mean), unit(Q - mean)
    elif sim == "whitened":
        E, Q = unit((E - mean) @ white), unit((Q - mean) @ white)
    s = (E @ Q.T).max(1)
    if sim == "qe":  # alpha query expansion: re-query with the query plus its top hits, weighted by score^alpha
        top = np.argsort(-s)[:qe_k]
        w = np.clip(s[top], 0, None) ** qe_a
        q2 = unit(Q.mean(0) + (w[:, None] * E[top]).sum(0))
        s = E @ q2
    return s


def grade(gc, s, notself, cents, P):
    top = t2.rank_nms(gc[notself], s[notself], P)
    e = t2.evaluate(top, cents, P)
    return {k: e[k] for k in ("p@10", "p@25", "r@100")}


def run(name, cfg, args, data, log, full=False):
    W, P = cfg["working"], cfg["patch"]
    data.working = W
    if full:
        t2.REGION_S0 = (np.zeros(3, int), np.array([6368, 1600, 12000]))
    else:
        t2.REGION_S0 = (np.array([640, 0, 512]), np.array([2688, 1600, 4608]))
    key = name + ("_full" if full else "")
    rlo, rhi = t2.REGION_S0[0] // 2 ** W, t2.REGION_S0[1] // 2 ** W
    gt = t2.instances(data.mirror, cfg["label"], W, rlo, rhi, cfg["min_voxels"])
    cents = gt["centroids"]
    rng = np.random.default_rng(args.seed)
    ok = gt["big"] & np.all((cents >= rlo + P) & (cents < rhi - P), axis=1)
    queries = []
    for j in rng.permutation(np.nonzero(ok)[0]):
        pt = t2.inside_point(data.mirror, cfg["label"], W, int(gt["ids"][j]), cents[j], P // 2)
        if pt is not None and np.all((pt >= rlo + P) & (pt < rhi - P)):
            queries.append(np.asarray(pt, int))
        if len(queries) >= args.queries:
            break
    gc = t2.grid_centers(rlo, rhi, P)
    log(f"[{key}] {len(cents)} instances, {len(gc)} candidate patches, {len(queries)} queries")

    def embed_all(model, tag):
        f = CACHE / f"test5_{key}_{tag}.npz"
        if f.exists():
            return dict(np.load(f))
        t = time.time()
        vol = data.read(W, rlo, rhi)
        out = {}
        for i in range(0, len(gc), 4096):
            d = model(t2.patches_from(vol, rlo, gc[i:i + 4096], P))
            for k, v in d.items():
                out.setdefault(k, []).append(v.astype(np.float16) if k != "tokens" else v)
        out = {k: np.concatenate(v) for k, v in out.items()}
        np.savez(f, **out)
        log(f"[{key}] embedded {len(gc)} patches ({tag}) in {time.time() - t:.0f}s")
        return out

    m112 = Model(112)
    C = embed_all(m112, "112")
    res = {"n_instances": int(len(cents)), "n_candidates": int(len(gc)), "queries": [q.tolist() for q in queries], "rows": []}
    per = {}
    base32 = {k: C[k].astype(np.float32) for k in ("cls", "mean", "center")}
    prep = {}
    for pool in POOLS:  # pooled candidates, their mean and a PCA whitening, from the candidates alone (no labels)
        E = pooled(base32, pool)
        mu = E.mean(0)
        Ec = E[::max(1, len(E) // 20000)] - mu
        _, sv, vt = np.linalg.svd(Ec, full_matrices=False)
        prep[pool] = (E, mu, vt[:128].T / (sv[:128] / np.sqrt(len(Ec)) + 1e-6))
    for qi, qc in enumerate(queries):
        rots, scs = query_patches(data, W, P, qc)
        Qr, Qs = m112(rots), m112(scs)
        notself = ~np.all(np.abs(gc - qc) < P / 2, 1)
        for pool in POOLS:
            E, mu, white = prep[pool]
            qr, qs = pooled(Qr, pool), pooled(Qs, pool)
            qsets = {"single": qr[:1], "rot-max": qr, "rot-mean": unit(qr.mean(0, keepdims=True)),
                     "scale-max": qs, "rot+scale-max": np.concatenate([qr, qs[1:]])}
            for qm, sim in itertools.product(QMODES, SIMS):
                s = scores(E, qsets[qm], sim, mu, white)
                per.setdefault((pool, qm, sim), []).append(grade(gc, s, notself, cents, P))
        log(f"[{key} q{qi}] done")

    for (pool, qm, sim), gl in per.items():
        res["rows"].append({"pool": pool, "query": qm, "sim": sim, "input": 112,
                            **{k: float(np.mean([g[k] for g in gl])) for k in ("p@10", "p@25", "r@100")}})

    # feedback: Rocchio (the page's current rule) vs good hits as extra queries (max), on cls+mean cosine
    E = pooled({k: C[k].astype(np.float32) for k in ("cls", "mean", "center")}, "cls+mean")
    fb = {"none": [], "rocchio": [], "multi-max": []}
    for qc in queries:
        rots, _ = query_patches(data, W, P, qc)
        q = pooled(m112(rots[:1]), "cls+mean")
        notself = ~np.all(np.abs(gc - qc) < P / 2, 1)
        s0 = (E @ q.T).max(1)
        top = t2.rank_nms(gc[notself], s0[notself], P)[:10]
        idx = [int(np.nonzero(np.all(gc == c, 1))[0][0]) for c in top]
        hitsets = np.all(np.abs(gc[idx][:, None, :] - cents[None]) < P / 2, 2).any(1)
        good, bad = [i for i, h in zip(idx, hitsets) if h], [i for i, h in zip(idx, hitsets) if not h]
        shown = np.zeros(len(gc), bool)
        shown[idx] = True
        keep = notself & ~shown
        fb["none"].append(grade(gc, s0, keep, cents, P))
        qq = q[0] + (0.75 * E[good].mean(0) if good else 0) - (0.25 * E[bad].mean(0) if bad else 0)
        fb["rocchio"].append(grade(gc, E @ unit(qq), keep, cents, P))
        fb["multi-max"].append(grade(gc, (E @ unit(np.concatenate([q, E[good]])).T).max(1), keep, cents, P))
    res["feedback"] = {k: {m: float(np.mean([g[m] for g in v])) for m in ("p@10", "p@25", "r@100")} for k, v in fb.items()}

    if name == "mito":  # input 224 (4x the compute), and token max-sim
        m224 = Model(224)
        C2 = embed_all(m224, "224")
        for pool in ("cls+mean", "cls+center", "center"):
            E = pooled({k: C2[k].astype(np.float32) for k in ("cls", "mean", "center")}, pool)
            gl = {"single": [], "rot-max": []}
            for qc in queries:
                rots, _ = query_patches(data, W, P, qc)
                qr = pooled(m224(rots), pool)
                notself = ~np.all(np.abs(gc - qc) < P / 2, 1)
                gl["single"].append(grade(gc, (E @ qr[:1].T).max(1), notself, cents, P))
                gl["rot-max"].append(grade(gc, (E @ qr.T).max(1), notself, cents, P))
            for qm, v in gl.items():
                res["rows"].append({"pool": pool, "query": qm, "sim": "cosine", "input": 224,
                                    **{k: float(np.mean([g[k] for g in v])) for k in ("p@10", "p@25", "r@100")}})
        if not full:
            mt = Model(112, keep_tokens=True)
            CT = embed_all(mt, "112tok")
            T = CT["tokens"].astype(np.float32)  # (n, 64, 384), unit tokens
            gl = []
            for qc in queries:
                rots, _ = query_patches(data, W, P, qc)
                qt = mt(rots[:1])["tokens"].astype(np.float32)[0]  # (64, 384)
                notself = ~np.all(np.abs(gc - qc) < P / 2, 1)
                s = np.einsum("ntd,qd->ntq", T, qt).max(1).mean(1)  # each query token's best match, averaged
                gl.append(grade(gc, s, notself, cents, P))
            res["rows"].append({"pool": "tokens", "query": "single", "sim": "max-sim", "input": 112,
                                **{k: float(np.mean([g[k] for g in gl])) for k in ("p@10", "p@25", "r@100")}})

    (RESULTS / f"test5_{key}.json").write_text(json.dumps(res, indent=1))
    rows = sorted(res["rows"], key=lambda r: -(r["p@10"] + r["p@25"]))
    base = next(r for r in res["rows"] if r["pool"] == "cls+mean" and r["query"] == "single" and r["sim"] == "cosine" and r["input"] == 112)
    log(f"== {key}: current (cls+mean, single, cosine, 112): p@10 {base['p@10']:.2f} p@25 {base['p@25']:.2f} r@100 {base['r@100']:.3f}")
    for r in rows[:8]:
        log(f"   {r['pool']:<10} {r['query']:<13} {r['sim']:<9} {r['input']}: p@10 {r['p@10']:.2f} p@25 {r['p@25']:.2f} r@100 {r['r@100']:.3f}")
    log(f"   feedback: " + " | ".join(f"{k} p@10 {v['p@10']:.2f} p@25 {v['p@25']:.2f}" for k, v in res["feedback"].items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=["mito", "vesicle"])
    ap.add_argument("--queries", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--full-mito", action="store_true")
    args = ap.parse_args()
    logf = open(RESULTS / "test5.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    data = t2.Data(Mirror())
    for name in args.targets:
        run(name, t2.TARGETS[name], args, data, log)
    if args.full_mito:
        run("mito", t2.TARGETS["mito"], args, data, log, full=True)


if __name__ == "__main__":
    main()
