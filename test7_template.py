"""Test 7: Mars-CTX-explorer-style matching (token template sliding) against the baseline, on EM.

mars_ctx_explorer embeds whole tiles once with DINO, keeps every patch token, and matches a
dragged box as a TEMPLATE of tokens: the box's token grid is slid over each candidate's token grid
and scored by the mean cosine of aligned tokens (and, in its re-rank, by the cosine of mean-pooled
tokens). Here the same scheme on the labelled EM, so it can be compared with test 5's baseline
(one DINOv2-small embedding per 64 px patch, CLS + mean token, cosine) on the same 30 queries:

  * each working-level xy plane of the region (at test 5's candidate z planes) is resampled by
    112/64, so a 64 px patch spans 8 x 8 tokens, as in the baseline's 112 px input;
  * cut into non-overlapping 224 px crops (DINOv2-small, 16 x 16 tokens each) whose token grids
    are stitched into one grid per plane (marsctx stitches neighbouring tiles' grids the same way);
  * the query template is the 8 x 8 tokens around the query, from its own plane (in context);
  * template: mean over the 64 template cells of the cosine with the aligned candidate token;
    pooled: cosine of the window's mean token with the template's mean token;
  * candidates every token (8 working px), ranked with the same NMS and graded as tests 5/6.

    uv run python test7_template.py [--targets mito vesicle mito_full]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import test2_coarse2fine as t2
from common import RESULTS, Mirror

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".hf"))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
CROP, TOK = 224, 14  # model input and its token size (input px)
REGIONS = {"sub": (np.array([640, 0, 512]), np.array([2688, 1600, 4608])),
           "full": (np.zeros(3, int), np.array([6368, 1600, 12000]))}


class Tokens:
    def __init__(self):
        from transformers import AutoModel

        torch.set_num_threads(int(os.environ.get("TORCH_THREADS", os.cpu_count())))
        self.m = AutoModel.from_pretrained("facebook/dinov2-small").eval()

    def plane(self, img: np.ndarray, f: float):
        """(Y, X) uint8 plane -> stitched unit token grid (Gy, Gx, 384) float16 and a validity mask."""
        x = torch.from_numpy(img.astype(np.float32) / 255.0)[None, None]
        Y, X = round(img.shape[0] * f), round(img.shape[1] * f)
        x = F.interpolate(x, size=(Y, X), mode="bilinear", align_corners=False)[0, 0]
        py, px = -Y % CROP, -X % CROP
        x = F.pad(x[None, None], (0, px, 0, py), mode="replicate")[0, 0]
        ny, nx = x.shape[0] // CROP, x.shape[1] // CROP
        crops = x.reshape(ny, CROP, nx, CROP).permute(0, 2, 1, 3).reshape(-1, 1, CROP, CROP)
        out = []
        with torch.inference_mode():
            for i in range(0, len(crops), 32):
                b = (crops[i:i + 32].repeat(1, 3, 1, 1) - MEAN) / STD
                h = self.m(pixel_values=b).last_hidden_state[:, 1:]  # (B, 256, 384)
                out.append(F.normalize(h, dim=2).half())
        g = CROP // TOK
        G = torch.cat(out).reshape(ny, nx, g, g, -1).permute(0, 2, 1, 3, 4).reshape(ny * g, nx * g, -1)
        valid = np.zeros(G.shape[:2], bool)
        valid[: int(np.floor(Y / TOK)), : int(np.floor(X / TOK))] = True
        return G, valid


def run(target, tk, data, log):
    name, full = ("mito", True) if target == "mito_full" else (target, False)
    cfg = t2.TARGETS[name]
    W, P = cfg["working"], cfg["patch"]
    base = json.loads((RESULTS / f"test5_{target}_n30.json").read_text())
    queries = [np.asarray(q, int) for q in base["queries"]]
    lo_s0, hi_s0 = REGIONS["full" if full else "sub"]
    rlo, rhi = lo_s0 // 2 ** W, hi_s0 // 2 ** W
    gt = t2.instances(data.mirror, cfg["label"], W, rlo, rhi, cfg["min_voxels"])
    cents = gt["centroids"]
    f = 112 / P                    # resampling: a P px patch spans 8 x 8 tokens
    tpx = TOK / f                  # working px per token
    nt = round(P / tpx)            # template side in tokens (8)
    zs = t2.grid_centers(rlo, rhi, P)[:, 0]
    zs = np.unique(zs)
    t0 = time.time()

    def read_plane(z):
        return data.read(W, [z, rlo[1], rlo[2]], [z + 1, rhi[1], rhi[2]])[0]

    # query templates, from each query's own plane
    T = []
    for q in queries:
        G, _ = tk.plane(read_plane(q[0]), f)
        ty, tx = (q[1] - rlo[1]) / tpx - nt / 2, (q[2] - rlo[2]) / tpx - nt / 2
        ty, tx = int(round(ty)), int(round(tx))
        T.append(G[ty:ty + nt, tx:tx + nt].float())
    T = torch.stack(T)                                          # (Q, nt, nt, D)
    Tw = T.permute(0, 3, 1, 2).contiguous()                    # conv weights (Q, D, nt, nt)
    Tm = F.normalize(T.mean((1, 2)), dim=1)                     # (Q, D)
    log(f"[{target}] templates {tuple(T.shape)} ({time.time() - t0:.0f}s); {len(zs)} planes")

    cents_all, s_tpl, s_pool = [], [], []
    m0 = P / 2
    for k, z in enumerate(zs):
        G, valid = tk.plane(read_plane(z), f)
        x = G.float().permute(2, 0, 1)[None]                     # (1, D, Gy, Gx)
        tpl = F.conv2d(x, Tw)[0] / (nt * nt)                      # (Q, oh, ow): mean aligned cosine
        pooled = F.normalize(F.avg_pool2d(x, nt, stride=1)[0], dim=0)  # (D, oh, ow)
        pool = torch.einsum("qd,dhw->qhw", Tm, pooled)
        oh, ow = tpl.shape[1:]
        r, c = np.meshgrid(np.arange(oh), np.arange(ow), indexing="ij")
        cy = rlo[1] + (r + nt / 2) * tpx
        cx = rlo[2] + (c + nt / 2) * tpx
        ok = valid[r + nt - 1, c + nt - 1] & (cy >= rlo[1] + m0) & (cy <= rhi[1] - m0) & (cx >= rlo[2] + m0) & (cx <= rhi[2] - m0)
        cents_all.append(np.stack([np.full(ok.sum(), z), cy[ok], cx[ok]], 1))
        s_tpl.append(tpl[:, ok].numpy())
        s_pool.append(pool[:, ok].numpy())
        if k % 10 == 0:
            log(f"[{target}] plane {k + 1}/{len(zs)} ({time.time() - t0:.0f}s)")
    C = np.concatenate(cents_all)
    S = {"template": np.concatenate(s_tpl, 1), "pooled": np.concatenate(s_pool, 1)}
    res = {"target": target, "n_candidates": int(len(C)), "seconds": time.time() - t0, "rows": []}
    b = next(r for r in base["rows"] if (r["pool"], r["query"], r["sim"], r["input"]) == ("cls+mean", "single", "cosine", 112))
    for mode, Sm in S.items():
        gl = []
        for qi, qc in enumerate(queries):
            notself = ~np.all(np.abs(C - qc) < P / 2, 1)
            top = t2.rank_nms(C[notself], Sm[qi][notself], P)
            e = t2.evaluate(top, cents, P)
            gl.append({k: e[k] for k in ("p@10", "p@25", "r@100")})
        row = {"mode": mode}
        for k in ("p@10", "p@25", "r@100"):
            v = np.array([g[k] for g in gl])
            d = v - np.array(b[k + "_per_query"])
            row |= {k: float(v.mean()), k + "_se": float(v.std(ddof=1) / np.sqrt(len(v))), k + "_per_query": v.tolist(),
                    k + "_vs_base": float(d.mean()), k + "_vs_base_se": float(d.std(ddof=1) / np.sqrt(len(d)))}
        res["rows"].append(row)
        log(f"[{target}] {mode:<9} p@10 {row['p@10']:.2f}+/-{row['p@10_se']:.2f} ({row['p@10_vs_base']:+.3f}+/-{row['p@10_vs_base_se']:.3f} vs base)"
            f"  p@25 {row['p@25']:.2f}+/-{row['p@25_se']:.2f} ({row['p@25_vs_base']:+.3f}+/-{row['p@25_vs_base_se']:.3f})  r@100 {row['r@100']:.3f}")
    log(f"[{target}] baseline  p@10 {b['p@10']:.2f}  p@25 {b['p@25']:.2f}  r@100 {b['r@100']:.3f}  ({res['seconds']:.0f}s)")
    (RESULTS / f"test7_{target}.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=["mito", "vesicle", "mito_full"])
    args = ap.parse_args()
    logf = open(RESULTS / "test7.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    tk, data = Tokens(), t2.Data(Mirror())
    for t in args.targets:
        run(t, tk, data, log)


if __name__ == "__main__":
    main()
