"""Test 1: does compressed chunk size (free, from a bucket listing) track content?

Chunk sizes come from chunkmap.py (listing only, no pixel reads). Reference labels are
the whole-volume COSEM segmentations (labels are used only to grade). Every map is
on the s0 EM chunk grid: one cell = 128^3 s0 voxels (0.67 x 0.51 x 0.51 um).

Reports Spearman(size, occupancy), ROC AUC of size for "cell contains class", with and
without the resin/background confounder (foreground-only), across four encodings of
the same image: zarr uint8 zstd at s0, s1, s2 and N5 uint16 gzip at s0. As a yardstick
it also scores a cheap pixel read: per-cell intensity std from EM s3 (1/512 of s0).

    uv run python test1_chunksize.py
"""

from __future__ import annotations

import argparse
import json
import time
from types import SimpleNamespace

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

import chunkmap
from common import CACHE, DATASET, EM_ZARR, LABELS_N5, RESULTS, BASE_URL, Mirror

CELL = 128  # s0 voxels per map cell (the s0 zarr chunk)
ORGANELLES = ["cent", "chrom", "endo", "er", "eres", "golgi", "ld", "lyso", "mito", "mt-out",
              "ne", "nhchrom", "np", "nucleolus", "nucleus", "pm", "ribo", "vesicle"]
FOREGROUND = f"{DATASET}/{DATASET}.zarr/recon-1/labels/masks/foreground"  # s1 resolution
S0_SHAPE = np.array([6368, 1600, 12000])
GRID = -(-S0_SHAPE // CELL)  # (50, 13, 94)


def sizes_from_chunkmap(url: str, max_chunks: int = 600000) -> dict[str, dict]:
    a = SimpleNamespace(url=chunkmap.normalize_url(url), level=None, max_chunks=max_chunks,
                        inner=False, threads=32)
    n5 = chunkmap.get_json(a.url + "/attributes.json")
    levels = chunkmap.n5_levels(a, n5) if n5 else chunkmap.zarr_levels(a)
    return {L["label"].split("/")[-1].split()[0]: L for L in levels}


def to_cell_grid(sizes: np.ndarray, chunks, level: int) -> np.ndarray:
    """Re-express a level's chunk-size grid on the s0 128^3 cell grid (sum or repeat)."""
    s0_extent = np.array(chunks) * 2**level  # s0 voxels per chunk
    s = np.nan_to_num(sizes, nan=0.0)
    if np.all(s0_extent == CELL):
        return s
    if np.all(s0_extent < CELL):  # finer chunks: sum
        f = CELL // s0_extent
        pad = [(0, (-n) % k) for n, k in zip(s.shape, f)]
        s = np.pad(s, pad)
        g = s.reshape(s.shape[0] // f[0], f[0], s.shape[1] // f[1], f[1], s.shape[2] // f[2], f[2]).sum((1, 3, 5))
        return g[: GRID[0], : GRID[1], : GRID[2]]
    f = s0_extent // CELL  # coarser chunks: spread over the cells they cover
    g = np.repeat(np.repeat(np.repeat(s, f[0], 0), f[1], 1), f[2], 2) / np.prod(f)
    return g[: GRID[0], : GRID[1], : GRID[2]]


def occupancy(mirror: Mirror, path: str, level: int, fmt_equal=None, slab_cells: int = 4):
    """Per-cell fraction of voxels that are non-zero (or == fmt_equal) for an array at
    `level` (factor 2**level vs s0), streaming z slabs."""
    info = mirror.info(path)
    b = CELL // 2**level  # label voxels per cell edge
    counts = np.zeros(GRID, dtype=np.float64)
    valid = np.zeros(GRID, dtype=np.float64)
    shape = np.array(info.shape)
    for c0 in range(0, GRID[0], slab_cells):
        c1 = min(GRID[0], c0 + slab_cells)
        lo = np.array([c0 * b, 0, 0])
        hi = np.array([c1 * b, GRID[1] * b, GRID[2] * b])
        arr = mirror.read(path, lo, hi)
        m = (arr > 0) if fmt_equal is None else (arr == fmt_equal)
        inside = np.zeros(arr.shape, bool)
        e = np.minimum(hi, shape) - lo
        inside[: e[0], : e[1], : e[2]] = True
        nz = (c1 - c0)
        r = lambda x: x.reshape(nz, b, GRID[1], b, GRID[2], b).sum((1, 3, 5), dtype=np.float64)
        counts[c0:c1] = r(m & inside)
        valid[c0:c1] = r(inside)
        del arr, m, inside
    return counts, valid


def em_std_s3(mirror: Mirror):
    """Per-cell intensity std and mean from EM s3 (16^3 voxels per cell)."""
    path = f"{EM_ZARR}/s3"
    info = mirror.info(path)
    b = CELL // 8
    arr = mirror.read(path, [0, 0, 0], GRID * b).astype(np.float32)
    r = lambda x: x.reshape(GRID[0], b, GRID[1], b, GRID[2], b)
    v = r(arr)
    return v.std(axis=(1, 3, 5)), v.mean(axis=(1, 3, 5))


def auc(score, y):
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return float("nan")
    return float(roc_auc_score(y, score))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()
    out = RESULTS
    out.mkdir(exist_ok=True)
    mirror = Mirror()
    t0 = time.time()

    # ---- chunk sizes (listing only) ----
    zl = sizes_from_chunkmap(f"{BASE_URL}/{EM_ZARR}")
    nl = sizes_from_chunkmap(f"{BASE_URL}/{DATASET}/{DATASET}.n5/em/fibsem-uint16")
    encodings = {}
    for lv in ("s0", "s1", "s2"):
        L = zl[lv]
        encodings[f"zarr uint8 zstd {lv} (chunk {L['chunks'][0]}^3)"] = to_cell_grid(L["sizes"], L["chunks"], int(lv[1]))
    L = nl["s0"]
    encodings["n5 uint16 gzip s0 (chunk 64^3, summed)"] = to_cell_grid(L["sizes"], L["chunks"], 0)
    raw_missing_n5 = np.isnan(nl["s0"]["sizes"]).mean()
    print(f"sizes listed in {time.time() - t0:.0f}s")

    # ---- reference labels (grading only) ----
    t1 = time.time()
    cache = CACHE / "test1_occupancy.npz"
    if cache.exists():
        occ = dict(np.load(cache))
    else:
        occ = {}
        fg, fgv = occupancy(mirror, FOREGROUND, 1)
        occ["foreground"] = fg / np.maximum(fgv, 1)
        for cls, lv in (("mito", 1), ("vesicle", 1)):
            c, v = occupancy(mirror, f"{LABELS_N5}/{cls}_seg/s{lv}", lv)
            occ[cls] = c / np.maximum(v, 1)
            print(f"  {cls} occupancy done ({time.time() - t1:.0f}s)")
        union = None
        for cls in ORGANELLES:  # union at s3 (32 nm): per-cell any-organelle fraction
            info = mirror.info(f"{LABELS_N5}/{cls}_seg/s3")
            b = CELL // 8
            arr = mirror.read(info.path, [0, 0, 0], GRID * b) > 0
            union = arr if union is None else (union | arr)
            occ[f"{cls}@s3"] = arr.reshape(GRID[0], b, GRID[1], b, GRID[2], b).mean((1, 3, 5))
        inside = np.zeros(GRID * (CELL // 8), bool)
        e = np.array(mirror.info(f"{LABELS_N5}/mito_seg/s3").shape)
        inside[: e[0], : e[1], : e[2]] = True
        b = CELL // 8
        rs = lambda x: x.reshape(GRID[0], b, GRID[1], b, GRID[2], b).sum((1, 3, 5), dtype=np.float64)
        occ["any"] = rs(union & inside) / np.maximum(rs(inside), 1)
        occ["valid"] = rs(inside) / b**3
        occ["em_std_s3"], occ["em_mean_s3"] = em_std_s3(mirror)
        np.savez_compressed(cache, **occ)
    print(f"labels in {time.time() - t1:.0f}s")

    valid = occ["valid"] > 0.99  # full cells only (drop the half cells at the y edge)
    fgm = valid & (occ["foreground"] > 0.5)
    targets = {
        "foreground (>50% of cell)": occ["foreground"] > 0.5,
        "mito (any voxel)": occ["mito"] > 0,
        "mito (>=5% of cell)": occ["mito"] >= 0.05,
        "vesicle (any voxel)": occ["vesicle"] > 0,
        "vesicle (>=0.5% of cell)": occ["vesicle"] >= 0.005,
    }

    rows = []
    signals = dict(encodings)
    signals.update(derived_signals(encodings, occ, valid))
    signals["EM s3 intensity std (cheap read, 1/512 of s0)"] = occ["em_std_s3"]
    for name, s in signals.items():
        row = {"signal": name}
        rho_all = spearmanr(s[valid], occ["any"][valid]).statistic
        rho_fg = spearmanr(s[fgm], occ["any"][fgm]).statistic
        row["spearman_any_all"] = float(rho_all)
        row["spearman_any_fg"] = float(rho_fg)
        for tname, y in targets.items():
            row[f"auc_all:{tname}"] = auc(s[valid], y[valid])
            if not tname.startswith("foreground"):
                row[f"auc_fg:{tname}"] = auc(s[fgm], y[fgm])
        for cls in ("mito", "vesicle"):
            row[f"spearman_{cls}_fg"] = float(spearmanr(s[fgm], occ[cls][fgm]).statistic)
        rows.append(row)

    summary = {
        "cells": int(valid.sum()),
        "foreground_cells": int(fgm.sum()),
        "positives_all": {k: int(v[valid].sum()) for k, v in targets.items()},
        "positives_fg": {k: int(v[fgm].sum()) for k, v in targets.items()},
        "n5_s0_blocks_missing_fraction": float(raw_missing_n5),
        "rows": rows,
        "zarr_s0_bytes_GB": float(np.nansum(zl["s0"]["sizes"]) / 1e9),
        "zarr_s0_raw_GB": float(np.prod(S0_SHAPE) / 1e9),
    }
    (out / "test1.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(out / "test1_grids.npz", **{k.split(" (")[0].replace(" ", "_"): v for k, v in encodings.items()}, **occ)

    # ---- table ----
    tcols = list(targets)
    print("\nsignal | rho(any) all | rho(any) fg | " + " | ".join(f"AUC all {t}" for t in tcols)
          + " | " + " | ".join(f"AUC fg {t}" for t in tcols[1:]))
    for r in rows:
        print(r["signal"], "|", f"{r['spearman_any_all']:.2f} | {r['spearman_any_fg']:.2f} | "
              + " | ".join(f"{r[f'auc_all:{t}']:.2f}" for t in tcols) + " | "
              + " | ".join(f"{r[f'auc_fg:{t}']:.2f}" for t in tcols[1:]))
    print(json.dumps({k: summary[k] for k in ("cells", "foreground_cells", "positives_all", "positives_fg")}, indent=1))

    figures(out, encodings, occ, valid, fgm, targets)
    print(f"total {time.time() - t0:.0f}s")


def trend_residual(x, y, mask, bins: int = 30):
    """y minus the binned median of y against x (bins = quantiles of x over mask)."""
    edges = np.unique(np.quantile(x[mask], np.linspace(0, 1, bins + 1)))
    centers, med = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = mask & (x >= a) & (x <= b)
        if m.sum() >= 5:
            centers.append(np.median(x[m]))
            med.append(np.median(y[m]))
    return y - np.interp(x, centers, med), (np.array(centers), np.array(med))


def derived_signals(encodings, occ, valid):
    """Free (listing-only) variants of the size signal, plus the addendum's residual
    from the size-vs-coarse-intensity trend (needs only the coarsest level)."""
    from scipy.ndimage import median_filter

    k0 = next(k for k in encodings if " s0 " in k and k.startswith("zarr"))
    k1 = next(k for k in encodings if " s1 " in k)
    ls0 = np.log10(np.maximum(encodings[k0], 1))
    ls1 = np.log10(np.maximum(encodings[k1], 1))
    out = {}
    out["free: s0/s1 size ratio (same region, two levels)"] = ls0 - ls1
    out["free: s0 size minus 5^3-cell local median"] = ls0 - median_filter(ls0, size=5, mode="nearest")
    res, _ = trend_residual(occ["em_mean_s3"], ls0, valid)
    out["residual: s0 log size vs coarse mean intensity"] = res
    out["|residual| (outlier rank)"] = np.abs(res)
    return out


def figures(out, encodings, occ, valid, fgm, targets):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s0 = encodings[next(iter(encodings))]
    n5 = encodings[[k for k in encodings if k.startswith("n5")][0]]
    ys = [3, 6, 9]
    panels = [
        ("zarr uint8 zstd s0 size (MB)", s0 / 1e6, "magma", None),
        ("n5 uint16 gzip s0 size (MB)", n5 / 1e6, "magma", None),
        ("EM s3 intensity std", occ["em_std_s3"], "magma", None),
        ("foreground fraction", occ["foreground"], "gray", (0, 1)),
        ("any-organelle fraction", occ["any"], "viridis", (0, 1)),
        ("mito fraction", occ["mito"], "viridis", (0, 0.5)),
        ("vesicle fraction", occ["vesicle"], "viridis", (0, 0.05)),
    ]
    fig, axs = plt.subplots(len(panels), len(ys), figsize=(4.8 * len(ys), 1.9 * len(panels)), squeeze=False)
    for i, (title, g, cmap, lim) in enumerate(panels):
        if lim is None:  # robust limits over foreground cells, so structure inside the cell shows
            v = g[fgm]
            lim = np.percentile(v, [2, 98])
        for j, y in enumerate(ys):
            ax = axs[i, j]
            im = ax.imshow(g[:, y, :], cmap=cmap, vmin=lim[0], vmax=lim[1], interpolation="nearest", aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(title, fontsize=8)
            if i == 0:
                ax.set_title(f"y cell {y} (z down, x across)", fontsize=9)
        fig.colorbar(im, ax=axs[i, :].tolist(), shrink=0.9, pad=0.01)
    fig.suptitle("jrc_hela-2: chunk size vs label occupancy on the s0 128^3 chunk grid", fontsize=11)
    fig.savefig(out / "test1_heatmaps.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (cls, key) in zip(axs[:2], [("any-organelle", "any"), ("mito", "mito")]):
        ax.scatter(s0[valid & ~fgm] / 1e6, occ[key][valid & ~fgm], s=2, alpha=0.3, c="tab:gray", label="background cells")
        ax.scatter(s0[fgm] / 1e6, occ[key][fgm], s=2, alpha=0.3, c="tab:blue", label="foreground cells")
        ax.set_xlabel("zarr s0 chunk size (MB)"); ax.set_ylabel(f"{cls} fraction"); ax.legend(markerscale=5, fontsize=8)
    ax = axs[2]
    for name, g in [("zarr s0 size", s0), ("n5 uint16 size", n5), ("EM s3 std", occ["em_std_s3"])]:
        for tname, ls in [("mito (any voxel)", "-"), ("vesicle (any voxel)", "--")]:
            y = targets[tname][fgm]
            fpr, tpr, _ = roc_curve(y, g[fgm])
            ax.plot(fpr, tpr, ls, label=f"{name}: {tname.split()[0]} (AUC {roc_auc_score(y, g[fgm]):.2f})")
    ax.plot([0, 1], [0, 1], ":", c="k")
    ax.set_title("ROC within foreground cells"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(fontsize=7)
    fig.savefig(out / "test1_scatter_roc.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # size vs coarse intensity: the trend and its residual
    ls0 = np.log10(np.maximum(s0, 1))
    res, (cx, cy) = trend_residual(occ["em_mean_s3"], ls0, valid)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    sc = axs[0].scatter(occ["em_mean_s3"][valid], ls0[valid], c=occ["any"][valid], s=2, cmap="viridis", vmin=0, vmax=1)
    axs[0].plot(cx, cy, "r-", lw=2, label="binned median trend")
    axs[0].set_xlabel("coarse mean intensity (EM s3 averaged to the chunk grid)")
    axs[0].set_ylabel("log10 zarr s0 chunk size"); axs[0].legend(fontsize=8)
    fig.colorbar(sc, ax=axs[0], label="any-organelle fraction")
    lim = np.percentile(np.abs(res[fgm]), 98)
    for ax, y in zip(axs[1:], [6, 9]):
        im = ax.imshow(res[:, y, :], cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest")
        ax.set_title(f"residual (size beyond trend), y cell {y}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axs[1:].tolist(), shrink=0.9)
    fig.savefig(out / "test1_residual.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
