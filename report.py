"""Plots and tables for RESULTS.md from results/test1.json and results/test2_*.json.

    uv run python report.py     # writes results/test2_*.png and results/summary.json
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from common import RESULTS

MATCH_REL = 0.9  # "roughly matching brute force": >= 90% of its mean precision...
MATCH_ABS = 0.05  # ...or within 0.05 absolute (small-number noise)


def mean_over_queries(rows, key):
    return float(np.mean([r[key] for r in rows]))


def load(target, region="sub"):
    """Merge test2_<target>*.json runs over the same region (c2f rows from all of them;
    brute force and sampling from the first). region 'sub' = subvolume, 'full' = whole volume."""
    files = sorted(RESULTS.glob(f"test2_{target}*.json"))
    runs = [json.loads(f.read_text()) for f in files]
    full = lambda r: r["region_s0"][0] == [0, 0, 0] and r["region_s0"][1] == [6368, 1600, 12000]
    runs = [r for r in runs if full(r) == (region == "full")]
    if not runs:
        return None
    base = max(runs, key=lambda r: len(r["brute"]))
    seen = {}
    for r in runs:
        for row in r["c2f"]:
            seen[(row["query"], row["coarsest"], row["k"], row["methods"])] = row
    base = dict(base)
    base["c2f"] = list(seen.values())
    return base


def c2f_table(r):
    """Group c2f rows by config, averaging over queries."""
    g = defaultdict(list)
    for row in r["c2f"]:
        g[(row["coarsest"], row["k"], row["methods"])].append(row)
    out = []
    for (c, k, m), rows in sorted(g.items()):
        e = {"coarsest": c, "k": k, "methods": m, "queries": len(rows)}
        for n in (10, 25, 50, 100):
            e[f"p@{n}"] = float(np.mean([x["final"][f"p@{n}"] for x in rows]))
            e[f"r@{n}"] = float(np.mean([x["final"][f"r@{n}"] for x in rows]))
        for key in ("frac_bytes", "frac_bytes_inner32", "frac_voxels", "frac_chunks", "seconds_cheap", "seconds_embed"):
            e[key] = float(np.mean([x[key] for x in rows]))
        e["bytes_per_query_MB"] = float(np.mean([x["cost"]["bytes"] for x in rows])) / 1e6
        # cheap-only precision of the first (coarsest) stage, the "rough results in seconds"
        e["coarse_p@10"] = float(np.mean([x["stages"][0]["p@10"] for x in rows]))
        e["coarse_seconds"] = float(np.mean([x["stages"][0]["seconds"] for x in rows]))
        e["last_cheap_p@10"] = float(np.mean([x["stages"][-1]["p@10"] for x in rows]))
        out.append(e)
    return out


def brute_means(r):
    q = r["brute"].values()
    return {k: float(np.mean([x[k] for x in q])) for k in next(iter(q)) if k[:2] in ("p@", "r@")}


def matches(e, b):
    return all(e[f"p@{n}"] >= min(MATCH_REL * b[f"p@{n}"], b[f"p@{n}"] - MATCH_ABS) for n in (10, 25))


def sampling_means(r, kind):
    curves = [v[kind] for v in r["sampling"].values()]
    n = min(len(c) for c in curves)
    return [{k: float(np.mean([c[i][k] for c in curves])) for k in curves[0][i] if not k.startswith("n@")}
            for i in range(n)]


def plot_target(name, r, table, b, tag=""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, metric in zip(axs, ("p@10", "p@25", "r@100")):
        for kind, style in (("random", "o:"), ("adaptive", "s--")):
            s = sampling_means(r, kind)
            ax.plot([x["frac_chunks"] for x in s], [x[metric] for x in s], style, c="tab:gray" if kind == "random" else "tab:olive",
                    label=f"{kind} chunk sampling (chunks read)")
        for mset, mk in (("all", "D-"), ("hist+tex", "^-")):
            ks = sorted([e for e in table if e["methods"] == mset and e["coarsest"] == TARGET_DEFAULT[name]],
                        key=lambda e: e["k"])
            if len(ks) < 2:
                continue
            for key, c, lab in (("frac_bytes", "tab:red", "bytes as stored"),
                                ("frac_bytes_inner32", "tab:orange", "bytes if 32^3 inner chunks"),
                                ("frac_voxels", "tab:blue", "voxels needed")):
                ax.plot([e[key] for e in ks], [e[metric] for e in ks], mk, c=c, label=f"c2f {mset}, {lab}")
                for e in ks:
                    if key == "frac_bytes":
                        ax.annotate(f"K={e['k']}", (e[key], e[metric]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.axhline(b[metric], c="k", lw=1, label=f"brute force (100%)")
        ax.axhline(r["chance_precision"] if metric.startswith("p") else 0, c="k", ls=":", lw=1, label="chance" if metric.startswith("p") else None)
        ax.axvspan(0.10, 0.20, color="tab:red", alpha=0.07, label="kill threshold 10-20%")
        ax.set_xscale("log")
        ax.set_xlabel("fraction of working-level data read")
        ax.set_ylabel(metric)
        ax.set_title(f"{name}: {metric} vs data read")
    axs[0].legend(fontsize=7, loc="lower right")
    fig.savefig(RESULTS / f"test2_{name}{tag}_recall_vs_data.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


TARGET_DEFAULT = {"mito": 4, "vesicle": 3}


def main():
    summary = {}
    t1 = RESULTS / "test1.json"
    if t1.exists():
        summary["test1"] = json.loads(t1.read_text())
    for name in ("mito", "vesicle"):
        for region in ("sub", "full"):
            r = load(name, region)
            if not r or not r["c2f"]:
                continue
            tag = "" if region == "sub" else "_full"
            key = name + tag
            table = c2f_table(r)
            b = brute_means(r)
            ok = [e for e in table if matches(e, b)]
            cheapest = {k: (min(ok, key=lambda e: e[k]) if ok else None)
                        for k in ("frac_bytes", "frac_bytes_inner32", "frac_voxels")}
            summary[key] = {"brute": b, "chance": r["chance_precision"], "n_instances": r["n_instances"],
                            "queries": len(r["brute"]), "table": table, "region_s0": r["region_s0"],
                            "cheapest_match": {k: (v and {kk: v[kk] for kk in ("coarsest", "k", "methods", k, "p@10", "p@25", "r@100")})
                                               for k, v in cheapest.items()},
                            "sampling_random": sampling_means(r, "random"),
                            "sampling_adaptive": sampling_means(r, "adaptive"),
                            "brute_patches": r["brute_patches"], "brute_cost": r["brute_cost"]}
            plot_target(name, r, table, b, tag)
            print(f"\n== {key}: {r['n_instances']} instances, chance p={r['chance_precision']:.2f}, "
                  f"brute p@10={b['p@10']:.2f} p@25={b['p@25']:.2f} r@100={b['r@100']:.2f}")
            print("coarsest K methods | p@10 p@25 r@100 | bytes inner32 voxels | coarse p@10 (s) | cheap s embed s | match")
            for e in table:
                print(f"s{e['coarsest']} {e['k']:>3} {e['methods']:<8} | {e['p@10']:.2f} {e['p@25']:.2f} {e['r@100']:.2f} | "
                      f"{e['frac_bytes']:.1%} {e['frac_bytes_inner32']:.1%} {e['frac_voxels']:.2%} | "
                      f"{e['coarse_p@10']:.2f} ({e['coarse_seconds']:.1f}) | {e['seconds_cheap']:.1f} {e['seconds_embed']:.1f} | "
                      f"{'yes' if matches(e, b) else ''}")
            print("sampling adaptive:", " ".join(f"{x['frac_chunks']:.2f}:p10={x['p@10']:.2f}/p25={x['p@25']:.2f}/r100={x['r@100']:.2f}" for x in summary[key]["sampling_adaptive"]))
            print("sampling random:  ", " ".join(f"{x['frac_chunks']:.2f}:p10={x['p@10']:.2f}/p25={x['p@25']:.2f}/r100={x['r@100']:.2f}" for x in summary[key]["sampling_random"]))
            print("cheapest match:", json.dumps(summary[key]["cheapest_match"], default=float))
    test3(summary)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1, default=float))


STRAT_STYLE = {"random": ("o:", "tab:gray"), "adaptive": ("s--", "tab:olive"), "adaptive+skip": ("s-", "tab:green"),
               "coarse": ("v--", "tab:purple"), "combined": ("*-", "tab:red")}


def test3(summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = {}
    for f in sorted(RESULTS.glob("test3_*.json")):
        r = json.loads(f.read_text())
        key = f.stem.replace("test3_", "")
        t2 = summary.get(key)
        mean = {}
        for sname, curves in r["curves"].items():
            n = len(curves[0])
            mean[sname] = [{k: float(np.mean([c[i][k] for c in curves])) for k in curves[0][i]} for i in range(n)]
        out[key] = {"empty_fraction": r["empty_fraction"], "coarse_level": r["coarse_level"],
                    "coarse_overhead": r["coarse_overhead"], "n_chunks": r["n_chunks"], "curves": mean,
                    "coarse_seconds": float(np.mean([q["coarse_seconds"] for q in r["queries"]]))}
        fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
        for ax, metric in zip(axs, ("p@10", "p@25", "r@100")):
            for sname, curve in mean.items():
                st, col = STRAT_STYLE[sname]
                ax.plot([x["frac_bytes"] for x in curve], [x[metric] for x in curve], st, c=col, label=sname, ms=7 if sname == "combined" else 5)
            if t2:
                ax.axhline(t2["brute"][metric], c="k", lw=1, label="brute force (100%)")
                ks = sorted([e for e in t2["table"] if e["methods"] == "hist+tex" and e["coarsest"] == TARGET_DEFAULT[r["target"]]],
                            key=lambda e: e["k"])
                ax.plot([e["frac_bytes"] for e in ks], [e[metric] for e in ks], "D", c="tab:blue", mfc="none",
                        label="coarse-to-fine hist+tex, K sweep")
            ax.axvspan(0.10, 0.20, color="tab:red", alpha=0.07)
            ax.set_xscale("log"); ax.set_xlabel("fraction of working-level bytes read (incl. coarse scan)")
            ax.set_ylabel(metric); ax.set_title(f"{key}: {metric}")
        axs[0].legend(fontsize=7, loc="lower right")
        fig.savefig(RESULTS / f"test3_{key}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"\n== test3 {key}: {r['n_chunks']} chunks, {r['empty_fraction']:.0%} empty by size, "
              f"coarse s{r['coarse_level']} overhead {r['coarse_overhead']:.2%}, scan {out[key]['coarse_seconds']:.1f}s")
        fr = [x["frac_chunks"] for x in next(iter(mean.values()))]
        for target_f in (0.01, 0.02, 0.05, 0.1, 0.2):
            i = fr.index(target_f)
            print(f"  {target_f:>4.0%} of chunks: " + "  ".join(
                f"{sn}={c[i]['p@10']:.2f}/{c[i]['p@25']:.2f}/{c[i]['r@100']:.2f} ({c[i]['frac_bytes']:.1%})" for sn, c in mean.items()))
    summary["test3"] = out


if __name__ == "__main__":
    main()
