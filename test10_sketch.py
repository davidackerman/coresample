"""Test 10: can a drawing be the query?

The page could let you draw what you are looking for instead of clicking an example. The
model has never seen drawings, so the question is whether a sketch lands near the real thing
in its embedding space. Stand-in for a person drawing: each of the 30 queries' own ground
truth, rendered as a sketch: the outline of the object clicked ("one") or of every labelled
object in the box ("all"), on the data's median grey, in the data's own tones (the outline in
the tone the data shows at object boundaries, the inside in the tone it shows inside them),
plain or with the data's blur and noise ("tex"). Controls: the real patch (the click), a
blank sketch (grey and noise only: what the tones alone find), and a random order.
Brute-force embeddings as tests 5-9 (CLS + mean, 112 px). Reported: p@10 / p@25.

    uv run python test10_sketch.py [--targets mito vesicle]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy import ndimage as ndi

import test2_coarse2fine as t2
import test6_models as t6
from common import CACHE, RESULTS, Mirror

REGION = (np.array([640, 0, 512]), np.array([2688, 1600, 4608]))


def unit(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)


def run(target, m, data, log):
    cfg = t2.TARGETS[target]
    W, P, h = cfg["working"], cfg["patch"], cfg["patch"] // 2
    data.working = W
    t2.REGION_S0 = REGION
    rlo, rhi = REGION[0] // 2 ** W, REGION[1] // 2 ** W
    base = json.loads((RESULTS / f"test5_{target}_n30.json").read_text())
    queries = [np.asarray(q, int) for q in base["queries"]]
    cents = t2.instances(data.mirror, cfg["label"], W, rlo, rhi, cfg["min_voxels"])["centroids"]
    gc = t2.grid_centers(rlo, rhi, P)
    z = np.load(CACHE / f"test5_{target}_112.npz")
    E = np.concatenate([unit(z["cls"].astype(np.float32)), unit(z["mean"].astype(np.float32))], 1)

    imgs = np.stack([data.read(W, q - [0, h, h], q + [1, h, h])[0] for q in queries]).astype(np.float32)
    labs = np.stack([data.mirror.read(f"{cfg['label']}/s{W}", q - [0, h, h], q + [1, h, h])[0] for q in queries]).astype(np.int64)
    one = np.stack([lab == lab[h, h] for lab in labs])
    every = labs > 0
    edge = lambda mk: mk & ~ndi.binary_erosion(mk, iterations=2)  # noqa: E731  (a 2 px outline)
    # the data's tones: its median, and what it shows at object outlines and inside objects
    bg = float(np.median(imgs))
    tone_edge = float(np.mean([im[edge(mk)].mean() for im, mk in zip(imgs, one)]))
    tone_in = float(np.mean([im[mk & ~edge(mk)].mean() if (mk & ~edge(mk)).any() else im[mk].mean() for im, mk in zip(imgs, one)]))
    noise = float(np.mean([(im - ndi.gaussian_filter(im, 1)).std() for im in imgs]))
    log(f"[{target}] tones: background {bg:.0f}, outlines {tone_edge:.0f}, inside {tone_in:.0f}; noise {noise:.1f}")
    rng = np.random.default_rng(0)

    def sketch(mk, filled, tex):
        s = np.full((P, P), bg, np.float32)
        if filled:
            s[mk] = tone_in
        s[edge(mk)] = tone_edge
        if tex:
            s = ndi.gaussian_filter(s, 1) + rng.normal(0, noise, s.shape)
        return np.clip(s, 0, 255)

    variants = {"real patch": imgs}
    for mname, masks in (("one", one), ("all", every)):
        for filled in (False, True):
            for tex in (False, True):
                variants[f"{'filled' if filled else 'outline'}{'+tex' if tex else ''} ({mname})"] = \
                    np.stack([sketch(mk, filled, tex) for mk in masks])
    variants["blank (tones only)"] = np.stack([sketch(np.zeros((P, P), bool), False, True) for _ in queries])
    np.save(CACHE / f"test10_{target}_sketches.npy", np.stack([variants[k][:4] for k in variants]))  # a look at a few
    res = {}
    real = None
    for name, pats in list(variants.items()) + [("random order", None)]:
        p10, p25 = [], []
        Q = m(np.round(pats).astype(np.uint8)) if pats is not None else None
        for i, q in enumerate(queries):
            mask = ~np.all(np.abs(gc - q) < h, 1)
            s = E @ Q[i] if Q is not None else np.random.default_rng(i).random(len(gc))
            e = t2.evaluate(t2.rank_nms(gc[mask], s[mask], P), cents, P)
            p10.append(e["p@10"]); p25.append(e["p@25"])
        p10, p25 = np.array(p10), np.array(p25)
        if real is None:
            real = (p10, p25)
        d = p10 - real[0]
        res[name] = {"p@10": p10.mean(), "p@25": p25.mean(), "d10": d.mean(), "se10": d.std(ddof=1) / np.sqrt(len(d))}
        log(f"   {name:<22} p@10 {p10.mean():.2f}  p@25 {p25.mean():.2f}   vs the real patch {d.mean():+.2f}+/-{res[name]['se10']:.2f}")
    (RESULTS / f"test10_{target}.json").write_text(json.dumps({"target": target, "tones": [bg, tone_edge, tone_in, noise], "rows": res}, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=["mito", "vesicle"])
    args = ap.parse_args()
    logf = open(RESULTS / "test10.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    os.environ.setdefault("TORCH_THREADS", "4")
    m, data = t6.Matcher("dinov2-small"), t2.Data(Mirror())
    for t in args.targets:
        run(t, m, data, log)


if __name__ == "__main__":
    main()
