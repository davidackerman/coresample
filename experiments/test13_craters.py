"""Test 13: the page's models on Mars craters, graded with the Robbins & Hynek (2012) catalogue.

Test 6 chose the models on one labelled EM volume; this grades them on planetary imagery. A
region of the CTX global mosaic (Murray Lab, 5.05 m/px, read as the page reads it: Esri's tile
cache) at one level; craters from the Robbins catalogue (complete for diameters >= 1 km;
USGS Astrogeology's feature service), by latitude, east longitude and diameter.

As test 6: the candidates are every P-px box at a stride of P/2 over the region; each model
embeds all of them once; a query is a box centred on a catalogued crater (what a click on it
picks); hits are ranked by cosine with non-overlapping boxes kept (NMS), the query's own box
left out. A crater "like it" is one filling 30-70% of the box (its diameter; the queries are
drawn from the same band), so a hit is correct when its box holds such a crater's centre
(test 6's rule); strict: the centre within a quarter box of the hit's centre. Chance: the
share of candidates that hold one.

Targets: two terrains (old, densely cratered highlands, Terra Sabaea at 25 E 5 S; younger
ridged plains, Lunae Planum at 63 W 8 N) at two levels (4: 64 px = 5.2 km box, craters
1.6-3.6 km, a region 331 km across; 6: 20.7 km box, 6-14 km, 1,320 km across).

    uv run python experiments/test13_craters.py [--targets ...] [--models ...] [--adapt] [--check]
"""

from __future__ import annotations

import argparse
import io
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import test2_coarse2fine as t2
from common import CACHE, RESULTS, _get
from test6_models import Matcher, unit

TILE = "https://astro.arcgis.com/arcgis/rest/services/OnMars/CTX/MapServer/tile/{z}/{y}/{x}"
CRATERS = "https://astrogeology.usgs.gov/pygeoapi/collections/mars/robbinsv1/items"
W0, H0, PX0, MAXZ = 4194304, 2097152, 5.05, 12  # full resolution: pixels, metres per pixel, zoom
P, SIZE = 64, 4096  # box (px at the level) and region side (px)
BAND = (0.3, 0.7)  # crater diameter / box side
# name: (centre east longitude, centre latitude, level)
TARGETS = {"sabaea4": (25.0, -5.0, 4), "lunae4": (-63.0, 8.0, 4),
           "sabaea6": (25.0, -5.0, 6), "lunae6": (-63.0, 8.0, 6)}
MODELS = ["dinov2-small", "pecore-t16-tok", "morphem-tok", "texture"]


def region(name):
    """The region's top-left pixel at its level, the level, and its lon/lat bounding box."""
    lon, lat, L = TARGETS[name]
    s = 2 ** L
    x0 = int(round((lon + 180) / 360 * W0 / s)) - SIZE // 2
    y0 = int(round((90 - lat) / 180 * H0 / s)) - SIZE // 2
    lon0, lon1 = x0 * s / W0 * 360 - 180, (x0 + SIZE) * s / W0 * 360 - 180
    lat1, lat0 = 90 - y0 * s / H0 * 180, 90 - (y0 + SIZE) * s / H0 * 180
    return x0, y0, L, (lon0, lat0, lon1, lat1)


def read_region(name):
    """The region's pixels (uint8, SIZE x SIZE) from the 512-px tiles at its zoom, cached."""
    x0, y0, L, _ = region(name)
    z = MAXZ - L
    f = CACHE / f"test13_{name}.npy"
    if f.exists():
        return np.load(f)
    from PIL import Image

    tx = range(x0 // 512, (x0 + SIZE - 1) // 512 + 1)
    ty = range(y0 // 512, (y0 + SIZE - 1) // 512 + 1)
    big = np.zeros((len(ty) * 512, len(tx) * 512), np.uint8)

    def tile(z, y, x):
        """One 512-px tile; a truncated one (Esri's cache has some) from the four finer ones."""
        for _ in range(2):
            try:
                return np.asarray(Image.open(io.BytesIO(_get(TILE.format(z=z, y=y, x=x)))).convert("L"))
            except OSError:
                pass
        if z >= MAXZ:
            raise RuntimeError(f"tile {z}/{y}/{x} unreadable")
        fine = np.block([[tile(z + 1, 2 * y + a, 2 * x + b) for b in (0, 1)] for a in (0, 1)]).astype(np.float32)
        return fine.reshape(512, 2, 512, 2).mean((1, 3)).round().astype(np.uint8)

    def one(ij):
        j, i = ij
        big[j * 512:(j + 1) * 512, i * 512:(i + 1) * 512] = tile(z, ty[j], tx[i])

    with ThreadPoolExecutor(6) as ex:
        list(ex.map(one, [(j, i) for j in range(len(ty)) for i in range(len(tx))]))
    ox, oy = x0 - tx[0] * 512, y0 - ty[0] * 512
    out = big[oy:oy + SIZE, ox:ox + SIZE].copy()
    np.save(f, out)
    return out


def craters(name):
    """Catalogued craters in the region: (row, col) at its level and diameter in pixels."""
    x0, y0, L, (lon0, lat0, lon1, lat1) = region(name)
    f = CACHE / f"test13_{name}_craters.json"
    if not f.exists():
        feats, off = [], 0
        while True:
            q = urllib.parse.urlencode({"f": "json", "limit": 10000, "offset": off, "bbox": f"{lon0},{lat0},{lon1},{lat1}"})
            # the service refuses urllib's default client name
            req = urllib.request.Request(f"{CRATERS}?{q}", headers={"User-Agent": "coresample-test13"})
            with urllib.request.urlopen(req, timeout=120) as r:
                page = json.loads(r.read())["features"]
            feats += [[p["properties"]["lat"], p["properties"]["lon_e"], p["properties"]["diamkm"]] for p in page]
            if len(page) < 10000:
                break
            off += 10000
        f.write_text(json.dumps(feats))
    c = np.array(json.loads(f.read_text()), float).reshape(-1, 3)
    s = 2 ** L
    lon = (c[:, 1] + 180) % 360 - 180
    rows = (90 - c[:, 0]) / 180 * H0 / s - y0
    cols = (lon + 180) / 360 * W0 / s - x0
    diam = c[:, 2] * 1000 / (PX0 * s)  # along the meridian (the equirectangular map stretches it east-west by 1/cos(lat))
    ok = (rows >= 0) & (rows < SIZE) & (cols >= 0) & (cols < SIZE)
    return np.stack([rows, cols], 1)[ok], diam[ok]


def check(name, img, cents, diam):
    """A picture of the region with the band's craters circled (to see the catalogue lines up)."""
    from PIL import Image, ImageDraw

    band = (diam >= BAND[0] * P) & (diam <= BAND[1] * P)
    im = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(im)
    for (r, c), dd, b in zip(cents, diam, band):
        d.ellipse([c - dd / 2, r - dd / 2, c + dd / 2, r + dd / 2], outline=(255, 210, 74) if b else (90, 176, 255), width=2)
    out = RESULTS / f"test13_{name}_check.jpg"
    im.resize((2048, 2048)).save(out, quality=85)
    sub = im.crop((SIZE // 2 - 512, SIZE // 2 - 512, SIZE // 2 + 512, SIZE // 2 + 512))
    sub.save(RESULTS / f"test13_{name}_check_centre.jpg", quality=85)
    return out


def run(name, args, log):
    img = read_region(name)
    cents, diam = craters(name)
    band = (diam >= BAND[0] * P) & (diam <= BAND[1] * P)
    pos = cents[band]
    if args.check:
        log(f"[{name}] {len(cents)} craters, {band.sum()} in the band: {check(name, img, cents, diam)}")
        return
    rng = np.random.default_rng(args.seed)
    inside = np.all((pos >= P) & (pos < SIZE - P), 1)
    queries = np.round(pos[rng.permutation(np.nonzero(inside)[0])[:args.queries]]).astype(int)
    h = P // 2
    ax = np.arange(h, SIZE - h + 1, h)  # test 2's grid: box centres at a stride of half a box
    gc = np.stack(np.meshgrid(ax, ax, indexing="ij"), -1).reshape(-1, 2)  # (row, col)
    gc3 = np.concatenate([np.zeros((len(gc), 1), int), gc], 1)  # test 2's helpers take (z, y, x)
    pos3 = np.concatenate([np.zeros((len(pos), 1)), pos], 1)

    def boxes(c):
        return np.stack([img[r - h:r + h, q - h:q + h] for r, q in c])

    holds = np.all(np.abs(gc3[:, None, 1:] - pos3[None, :, 1:]) < P / 2, 2).any(1)
    log(f"[{name}] level {region(name)[2]}, {len(cents)} craters, {len(pos)} in the band "
        f"({BAND[0] * P * PX0 * 2 ** region(name)[2] / 1000:.1f}-{BAND[1] * P * PX0 * 2 ** region(name)[2] / 1000:.1f} km), "
        f"{len(gc)} candidates ({holds.mean():.1%} hold one: chance), {len(queries)} queries")
    res = {"target": name, "region": region(name)[:3], "n_craters": int(len(cents)), "n_band": int(len(pos)),
           "n_candidates": int(len(gc)), "chance": float(holds.mean()), "queries": queries.tolist(), "rows": []}
    Qp = boxes(queries)
    for mname in args.models:
        t = time.time()
        m = Matcher(mname)
        f = CACHE / f"test13_{name}_{mname}.npy"
        if f.exists():
            E = np.load(f).astype(np.float32)
        else:
            E = np.concatenate([m(boxes(gc[i:i + 2048])) for i in range(0, len(gc), 2048)])
            np.save(f, E.astype(np.float16))
            E = E.astype(np.float16).astype(np.float32)
        Q = m(Qp)
        if mname == "texture":
            mu, sd = E.mean(0), E.std(0) + 1e-6
            E, Q = unit((E - mu) / sd), unit((Q - mu) / sd)
        elif args.adapt:  # as test 6 --adapt: each [CLS | mean] half less the candidates' mean
            w, mu = E.shape[1] // 2, E.mean(0)

            def adapt(X):
                return np.concatenate([unit(X[:, :w] - mu[:w]), unit(X[:, w:] - mu[w:])], 1) / np.sqrt(2)
            E, Q = adapt(E), adapt(Q)
        gl = []
        for qc, q in zip(queries, Q):
            notself = ~np.all(np.abs(gc - qc) < P / 2, 1)
            top = t2.rank_nms(gc3[notself], (E @ q)[notself], P)
            e = t2.evaluate(top, pos3, P)
            strict = t2.evaluate(top, pos3, P // 2)  # the centre within a quarter box
            gl.append({"p@10": e["p@10"], "p@25": e["p@25"], "strict p@10": strict["p@10"], "strict p@25": strict["p@25"]})
        row = {"model": mname, "seconds": time.time() - t}
        for k in gl[0]:
            v = np.array([g[k] for g in gl])
            row[k], row[k + "_se"], row[k + "_per_query"] = float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))), v.tolist()
        res["rows"].append(row)
        log(f"[{name}] {mname:<15} p@10 {row['p@10']:.2f}+/-{row['p@10_se']:.2f}  p@25 {row['p@25']:.2f}+/-{row['p@25_se']:.2f}"
            f"  strict p@10 {row['strict p@10']:.2f}  ({row['seconds']:.0f}s)")
        (RESULTS / f"test13_{name}{args.tag}.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=list(TARGETS))
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--queries", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--tag", default="")
    ap.add_argument("--adapt", action="store_true", help="compare as the page's adapt to data does")
    ap.add_argument("--check", action="store_true", help="only draw the band's craters over each region")
    args = ap.parse_args()
    logf = open(RESULTS / f"test13{args.tag}.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    for t in args.targets:
        run(t, args, log)


if __name__ == "__main__":
    main()
