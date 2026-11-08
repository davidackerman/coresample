# coresample

Search a huge chunked array (volume EM, slides, satellite rasters, climate grids) for
"regions like this one" with **no precomputation**: no embedding pass, no index build,
nothing before the first query. The multiscale pyramid the data owner already wrote is
the index. Cheap model-free scores rank the coarsest level, only the best candidates
are read at finer levels, and an embedding model (DINOv2-small) re-ranks the survivors.
Every embedding is cached, so the index builds itself from use. Compressed chunk sizes,
read from storage metadata without touching pixels, give a free map of where the
content is.

**Try it: https://davidackerman.github.io/coresample/** (Chrome recommended: the model runs on
your GPU through WebGPU; without WebGPU it falls back to the CPU, about 10x slower).

### Using the page

Pick a dataset (the menu lists them by size, type or name, and takes your own data), drag to
pan, scroll to zoom, then click something: hits stream into the right panel, best first. The
search keeps going, best-ranked tiles first, until it has read the whole region or you press
**Pause** (**Resume** carries on); the first hits usually come within seconds and later reads
only add to them. The progress line is the share of the searched area (Options > region;
auto = the whole dataset whenever the quick first pass can cover it) read at full detail so far,
and the **Where it has looked** map shows it: the area from above and, for a volume, from the
side, with the tiles read, the hits, the query and your view (click it to go there). Click a
thumbnail to go there; hover one to mark it good or bad, which re-ranks the list. Each hit shows
how far it is from the query; the list can be ordered best first, nearest or farthest, and
Options > spacing leaves out hits within 2-25 boxes of the query (its own surroundings are the
easy matches) or spreads hits 2-10 boxes apart. The search runs at the zoom level you are looking at, so zoom until the
thing you care about is clear, then click. However long it reads, memory stays bounded: it
keeps the best 3,000 candidates (and any you marked) for re-ranking.

How it decides: your clicked patch is embedded by DINOv2-small (a general image model). A
cheap scan of a coarse level (intensity histogram and texture statistics, no model) ranks
where similar things might be; the full-resolution tiles under the best-ranked places are
read first, every patch in them is embedded, and hits are ranked by embedding similarity to
your click (the number under each thumbnail). Nothing is precomputed or uploaded anywhere.

### How matching works

The number under each hit is the similarity of its DINOv2 embedding to your click's (1 =
identical). The controls under **Options** choose how that is computed. Tests 5 and 7 measured
each on labelled EM: 30 queries per target, the same candidates and queries for every option,
differences paired per query (`results/RESULTS.md` has the tables):

| control | what it does | top-10 precision vs the default (90 queries) |
|---|---|---|
| (default) fast, whole patch | 112 px input, CLS + mean of tokens, one embedding per patch | 0.48 mito / 0.81 vesicles / 0.44 mito whole volume |
| **detailed** | 224 px input | +0.01 +/- 0.02 |
| **object at the centre** | CLS + centre-weighted tokens (where you clicked) | +0.00 +/- 0.01 |
| **any rotation** | also your box rotated/mirrored 8 ways | +0.00 +/- 0.02 |
| **any size** | also your box at half and double the field | not in the 30-query run |
| **adapt to data** | compare after removing what all candidates share | +0.00 +/- 0.01 |
| **method: tiles + tokens** | the Mars CTX explorer's way: each tile through the model once (crops twice the box), every window of the box's size, at 1/8 of it, matched by the mean of its tokens | -0.01 / +0.06 / -0.02 per target (each +/- 0.04): as good |
| good / bad | re-rank toward good hits, away from bad | +0.00 / +0.04 / +0.04 per target |

No option changes precision by more than a few hundredths, so the default stays and the rest
are there for data where they make sense: rotation for things with no fixed orientation
(craters, galaxies), size for objects that come in many sizes, the centre when surroundings
vary, tiles + tokens to place hits more finely (it embeds fewer but larger, 224 px crops; on
the CPU that was slower than patches here, on a GPU it is not measured yet). The model, not
the comparison, is the lever. The cheap
scan that decides which tiles to read never sees the model; the match options only re-rank what
has been read, instantly ("detailed" and the method re-embed, so they search again).

Ways to load data:

| how | example |
|---|---|
| a preset from the menu | Mars CTX (5 m/px), Moon LRO WAC, Hubble Andromeda, Webb Carina, Legacy Surveys around NGC 6129, Sentinel-2 cloudless, IDR organoid EM and lung histology, OpenOrganelle jrc_hela-2 in 3-D |
| a link to a preset | `?d=mars-ctx`, `moon-wac`, `andromeda`, `webb`, `legacy-ngc6129`, `s2`, `hela`, `idr-organoid`, `idr-lung` |
| your own OME-Zarr (v0.4/0.5, zarr v2/v3; 2-D or 3-D) | `https://...`, `s3://bucket/path`, `gs://bucket/path` under **Your own data** in the dataset menu |
| your own web-map tiles | `https://host/.../{z}/{x}/{y}.png` (add `#maxZoom=12` to set the finest zoom) |
| your own Zoomify image | `https://host/.../ImageProperties.xml` |
| a DESI Legacy Surveys field | `legacy:245.43,37.996` (RA, Dec in degrees) |
| a link to your data | `?url=<any of the above>`: opening data puts this in the address bar, so the link can be shared |

The page shows each dataset's size (pixels or voxels, and levels) when it opens. Volumes and
OME-Zarr open in Neuroglancer (the raw data streams natively; hits appear as annotation boxes),
tile pyramids in OpenLayers; the search runs behind the scenes in either. Neuroglancer opens
at the right place from the start (its first state carries the place, zoom and coordinate
space), and the model downloads only once the view has its first data, so the two do not
compete for the network. A dataset can carry more layers over the same space: the HeLa volume
comes with OpenOrganelle's organelle segmentations (nucleus, mito, ER, Golgi, vesicles,
endosomes, lysosomes, lipid droplets, microtubules, plasma membrane), toggled under **Layers**,
each class in one colour. They are for looking only so far; the search uses the raw data.

The data must be a multiscale pyramid (each level 2x coarser) on a host that allows
cross-origin reads (CORS), as public buckets, NASA GIBS/Trek, ESA and IDR do. Sources without
CORS (the Legacy Surveys' own map tiles, for one) cannot be read by a web page; the page uses
their cutout service instead. More formats (TIFF/SVS slides, HDF5, FITS, Sentinel-2 SAFE...)
are planned through refkv.

This repository currently holds the **feasibility tests** (Python, workstation scale)
and the `chunkmap.py` prototype. Results and the go/no-go verdict are in
[results/RESULTS.md](results/RESULTS.md).

## Run

```bash
uv sync                                        # numpy, scipy, tensorstore, torch (CPU), transformers
export CORESAMPLE_MIRROR=/local/disk/mirror    # chunk mirror: many small files, keep off NFS

uv run python explore_layout.py                # arrays, levels, chunking, compressors of jrc_hela-2
uv run python test1_chunksize.py               # chunk size vs label occupancy
uv run python test2_coarse2fine.py             # coarse-to-fine vs brute force vs sampling
uv run python test3_combined.py                # chunk-wise strategies: coarse prior, size map, feedback
uv run python report.py                        # plots + summary for RESULTS.md
```

In the browser (raw data only, DINOv2 on your GPU through WebGPU; Chrome recommended):

```bash
cd web && npm install && npm run model   # exports DINOv2-small to ONNX (88 MB, gitignored)
npm run build && npx vite preview        # http://localhost:5180
```

Presets go beyond EM: Mars CTX (5 m/px), Moon LRO WAC, Hubble Andromeda, Webb Carina, Legacy
Surveys around NGC 6129, Sentinel-2 cloudless, IDR images, and jrc_hela-2 in 3-D. Any OME-Zarr
URL works too, or `legacy:RA,Dec` for a Legacy Surveys field. `web/scripts/deploy_pages.sh`
publishes the build by hand; the site itself is built by `.github/workflows/pages.yml`
(Neuroglancer from source, the ONNX model, chunkmirage at the commit in `web/chunkmirage.lock`).

Search with raw data only (no labels anywhere), on any OME-Zarr or N5 pyramid:

```bash
uv run python query.py zarr://s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8 \
    --point 1700,400,2600 --working 2 --coarsest 4 --patch 64 --k 100 --top 20
```

It prints the ranked hits and writes a Neuroglancer link with them as annotations.

Chunk-size maps (no pixel reads; zarr v2/v3, sharded zarr, N5, Neuroglancer precomputed):

```bash
uv run chunkmap.py zarr://s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8 --max-chunks 5000000
uv run chunkmap.py precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg --max-chunks 5000000
```

## Layout

| file | what |
|---|---|
| `search.py` | portable search logic: box sums, FFT NCC, histogram EMD, texture stats, peaks, coarse-to-fine descent, coarse scan, best-first chunk search, size-based empty mask. numpy/scipy only, no I/O, no labels |
| `embed.py` | DINOv2-small patch embeddings with the sqlite cache (`cache/embeddings.sqlite`) |
| `common.py` | S3 listing, local chunk mirror, chunk-touch accounting |
| `chunkmap.py` | chunk-size maps from listings, shard indexes, or HEAD requests |
| `query.py` | raw-data-only search CLI |
| `web/` | the browser tool: TypeScript scan (worker), chunk-wise search, readers for OME-Zarr, tile pyramids, Zoomify and the Legacy Surveys cutout service; reuses chunkmirage's ONNX Runtime Web wrapper |
| `explore_layout.py`, `test1_chunksize.py`, `test2_coarse2fine.py`, `test3_combined.py`, `report.py` | the feasibility tests |

Data: [OpenOrganelle](https://openorganelle.janelia.org) `jrc_hela-2` from the public
bucket `s3://janelia-cosem-datasets`, read lazily with range requests.
