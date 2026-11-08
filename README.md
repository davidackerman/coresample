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
search keeps going until it has read everything or you press **Pause** (**Resume** carries on);
the first hits usually come within seconds and later reads only add to them. With Options >
region on auto it searches everywhere: first the largest area around your click that the quick
first pass can cover (the whole dataset when it fits), best-ranked tiles first, then the areas
around it, ring by ring, nearest first (at full resolution on Earth or Mars one area is about
100 km across, and the whole planet takes far longer than anyone will wait). The other region
choices stop at their area. The progress line is the share of the dataset (or the chosen area)
read at full detail so far,
and the **Where it has looked** map shows it: the area from above and, for a volume, from the
side, with the tiles read, the hits, the query and your view (click it to go there). Click a
thumbnail to go there. Hover one to mark it: **✓ more like this** moves the query toward it (the
hit turns green; click ✓ again to undo), **✗ fewer like this** hides it and moves the query a
little away from it; either way every kept candidate is scored again and the tiles not read
yet are re-ordered, so the reads follow your marks too. A line under **Hits** counts the marks,
with **undo last** and **clear marks**. Each hit shows
how far it is from the query; the list can be ordered best first, nearest or farthest, and
Options > spacing leaves out hits within 2-25 boxes of the query (its own surroundings are the
easy matches) or spreads hits 2-10 boxes apart. The bottom right shows the coordinates under
the cursor (full-resolution pixels; latitude and longitude on Mars, the Moon and Earth;
physical position on volumes), and takes coordinates to go to (x, y[, z], or lat, lon). The search runs at the zoom level you are looking at, so zoom until the
thing you care about is clear, then click. However long it reads, memory stays bounded: it
keeps the best 3,000 candidates (and any you marked) for re-ranking.

How it decides: your clicked patch is embedded by DINOv2-small (a general image model; it
sees colour on colour data). A cheap scan of a coarse level (brightness histogram and texture
statistics, no model, no colour) ranks
where similar things might be; the full-resolution tiles under the best-ranked places are
read first, every patch in them is embedded, and hits are ranked by embedding similarity to
your click (the number under each thumbnail). Nothing is precomputed or uploaded anywhere.

**More examples.** Shift-click (or press **+ example**, then click) to add another example of the
same thing: the query becomes the mean of the examples' embeddings (each part unit length
again), as Earth Engine's similarity search and similar.earth combine several points, and the
list re-ranks at once. The thumbnail in the status card shows every example; no hit sits on
one, and distances count from the nearest. With a GPU the rough pass's query moves too; the
cheap scan that ranked the tiles at the start stays the first example's. On labelled EM (test
9) eight averaged examples raised top-10 precision on vesicles from 0.81 to 0.91; on mito,
whose instances look more varied, averaging did not help and scoring each hit by its best match
among the examples did a little better (+0.04 with two), so Options > examples offers both.

**Drawing.** **draw** opens a pad: draw what to find as one box at the search level, in the
data's own greys or colours (dark, middle and light pens from the view; start blank or from the box at the
view centre; "look like the data" softens it and adds the data's grain), then search for it.
The model has never seen drawings: bold shapes such as craters find crater-like boxes, but on
EM a drawing finds nothing (test 10: top-10 precision 0.00-0.13, about what a random order
finds), so there, add real examples instead.

**Heat.** The map of where it has looked also shows how well each place matched: every place
compared, in one magenta ramp from faint (the lowest score shown) to solid (the best), for the
best quarter of the scores by default (the slider sets the share). On 2-D maps the heat lies
over the image too (a checkbox and an opacity slider). Each place keeps the score it had when
it was compared, except the best 3,000, which are re-scored whenever feedback or a new example
changes the query.

With a GPU, the model also takes a rough look first: two levels coarser than the search level
(the same physical box, a quarter of the pixels each way), a few hundred places at a time in
the cheap scan's order, and the full-detail tiles are read best rough match first; a new round
starts when only below-median tiles are left. It only reorders the reads, never skips any. On
labelled EM (test 8, mito) it found the right tiles sooner than the cheap statistics at the
same download: p@10 0.42 vs 0.33 after 1% of the tiles, 0.53 vs 0.41 after 5% (reading all of
them gives 0.48). Each round takes the coarse chunks under the next tiles and scores every
tile under each chunk from it, since a chunk downloads whole anyway. Marking a hit good or bad also moves this rough query, so the tiles read next
follow your marks, not only the list. Add `&rough=0` to the address to turn it off (`&rough=1`
forces it on a CPU, where it is slow).

### How matching works

The number under each hit is the similarity of its DINOv2 embedding to your click's (1 =
identical): the box is resized to 112 px and DINOv2-small turns it into a summary token (CLS)
and 8 x 8 tokens for its pieces; the score is the mean of two cosines, **look** (CLS: the box as
a whole) and **parts** (the average of the piece tokens: what is in it). **why**, above the
hits, shows this per hit (the two parts, the rotation, size or example that matched best, what
your good/bad marks added) and colours each thumbnail by how the model sees its pieces: the
three main directions of the 224 px model's 16 x 16 tokens over the query and the hits, as red,
green and blue, so pieces it sees alike share a colour and a hit's colours show which of its
parts match which parts of the query (coloured the same way). "where alike" instead shows, in
magenta, the pieces most like the query's average piece. The header names the model; it is DINOv2-small in
every mode: test 6 compared it on the EM with five other models (an EM-trained one among them) and a
hand-made texture descriptor, and none did better on both targets. The cheap statistics and the rough pass only decide
what is read first; this score alone ranks the hits. The controls under **Options** choose how that is computed. Tests 5 and 7 measured
each on labelled EM: 30 queries per target, the same candidates and queries for every option,
differences paired per query ([experiments/results/RESULTS.md](experiments/results/RESULTS.md) has the tables):

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
| **examples** (shift-click adds one) | their mean, or each hit's best match among them | mean of 8: -0.03 / +0.11 / +0.06; best of 2: +0.04 / +0.04 / +0.02 (each +/- 0.04, test 9) |

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
| your own Neuroglancer precomputed volume (sharded or not; raw, JPEG or segmentation) | `precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg` (the FlyEM hemibrain, 56 trillion voxels) |
| your own web-map tiles | `https://host/.../{z}/{x}/{y}.png` (add `#maxZoom=12` to set the finest zoom) |
| your own Zoomify image | `https://host/.../ImageProperties.xml` |
| a DESI Legacy Surveys field | `legacy:245.43,37.996` (RA, Dec in degrees) |
| a link to your data | `?url=<any of the above>`: opening data puts this in the address bar, so the link can be shared |
| a link to a place | `&v=x,y,z,scale` (full-resolution pixels and zoom): the address bar keeps the current view, so a copied link opens the same place |

The page shows each dataset's size (pixels or voxels, and levels) when it opens. Volumes and
OME-Zarr open in Neuroglancer (the raw data streams natively; what you clicked appears as blue
annotation boxes and the hits as yellow ones; its own controls, the layer bar, position box and
layer panel, work as usual, and only clicks on the data search),
tile pyramids in OpenLayers, and the planetary mosaics (Mars, Moon) on a CesiumJS globe (the
**map** button switches to the flat map and writes `&globe=0` into the address); the search runs
behind the scenes in each. The globe and the map load tiles the same way, so a tile a server
sends broken (Esri's Mars cache has cut-off images and error pages at several zooms) is rebuilt
from its children or its parent in both; the black strips that remain are gaps in the CTX
mosaic itself. While a search reads, the boxes on the view update at most once a second. Neuroglancer opens
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

The browser tool is in `web/`. The ten feasibility tests behind its design (Python, workstation
scale, graded on labelled EM) are in [experiments/](experiments/), with their results in
[experiments/results/RESULTS.md](experiments/results/RESULTS.md).

## Run

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

The same search from the command line (Python, raw data only, any OME-Zarr or N5 pyramid):

```bash
uv sync                                        # numpy, scipy, tensorstore, torch (CPU), transformers
uv run python experiments/query.py zarr://s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8 \
    --point 1700,400,2600 --working 2 --coarsest 4 --patch 64 --k 100 --top 20
```

It prints the ranked hits and writes a Neuroglancer link with them as annotations.

Chunk-size maps (no pixel reads; zarr v2/v3, sharded zarr, N5, Neuroglancer precomputed):

```bash
uv run experiments/chunkmap.py zarr://s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8 --max-chunks 5000000
uv run experiments/chunkmap.py precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg --max-chunks 5000000
```

The tests themselves: [experiments/README.md](experiments/README.md).

## Layout

| path | what |
|---|---|
| `web/` | the browser tool: TypeScript scan (worker), chunk-wise search, readers for OME-Zarr, tile pyramids, Zoomify and the Legacy Surveys cutout service; DINOv2-small through ONNX Runtime Web (chunkmirage's wrapper); OpenLayers, Neuroglancer and Cesium viewers |
| `experiments/` | the Python side: the search logic and CLI (`search.py`, `query.py`), embeddings with a cache (`embed.py`), data access (`common.py`), chunk-size maps (`chunkmap.py`), and tests 1-10 |
| `experiments/results/` | the tests' numbers, plots and write-up (`RESULTS.md`) |
