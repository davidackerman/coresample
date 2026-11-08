# coresample

Click something in a huge image or volume (EM, slides, planetary maps, sky surveys, satellite
imagery) and find more regions like it, in the browser, with **no precomputation**: no
embedding pass, no index. The data's own multiscale pyramid is the index: a quick pass over a
coarse level decides what to read first, and DINOv2-small scores everything read.

**Try it: https://davidackerman.github.io/coresample/** (Chrome recommended: the model runs on
your GPU through WebGPU; without it, on the CPU, about 10x slower).

## Using it

- **Search:** pick a dataset (or paste your own, below), zoom until the thing is clear, click.
  Hits stream in, best first; it keeps reading until it is done or you press **Pause**. With
  region on auto it works outward from your click, nearest areas first.
- **More examples:** shift-click (or **+ example**) adds another; hits then match their mean
  (Options > examples: or each hit's best match).
- **✓ / ✗ on a hit:** more / fewer like this. The query moves, the list re-ranks, and the tiles
  still to read are re-ordered. ✓ turns the hit green; undo under **Hits**.
- **match:** lights the parts of each hit most like what you clicked and dims the rest (on the
  query, the part being matched). **why:** each hit's score broken down, and its pieces
  coloured the way the model sees them.
- **Heat:** how well every place compared matched, on the **Where it has looked** map and over
  the image (in a volume, at the slice's depth).
- **draw:** sketch what to find. It finds bold shapes such as craters; on EM it finds nothing,
  so add real examples there.
- **Viewers:** Neuroglancer for OME-Zarr and precomputed volumes (your clicks in blue, hits in
  yellow), OpenLayers for tile maps, a globe for Mars and the Moon (**map** switches to the
  flat map). The bottom right shows the coordinates under the cursor and takes coordinates
  to go to.

## How it works

1. **Quick pass:** brightness and texture statistics over a coarse level rank where similar
   things might be (no model; a few MB).
2. **Rough pass** (GPU): the model looks two levels coarser (a sixteenth of the download) and
   picks what to read in full detail: the most promising 10% of what it has looked over (on EM,
   the top 5% already matched reading everything). **Read the rest** reads the others.
3. **Score:** every patch in the tiles read goes through DINOv2-small (112 px); the score
   under each hit is the cosine of its embedding with your click's (the mean of **look**, the
   CLS token, and **parts**, the average piece token). Memory stays bounded: the best 3,000
   candidates are kept for re-ranking.

Nothing is uploaded. The **Options** (detailed 224 px input, centre object, any rotation, any
size, adapt to data, tiles + tokens, combining examples) are for data where they make sense:
on labelled EM none moved top-10 precision by more than a few hundredths, and no other model
beat DINOv2-small on both targets (tests 5-9, [RESULTS.md](experiments/results/RESULTS.md)).

## Data

| how | example |
|---|---|
| a preset | `?d=mars-ctx`, `moon-wac`, `andromeda`, `webb`, `legacy-ngc6129`, `s2`, `hela`, `idr-organoid`, `idr-lung` |
| OME-Zarr (v0.4/0.5, 2-D or 3-D) | `https://...`, `s3://bucket/path`, `gs://bucket/path` |
| Neuroglancer precomputed | `precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg` |
| web-map tiles | `https://host/.../{z}/{x}/{y}.png` (`#maxZoom=12` sets the finest zoom) |
| Zoomify | `https://host/.../ImageProperties.xml` |
| a Legacy Surveys field | `legacy:245.43,37.996` (RA, Dec) |

Paste one under **Your own data**; the address bar then holds `?url=...` and the view
(`&v=x,y,z,scale`), so a copied link opens the same place. Other address options: `&globe=0`
(flat map), `&rough=0` / `&rough=1` (rough pass off / forced on a CPU). The data must be a
multiscale pyramid on a host that allows cross-origin reads (CORS), as public buckets, NASA
Trek, ESA and IDR do.

## Run locally

```bash
cd web && npm install && npm run model   # exports DINOv2-small to ONNX (88 MB, gitignored)
npm run build && npx vite preview        # http://localhost:5180
```

The site is built by `.github/workflows/pages.yml` (Neuroglancer from source with
`web/scripts/patch_neuroglancer.sh`, the ONNX model, chunkmirage at `web/chunkmirage.lock`).
The same search runs from the command line too, and chunk sizes alone make a map of where the
content is (no pixel reads):

```bash
uv sync
uv run python experiments/query.py zarr://s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8 \
    --point 1700,400,2600 --working 2 --coarsest 4 --patch 64 --k 100 --top 20
uv run experiments/chunkmap.py zarr://s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8
```

## Layout

| path | what |
|---|---|
| `web/` | the browser tool (TypeScript): readers, the search, DINOv2 through ONNX Runtime Web, the viewers |
| `experiments/` | the Python side: search logic, CLI, chunk-size maps, and the ten feasibility tests ([README](experiments/README.md)) |
| `experiments/results/` | the tests' numbers and write-up ([RESULTS.md](experiments/results/RESULTS.md)) |
