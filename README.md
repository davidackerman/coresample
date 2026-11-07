# coresample

Search a huge chunked array (volume EM, slides, satellite rasters, climate grids) for
"regions like this one" with **no precomputation**: no embedding pass, no index build,
nothing before the first query. The multiscale pyramid the data owner already wrote is
the index. Cheap model-free scores rank the coarsest level, only the best candidates
are read at finer levels, and an embedding model (DINOv2-small) re-ranks the survivors.
Every embedding is cached, so the index builds itself from use. Compressed chunk sizes,
read from storage metadata without touching pixels, give a free map of where the
content is.

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
publishes the build to GitHub Pages.

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
