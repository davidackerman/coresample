# Experiments

The feasibility tests behind the browser tool: Python at workstation scale, graded on
labelled EM. Each test asks one question about the search, and the answer decided a default
or a feature of the page. The numbers, plots and full write-up are in
[results/RESULTS.md](results/RESULTS.md).

Data: [OpenOrganelle](https://openorganelle.janelia.org) `jrc_hela-2` from the public bucket
`s3://janelia-cosem-datasets`, read lazily with range requests. Its organelle segmentations
are used only to choose queries and grade hits; the search code never sees them.

| test | question | answer |
|---|---|---|
| [1](test1_chunksize.py) | Do compressed chunk sizes show where the content is? | Cell vs resin, yes (AUC 0.95-0.98); organelles, weakly |
| [2](test2_coarse2fine.py) | Can cheap scores at coarse levels find the regions, point by point? | Vesicles yes at 3-8% of the bytes; mito no |
| [3](test3_combined.py) | Read the coarse scan's top chunks whole and embed everything in them? | Matches brute force on the top hits at about 2.5% of the bytes |
| [4](test4_feedback.py) | Do good/bad marks improve the hits? | A little, not consistently |
| [5](test5_matching.py) | Which embedding pool, input size and similarity? | None moves precision by more than 0.02; the default stays |
| [6](test6_models.py) | Would another model find more? | None of the general ones; MorphEm (fluorescence, one channel) does: +0.06 p@10, +0.04 p@25 |
| [7](test7_template.py) | Match by tile tokens (the Mars CTX explorer's way)? | Pooled tokens match the default at far fewer model runs |
| [8](test8_levels.py) | What should decide which tiles are read first? | The model two levels up (the page's rough pass) |
| [9](test9_examples.py) | Do several examples find more? | Yes where the class looks alike (vesicles +0.11 with 8) |
| [10](test10_sketch.py) | Can a drawing be the query? | Not on EM |
| [11](test11_context.py) | Can coarser levels rule out regions by their surroundings? | At 4x the box, three quarters at no loss (half the model's work); wider loses on mito |
| [12](test12_brush.py) | Does marking what matters in the box find more? | A rough brush on its own: +0.04 in the top 10, nothing in the top 25 |

## Run

From the repository root:

```bash
uv sync                                          # numpy, scipy, tensorstore, torch (CPU), transformers
export CORESAMPLE_MIRROR=/local/disk/mirror      # chunk mirror: many small files, keep off NFS

uv run python experiments/explore_layout.py      # arrays, levels, chunking, compressors of jrc_hela-2
uv run python experiments/test1_chunksize.py     # then test2_..., test3_... (each script's docstring has its options)
uv run python experiments/report.py              # plots and summary for tests 1-2
```

Downloads are cached in `experiments/cache/` (gitignored), model weights in `.hf/` at the
repository root.

## Shared code

| file | what |
|---|---|
| `search.py` | portable search logic: box sums, FFT NCC, histogram EMD, texture stats, peaks, coarse-to-fine descent, coarse scan, best-first chunk search, size-based empty mask. numpy/scipy only, no I/O, no labels |
| `query.py` | the search as a command-line tool, on raw data only (any OME-Zarr or N5 pyramid) |
| `embed.py` | DINOv2-small patch embeddings with a sqlite cache (`cache/embeddings.sqlite`) |
| `common.py` | S3 listing, local chunk mirror, chunk-touch accounting |
| `chunkmap.py` | chunk-size maps from listings, shard indexes, or HEAD requests |
| `explore_layout.py`, `report.py` | the dataset's layout; plots and tables for tests 1-2 |
