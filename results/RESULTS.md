# coresample feasibility: results

> **Partial report (Test 2 still running).** Test 1 is complete. Test 2 numbers below
> are from the first query only and will be replaced by 5-query means and the verdict.

Dataset: `jrc_hela-2` (OpenOrganelle), EM `recon-1/em/fibsem-uint8`, 6368 x 1600 x 12000
voxels at 5.24 x 4 x 4 nm, six 2x levels (s0 chunks 128^3, s1-s5 chunks 64^3, zstd 6).
Reference labels are the whole-volume COSEM segmentations (`*_seg`, N5), used only to
grade; manual ground truth exists only in 26 small crops. Layout: [layout.md](layout.md).

## Test 1: compression heatmap

Map cell = one s0 chunk (128^3 voxels, 0.67 x 0.51 x 0.51 um); 54,684 full cells,
23,591 of them inside the cell (foreground mask > 50%). Sizes come from `chunkmap.py`
(bucket listing, 61,100 s0 sizes in 2 s). Label occupancy: mito and vesicle from 8 nm
labels, the any-organelle union of 18 classes from 32 nm labels.

| signal | AUC foreground vs background | AUC mito, within foreground | AUC vesicle, within foreground | Spearman any-organelle, within foreground |
|---|---|---|---|---|
| zarr uint8 zstd s0 size (free) | 0.95 | 0.67 | 0.57 | 0.79 |
| zarr uint8 zstd s1 size (free) | 0.89 | 0.69 | 0.66 | 0.62 |
| zarr uint8 zstd s2 size (free) | 0.92 | 0.66 | 0.55 | 0.62 |
| N5 uint16 gzip s0 size (free) | 0.94 | 0.69 | 0.57 | 0.81 |
| s0/s1 size ratio, same region (free) | **0.98** | 0.65 | 0.56 | 0.77 |
| s0 size minus local median (free) | 0.72 | 0.53 | 0.61 | 0.04 |
| residual from size-vs-coarse-intensity trend | 0.90 | 0.64 | 0.66 | 0.41 |
| abs(residual), outlier rank | 0.73 | 0.59 | 0.59 | 0.24 |
| EM s3 intensity std (reads 1/512 of s0) | 0.92 | 0.57 | 0.50 | 0.59 |

Across all cells (resin included) size predicts mito at AUC 0.87 and vesicles at 0.88,
but that is mostly "is this cell or resin": within the cell it drops to 0.57-0.71.

**Verdict: pass as a where-is-the-content map, not as an organelle detector.**
Chunk size separates cell from resin (AUC 0.95; 0.98 for the free s0/s1 ratio) and
visibly tracks structure: the nucleus and its chromatin texture, the cell outline, the
mito-rich ring around the nucleus ([test1_heatmaps.png](test1_heatmaps.png)). Within the
cell it is a weak class signal.

Findings and confounders:

- **Free beats a cheap read.** Within the cell, s0 chunk size (no pixels) predicts mito
  and vesicles better than the per-chunk intensity std of EM s3: the size channel carries
  fine-scale information that downsampling averages away.
- **Raw EM barely compresses.** s0 chunks are 1.0-1.6 MB of 2.1 MB raw (59.6 GB stored
  for 122 GB raw); the median s1 chunk is 258 KB of 262 KB, i.e. zstd stores it nearly
  raw. Size saturates where the texture is, so dynamic range inside the cell is small.
- **Compressor type barely matters.** N5 uint16 gzip (64^3, 19% of blocks not stored)
  gives the same picture as zarr uint8 zstd.
- **Acquisition state shows up.** A z-plane with anomalous sizes and a z-band of larger
  background chunks are visible in every encoding; so is a missing row in the foreground
  mask itself. This is the QA use (corruption, imaging changes, write progress).
- **The texture-vs-intensity residual** (addendum idea) helps vesicles a little (0.66 vs
  0.57) and not mito; its outlier rank is near chance for organelles. Constant edge
  chunks must be excluded from the trend fit or they dominate it.
- **Display.** A few hundred near-empty edge chunks used to set the colour range and wash
  out the cell; `chunkmap.py` now stretches over non-empty chunks only.

Real-bucket runs of `chunkmap.py` (handoff's open item):

| dataset | levels mapped | chunk sizes | time |
|---|---|---|---|
| jrc_hela-2 zarr, s0-s5 (`--max-chunks 5000000`) | 6 | 122,850 | 10.7 s |
| jrc_hela-2 N5 uint16, s0-s4 (new N5 support) | 5 | ~540,000 | 17 s |
| hemibrain precomputed (sharded JPEG), 32 nm-2048 nm | 7 | 3.47 M | 59 s (32 nm: 3.41 M in 44 s, 7,140 minishard index reads) |

The hemibrain map shows neuropil compartments and acquisition-slab seams from sizes alone
([chunkmap/hemibrain_32nm.png](chunkmap/hemibrain_32nm.png)).

## Test 2: coarse-to-fine recall (running)

Subvolume: s0 z 640-2688, all y, x 512-4608 (10.7 x 6.4 x 16.4 um of cytoplasm left of
the nucleus, a nucleus edge and some resin; [overview_s4.png](overview_s4.png)).

First mito query (working level s2 = 16 nm, 64-px patches, 62 instances, chance
precision 0.07): brute force p@10 0.50, p@25 0.52; coarse-to-fine from s4 with K=100
p@10 0.50, p@25 0.48, while reading 29.5% of the working level's bytes as stored (16.9%
with 32^3 inner chunks, 4.4% of the voxels). The cheap s4 scan alone, before any
embedding, gave p@10 0.60 in 3.7 s.
