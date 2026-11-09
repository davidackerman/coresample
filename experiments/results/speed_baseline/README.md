# Speed baseline on a real GPU (design Goal 2)

The page at commit `1da1b37`, unchanged, in headless Chromium on `ackermand-ws2` (RTX 2080 Ti, NVIDIA driver 535; WebGPU through ANGLE on Vulkan). Script: [`experiments/speed_baseline.mjs`](../../speed_baseline.mjs). Raw records: [`baseline10.jsonl`](baseline10.jsonl). Run on 2026-11-08.

Method: for each dataset, 10 clicks at fixed spots in the preset's view, each in a fresh browser context (empty HTTP cache), after the model had loaded. A search ran until done or 90 s. Every figure comes from the page's own status line and model chip. "Model busy" is model runs (rough places plus close-read patches) divided by the model's median rate and by search time, so it is an estimate.

| dataset | searches | first hit (s) | cheap scan (s) | model busy | rough share of runs | model rate (patches/s) | tiles looked over / read | MB |
|---|---|---|---|---|---|---|---|---|
| HeLa (`hela`) | 6 (whole search area done in 23-25 s) | 8.9 (8.6-9.0) | 5.6 | 0.67 (0.56-0.72) | 0.56 | 440 | 7,050 / 386 | 275 |
| Mars CTX (`mars-ctx`) | 10 | 9.1 (9.0-10.1) | 2.0 | 0.78 (0.77-0.78) | 0.71 | 625 | 2,132 / 200 | 137 |
| Sentinel-2 (`s2`) | 9 | 12.3 (12.1-12.5) | 6.9 | 0.62 (0.61-0.65) | 0.72 | 675 | 7,044 / 665 | 26 |
| FlyEM hemibrain (precomputed) | 8 | 13.4 (10.5-14.7) | 6.0 | 0.24 (0.22-0.27) | 0.60 | 390 | 6,208 / 456 | 150 |

Medians, with ranges in brackets.

## Findings

1.  **The model is the limit only on Mars** (busy 0.78, above the design's 0.70). HeLa and Sentinel-2 are in between, and on hemibrain the model is idle three quarters of the time: reading is the limit there, as test 11 noted.
2.  **The rough pass is 56-72% of model runs**, in line with "about three quarters" in RESULTS.md.
3.  **The cheap scan is the largest single part of the time to first hit** on three of four datasets: 5.6 of 8.9 s on HeLa, 6.9 of 12.3 s on Sentinel-2, 6.0 of 13.4 s on hemibrain (2.0 of 9.1 s on Mars). This is a lever for early results that the design did not list.
4.  **On Sentinel-2 the model idles about 38% of the time** although only 26 MB is downloaded. The cause is not known (tile latency, round scheduling or the main thread are candidates).
5.  **The progress line divides by the whole planet.** Mars reads 0.40% and Sentinel-2 0.17% after 90 s, although each covered an area of about 1,500 x 1,100 km around the click.

## Where the wait for the first hit goes

`experiments/scan_timing.mjs`, 3 clicks per dataset, the page unchanged. An init script timestamps messages to and from the scan worker and watches the phase line and the hit list; Playwright records the requests and CDP the main thread's busy time. Raw records: [`scan_timing.jsonl`](scan_timing.jsonl). Seconds from the click:

| dataset | scan starts | coarse read | scan worker compute | scan end to reading | reading to first hit | first hit | coarse read: requests, MB | main thread busy, click to scan end |
|---|---|---|---|---|---|---|---|---|
| HeLa | 0.1 | 0.3-0.4 | 5.0-5.1 | 0.4 | 3.1-3.4 | 9.0-9.3 | 23, 1.6 | 0.6-0.7 |
| Sentinel-2 | 0.3 | 1.6 | 5.4-5.7 | 0.5 | 4.6-4.7 | 12.5-12.8 | 80, 1.9 | 1.0 |
| hemibrain | 0.4-1.4 | 1.1-1.3 | 4.5-4.6 | 0.7-0.9 | 4.3-7.4 | 11.3-15.4 | 29, 0.4 | 0.8 |

The scan's arithmetic in one worker is 4.5-5.7 s, about 40-55% of the wait. Downloads and the main thread are small beside it. Mars is left out: in this run the page opened zoomed out to the whole planet and searched 32 tiles at level 10, so it is not comparable with the baseline (the cause, the init script or the view's loading, was not found).

## The cheap scan split across workers (branch `feature/split-scan`)

The scan cut into slabs along its slowest axis, each with an 8-voxel border (the reach of its widest filter), scored in a pool of up to 4 workers and joined in one; the percentile sample now sorts as a typed array (same values, about 10x faster). Parity: `web/test/scan_split.test.ts` (the scan in one piece equals the scan before the change exactly; 2-4 slabs agree to under 1e-4).

Both builds (`816aa0b` and the branch) served by `vite preview`, timed back to back, alternating by dataset, 5 clicks each at the same spots (`split_scan_before.jsonl`, `split_scan_after.jsonl`). Medians in seconds:

| dataset | scan phase (scan start to reading) | first hit |
|---|---|---|
| HeLa | 5.9 → 2.9 | 9.1 → 6.1 (−33%) |
| Sentinel-2 | 7.5 → 3.9 | 12.5 → 8.9 (−29%) |
| hemibrain | 6.5 → 4.8 | 12.9 → 11.0 (−14%) |

The scan's compute alone (first message to a scan worker to its last reply; 3 more clicks on the branch, `split_scan_after_compute.jsonl`): HeLa 5.2 → 2.2 s (42% of before), Sentinel-2 5.4 → 1.7 s (32%), hemibrain 4.5 → 2.9 s (64%). The design's target of 40% or less is met on Sentinel-2 and missed on HeLa (just) and hemibrain; why hemibrain gains less was not looked into. (`scan_timing.mjs` first took the first scan reply as the end of the scan; with several workers that is only the histogram-bin step, so it now takes the last.)

## Limits

*   The page was served live from a checkout that other work moved during the runs (to `816aa0b` and later). The search code (`search.ts`, `scan.ts`, `embedder.ts`, `sources.ts`) did not change between `1da1b37` and `816aa0b`; `main.ts`, `sketchpad.ts`, `datasets.ts` and `index.html` did. Since `1226044` the Mars preset opens on the whole planet, so the Mars row above used the earlier, zoomed-in view.
*   One GPU, faster than most viewers' GPUs. A laptop run is still to do.
*   Clicks are all inside one preset view per dataset, so they sample one region, not the whole dataset.
*   HeLa: 4 of 10 clicks were refused as "too close to the edge" and two others landed on the same voxel, so 5 distinct queries. Hemibrain: 2 clicks did not start a search and two landed on the same voxel. Sentinel-2: 1 click did not start a search.
*   "Model busy" uses the chip's recent rate, which varies with batch size; treat differences under about 0.05 as noise.
