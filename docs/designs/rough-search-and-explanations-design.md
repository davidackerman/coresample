# coresample: faster early results and clearer match explanations

*   **Author:** David Ackerman
*   **Status:** Draft v3
*   **Date:** 2026-11-08

The idea list this design came from is [rough-search-and-explanations.md](./rough-search-and-explanations.md). Evidence tags: **[R]** means [RESULTS.md](../../experiments/results/RESULTS.md) has the number; **[G]** means it is a guess.

--------------------------------------------------------------------------------

## Overview

coresample is a browser page. You click something in a large image or volume and it finds more regions that look like it, with no precomputed index. The data's own multiscale pyramid decides what to read first, and DINOv2-small, running on the GPU through WebGPU, scores what it reads. It is a demo: a hit is good if a viewer sees at once that it matches, and the page does not have to find every match.

This design commits to five things and makes a sixth depend on measurements:

*   **Committed:** measure where the time goes on real GPUs (done on one workstation); fix the ambiguous scale label; make the match explanation plainer and show two named stages; count failed reads and model time per pass; split the cheap scan across several workers, because it is the largest single part of the wait for the first hit.
*   **Contingent:** rule out regions by their surroundings before the rough pass (test 11), on map-like data only, where the measurements show the model is the limit, and only if a cheap offline estimate and the ship rule both pass.

## How a search runs today

1.  **Click.** The page reads a `P` x `P` patch around the click at the working level `W` and embeds it. It also reads the same physical box at the rough level `L` and embeds that.
2.  **Cheap scan.** A model-free pass over one coarse level scores every window by histogram distance and texture statistics, and ranks the tiles of the search area. This is the tile order. It runs in a Web Worker and takes seconds.
3.  **Rough pass, in rounds.** Each round takes the next tiles in the tile order, downloads the level-`L` chunks under them, embeds every place in those chunks, and gives each tile the score of its best place. A round holds up to 768 places on a GPU. The next round downloads while the model works on this one. On a CPU the rough pass is off by default.
4.  **Close read.** Tiles are read best rough score first, but only up to `DETAIL` (10%) of the tiles the rough pass has looked at. Every patch in a read tile goes through the model and is scored; hits stream in, best first. The other 90% are skipped.
5.  **Read the rest.** A button reads the skipped tiles.
6.  **Feedback.** ✓ and ✗ under a hit (Rocchio feedback) and shift-click (another example) move the query. The rough scores of tiles not yet read are recomputed, so the reads follow.

Rounds of step 3 and reads of step 4 overlap. Example cost, HeLa with `P` = 64: a tile is 64^3 voxels at the working level; the rough pass spends about one model run on it; reading it costs 8 (its patches, at a step of `P/2`).

## Glossary

| Term | Meaning |
|---|---|
| level, `W`, `L` | One level of the pyramid; each is 2x coarser than the one below. `W` is the working level, where hits are shown. `L` is the rough level: `W+2`, or `W+1` when `P/4` would be under 16 px. |
| `P` | Patch size in pixels at `W`: 32, 64, 96 or 128, 64 by default. It is the size of the query box. |
| patch | A `P` x `P` square at `W`, the unit of the close read. |
| tile | A block of `W` aligned to its chunks. Tiles are what the page reads; a tile holds several patches. |
| place | Where the rough pass cuts a patch at level `L`: `P/4` px wide at `W+2`, covering the same ground as a patch. |
| surroundings box | A `P` px square at `W+2`. Two levels up, that covers 4x the width of a patch, so it shows the patch and what is around it. |
| model input | Every patch, place or box is resized to 112 x 112 px before the model; the model sees it as an 8 x 8 grid of 14 px pieces. The "why" colour map uses a second model at 224 px (a 16 x 16 grid). |
| model run | One patch, place or box through the model: the unit of model cost. |
| skipped | A tile the rough pass looked at but that is not among its best 10%. |
| pruned | A tile that surroundings pruning removes before the rough pass looks at it (contingent goal). |
| score | Cosine similarity between a hit's embedding and the query's: the mean of the CLS-token cosine ("look") and the mean-token cosine ("parts"). |
| precision at 10 (p@10) | Of the 10 best hits, the share that contain an instance of the target, graded against whole-volume labels that are COSEM model output (not manual annotation). [R] |
| speed baseline | The page at commit `1da1b37`, measured on real GPUs (Goal 2). |
| typical score | The median score of 64 places drawn at random from the search area when the search starts (Goal 3). |
| reference variant | The query as clicked, unrotated and unscaled ("id" in the code). |

**Tests cited** (all on one grey EM volume, OpenOrganelle `jrc_hela-2`, with targets mito in a subvolume, vesicles, and mito over the whole volume): **test 3**, reading the cheap scan's top tiles matches brute force on the top hits at about 2.5% of the bytes, for vesicles and whole-volume mito but not the mito subvolume; **test 5**, no matching option moves p@10 by more than 0.02 (rotation: +0.004 ± 0.015); **test 8**, the model two levels up orders mito tiles better than the cheap scan; **test 11**, surroundings pruning. [R]

## The Problem

**Where the time goes, measured.** On an RTX 2080 Ti, with the page at commit `1da1b37` and 6-10 searches per dataset ([speed baseline](../../experiments/results/speed_baseline/README.md)):

| dataset | first hit | cheap scan, of which worker compute | model busy | rough share of model runs |
|---|---|---|---|---|
| HeLa | 8.9 s | 5.6 s, 5.0 s | 0.67 | 0.56 |
| Mars CTX | 9.1 s | 2.0 s, not timed | 0.78 | 0.71 |
| Sentinel-2 | 12.3 s | 6.9 s, 5.5 s | 0.62 | 0.72 |
| FlyEM hemibrain | 13.4 s | 6.0 s, 4.5 s | 0.24 | 0.60 |

*   **The cheap scan is the largest single part of the wait for the first hit** on HeLa, Sentinel-2 and hemibrain. Almost all of it is the scan's own arithmetic in one Web Worker (4.5-5.7 s). Its coarse read takes 0.3-1.6 s for under 2 MB, and the main thread is busy for under 1 s.
*   **The model is the limit only on Mars** (busy 0.78). On hemibrain it is idle three quarters of the time, so reading is the limit there, as test 11 noted. HeLa finishes its whole search area in about 24 s.
*   **On Sentinel-2 the model idles 38% of the time** with only 26 MB downloaded. The cause is not known.
*   **The progress figure divides by the whole dataset.** After 90 s Mars shows 0.40% and Sentinel-2 0.17%, although each covered about 1,500 x 1,100 km around the click.

**More coverage may not improve the first screen.** The cheap scan's top tiles already match brute force on the top hits for vesicles and whole-volume mito. [R] The first screen's ceiling is the model: p@10 is about 0.45 for mito and 0.81 for vesicles. [R]

**The explanation is hard to read.**
- "look", "parts", "centre", "tokens", "cosine" and "rough pass" appear without explanation.
- A score of 0.7 reads like "70% likely right", and nothing shows what an unrelated place scores.
- "Why" implies a cause. The view shows a score split and a colour map of the 16 x 16 pieces the 224 px model sees, a finer grid than the 8 x 8 the score uses. Neither shows what drove the score.
- One progress line mixes "looked over" (rough pass) with "read" (close read), so a viewer cannot tell the list may still change.
- The scale variant "s-1" reads the next finer level (`W-1`) over the same pixel box, which is half the field of view. A hit that matches it best holds something about twice the size of what was clicked, yet the label says "at half size". It describes the query box, not the hit, so a viewer may read it backwards.

## Goals

Committed, in this order:

1.  **Scale label.** "s-1" reads "about twice the size of what you clicked" and "s+1" "about half the size", checked on a fixed patch with the scales option on. No other scale or orientation text ships.
2.  **Speed baseline.** Done on the author's workstation: headless Chromium on an RTX 2080 Ti (WebGPU through ANGLE on Vulkan, `--use-angle=vulkan --use-gl=angle --enable-unsafe-webgpu`, run outside the agent sandbox, which hides the GPU), ten clicks on each of `hela`, `mars-ctx`, `s2` and the FlyEM hemibrain volume, with "the model is the limit" fixed in advance as busy more than 70% of the search time. Scripts: `experiments/speed_baseline.mjs` and `experiments/scan_timing.mjs`. Still to do: one run by hand on a laptop's integrated GPU.
3.  **Plain explanation.** Every term the explanation shows is defined in the help text in words from the glossary; the typical score is shown next to each hit's score; the clicked crop is shown beside each hit; the control reads "how it compared", not "why". Checked by going through each UI string against the glossary.
4.  **Two named stages and an honest progress figure.** "Quick scan: this list may change" while the rough pass is still finding tiles; "close read: n of m tiles" once it has looked at the whole search area; "finished" when the close read is done; "paused" while paused. When the search works outward from the click, the share shown is of the areas around the click so far, with the area in km or µm, not of the whole dataset.
5.  **Observability.** Failed tile reads are counted and shown, not dropped silently. Model time is recorded per pass, by a label each caller gives the embedder.
6.  **A faster cheap scan.** Split the scan's arithmetic across several workers, so the scan's compute falls from 4.5-5.7 s to at most 40% of that on `hela`, `s2` and hemibrain, with the same tile order as today (equal up to floating-point rounding). **Built on branch `feature/split-scan`, not merged.** Measured: compute 42% (HeLa), 32% (Sentinel-2), 64% (hemibrain) of before; first hit 9.1 → 6.1 s, 12.5 → 8.9 s and 12.9 → 11.0 s ([results](../../experiments/results/speed_baseline/README.md)). The target is met on Sentinel-2 only; the gain on hemibrain is not yet explained.

Contingent:

7.  **Surroundings pruning, on map-like data, if all three gates pass.**
    *   **Gate 1, passed only on Mars:** pruning saves model runs, not downloads [R], so it helps only where the model is the limit. Of the four datasets that is Mars CTX (busy 0.78); HeLa (0.67), Sentinel-2 (0.62) and hemibrain (0.24) fail. Pruning is therefore considered for 2-D tile maps only.
    *   **Gate 2, a cheap offline estimate:** with test 11's code on HeLa, model runs at keep 25% stay at or below 0.6 of the rough pass alone, using the box layout the page would use; and the pre-pass, timed with Goal 2's patches/s, adds no more than half the measured time to first hit on `mars-ctx`.
    *   **Gate 3, the ship rule:** on HeLa, 30 queries per target, paired against the speed baseline: the mean p@10 difference pooled over the three targets is no worse than -0.02, and no single target's is worse than -0.04. On `mars-ctx` and `s2` in the page, the pruned search finds at least 8 of the unpruned search's top 10 on average over ten clicks. This checks agreement with the baseline, not correctness.
    *   If any gate fails, the page is unchanged and the result goes into `RESULTS.md`.

## Non-Goals

*   No human judging of matches.
*   No precomputed index and no server.
*   No orientation statement in the page, and no new orientation experiment in this design (Future Improvements).
*   No non-EM test harness in Python; non-EM checking is in the page (Goals 2 and 7).
*   No new or gated model. DINOv3-small was +0.04 ± 0.04 on vesicles and level or worse on mito at 2-4x the time, with gated weights. [R]
*   No shape-aware cheap score, 3-D context, new matching options, drawing queries for EM, or learning from marks beyond Rocchio. Test 8 already closed most of the mito ordering gap, and tests 4, 5 and 10 found little or no gain from marks, matching options and drawings. [R] 3-D context is untested and would cost 2-3x the model work. [G]
*   No refactor of `search.ts`.

## Critical User Journeys

*   **A viewer clicks.** The list is labelled "quick scan: this list may change" while the rough pass runs, then "close read: n of m tiles", then "finished". Skipped and pruned tiles are read only by "Read the rest".
*   **A viewer reads a score.** "How it compared" shows the score beside the typical score and the clicked crop beside the hit, in plain terms, with no claim about orientation.
*   **The author decides on pruning.** The three gates of Goal 7 run in order and stop at the first failure.

## Technical Plan

**Scale label, explanation and stages (Goals 1, 3, 4).** Text and layout in `main.ts` and the README. The typical score needs 64 extra model runs at the start of a search and one field in the search state.

**Speed baseline (Goal 2).** Done for the workstation, with the page unchanged: the figures come from the page's own status line and model chip, and the scan's parts from timestamps on messages to and from its worker, taken by the test script. Results are in `experiments/results/speed_baseline/`; they move into `RESULTS.md` when the other work on that file allows.

**Faster cheap scan (Goal 6).** The scan's arithmetic is per-voxel filters (a Sobel gradient, Gaussian blurs at σ 1 and 2, the second reaching 8 voxels) followed by window sums, for four texture maps and 15 histogram thresholds. All of it is local, so a block of the volume gives the same values as the whole, given a border as wide as the filters' reach plus the window. The page cuts the coarse volume into slabs along its longest axis, one per worker (the decode pool's rule: one per spare core, up to 4), each with that border. Each worker returns its slab's raw texture maps and histogram distances. The steps that need the whole volume stay in one place: the histogram edges (percentiles of a sample) are computed before the split and sent to every worker, and the per-feature spread, the z-scores and the combined score are computed after the slabs are joined. The existing scan parity test (`web/test/scan_parity.ts`) is extended to compare the split scan with the single one. Files: `scan.ts`, `scan.worker.ts`, and the scan call in `main.ts`; `search.ts` is not touched.

**Observability (Goal 5).** The embedder keeps seconds and counts per pass label ("query", "rough", "close"), so overlapping rough and close work is split correctly. `search.ts` counts read failures where it now drops them.

**Surroundings pruning (Goal 7).** As measured in test 11:
1.  Embed a surroundings box centred on the click.
2.  Lay surroundings boxes over the search area at level `W+2`, half a box apart (`P/2` px there, `2P` px at `W`), embed each, and score it by cosine with the click's box.
3.  Give each tile the score of its nearest box and keep the best 25% of tiles in the tile order.
4.  Run the rough pass on the kept tiles only. Pruned tiles join the skipped ones for "Read the rest" and in the counts.

Because one box serves several tiles, the pre-pass costs fewer runs than tiles. It runs before the first rough round, so it delays the first hit; Gate 2 measures by how much. Above a tile-count cap, set from Gate 2, the page skips pruning. It sits behind a URL flag, off by default; the default changes in a separate commit after Gate 3.

## Alternatives Considered

**Speed**
*   **Change nothing until the speed baseline exists.** Chosen as the first step, and done.
*   **Run the cheap scan on the GPU.** Likely faster than splitting it, and the GPU is free while the scan runs, but it needs new shader code for the blurs, gradient, window sums and histogram. Deferred until the split shows how much of the scan is left.
*   **Scan fewer voxels.** Lowering the 4-million-voxel budget saves time in proportion with one constant, but the same budget sets the automatic search area and the scan level, so it changes what is searched. Not chosen.
*   **Start the rough pass on the tiles nearest the click while the scan runs.** Could hide the whole scan, but changes the tile-order logic in `search.ts`, and the first tiles would be chosen by distance, not by likeness. Deferred until after the split.
*   **Make the single-worker scan cheaper** (fewer histogram bins, no σ 2 blur). Changes the scores that tests 2 and 3 validated. Not chosen.
*   **A rolling threshold per round instead of a pre-pass.** No pre-pass delay and works on any area, but not measured, the first rounds run unpruned, and the threshold drifts. Deferred.
*   **A fixed margin below the click's surroundings score.** Cheapest, but the share kept varies, so test 11's numbers do not apply. Not chosen: no basis for the margin.
*   **Keep 50% instead of 25%.** Lost 0.03 ± 0.02 on whole-volume mito and saved less (0.67-0.71 of the runs). [R] Not chosen.
*   **A smaller rough input, lower `DETAIL`, fp16.** Untested; a smaller input must be a multiple of 14 (56 or 70 px). Deferred until the speed baseline shows where time goes.
*   **Drop the cheap scan for a model-based order.** Surroundings alone in place of the rough pass lose precision (mito 0.39 vs 0.53). [R] Not chosen.

**Explanation**
*   **An orientation and scale statement per hit.** Would tell the viewer something, but rotation moved p@10 by +0.004 ± 0.015 on EM [R], there is no margin procedure, and a confident wrong statement lowers trust. Deferred.
*   **Typical score as the median of places compared so far.** Free, but biased toward similar places, because the rough pass compares a pre-ranked subset. Not chosen; the random sample costs 64 runs.

**Measurement**
*   **An augmentation self-retrieval test and Python readers for non-EM data.** Would answer whether orientation matters on any data, but no shipped change depends on it, and it is most of the harness work. Deferred.
*   **Human judging.** Ruled out by the author.
*   **A headless-browser bench.** Chosen for Goal 2. With ANGLE on Vulkan and outside the agent sandbox, headless Chromium uses the workstation GPU; the 8 patches/s in RESULTS.md was its CPU fallback.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| The pre-pass delays the first hit, worst on large areas. | Gate 2 bounds it; a tile-count cap skips pruning on large areas. |
| The page's box layout costs more runs than test 11 measured. | Use test 11's layout; Gate 2 recounts with the page's tiles. |
| Test 11 is one dataset. | Gate 3 adds two non-EM presets in the page. |
| WebGPU speed varies by machine; a 2080 Ti is faster than most viewers' GPUs. | Report the workstation and the laptop run side by side; judge Gate 2's delay on the laptop numbers. |
| A laptop has fewer spare cores, so the split scan gains less there. | The laptop run (Goal 2) measures it; with one spare core the page falls back to today's single scan. |
| Slab borders give slightly different values if the border is too narrow. | The border is set from the filters' reach and the window; the parity test compares every score. |
| `search.ts` and `main.ts` change under this work. | Small changes behind a flag; re-check call sites before starting. |
| Memory limits (3,000 kept candidates, 6 million heat-map places, 20,000 rough tiles with embeddings) at much higher coverage. | Watch them in the Goal 2 runs; out of scope unless they bind. |

**Security and privacy.** Nothing is uploaded; the model runs locally; the new flag is parsed as a number. No new endpoint, credential or user data.

## Notes for the implementation plan

*   The rough query is embedded in four places in `search.ts` (`setupRough`, `roughRound`, `roughFeedback`, `addExample`). Any change to how rough places are embedded must change all four, or embeddings of different kinds mix with no error.
*   Pruned tiles must reach "Read the rest", the skipped count, the tile total and expand mode.
*   `embedder.ts` imports the chunkmirage ONNX wrapper at the top, so a module that imports it cannot be unit-tested without that checkout. `web/test/pw_smoke.mjs` has no assertions yet.

## Open Questions

*   How fast is a laptop's integrated GPU, and is the model the limit there? (Goal 2.)
*   Why does the model idle 38% of the time on Sentinel-2, with only 26 MB downloaded?
*   What tile-count cap keeps the pre-pass worthwhile? (Gate 2.)

## Future Improvements

*   An augmentation test to answer whether orientation matters, on EM and non-EM data, and a per-hit orientation statement if it does.
*   A smaller rough input, lower `DETAIL` or fp16, if the model is the limit.
*   A rolling-threshold version of pruning for large areas.
*   DINOv3-small or a satellite-trained model compared on Sentinel-2.
*   A nudge toward several examples, and groups of similar hits.
