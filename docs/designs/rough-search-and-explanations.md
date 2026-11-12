# coresample concepts brief: faster rough search, clearer "why", and what else is worth doing

Author: David Ackerman. Date: 2026-11-08. Mode: brainstorm. Six review lenses (design, codebase details, zoom-out, comprehension, implementation readiness, structure) all completed.

Evidence labels used below:
- **[R]** backed by a number in `experiments/results/RESULTS.md`.
- **[G]** a guess: no measurement yet.

## Overview

coresample is a click-to-find-more-like-this demo that runs entirely in the browser, with no precomputed index. It only has to show matches a person recognises at once, not find everything. Success is a viewer saying "yes, that clearly matches". No human judging will be done (author decision), so success is measured with labels where they exist and with a self-retrieval test with augmented copies everywhere else.

The three pieces of work:
- (a) A faster rough search that covers far more of the data before the first screens are built (today about 0.15%), followed by a slower fine pass.
- (b) Clearer "why this matched" explanations, including whether orientation and scale mattered.
- (c) Other robustness or interest ideas: shape-aware scoring for mito, 3-D context, clustering or outliers, learning from marks, a second dataset.

Summary of the reviewers' view:
- The cheap, well-evidenced wins are (a) tuning the rough pass, measured honestly, and (b) rewording and relabelling the explanation.
- The biggest open credibility gap is that everything is validated on one EM dataset, so a second-dataset check should come early.
- Most of (c) either already exists or has no supporting evidence, and should be cut or deferred.

## Analysis

### What the evidence says
- **No option or model moves precision.** No matching option or model moved p@10 by more than 0.02. The best of 24 variants gained 0.03 to 0.04, which is about 2 standard errors and looks like noise. [R, tests 5 and 6]
- **Mito is the weak class.** Mito p@10 is about 0.45, against 0.81 for vesicles, so roughly half of the first mito screen is wrong. [R]
- **Mito ordering is already improved.** The model two levels up orders the reads better than the cheap scan. Mito p@10 is about 0.42 after 1% of tiles, against 0.33 for the cheap scan. [R, test 8] This closes much of the gap that "shape-aware coarse scoring" was meant to fix.
- **The model is the speed limit.** The rough pass is about three quarters of the model's work. The Mars run covered 5,688 places in 90 s, about 30x more ground than before. [R] That figure used a stand-in model at 400 patches/s. Real-GPU throughput is unmeasured. [G]
- **Several examples help only some classes.** Averaging 8 examples gave +0.11 on vesicles. On mito, averaging was slightly negative and "best match" was better. [R, test 9]
- **Feedback is inconsistent.** Rocchio gave 0 to +12% and logistic regression -10% to +20%. [R, test 4]
- **Drawings find nothing on EM.** [R, test 10]
- **Coverage is not precision.** The coarse scan's top chunks already match brute force on the top hits at about 2.5% of bytes. [R, test 3] More coverage mainly buys recall and fresher first screens. The first-screen quality ceiling is the model (mito about 0.45), not the data fraction.

### Code facts the reviewers found (approximate lines in `web/src/search.ts`, which is being edited)
- The rough pass lives in `Rough` (~72), `setupRough` (~589), `roughRound` (~620), `roughFetch` (~651), `roughRequery` (~704), `takeTiles` (~571). `DETAIL` is ~90, and the per-round budget is 768 places on GPU (48 on CPU).
- The rough pass is GPU-only. The WASM fallback gets no rough pass, so any speedup applies to WebGPU users only.
- At level W+2 the patch is P/4 (16 px) and is upsampled to the 112 px model input. A smaller input is therefore a plausible cost saving, but precision at 56 or 64 px is untested. [G]
- The page scores every place under a level-L chunk, whereas test 8 used one patch per tile. Test 8 numbers therefore do not transfer exactly. The rough pass also uses only the unrotated query.
- `explain()` (~999) and `queryVariants` (~870) hold the orientation and scale data. The variant is non-identity only when rotations or scales are switched on, and both are off by default. So orientation never appears in the default "why".
- The UI side is `whyText` (~main.ts:984), `showWhyHelp` (~994) and the `VARIANT` map (~983).
- **Possible bug to verify before shipping any explanation:** scale label "s-1" reads level W-1 over the same pixel box, which looks like a view half the size, yet it is labelled "at half size". A hit matching it would be about twice as large as the query, so the label may be inverted. Rotation labels r1 to r7 are printed as "turned (N)", which means nothing to a reader.
- `readTile` and `readAt` swallow read failures silently (~450, ~613). `State` has no failed-tile counter and no split of model time between the rough and fine passes.
- `search.ts` is a ~1,080-line class holding the rough pass, reading, ranking, Rocchio feedback, explain, heat and examples. The token-mode and patch-mode branches duplicate ranking and heat.

### Comprehension problems in the current explanation
- Terms such as "look", "parts", "centre", "tokens", "cosine", "rough pass" and "level 3" are never explained.
- Scores read as confidence. A cosine of 0.7 reads as "70% likely right" with no baseline.
- "Why" implies a cause, but the display is a score split plus a PCA colour map, which shows how the model groups pieces, not what drove the score.
- When rotation and scale options are off, the viewer cannot tell "did not matter" from "not tested".
- One progress bar mixes "looked over" (rough) with "read" (fine), so the viewer cannot tell the list is provisional.

## Goals vs Non-Goals

**Goals**
1. A measurably earlier first good screen: more ground covered per second at no loss of p@10 (within 0.02, paired, 30 queries, across targets).
2. An explanation a first-time viewer reads correctly, including an honest orientation and scale statement, without overclaiming.
3. A clear provisional versus refined display of results.
4. A check of the core claims (tests 3, 5 and 8) on non-EM data as well as a second EM volume.

**Non-Goals**
- No new or gated model unless it clearly pays. DINOv3-small is +0.04 +/- 0.04 on vesicles and level or worse on mito at 2 to 4x the time, with gated weights. [R] It is not justified now.
- No precomputed index and no server.
- No further investment in drawing queries on EM [R], or in new matching options. [R]
- No promise that explanations improve precision. They are for trust and judgement.
- No shape-aware cheap score for mito, because test 8 already closed most of that gap. [R]
- No large refactor of `search.ts` while it is being edited concurrently.

## Alternatives Weighed

Ideas are ranked by expected gain against effort. Each carries a conflict-risk note for the concurrent `search.ts` edits.

| # | Idea | Gain / effort | Evidence | Conflict risk with concurrent edits |
|---|---|---|---|---|
| 1 | **Measure first: a headless coverage-vs-time benchmark** (extend untracked `web/bench/`) with two metrics: p@10 on labelled EM, and an **augmentation self-retrieval test** that works on any data (below). No human judging | Defines success; low effort | Needed because the 30x figure came from a stand-in model [R] and p@10 is region-level against model-output labels [R]. Self-retrieval is new [G] | None (new files, docs) |
| 2 | **Validation beyond HeLa EM**: another EM volume plus non-EM data (Mars CTX, a Legacy Surveys field, Sentinel-2, an IDR organoid or lung image). Labelled data gets p@10; unlabelled data gets the self-retrieval test and the rough-pass-versus-full-read agreement (does the rough pass's top 10% contain the full read's top hits?) | High credibility; medium effort | RESULTS itself says test 3 should be confirmed elsewhere [R] | None (Python in `experiments/`, or the headless bench) |
| 3 | **Cheaper rough pass**: smaller rough input (56 or 64 px), fp16 or int8, batch tuning, lower `DETAIL` (0.1 to 0.05), stride | Highest gain for (a); medium effort | Rough is 3/4 of model work and top 5% matches full read [R]. Smaller input, fp16 and batching are guesses [G]. 224 px vs 112 px moved p@10 by only +0.009 +/- 0.019 [R], which hints input size matters little but never tested 56 px | HIGH (`setupRough`, `roughFetch`, `DETAIL`, budget in `search.ts`; a new ONNX export in `embedder.ts` and `web/scripts/export_model.py`). Prototype offline in Python on cached embeddings (`test8_levels.py`), then add behind flags (`&roughpx=`) after the other edit lands |
| 4 | **Plainer "why" wording and honesty** (rewrite help text, define terms, give a baseline such as typical unrelated-place score, say precision rarely changed in tests, rename "why" to something like "how it compared") | High comprehension gain; low effort | Option flatness [R]; baseline needs one new number [G] | LOW (`main.ts` text, README). MEDIUM only if a baseline field is added to `Explain` |
| 5 | **Orientation and scale statement**: compute the best rotation/scale variant for displayed hits only, using a separate explain-only variant list that `rank` ignores; show the matched variant as a thumbnail and name it ("rotated 90 degrees", "mirrored"); say "not tested" when off; show it only above a measured margin | High trust gain; low to medium effort | Rotation moved p@10 by +0.004 +/- 0.015 [R], so the best variant may be noise on EM and the margin threshold must be measured [G] | MEDIUM (`explain`, `queryVariants` in `search.ts`; `whyText` and the `VARIANT` map in `main.ts`). Keep it to one added field on the item, with rendering in `main.ts` |
| 6 | **Query crop beside the hit** with the existing match heat | High; low effort | Lets a viewer judge directly, which matters since mito is about half wrong [R] | LOW (`main.ts`) |
| 7 | **Two named stages in the UI** ("quick scan: list may change", "close read"), with a "still changing" badge | Medium; low effort | Guess [G] | LOW to MEDIUM (`main.ts`, existing `State` fields) |
| 8 | **Nudge toward several examples**, with "match any one" suggested for mito-like targets | Medium; low effort | +0.11 on vesicles with 8 averaged; mito mixed [R] | LOW as a UI nudge; HIGH if the combine mode is chosen automatically |
| 9 | **Cluster hits / show one representative per group, near-duplicate suppression** | Medium; low effort | No evidence [G]. A coherent cluster of wrong mito hits would look convincing, so show it only as "groups of similar hits" | LOW if post-processing in `main.ts`; MEDIUM if inside `rank` |
| 10 | **Observability**: failed-tile counters, rough versus fine seconds, places/s | Enables idea 3; low effort | Silent swallow of read errors found in code | MEDIUM (`State`, `readTile`, `readAt`) |
| 11 | **Model-based rough ranking as primary, cheap scan demoted to scheduler; try W+1 on the cheap scan's top 20%** | Medium; medium effort | W+2 beats cheap scan for mito [R]; W+1 costs about 10% of working level when applied to all [R] | HIGH (`scanRegion`, `roughFetch`) |
| 12 | **3-D / orthogonal context** | Speculative; high effort | No measurement; xy only [R caveat]. Costs 2 to 3x model work while model is the limit | HIGH (`readAt`, `readTile`). If tried, do it offline in Python first (embed xz and yz slices, require more than +0.03) |
| 13 | **Learning from checks/crosses (logistic regression etc.)** | Low | Inconsistent [R, test 4] | HIGH (`applyFeedback`, `roughRequery`). Keep Rocchio, no new work |
| 14 | **Shape-aware cheap score for mito** | Low now | Test 8 already beats the cheap scan [R]; NCC with rotations already failed [R] | MEDIUM (`scan.ts`, scan wiring) |
| 15 | **Gated models** | None now | [R, test 6] | Low, but not recommended |

### Augmentation self-retrieval test (replaces human judging)

Take a patch, make an altered copy, hide the copy among the data, click the original, and record where the copy ranks (top-1, top-10, and the score gap to the best unrelated hit). The alterations are rotation, flip, scale, brightness and contrast, blur and noise, each at several strengths. [G]

- **What it gives:** a ground truth on any data, labelled or not. It sets the margin for showing an orientation or scale statement: if rotation by 90 degrees is retrieved only with the rotations option on, the page can say so with evidence. It also checks that a faster rough pass still finds the copy, at equal wall-clock time.
- **What it does not give:** other instances of the same class. DINOv2 is trained to be invariant to these alterations, so the test may be too easy, and a pass says nothing about whether mito match mito. Keep p@10 on labelled EM for that.
- **Layout-shuffle alteration:** also cut a patch into pieces and shuffle them. DINOv2 averages its 8x8 tokens, so a shuffled copy may still match. If it does, the page should say "this match ignores orientation and layout", not "orientation did not matter here". [G]
- **Design points to settle:** where the copy is planted (inside a tile the search will read, or the rough pass could never see it), how many distractors, and whether near-duplicates in the data (Mars has repeated tiles) count as hits. Plant it in a region drawn at random from the whole search area, and report results per alteration and per dataset.

### Top objections raised
1. **Coverage may not improve first-screen quality.** Top hits were already at brute-force level at about 2.5% of bytes [R]. The first screen is capped by the model, so "much more data" may mainly improve recall and freshness. Any speedup needs a before/after p@10 check at equal wall-clock time.
2. **The metric does not match the success statement.** "Yes, that clearly matches" is a human judgement, and the author will not run human judging. p@10 is region-level against model-output labels on a single dataset, and mito at about 0.45 does not meet that bar. The augmentation test replaces the human check but measures something narrower: see "Augmentation self-retrieval test" below.
3. **Explanations can hurt.** A confident story on a wrong hit lowers trust (author agrees this is a real risk). Show orientation and scale only above a margin measured with the augmentation test, and keep the wording as "how it compared", not "why".
4. **Speed work competes with every other model-based idea.** Explain variants, orthogonal slices and rotations all add embeds while the model is the limit.
5. **Numbers may not transfer** to sky, Mars or organoid data, where no labels exist to grade with. Decision: test on non-EM data, using the augmentation test (idea 2).

### Note on the model's resolution and domain-specific models
- At the default 112 px input, a 64 px patch becomes an 8x8 grid of tokens (about 8 source pixels each); the detailed 224 px mode gives 16x16 (about 4 each). Each token is a 384-number description in context, so content is rich but layout is coarse. A smaller rough-pass input is plausible but untested. [G]
- A satellite-trained DINOv3 exists (from memory, unverified: a large model, gated weights). It could only help on Earth imagery and is likely too slow for the rough pass. Optional later check: on Sentinel-2, compare DINOv2-small, DINOv3-small and the satellite model with the augmentation test. Not a default; not a goal now. [G]

## Key Decisions & Staging for Next Steps

### Decisions to take now
1. Adopt the minimal bundle: (3) a measured cheaper rough pass, (4)+(5)+(6) clearer explanation, and (2) the second-dataset check. Everything else waits for evidence.
2. No human judging. "Recognised at once" is approximated by the augmentation self-retrieval test, alongside p@10 where labels exist.
3. Test on non-EM data, not only a second EM volume.
4. Rename or reframe "why" as a similarity breakdown, and fix or confirm the scale label before shipping any orientation or scale text.
5. Keep Rocchio, the current model, and current matching defaults. Do not build a shape-aware cheap score.
6. No new feature may add fields to the `Search` class. Each new piece is a module with an explicit interface.

### Staging
- **Stage 0, now (no conflict):** write the benchmark with the augmentation test (idea 1). Verify the "s-1" scale label. Draft the plain-wording help text. Start the second-dataset experiment (idea 2).
- **Stage 1, offline (no conflict):** in Python on cached embeddings, test the smaller rough input, lower `DETAIL`, and W+1-on-top-20% variants. Acceptance: p@10 within 0.02 of baseline (paired, 30 queries, three targets) at 2x or more places/s. Also measure real-GPU patches/s on at least two GPUs by batch size, and the rough-pass split between download and model time.
- **Stage 2, UI-only changes in `main.ts` and README (low conflict):** plain wording, crop beside hit, two-stage progress display, named transforms, multi-example nudge.
- **Stage 3, after the concurrent `search.ts` edit lands (high conflict):** add the rough-pass flags (`&roughpx=` or similar, off by default), the one-field explain extension, and observability counters. Work in a worktree off main, not the dirty `feature/legacy-sky` checkout, and leave `search_old.ts` alone.
- **Stage 4, optional:** pure-move extraction (`explain.ts`, then `rough.ts`, then a feedback store) as its own commit once things settle. Clustering as a separate main.ts module. Each extra behind a flag with its own kill criterion.

### Open risks
- WebGPU throughput and memory are unmeasured on real hardware. The WASM path is about 10x slower and has no rough pass.
- Smaller or quantised models need new ONNX exports (44 MB files, gitignored), so the build and cache key change.
- Higher coverage may push past `KEEP=3000`, `HEAT_MAX=6M` and `POOL_EMBS=20000`. Check at 10x coverage.
- The augmentation test may be too easy for DINOv2 and needs harder alterations (strong crops, partial occlusion) to separate variants.
- The "0.15%" figure is not in RESULTS (the nearest is Mars at 0.11% in 122 s), so confirm it is the same measurement before using it as a baseline.
- The model download is large, and a slow connection leaves the first screen empty.
- Untracked files (`search_old.ts`, `web/bench/`, `experiments/results/*`) will pollute diffs and need cleaning or ignoring before staging.
- Test 3 and test 8 defaults (`DETAIL=0.1`, W+2) were tuned on one grey EM volume and may not transfer to colour data.
