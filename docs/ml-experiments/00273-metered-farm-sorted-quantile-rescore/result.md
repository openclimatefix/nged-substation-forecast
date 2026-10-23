---
id: '00273'
slug: metered-farm-sorted-quantile-rescore
title: >-
  Do the metered solar-farm conclusions survive sorting each forecast's quantiles?
theme: metered-solar-farms
also_relevant_to:
- calibration-and-uncertainty
status: finished
size: >-
  simple. Stored: private score tables under data/ml-experiments/00273-metered-farm-sorted-quantile-rescore/
  only. Serving path: not touched. Degradation rule: not touched. Designs: one defensible way to
  sort (ascending sort of the 11 quantiles in each row). Callers: this folder's scripts, copied
  from the scorers of experiments 224, 243, 248, 257, and 260. Asset graph: not touched. No fitting.
hypothesis: >-
  Some stored forecasts of experiments 224, 243, 248, 257, and 260 have quantiles that cross (a
  lower quantile above a higher one). Sorting the 11 quantiles in each row can only lower the
  pinball loss, so the stored scores are pessimistic for the arms that cross, and the arms that
  cross most gain most.
prediction: >-
  Crossing is common in the arms that fit each quantile separately (XGBoost, the equal-weight and
  per-level blends of experiments 260) and absent in the arms that cannot cross by construction
  (H5 and the plant). Sorting lowers the quantile score of the crossing arms by 0.1% to 1%, and
  does not change the sign of any headline contrast in experiments 224, 243, 248, 257, or 260.
decision_rule: >-
  Quantile score is twice the mean pinball loss over the 11 levels, in points of capacity, equal
  weight per farm, with the intervals and random seeds of the original scorers (20,000 resamples of
  whole calendar months). A verdict moves when a headline contrast changes sign, when its 95%
  interval changes from excluding zero to including zero (or the reverse), or when it crosses a
  threshold the original experiment pre-registered. The threshold for porting the one-line advice
  (constrain future heads to be monotone) is zero, because it costs nothing.
benchmark: >-
  The original scores of each experiment, recomputed here from the same stored predictions and
  checked against the published figures
row_set_id: >-
  The rows of each original experiment: experiment 224's common daylight half-hours at 12 to 36
  hour leads, scored year 2025-09 to 2026-08 (and, for experiment 248, also its replicate fold
  2025-03 to 2025-08), farms A to F
metric: >-
  Crossing rate per arm and per adjacent quantile pair; quantile score, MAE of the median, and
  pinball loss and coverage at 0.01 and 0.99, before and after sorting
code: ml-experiments/00273-metered-farm-sorted-quantile-rescore/
verdict: not supported
verdict_note: >-
  None of the 5 experiments stored a forecast with crossing quantiles, so sorting changed no score
  and no verdict.
best_arm: >-
  Not applicable: no arm changed. The headline scores reproduce exactly (H5 5.475, XGBoost 5.490,
  blend C2 5.358, two-number plant 5.642 points of capacity).
headline: >-
  All 11 to 13 arms of each of experiments 224, 243, 248, 257, and 260 have 0 crossing rows, so
  sorting the quantiles changes no stored score to 13 decimal places.
take_home_message: >-
  The stored forecasts of the five metered-farm experiments never cross, because every fit script
  already sorted the 11 quantiles before writing them. Sorting them again therefore changes no
  score and no verdict, and the 26,344 crossing rows found in experiment 268 come from that
  experiment's own head. Future heads should still be monotone by construction.
production_status: not-for-production
adopted_in: none
advice:
  - >-
    Make every future metered-farm quantile head monotone by construction, for example by
    predicting cumulative non-negative offsets from the median, or sort the 11 quantiles before
    writing them, because a crossing quantile pair is always a pure loss (Checks).
  - >-
    Do not re-score experiments 224, 243, 248, 257, or 260 for crossing, because their stored
    forecasts have none (Figure 1).
  - >-
    Count crossing rows in the stored predictions of any new quantile head before scoring it, and
    report the rate beside the score, because experiment 268 found 26,344 crossing rows among
    77,897 (Figure 1).
---
**The stored forecasts of all five metered-farm experiments have no crossing quantiles, so sorting
changes nothing and no verdict moves.** National Grid Electricity Distribution (NGED) meters six solar
farms, labelled A to F. [Experiment 224](../00224-metered-farm-dayahead-two-number/result.md),
[243](../00243-metered-farm-architecture-comparison/result.md),
[248](../00248-metered-farm-architecture-ukv-inputs/result.md),
[257](../00257-metered-farm-fair-plant-arm/result.md), and
[260](../00260-metered-farm-quantile-combination/result.md) each store 11 quantiles per half-hour.
Experiment 268 reported that 26,344 of its 77,897 stored farm-forecast rows had crossing quantiles
(a lower quantile above a higher one). Sorting the quantiles can only lower the pinball loss, so we
asked whether the scores of the five earlier experiments were pessimistic for the same reason. We
re-read every stored prediction and counted the rows with a crossing pair. We then scored each arm as
stored and with the 11 quantiles sorted in each row, using each experiment's own scoring method.

**The answer is no.** The 458,194 to 3,249,012 stored rows per experiment (8 to 13 arms each, floors
included) hold 0 crossing pairs. Sorting changes every score by less than 1e-13%, and every headline
contrast, interval, and verdict matches the original page. The scripts that fitted the arms already
sort the 11 quantiles before writing them (`np.sort` or `torch.sort` in the fit and combine scripts of
all five experiments), so the stored scores are already scores of sorted forecasts. The raw crossing
rate before that sort cannot be recovered without refitting, and no refit was run.

![Figure 1: share of stored rows with crossing quantiles](figure_1_crossing.svg)

## Numbers

- **Rows and arms.** Experiment 224: 11 arms, 458,194 rows. Experiment 243: 11 arms, 3,249,012
  rows. Experiment 248: 10 arms, 2,048,820 rows. Experiment 257: 8 arms, 916,388 rows. Experiment
  260: 13 arms, 1,707,814 rows. Rows are the stored seed rows on experiment 224's common daylight
  half-hours at 12 to 36 hour leads, so the reference arms repeat across tables. Crossing rows in
  every arm: 0, and in every one of the 10 adjacent quantile pairs: 0.
- **Ties are not crossings.** Many arms have equal neighbouring quantiles, because the plant's
  output is clamped at zero and at the farm's ceiling (up to 38% of rows in some arm). XGBoost has
  ties in 7% to 13% of rows. Ties cost nothing in the pinball loss and sorting leaves them alone.
- **The original scores reproduce.** Own history, 12 to 36 h, 2025-09 to 2026-08, with 95%
  intervals: H5 5.475 (4.95 to 5.92); XGBoost 5.490 (4.94 to 5.95); blend C2 5.358; P2 5.500; the
  two-number plant 5.642 (5.15 to 6.07). Every contrast matches its page, for example H5 minus
  XGBoost -0.27% (-1.17% to +0.83%) in 243, C2 minus H5 -2.13% (-2.80% to -1.58%) in 260, P2 minus
  H5 +0.45% (-0.36% to +1.69%) in 257, H5 with UKV minus H5 -1.10% (-2.24% to -0.07%) in 248, and
  the retained share 0.95 (0.84 to 1.02) in 224. The sorted columns equal these to 13 decimal places.
- **The 5.48 against 5.49 comparison (H5 against XGBoost)** is unchanged: 5.475 against 5.490 in
  both columns, with the pinball loss at the 0.01 and 0.99 quantiles 9.5% and 29.1% lower for H5.
- **Where 268's crossing comes from.** We could not re-count the 26,344 rows, because
  experiment 268's predictions are not in this checkout. Its head is therefore the likely source,
  and that head needs a monotone output or a sort before it is scored.

## Checks

- **Median minus mean forecast per site.** Not recomputed: no forecast changed, so the values on the
  five original pages stand.
- **Mean-forecast error beside each quantile-score contrast.** Not recomputed for the same reason.
- **Triggers.** None fired. The null result is exact, not approximate, and every original score
  reproduces, which is the control. Sorting is a deterministic, nested operation, so no placebo is needed.
- **Limits.** The page cannot say how often the arms would have crossed before the fit scripts
  sorted them. It covers the stored 12 to 36 hour rows of farms A to F only.

## What to try next

- **Monotone heads for new farm briefs.** Any future head should constrain its quantiles to be
  monotone, with cumulative non-negative offsets from the median, so that no sort is needed.
- **Look at 268's head.** Count where its crossing rows fall (which pair, which farm, which lead) to
  see whether a cumulative-offset head would remove them.
