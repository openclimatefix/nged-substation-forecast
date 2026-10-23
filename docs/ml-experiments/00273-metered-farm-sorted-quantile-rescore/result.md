---
id: '00273'
slug: metered-farm-sorted-quantile-rescore
title: >-
  Do the metered solar-farm conclusions survive sorting each forecast's quantiles?
theme: metered-solar-farms
also_relevant_to:
- calibration-and-uncertainty
status: planned
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
---
Planned. Written before any score was computed.
