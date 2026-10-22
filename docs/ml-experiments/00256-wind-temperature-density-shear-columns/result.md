---
id: "00256"
slug: wind-temperature-density-shear-columns
title: >-
  Do air temperature, air density, or a shear column lower the XGBoost wind forecast's quantile
  score on the three metered wind farms?
theme: generator-forecasting
also_relevant_to: []
status: planned
date: 2026-10-22
size: >-
  simple. Stored: private files under data/ml-experiments/00256-wind-temperature-density-shear-columns/
  only. Serving path: not touched (a column the ENS table already carries). Degradation rule: not
  touched. Designs: one defensible test, a column permutation of experiment 230's recipe with a
  placebo. Callers: this folder's scripts, which copy experiment 230's helpers. One Sonnet code
  review, one Sonnet write-up review, one Sonnet prose review.
hypothesis: >-
  Adding 2 m air temperature (arm T1) to the six wind columns of the XGBoost wind model lowers the
  quantile score on the three metered wind farms (W1 to W3), because air density and stability change
  the power a turbine produces at a given forecast wind speed. A density column (T2) carries the
  same information as temperature, and a shear and solar-elevation column set (T3) adds a further
  gain at night.
prediction: >-
  Written before any fit. C1, T1 minus T0, kept rows, all leads, is -0.04 to -0.12 points of
  capacity. C2, T2 minus T1, is inside -0.03 to +0.03. C3, T3 minus T1, is -0.05 to +0.02, with any
  gain at night and at lead days 1 to 6. C4, the placebo P1 minus T0, is inside -0.02 to +0.02. The
  0.99 pinball loss falls more than the 0.01 loss in T1.
decision_rule: >-
  Four decisive contrasts, C1 to C4, each on the quantile score over all leads on the rows kept by
  experiment 210's mask, read at 98.75% (Bonferroni across four), resampling whole months of the ENS
  run and the three seeds, 50,000 resamples. The porting threshold is zero, because a column adds no
  pipeline step (the ENS table already carries temperature and surface pressure). A column is
  recommended when its contrast interval lies wholly below zero and the placebo's interval includes
  zero. If C1 holds and C4 does not, the verdict is inconclusive (the gain may be the extra column).
  All other contrasts, per farm, per lead band, and per season are descriptive. Intervals describe
  these three farms, because farms are not resampled.
benchmark: >-
  T0, experiment 230's A0r recipe refitted here. T0 must reproduce experiment 230's stored A0r to
  within 0.02 points on the kept rows, or the experiment stops.
row_set_id: >-
  experiment 33's wind cases (`fold_a_tuesdays_v1`: 53 Tuesday runs from 2025-07-01 to 2026-06-23,
  W1 to W3, three seeds). Headline on the rows kept by experiment 210's mask; all rows descriptive.
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over nine levels), in points of capacity;
  pinball loss at 0.01, 0.05, 0.5, 0.95, and 0.99; coverage of the 5% to 95% and 1% to 99% bands
production_status: not-for-production
adopted_in: none
code: ml-experiments/00256-wind-temperature-density-shear-columns/
---

**Question.** [Experiment 230](../00230-wind-physics-core-part-1/result.md) found, after its plan,
that adding the 2 m air temperature to the XGBoost wind model lowered the kept-row quantile score by
0.083 points (arm A0rT). That contrast was post hoc, so this experiment repeats it as a planned
contrast and asks which physical route explains it.

## Arms

All arms are experiment 230's A0r recipe: an XGBoost quantile model with the power-curve base margin,
trained on ERA5 valid times from 2019-09-01 to 2025-06-16, run on the 51 ENS members, mixed with equal
weights, and given the spread factor. They differ only in their columns, and `colsample_bytree` stays
at 1 (colsample invalidates feature ablations).

- **T0:** the six wind columns (10 m and 100 m speed, and the sine and cosine of each direction).
- **T1:** T0 plus 2 m air temperature (ERA5 in training; the ENS member's own at inference).
- **T2:** T0 plus air density from 2 m temperature and surface pressure (ERA5 `sp` at the nearest
  cell in training; the ENS member's `pressure_surface` at inference), without temperature.
- **T3:** T1 plus density, plus a shear column (the natural log of the 100 m over 10 m wind speed
  ratio, divided by the natural log of 10), plus the solar elevation angle.
- **P1 (placebo):** T1 with the temperature column permuted by whole calendar months within each farm.
- **References:** climatology, and experiment 230's stored A0r and A0rT.

**Weather at lag times.** The experiment uses no lagged weather. ERA5 is the training-time weather
and is not available at forecast time; the ENS members supply every column at inference.

## Stop rules

- If T0 differs from experiment 230's A0r by more than 0.02 points on the kept rows, stop and find the
  cause.
- If the monthly mean of ENS surface pressure differs from ERA5's by more than 5 hPa at any farm, run
  T2 and T3 with a fixed 100 kPa pressure and say so.

## Compute

CPU for the column work and XGBoost on the GPU one fit at a time, about 2 to 3 hours at 4 cores, run
as `systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue`.
