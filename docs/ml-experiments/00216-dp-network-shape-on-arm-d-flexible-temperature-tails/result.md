---
id: '00216'
slug: dp-network-shape-on-arm-d-flexible-temperature-tails
title: >-
  Differentiable physics: can a more flexible temperature curve in the convex model carry the
  network's correction?
theme: differentiable-physics-forecast-engine
also_relevant_to: [calibration-and-uncertainty]
status: planned
size: >-
  medium. Stored: private files under
  data/ml-experiments/00216-dp-network-shape-on-arm-d-flexible-temperature-tails/ only. Serving
  path: not touched. Degradation rule: not touched. Designs: more than one is defensible (where the
  extra knots sit, how weak the tail penalty is, which smoothing windows carry tail terms), and this
  plan pre-registers a small grid chosen on the tuning months. Callers: this folder's scripts, which
  copy the fit, forecast, and correction code of experiments 136, 155, and 201 and call the
  reviewed `studies.*` modules without changing them. Refitting the convex model at six
  substations is the only new fitting.
hypothesis: >-
  The neural network of experiments 200 and 201 lowers the quantile score of arm D (the best-guess
  convex forecast of experiment 155) by 0.09 to 0.125 points, mainly by raising the median on very
  hot and very cold days. That correction is a shape the convex model could carry. The convex
  model's only temperature terms are one-sided heating hinges at 15.5 degrees Celsius, so it has no
  term for cooling load and no extra slope in the cold tail. Extra hinge columns in the tails of
  the site's own temperature range, with a weaker ridge penalty on those columns, recover at least
  half of the network's gain without a network.
prediction: >-
  Written before any score is read. (1) F1 (the tail columns) or F2 (F1 plus the 72-hour thermal
  lag terms) lowers the quantile score at the five decision substations in the scored year by at
  least 0.05 points against arm D, and lowers it in the replicate year as well (the replicate year
  has no stored network, so the share of the network's gain is read in the scored year only).
  Experiment 209's temperature-extremes lift on a binned table gained under 0.001 points, so the
  more likely outcome is a gain of 0.01 to 0.04, with F2 no better than F1 by more than 0.01. (2)
  The convex tensor surface S explains at least 70% of the variance of the network's correction on
  held-out months. (3) More than half of the network's gain sits on rows whose temperature lies in
  the outer 10% of the site's range (the outer deciles 1 and 10), against 20% of rows by
  construction.
decision_rule: >-
  C1 is F1 minus R0 and C2 is F2 minus R0, each the paired difference in the quantile score over
  leads 0.5 to 168 hours at the five decision substations (D5, D10, D11, D14, D16; D9 beside them
  and in no contrast), read with the 12 calendar months of the year as the resampling unit (20,000
  resamples, 95% level, and the 97.5% level that two planned contrasts call for under a Bonferroni
  correction). The convex route is supported (prediction 1 passes) when F1 or F2 has the point
  estimate at or below -0.05 and an interval wholly below zero in the scored year, and a point
  estimate below zero in the replicate year. It is partly supported when the interval is wholly
  below zero in the scored year but the gain is under 0.05 or the replicate year disagrees. If
  predictions 1 and 2 both fail, the network uses an input that arm D lacks, and the write-up says
  which of calendar, lagged errors, and holiday flags the stored network arms add. The threshold of
  0.05 is the brief's, half of the network's gain. A gain of 0.02 would already pay for the 32
  columns the tail terms add per substation, because they ride on the existing fit and add no
  trained component to the serving path, so the write-up states the verdict at 0.05 and at 0.02. The
  known-answer gate runs first (below) and caps the verdict at `inconclusive` if it fails.
benchmark: >-
  R0, arm D as stored by experiment 155 (2.756 points at the five decision substations in the scored
  year, to be reproduced within 0.005 from the stored rows); the stored network (arm 2 of
  experiment 201, 2.587 against 2.691 at six substations); experiment 209's binned table
row_set_id: >-
  fold_a_daily_gap_v1: experiment 155's daily runs of the scored year 2025-07-02 to 2026-06-29 (292
  runs) outside switching events, and the even months of 2024-07 to 2025-06 for the replicate year
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles) in points of effective capacity,
  leads 0.5 to 168 hours; coverage and pinball loss at q01, q05, q95, q99
code: ml-experiments/00216-dp-network-shape-on-arm-d-flexible-temperature-tails/
---

**Status: planned. No score has been read.** The sections below are the plan, written before any
fit.

This experiment asks whether the correction that a small neural network gives our best day-ahead
convex forecast is a shape the convex model can express, or an interaction its inputs cannot.
[Experiments 200 and 201](../00201-dp-gam-residual-mlp/result.md) found that the network lowers the
quantile score by 0.09 to 0.125 points, mainly on very hot and very cold days.
[Experiment 209](../00209-dp-pinball-table-vs-network-head-on-arm-d/result.md) found that a binned
table recovers only 0.035 to 0.041 points. The plan refits the convex model with extra temperature
terms at the temperature extremes and compares it with arm D and with the network.

## Design

- **Sites.** Six substations that have arm D: D5, D9, D10, D11, D14, and D16. Contrasts use the
  five decision substations. The other 10 Primary substations have no pinned solar capacity vintage
  or ensemble frames and are not tried.
- **Schedule.** Monthly refit on the last 24 months, as in
  [experiment 155](../00155-dp-bg-stage2-assembly-night-offset-and-error-correction/result.md),
  with the same pinned solar capacity, the T-all interaction columns, the pooled filter noise
  setting, the ensemble-mean weather, the winter night offset, recipe R, and the spread refitted on
  the tuning months. The scored year, tuning months (odd months of 2024-07 to 2025-06) and replicate
  months (even months) are experiment 155's. The write-up says that monthly refits were used and
  that the contrast could depend on the cadence.
- **R0.** Arm D's stored rows, rescored. A control rebuilds arm D with the new code and no tail
  columns and must match R0's score within 0.005.
- **F1, flexible tails.** The model's heating block is `relu(15.5 - T)` on the temperature
  averaged over 6, 24, 72, and 168 hours, one slope per hour band. F1 adds, for the 6-hour and
  24-hour averages and each of the 4 hour bands, two cold hinges `relu(c - T)` with knots `c` at the
  site's 5th and 1st percentiles of the training window's averaged temperature, and two hot hinges
  `relu(T - h)` with knots `h` at its 95th and 99th percentiles. That is 32 columns, each with a
  non-negative slope. Each hinge's slope is an increment of the curve's slope, so a ridge penalty on
  the tail columns is a second-difference penalty on the temperature curve. The tail columns carry
  a ridge `lambda_tail` times the base ridge of 0.1, with `lambda_tail` chosen from {1, 0.1, 0.01}.
- **F2, thermal lag.** F1 plus the same tail hinges on the 72-hour average (16 more columns) and an
  unconstrained linear term in the 72-hour average per hour band (4 more columns), with the same
  tail ridge. The 72-hour average already enters the heating hinge at 15.5 degrees Celsius; the
  linear term lets the curve slope on both sides of it.
- **Choosing `lambda_tail` (tuning months only).** For each of the three values, refit only the
  monthly cuts that serve the tuning months (2024-06 to 2025-05), forecast the tuning-month runs
  from the ensemble-mean weather without the filter, and take the value with the lowest mean
  absolute error of the mean forecast over leads 0.5 to 168 hours at the six substations. A value
  within 0.5% of the lowest error loses to the larger (stronger) penalty. The same value serves F2.
  The knots are fixed above, not tuned.
- **S, explaining the network.** The target is the network's median correction (arm 2 of
  experiment 201 minus arm D at q50, averaged over the three seeds) on the scored year's rows. S
  is a convex surface over ensemble-mean temperature (knots every 2 degrees Celsius from -6 to 26),
  hour band (4), and day type (weekday, Saturday, Sunday and bank holiday), with a second-difference
  penalty along temperature and a ridge, no sign constraint. Its explained variance is measured on
  held-out months by leave-one-month-out folds across the 12 months of the scored year (the
  network's correction is stored for the scored year only). A second S adds arm D's filter level
  and anomaly as inputs, to see whether the correction needs them.
- **Decile analysis.** Each row's temperature decile is the decile of its 24-hour-averaged
  ensemble-mean temperature among the site's training-window values. The gain by decile is reported
  for N, F1, and F2 against R0.

## Gate (before any contrast is read)

On real rows at each substation, at the monthly cuts of the tuning months, multiply the target by
1.03 on training rows whose 24-hour-averaged temperature is above the training window's 95th
percentile, and refit F1 and arm D on that altered target. F1 passes at a substation when its fitted
values on the injected rows rise by at least 50% of 3% of the target, averaged over the rows, more
than arm D's do. The same test runs on held-out months of the same cuts, and with the injection in
the cold tail (below the 5th percentile). If F1 recovers under 50% of the hot-tail injection at 4 of
the 6 substations, the tail basis cannot express the shape and the verdict is `inconclusive`.

## Scoring

- Month-resampled paired bootstrap through `studies.bootstrap.bootstrap_difference_at_level`, 20,000
  resamples, at 95% and 97.5%.
- Quantile score over the 11 quantiles, absolute for every arm, plus the error of the mean
  forecast beside each contrast and the median minus the mean forecast per substation.
- Tail calibration at q01, q05, q95, and q99 (coverage and pinball loss,
  `studies.tail_calibration`). Curtailment cost is not scored, because the metric needs
  node-plus-generator frames that arm D does not have.

## Figures

1. Gain against R0 by temperature decile, for N, F1, and F2, at the five decision substations.
2. Headline: absolute quantile score and the contrasts with intervals, scored year and replicate
   year.
3. The fitted temperature curve at two substations (heating response at a midday weekday hour)
   with and without the tail columns.
4. S's out-of-sample fit: the network's correction against S's, by month, and the explained
   variance by temperature decile.

## Size and prior advice

- **Stored:** private files under `data/ml-experiments/00216-...` only.
- **Serving path:** not touched.
- **Degradation rule:** not touched.
- **Designs:** more than one is defensible; the grid is small and chosen on the tuning months.
- **Callers:** none outside this folder.

The plan follows earlier advice to test extra heating and cooling terms before using a network
([experiment 201](../00201-dp-gam-residual-mlp/result.md)), and departs from experiment 209's
finding that a temperature-extremes median lift gained under 0.001 points: that term acted on a
binned table of the recalibration, whereas these terms act inside the fit and so reshape the
Kalman filter's residuals as well.
