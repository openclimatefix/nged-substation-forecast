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
  correction, which decides the verdict; the replicate year has only 6 resampling units, the even
  months, and its median monthly difference must be negative too). The convex route is supported
  (prediction 1 passes) when F1 or F2 has the point
  estimate at or below the larger of -0.05 and minus half of the network's measured gain at the
  same substations (reported with its interval), and an interval wholly below zero in the scored
  year, and a point
  estimate below zero in the replicate year. It is partly supported when the interval is wholly
  below zero in the scored year but the gain is under 0.05 or the replicate year disagrees. If
  predictions 1 and 2 both fail, the network uses an input that arm D lacks, and the write-up says
  which of calendar, lagged errors, and holiday flags the stored network arms add. The threshold of
  0.05
  is the brief's, half of the network's gain; the replicate year has no stored network, so that
  year is read against arm D only, which departs from the brief. A gain of 0.02 would already pay
  for the 32
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
- **R0 and the rebuilt control.** R0 is arm D's stored rows, rescored. The control rebuilds arm D
  with the new code and no tail columns, and must match R0's score within 0.005. The contrasts C1
  and C2 are taken against the rebuilt control, so a difference between the stored pipeline and the
  new code cannot enter them; the contrast against R0 is also reported. Every arm refits the winter
  night offset, recipe R, the filter, and the spread on its own residuals.
- **F1, flexible tails.** The model's heating block is `relu(15.5 - T)` on the temperature
  averaged over 6, 24, 72, and 168 hours, one slope per hour band. F1 adds, for the 6-hour and
  24-hour averages and each of the 4 hour bands, two cold hinges `relu(c - T)` with knots `c` at the
  5th and 1st percentiles of the training rows' averaged temperature (as the fit's weather frame
  holds it, per smoothing window, recomputed at each monthly cut), and two hot hinges
  `relu(T - h)` with knots `h` at its 95th and 99th percentiles. That is 32 columns, each with a
  non-negative slope. Each hinge's slope is an increment of the curve's slope, so a ridge penalty on
  the tail columns is a second-difference penalty on the temperature curve. The tail columns carry
  an absolute ridge `lambda_tail` times the base ridge of 0.1 (the columns are not split by day type
  or half-band, and use the base columns' scale of 10 degrees Celsius), with `lambda_tail` chosen
  from {1, 0.1, 0.01}.
- **F2, thermal lag.** F1 plus the same tail hinges on the 72-hour average (16 more columns) and an
  unconstrained linear term in the 72-hour average per hour band (4 more columns), with the same
  tail ridge. The 72-hour average already enters the heating hinge at 15.5 degrees Celsius; the
  linear term lets the curve slope on both sides of it.
- **Choosing `lambda_tail` (tuning months only).** For each of the three values, refit the monthly
  cuts (the stored cuts of experiment 111 run from 2024-04, so the tuning cuts exist at all six
  substations), run the pipeline's pooled filter, and forecast the tuning-month runs. The criterion
  is the mean absolute error of the filtered mean forecast over leads 0.5 to 168 hours on tuning-
  month rows in the outer two deciles of the site's training temperature (deciles 1 and 10 of the
  24-hour average), pooled over the six substations, because the tail columns act on about 5% of
  rows and an all-row error is nearly flat in `lambda_tail`. A larger penalty wins only if its error
  is lower by more than 0.5%; otherwise the smaller penalty wins. The error on all tuning rows is
  reported for all three values. If the tuning months hold fewer than 200 outer-decile rows at a
  substation, that substation drops out of the criterion. The same value serves F2. The knots are
  fixed above, not tuned.
- **S, explaining the network.** The target is the network's median correction (arm 2 of experiment
  201 minus arm D at q50, averaged over the three seeds) at the six substations pooled, on the
  scored year's rows, weighted so that each distinct valid time counts once. S is a convex surface
  over ensemble-mean temperature (knots every 2 degrees Celsius from -6 to 26), hour band (4), and
  day type (weekday, Saturday, Sunday and bank holiday), with a second-difference penalty along
  temperature and a ridge, and no sign constraint. Explained variance is 1 - SSE/SST. It is reported
  in sample, by leave-one-month-out over the 12 months of the scored year, and by temperature
  decile, with month-block standard errors. Prediction 2 is read on held-out rows inside the other
  11 months' temperature range; held-out rows outside that range are reported separately, because
  the hottest and coldest months hold most tail temperatures. Variants add one input at a time to
  see which the correction needs: arm D's filter level and anomaly, day of year, and the bank-
  holiday flags.
- **Decile analysis.** Each row's temperature decile is the decile of its 24-hour-averaged ensemble-
  mean temperature among the site's training-window values. The gain by decile is reported for N,
  F1, and F2 against R0, and the share of the net gain in deciles 1 and 10 is compared with both 20%
  (the share of rows) and the share of R0's total quantile score in those deciles, with a month-
  resampled interval. N is arm 2, which has calendar and holiday inputs that arm D lacks, so arm 1
  is read as well if its rows are stored. The seed-to-seed spread of N's gain is reported. Hot-hinge
  gain is also reported by hour band, because a non-negative hot hinge could absorb errors in the
  pinned solar output at sunny, hot midday hours, and the share of scored rows on which each tail
  hinge is active is reported per substation, because the ensemble mean is smoother than the weather
  the fit saw.

## Gate (before any contrast is read)

The gate runs at the chosen `lambda_tail`, for F1 and for F2, at each of the 6 substations and at
the monthly cuts of the tuning months. It is a known-answer test with an altered training target.

- **Injection.** On training rows whose 24-hour-averaged temperature is above the training window's
  95th percentile, add 3% of the site's mean absolute demand over the training window (net power
  plus the pinned solar output) to the target. An additive size is used because net power is near
  zero or negative at midday at solar-heavy substations, so a multiplier has no fixed meaning.
- **Recovery.** Refit F1 and the rebuilt arm D (T-all) on the altered target and on the unaltered
  target at the same cut. The recovery of a fit is the mean over the injected rows of its fitted
  value on the altered target minus its fitted value on the unaltered target, as a share of the
  injected amount. F1 passes at a substation when its recovery is at least 50%. Arm D's recovery is
  reported beside it, because extra flexibility alone could raise the fitted values.
- **Variants.** The same test runs with a ramp injection (zero at the 95th percentile, 3% at the
  99th), with the injection in the cold tail (below the 5th percentile), on held-out months (the
  next 3 months' predictions with and without the training injection), with a placebo injection in
  the middle half of the temperature range (the tail rows must show under 10% recovery), and with a
  zero injection (the difference must be under 1% of the injected amount). Only injections that
  raise demand are tested, because the tail slopes are non-negative.
- **Pass rule.** If F1's recovery of the hot-tail step injection is under 50% at 4 or more of the 6
  substations, or the placebo shows 10% or more, the tail basis cannot be read as evidence and the
  verdict is `inconclusive`.

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
