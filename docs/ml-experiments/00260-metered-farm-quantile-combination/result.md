---
id: '00260'
slug: metered-farm-quantile-combination
title: >-
  Does combining three solar-farm forecasters per quantile level beat the best one alone?
theme: metered-solar-farms
also_relevant_to:
- calibration-and-uncertainty
status: planned
size: >-
  medium. Stored: combined predictions and tables under
  data/ml-experiments/00260-metered-farm-quantile-combination/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one defensible way to fit the weights (squared
  error or pinball loss, rolling origin or leave-one-month-out) and to adjust the tails, so each is
  fixed in this plan before any scoring. Callers: this folder's scripts, copied from experiment 243's
  scorer, and the unchanged studies.tail_calibration. Asset graph: not touched.
hypothesis: >-
  The three best solar-farm forecasters of experiment 243 win at different quantile levels (the
  neural network without physics at 0.99, H5 at 0.01), so convex per-level weights fitted on
  earlier months beat H5 alone, and a rolling adjustment of the outer levels brings their coverage
  to the nominal level without hurting the quantile score.
prediction: >-
  The per-level combination (arm C2) beats H5 by at least 1.5% in quantile score, with a 95%
  month-block interval wholly below zero, and its pinball loss at the 0.99 quantile is within 5% of
  the neural network's. The rolling tail adjustment (arm C3) brings coverage at the 0.99 quantile
  to between 0.985 and 0.995 and raises the quantile score by no more than 0.2%.
decision_rule: >-
  Quantile score (twice the mean pinball loss over 11 levels, in points of capacity), equal weight
  per farm, on the 41,654 daylight half-hours of experiment 243, 12 to 36 hours ahead, 2025-09 to
  2026-08, with intervals from resampling the 12 scored months 20,000 times. C1: C2 minus H5,
  quantile score, own history. C2: C2 minus the neural network, pinball loss at 0.99. C3: C3 minus
  C2, quantile score and coverage at 0.99. C4: C2 minus H5, quantile score, zero history. Supported
  when C1 is at least 1.5% below zero with a 95% interval wholly below zero and C2 is under +5%.
  The 1.5% threshold is the cost of serving and retraining two forecasters beside one. Stop rule:
  if C1 is within 0.5% of zero, report that combining gains nothing detectable and do not score C3.
  Intervals are 95% as the prediction states, and 99% intervals (Bonferroni over four contrasts)
  are reported beside them.
benchmark: >-
  H5, the neural network without physics, XGBoost, and the plant alone (experiment 243, seed means,
  own and zero history), and the two floors of experiment 224
row_set_id: >-
  Experiment 243's rows: daylight half-hours (sun more than 5 degrees above the horizon) at 12 to
  36 hour leads, 2025-09 to 2026-08, farms A to F
metric: >-
  Quantile score, MAE of the median, and pinball loss and coverage at 0.01, 0.05, 0.95, and 0.99,
  per farm and arm, with the fitted weights per level
code: ml-experiments/00260-metered-farm-quantile-combination/
---

**This page is the plan; the write-up replaces it when the experiment finishes.**

## Question

**We ask whether a convex blend of three stored forecasters, with weights chosen per quantile
level, beats the best single forecaster, and whether a rolling adjustment can fix the outer
levels.** [Experiment 243](../00243-metered-farm-architecture-comparison/result.md) stored the
forecasts of H5 (the plant, a solar-farm power model, plus a neural network that outputs 11
quantile offsets), the neural network without the plant, and XGBoost. At the 0.99 quantile the
neural network without the plant scores 0.328 pinball loss, H5 0.344, and XGBoost 0.485. At the
0.01 quantile H5 scores best, with 0.334. No new fit is needed.

**Honesty note: an equal-weight average of the three forecasters' quantiles was scored once
before this plan was written.** It scored about 1.4% below H5 on the stored seed means, with
capacity taken as each farm's 99.5th percentile of actual output. That arm (C1) has been seen and
is reported as exploratory. The pre-registered arm is C2.

## Arms

- **Singles:** H5, the neural network without the plant, and XGBoost alone (experiment 243, seed
  means, own history).
- **C1 (exploratory, already seen):** equal-weight average of the three forecasters' quantiles.
- **C2:** at each of the 11 levels, weights on a simplex (non-negative, summing to 1) that
  minimise the pinball loss of the weighted sum of the three forecasters' quantiles, fitted by
  linear programming on the scored rows of the months before the scored month (rolling origin).
  The first 3 scored months are burn-in, scored with equal weights, and the page says so. The
  combined quantiles are sorted.
- **C2-LOMO (secondary):** weights fitted on every scored month except the scored month. These
  weights use later months, which do not exist at forecast time, so the page labels the arm
  optimistic.
- **C3:** C2 plus an adaptive conformal adjustment ([Gibbs and Candès
  (2021)](https://arxiv.org/abs/2106.00170)) of the 0.01, 0.05, 0.95, and 0.99 levels, with one
  offset (in fractions of capacity) per lead band and clear-sky band. Before each forecast run, the
  offset of each group moves by 0.5 times the group's mean miss rate minus the nominal rate, over
  the rows whose valid time is before the run's initialisation time, so only observed actuals are
  used. The step size of 0.5 is fixed here before scoring.
- **Zero history:** C2 on the zero-history predictions of experiment 243.

## Design

- **Rows:** experiment 243's 41,654 daylight half-hours; the three forecasters' predictions are
  already stored for those rows, so no model is refitted.
- **Capacity:** experiment 243's definition, so predictions stay in fractions of capacity.
- **Fit rows:** a month's weights use only the earlier scored months of the same setting. The
  burn-in months never fit.
- **Scoring:** copied from experiment 243's scorer, farm-equal means and month-block intervals;
  `studies.tail_calibration` reports coverage and pinball loss at 0.01, 0.05, 0.95, and 0.99.
- **Lower tail:** the page splits the misses at the 0.01 level by actual output under 3% of
  capacity, and says whether the adjustment can repair misses that come from outages.
- **Weather and forecast time:** as experiment 243, with the ECMWF ensemble 00 UTC run at the
  nearest cell and farm power one or two days earlier, all available at forecast time.
- **Anonymisation:** farms A to F only.

## Earlier advice followed

- Experiment 243: prefer H5 or the neural network without the plant to XGBoost for the 0.99
  quantile, so the weights are expected to lean on them at the top level.
- Experiment 257: a linear fit on the plant forecast, the ensemble spread, and the lagged power ties
  H5, so the combination is compared against H5 as in experiment 243 and not against the plant
  alone.
