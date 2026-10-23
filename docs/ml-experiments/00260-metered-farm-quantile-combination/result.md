---
id: '00260'
slug: metered-farm-quantile-combination
title: >-
  Does combining three solar-farm forecasters per quantile level beat the best one alone?
theme: metered-solar-farms
also_relevant_to:
- calibration-and-uncertainty
status: finished
size: >-
  medium. Stored: combined predictions and tables under
  data/ml-experiments/00260-metered-farm-quantile-combination/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one defensible way to fit the weights (squared
  error or pinball loss, rolling origin or leave-one-month-out) and to adjust the tails, so each is
  fixed in this plan before any scoring. Callers: this folder's scripts, copied from experiment
  243's
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
verdict: partly supported
verdict_note: >-
  The per-level combination C2 beats H5 by 2.13% in quantile score (95% interval -2.81% to -1.59%),
  past the 1.5% threshold, and its 0.99 pinball loss is 2.1% below the neural network's. The tail
  adjustment C3 fails its prediction: 0.99 coverage rises to 0.9951 (the band ends at 0.995) but the
  quantile score worsens by 1.18% and the 0.99 pinball loss by 26.5%.
best_arm: >-
  C2, the per-level pinball-loss weights fitted on earlier months, with a quantile score of 5.358
  points of capacity (95% interval 4.82 to 5.82) against H5's 5.475 (4.95 to 5.93), own history
headline: >-
  Weighting H5, the neural network, and XGBoost per quantile level beats H5 by 2.1% (95% interval
  1.6% to 2.8%), and the tail adjustment makes the forecast worse.
take_home_message: >-
  Blending the three stored forecasters with a separate weight at each quantile level scores 2.1%
  better than H5 alone, with every one of the six farms improving, because the neural network is
  best at the top quantile and H5 is best at the bottom one. The rolling adjustment of the outer
  quantiles did not help: it pushed the 0.99 quantile slightly too high and raised the error.
  Misses at the 0.01 quantile come from outages at farm D that no adjustment can predict.
production_status: candidate
adopted_in: none
advice:
  - >-
    Blend H5, the neural network without the plant, and XGBoost with a separate weight at each
    quantile level before choosing one forecaster, because the blend scores 2.1% better than H5
    (C1, Figure 2).
  - >-
    Weight the neural network at the 0.95 and 0.99 quantiles and H5 at the 0.01 quantile,
    because the fitted weights lean that way with own history (Figure 4).
  - >-
    Do not add an adaptive conformal adjustment of the 0.99 quantile to a forecast whose 0.99
    coverage is already 0.988, because it raised the 0.99 pinball loss by 26.5% (C3, Figure 3).
  - >-
    Treat farm D's 0.01 misses as outages to model separately, because 12% of its half-hours fall
    below the 0.01 quantile for every arm, including the adjusted one (Numbers).
---
**Weighting the three stored forecasters separately at each quantile level scores 2.1% better than
H5 alone, and the rolling tail adjustment makes the forecast worse.** National Grid Electricity
Distribution (NGED) meters six solar farms, labelled A to F. [Experiment
243](../00243-metered-farm-architecture-comparison/result.md) stored the forecasts of H5 (the plant,
a physical solar-farm power model, plus a neural network that outputs 11 quantile offsets), the
neural network without the plant, and XGBoost. Arm C2 weights the three at each of 11 quantile
levels, fitting the weights on earlier months by minimising pinball loss. With own history (fitted
on each farm's own earlier months), C2's quantile score (twice the mean pinball loss over the 11
levels, in points of capacity; lower is better) is 5.358 against 5.475 for H5. C2 is 2.13% better
(95% interval -2.81% to -1.59%; 99% interval -3.11% to -1.43%), and its 0.99 pinball loss is within
5% of the neural network's (2.1% lower, 95% interval -4.2% to +0.5%). **The first prediction
holds.** **The second prediction fails.** The rolling adjustment of the 0.01, 0.05, 0.95, and 0.99
quantiles (C3) lifts coverage at the 0.99 quantile from 0.9877 to 0.9951, just past the 0.995 limit
of the predicted band, and raises the quantile score by 1.18% (95% interval +0.88% to +1.46%)
against the 0.2% allowed. The hypothesis is partly supported. **The neural network carries the top
quantiles and H5 the bottom one.** The fitted weights give the neural network 0.8 of the weight at
the 0.99 quantile and 0.7 at the 0.95 quantile, and H5 almost all of the 0.01 quantile. With zero
history (fitted on the other five farms), C2 is 1.67% better than H5 (95% interval -2.19% to
-1.16%).

![Figure 1: quantile score per arm](figure_1_headline.svg)

![Figure 2: planned contrasts](figure_2_contrasts.svg)

![Figure 3: pinball loss at the outer quantiles](figure_3_tails.svg)

![Figure 4: weights per quantile level](figure_4_weights.svg)

## Numbers

- **Quantile score** (points of capacity, mean over six farms, own history): C2-LOMO 5.349, C2
  5.358, C1 5.370, C3 5.421, H5 5.475, XGBoost 5.490, neural network 5.541, plant 5.726, climatology
  8.638, manual heuristic 8.687. Zero history: C2-LOMO 5.508, C2 5.517, C1 5.522, XGBoost 5.610, H5
  5.610, neural network 5.650, plant 5.871.
- **Burn-in.** The first three scored months (2025-09 to 2025-11) are scored with equal weights, so
  C2 is C1 there. C2-LOMO uses later months and cannot exist at forecast time, so it is an
  optimistic reference, and it scores only 0.16% better than C2.
- **Exploratory arms** (not pre-registered): C1, equal weights, is 1.91% better than H5 (95%
  interval -2.52% to -1.42%), seen before the plan and now scored on experiment 243's capacity. C2
  is 2.40% better than XGBoost (95% interval -2.89% to -1.80%).
- **Per farm** (own history, quantile score, H5 then C2): A 5.277 and 5.220, B 5.252 and 5.138, C
  4.903 and 4.777, D 6.011 and 5.868, E 6.351 and 6.214, F 5.055 and 4.931. C2 is better at all six
  farms.
- **Tail calibration, own history** (coverage of the 0.01 and 0.99 quantiles, then pinball loss at
  0.01 and 0.99): H5 0.033 and 0.985, 0.334 and 0.344; neural network 0.048 and 0.980, 0.400 and
  0.328; XGBoost 0.038 and 0.994, 0.369 and 0.485; C2 0.033 and 0.988, 0.345 and 0.321; C3 0.033 and
  0.995, 0.338 and 0.406.
- **Lower-tail misses.** Of the 3.3% of C2's half-hours below the 0.01 quantile, 93% have actual
  output under 3% of capacity (H5: 92%). Farm D alone has 12% of its half-hours below the 0.01
  quantile for H5, C2, and C3 alike, so the cause is output that sits at zero while the forecast
  expects sun. The adjustment leaves the 0.01 coverage unchanged at 0.033 and cannot repair those
  misses.

## Checks

- **Median beside the contrasts.** The median error (mean absolute error, MAE, of the median) is
  9.72 for C2 and 9.85 for H5, so the gain in quantile score is not carried by the median alone: the
  0.99 pinball loss is 6.6% lower (95% interval -10.9% to -1.8%) and the 0.01 pinball loss 3.2%
  higher (95% interval -0.4% to +9.7%).
- **Triggers.** The prediction for C3 fails, so the plan's audit trigger for a contradicted
  prediction fires for the second prediction only. C2 gives more than the 1.5% threshold, so the
  stop rule does not apply.
- **Departures from the plan.** The weights are found by a grid search of the simplex to 0.002 in
  each weight, because the linear programme of the plan took tens of seconds per fit. A linear
  programme on one fit (the 0.5 level, six months) gave weights within 0.002 of the grid's. The MAE
  of the mean forecast is not reported, because the combined arms output quantiles only.

## What the next experiment should answer

**Whether a conformal adjustment that moves only the 0.99 quantile, by a smaller step, closes the
gap between C2's 0.99 coverage of 0.988 and the nominal 0.99, and whether farm D's outages can be
forecast from its recent output.** C3's step of 0.5 overshoots, and the lower-tail misses of farm D
are the largest remaining tail error.
