---
id: '00216'
slug: dp-network-shape-on-arm-d-flexible-temperature-tails
title: >-
  Differentiable physics: can a more flexible temperature curve in the convex model carry the
  network's correction?
theme: differentiable-physics-forecast-engine
also_relevant_to: [calibration-and-uncertainty]
status: done
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
date: 2026-10-22
verdict: inconclusive
best_arm: >-
  F2 (tail columns plus 72-hour terms): 2.7205 points against arm D's 2.7562 over leads 0.5 to 168
  hours at the five decision substations in the scored year (-0.036; 97.5% interval -0.074 to
  -0.007). The stored network scored 2.62 (-0.117 to -0.141 across 3 seeds).
production_status: not-for-production
adopted_in: none
headline: >-
  The tail columns recover 20% (F1) to 27% (F2) of the neural network's gain in the scored year and
  lose points in the replicate year, so the network's correction is not mainly a temperature-tail
  shape. A convex surface in temperature, hour band, and day type explains under 1% of the network's
  correction on held-out months.
take_home_message: >-
  Adding extra temperature terms for the hottest and coldest days to the convex demand model
  recovers only a fifth to a quarter of the neural network's improvement, and does not repeat in a
  second year. The network's correction is spread over all temperatures instead of sitting at the
  extremes, and a smooth function of temperature explains under 1% of it, so the network uses
  something other than a missing temperature curve. The formal verdict is capped at inconclusive
  because a placebo check narrowly failed, but the shortfall does not depend on that check.
lessons:
  - >-
    The network's gain is spread over the temperature deciles: deciles 1 and 10 hold 24.8% of it
    (95% interval 11.0% to 49.3%) against 23.5% of the rows, so the 'extremes' reading of
    experiment 201 does not hold for the quantile score (decile table, figure 2).
  - >-
    The tail columns raise the forecast by up to 7 points at 27 degrees Celsius, against 2 points
    for the network, so the hot-tail slope overshoots on the few hot rows (figure 3).
  - >-
    Weaker ridge penalties on the tail columns scored worse than the base ridge on the tuning
    months (error on tail rows 4.68 and 4.72 against 4.59), so the tail basis wants more
    regularisation, not less.
advice:
  - >-
    Do not expect extra temperature-tail terms in the convex demand model to replace the network:
    they recover 20% to 27% of its gain in the scored year and none in the replicate year (C1, C2,
    figure 1).
  - >-
    Look for the network's gain in calendar and slow-seasonal inputs: a surface with day of year
    explains 10.6% of its correction on held-out months, against 0.8% without it (arm S, figure
    4).
  - >-
    If a hot-tail term is tried again, cap its slope or use a stronger ridge, because the tuning-
    month grid preferred the base ridge and the fitted hot slope overshoots at 27 degrees Celsius
    (figure 3).
---

**Extra temperature terms recover only a fifth to a quarter of the neural network's improvement, and
the improvement does not repeat in a second year.** The network's correction is spread over all
temperatures instead of sitting at the extremes, and a smooth function of temperature explains
under 1% of it. The network therefore uses something other than a missing temperature curve, and
the next experiment should look at calendar and slow-seasonal inputs. The verdict is `inconclusive`
because the placebo check of the pre-registered known-answer test narrowly failed (10.2% against a
limit of under 10%); the shortfall against the network does not depend on that check.

This experiment asked whether the correction that a small neural network gives our best day-ahead
convex forecast (arm D of
[experiment 155](../00155-dp-bg-stage2-assembly-night-offset-and-error-correction/result.md)) is a
shape the convex model could carry. [Experiments 200 and 201](../00201-dp-gam-residual-mlp/result.md)
found that the network lowers the quantile score by 0.09 to 0.125 points, and
[experiment 209](../00209-dp-pinball-table-vs-network-head-on-arm-d/result.md) found that a binned
table recovers only 0.035 to 0.041. We refitted the convex model with extra hinge terms in the hot
and cold tails of each substation's temperature range, rebuilt arm D from the refits, and compared
the result with the stored network on the same rows. We then fitted a convex surface to the
network's correction to see how much of it a function of temperature, hour band, and day type can
express. The method of adding hinge terms to an additive model is standard; the novel step is only
the comparison with the network.

![Quantile score gain of the stored network and of the tail columns against arm D, in the scored
year and the replicate year, with 97.5% intervals.](figure_1_gain_against_arm_d.svg)

![Gain over arm D by temperature decile for the network and the tail columns, scored
year.](figure_2_gain_by_decile.svg)

![Mean shift of the median forecast against arm D by temperature, for the network and the tail
columns.](figure_3_shape_of_the_correction.svg)

![Share of the network's correction explained by a convex surface in temperature, with and without
extra inputs.](figure_4_convex_surface_explains_the_correction.svg)

## Results

- **Arms.** R0 is arm D as stored; the rebuilt control reproduces it exactly (2.7562 points in the
  scored year, all 464,363 rows). F1 adds 32 tail hinge columns per substation (cold knots at the
  5th and 1st percentiles and hot knots at the 95th and 99th percentiles of the training rows'
  6-hour and 24-hour averaged temperature). F2 adds the same on the 72-hour average plus an
  unconstrained linear term in it (20 more columns). The stored network is arm 2 of experiment 201,
  three seeds.
- **Schedule and rows.** Monthly refit on 24 months, as in experiment 155; six substations, with
  the five decision substations (D5, D10, D11, D14, D16) in the contrasts. A weekly refit might
  change the contrast, because the tail knots move at each cut.
- **Prediction 1 (convex route recovers half of the network's gain): failed.** In the scored year
  F1 scored 2.7293 (-0.027 against R0; 97.5% interval -0.052 to -0.007) and F2 2.7205 (-0.036;
  -0.074 to -0.007). The network scored 2.6197, 2.6153, and 2.6396 across seeds (-0.137, -0.141,
  -0.117 against R0), so F1 recovers 20% and F2 27% of the mean gain. In the replicate year (the
  even months of 2024-07 to 2025-06; the network is not stored there, so the comparison is with R0
  only) F1 was +0.007 (-0.006 to +0.023) and F2 +0.026 (-0.010 to +0.086).
- **Prediction 2 (convex surface explains 70% of the correction): failed.** Temperature by hour
  band by day type explains 9.9% of the network's median correction in sample and 0.8% on held-out
  months (leave-one-month-out; 0.3% to 0.8% across three smoothing weights). Adding arm D's filter
  state gives 0.9%, day of year 10.6%, and both 9.5%. Held-out rows inside the other months'
  temperature range are 99.97% of the rows, so the low score is not an extrapolation artefact.
- **Prediction 3 (gain sits in the outer deciles): failed.** Deciles 1 and 10 hold 24.8% of the
  network's gain (95% interval 11.0% to 49.3%), 28.2% of F1's (-33.7% to 64.7%) and 23.0% of F2's,
  against 23.5% of the rows and 22.3% of arm D's score. The network's gain peaks in deciles 3, 8,
  and 9.
- **Tuning.** The tail ridge ratio was chosen on the tuning months from {1, 0.1, 0.01}. The mean
  absolute error of the filtered mean forecast on outer-decile rows was 4.591, 4.680, and 4.721
  against 4.602 for arm D, so the base ridge (ratio 1) was chosen and the weaker penalties the
  brief asked for lost.
- **Where F1 gains.** The scored-year gain is at leads of 6 hours or more (-0.028; -0.054 to
  -0.007); under 6 hours it is -0.001 (-0.021 to +0.021). It is -0.015 to -0.035 at every
  substation, and no leave-one-out flips it (-0.025 to -0.030).
- **Tail calibration** (five decision substations, scored year; coverage and pinball loss):

  | Arm | q01 | q05 | q95 | q99 |
  |---|---|---|---|---|
  | R0 | 2.4%, 0.215 | 7.0%, 0.712 | 94.7%, 0.749 | 98.3%, 0.232 |
  | F1 | 2.3%, 0.213 | 6.9%, 0.706 | 94.9%, 0.745 | 98.4%, 0.232 |
  | F2 | 2.3%, 0.213 | 6.8%, 0.704 | 95.1%, 0.741 | 98.4%, 0.231 |

## Gate

- **Known-answer test (hot-tail step).** With 3% of the training window's mean absolute demand
  added to the target above the 95th percentile of temperature, F1 recovered 62% (F2 66%, arm D
  20%) on the injected rows and 55% (58%, 21%) on held-out months, at every one of the 6
  substations (59% to 66%). A ramp injection was recovered 68% and 66% against 11%; a cold-tail
  step 65% and 66% against 45%; a zero injection changed nothing.
- **Placebo.** An injection in the middle half of the temperature range changed F1's fitted values
  on hot-tail rows by 10.2% of the amount (F2 -0.8%, arm D 23.7%), so F1 narrowly fails the
  pre-registered limit of under 10%. Arm D leaks more than F1, so the leak is the smoothness of
  the model and not the tail columns. The verdict follows the pre-registered cap.

## Checks

- **Median minus mean forecast, per substation:** 0.000 at all six, because the stored quantiles
  centre on the corrected mean.
- **Error of the mean forecast beside each contrast:** F1 -0.053 (97.5% interval -0.108 to -0.012)
  and F2 -0.069 (-0.149 to -0.011) in the scored year, the same direction as the quantile score;
  in the replicate year +0.011 and +0.038, both with intervals that include zero.
- **Triggers fired:** trigger 4. In the replicate year F1's contrast is -0.017 without D10 and +0.009
  to +0.019 without any other substation, because D10 alone is +0.107. Triggers 1 (the placebo
  narrowly fails, which is a control failure) and 7 (the series of network experiments depends on
  the headline) are also arguable. An audit by a fresh Opus agent is requested from the
  manager before merge; its findings are not yet in.

## Caveats

- **The replicate year has 6 resampling units** (the even months), and the network is not stored
  there, so prediction 1's replicate clause compares the tail columns with arm D only.
- **Deciles use the analysis-frame 24-hour temperature at the valid time**, not the ensemble-mean
  temperature. The share of scored rows on which each tail hinge is active under the ensemble-mean
  weather was not computed.
- **The hot-hinge gain by hour band** is mixed: F1 gains 0.10 in the night band and 0.05 at midday
  but loses 0.04 in the evening band on hot rows, so some of the hot-tail gain may be solar error.

## What to try next

- **Test calendar and slow-seasonal inputs.** Day of year takes a convex surface from 0.8% to 10.6%
  of the network's correction on held-out months, which points at a seasonal or calendar effect
  that arm D's heating terms do not carry. A learned small network on day of year and hour, with
  the convex model as the comparison arm, is the ranked suggestion; it could fail if the gain
  vanishes in the replicate year.
- **If hot-tail terms are tried again,** cap the hot slope or use a stronger ridge.
