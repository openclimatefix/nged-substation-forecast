---
id: '00279'
slug: dp-heating-curve-monotone-lagged-temperature-and-wind
title: >-
  Does a smooth monotone heating curve, with thermal lag and wind, lower the demand forecast's error
  on cold nights and winter days?
theme: differentiable-physics-demand-calendar-and-weather
also_relevant_to:
  - production-reference-and-best-guess
status: planned
size: >-
  medium. Stored: private files under
  data/ml-experiments/00279-dp-heating-curve-monotone-lagged-temperature-and-wind/ only. Serving
  path: not touched. Degradation rule: not touched. Designs: more than one is defensible (basis
  knots, which lags, where the network sits), so the plan fixes one. Callers: this folder's scripts
  copy the fit, filter, and scoring code of experiments 126 and 250 and call reviewed `studies.*`
  modules unchanged. Asset graph: not touched.
hypothesis: >-
  The heating response of the convex demand model T-all-K (a hinge at 15.5 C on four trailing-mean
  temperatures, split by day type) is too rigid. A smooth, monotone, non-negative curve on the same
  temperatures (A1), the same curve also on exponentially weighted and previous-day mean
  temperatures that carry the building's thermal lag (A2), or a small monotone neural network on
  three daily mean temperatures and the wind speed (A3), lowers the day-ahead quantile score on
  cold nights and winter days at the five decision Primaries, and the fitted curve is not a
  straight line.
prediction: >-
  Written in the maintainer's brief before any score was read, with the intervals as tests that can
  fail. (P1) A1 minus T-all-K is within +/-0.02 points. (P2) A2 minus T-all-K is -0.01 to -0.04
  pooled and -0.05 to -0.15 on cold nights, with at least 4 of 5 sites better. (P3) A3 minus A2 is
  -0.01 to +0.01 pooled, and A3 beats A2 by 0.02 or more on cold nights only if the curve flattens
  above about 12 C and steepens below 3 C at 3 or more sites. (P4) Wind adds less than 0.01 pooled.
  Earlier experiments argue for ties: experiment 126 found its monotone temperature ramps (-0.007),
  exponentially weighted temperatures (-0.002), and wind chill (-0.026, interval -0.056 to +0.018)
  all within noise on winter rows, and experiment 121 found five heating hinges gained 0.016.
decision_rule: >-
  Paired month-resampled bootstrap on the scored year (2025-07 to 2026-06), 99.17% intervals
  (Bonferroni over the three planned contrasts C1 A1-A0, C2 A2-A0, C3 A3-A0). The arm is supported
  when its scored-year interval on its primary rows (cold nights for the cold-night contrast,
  November to March otherwise) is wholly below zero, its point estimate on the replicate year
  (2024-07 to 2025-06) is below zero, and at least 4 of 5 decision sites improve. The threshold is
  -0.02 points for A2 (two extra fitted weights per hour band) and -0.04 points for A3 (a network
  per site), because those arms add parameters or a trained component to the serving path; A1 has
  no threshold beyond an interval below zero, because it replaces columns with columns. A3 minus A2
  (C4) and A2w minus A2 (C5, the wind term) are secondary and read at the same level. The curve on
  the card changes only for an arm that passes.
benchmark: >-
  T-all-K (experiment 114's day-type split, tuned penalty weight 1.0, with the Kalman filter whose
  hyperparameters experiment 126 tuned for it), refitted here and checked against experiment 126's
  stored scored-year forecast
row_set_id: dp_mw_v1
metric: >-
  Quantile score (twice the mean pinball loss over the 11 quantiles) in points of effective
  capacity, day ahead, outside the hand-filtered events; cold nights are 22:00 to 08:00 local time
  on days whose mean temperature is at most 3 C; winter days are November to March
code: ml-experiments/00279-dp-heating-curve-monotone-lagged-temperature-and-wind/
---

## Plan

**Question.** The convex demand model fits heating as a straight line below 15.5 C. This
experiment asks whether a smooth monotone curve, a thermal lag, or a small monotone neural network
does better on the rows where the line is most likely to be wrong: cold nights and winter days.

**Arms.** All arms start from T-all (A0) and keep its day-type split, its lighting, and its
calendar terms.

- **A0 (T-all-K):** the hinge at 15.5 C on the trailing 6, 24, 72, and 168 hour mean temperatures.
- **A1:** A0 plus a smooth monotone curve (integrated quadratic B-splines with non-negative
  coefficients, six functions with knots at 0, 4, 8, 12, 16, and 20 C) on each of the four
  trailing means, per hour band.
- **A2:** A1 plus the same curve on exponentially weighted temperatures (time constants 1 and 3
  days) and on the previous two days' mean temperatures, per hour band. The non-negative
  coefficients are the fitted lag weights.
- **A2w:** A2 plus the 24-hour mean wind speed times the 24 and 72 hour hinge, per hour band.
- **A3:** A0's non-heating terms held fixed, and a monotone network (2 hidden layers of 16 units,
  non-negative weights on the temperature and wind paths, a free bias per hour band and day type)
  replacing the heating term. Inputs are the daily mean temperature of the last day and the two
  days before it, and the 24-hour mean wind speed. Three seeds are averaged.
- **A3 without wind:** the same network without the wind input, to isolate wind in the network.

**Data and schedule.** Row set `dp_mw_v1` on past weather (ERA5 temperature and wind speed, CAMS
irradiance), the same frozen yearly fits and Kalman filter as experiments 114, 121, and 126. Past
weather arrives about 5 days late, so every score is a ceiling for a forecast at the origin, and
the conditional ENS retest (`fold_a_daily_gap_v1`) only runs for an arm that passes. The filter
uses T-all's per-site hyperparameters for every arm, so the terms are the only difference. Yearly
refits, not weekly: contrasts of 0.02 points or more do not depend on the cadence.

## Earlier evidence read before planning

- Experiment 126: A-mono -0.007 (interval -0.062 to +0.032), S-ewma -0.002 (not powered), C-wind
  -0.026 (-0.056 to +0.018) on November to March, all labelled undecidable or tie.
- Experiment 121: five heating hinges -0.016 (-0.035 to +0.004).
- Experiment 114: T-all-K -0.039 against B0-K, about 38% of the gain on warm rows.
