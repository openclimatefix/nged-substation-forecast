---
id: '00169'
slug: calibration-combined-fixes-and-ensemble-spread
part: CAL-3
theme: calibration-and-uncertainty
also_relevant_to:
- production-reference-and-best-guess
- curtailment-cost
status: planned
size: >-
  simple. Stored: private files under
  data/ml-experiments/00169-calibration-combined-fixes-and-ensemble-spread/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: the arms and the spread model are pre-registered
  below, so no choice of design is left open. Callers: this folder's scripts, which read experiment
  164's stored forecasts and experiment 111's stored ensemble temperature and shortwave radiation
  and call the reviewed `studies.*` modules without changing them. No refit of the mean forecast.
hypothesis: >-
  The spread across the 51 ensemble members of the temperature and of the downward shortwave
  radiation at a substation predicts how wide the forecast's residuals are, beyond the hour band,
  calendar quarter, lead, and the cold-day widening of experiment 164. A convex quantile model that
  scales the quantile offsets by a non-decreasing function of those two spreads lifts the
  cold-night q95 coverage of the combined forecast to 0.90 or more.
prediction: >-
  Written before any score is read. The combined forecast (experiment 160's interpolated table plus
  experiment 164's one-sided cold-day widening) reproduces experiment 164's fix O: cold-night q95
  coverage 0.896 and all-row score 2.7519 in the scored year. The spread model lifts cold-night q95
  coverage by 0.00 to 0.03 (to 0.90 or more with probability about 0.4), because experiment 164's
  scaling by the temperature spread alone moved it only from 0.846 to 0.865. The all-row score
  changes by -0.01 to +0.005 points. The fitted coefficients on shortwave spread are larger than on
  temperature spread in the summer quarters and near zero in winter.
decision_rule: >-
  The reference is the combined forecast (arm A: experiment 164's fix O applied to experiment 160's
  arm I), read from experiment 164's stored forecasts. The contrasts are C1 (arm A minus
  experiment 160's arm I, the assembly) and C2 (arm SP, arm A with the spread model, minus arm A),
  in quantile score over leads 0.5 to 168 hours on all rows at the five decision substations D5,
  D10, D11, D14, and D16 in the scored year (D9 beside them, in no contrast), read as a paired
  month-resampled bootstrap with 20,000 resamples at the 95% level (two contrasts, one of them
  descriptive, so no correction). Arm SP passes when its cold-night q95 coverage, pooled over the
  five substations in the scored year, is 0.90 or more and the upper bound of C2 is at or under
  +0.01 points. Cold nights are winter months (November to March), the night and evening hour
  bands, and an ERA5 day mean below 2 °C, the definition of experiments 160 and 164. The hypothesis
  is `supported` when SP passes on both the scored year and the replicate months, `partly supported`
  when SP passes on the scored year only, `refuted` when SP's cold-night q95 coverage exceeds arm
  A's by less than 0.01 in the scored year, and `inconclusive` otherwise. The threshold is set by the
  effort of porting: the spread model adds 8 quantile levels times 9 fitted coefficients, stored
  beside the spread table, and two ensemble columns that the weather pipeline already computes, so
  a change whose upper bound stays under 0.01 points (under 0.5% of the score) is worth porting.
  Every coefficient is fitted on the tuning months and nothing is fitted on the replicate months or
  the scored year. The verdict is capped at `inconclusive` if arm A rescored from experiment 164's
  stored forecasts differs from experiment 164's printed fix O score of 2.7519 by more than 0.001
  points.
benchmark: >-
  Arm A (experiment 164's fix O, 2.7519 points over leads 0.5 to 168 hours in the scored year),
  experiment 160's arm I (2.7529), BG2, climatology, and manual_heuristic as stored by experiment
  87, for absolute skill
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over the 11 quantiles of experiment 87) in points of
  effective capacity, on rows outside the hand-filtered switching events, over leads 0.5 to 168
  hours; coverage and pinball loss at the 0.01, 0.05, 0.95, and 0.99 quantiles by lead day, hour
  band, on cold nights, on bank holidays, and over Christmas; and a scaled-limit curtailment-cost
  proxy
code: ml-experiments/00169-calibration-combined-fixes-and-ensemble-spread/
---

# Experiment 169: combine the calibration fixes and test the ensemble spread

Planned; the write-up follows once the scripts have run. See the front matter for the hypothesis,
prediction, and decision rule.

## Plan

- **Arms.** I: experiment 160's interpolated spread table (the reference of experiment 164). A: I
  with experiment 164's one-sided widening of q75 to q99 on forecast-cold days (fix O). AT: A with
  the lower-tail multipliers of experiment 164's fix T (q01, q05, q10 by temperature class), to
  test whether T adds to O. SP: A with the spread model below.
- **Spread model.** For each of the levels 0.01, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, and 0.99, the
  row's offset from the mean forecast is multiplied by `c + sum_k a_k * max(z1 - k1, 0) + sum_k b_k
  * max(z2 - k2, 0)`, where z1 is the standard deviation across members of the 24-hour mean
  temperature and z2 the same for shortwave radiation, the knots are the 25th, 50th, 75th, and 90th
  percentiles of the tuning rows, and the coefficients a and b are non-negative. The pinball loss
  is linear in these coefficients, so each level is a linear programme fitted on the tuning months
  (solved with HiGHS), on at most 150,000 rows.
- **Not available.** Experiment 111's store holds temperature and shortwave radiation for the 51
  members, not wind, so the wind spread and the spread of the GAM-K forecast across members are not
  tested.
- **Size triggers.** Stored: private files only. Serving path: untouched. Degradation rule:
  untouched. Designs: pre-registered. Callers: this folder only.
