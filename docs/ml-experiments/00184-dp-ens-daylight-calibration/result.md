---
id: '00184'
slug: dp-ens-daylight-calibration
part: CAL-4
theme: differentiable-physics-forecast-engine
also_relevant_to:
- calibration-and-uncertainty
- weather-products-and-blending
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00184-dp-ens-daylight-calibration/ only.
  Serving path: not touched. Degradation rule: not touched. Designs: more than one is defensible (a
  solar-only or all-weather member pass, a summary-statistic or a per-member route), and this plan
  pre-registers each as its own arm. Callers: this folder's scripts, which copy the member and
  weather code of experiments 111 and 136 and the scale-model code of experiment 169, and call the
  reviewed `studies.*` modules without changing them. No refit of any mean forecast.
hypothesis: >-
  Experiment 169 found that passing the ensemble's spread to the width of the forecast's bands adds
  nothing when scored over all half-hours. The benefit may be confined to daylight half-hours,
  especially sunny and partly cloudy ones, where the clouds that the 51 ensemble members disagree
  about change the output of embedded solar generation, and so the demand. Passing each member's
  irradiance through the convex model's solar term (arm M), or using the spread of the members'
  irradiance as a fraction of clear-sky to set the band width in daylight (arm S), calibrates the
  daylight bands better than the existing bands (arm A).
prediction: >-
  Written before any score is read. The prediction that can fail: arm M or arm S lowers the
  daylight quantile score by at least 1% against arm A at the five decision substations (leads 0.5
  to 168 hours, scored year), keeps the night quantile score within 0.5% of arm A, and moves the
  daylight q05-to-q95 coverage (nominal 0.90) closer to 0.90. My own expectation is that it fails
  (probability about 0.7): arm M changes the daylight score by -0.5% to +1.0%, because the members'
  irradiance varies more than the weather that the model was fitted on, and arm S changes it by
  -0.5% to 0.0%. If a gain exists it sits at leads under 72 hours and on the partly cloudy third of
  the daylight half-hours, and not on clear or overcast ones. The shuffled-spread placebo (arm SP)
  recovers more than half of arm S's gain if the gain is only a wider band. The daylight gap of
  arm A to BG2 (experiment 178, +0.05 MW on sunny rows, 63% at D5) changes by less than 0.01 MW.
decision_rule: >-
  The reference is arm A (experiment 164's fix O on experiment 160's table on experiment 155's arm
  D mean, as stored, the arm A of experiment 169). The contrasts are C1 (M minus A) and C2 (S minus
  A) in the quantile score (twice the mean pinball loss over the 11 quantiles of experiment 87) over
  daylight half-hours (solar elevation of 5 degrees or more at the valid time), leads 0.5 to 168
  hours, outside the hand-filtered switching events, at the five decision substations D5, D10,
  D11, D14, and D16 in the scored year (D9 beside them, in no contrast). Each contrast is read as a
  percentage change of the daylight score of arm A, with a paired month-resampled bootstrap (12
  calendar months, 20,000 resamples) at the 97.5% level (two planned contrasts, Bonferroni). An arm
  passes when the upper bound of its interval is under zero, its point estimate is at or under
  -1%, its night score (elevation under 0 degrees) is within +0.5% of arm A's, and its daylight
  q05-to-q95 coverage is closer to 0.90 than arm A's. The hypothesis is `supported` when an arm
  passes on both the scored year and the replicate months, `partly supported` when an arm passes on
  the scored year only, `small gain` when an arm's interval is wholly under zero but its point
  estimate is above -1%, `not supported` when no interval lies wholly under zero, and `inconclusive`
  when a known-answer gate fails. The threshold of 1% is set by cost. Arm M adds a 51-member pass
  of the solar term and four stored spread factors to the weather layer, and arm S adds a handful
  of stored coefficients and three columns computed from the ensemble, so neither is worth porting
  for a change that the interval cannot separate from zero, and a change under 1% is worth less than
  the weather-layer work arm M adds. Every coefficient and factor is fitted on the tuning months
  and nothing is fitted on the replicate months or the scored year. Four cuts are reported and
  none decides the verdict: lead bands (under 24, 24 to 72, 72 to 168 hours, and 12 to 36 hours),
  the clear-sky index band of the ensemble-mean irradiance (overcast under 0.4, partly cloudy 0.4 to
  0.7, clear 0.7 and over), the sunniest third as experiment 178 defines it, and D5 against the
  other four substations.
benchmark: >-
  Arm A (experiment 169's arm A, 2.7519 points over all rows in the scored year), BG2 as stored by
  experiment 87 for the daylight gap, climatology and manual_heuristic for absolute skill
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score in points of effective capacity over daylight and night half-hours, as a
  percentage change from arm A, with absolute pinball loss, q05 and q95 coverage, q05-to-q95
  interval width, and the tail pinball loss at 0.01, 0.05, 0.95, and 0.99
code: ml-experiments/00184-dp-ens-daylight-calibration/
---

# Experiment 184: does the ensemble help calibrate the bands in daylight?

This page is the plan. It holds the question, the arms, and the decision rule; the write-up replaces
it when the scores exist.

## Question

Experiment 169 showed no gain from the spread of the 51 ensemble members over all half-hours.
Cloud is the one weather quantity that the ensemble forecasts with real uncertainty, and cloud
changes demand only through embedded solar generation, which exists only in daylight. This
experiment scores daylight and night separately, so that a gain confined to daylight is not diluted
by the 50% of half-hours at night.

## Arms

All arms keep experiment 155's arm D mean for the centre of the forecast, and all start from arm
A's quantile offsets (the 11 quantiles minus the mean).

- **A:** the reference, as stored.
- **M:** each of the (up to) 51 members' irradiance goes through the convex model's solar term, and
  every other input stays at the ensemble mean. The fitted model is linear in its coefficients, so
  member m's forecast is arm A's mean plus the model's response to the member's unit PV shape
  minus its response to the ensemble mean's unit PV shape (experiment 136's T-all fit of the run's
  cut, the same fit that made the mean). The 11 quantile offsets of arm A, scaled by one factor per
  calibration lead band, go around each member, and the 51 members are pooled as in experiment
  111. Each factor is chosen on the tuning months' daylight rows from 0.5 to 1.3 in steps of 0.05.
  Mc (a control) centres the members' responses on zero, which isolates the spread from the shift
  in the mean that a nonlinear response produces.
- **F:** as M, but every weather input (temperature, smoothed temperature, and irradiance) comes from
  the member, the all-members full pass.
- **S:** daylight offsets are multiplied by `c + sum_k a_k max(z1 - k1_k, 0) + sum_k b_k max(z2 - k2_k, 0) + sum_k d_k max(z3 - k3_k, 0)`,
  fitted per quantile level by a linear programme on the pinball loss (the method of experiment
  169) on the tuning months' daylight rows. The statistics are z1, the standard deviation across
  members of the clear-sky index (calibrated irradiance over the Haurwitz clear-sky irradiance); z2,
  the range of that index across members; and z3, `4 p (1 - p)` where p is the share of members
  with a clear-sky index of 0.8 or more (0 when members agree, 1 when they split evenly). All
  coefficients are non-negative, so the width is non-decreasing in each statistic. Night offsets
  stay at arm A's.
- **SP:** arm S's fitted coefficients applied to statistics taken from another day of the same
  substation, calendar month, and hour band (fixed seed). If SP recovers the gain of S, the gain
  comes from a wider band, not from information.
- **K (optional, descriptive, outside the verdict):** the ensemble spread as the filter's
  observation variance in daylight, if the stored filter code allows it without a refit of the
  mean.

## Known-answer gates

A failed gate caps the verdict at `inconclusive`.

- **G1, reproduction.** Rebuilding the ensemble-mean-driven T-all forecast plus the filter's level
  and anomaly reproduces experiment 136's stored arm A2 mean to within 1e-3 points on every row of
  the scored year.
- **G2, unit shape.** The fleet's unit PV shape that `SunGeometry.pv_shape_and_ghi` builds from
  CAMS irradiance equals the shape in the cache that the model was fitted on, to a through-origin
  slope of 0.95 to 1.05 per substation on daylight tuning rows (the three-places trap of experiment
  180). The slope is reported by substation, D5 included.
- **G3, fake ensemble.** A fake ensemble of 51 members equal to CAMS irradiance times 51 equally
  spaced multipliers from 0.6 to 1.4, with the identity calibration, gives member responses that
  fall monotonically with the multiplier on every daylight row, that are exactly zero at night, and
  that are zero for the member with multiplier 1.0 (the mean). The statistics z1 and z2 of the fake
  ensemble equal the known multiplier spread to within 1%.
- **G4, raw against calibrated spread.** The standard deviation of the clear-sky index across
  members, before and after the calibration and solar shrink of experiment 111, is reported by
  clear-sky-index band, so a null cannot be read as "the ensemble has no information" if the
  calibration shrank the spread.
- **G5, night placebo.** Arms leave rows at night unchanged by design, and the night score is
  reported for every arm. Arm SP is the information placebo.

## Row definitions, fixed before any score

- **Daylight:** solar elevation of 5 degrees or more at the valid time (the cache's
  `elevation_deg`). **Night:** under 0 degrees. Rows between 0 and 5 degrees are scored and
  reported as twilight and are in neither contrast; every arm leaves them unchanged.
- **Leads:** 0.5 to 168 hours. Cuts at under 24, 24 to 72, and 72 to 168 hours, and 12 to 36 hours
  to match experiment 178.
- **Sunniest third:** experiment 178's rule, the top third of the daylight half-hours of each
  substation by the unit PV shape above 0.01. Experiment 178's clear-sky-index definition is also
  shown.
- **Periods:** the tuning months fit every coefficient and factor; the scored year (2025-07-02 to
  2026-06-29) decides; the replicate months replicate.
