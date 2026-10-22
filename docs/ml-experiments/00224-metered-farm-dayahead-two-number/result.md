---
id: '00224'
slug: metered-farm-dayahead-two-number
title: >-
  Does the two-number summary of the 51 ensemble members forecast a metered solar farm day ahead as well as XGBoost?
theme: metered-solar-farms
also_relevant_to:
- calibration-and-uncertainty
- differentiable-physics-embedded-generation
status: planned
size: >-
  medium. Stored: private tables and predictions under
  data/ml-experiments/00224-metered-farm-dayahead-two-number/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one defensible design for the band (a Gaussian
  mixture of the two numbers, or a closed-form shift of the residual offsets), so the Gaussian
  mixture of experiment 193 is registered. Callers: this folder's scripts, copied from experiments
  187, 193, and 218, and no new reviewed module. Asset graph: not touched.
hypothesis: >-
  Passing each of the 51 ensemble members through a solar farm's plant model gives the farm's day-ahead
  output a spread that carries information the ensemble mean lacks, and the mean and standard deviation of
  the 51 plant responses keep most of that information, as they did for Primary substations in
  [experiment 193](../00193-dp-ens-two-number-solar-summary/result.md). A two-number plant forecast
  therefore forecasts a metered solar farm 12 to 36 hours ahead about as well as XGBoost, and gives
  calibrated tails.
prediction: >-
  The arm with the mean and standard deviation of the 51 plant responses (arm 3) keeps at least 80% of the
  gain that the arm with all 51 responses (arm 4) has over the ensemble-mean arm (arm 2), in the quantile
  score on daylight half-hours at 12 to 36 hour leads; the MAE of arm 3's median forecast is within 5% of
  XGBoost's (arm 5); and arm 3's pinball loss at the 0.01 and 0.99 quantiles is lower than XGBoost's.
decision_rule: >-
  Quantile score is twice the mean pinball loss over the 11 levels of `studies.ens_members`, in points of
  capacity, over daylight half-hours (sun more than 5 degrees above the horizon) at 12 to 36 hour leads in
  the scored year (2025-09-01 to 2026-08-31), averaged over the six farms with equal weight. C0 is arm 4
  minus arm 2 as a percentage of arm 2. C1 is arm 3 minus arm 2 as a percentage of arm 2. Supported when
  C1 is at least 80% of C0 (the retained share), the interval of the share from resampling the 12
  scored months excludes 0.8 from below, and the other two conditions of the prediction hold; partly
  supported when only the share holds; not supported otherwise. Stop rule: if C0 is a gain of under 2%
  at three or more farms, members do not help a single farm, and the page says so first. If arm 3 keeps
  under 50% of the gain, report arm 4 only. The 80% is set by cost: a two-column summary replaces a
  51-member pass through the plant in the weather layer, and a summary that gave up a fifth of the gain
  would not justify a second path beside the member pass. The 5% on MAE is set by the cost of a physical
  plant beside XGBoost: a plant forecast is only worth keeping if it is not worse in the middle either.
benchmark: >-
  Arm 2 (plant on the ensemble mean), arm 5 (XGBoost on the same rows), and the manual_heuristic and
  climatology floors
row_set_id: >-
  Metered solar-farm half-hours with sun more than 5 degrees above the horizon, a null `drop_reason` in
  experiment 179's cleaned series, an observed power at every floor and arm, in 12 to 36 hour leads (and
  60 to 84 and 108 to 132 hour leads as secondary bands), scored 2025-09-01 to 2026-08-31 with a
  replicate of 2025-03-01 to 2025-08-31
metric: >-
  Quantile score in points of capacity, MAE of the median, and coverage and pinball loss at 0.01, 0.05,
  0.95, and 0.99, per farm and arm
code: ml-experiments/00224-metered-farm-dayahead-two-number/
production_status: not-for-production
adopted_in: none
---

Planned. The first commit of this branch is the plan, and the write-up replaces this page once the
experiment has run.

## Plan

- **Weather and forecast-time availability.** ECMWF ENS, 51 members, the 00 UTC run, at the 0.25 degree
  H3 cell nearest the farm. Each half-hour takes the run whose 12 to 36 hour window holds it (the same
  mapping as experiment 218's forecast-time repeat), so the weather exists at forecast time. Power
  lags are used only by the manual_heuristic floor, at 168 hours and longer, which exist at forecast time
  for every lead band scored here.
- **Plant.** The fleet shape of experiment 218 (tilt 35 degrees facing south, 0.4% loss per kelvin of
  cell temperature, cell temperature air plus 25 K times the shape) with a per-farm scale fitted on the
  tuning months to the ensemble-mean response, and a per-farm ceiling at the 99.5th percentile of the
  tuning output. [Experiment 218](../00218-metered-solar-farm-plant-fit/result.md) found that per-farm
  parameters add 0.003 of the output fitted, so no per-farm plant arm is run.
- **Arms.** 1: manual_heuristic and climatology. 2: plant on the ensemble-mean irradiance and
  temperature, with 11 offset quantiles from the tuning residuals in four bins of the forecast. 3: the
  plant responses' mean and standard deviation, with the 11 quantiles from 51 Gaussian points times the
  offsets (experiment 193's mixture), the spread factor fitted per lead band. 4: all 51 responses as the
  mixture centres. 5: XGBoost with 11 quantile outputs on the ensemble-mean and ensemble-spread
  weather, sun geometry, and lead, with no calendar columns.
- **Controls.** The rescale-only placebo (every response set to the mean, offsets rescaled), the shuffle
  of the spreads across rows of the same farm, month, and hour band, and the stricter shuffle that also
  matches the lead band and the forecast bin. The night placebo of experiment 187 is not run: a farm's
  output at night is zero in the plant and in the metering, so shifting a daylight spread onto night
  rows tests only the clip at zero.
- **Rows, folds, and shared parameters.** The ensemble store begins on 2024-04-01, so the tuning months
  of the ensemble-driven parts are April 2024 up to the fold, and the replicate fold scores six months
  with a shorter tuning window. The offset factor and spread factor are shared across farms and fitted
  leave-one-farm-out. XGBoost is fitted once per farm and fold on the tuning window and not refitted in
  the scored year; this is the cheapest schedule and the plant arms have no refit either.
- **Scoring.** `studies.tail_calibration` for coverage and pinball loss; per-farm and equal-weight farm
  means; 95% interval from resampling the scored months in blocks, the same months for all farms.
- **Curtailment value.** Scored against manual_heuristic with `studies.node_curtailment` only if a farm
  sits on a node of its pilot set; otherwise the page says why not.
