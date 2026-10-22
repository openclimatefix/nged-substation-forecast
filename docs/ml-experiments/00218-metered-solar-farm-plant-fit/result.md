---
id: '00218'
slug: metered-solar-farm-plant-fit
title: >-
  Does a per-farm plant model fit the six metered solar farms better than the fleet shape?
theme: metered-solar-farms
also_relevant_to:
- solar-disaggregation
- differentiable-physics-embedded-generation
status: planned
size: >-
  medium. Stored: private tables, fitted parameters and predictions under
  data/ml-experiments/00218-metered-solar-farm-plant-fit/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one defensible design (a grid search or a
  gradient fit for the plant, and several definitions of "fraction of output fitted"), so the grid
  and the score are fixed here before any fit. Callers: this folder's scripts and one new reviewed
  module in `packages/studies`, which nothing else imports. Asset graph: not touched.
hypothesis: >-
  The six metered solar farms differ in panel tilt, panel azimuth, ratio of direct-current to
  alternating-current capacity, heat loss, and inverter clipping, so a plant model with those
  parameters fitted per farm reproduces each farm's output better than the one fleet shape (tilt 35
  degrees, facing south, DC:AC ratio 1.15, 0.4% loss per kelvin, cell temperature equal to air
  temperature plus 25 K times the shape), and explains why the flat top of farms C, D, E, and F
  falls 6% to 9% from spring to summer ([experiment 183](../00183-metered-farm-flat-top-cause/result.md)).
prediction: >-
  The median over the six farms of the fraction of metered output fitted rises from about 0.93
  (the ceiling of one scale of the fleet shape in [experiment
  157](../00157-differentiable-physics-joint-pv-capacity-known-answer/result.md), measured on other
  rows) to 0.97 or more, scored on the 12 months after the tuning months, with CAMS irradiance.
decision_rule: >-
  The fraction of output fitted is 1 minus the sum of squared errors over the sum of squared metered
  output, over daylight half-hours (sun more than 5 degrees above the horizon). The headline is the
  median over farms, with a 98.75% interval from resampling the 12 scored calendar months as blocks
  (a Bonferroni level for 4 contrasts). C1: arm 2 minus arm 1. C2: arm 3 minus arm 2. C3: arm 2
  minus arm 4. C4: arm 2 minus arm 2 with the geometry and heat parameters pooled from the other five
  farms. Supported when C1 is at least 0.02 with an interval above zero and arm 2's median is at
  least 0.97. Stop rule: if C1 is under 0.02, the ceiling is in the weather and not the plant; say
  so, and stop. If C2 is under 0.01, keep arm 2's parameters and not the learned response. The
  threshold of 0.02 is set by the cost of porting: a per-farm parameter set is a new fitted object
  per farm, and a gain under 0.02 of the output is smaller than the difference between CAMS and ERA5
  irradiance seen in earlier experiments.
benchmark: >-
  Arm 1, the fleet shape with only a scale fitted per farm, on the same rows
row_set_id: >-
  Metered solar-farm half-hours with sun more than 5 degrees above the horizon and a null
  `drop_reason` in experiment 179's cleaned series, scored for 2025-09-01 to 2026-08-31 (fold 1) and
  2024-09-01 to 2025-08-31 (fold 2, replication). Tuning uses only months before the fold's scored year.
metric: >-
  Fraction of output fitted, MAE in MW and in points of capacity, and the seasonal range of the flat
  level (experiment 183's definition) per farm and arm
code: ml-experiments/00218-metered-solar-farm-plant-fit/
verdict: null
production_status: not-for-production
adopted_in: none
---

**Plan only: no fit has run yet.** This page is the pre-registration of experiment 218. The write-up
replaces it when the fits finish.

**Question.** The six metered solar farms (labelled A to F) output less in summer at the same
irradiance than in spring, and four of them flatten at a level that falls 6% to 9% from April to
August ([experiment 183](../00183-metered-farm-flat-top-cause/result.md)). We ask whether giving
each farm its own panel tilt, azimuth, DC:AC ratio, heat loss, and soft inverter clip fits its output
better than the one fleet shape the demand models use, and which of those parameters explains the
falling flat top. No weather model is trained. The plant model is the one in `studies.pv_physics`.

## Arms

- **Arm 1 (control).** The fleet shape (`studies.pv_physics.PlantParameters` with tilt 35, azimuth
  180, 0.4% per kelvin, cell temperature = air + 25 K × shape) with one scale fitted per farm.
- **Arm 2.** Per-farm tilt (10 to 60 degrees in steps of 10), azimuth (120 to 240 in steps of 20),
  heat coefficient (0, 0.2, 0.4, 0.6%/K), and, at each grid point, a DC scale and an AC limit with a
  soft clip (a p-norm minimum, p = 6) fitted by least squares. The grid point with the smallest
  squared error on the tuning months is the farm's plant.
- **Arm 3.** Output over DC scale as a monotone surface in irradiance and sun elevation (hat basis,
  non-negative increments, a convex bounded quadratic), ridge-penalised towards arm 2's cell means.
  The penalty weight is picked by month-block cross-validation inside the tuning months.
- **Arm 4.** Arm 2 with the heat coefficient fixed at 0.
- **Ablations for the flat top (not contrasts).** Arm 2 with the clip removed, and arm 2 with the
  fleet's tilt and azimuth.
- **Leave-one-farm-out (C4).** Tilt, azimuth, and heat coefficient chosen to minimise the summed
  tuning error of the other five farms, each with its own scale and clip; the held-out farm's own
  scale and clip are then fitted.

## Rows and weather

- **Farms and rows.** Experiment 179's cleaned series with a null `drop_reason`, in daylight. Power
  stamps mark the end of the half-hour.
- **Irradiance at the fit.** CAMS hourly global irradiance at each farm's location, split into
  half-hours by the cosine of the zenith and into beam and diffuse parts with the Erbs correlation.
  It exists only about two days after the fact, so it does not exist at forecast time.
- **Temperature at the fit.** ERA5 2 m temperature (Open-Meteo) at the nearest cell; it arrives about
  5 days late.
- **Forecast-time repeat.** The best arm repeated with the ECMWF 51-member ensemble mean at 12 to 36
  hour leads, which does exist at forecast time, over the months the ensemble store covers.

## Prior advice followed

- Judge a flat level at the 99th percentile as well as the 90th, because the 90th is exposed to cloud
  (experiment 183).
- Fit tilt, azimuth, and a heat term, not only a clip level (experiment 183).
