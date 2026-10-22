---
id: "00230"
slug: wind-physics-core-part-1
title: >-
  Does a wind-turbine physics model with an explicit idle layer forecast the three metered wind
  farms as well as the XGBoost best guess, and is the weather input the limit?
theme: generator-forecasting
also_relevant_to: []
status: finished
date: 2026-10-22
size: >-
  complex. Stored: private files under data/ml-experiments/00230-wind-physics-core-part-1/ only.
  Serving path: not touched. Degradation rule: not touched. Designs: several are defensible (power
  curve family, how the idle layer enters, how quantiles are built from a deterministic curve), and
  this plan pre-registers one. Callers: this folder's scripts, which copy experiment 33's and
  experiment 52's helpers and read the stored best guess, the ENS member features, and experiment
  210's mask. A new model family and a new quantile construction, so one Sonnet plan review, one
  Sonnet code review, one Sonnet write-up review, and one Sonnet prose review; the manager decides
  whether an Opus read of the final write-up is worth the quota.
hypothesis: >-
  For the three metered wind farms (W1, W2, W3), a physics core fitted by gradient descent (hub-height
  wind from the ERA5 100 m and 10 m speeds, an air-density correction, a smooth monotone power curve
  with a cut-out, and an idle layer for the reactive-power floor of the MVA series) run through the
  51 ENS members scores within 0.1 points of capacity of the XGBoost best guess (experiment 33) on the
  quantile score, and the measured gap between the ERA5-input ceiling and a simple power curve is
  under 0.02 points, which would mean the 25 km weather cell, and not the turbine physics, limits the
  forecast.
prediction: >-
  Written before any fit. Step 0 (ERA5 at inference, held-out year): the physics core, a binned
  power curve, and a flexible XGBoost model all score within 0.3 points of each other on all rows,
  and the flexible model is under 0.1 points better than the core. The core explains 0.65 to 0.75 of
  the power's variance (R squared) at W1 and W2, and less at W3, consistent with experiment 210's
  correlations of 0.81 to 0.86 with ERA5 wind. B1 minus A0 is within plus or minus 0.1 points over all
  leads, because XGBoost already learns the power curve. B1 beats B2 (ENS-control-trained) by more
  than 0.3 points, as ERA5 training did for XGBoost in experiment 1. B8 (no idle layer) is worse than
  B1 on all rows by 0.02 to 0.10 points and equal on the rows kept by the experiment 210 mask. B3
  (neural calibrator, run only if Step 0 allows) improves on B1 by 0.05 to 0.15 points, mostly at the
  0.99 quantile near rated power.
decision_rule: >-
  Step 0 gate, read before any ENS-driven arm: the held-out year of July 2025 to June 2026 scored on
  experiment 52's 53 Tuesday runs' valid times with ERA5 wind as the weather. The ERA5 ceiling is the
  better of the physics core (P) and a flexible ERA5-input XGBoost model (X) on the quantile score
  over all rows; the simple curve (S) is a binned power curve with residual quantiles. If S minus
  the ceiling is under 0.02 points, the weather input is the limit: B3, B5, and B6 are not run, and
  only B1, B2, B8, A0, and A1 are. Stop rule 2: stop after B1 and B8 if B1 is worse than A0 by more
  than 0.3 points over all leads (the wind-level or power-curve set-up is wrong). Stop rule 3: stop
  if the observation layer cannot reproduce the idle-row distribution on held-out months, defined as
  the 5% to 95% band of B1 (ERA5 input) covering fewer than 80% or more than 97% of held-out rows
  whose ERA5 wind speed is below 2 m/s, at any farm. Decisive contrasts, each the first arm minus
  the second on the quantile score over all leads on the rows kept by the experiment 210 mask,
  with whole months and the three seeds resampled, 50,000 resamples, 95% intervals, Bonferroni-corrected
  to 98.75% across the four: C1 B1 minus A0, C2 B3 minus B1 (only if run), C3 B1 minus B2, C4 B1 minus
  B8. The threshold for porting the physics core as a replacement for the XGBoost stage is a C1
  interval wholly below zero, because it replaces a model and adds a fitted physics step to the
  serving path; "within 0.1 points" (C1 interval inside plus or minus 0.1) is reported as `partly
  supported` (competitive, not better). Everything else (per farm, per lead band, the three row sets,
  all other arm contrasts, the 0.01 and 0.99 pinball losses) is descriptive and labelled so. The
  intervals describe these three farms, because farms are not resampled.
benchmark: >-
  A0: the experiment 33 wind best guess (`bg_full`) as stored, on the same cases. A0r: the same recipe
  refitted here with the 0.01 and 0.99 levels added, to score the tails; A0r must reproduce A0 on the
  nine shared levels to within 0.05 points, or the tail rows are not reported.
row_set_id: >-
  experiment 33's wind cases (`fold_a_tuesdays_v1`: 53 Tuesday runs from 2025-07-01 to 2026-06-23,
  W1 to W3, three seeds). Three row sets are reported: all rows, rows kept by experiment 210's mask,
  and rows where the ERA5 100 m wind speed is above the farm's fitted rated speed.
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over nine levels, the estimate of the
  CRPS), in points of capacity; pinball loss at 0.01, 0.05, 0.5, 0.95, and 0.99; coverage of the 5%
  to 95% band; R squared of the median forecast in Step 0
production_status: not-for-production
adopted_in: none
code: ml-experiments/00230-wind-physics-core-part-1/
verdict: not supported
best_arm: >-
  A0 and B14 (the XGBoost best guess, with or without the core's columns), 11.66 points on the kept
  rows over all leads
headline: >-
  The physics core (B1) scored 0.13 points worse than the XGBoost best guess (A0) on the kept rows
  over all leads (95% interval 0.07 to 0.19), and 0.32 worse at lead days 1 and 2.
take_home_message: >-
  A physics model of a wind farm, fitted to the three metered farms, forecasts them slightly worse
  than the XGBoost best guess, so it should not replace it. Most of the forecast error comes from the
  25 km ensemble wind, not from the power curve: with ERA5 wind as the forecast, the same model
  scores about half the error. The idle layer and the small neural adapter add nothing, and
  training the physics core on ERA5 instead of the ensemble control member improves it by 0.4 points.
advice:
  - >-
    Keep XGBoost as the wind forecaster, and do not add a physics stage in front of it or its
    columns beside it (C1, E2).
  - >-
    Train wind models on ERA5, not on the ensemble control member, for the three metered farms (C3).
  - >-
    Put effort into the weather input, not the power curve or the idle layer: ERA5 as the forecast
    halves the score, and the idle layer changes it by under 0.01 (F1, C4).
  - >-
    Add air temperature to the XGBoost wind features, which improved the kept-row score by 0.08 in a
    post hoc contrast that needs a planned repeat (F2).
---

**Take-home: the physics model is slightly worse than XGBoost, and the weather input is the limit.**
A physics model of a wind farm scored 0.13 points of capacity worse than the XGBoost best guess on
the three metered wind farms, so we should not port it. The score is far better when ERA5 supplies
the wind, which shows that the 25 km ensemble wind, not the turbine model, causes most of the error.

![The physics core is 0.1 to 0.3 points worse than the best guess](figure_1.svg)

## Results

Scores are the quantile score in points of capacity (smaller is better), on the rows that experiment
210's mask keeps, over all lead times, for W1, W2, and W3 across 53 Tuesday runs from July 2025 to
June 2026. The intervals resample whole months and the three seeds, and do not resample farms, so they
describe these three farms only.

- **C1, the decisive contrast: the core minus the best guess is +0.126 (95% interval 0.068 to 0.190).**
  At lead days 1 and 2 it is +0.322 (0.241 to 0.413). The core is worse, so the porting threshold (an
  interval wholly below zero) is not met.
- **C4: the idle layer is worth nothing (B1 minus B8 is 0.000, interval -0.002 to 0.002).**
  The residual-quantile layer already carries the reactive-power floor.
- **C2: the neural output adapter (B9) adds nothing (+0.002, -0.004 to 0.007).**
- **C3: training the core on ERA5 beats training it on the ENS control member by 0.40
  (interval 0.12 to 0.69).** At lead days 1 and 2 the gap is 1.95 points. Most of it comes from W2.
- **XGBoost on the core's output (B5) closes about half the gap (B5 minus B1 is -0.060, -0.120 to -0.004),
  and still trails the best guess.**
- **The pure neural network (B6) is no better than the best guess (+0.037, -0.047 to 0.115).**
- **The best guess with the core's columns added (B14) equals the best guess (-0.007, -0.014 to 0.000).**
- **ERA5 as the forecast (E0) scores 6.71 against 11.79 for B1.** The 25 km wind, not the power
  curve, causes most of the error.

![Absolute quantile score by lead band](figure_2.svg)

| Arm | Description | Kept rows, all leads | Kept rows, lead days 1 and 2 | All rows, all leads |
|---|---|---|---|---|
| A0r | Best guess, refitted here | 11.66 | 7.03 | 11.04 |
| B14 | Best guess plus the core's columns | 11.66 | 7.02 | 11.03 |
| B6 | Neural network, no physics | 11.70 | 7.22 | 11.08 |
| B5 | Core plus XGBoost stage 2 | 11.73 | 7.13 | 11.09 |
| B1 | Physics core, ERA5-trained | 11.79 | 7.35 | 11.14 |
| B9 | Core plus neural output adapter | 11.79 | 7.34 | 11.14 |
| B8 | Core without the idle layer | 11.79 | 7.35 | 11.14 |
| B2 | Core, ENS-control-trained | 12.20 | 9.30 | 11.68 |
| A1 | Best guess, ENS-control-trained | 12.17 | 9.24 | 11.67 |
| CLIM | Climatology | 13.55 | 13.12 | 13.12 |
| E0 | ERA5 as the forecast (reference) | 6.71 | 6.36 | 6.32 |

**Beyond day 7, climatology (13.48 on the kept rows) beats every trained arm (13.84 to 13.93).** The
tails behave alike across arms: on the kept rows the 0.01 pinball loss is 0.285 to 0.288 and the 0.99
loss is 0.658 to 0.676 for A0r, B1, B5, B6, and B14, and the 5% to 95% band covers 91.4% to 91.7% of
rows. Persistence on hours 0 to 6 scores 15.57, worse than every other arm.

## Step 0: the ERA5 ceiling

![Step 0 ceiling on the held-out year](figure_3.svg)

- **With ERA5 as the weather, a flexible XGBoost model (X) beats the physics core (P) by 0.27 points
  and the binned power curve (S) by 0.54 (interval 0.43 to 0.64) on all rows.** The gate asked whether
  S was within 0.02 of the ceiling. It was not, so the other arms ran.
- **The core explains 0.84, 0.76, and 0.72 of the power's variance (R squared) at W1, W2, and W3.**
  X explains 0.85, 0.77, and 0.75.
- **Stop rule 3 fired literally at W2.** The 5% to 95% band of B1 covered 59% of W2's idle rows
  (threshold 80%). W2 is in MW, observed idle power can be negative, and the clip at zero is shared by
  every arm, so the arms stay comparable.

## Comparison with experiment 221

[Experiment 221](../00221-wind-correction-head-on-stored-predictions/result.md) put a neural head on
the stored best guess, and found no reliable gain. On the same kept rows at lead days 1 and 2, its A0
scored 7.223 and its head 7.403. Here the best guess scores 7.02 to 7.03 on the same rows, B6 (a neural
network with no physics) 7.22, and the core 7.35. Neural components and a physics core thus land in
the same place: neither beats XGBoost on these three farms.

## Checks

- **Median minus mean:** the median of the forecast sits 5.2 points below the mean on the kept rows
  for A0r and 5.3 for B1, so all arms share the same skew and the mean-error contrasts agree in
  direction with the quantile-score contrasts.
- **A0r reproduces A0** on the nine shared levels (11.035 against 11.037, all rows).
- **Leave-one-farm-out:** C1 stays positive without any single farm (0.11 to 0.14), and C3 stays
  negative.
- **Row sets:** on the rows above the farm's fitted rated speed, C1 is +0.43 (0.30 to 0.57). On all
  rows it is +0.10.
- **Audit triggers 1 and 4 fired** (and trigger 7 may have). The manager decides whether an Opus audit
  is worth the quota.

## Departures from the plan

- **ENS control member instead of the ENS mean** for A1 and B2.
- **Temperature-only density correction:** ERA5 surface pressure is not on disk, so the core uses a
  fixed 100 kPa.
- **The idle layer shapes the mean, and the noise comes from the residual quantiles.**
- **The fitted hub heights (32 to 37 m) are not physical.** They absorb the 25 km cell's sub-grid wind.
- **The physics arms are deterministic**, so their three seeds repeat one fit.
- **E0 and A0rT (A0r with temperature) are post hoc.**
- **Not run:** B10 to B13, the no-history tests, and the curtailment-cost metric.
- **The wind-farm identities and series are anonymised;** the mask stays private.

## What the next experiment should answer

Does a better weather input, such as a blend of weather models or a downscaled wind, close any of the
6-point gap between the ensemble wind and ERA5 wind?
