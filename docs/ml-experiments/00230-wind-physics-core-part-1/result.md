---
id: "00230"
slug: wind-physics-core-part-1
title: >-
  Does a wind-turbine physics model with an explicit idle layer forecast the three metered wind
  farms as well as the XGBoost best guess, and is the weather input the limit?
theme: generator-forecasting
also_relevant_to: []
status: planned
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
---

**Planned; no result yet.** This page is the plan, committed before any code runs.

## Question

We ask whether a physics model of a wind farm (hub-height wind, air density, a smooth power curve, and
an idle layer for the reactive-power floor that NGED's MVA meters show at zero output), fitted by
gradient descent on a 25 km ERA5 wind cell, forecasts the three metered wind farms as well as the
XGBoost best guess of [experiment 33](../00033-best-guess-architecture-vs-production/result.md).
Before anything else, Step 0 asks how much a perfect weather forecast (ERA5, the past-weather
estimate, as the forecast) could improve the fit, to learn whether the weather input or the turbine
physics is the limit. [Experiment 218](../00218-metered-solar-farm-plant-fit/result.md) found for solar
farms that per-farm plant parameters added only 0.003 over the fleet shape, so the weather input was
the ceiling.

## Arms

- **A0:** the stored best guess. **A0r:** the same recipe refitted with the 0.01 and 0.99 levels added.
- **A1:** A0r trained on the ENS control member's forecast wind (the ENS training table holds one
  member per run) and run on the 51 members.
- **B1:** the physics core trained on ERA5, run on the 51 ENS members, with the spread factor of
  experiment 33. **B2:** the core trained on the ENS control member's forecast wind. **B8:** B1
  with no idle layer.
- **B3:** B1 plus a small neural calibrator on the power-curve residual. **B5:** B1 plus a per-farm
  quantile XGBoost stage 2, as in [experiment 29](../00029-physics-stage1-solar-pytorch/result.md).
  **B6:** a neural network on the same inputs with no physics. Run only if Step 0 allows.
- **B4 and B7** (B3 and B1 trained per ENS member) wait for W-A's first report.
- **Reference rows:** the month-by-half-hour climatology, and persistence at 0 to 6 hours.

## Physics core, as pre-registered

- **Hub-height wind:** the shear exponent between the ERA5 10 m and 100 m speeds, applied to a fitted
  hub height with a prior of 50 to 80 m (the turbine model and hub height are not on disk).
- **Air density:** the dry-air density from the 2 m temperature and the surface pressure, applied as
  the cube root ratio to the standard density.
- **Power curve:** a smooth monotone curve with a cut-in speed, a rated speed, a shape exponent, a
  logistic cut-out, and a Gaussian turbulence blur that stands for the 25 km cell's sub-grid wind.
  The rated power starts from the registered turbine count times rating (W1 26 MW, W2 9.6 MW, W3
  2.6 MW, where the effective capacity is that of experiment 52), and a per-farm scale is fitted.
- **Idle layer (W1 and W3, in MVA):** the expected apparent power is the square root of the squared
  active power plus a squared idle floor, with a small fitted dip as the wind rises from zero; the
  floor is fitted on the rows experiment 210 marks as the idle band. W2 (in MW) has no floor. The
  spread of the observation at each forecast level comes from the empirical residual quantiles,
  fitted on the ERA5-training rows in bins of the predicted power, which carry the meter noise.
- **Fit:** PyTorch, full batch, Adam, on CPU with one thread per process, an L1 loss on the power
  fraction, on valid times before 2025-06-17.

## Checks and caveats fixed in advance

- **Departures from the design brief:** ENS-mean training (A1, B2) uses the ENS control member,
  because the ENS training table holds only member 0 and extracting the 51 members of 456 runs would
  cost days; the idle layer shapes the mean, and the noise comes from the empirical residual
  quantiles; the physics arms are deterministic, so their three seeds repeat one fit, as the
  climatology rows do.
- **Comparison with experiment 221:** the write-up compares B1 and A0 with [experiment
  221](../00221-wind-correction-head-on-stored-predictions/result.md), which put a neural head on A0.
- **Anonymisation:** farms are W1 to W3. No farm's identity, coordinates, or series is shown.
