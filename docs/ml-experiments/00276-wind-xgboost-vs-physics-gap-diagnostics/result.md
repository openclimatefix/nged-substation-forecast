---
id: "00276"
slug: wind-xgboost-vs-physics-gap-diagnostics
title: >-
  Why does an XGBoost model forecast the three metered wind farms better than a physics model of the
  wind turbines, and is the metering noise the reason?
theme: generator-forecasting
also_relevant_to: []
status: planned
date: 2026-10-23
size: >-
  medium. Stored: private tables under data/ml-experiments/00276-wind-xgboost-vs-physics-gap-diagnostics/
  only. Serving path: not touched. Degradation rule: not touched. Designs: one hypothesis (metering
  noise) is pre-registered and the other diagnostics are descriptive, so there is little room for
  another defensible design. Callers: this folder's scripts, which copy experiment 230's. One
  Sonnet code review, and one Sonnet write-up and prose review.
hypothesis: >-
  The XGBoost model beats the physics core on the three metered wind farms (W1, W2, W3) because it
  learns the metering noise (the reactive-power floor of the apparent-power series, the deadband, and
  the dip near cut-in) that the physics core ignores.
prediction: >-
  Written before any run, and expected to fail. With ERA5 as the weather, the core-minus-XGBoost gap
  in quantile score (0.27 points on all rows in experiment 230) shrinks by at least half when the
  rows of experiment 210's idle-band mask are removed from the evaluation, and shrinks further when
  they are also removed from training. Experiment 230 already hints against this, since its gap was
  0.126 points on the kept rows against about 0.10 on all rows. The more likely finding is that the
  gap sits at wind speeds between cut-in and rated, and is carried by wind direction or air
  temperature (D4), or by slow change over the years (D8).
decision_rule: >-
  H-noise is supported if, in the ERA5-weather setting, the gap on the kept rows is at most half the
  gap on all rows when both models train on all rows (D1), and the gap on the kept rows is smaller
  again when both train without the idle-band rows (D2). It is refuted if the gap on the kept rows is
  at least 0.7 of the gap on all rows. Anything between is `partly supported`. The porting threshold
  is not relevant, since nothing here goes to production. Intervals resample whole months of the
  ENS run date and the three seeds (50,000 resamples, 95%); farms are not resampled. Every
  diagnostic other than D1 and D2 is descriptive and labelled so. Stop after D1 to D3 if the gap is
  under 0.05 points in every ERA5-weather cell, and drop D9 and D10 first if the job runs past 6
  hours.
benchmark: >-
  The physics core B1 and the XGBoost best guess A0r of experiment 230 (ERA5-trained, 51 ensemble
  members, spread factor), and the Step 0 models P (core), S (binned curve), and X (XGBoost) with
  ERA5 as the weather.
row_set_id: >-
  experiment 33's wind cases (`fold_a_tuesdays_v1`: 53 Tuesday runs from 2025-07-01 to 2026-06-23,
  W1 to W3, three seeds). D3's by-year table also uses every half-hour from 2019-09 to 2026-06 under
  a leave-one-July-to-June-year-out fit, which is a different row set and is labelled.
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over nine levels, the estimate of the
  CRPS), in points of capacity; contrasts are core minus XGBoost, so a positive number means XGBoost
  is better
code: ml-experiments/00276-wind-xgboost-vs-physics-gap-diagnostics/
---

**Plan.** The three metered wind farms forecast better with an XGBoost model than with a physics
model of the turbines (experiment 230: 0.126 points of capacity on the rows that experiment 210's
mask keeps, 0.27 points with ERA5 as the weather). The maintainer's first guess is that XGBoost
learns the metering noise. This experiment tests that guess (D1 and D2) and then narrows down what
else XGBoost uses (D3 to D10). Each diagnostic runs with ERA5 as the weather, which isolates the
model, and D1 and D2 also run on the 51 ensemble members, which is the production setting.

## Diagnostics

- **D1:** score the core and XGBoost on (a) all rows, (b) the rows the mask keeps, (c) the mask's
  idle-band rows only, (d) ERA5 100 m wind above the core's fitted cut-in, and (e) above its rated
  speed. Both train on all rows.
- **D2:** repeat D1 with the mask's idle-band rows removed from training, giving four train-mask by
  evaluation-mask combinations.
- **D3:** the gap by 1 m/s wind-speed bin, 8 direction sectors, farm, season, hour of day, year, and
  lead day.
- **D4:** XGBoost fitted on 100 m wind speed alone, then adding direction, air temperature, the 10
  m level, and hour of day with day of year, with no column subsampling. ERA5 surface pressure, dew
  point, and gusts are not on disk for these farms, so air density rests on temperature alone.
- **D5:** an XGBoost model of the core's median residual, with permutation importance and
  partial-dependence plots.
- **D6:** the parametric core against the binned curve and a monotone neural curve, then the core
  with a direction-dependent term and a free density exponent added one at a time.
- **D7:** the core refitted on pinball loss with a hub-speed-dependent spread (7b), and the core's
  residual quantiles binned on hub speed instead of expected power (7a).
- **D8:** the core with a per-year, then a per-month, scale; XGBoost without date columns, with hour
  and day of year, and with year as well.
- **D9:** both models trained on 3, 6, 12, 24, and 48 months.
- **D10:** both models given ERA5 speeds scaled by a farm-specific factor fitted on the training
  years.

Experiment 230's scripts supply the frames, the core, the residual quantiles, and the scoring; the
scripts here copy them with extra options.
