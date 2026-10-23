---
id: "00285"
slug: wind-effective-wind-adapters
title: >-
  Can a small correction to the wind speed, before the physics model of a wind farm or after the
  ensemble, close the gap to XGBoost?
theme: generator-forecasting
also_relevant_to: []
status: planned
date: 2026-10-23
size: >-
  medium. Stored: private tables under data/ml-experiments/00285-wind-effective-wind-adapters/ only.
  Serving path: not touched. Degradation rule: not touched. Designs: the arms are fixed by the
  brief, so little room for another defensible design. Callers: this folder's scripts, which copy
  experiments 259 and 276. One Sonnet code review and one Sonnet write-up and prose review.
hypothesis: >-
  A small bounded correction of the wind speed that the physics core of a wind farm reads, learned
  from the wind direction, the 10 m to 100 m speed ratio, the 2 m temperature, and the hour of day,
  closes at least half of XGBoost's lead over the core on the three metered wind farms (W1, W2, W3)
  when ERA5 is the weather. A correction of each ensemble member's wind speed towards ERA5's
  removes the ensemble's low bias.
prediction: >-
  Written before any run. Part A (ERA5 as the weather, kept rows, gap 0.31 points in experiment
  276): a shear term plus direction sectors close at least a quarter of the gap; the neural adapter
  closes at least half (0.15 points) and beats the shear plus sector adapters by at least 0.05; the
  blind adapter (farm and 100 m speed only) closes under 0.05. Part B (51 ensemble members): the
  best of quantile mapping by lead day, by lead day and direction sector, or a neural mapping
  raises the share of observed power below the median from 34% to above 42% and lowers the score
  by at least 0.2 points against the raw members; it does not beat the experiment 259 offset table
  by more than 0.05; stacking the best mapping with the offset table helps the 0.99 quantile.
decision_rule: >-
  Supported if the neural adapter A4 closes at least 0.15 points of the core-minus-XGBoost gap on
  the kept rows with a 95% interval above zero and beats A2 plus A3 by at least 0.05 (Part A), and
  the best of B1 to B3 lowers the score by at least 0.2 points against B0 with the observed-below-
  median share above 42% (Part B). Partly supported if one part meets its rule. Not supported if
  neither does. Nothing here goes to production, so there is no porting threshold. Intervals
  resample whole months of the ensemble run date and the three seeds (50,000 resamples, 95%);
  farms are not resampled. Stop Part A if no arm closes 0.05 points; stop Part B if quantile
  mapping B1 changes the observed-below-median share by under 3 points. Run Part B even if Part A
  stops. Cap at 6 hours.
benchmark: >-
  The physics core and the XGBoost model of experiment 276 (core P, XGBoost X) with ERA5 as the
  weather for Part A. For Part B, the same two models trained on ERA5 before 2024-04-01 on raw
  ensemble members, and experiment 259's lookup table of offsets per lead day, quantile, and farm.
row_set_id: >-
  experiment 33's wind cases (`fold_a_tuesdays_v1`: 53 Tuesday runs from 2025-07-01 to 2026-06-23,
  W1 to W3, three seeds). Part B's mapping and offsets are fitted on the daily ensemble hindcast
  runs of 2024-04-01 to 2025-06-16, as in experiment 259.
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over nine levels, the estimate of the
  CRPS), in points of capacity; contrasts are first model minus second, so a positive number means
  the second is better
code: ml-experiments/00285-wind-effective-wind-adapters/
---

**Plan.** Experiment 276 found that XGBoost beats the physics core of the three metered wind farms
because it corrects the wind speed, mainly at 9 to 13 m/s, using the wind direction and the 10 m
wind speed. Experiment 261 found that 89% of the ensemble's low median enters at the ensemble's
weather. This experiment tests one adapter on each side of the weather: a small bounded correction
of the wind speed the core reads (Part A, ERA5 as the weather), and a correction of each ensemble
member's wind speed towards ERA5 before the core (Part B, the production setting).

## Arms

- **A0:** the core as in experiment 276. **A1:** XGBoost as in experiment 276.
- **A2:** effective speed = 100 m speed times (10 m to 100 m speed ratio) to the power k, with k
  fitted per farm.
- **A3:** a speed scale for each of 8 direction sectors, fitted per farm. **A23:** A2 and A3 together.
- **A4:** a neural adapter (two layers of 32 units; speed scale 0.8 to 1.25 and offset within ±1 m/s)
  fed direction (sine, cosine), the speed ratio, 2 m temperature, and hour of day, trained end to end
  through the core.
- **A5:** A4 without the hour of day. **A6:** a blind adapter fed only the 100 m speed.
- **B0:** the core and XGBoost on raw members. **B1:** quantile mapping of each member's 100 m speed
  to ERA5's by lead day. **B2:** B1 by lead day and direction sector. **B3:** a neural mapping.
  **B4:** the 259 offset table on top of B0. **B5:** the best of B1 to B3 followed by the offset table.

Deviations from the brief, fixed before any run: the adapters are trained on the median pinball loss
(the L1 loss that trains the core), and the quantiles come from residuals as in experiment 276; Part
B trains the core and XGBoost on ERA5 before 2024-04-01, as experiment 259 did, so that the
hindcast months are out of sample for the offset table.
