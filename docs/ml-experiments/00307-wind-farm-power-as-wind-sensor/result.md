---
id: "00307"
slug: wind-farm-power-as-wind-sensor
title: >-
  Which property of the ensemble's wind speed is wrong, judged against the wind that the farms' own
  power implies?
theme: generator-forecasting
also_relevant_to:
  - weather-products-and-blending
status: planned
date: 2026-10-23
size: >-
  medium. Stored: private tables under data/ml-experiments/00307-wind-farm-power-as-wind-sensor/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one defensible
  way to correct a speed series, so each correction is a registered arm. Callers: this folder's
  scripts, which copy experiments 261 and 285. One Sonnet code review and one Sonnet write-up and
  prose review.
hypothesis: >-
  The wind that the three metered wind farms' own power implies (the fitted physics core's power
  curve run backwards) is closer to a high-resolution analysis than to ERA5 at 9 to 14 m/s, and at
  least half of the ensemble's shortfall (observed power below the forecast's median on 34.1% of
  rows instead of 50%) is in the spread and timing of the members, not in their mean speed.
prediction: >-
  Written before any run. (1) The implied wind agrees with ERA5 to within 1 m/s on average at the
  invertible speeds. (2) The mean-speed correction (a) explains under 50% of the 18.7-point gap
  between the ensemble mixture's 34.1% and the ERA5 oracle's 52.8% of kept rows below the median,
  and the spread (b) and timing (c) corrections together explain at least 50%. (3) Where
  ICON-D2 is available, its first forecast hours are closer to the implied wind than ERA5 is at 9 to
  14 m/s.
decision_rule: >-
  Descriptive attribution, with no hypothesis test on a score. Supported when (1) holds, (2) holds
  and the first-hours ICON-D2 or UKV wind is closer to the implied wind than ERA5 is at 9 to 14 m/s.
  Partly supported when (2) holds alone or (3) holds alone. Not supported when the mean explains over
  70%. Stop after step 2 if the implied wind differs from ERA5 by more than 1 m/s on average at the
  invertible bins (the curve fit is then the issue). Shares carry 95% intervals that resample the 12
  months of the scored year; farms are not resampled. Nothing here goes to production, so there is
  no porting threshold. Cap at 5 hours.
benchmark: >-
  Experiment 261's stage shares for the same rows: ERA5 oracle 52.8%, ensemble-mean wind 36.6%,
  51-member mixture 34.1% of kept rows below the median; experiment 285's quantile mapping by lead
  day, 37.5%
row_set_id: >-
  experiment 33's wind cases (`fold_a_tuesdays_v1`: 53 Tuesday runs from 2025-07-01 to 2026-06-23,
  W1 to W3); corrections are fitted on experiment 259's 442 daily hindcast runs of 2024-04-01 to
  2025-06-16. Headline on the rows kept by experiment 210's mask, all rows beside it.
metric: >-
  Share of rows where observed power is below the forecast's median (calibrated: 50%), in points of
  PIT share explained out of the 18.7-point gap; implied-minus-reference wind speed in m/s by speed
  bin; quantile score in points of capacity as a secondary check
code: ml-experiments/00307-wind-farm-power-as-wind-sensor/
---

**Planned, not yet run.** This page is the plan; the write-up replaces it.

## Plan

- **Step 1, implied wind.** Fit experiment 261's physics core (R1) to ERA5 before 2024-04-01. Run its
  power curve backwards on half-hours outside experiment 210's mask whose observed power lies
  between 5% and 90% of the curve's plateau, and outside suspected outage runs (48 or more
  consecutive half-hours at under 1% of capacity). Compare with ERA5 by speed bin, direction
  sector, farm, and season.
- **Step 2, ensemble against implied wind.** Compare the members' 100 m speed (mean, 10th and 90th
  percentiles) with the implied wind by lead day, speed bin, direction, and farm. ERA5 against implied
  wind is the control.
- **Step 3, one property at a time.** Corrections fitted on the hindcast and applied to every member
  of the scored year, then the core, the residual quantiles, and the member mixture: (a) one
  multiplicative factor on the speed, (b) member deviations about the case mean inflated by a
  per-lead-day factor, (c) members shifted by the hour offset (-3 to +3 h) that best matches the
  implied wind, (d) a factor per 45-degree direction sector, (e) a factor per lead day. Each is
  scored by points of the 18.7-point PIT gap it closes.
- **Step 4, high-resolution gate.** ICON-D2 (Open-Meteo previous runs, 100 m wind, 2024-01 to
  2026-09) and UKV from AWS (height-level wind, 2024-10 to 2026-10) cover the scored year, so the
  first hours of each run are used as an analysis-like wind and run through the core. UKV on AWS is
  the live feed, so no research-archive caveat applies to that arm.

Known limit, stated before the run: the implied wind exists only where observed power is between 5%
and 90% of the plateau, so any comparison binned by implied speed is biased by that selection. Bins
are by the reference product's speed, and ERA5 against implied wind is run with the identical
selection as the control.
