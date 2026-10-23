---
id: '00286'
slug: metered-farm-drift-soiling-injection
title: >-
  How small a yearly drift or a soiling loss can the six metered solar farms' output show, with and
  without comparing each farm to its neighbours?
theme: metered-solar-farms
also_relevant_to: []
status: planned
size: >-
  simple. Stored: private tables under
  data/ml-experiments/00286-metered-farm-drift-soiling-injection/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one index could be built, so both are fixed in
  this plan before any run. Callers: this folder's scripts, copied from experiment 179, and no new
  reviewed module. Asset graph: not touched.
hypothesis: >-
  A yield index that divides each farm's output by the output of the other farms, on half-hours when
  the other farms are near clear-sky output, reads a yearly trend more precisely than the monthly
  yield index of experiment 179, and it can recover a soiling loss that the monthly index cannot.
prediction: >-
  At the farms with 4 or more years of data, the neighbour-relative index has a 90% interval on the
  yearly trend narrower than plus or minus 0.6 %/yr, against about plus or minus 1 %/yr for
  experiment 179's index. Soiling (0.1% per day, reset by 5 mm of rain in a day) is recovered at 2
  or more farms.
decision_rule: >-
  Drift is recovered at a farm when, over 5 random start dates, the median of recovered over
  injected trend lies in 0.7 to 1.3, and the 90% month-resampled interval of the injected series'
  trend excludes zero in at least 4 of the 5. The injection test passes for a method when drift of
  0.5 or 1 %/yr is recovered at 4 or more of the 6 farms. Stop rule: if neither method passes at 1
  %/yr, report that real drift under 1 %/yr cannot be read from output alone and do not run the
  real-data stage. C1: the width of the baseline 90% trend interval, neighbour-relative against
  experiment 179's index, at farms with 4 or more years of data (supported if under 0.6 %/yr
  half-width). C2: soiling recovered (median of recovered over injected slope in 0.7 to 1.3, interval
  excluding zero) at 2 or more farms. No model is ported, so the threshold for porting is not
  applicable; the minimum detectable drift is reported instead.
benchmark: >-
  Experiment 179's monthly yield index trend (drift.csv)
row_set_id: none (no forecast is scored; the rows are each farm's cleaned series with a null drop_reason)
metric: >-
  Recovered over injected trend (drift) and recovered over injected soiling slope, per farm,
  injected size, and method, with 90% intervals; minimum detectable drift
code: ml-experiments/00286-metered-farm-drift-soiling-injection/
production_status: not-for-production
adopted_in: none
---

# Plan

**Status: planned. No result exists yet.** Experiment 179 read yearly trends of -1.8 to +0.93 %/yr
from the six metered solar farms, each with a 90% interval about 1 %/yr wide. This experiment asks
how small a loss it could have seen, by adding known losses to the series and checking what comes
back.

## Design

- **Series.** Each farm's cleaned half-hourly output from experiment 179, rows with a null
  `drop_reason`, divided by the farm's effective capacity. Timestamps mark the end of the half-hour.
- **Injected drift.** A loss that grows linearly from a random start date: output multiplied by
  `1 - r × (years since start)` for r of 0.5, 1, and 2 %/yr. Five random start dates in the first
  quarter of each farm's span, with the same dates for every size and method.
- **Injected soiling.** Output multiplied by `1 - 0.001 × (days since the last reset)`, reset to
  zero on any UTC day on which the total rainfall at the farm's nearest grid cell is 5 mm or more.
  Five random start dates, as for drift.
- **Rainfall product.** ERA5 total precipitation (`tp`, hourly accumulation, 0.25 degree grid,
  [Hersbach et al. (2020)](https://doi.org/10.1002/qj.3803)) from the private ARCO-ERA5 copy under
  `data/studies/weather/ERA5/stage1_box/`. ERA5 arrives about 5 days after the event, so it does not
  exist at forecast time. This does not matter here, because the experiment reads past drift and
  injects soiling into history; it does matter for any live soiling correction.
- **Method A.** Experiment 179's monthly yield index: the 90th percentile of power over clear-sky
  irradiance, for half-hours with clear-sky irradiance above 300 W/m², divided by the mean of the same
  calendar month, then a straight line fitted to months with at least 40 half-hours and a relative
  yield of 0.75 or more, with a 90% interval from resampling months 2,000 times (here 500).
- **Method B.** For each pair of the target farm and another farm, the monthly median of the ratio of
  the target's power to the other farm's power, over half-hours when the clear-sky irradiance is
  300 to 650 W/m² (below the flat top of experiment 179) and every other farm available at that time
  has power within 0.85 to 1.15 of its own clear-sky yield (its 90th percentile of power over
  clear-sky irradiance, over the whole record), with at least 2 other farms available. Each pair's
  months are divided by the mean of the same calendar month and averaged over pairs, so a farm
  entering or leaving the comparison does not step the index. The mask uses only the other farms, so
  it does not depend on the injected series. The trend and its interval are fitted as for method A.
- **Soiling readout.** A daily version of each index (method A: the daily 90th percentile of yield;
  method B: the daily median pair ratio, at least 4 half-hours) is regressed on the days since the
  last reset, with a separate level for each calendar month of each year, so that the slow level is
  absorbed and only the within-month sawtooth is read. The interval resamples months.
- **Recovery.** The estimate on the injected series minus the estimate on the un-injected series,
  over the true injected quantity. For drift the true quantity is the straight-line slope of the
  injected multiplier over the same months; for soiling it is 0.1 %/day.
- **Minimum detectable drift.** The smallest of 0.5, 1, and 2 %/yr that meets the recovery rule at a
  farm, and the half-width of the 90% baseline interval as a second figure.
- **Real data stage.** Runs only if the injection test passes for a method. Reports each farm's trend
  with its interval for each method, anonymised.

## What this plan changes if it fails

- If neither method recovers 1 %/yr at 4 farms, the real-data stage is not run.
- If method B is no narrower than method A, the neighbours do not carry the weather noise away, and
  the next step is a better irradiance product rather than a farm comparison.

## Departures from the brief

- Method B averages normalised pairwise ratios rather than dividing by the mean of the other
  farms, because the set of farms with data changes over the record, and a changing mean would
  step the index.
- The experiment scripts stay in the experiment folder and no module is added to `packages/studies`,
  so that the pull request touches only `ml-experiments/` and `docs/ml-experiments/`.
