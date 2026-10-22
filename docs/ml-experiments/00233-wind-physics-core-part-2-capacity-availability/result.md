---
id: "00233"
slug: wind-physics-core-part-2-capacity-availability
title: >-
  Can a fitted wind-turbine physics model show that a wind farm has lost 25% of its capacity within
  60 days, without raising false alarms on a slow seasonal drift?
theme: generator-forecasting
also_relevant_to: []
status: planned
date: 2026-10-22
size: >-
  medium. Stored: private files under data/ml-experiments/00233-wind-physics-core-part-2-capacity-availability/
  only. Serving path: not touched, because no forecast is produced. Degradation rule: not touched.
  Designs: several are defensible (the statistic, the window, the alarm threshold), and this plan
  pre-registers one. Callers: this folder's scripts, which copy the physics core of experiment 230
  and read the observed power and ERA5 weather of W1 to W3. The change is a new analysis that fits
  one small model (experiment 230's core, refitted here), so one Sonnet plan review, one Sonnet code
  review, one Sonnet write-up review, and one Sonnet prose review.
hypothesis: >-
  For the three metered wind farms (W1, W2, W3), the ratio of observed power to the power that the
  fitted physics core of experiment 230 expects, taken over the half-hours when ERA5 wind is above
  the core's rated speed and averaged over a trailing 60 days, shows an injected capacity loss of
  25% within 60 days of the loss starting, at every farm, and does not raise an alarm on a slow
  seasonal drift of the size seen in the flat top of metered solar farms (experiment 183).
prediction: >-
  Written before any fit. About 8% to 14% of half-hours are above rated speed (experiment 230's
  Step 0), so a 60-day window holds roughly 230 to 400 such half-hours in an average season and
  far fewer in a calm summer. The relative ratio's noise over a 60-day window is about 3% to 5%, so
  the alarm threshold, set at the lowest value seen in the calibration years, lands near 0.85 to
  0.90. A 50% loss is found at every farm within 40 days. A 25% loss is found within 60 days at
  W1 and W2 and within 90 days at W3 (the farm with the fewest above-rated half-hours). A 10% loss
  is found at no farm within 120 days in more than half of the starts, because its size is close to
  the 9% seasonal drift. The seasonal-drift null raises no alarm at W1 and W2 and may do so at W3.
  On the real series, the monthly ratio of observed to expected power above rated differs from
  month to month by under 10% at W2 and W3, and shows a level below 1 for W1, consistent with its
  observed 99th percentile of about 21 MVA against 26 MW registered.
decision_rule: >-
  Known-answer test on observed power with a multiplicative loss injected from a random start date.
  The statistic is the relative ratio: the sum of observed power over the sum of expected power on
  above-rated half-hours in the trailing window, divided by the same ratio over all above-rated
  half-hours before 2024-07-01. The primary window is 60 days; 30 and 90 days are descriptive. A
  window with fewer than 48 above-rated half-hours has no statistic and cannot alarm. The alarm
  threshold per farm and window is the lowest relative ratio seen on any day from 2021-09-01 to
  2024-06-30 (calibration years, no injection), so those years raise no alarm by construction.
  Injection starts are drawn uniformly from 2024-07-01 to 2025-12-31 (30 starts per farm and loss
  size, fixed seed). The loss sizes are 10%, 25%, and 50% of the farm's observed power, and the
  turbine-count steps of 1, 3, and 6 of 13 turbines at W1, 1, 2, and 4 of 8 at W2, and 1 of 2 at
  W3. The detection delay is the number of days from the start to the first alarm; a start with
  no alarm within 180 days is undetected. The seasonal-drift null multiplies the observed power from
  2024-07-01 to 2026-06-30 by one minus 4.5% times one minus the cosine of the day of the year,
  halved (a 9% range, lowest on day 196), with no injected loss. Supported when, with the primary
  window, the median delay at a 25% loss is at most 60 days with at least 80% of starts detected
  within 120 days at all three farms, and the drift null raises no alarm at any farm. Partly
  supported when that holds at two of the three farms, or when the drift null alarms at one farm.
  Not supported otherwise. Stop rule: the median delay at a 25% loss exceeds 60 days at two or more
  farms; the real-data description still runs, and the write-up states that the physics core cannot
  find a loss of this size fast enough. The real-data part is descriptive and has no decision rule.
  Intervals: Wilson 95% intervals for detection rates and bootstrap 95% intervals (starts
  resampled) for median delays; starts overlap in time, so the intervals are narrower than the
  uncertainty about a different farm, and the farms are not resampled. No threshold for porting
  applies, because the experiment produces no forecast.
benchmark: >-
  Within-experiment contrasts only: the primary 60-day window against 30 and 90 days, and the
  statistic on above-rated half-hours against the same statistic on all half-hours whose expected
  power exceeds 20% of capacity (descriptive). No earlier experiment tests capacity inference.
row_set_id: >-
  none shared. All half-hours of W1 to W3 from 2019-09-01 to 2026-06-30 with ERA5 weather and an
  observed power value, dropping the first weeks after a new meter as experiment 230 does. No
  valid time on or after 2026-07-01 is used.
metric: >-
  Detection delay in days (median, per farm, loss size, and window), the share of starts detected
  within 120 days, alarms per farm-year on the null runs, and the monthly ratio of observed to
  expected power above rated on the real series
production_status: not-for-production
adopted_in: none
code: ml-experiments/00233-wind-physics-core-part-2-capacity-availability/
---

**Planned; no result yet.** This page is the plan, committed before any code runs.

## Question

We ask whether the physics model of experiment 230 can act as a monitor for a wind farm's effective
capacity: if some turbines stop generating, does the gap between observed power and the power that
the model expects in strong wind show the loss quickly, and without false alarms? In strong wind
(above the speed at which a turbine reaches rated power) the power curve is flat, so a fall in
observed power cannot be blamed on a weather forecast error, and the gap measures capacity. The
experiment first tests the idea on a known answer, a loss that we inject into real observed power,
and only then describes what the real series shows. It forecasts nothing.

## Method, as pre-registered

- **Core:** experiment 230's physics core (hub-height wind from the ERA5 10 m and 100 m speeds, air
  density, a smooth monotone power curve), refitted here on valid times before 2025-06-17 with
  experiment 230's code copied into this folder. The fitted hub heights and rated speeds must match
  [experiment 230's Step 0](../00230-wind-physics-core-part-1/result.md) (32, 33, and 37 m) to within
  1 m, or the refit is wrong.
- **Expected power:** the core's expected power on ERA5 for every half-hour of 2019-09 to 2026-06.
  ERA5 arrives about 5 days late, so this is a monitor run after the fact. A live monitor would use
  the ERA5 values of 5 days ago, which is the same statistic with a 5-day lag.
- **Injection:** the observed power from the start date onward is multiplied by one minus the loss.
  Multiplication matches the loss of whole turbines when wind is above rated. Below rated the
  true effect of a missing turbine is similar (the same share of the farm's power), so the
  statistic is not sensitive to that choice.
- **Reference series:** the same statistic on the real series, with no injection, over 2024-07-01 to
  2026-06-30, to count the alarms that the real series raises.
- **Real-data description:** the monthly sum of observed power over the sum of expected power on
  above-rated half-hours, and the monthly 99th percentile of observed power in MVA or MW, per farm.
  No cause is claimed.

## Checks and caveats fixed in advance

- **Departure from the design brief:** the brief asks for a rolling-window fit of the per-farm scale
  on above-rated rows. The ratio of sums is the least-squares scale on those rows, so the statistic
  is the same quantity without an iterative fit.
- **In-sample core:** the core is fitted on data up to 2025-06-17, so injections that start before
  then use a core that has seen those months. The expected power is a smooth six-parameter function,
  so the in-sample effect on the ratio is small; the write-up reports the results for starts before
  and after 2025-06-17 separately.
- **Tail calibration and curtailment cost:** not applicable, because no forecast is produced.
- **Anonymisation:** farms are W1 to W3. No farm's identity, coordinates, or series is shown.
