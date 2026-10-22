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
  pre-registers one. Callers: this folder's scripts, which copy the helpers of experiment 230 and
  read the observed power of W1 to W3 and the fitted cores and ERA5 expected power that experiment
  230 exported. The change is a new analysis that fits nothing (it reads experiment 230's fitted
  core), so one Sonnet plan review, one Sonnet code review, one Sonnet write-up review, and one
  Sonnet prose review.
hypothesis: >-
  For the three metered wind farms (W1, W2, W3), the ratio of observed power to the power that the
  fitted physics core of experiment 230 expects, taken over the half-hours when the core expects
  rated power and averaged over a trailing 60 days, shows an injected capacity loss of 25% within
  60 days of the loss starting, at every farm, and raises no alarm on a slow drift of the observed
  power of the size seen in the flat top of metered solar farms (experiment 183).
prediction: >-
  Written before any run. Predicted verdict: partly supported. About 3% to 14% of half-hours are
  above rated (2,239 of about 46,000 at W1 over the whole record, 4,152 at W2, 722 at W3), so a
  60-day window holds a few hundred such half-hours at W1 and W2 in a windy season, and none for
  weeks in a calm summer, and far fewer at W3. The relative ratio's noise over a 60-day window is
  3% to 5% at W1 and W2 and 8% to 12% at W3, so the alarm threshold lands near 0.85 to 0.90 at W1
  and W2 and near 0.70 to 0.80 at W3. A 50% loss is found at every farm within 40 days. A 25% loss
  is found within 60 days at W1 and W2 and takes longer than 60 days at W3. A 10% loss is found
  in under half of the starts at every farm. The 9% drift raises no alarm at W1 and W2, and the
  largest drift range that raises no alarm is between 10% and 20% at W1 and W2. The statistic with
  a constant expected power in place of the core's performs within 10% of the core's statistic,
  because the expected power on the flat top is nearly constant, so the core's value is the
  selection of the half-hours and not the curve. The monthly ratio of observed to expected power
  above rated, in months with enough half-hours, moves by under 10% from month to month at W1 and
  W2, and by more at W3.
decision_rule: >-
  Rule 1, the statistic. The relative ratio for a farm on a day is the sum of observed power over
  the sum of expected power, over above-rated half-hours in the trailing window ending that day,
  divided by the same ratio over all above-rated half-hours before 2024-07-01 (the reference level).
  A half-hour is above rated when the core expects at least 95% of the farm's fitted scale. The
  primary window is 60 days, and 30 and 90 days are descriptive. A window with fewer than 6 days
  that each hold at least 4 above-rated half-hours has no statistic and cannot alarm. Rule 2, the
  threshold. The alarm threshold per farm and window is the lowest relative ratio on any day from
  2021-09-01 to 2024-06-30, so those years raise no alarm by construction, and the write-up reports
  each threshold and its margin below 1. A hold-out check calibrates on 2021-09-01 to 2023-06-30 and
  counts alarm days from 2023-07-01 to 2024-06-30, and is descriptive. Rule 3, the injection.
  Observed power from the start day onward is multiplied by one minus the loss, with the core
  frozen. Losses are 10%, 25%, and 50% (whole-turbine losses are the same multipliers, 1 of 13
  turbines at W1 being 7.7%, 1 of 8 at W2 being 12.5%, and 1 of 2 at W3 being 50%, and are reported
  under the nearest size). Starts are 30 dates drawn uniformly from 2024-07-01 to 2025-12-31 with a
  fixed seed, the same 30 for every loss size and farm. The delay is the number of days from the
  start to the first day on which the injected run alarms and the un-injected run does not. A start
  where the un-injected run already alarms on the day before is excluded and counted. A start with
  no alarm within 180 days is undetected and counts as an infinite delay in the median. Rule 4,
  the drift null. Observed power from 2024-07-01 to 2026-06-30 is multiplied by 1 minus half the
  range times one minus the cosine of 2 pi times the days since 2024-07-01 over 365.25, with no
  loss injected. The range is 9% (experiment 183's 6% to 9%), 15%, and 25%, and a separate linear
  drift of 10% per year is run. The un-injected real series is run for comparison. Rule 5, the
  verdict, with the primary window. Supported when the median delay at a 25% loss is at most 60
  days and at least 80% of starts are detected within 120 days at all three farms, and the 9% drift
  raises no alarm at any farm. Partly supported when both hold at two of the three farms. Not
  supported otherwise. The stop rule is the not supported case: the real-data description still
  runs, and the write-up states that the physics core cannot find a loss of this size fast enough.
  Rule 6, intervals. A bootstrap over whole calendar months of start dates gives 95% intervals for
  the median delay and the detection rate, because starts within a month overlap, and farms are not
  resampled. About 18 months of starts give 18 resampling units, which the write-up states. Rule 7,
  descriptive contrasts, each labelled so: 60-day against 30-day and 90-day windows; above-rated
  half-hours against the half-hours at 20% or more of the scale; the core's expected power against
  a constant expected power on the same half-hours; and starts before and after 2025-06-17, when the
  core's fitting data ended. No threshold for porting applies, because the experiment produces no
  forecast. The 25% test mostly checks whether the threshold lies above 0.75, so the 10% loss and
  the larger drifts are the informative results.
benchmark: >-
  Within-experiment contrasts only (Rule 7 above). No earlier experiment tests capacity inference.
row_set_id: >-
  none shared. All half-hours of W1 to W3 from 2019-09-01 to 2026-06-30 with an observed power value
  and an expected power from experiment 230's export, dropping the first weeks after a new meter as
  experiment 230 does. No valid time on or after 2026-07-01 is used.
metric: >-
  Detection delay in days (median, per farm, loss size, and window), the share of starts detected
  within 120 days, alarms per farm on the drift and un-injected runs, the share of days with a
  statistic by month, and the monthly ratio of observed to expected power above rated on the real
  series
production_status: not-for-production
adopted_in: none
code: ml-experiments/00233-wind-physics-core-part-2-capacity-availability/
---

**Planned; no result yet.** This page is the plan, committed before any code runs.

## Question

We ask whether the physics model of experiment 230 can act as a monitor for a wind farm's effective
capacity. If some turbines stop generating, does the gap between observed power and the power that
the physics model expects in strong wind show the loss quickly, and without false alarms? In strong
wind (above the speed at which a turbine reaches rated power) the power curve is flat, so a fall in
observed power cannot be blamed on a weather forecast error, and the gap measures capacity. The
experiment first tests the idea on a known answer, a loss that we inject into real observed power,
and then describes what the real series shows. The experiment forecasts nothing.

## Method, as pre-registered

- **Core:** the fitted B1 core and its ERA5 expected power, exported by the session that ran
  [experiment 230](../00230-wind-physics-core-part-1/result.md) (fitted on valid times before
  2025-06-17, and held frozen here). A monitor that refitted the core after a loss would absorb the
  loss, so the core is not refitted.
- **Expected power:** ERA5 arrives about 5 days late, so the statistic is computed after the fact.
  A live monitor would use the ERA5 values of 5 days earlier, which gives the same statistic with a
  5-day lag.
- **Injection:** multiplication matches the loss of whole turbines when wind is above rated. Farms
  with an export cap could hide a loss at strong wind, and W1's observed 99th percentile of about
  21 MVA against 26 MW registered may be such a cap, so the delays are an optimistic bound where a
  cap binds.
- **Selection effect:** the half-hours are chosen by ERA5 wind, so some have actual wind below rated
  and pull the ratio below 1 by an amount that varies with ERA5 error. The reference level absorbs
  the average of that bias, but not its seasonal variation.
- **Real-data description:** the monthly ratio of observed power to expected power on above-rated
  half-hours (months with no statistic are blank), and the monthly 99th percentile of observed
  power in MW or MVA, per farm. No cause is claimed.

## Checks and caveats fixed in advance

- **Departure from the design brief:** the brief asks for a rolling-window fit of the per-farm
  scale on above-rated rows. The ratio of sums is the least-squares scale on those rows, so the
  statistic is the same quantity without an iterative fit.
- **Calibration years:** the calibration starts on 2021-09-01 so that the reference level has two
  years of data before the first calibration day.
- **Tail calibration and curtailment cost:** not applicable, because no forecast is produced.
- **Anonymisation:** farms are W1 to W3. No farm's identity, coordinates, or series is shown.
