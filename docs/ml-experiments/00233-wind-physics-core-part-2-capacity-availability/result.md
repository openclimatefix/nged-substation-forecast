---
id: "00233"
slug: wind-physics-core-part-2-capacity-availability
title: >-
  Can a fitted wind-turbine physics model show that a wind farm has lost 25% of its capacity within
  60 days, without raising false alarms on a slow seasonal drift?
theme: generator-forecasting
also_relevant_to: []
status: done
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
verdict: not supported
best_arm: >-
  none of the pre-registered arms. The post hoc arm that uses every half-hour at 20% of scale or
  more finds a 25% loss within 120 days at 100%, 87%, and 100% of start dates at W1, W2, and W3
headline: >-
  The planned statistic found a 25% loss within 120 days at 50% (W1), 63% (W2), and 0% (W3) of the
  30 start dates, and a 9% seasonal drift raised new alarms at W1 and W2. The stop rule fires.
lessons: >-
  Half-hours above rated speed are too rare for a 60-day monitor (2% of half-hours at W1, 4% at W2,
  and 0.6% at W3), and the core adds nothing over a constant on them.
take_home_message: >-
  A monitor that compares observed power with the physics model only in strong wind, where the power
  curve is flat, cannot show a 25% loss of capacity within 60 days at any of the three wind farms,
  because strong-wind half-hours are too rare (a 60-day window has enough of them on 21% of days at
  W1, 45% at W2, and 5% at W3). Using every half-hour at 20% of rated power or more finds a 25% loss
  within 60 days at the median start date at W1 and W3 and in 52 days at W2, but a smooth 9%
  seasonal drift then raises false alarms, and a 10% loss is found at only 27% to 73% of start
  dates. The next step is a monitor that uses the whole power curve and a seasonal baseline, and
  the real series already shows unexplained shortfalls (W2 in winter 2025 to 2026, W3 from late
  2024).
advice:
  - >-
    Do not monitor wind-farm capacity from half-hours above rated speed alone, because they occupy
    under 4% of half-hours and a 60-day window has a statistic on only 5% to 45% of days (Figures 1
    and 3).
  - >-
    Use every half-hour at 20% of the fitted scale or more for a capacity monitor, with a seasonal
    baseline, because that statistic found a 25% loss in a median of 27 to 52 days (Figure 1).
  - >-
    Do not credit the fitted power curve for a strong-wind capacity statistic, because a constant
    expected power gave the same detection shares as the core's (Figure 1).
production_status: not-for-production
adopted_in: none
code: ml-experiments/00233-wind-physics-core-part-2-capacity-availability/
---

**A monitor that compares observed power with the physics model only in strong wind cannot show a
25% loss of capacity in time, because strong-wind half-hours are too rare.** The verdict is **not
supported**, and the pre-registered stop rule fires. Using every half-hour at 20% of rated power or
more (a post hoc variant) finds a 25% loss quickly, but a smooth 9% seasonal drift then raises
false alarms.

## Question and answer

We asked whether the physics model of [experiment 230](../00230-wind-physics-core-part-1/result.md)
can show that a wind farm has lost effective capacity. We injected losses of 10%, 25%, and 50% into
the observed power of three wind farms (W1, W2, W3) from 30 random start dates, and measured how many
days passed before the ratio of observed to expected power, over a trailing 60 days of strong-wind
half-hours, fell below an alarm threshold set on earlier years. A 25% loss was found within 120 days
at 50% of start dates at W1, 63% at W2, and 0% at W3. The post hoc variant that uses every half-hour
at 20% of the fitted scale or more found it at 100%, 87%, and 100%. The result narrows a capacity
monitor to the busier part of the power curve with a seasonal baseline.

![Share of start dates detected within 120 days, per farm and loss size](figure_1_detection_share.svg)

![Relative ratio over time for the real series, a 9% drift, and a 25% loss](figure_2_ratio_series.svg)

![Share of days with at least 4 qualifying half-hours, by calendar month](figure_3_days_with_enough_half_hours.svg)

![Monthly observed over expected power on the real series](figure_4_real_monthly_ratio.svg)

![Monthly 99th percentile of observed power](figure_5_monthly_p99.svg)

## Results

- **Row set:** half-hours with the core expecting at least 95% of its fitted scale are 2% of
  half-hours at W1, 4% at W2, and 0.6% at W3 over 2019-09 to 2026-06. The plan predicted 3% to 14%,
  because experiment 230's rated-speed rule used the 100 m wind speed and this experiment uses the
  core's expected power.
- **Days with a statistic (60-day window, planned rows):** 21% (W1), 45% (W2), and 5% (W3). With the
  post hoc rows (20% of scale or more), 97%, 97%, and 93%.
- **Thresholds (60-day, planned rows):** relative ratio below 0.956 (W1), 0.832 (W2), and 0.926
  (W3). The W1 threshold sits only 4% below 1, so any slow drift of 4% or more alarms there.
- **25% loss, planned rows:** median delay 121.5 days (W1, bootstrap 95% interval 79 to not
  reached), 105.5 days (W2, 77.5 to 162), and no detection (W3). Shares detected within 120 days:
  0.50 (0.22 to 0.76), 0.63 (0.37 to 0.88), and 0.00. The interval is over 18 calendar months of
  start dates.
- **25% loss, post hoc rows:** median delay 27 days (W1, 24.5 to 36), 51.5 days (W2, 42 to 75), and
  34 days (W3, 27 to 43); shares within 120 days 1.00, 0.87 (0.64 to 1.00), and 1.00. A 50% loss is
  found in 12.5, 27, and 15 days. A 10% loss is found within 120 days at 0.27 (W1), 0.60 (W2), and
  0.73 (W3).
- **Constant expected power (planned rows):** the shares and median delays equal the core's at
  every farm, so the fitted power curve adds nothing on the flat top.
- **9% seasonal drift, new alarm days over two years:** planned rows 50 (W1), 42 (W2), and none at
  W3 (no statistic exists); post hoc rows 10, 28, and 62. Drifts of 15% and 25% raise 152 and 340
  new alarm days at W1 with the post hoc rows.
- **Real series, alarm days after 2024-07-01 with no injection:** planned rows 0, 102, and 36
  (W1, W2, W3); post hoc rows 13, 32, and 202. Monthly observed over expected power (post hoc rows)
  ranges from 0.71 to 1.20 (W1), 0.74 to 1.31 (W2), and 0.35 to 1.27 (W3). On planned rows W2's monthly ratio
  falls to 0.39 in one month of winter 2025 to 2026, and W3's falls to 0.51 in one month of 2024 to
  2025; on post hoc rows W3 stays near 0.8 through 2026.
  No cause is claimed. W1's monthly 99th percentile is mostly 18 to 21 MVA and falls to 9 to 15 MVA in some summer
  months, so it follows the wind and shows no step.
- **Starts before and after 2025-06-17** (the end of the core's fitting data) are in `detect.json`;
  the post hoc rows detect a 25% loss at all starts in both halves at W1 and W3.

## Checks

- **Median minus mean forecast, and the error of the mean forecast:** not applicable, because the
  experiment produces no forecast.
- **Audit triggers fired:** trigger 1 (the 9% drift control raises new alarms at W1 and W2) and
  trigger 2 (the prediction that W1 and W2 detect a 25% loss within 60 days failed; the explanation,
  that qualifying half-hours are too rare, is tested in Figures 1 and 3 and by the post hoc rows).
  The audit and its triage are recorded below.
- **Hold-out check of the threshold:** calibrating on 2021-09 to 2023-06, 51 of 104 days with a
  statistic in 2023-07 to 2024-06 alarm at W1 and 50 of 50 at W3, so the minimum-over-years
  threshold is not stable out of sample at those farms; at W2 none of 159 alarm.
- **Tail calibration and curtailment cost:** not applicable.

## Caveats and what to try next

- **Post hoc rows:** the 20%-of-scale variant was in the plan only as a descriptive contrast, so its
  numbers are post hoc and have not met the pre-registered drift rule (9% drift raises new alarms
  at all three farms).
- **Lab conditions:** the core is frozen, the loss is a multiplier of observed power, and an export
  cap would hide a loss at strong wind, so the delays are optimistic.
- **Next:** a monitor on all half-hours above cut-in with a seasonal baseline (the 9% drift is
  mostly absorbed by a baseline that is a function of month), then a check of the W2 and W3
  shortfalls against NGED's outage records. A learned correction of the ratio by wind speed is a
  candidate neural component; its prediction that could fail is that the 9% drift then raises no new
  alarm at W1.
