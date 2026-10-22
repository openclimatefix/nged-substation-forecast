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
  Within-experiment contrasts only (Rule 7 above). No earlier experiment in this repository tested
  capacity inference.
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
  none of the pre-registered arms. A variant added after the results were seen, which uses every
  half-hour at 20% or more of the fitted maximum output, finds a 25% loss within 120 days at 100%,
  87%, and 100% of start dates at W1, W2, and W3, but it still fails the 9% drift test
headline: >-
  In a simulation that cut observed power by 25% from random dates, the planned monitor raised an
  alarm within 120 days at 50% (W1), 63% (W2), and 0% (W3) of the start dates. A smooth seasonal
  swing of 9% in observed power, with no loss, also raised extra alarm days at W1 and W2. The
  pre-registered rule for stopping is met.
lessons: >-
  Strong-wind half-hours are too rare for a 60-day monitor (2% of half-hours at W1, 4% at W2, and
  0.6% at W3), and the fitted power curve adds almost nothing over a constant on them.
take_home_message: >-
  A monitor that compares observed power with the wind-turbine model only in strong wind, where the
  power curve is flat, cannot show a 25% loss of capacity in time at any of the three wind farms,
  because strong-wind half-hours are rare: a trailing 60-day window holds enough of them to give a
  reading on only 21% of days at W1, 45% at W2, and 5% at W3. A variant added after the results were
  seen, which uses every half-hour at 20% of maximum output or more, finds a 25% loss with a median
  delay of 27 days at W1, 52 at W2, and 34 at W3, but a smooth 9% seasonal swing then raises false
  alarms, and a 10% loss is found at only 27% to 73% of start dates. The next step to test is a
  monitor on the whole power curve with a seasonal baseline; the real series also shows single
  months with low output (W2 in winter 2025 to 2026, W3 in 2024 to 2025) that nothing here explains.
advice:
  - >-
    Do not monitor wind-farm capacity from half-hours when the model expects at least 95% of maximum
    output alone, because those are at most 4% of half-hours and a 60-day window gives a reading on
    only 5% to 45% of days (Figure 1; the seasonal pattern is in Figure 3).
  - >-
    Use every half-hour at 20% of the fitted maximum output or more for a capacity monitor, with a
    seasonal baseline, because that variant found a 25% loss with a median delay of 27 to 52 days
    (Figure 1; the seasonal baseline is untested).
  - >-
    Do not credit the fitted power curve for a strong-wind capacity reading, because a constant
    expected power on the same half-hours gave the same shares and median delays at the 25% and 50%
    losses (Results).
production_status: not-for-production
adopted_in: none
code: ml-experiments/00233-wind-physics-core-part-2-capacity-availability/
---

**The planned monitor found a 25% loss within 120 days at only about half the start dates at W1
and W2, and at none at W3.** The verdict is **not supported**: the pre-registered rule for stopping
(a median delay above 60 days at 2 or more farms at a 25% loss) is met. W1, W2, and W3 are three
metered wind farms, anonymised.

## Question and answer

**We asked whether the fitted wind-turbine model of [experiment 230][e230] (called "the core"
below) can show that a farm has lost capacity.** The monitor divides observed power by the power the core expects, adds up both
over a trailing 60 days of strong-wind half-hours (those where the core expects at least 95% of
the farm's fitted maximum output, its "scale"), and divides by the same ratio before July 2024, so
that 1.0 means normal. An alarm is raised when that relative ratio falls below the lowest value seen
in 2021-09 to 2024-06.

**The planned monitor was too slow in a simulation.** We cut observed power by 10%, 25%, and 50%
from 30 random start dates and counted the days until an alarm. A 25% loss raised an alarm within
120 days at 50% of start dates at W1, 63% at W2, and 0% at W3 (29 start dates at W3, after one
that already alarmed was excluded).

**A variant added after the results were seen did better but is not trustworthy.** It uses every
half-hour at 20% of scale or more. It found a 25% loss within 120 days at 100%, 87%, and 100% of
start dates (26 at W3 after exclusions). A smooth seasonal swing of 9% in observed power with no
loss, the size seen in experiment 183's solar farms, raised extra alarm days at all three farms.

![Figure 1: share of start dates detected within 120 days, per farm and loss size](figure_1_detection_share.svg)

![Figure 2: relative ratio over time for the real series, a 9% seasonal swing, and a 25% loss](figure_2_ratio_series.svg)

![Figure 3: share of days with at least 4 qualifying half-hours, by calendar month](figure_3_days_with_enough_half_hours.svg)

![Figure 4: monthly observed over expected power on the real series](figure_4_real_monthly_ratio.svg)

![Figure 5: monthly 99th percentile of observed power, all three farms](figure_5_monthly_p99.svg)

## Results

- **Strong-wind half-hours are rare.** They are 2% of half-hours at W1, 4% at W2, and 0.6% at W3
  over 2019-09 to 2026-06. The plan predicted 3% to 14%, from counts that its own text gave
  wrongly; its stated counts imply 1.6% to 9%.
- **The planned monitor has a reading on few days.** A 60-day window holds enough strong-wind
  half-hours (at least 4 half-hours on each of at least 6 days) on 21% of days at W1, 45% at W2, and
  5% at W3. The post hoc rows give 97%, 97%, and 93%. W3's planned result is mostly a coverage
  failure, not a measure of sensitivity.
- **Alarm thresholds (planned rows):** 0.956 at W1, 0.832 at W2, and 0.926 at W3. The W1 threshold
  sits 4% below 1, so a slow drift of 4% or more alarms there.
- **A 25% loss, planned rows:** median delay 121.5 days at W1 (bootstrap 95% interval 79 to not
  reached), 105.5 days at W2 (77.5 to 162), and never at W3. Shares within 120 days: 0.50 (0.22 to
  0.76), 0.63 (0.37 to 0.88), and 0.00. The intervals resample 18 calendar months of start dates.
- **A 25% loss, post hoc rows:** median delay 27 days at W1 (24.5 to 36), 51.5 at W2 (42 to 75),
  and 34 at W3 (27 to 43). A 50% loss: 12.5, 27, and 15 days. A 10% loss is found within 120 days
  at 0.27 (W1), 0.60 (W2), and 0.73 (W3) of start dates.
- **Constant expected power on the planned half-hours** gave the same shares and median delays as
  the core at the 25% and 50% losses at every farm, and within 0.04 in share at the 10% loss. The
  fitted power curve adds almost nothing on the flat top.
- **9% seasonal swing, extra alarm days over two years:** planned rows 50 at W1 and 42 at W2, and 0
  at W3 (where only 5% of days have a reading); post hoc rows 10, 28, and 62. Swings of 15% and 25%
  raise 152 and 340 extra alarm days at W1 on the post hoc rows.
- **Real series with no loss, alarm days after 2024-07-01:** planned rows 0, 102, and 36 (W1, W2,
  W3); post hoc rows 13, 32, and 202. The monthly observed over expected power on the post hoc rows
  ranges from 0.71 to 1.20 (W1), 0.74 to 1.31 (W2), and 0.35 to 1.27 (W3). On the planned rows, W2
  falls to 0.39 in one month of winter 2025 to 2026, and W3 to 0.51 in one month of 2024 to 2025.
  No cause is claimed.
- **W1's output shows no step.** Its monthly 99th percentile is mostly 18 to 21 MVA and falls to 9
  to 15 MVA in some summer months, so it follows the wind.
- **Starts inside and outside the core's fitting data** (before and after 2025-06-17), post hoc
  rows, 25% loss: shares within 120 days of 1.00 and 1.00 at W1, 0.79 and 1.00 at W2, and 1.00 and
  1.00 at W3. Starts before 2025-06-17 sit inside the data the core was fitted on, so the
  after-2025-06-17 figures are the honest ones.

## Checks

- **No forecast is produced,** so forecast-error and tail-calibration checks do not apply.
- **Audit triggers** (listed in the `ml-experiment` skill): trigger 1 fired (the 9% control raises
  extra alarms at W1 and W2) and trigger 2 fired (the predicted detection within 60 days at W1 and
  W2 failed; the rarity of strong-wind half-hours explains it, shown in Figure 3 and by the post
  hoc rows). A fresh Sonnet auditor re-derived the shares, thresholds, median delays, and drift
  alarm days with its own code from the raw inputs and matched every one. It found no error in
  window indexing, loss timing, or missing-data handling. It noted that the calibration years
  overlap the core's fitting data, which tightens the thresholds, and that the loss-free run's
  alarm days are masked, which makes the delays slightly longer than a plain count.
- **Hold-out check of the threshold:** calibrating on 2021-09 to 2023-06, 51 of 104 days with a
  reading in 2023-07 to 2024-06 alarm at W1 and 50 of 50 at W3, so the minimum-over-years threshold
  is not stable out of sample there; at W2 none of 159 alarm.

## Caveats and what to try next

- **The post hoc rows are not pre-registered,** and they fail the 9% drift test at all three farms.
- **Lab conditions:** the core is frozen, the loss multiplies observed power, and an export cap
  would hide a loss at strong wind, so the delays are optimistic. Start dates that already alarmed
  without a loss are excluded, so the farms are compared on slightly different start sets.
- **Next:** a monitor on all half-hours above cut-in with a baseline that depends on the month, which
  may absorb the 9% swing (untested), then a check of the W2 and W3 low months against NGED's outage
  records. A learned correction of the ratio by wind speed is a candidate neural component; its
  prediction that could fail is that the 9% swing then raises no extra alarm at W1.

[e230]: https://github.com/openclimatefix/nged-substation-forecast/blob/ml-exp-wind-physics-core-part-1/docs/ml-experiments/00230-wind-physics-core-part-1/result.md
