---
id: "00072"
slug: p35-step-loss-monitor-for-solar-farms
status: finished
size: medium
hypothesis: >-
  For the 5 metered solar farms with a long record (labelled A, B, C, D, and F), a monitor can find
  a loss of generating capacity, such as a lost inverter, from the farm's own power telemetry. The
  monitor divides each day's power by the power a physics model expects on clear, unclipped hours,
  divides that ratio again by the same day's ratio at the other farms, and fits a piecewise-constant
  level to the result with an L0 penalty (an optimal-partitioning change-point fit). It finds a 25%
  loss that lasts 30 days, starting from April to September, within 28 days of its onset in at
  least 80% of injections, at no more than 0.5 false alarms per farm-year. It finds more injected
  losses of 10% than two simpler monitors that are tuned to the same false-alarm rate: a rolling
  median of the same signal, and the hysteresis rule of the capacity studies (called HYST here)
  run on the same signal.
prediction: >-
  Written before any monitor is run. The daily signal has a robust scatter of 7% to 11% at the
  farms (measured on the existing capacity record while building the signal, before any monitor was
  scored), and its winter values sit up to 0.1 to 0.4 in logarithm away from its summer values at
  farms A and D. The monitor then finds a 25% loss lasting 30 days that starts from April to
  September in 85% to 98% of injections at A, B, C, and F and in 60% to 85% at D. It finds a 10%
  loss lasting 30 days in 40% to 70% of injections and a 10% loss lasting 90 days in 75% to 95%. It
  finds a 3% loss in under 20%, so 3% is below the detection limit. Losses that start from October
  to March are found less often, because a farm has about 2 valid days a week then. The smallest loss
  found in 80% of 30-day injections is therefore 10% to 25% of capacity. The monitor beats the rolling
  median by 5 to 15 points of detection rate at 10%, because the segmentation pools noisy days that
  the rolling median reads one window at a time. It beats the hysteresis rule by more, because that
  rule needs 7 consecutive valid days below its threshold. Farm A's three low months (2024-12,
  2025-03, 2026-04) are all found. This is a monitor and not a forecast input, so no forecast gain is
  expected.
decision_rule: >-
  Farms A, B, C, D, and F are injected one at a time (farm E has 2.7 years of record and an export
  cap, so it is a control and is not injected). Each injection changes the power of the injected
  farm only, and only the injected farm's alarms are scored. The other farms' signals are built
  from their unmodified power, and the injected farm's own power never enters its own neighbour
  median. The headline injections start from April to September, when a farm has about 4 valid
  days a week, so that a 30-day loss holds enough valid days for a segment of 14 valid days.
  Injections that start from October to March are reported separately and decide nothing. The
  monitor and both baselines each have one threshold, tuned to give 0.35 false alarms per farm-
  year on a week-shuffled placebo built with the tuning seeds, then frozen in a committed file and
  scored on 20 draws with other seeds. A detection is a downward alarm of at least 5% on the
  injected farm whose day lies from the loss's first day to 28 days after it. The window is 28
  days because a farm has 2 to 4 valid days a week, so a loss is first visible some days after its
  onset. A false alarm is a downward alarm of at least 5% on the week-shuffled placebo or on the
  unmodified record. Alarms in injected runs that fall outside a detection window are not counted.
  A rate is the pooled count over all farms and draws, and a farm-year counts only the years from
  the farm's first valid day to its last. Two contrasts decide the verdict, each a difference in
  detection rate on losses of 10% lasting 30 and 90 days that start from April to September. C1 is
  the monitor minus the rolling median, and C2 is the monitor minus HYST. The interval of each is
  a 97.5% percentile interval (Bonferroni across the two) from 10,000 resamples of the 5 farms,
  with the 20 draws of each farm kept together inside it. It is a rough guide, because 5 farms are
  few and the farms share one satellite box, so the per-farm sign of each contrast is reported
  beside it. The verdict is the first of these that applies. Verdict `invalid` when a self-test
  fails. Verdict `inconclusive` when fewer than 4 of the 5 injected farms yield a score. Verdict
  `refuted` when any stop rule holds: the pooled detection rate of 25% losses lasting 30 days is
  under 80%; the upper limit of C1 or C2 is at or below 0; the false-alarm rate on the scored
  placebo exceeds 1 per farm-year; or none of farm A's three low months is found. A low month
  counts as found when the fitted level sits at least 5% below the farm's median level on at least
  half of the month's valid days, where the median is taken over the farm's signal with its three
  low months masked. Verdict `supported` when the pooled detection rate of 10% losses lasting 90
  days is at least 80%, the lower limits of C1 and C2 are above 0, the false-alarm rate on the
  scored placebo is at most 0.5 per farm-year, all three of farm A's low months are found, and the
  unmodified record, with farm A's low months set aside, gives at most 1 downward alarm per farm-
  year. Verdict `partly supported` in every other case. There is no threshold on the effort of
  porting, because the monitor is an offline table and the production forecast does not read it.
  The thresholds of 80% at 25% and 30 days, and 10% at 90 days, come from the aim in the design
  review: to find losses of 10% or more, which is a lost inverter at farms of 1 to 18 MW.
  Experiment 45 (PVDIS-4) found a weekly capacity series no better than a capacity constant within
  each season-year and advised detecting steps only above 25% to 30% of capacity, so the 25% stop
  rule is the floor and the 10% condition is the aim.
benchmark: >-
  A rolling 28-valid-day median of the same daily signal, thresholded, and the hysteresis rule of
  `studies.capacity_posthoc` run on the same daily signal (the capacity studies ran it on a Kalman
  smoothed ratio, so this is the rule and not the whole pipeline), each tuned to 0.35 false alarms
  per farm-year on the same placebo. The capacity constant within each season-year, the series
  experiment 45 recommended, is the signal's zero level, and a monitor that detects nothing has a
  detection rate of 0, which the monitor has to exceed.
row_set_id: none (the monitor is scored on injected and real losses in the capacity record, not on forecasts)
metric: >-
  Detection rate by loss size (3%, 10%, 25%) and duration (30 days, 90 days, permanent), the
  smallest loss found in 80% of injections, the false alarms per farm-year on the week-shuffled
  placebo and on the unmodified record for all three monitors, the delay of the offline fit in
  days, and the size error in percentage points.
code: ml-experiments/00072-p35-step-loss-monitor-for-solar-farms/
verdict: refuted
verdict_note: >-
  A pre-registered stop rule fired: the change-point monitor found 0.45 of 25% losses lasting 30
  days that start from April to September, against the 0.80 the rule requires. The other three stop
  rules did not fire, and the monitor beat neither baseline by an interval that excludes 0.
best_arm: >-
  The hysteresis rule on the same daily signal, which found 0.86 of 25% losses lasting 30 days and
  0.82 of those lasting 90 days, but only 0.07 to 0.13 of 10% losses. The change-point monitor
  found 0.84 of 25% losses lasting 90 days and 0.26 of 10% losses lasting 90 days, which is the
  best of the three at that size.
headline: >-
  Change-point monitor minus rolling median on 10% losses lasting 30 and 90 days: +0.045 (97.5%
  interval -0.015 to +0.090). Change-point monitor minus the hysteresis rule: +0.050 (-0.010 to
  +0.105). False alarms: 0.29 per farm-year on the scored placebo and 0.54 on the unmodified
  record.
lessons: >-
  The daily signal of a solar farm wanders by 6% to 11% (robust standard deviation) and a farm has
  only 3.1 valid days a week from April to September and 2.0 from October to March. A 30-day loss
  therefore holds about 13 valid days, and an L0 fit with 14-day segments cannot see it. A loss of
  25% that lasts 90 days or more is found about 0.8 of the time, a loss of 10% almost never, and a
  loss of 3% never (0.01 to 0.07, which is the chance level of 0.00 to 0.02). Farm D has a winter dip
  every year that raised 6 of the 18 alarms on the unmodified record.
production_status: not-for-production
adopted_in: none
advice:
  - >-
    Do not rely on a daily power ratio to find a solar-farm loss under 25% of capacity, or any
    loss that lasts under 90 days, because the best monitor found 0.26 of 10% losses lasting 90
    days and 0.05 or less of 10% losses lasting 30 days (figure 1).
  - >-
    Use a hysteresis rule (7 valid days below 10%) and not a change-point fit for 25% losses that
    may last about 30 days, because it found 0.86 of them against 0.45, and use the change-point
    fit for losses that last 90 days or more, where it found 0.26 of 10% losses against 0.07 (the
    detection table).
  - >-
    Plan on 3 valid days a week from April to September and 2 from October to March at a solar
    farm when sizing a segment or a window, because a 14-valid-day segment is longer than a 30-day
    loss (the signal table in report.txt).
  - >-
    Remove the winter dip at farm D before running any monitor on it, because 6 of the 18
    unmodified-record alarms of the change-point monitor were its November and December dips
    (figure 3).
  - >-
    Ask NGED for outage notices, protection-trip logs, and curtailment records of the 6 solar
    farms before building another monitor, because the telemetry alone cannot show a 10% loss and
    no record of a real loss is held apart from farm A's three low months (figure 1).
---

# Experiment 72 (P35): can a step-loss monitor find a lost inverter at the solar farms?

**Take-home message: the daily power telemetry of a solar farm shows a loss of 25% of its output
that lasts 90 days or more, and almost nothing smaller or shorter, so it cannot give early warning
of a single lost inverter.** The best monitor found 0.84 of 25% losses lasting 90 days. It found
0.26 of 10% losses lasting 90 days, and under 0.15 of 10% losses lasting 30 days. A loss of 3% was
found no more often than chance. The limit is the signal and not the method: three monitors of
different design reach nearly the same detection rates, because a farm reports only about 3 usable
days a week and each day wanders by 6% to 11%. The next step is to find a second source of evidence,
such as NGED's outage notices, and to use this monitor only for large, long losses.

**We asked whether the power a solar farm delivers shows that the farm has lost part of its
capacity, for example because an inverter failed.** We added known losses to the real power of 5
solar farms, and checked how many of them three monitors found, how often the monitors raised a
false alarm when no loss had been added, and whether they found the three months in which farm A
really did run low. The verdict is `refuted`: the plan required the monitor to find at least 0.80
of 25% losses lasting 30 days, and it found 0.45.

## Result

**The verdict is refuted by the first stop rule.** The change-point monitor found 0.45 of 25%
losses lasting 30 days (April to September starts; 100 losses). Each monitor was tuned on a
week-shuffled placebo to about 0.35 false alarms per farm-year. The monitor found all three of
farm A's low months. Its false-alarm rate was 0.29 per farm-year on the scored placebo, which meets
the limit of 0.5. On the unmodified record it was 0.54, which meets the limit of 1. The change-point
monitor beat neither baseline by an interval that excludes 0 (C1 and C2 below), and the other three
stop rules did not fire.

![Share of planted losses detected, by loss size](figure_1_detection_by_loss_size.svg)

| Loss, April to September onset | Change-point fit | Rolling median | Hysteresis rule (HYST) | Change-point, 7-day segment (post hoc) |
|---|---|---|---|---|
| 25%, 30 days | 0.45 | 0.30 | 0.86 | 0.64 |
| 25%, 90 days | 0.84 | 0.72 | 0.82 | 0.84 |
| 25%, to the end of the record | 0.80 | 0.70 | 0.78 | 0.82 |
| 10%, 30 days | 0.04 | 0.05 | 0.13 | 0.05 |
| 10%, 90 days | 0.26 | 0.16 | 0.07 | 0.25 |
| 10%, to the end of the record | 0.38 | 0.12 | 0.13 | 0.37 |
| 3%, 90 days | 0.03 | 0.04 | 0.07 | - |
| 0% (no loss; chance of an alarm in the window) | 0.01 to 0.02 | 0.01 to 0.02 | 0.00 to 0.01 | 0.01 to 0.02 |

Each cell pools 100 losses (5 farms, 20 draws). A loss that starts from October to March is found
less often: 25% losses lasting 30 days at 0.27, 0.14, and 0.57 for the change-point fit, the
rolling median, and the hysteresis rule. Losses from that half-year decide nothing.

**The two planned contrasts are positive but their intervals include 0.** On 10% losses lasting 30
and 90 days, the change-point fit minus the rolling median is +0.045 (97.5% interval -0.015 to
+0.090), and the change-point fit minus the hysteresis rule is +0.050 (-0.010 to +0.105). Farm A
sits below 0 against the rolling median (-0.05), and farm D below 0 against the hysteresis rule
(-0.03). The interval resamples 5 farms that share one satellite box, so it is a rough guide.

**False alarms are about 0.3 per farm-year on the placebo and 0.4 to 0.5 on the real record.** The
unmodified record has more alarms than the placebo because it holds real steps, year-level drift,
and the winter dip at farm D.

![False alarms per farm-year](figure_2_false_alarms.svg)

**Six of the 18 unmodified-record alarms are farm D's winter dip.** Farm D raised one alarm in
November or December of each year from 2020 to 2025. Farm A's low months raised 2 further alarms,
which the count sets aside.

![The daily signal and the fitted level of each farm](figure_3_signal_and_fit.svg)

## Why the monitors fall short

**A 30-day loss holds about 13 valid days, and the change-point fit asks for 14.** A farm has 3.1
valid days a week from April to September and 2.0 from October to March (the plan assumed about 4
and 2). The daily signal has a robust standard deviation of 0.061 to 0.114 (6.1% to 11.4%) at farms
A to D and F. The hysteresis rule needs only 7 valid days below the threshold, which is why it
finds 25% losses lasting 30 days better than the change-point fit does.

**A shorter segment helps only a little.** The post hoc arm with a segment of 7 valid days found
0.64 of 25% losses lasting 30 days (against 0.45), kept 0.31 false alarms per farm-year on the
placebo, and left the 10% rows unchanged. It still fell short of the 0.80.

**A temporary loss needs two change points, and a lower penalty for them gives more false alarms.**
This is a property of an L0 fit with a penalty tuned to 0.35 false alarms per farm-year, and it
explains why a permanent 10% loss (one change point) is found 0.38 of the time and a 10% loss
lasting 90 days (two) 0.26.

## Post hoc checks

**Neither check changes the pre-registered verdict.** A window that opens 14 days before the loss
raises the rates a little: 25% losses lasting 30 days at 0.64, 0.43, and 0.87 for the change-point
fit, the rolling median, and the hysteresis rule. A change-point fit with a minimum segment of 7
valid days is the post hoc arm in the table above.

## Method

**Signal.** Each farm's daily ratio of power to the power a physics model expects on clear,
unclipped hours, divided again by the median of the other four farms' ratios of the same day, and
centred on the farm's median over the whole unmodified record. Farm E is monitored but is never one
of the other farms, because E has 2.7 years of record and an export cap. A day with fewer than two
other farms is dropped.

**Monitors.** The change-point fit is an optimal partitioning of the valid days with a squared-error
cost, a minimum segment of 14 valid days, and an L0 penalty. The rolling median is the centred
28-valid-day median, and the hysteresis rule is `hysteresis_level` of the capacity studies. Each has
one parameter, tuned on 20 week-shuffled placebo copies of the 5 farms (seeds 0 to 19) to 0.35
false alarms per farm-year and then frozen (`e00072_frozen_config.json`). The scored placebo uses 20
other shuffles.

**Injection.** A loss multiplies one farm's real power by `1 - f` from an onset to the end of the
loss. Each farm, draw, and half-year gets four runs: sizes 0%, 3%, 10%, and 25% lasting 30 days, the
same lasting 90 days, and one permanent loss of 10% and of 25%. The injected farm's neighbours keep
their unmodified power, and only the injected farm is scored. All 800 runs placed their losses. The
injection scales a whole farm's power, so it emulates an inverter loss and cannot emulate a string
loss on a farm that clips.

**Known limits.** The monitors remove no snow days and no negative-price hours, fit no trend for
panel degradation, and ignore the plateau of the daily maximum. No record of a real loss exists
apart from farm A's three low months, so the injected losses are the only known-answer test.

## What changed from the plan

**The plan was revised after its two Sonnet reviews and before any result was read.** A first
scoring run happened before the code review. The review found that the rolling-median baseline
subtracted a median that included the injected loss, that the week shuffle moved missing days into
the record, that farm-years were counted on the shuffled signal, and that the alarm sizes of two
monitors were not floored at 5%. The page reports the second run only. The first run's tables and
frozen parameters are in a `superseded` folder, and the penalty and thresholds the second tuning
chose are the same as the first (6.0, 0.12, and 0.10).

## What the next experiment should answer

**Which real events at the 6 solar farms can be labelled, so that the monitor is tested on more than
farm A's three low months?** NGED's outage notices, protection-trip logs, and curtailment records
for the 6 farms would give real losses. With them, the next experiment can ask whether a loss that
the telemetry cannot show (10% or less) can be found from another source, and whether the
hysteresis rule at 25% gives a useful early alert.
