---
id: "00072"
slug: p35-step-loss-monitor-for-solar-farms
status: planned
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
  farm only, and only the injected farm's alarms are scored. The other farms' signals are built from
  their unmodified power, and the injected farm's own power never enters its own neighbour median.
  The headline injections start from April to September, when a farm has about 4 valid days a week,
  so that a 30-day loss holds enough valid days for a segment of 14 valid days. Injections that
  start from October to March are reported separately and decide nothing. The monitor and both
  baselines each have one threshold, tuned to give 0.35 false alarms per farm-year on a
  week-shuffled placebo built with the tuning seeds, then frozen in a committed file and scored on
  20 draws with other seeds. A detection is a downward alarm of at least 5% on the injected farm
  whose day lies from the loss's first day to 28 days after it. The window is 28 days because a
  farm has 2 to 4 valid days a week, so a loss is first visible some days after its onset. A false
  alarm is a downward alarm of at least 5% on the injected farm that lies more than 14 days before
  the loss's first day and more than 30 days after the loss's last day. Alarms inside the loss's
  span, and within the 14 days before it, are neither detections nor false alarms. A rate is the
  pooled count over all farms and draws, and a farm-year counts only the days from the farm's first
  valid day. Two contrasts decide the verdict, each a difference in detection rate on losses of
  10% lasting 30 and 90 days that start from April to September. C1 is the monitor minus the
  rolling median, and C2 is the monitor minus HYST. The interval of each is a 97.5% percentile
  interval (Bonferroni across the two) from 10,000 resamples of the 5 farms, with the 20 draws of
  each farm kept together inside it. It is a rough guide, because 5 farms are few and the farms
  share one satellite box, so the per-farm sign of each contrast is reported beside it. The
  verdict is the first of these that applies. Verdict `invalid` when a self-test fails. Verdict
  `inconclusive` when fewer than 4 of the 5 injected farms yield a score. Verdict `refuted` when
  any stop rule holds: the pooled detection rate of 25% losses lasting 30 days is under 80%; the
  upper limit of C1 or C2 is at or below 0; the false-alarm rate on the scored placebo exceeds 1
  per farm-year; or none of farm A's three low months is found. A low month counts as found when
  the fitted level sits at least 5% below the farm's median level on at least half of the month's
  valid days, where the median is taken over the farm's signal with its three low months masked.
  Verdict `supported` when the pooled detection rate of 10% losses lasting 90 days is at least
  80%, the lower limits of C1 and C2 are above 0, the false-alarm rate on the scored placebo is at
  most 0.5 per farm-year, all three of farm A's low months are found, and the unmodified record,
  with farm A's low months set aside, gives at most 1 downward alarm per farm-year. Verdict
  `partly supported` in every other case. There is no threshold on the effort of porting, because
  the monitor is an offline table and the production forecast does not read it. The thresholds of
  80% at 25% and 30 days, and 10% at 90 days, come from the aim in the design review: to find losses
  of 10% or more, which is a lost inverter at farms of 1 to 18 MW. Experiment 45 (PVDIS-4) found a
  weekly capacity series no better than a capacity constant within each season-year and advised
  detecting steps only above 25% to 30% of capacity, so the 25% stop rule is the floor and the 10%
  condition is the aim.
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
---

# Experiment 72 (P35): can a step-loss monitor find a lost inverter at the solar farms?

**This page is the plan, committed before any code runs.** The results and the take-home message
replace this paragraph when the experiment finishes.

## Question, in plain words

**We asked whether the power telemetry of a solar farm shows that the farm has lost part of its
generating capacity, for example because one inverter failed.** We add known losses to the real
power of a healthy farm and check how many of them the monitor finds, how often it raises a false
alarm on data with no loss, and whether it finds the three months in which farm A really did run low.

## Size, as an answer to the five triggers

- **What gets stored:** results tables and figures under `data/ml-experiments/` and `docs/`, and no
  predictions or model.
- **The production serving path:** untouched.
- **A degradation rule:** untouched.
- **More than one defensible design:** yes, so this plan fixes the monitor before any result and
  tunes only one threshold per monitor, on a separate draw.
- **Callers that cannot be named:** none. The scripts import `studies.capacity_estimators`,
  `studies.capacity_recovery`, and `studies.capacity_seasonal`, which the two earlier capacity
  studies already review.

The size is medium. It gets a Sonnet plan review, a Sonnet code review, a Sonnet check of the
write-up against the printed report, and a Sonnet prose review.

## Advice from earlier experiments that this plan follows

- **Follows experiment 45's advice** to keep the season-year constant as the capacity series and
  add step detection only above 25% to 30% of capacity. The monitor reports a level relative to
  that constant, and the 25% stop rule is the floor.
- **Follows experiment 45's advice** to score against an estimator that detects nothing.
- **Follows experiment 45's advice** to build a false-alarm unit on a placebo and not on the real
  series: the placebo shuffles whole weeks across years, which removes real steps and keeps the
  noise and the season.
- **Departs from experiment 45** in using the 6 solar farms and not the Primary substations, and in
  using a daily ratio and an L0 change-point fit and not a weekly capacity fit with an asymmetric
  penalty. The farm record has 5 farms, so the intervals are wide and the verdict leans on the
  stop rules.
- **Experiment 45's weekly table (`capacity_weekly.parquet`) covers the Primaries and not the
  farms**, so the monitor reuses its method (injecting known losses into a real record, and a
  placebo) and not its table.

## The monitor

1. **Signal.** For each farm and day, the sum of power over the sum of the modelled output
   (`cams_shape`) on selected hours, then its logarithm. The selected hours have a clear-sky index
   (satellite irradiance over clear-sky irradiance) of at least 0.8 where the clear-sky irradiance
   is at least 50 W/m2, a modelled output between 0.3 and 0.85 of the farm's 99th percentile of
   modelled output (a proxy for "not clipped by the inverter"), no run of 6 or more equal readings,
   and a valid day (the capacity studies' rule). A day needs at least 3 selected hours.
2. **Neighbour normalisation.** Centre each farm's logarithmic ratio on its median over the whole
   unmodified record, with farm A's three low months masked, and compute these constants once. They
   are never recomputed after an injection. Then subtract the median of the other farms' centred
   ratios on the same day. The pool of neighbours holds farms A, B, C, D, and F, and farm E is
   never in it, because E has under 1 valid day a week and an export cap. A day with fewer than two
   other pool farms is dropped, and is not replaced by the single ratio.
3. **Change-point fit.** Optimal partitioning on the valid days with a squared-error cost, a
   minimum segment of 14 valid days, and a penalty of `c` times the robust variance times the
   logarithm of the number of valid days. Each value is first clipped to 3 robust standard
   deviations around the median of the 15 valid days centred on it, which keeps a step (the local
   median follows it). The fit reports a step only when the level changes by 5% or more.
4. **Baselines.** The rolling median is the centred 28-valid-day median of the same signal, and an
   alarm starts on the first day it is below minus the threshold and ends when it rises above
   half of the threshold. HYST is `hysteresis_level` of `studies.capacity_posthoc` on the exponential
   of the same signal, and an alarm starts when it begins to follow a drop. Each baseline has one
   threshold, tuned like the penalty multiplier of the monitor.
5. **Known limits, stated now.** The monitor removes no snow days and no negative-price hours, adds
   no trend term for degradation, and ignores the inverter-plateau channel of the design review. Each
   may raise false alarms in winter, which the results report by season.

## Known-answer tests

**Real-noise injection.** Multiply a farm's real power by `1 - f` from an onset date to the end of
the loss, for `f` of 3%, 10%, and 25%, and durations of 30 and 90 days. A separate run injects 10%
and 25% permanent losses. Each draw places, per farm, 6 temporary losses (one per cell of the
grid), spaced at least 150 days apart, with onsets drawn at random from days with a valid day;
half of the draws place all onsets from April to September and the other half from October to
March. A separate run per draw places one permanent loss per size. The tuning needs no injection.
The 20 scored draws use seeds that no tuning draw uses. The injection emulates an inverter loss
and cannot emulate a string loss on a farm that clips; the write-up says so.

**Week-shuffled placebo.** First set farm A's three low months to missing. Then permute whole weeks
of each farm's signal among the years that share a week of the year. It destroys real steps and
keeps the noise and the season. The tuning placebos and the scored placebos use different seeds
(the tuning set holds 20 shuffles and the scored set holds 20 others). A placebo is easier than the
real record, since it removes year-level drift, so each monitor's rate on the unmodified record is
reported beside its placebo rate.

**Real controls.** Farm A's three low months must be found. Farm E, whose export cap is masked in
`power_mw`, is scored on the unmodified record only and must show no alarm.

**Self-tests, run before any scoring.** A constant signal gives no alarm. A signal with a planted
25% step and no noise gives one alarm within one day of the step with a size of 25% (to within
1 percentage point). The same step with noise at 8% gives a size within 5 percentage points. A
signal with a planted rise gives no alarm. A failed self-test makes the verdict `invalid`.

## Planned outputs

- A table of detection rate by farm, loss size, and duration for the monitor and both baselines.
- A figure of detection rate against loss size, with intervals, and the false alarms per farm-year.
- A figure of the signal and the fitted level for each farm, with farm A's low months marked.
- The list of every downward change on the unmodified record, with an explanation for each.

## What the next experiment should answer

To be written after the results are read.
