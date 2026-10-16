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
  loss that lasts 30 days within 14 days of its onset in at least 80% of injections, at no more than
  0.5 false alarms per farm-year. It finds more injected losses than two simpler monitors that are
  tuned to the same false-alarm rate: a rolling median of the same ratio, and the HYST2 hysteresis
  rule from the capacity studies.
prediction: >-
  Written before any monitor is run. The daily ratio has a scatter of 6.7% to 7.5% at farms A, B, C,
  and F and 12.8% at farm D (design review, measured on the existing capacity record). The monitor
  then finds a 25% loss lasting 30 days in 90% to 98% of injections at A, B, C, and F and in 60% to
  85% at D. It finds a 10% loss lasting 30 days in 40% to 70% of injections and a 10% loss lasting 90
  days in 75% to 95%. It finds a 3% loss in under 20%, so 3% is below the detection limit. The
  smallest loss found in 80% of 30-day injections is therefore 10% to 25% of capacity. The monitor
  holds to 0.5 false alarms per farm-year on a week-shuffled placebo after tuning, and beats the
  rolling median by 5 to 15 points of detection rate at 10% (the segmentation pools noisy days that
  the rolling median reads one window at a time). It beats HYST2 by less, because HYST2 already
  holds a level until a drop persists. Farm A's three low months (2024-12, 2025-03, 2026-04) are
  all found. This is a monitor and not a forecast input, so no forecast gain is expected.
decision_rule: >-
  Scores are medians over farms and draws of the detection rate, with farms A, B, C, D, and F
  injected one at a time (farm E has only 2.7 years of record and an export cap, so it is a control
  and is not injected). The monitor and both baselines each have one threshold, tuned on a tuning
  draw to give 0.5 false alarms per farm-year on a week-shuffled placebo, then frozen in a committed
  file and scored on 5 fresh draws. A detection is a downward change in the fitted level of at least
  5% whose date lies from 7 days before the loss starts to 14 days after (the window is asymmetric
  because a farm has only 2 to 4 valid days a week, so the first valid day after a loss falls late).
  A false alarm is a downward change of at least 5% that is not within 14 days before to 30 days
  after a planted loss. Two contrasts decide the verdict, each a difference in detection rate pooled
  over losses of 10% and 25% lasting 30 and 90 days, with a 97.5% interval (Bonferroni across the
  two) from 10,000 resamples of the 5 farms and the 5 scored draws as clusters. C1 is the monitor
  minus the rolling median, and C2 is the monitor minus HYST2. Verdict `refuted` when any of these
  stop rules holds: the 25%, 30-day detection rate is under 80%; the upper limit of C1 or C2 is at
  or below 0; the false-alarm rate on the scored week-shuffled placebo exceeds 1 per farm-year; or
  none of farm A's three low months is found. A low month counts as found when the fitted level
  sits at least 5% below the farm's median level on at least half of the month's valid days. Verdict
  `supported` when no stop rule holds and also: the 10%, 90-day detection rate is at least 80%; the
  lower limits of C1 and C2 are above 0; the false-alarm rate on the scored placebo is at most 0.5
  per farm-year; all three of farm A's low months are found; and the unmodified record, after the
  low months of farm A are set aside, gives at most 1 downward change of 5% or more per farm-year.
  Verdict `partly supported` when no stop rule holds and at least one of those conditions fails. A
  verdict `inconclusive` applies when a self-test fails or a monitor cannot be run on enough farms.
  There is no threshold on the effort of porting, because the monitor is an offline table and the
  production forecast does not read it. The thresholds of 80% at 25% and 30 days, and 10% at 90
  days, come from the aim in the design review: to find losses of 10% or more, which is a lost
  inverter at farms of 1 to 18 MW. Experiment 45 (PVDIS-4) found a weekly capacity series no better
  than a capacity constant within each season-year and advised detecting steps only above 25% to
  30% of capacity, so the 25% stop rule is the floor and the 10% condition is the aim.
benchmark: >-
  A rolling 28-valid-day median of the same neighbour-normalised ratio, thresholded, and the HYST2
  rule of `studies.capacity_seasonal` run on the same injected record, each tuned to 0.5 false alarms
  per farm-year on the same placebo. The capacity constant within each season-year, the series
  experiment 45 recommended, is the unmodified record's reference level, and a monitor that detects
  nothing has a detection rate of 0, which the monitor has to exceed.
row_set_id: none (the monitor is scored on injected and real losses in the capacity record, not on forecasts)
metric: >-
  Detection rate by loss size (3%, 10%, 25%) and duration (30 days, 90 days, permanent), the
  smallest loss found in 80% of injections, the false alarms per farm-year on the week-shuffled
  placebo and on the unmodified record, the detection delay in days, and the size error in
  percentage points.
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
   (satellite irradiance over clear-sky irradiance) above 0.8, a modelled output between 0.3 and
   0.85 of the farm's 99th percentile of modelled output (a proxy for "not clipped by the
   inverter"), no run of 6 or more equal readings, and a valid day (the capacity studies' rule). A
   day needs at least 3 selected hours.
2. **Neighbour normalisation.** Subtract the median of the other farms' same-day signals (at least
   two others must have one), then centre each farm on its median over the reference year. A day
   with fewer than two neighbours keeps the single ratio and is flagged.
3. **Change-point fit.** Optimal partitioning on the valid days with a robust cost (the daily
   values are winsorised at 3 robust standard deviations), a minimum segment of 14 valid days, and a
   penalty of `c` times the robust variance times the logarithm of the number of valid days. The fit
   reports a step only when the level changes by 5% or more.
4. **Known limits, stated now.** The monitor removes no snow days and no negative-price hours, adds
   no trend term for degradation, and ignores the inverter-plateau channel of the design review. Each
   may raise false alarms in winter, which the results report by season.

## Known-answer tests

**Real-noise injection.** Multiply a farm's real power by `1 - f` from an onset date to the end of
the loss, for `f` of 3%, 10%, and 25%, and durations of 30 and 90 days. A separate run injects 10%
and 25% permanent losses. Each draw places 6 temporary losses per farm, one per cell of the grid,
spaced at least 150 days apart, and one permanent loss in a separate run. The tuning draw and the 5
scored draws use different seeds. The injection emulates an inverter loss and cannot emulate a string
loss on a farm that clips; the write-up says so.

**Week-shuffled placebo.** Permute whole weeks of each farm's signal among the years that share a
week of the year. It destroys real steps and keeps the noise and the season.

**Real controls.** Farm A's three low months must be found. Farm E, whose export cap is masked in
`power_mw`, must show no alarm; farm E is not injected.

**Self-tests, run before any scoring.** A constant signal gives no change point. A signal with a
planted 25% step and no noise gives one change point within one day of the step. A zero-penalty fit
changes at every allowed place. A failed self-test makes the verdict `invalid`.

## Planned outputs

- A table of detection rate by farm, loss size, and duration for the monitor and both baselines.
- A figure of detection rate against loss size, with intervals, and the false alarms per farm-year.
- A figure of the signal and the fitted level for each farm, with farm A's low months marked.
- The list of every downward change on the unmodified record, with an explanation for each.

## What the next experiment should answer

To be written after the results are read.
