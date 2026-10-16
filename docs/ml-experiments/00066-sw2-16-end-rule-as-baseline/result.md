---
id: "00066"
slug: sw2-16-end-rule-as-baseline
status: planned
size: medium
hypothesis: >-
  Experiment 61's end rule (each alarm interval ends at the strongest opposite step within a cap of
  14 days) can be improved by a longer cap, a shorter step window, or both. A cap of 30 or 60 days
  finds more of the MW moved by transfers that last over 14 days, with a loss of at most 0.01 on the shorter transfers of experiment 58.
  The 4-hour step window finds more of the moved MW than the 24-hour window at the same number of
  unexplained alarms. The winning window and cap then become the benchmark for every later
  switching-detector experiment.
prediction: >-
  Pooled over the shapes S2, S3, and S4, in MW-weighted recall (the share of the moved MW that sits
  in found transfers). On a new set of transfers lasting 15.6 to 60 days (the long set), the 24-hour
  window with a 30-day cap gains +0.30 to +0.55 over the 14-day cap (C1), because the 14-day cap
  cannot end a transfer longer than 15.6 days within tolerance, and the 60-day cap gains
  +0.40 to +0.65 (C2), the extra +0.00 to +0.20 over the 30-day cap coming from the roughly 1 in 5
  long transfers that last over 30 days. On experiment 58's 620 transfers, a no-harm check rather
  than evidence of a gain, of which 17 of 1,160 headline-band transfers (all months) last over 14
  days and none over 30, the 30-day cap changes recall by -0.01 to +0.01 (C3) and the 60-day cap
  by -0.03 to +0.01 (C4), since a longer cap can choose a later and larger opposite step that
  belongs to another event. The 4-hour window with 80
  unexplained alarms changes recall by -0.15 to 0.00 against the 24-hour window (C5), because the
  4-hour step is noisier for the slow steps of a transfer that follows a daily profile. The 24-hour
  window with a 30-day cap is therefore the expected winner.
decision_rule: >-
  Five planned contrasts, read at 99% (Bonferroni across C1 to C5) from 20,000 resamples of whole
  calendar months, each the paired difference in MW-weighted recall pooled with equal weight over
  S2, S3, and S4, on transfers that start in even months. C1 and C2 are on the long set (400
  injected transfers of 25% to 40% of the receiving load lasting 15.6 to 60 days, with durations
  from the 26 labelled real operations that last that long), and C3 and C4 are on experiment 58's 620
  headline-band transfers. C1 is the 24-hour window with a 30-day cap minus the same window with a
  14-day cap, C2 is a 60-day cap minus the 14-day cap, C3 and C4 are C1 and C2 on the 620
  transfers, and C5 is the 4-hour window (threshold matched to 80 unexplained alarms on odd months)
  with a 14-day cap minus the 24-hour window with a 14-day cap, on the 620 transfers. The
  benchmark is the 24-hour window with a 14-day cap (PE of experiment 61). The window changes
  to 4 hours when the C5 interval lies wholly above zero and its point value is at least +0.01.
  A cap of c days replaces 14 when the interval of its contrast on the long set (C1 or C2) lies
  strictly above zero, its point value on the long set is at least +0.01, and its point value on
  the 620 transfers (C3 or C4) is at least -0.01. The caps are always chosen from the 24-hour
  contrasts C1 to C4. If the window changes to 4 hours, the cap carried forward is the one chosen at
  24 hours, and the write-up says it was not tested at 4 hours; the 4-hour cap contrasts are
  reported without a verdict. When both caps pass, the 60-day cap is used only if its recall on the
  long set exceeds the 30-day cap's by at least +0.01 (a comparison of point values). The verdict
  is `supported` when the window or the cap changes. It is `refuted` when neither changes and the
  upper ends of the C1, C2, and C5 intervals are all below +0.01, so that no change could be worth
  the cost (C3 and C4 only guard against loss and do not enter this test). Every other outcome,
  including a cap that passes on the long set and fails the guard, is `inconclusive`. The threshold of +0.01 is the cost
  of the longer hindsight a cap implies (a cap of 60 days means the last 60 days before a
  forecast cannot be masked) and of switching to the 4-hour windowed statistic; setting a cap
  costs nothing else to port.
benchmark: >-
  The 24-hour window with a 14-day cap (experiment 61's PE, at D0M's threshold of 4.0 noise units
  and 80 odd-month unexplained alarms), and D0M's own tracked end at each alarm setting.
row_set_id: >-
  none (this experiment scores detector alarms against injected transfers, as experiments 43, 55,
  58, 60, and 61 do, not forecasts)
metric: >-
  MW-weighted recall (headline): the share of the MW moved by injected transfers that sits in
  transfers the detector finds, where a transfer is found when an alarm interval starts within 2
  hours and ends within the larger of 6 hours and 10% of the duration. The share of the moved MW
  missed, unweighted recall, recall by donor step in noise units split by metered receiver, and the
  count of the 35 labelled real transfers over 1 day that are found follow.
code: ml-experiments/00066-sw2-16-end-rule-as-baseline/
---

# Experiment 66 (SW2-16, part of the switching-event detector series): which window and cap should the end rule use?

**Planned.** A load transfer is a switching operation that moves part of one Primary substation's
demand onto neighbouring substations for some hours or days. Experiment 61 found that ending each
detected interval at the strongest opposite step in the following 14 days finds 0.19 more of the
moved power than the tracker that experiment 43's detector used. This experiment asks whether a
longer search, or a shorter window for the step statistic, makes the end rule better, so that one
setting can serve as the benchmark for every later detector.

## Arms

All arms keep the alarms of D0M, the single-series detector that raises an alarm when the average
of a Primary's residual steps by a threshold number of noise units. They differ in the window of
the step statistic and in where each interval ends.

- **D0M** has its end from the level tracker, with a 60-day cap, at each of the three alarm
  settings below.
- **PE14, PE30, and PE60** take the end from the stamp between 24 hours and the cap (14, 30, or 60
  days) after the onset where the step back is largest, when that step is at least 1.5 noise
  units, as in experiment 61. Otherwise the tracked end stays.
- **Alarm settings.** The 24-hour window at 4.0 noise units (80 odd-month unexplained alarms); the
  4-hour window at 6.0 noise units (120 alarms, experiment 43's setting, descriptive); and the
  4-hour window at the threshold that gives 80 odd-month unexplained alarms, so that the two
  windows are compared at the same alarm rate (C5).

Twelve arms in all: three alarm settings, each with D0M and three caps.

## Populations

- **The 620 transfers** are experiment 58's headline band (25% to 40% of the receiving load,
  lasting over 1 day) that start in even months, in the five daily shapes of experiment 60. S2,
  S3, and S4 are the shapes no detector assumes and form the headline. Only 17 of the 1,160
  headline-band transfers last over 14 days and none last over 30, so these 620 cannot tell the
  caps apart.
- **The long set** is 400 new injected transfers of 25% to 40% of the receiving load, lasting 15.6
  to 60 days, with the same quietness rules as experiment 58's (no logged closure within 7 days
  and no experiment 43 alarm within 2 days, on every series the transfer touches). A fixed seed,
  chosen before any result is read, draws it. Each transfer takes its duration and its pair of
  substations from one of the 26 labelled real operations that last 15.6 to 60 days (5 of the 26
  last over 30 days, so about 1 in 5 of the long set does), and its onset is drawn uniformly. A
  transfer counts as an even-month transfer by the month of its onset. The lower limit of 15.6
  days is 14 days divided by 0.9: a shorter transfer has its end within the 10% tolerance of the
  latest end a 14-day cap allows, so every cap repairs it and it cannot separate the caps. The
  long set is injected in all five shapes and scored like the 620.
- **The labelled real transfers** over 1 day (35 transfers, 24 with no D0M alarm at the start) are
  scored as in experiment 61 and reported as counts only, with no interval. The long set takes its
  durations and substation pairs from the same labelled operations, so the real transfers and the
  long set are not independent evidence.

## Scores

Print for every arm, even months and odd months: MW-weighted recall by shape, the share of the
moved MW missed (split into MW with no alarm at the start and MW whose start is found but whose
end is wrong), unweighted recall, recall by donor step in noise units split by metered
receiver (the standard table of the switching line), recall by duration, and the count of
transfers whose start is found but whose end is wrong. The window and the cap are chosen on the
decision rule above from even months, as the contrasts are. Because that choice selects among the
arms on the even months, the chosen setting's recall is also reported on odd months, which the
selection does not bias. The end rule's 1.5 noise units and the alarm thresholds (4.0, 6.0) are
inherited from experiments 43 and 61 and are not retuned. The 4-hour threshold that matches 80
alarms is searched on odd months only. D0M and the three end rules at one alarm setting share
their alarms, so their alarm rate is equal by construction, and the two windows have the same rate
at the 80-alarm setting.

## Sizing

- **What gets stored:** private parquet tables in `data/ml-experiments/` only.
- **Production serving path:** not touched; the end rule needs up to the cap's worth of hindsight.
- **Degradation rule:** not touched.
- **More than one defensible design:** yes, a cap test needs transfers longer than 14 days, which
  experiment 58's set barely has, so the long set is a design choice.
- **Callers that need searching:** none; the code is a copy of experiment 61's harness, with three
  changes: the end rule takes the cap as an argument, the threshold search takes a target alarm
  count, and a new module draws the long set.

The size is medium. A Sonnet agent reviews this plan, a Sonnet agent reviews the code, and two
Sonnet agents review the write-up.

## Earlier advice followed

- Experiment 61: take the end from the strongest opposite step (followed), keep the onset
  single-step at 4.0 noise units (followed), test the 4-hour window and a longer cap before
  adopting (this experiment).
- Experiment 60: the standard donor-step table and non-matching shapes (followed).

## Caveat

The end rule needs up to the cap's worth of hindsight, so detectors built on it serve training-time
masks (SW2-12) and the event table for NGED, not live forecast inputs.
