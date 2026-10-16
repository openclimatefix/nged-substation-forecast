---
id: "00069"
slug: sw2-17-missed-onsets-of-the-end-rule
status: planned
size: medium
hypothesis: >-
  Most of the 24 labelled real transfers over 1 day that raise no alarm at their onset have a step
  that is real but under D0M's threshold of 4.0 noise units, and an onset rule that accepts a
  lower-threshold alarm only when a matching opposite step follows within 14 days recovers more of
  the moved MW than D0M with the same end rule, at the same number of unexplained alarms.
prediction: >-
  The classification puts at least 12 of the 24 unalarmed real transfers in the classes whose
  strongest step lies between half of D0M's threshold and the threshold, and at most 4 in the data
  gap class. On injected transfers (shapes S2, S3, and S4 pooled, even months), the confirmation
  step finds PF14 minus PE14 at +0.02 to +0.06 on a fresh draw of experiment 58's headline-band
  transfers (A1) and +0.10 to +0.40 on a fresh draw of the long set (A2), both smaller than
  experiment 66's post hoc +0.042 and +0.412 because a rule chosen after seeing a draw does better on
  that draw than on a new one. The paired low-threshold onset (PP) gains +0.04 to +0.12 over the
  benchmark on the fresh headline-band transfers (C1) and -0.03 to +0.04 on the fresh long set (C2).
  On the 35 labelled real transfers, PP finds 2 to 6 more than the benchmark's 8.
decision_rule: >-
  Four planned contrasts, each the paired difference in MW-weighted recall (the share of the moved
  MW that sits in found transfers), pooled with equal weight over the injected shapes S2, S3, and S4
  and read on transfers that start in even months. Intervals are 98.75% (Bonferroni across the four
  contrasts, 1 minus 0.05 over 4), from 20,000 resamples of whole calendar months. A1 and A2 are PF14
  minus PE14 on the fresh headline-band draw and the fresh long draw (new draws, new injection
  seeds). C1 and C2 are PP minus the benchmark end rule on the same two draws. The benchmark end
  rule is PF14 when the lower ends of both A1 and A2 lie above zero, and PE14 otherwise; PP uses the
  benchmark end rule, and the matching PE14 or PF14 version of PP is the one read. The verdict
  concerns PP only. It is `supported` when the lower end of C1 lies above zero, the point value of
  C1 is at least +0.02, and the point value of C2 is at least -0.02. It is `small gain` when the
  lower end of C1 lies above zero and its point value is under +0.02. It is `refuted` when the upper
  end of C1 is under +0.02. It is `inconclusive` otherwise, including a C1 that passes while the
  point value of C2 is under -0.02. The threshold of +0.02 is the smallest gain that justifies a
  second tuned threshold and a pairing step in a detector that serves only training-time masks and
  the event table: it is one transfer in 50 of the moved MW. The classification and the real
  transfers carry no verdict and no interval, because 35 transfers are too few.
benchmark: >-
  D0M at its threshold of 4.0 noise units (80 odd-month unexplained alarms) with the end rule chosen
  by A1 and A2: PE14 (experiment 61's strongest opposite step within 14 days) or PF14 (the first
  opposite step of at least half the onset step, otherwise the tracked end).
row_set_id: >-
  none (this experiment scores detector alarms against injected and labelled real transfers, as
  experiments 43, 55, 58, 60, 61, and 66 do, not forecasts)
metric: >-
  MW-weighted recall (headline), pooled over S2, S3, and S4 on even months at the matched
  unexplained-alarm count. Unweighted recall, the share of the moved MW missed (split into no alarm
  at the onset and a wrong end), recall by donor step in noise units split by metered receiver, and
  the count of the 35 labelled real transfers over 1 day found follow.
code: ml-experiments/00069-sw2-17-missed-onsets-of-the-end-rule/
production_status: not-for-production
adopted_in: none
---

# Experiment 69 (SW2-17, part of the switching-event detector series): why do real transfers raise no alarm at the start?

**Planned; this page is rewritten with the results.** We ask why 24 of the 35 labelled real load
transfers lasting over 1 day raise no alarm at their start, and whether an alarm rule that accepts a
weaker step when a matching step back follows recovers them without raising the number of false
alarms. A load transfer is a switching operation that moves part of one Primary substation's demand
to neighbouring substations for some hours or days. A Primary substation is the 11 kV or 6.6 kV
substation that feeds local distribution circuits. The detector D0M raises an alarm when the
24-hour average of a substation's residual (measured power minus an XGBoost forecast that uses no
past power) steps away from the previous 24-hour average by 4.0 noise units or more. Experiment 66
found that ending each alarm interval at the first opposite step of at least half the onset step
(PF14) beat a search for the strongest opposite step within 14 days (PE14), but chose that rule after
seeing the results. We first re-test that rule on newly drawn injected transfers. We then sort the 24
unalarmed real transfers by cause and test the fix the sorting points to.

## Size

**Medium: one trigger fires, the rest do not.**

- **What gets stored:** private tables under `data/ml-experiments/`; no production table.
- **Production serving path:** not touched. The end rule and the paired onset use up to 14 days of
  hindsight, so they serve training-time masks (SW2-12) and the event table for NGED, not live
  forecast inputs.
- **Degradation rule:** not touched.
- **More than one defensible design:** yes. A lower onset threshold confirmed by a return step is the
  leading candidate, but the classification may point to another cause.
- **Code whose callers are hard to name:** no. The scripts copy experiment 66's harness and add the
  classification and the paired onset.

Per the brief, Sonnet writes the plan, implements, runs, and reviews; Opus is not used.

## Earlier advice followed and departed from

- **Followed:** keep the 24-hour window (experiment 66, C5); confirm PF14 on a fresh draw before using
  it as a baseline (experiment 66); test on transfers over 15.6 days as well as the headline band
  (experiment 66); report recall by donor step split by metered receiver (experiments 58 and 60); look
  at why the real transfers raise no alarm (experiments 60 and 61).
- **Departed from:** experiment 61 advised keeping the onset a single step and not folding the return
  step into the alarm score, because a threshold that keeps 80 unexplained alarms needed a lone onset
  step of about 8.8 noise units. This experiment folds the return step in only for weak onsets, and
  tunes the strong-onset threshold to the same alarm count, so the cost of the departure is read
  directly from the matched-rate contrast C1.

## What we will do

**Step 1, confirm the first-return rule.** We draw new injected transfers with new seeds: 1,160
headline-band transfers (25% to 40% of the receiving load, lasting over 1 day, drawn as experiment 58
drew its extra set, seed 6900) and 400 long-set transfers (15.6 to 60 days, drawn as experiment 66
drew the long set, seed 7900). Each is injected under the five shapes of experiment 60, and the
headline pools S2, S3, and S4, the shapes none of the detectors assumes. D0M, PE14, and PF14 run at
D0M's threshold of 4.0, which gives 80 unexplained alarms on odd months. A1 and A2 read PF14 minus
PE14.

**Step 2, sort the 24 unalarmed real transfers.** A real transfer has an alarm at its start when D0M
(24-hour window, threshold 4.0) raises an alarm on one of its series within 4 half-hours (2 hours) of
its labelled start. Each of the other transfers goes into the first of these classes whose test it
meets, using the step statistic `z` (the 24-hour step in noise units, as above):

1. **Data gap:** `z` is missing (under 75% of either 24-hour window present) at every stamp within 4
   half-hours of the start on every series of the transfer.
2. **Monthly refit:** the largest |`z`| within 4 half-hours of the start is at least 4.0, but a
   monthly refit of the residual model lies within 48 half-hours of the stamp, which suppresses the
   alarm.
3. **Alarm outside the tolerance:** D0M raises an alarm on a series of the transfer between 4 and 96
   half-hours (2 hours and 2 days) from the labelled start. A second count splits this class into
   alarms within 12 hours and later alarms.
4. **Peak suppressed:** the largest |`z`| within 96 half-hours is at least 4.0 but D0M raises no alarm
   there, because a larger peak within the window or a refit takes the alarm.
5. **Sharp step under the threshold:** the largest |`z`| within 4 half-hours of the start is
   between 2.0 and 4.0.
6. **Spread or late step under the threshold:** the largest |`z`| within 96 half-hours is between 2.0
   and 4.0 and lies more than 4 half-hours from the start.
7. **No step visible:** the largest |`z`| within 96 half-hours is under 2.0.

The write-up gives, per class, the count, the median moved MW (`moved_mw_p50` of the labelled set),
the median donor step in noise units, and whether the transfer has a metered receiver. Classes 5 and
6 are the classes the paired onset can recover. When they hold fewer than 12 of the 24 transfers, the
write-up says the paired onset targets a minority and names the fix the largest class points to; that
fix is a recommendation for the next experiment, because most classes (a data gap, a timing error in
the log) cannot be tested on injected transfers.

**Step 3, the paired low-threshold onset (PP).** PP raises every D0M alarm at a strong threshold
`T`. It also raises a weaker alarm at a series and stamp where the step statistic is a local peak
(the largest |`z`| within 24 hours either side) of at least `T/2` but under `T`, when a matching step
back follows. A matching step back is a stamp from 48 half-hours (24 hours) to 14 days later that is
a local peak with the opposite sign, at least `T/2` noise units, and between 0.5 and 2.0 times the
size of the onset step (the same ratio range as the balance test of detector D1). The first such stamp
counts. The interval's end comes from the benchmark end rule (PE14 or PF14), as in Step 1. `T` is
tuned on odd months, on a 0.05 grid, to the lowest value whose unexplained alarms (D0M alarms plus
accepted weak alarms, more than 24 hours from a logged closure on the series) number at most 80, so
that PP and the benchmark are scored at the same alarm count. The write-up reports `T`. A second
version with the weak threshold at `3T/4` is descriptive, with no contrast. The rule has one extra
parameter (the half), taken as the first value tried and not tuned.

**Scoring.** Everything is scored as in experiment 66. A transfer counts as found when an alarm
interval on one of its series starts within 2 hours of the true start and ends within the larger of 6
hours and 10% of the duration. The odd-month and even-month recall, unweighted recall, the missed MW
split into no alarm at the start and a wrong end, and recall by donor step split by metered receiver
follow, as does the count of the 35 labelled real transfers found, with the alarm counts on the real
residual. Before the fresh draws are scored, a harness check reproduces experiment 66's PE14 and PF14
`found` flags on the 1,900 transfers of experiments 43 and 58 under all five shapes; the run stops
when any flag differs.

## Limits to keep in mind

- **The fresh draws re-use experiment 66's code,** so PF14's definition cannot change between the
  draw that suggested it and the draw that tests it.
- **PP and the benchmark are scored on the same fresh draws that choose the benchmark end rule.** The
  contrast C1 is paired, so the choice moves both arms; the write-up says so.
- **Injected transfers are clean boxcars,** so a return step is easier to find than a real one.
- **The real-transfer count of 35 is too small for an interval,** so it carries no verdict.
