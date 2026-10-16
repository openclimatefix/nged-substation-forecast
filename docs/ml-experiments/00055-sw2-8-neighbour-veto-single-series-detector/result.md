---
id: "00055"
slug: sw2-8-neighbour-veto-single-series-detector
status: planned
size: medium
hypothesis: >-
  Keeping every alarm of the single-series detector D0 at the 24-hour window, except those vetoed
  because at least two neighbouring Primary substations stepped the same way and none stepped the
  opposite way, lets the alarm threshold fall at the same rate of unexplained alarms, and the
  detector then finds more injected load transfers than D0 alone.
prediction: >-
  Experiment 43's D0 24-hour alarms are the benchmark. At a veto of 2.0 noise units the veto removes
  14 of the 80 unexplained odd-month alarms (18%) and 16 of the 137 alarms that a logged closure
  explains. The matched 24-hour threshold falls by 0.05 to 0.20 from 4.00, so the gain through the
  threshold is small. C1 (24-hour window, injected transfers of 25% to 40% of the receiving load
  lasting over 1 day, starting in even months) rises by 0.00 to 0.02 from about 0.35, with a point
  value near +0.01. Experiment 47 saw the odd-month share found change by -0.019 when its threshold
  rose by 0.22, which puts the gain from a fall of 0.05 to 0.20 near +0.005 to +0.015. The veto
  removes the alarm that found fewer than 10% of the transfers the benchmark finds. Of the alarms on the real residual, the confirmed tier (an opposite neighbour
  step that balances the alarm's step) is explained by a logged closure more often than the
  unconfirmed tier.
decision_rule: >-
  The headline is C1, D2 minus D0M in the share of injected transfers found, paired by transfer, on
  the 24-hour window and the headline band, over transfers that start in even months, read at 98.75%
  (Bonferroni across C1 to C4) from 20,000 resamples of whole calendar months. D2's veto strength and
  threshold are chosen on odd months and on the injected transfers that start in odd months, never
  on even months. `supported` when the C1 interval lies wholly above zero, the C1 point value is
  at least +0.01, and D2's even-month unexplained-alarm count is no more than 10% above D0's.
  `refuted` when the upper end of the C1 interval lies below +0.01. `inconclusive` otherwise. C2 to
  C4 and the real-event counts are reported without changing the verdict.
benchmark: >-
  D0M, D0 with its threshold chosen by the search D2 uses and no veto (so that neither detector
  gets a threshold step the other was not offered); D0 at experiment 43's thresholds (4-hour window
  6.0, 24-hour window 4.0) through experiment 43's harness, whose 600 injected transfers and outcomes the harness check must reproduce; D1 at its
  experiment 43 thresholds for reference; and a detector that raises no alarm.
row_set_id: >-
  none (this experiment scores detector alarms against injected and labelled transfers, as
  experiment 43 does, not forecasts)
metric: >-
  The share of injected transfers found (start within 2 hours and end within the larger of 6 hours
  and 10% of the duration), event F1 at the two precision bounds of experiment 43, the
  even-month unexplained-alarm count, and the share of found transfers whose receiver the alarm names.
code: ml-experiments/00055-sw2-8-neighbour-veto-single-series-detector/
---

# Experiment 55: does a neighbour veto help the single-series switching detector?

**Status: planned.** This page is the plan, committed before any recall is scored. The write-up
replaces it when the experiment has run.

## Question

**The single-series detector D0 raises an alarm when the 24-hour mean of one Primary substation's
residual (measured minus predicted power) steps up or down.** Experiment 43 found that the
neighbour-balance detector D1, which requires a neighbouring substation to step the opposite way,
finds fewer transfers than D0 at the 24-hour window, because about half the injected transfers have
no metered receiver. SW2-8 keeps D0's alarms and uses the neighbours only to annotate and to veto:

- **Every alarm stands**, unless it is vetoed.
- **A veto** applies when at least two neighbours (or the one neighbour, for a substation with only
  one) stepped the same way as the alarm by more than the veto strength, in units of the neighbour's
  noise scale, and no neighbour stepped the opposite way. A same-sign step on the neighbours is what
  a weather or calendar miss looks like, and a transfer between substations never produces it.
- **Confirmed alarms** have an opposite-sign neighbour step of more than 2.0 noise units whose sum
  balances the alarm's step within 0.4 to 2.5 times (D1's test). The confirming neighbours are the
  receivers the alarm names.
- **Unconfirmed alarms** are the rest, and keep their single-series confidence.

The veto removes only alarms, so at D0's threshold D2 raises a subset of D0's alarms. The gain can
come only through the threshold: with the vetoed alarms removed, the threshold can fall until the
unexplained-alarm count on odd months equals D0's, and the lower threshold finds more transfers.

## Design

- **Residual and harness.** The residual is experiment 17's `H0.5` residual as experiment 37 stored it
  (`panel.parquet`), the harness is experiment 43's (copied with the experiment number changed), and
  the detector changes, not the residual model. The fast residual (`residual + moved MW`) is therefore
  exact, as in experiment 47. The harness check must reproduce experiment 43's alarm counts
  (432, 444, 297, 110) and every one of its 600 injected-transfer outcomes before anything else runs.
- **Detector D2.** D0's alarms, annotated by the neighbour check above. The interval builder runs on
  the alarm series only (partners are not masked), so the headline contrast differs from D0 only by
  which alarms exist. The partners are used only to score receiver naming (C4 below is not this).
- **Ceiling check first.** Before drawing the larger sets, D0M alone (veto switched off, threshold
  matched as below) scores experiment 43's headline-band transfers that start in odd months. If D0M
  does not gain on D0 at experiment 43's threshold, no veto can pass the bar, and the write-up says
  so. The check reads odd months only and decides nothing.
- **Tuning on odd months.** The veto strength is chosen from 1.5, 2.0, and 3.0 noise units, and for
  each the threshold is the lowest on a 0.05 grid whose odd-month unexplained-alarm count does not
  exceed D0's (80 at the 24-hour window, 120 at the 4-hour window). The choice maximises the share of
  injected transfers found that start in odd months, 24-hour window, headline band. Ties go to 2.0.
- **Scoring on even months.** The chosen strength and threshold score the transfers that start in
  even months. The even-month unexplained-alarm counts of D0 and D2 are reported, because matching
  on odd months does not hold them equal.
- **Transfers.** Experiment 43's 600 transfers (160 in the headline band) are scored as they are.
  A further 1,000 headline-band transfers are drawn with the same generator and a new seed, which
  gives about 500 to 600 in even months against experiment 47's 89. A third set of 300 headline-band
  transfers is drawn with the onset within 1 day of a bank holiday in England outside Christmas and
  New Year, where a calendar miss is most likely and a wrong veto most costly. The original and extra sets are
  quiet on every series they use (no logged closure within 7 days, no alarm of experiment 43's four
  settings within 2 days), as in experiment 43. That mask removes the onsets where a series had a
  weather or calendar miss, so it understates how often a wrong veto happens. The holiday set
  therefore uses logged closures only, and the write-up counts the holiday onsets that the alarm
  part of the mask would have rejected.

## Veto details

- **Opposite step:** a neighbour counts as stepping the opposite way only above 2.0 noise units, so
  an opposite step of 1.9 does not block a veto.
- **Neighbours with no data:** a neighbour whose step window is all missing counts as a neighbour
  but cannot step, so a substation with two neighbours, one of them missing, cannot be vetoed.
- **Receiver naming:** a confirmed alarm names the transfer's other metered series, which is the
  donor when the alarm is on a receiver.
- **Reported beside the tiers:** the share of vetoed alarms that a logged closure explains, and
  the veto counts by substation (substation 7 has one neighbour, so one same-sign step vetoes it).

## Planned contrasts

| Contrast | Detector, window | Transfers | Reads |
|---|---|---|---|
| C1 | D2 minus D0M, 24 h | headline band, even months, all sets except the holiday set | the verdict |
| C2 | D2 minus D0M, 4 h | same | reported |
| C3 | D2 minus D0M, 24 h | C1's transfers that have a metered receiver | reported |
| C4 | D2 minus D0M, 24 h | C1's transfers that have no metered receiver | reported |

Also reported: the holiday set as counts (transfers D0 finds, D2 finds, and D2 loses to a veto); the
odd-month and even-month unexplained-alarm counts; the precision bounds and event F1 as in
experiment 43; the share of real alarms by tier and the share of each tier that a logged closure
explains; the share of found transfers with a metered receiver whose receiver the confirmed alarm
names; the labelled real transfers as counts without a verdict.

## Size

- **Stored:** private tables under `data/ml-experiments/`, nothing in a Delta table.
- **Serving path:** not touched. The detector runs offline on a residual that needs ERA5.
- **Degradation rule:** not touched.
- **More than one defensible design:** yes. The veto count, the veto strength, and whether the
  partners are masked in the intervals all admit alternatives, so the plan fixes them above.
- **Callers:** the harness is a copy of experiment 43's, and no caller is outside this folder.

This is a medium experiment: an Opus plan review, a Sonnet implementation, a second Sonnet code
review, and a Sonnet write-up review.

## Earlier advice followed

- Match thresholds by recounting unexplained alarms on odd months, and report the even-month count
  (experiment 47).
- Score several hundred headline-band transfers, because 89 cannot resolve a gain below 0.04
  (experiment 47).
- Score first on the transfers with a metered receiver and on those without (experiment 43).
- Reuse experiment 43's injection harness unchanged (experiment 43). One departure: the draw takes a
  count and an optional filter on the onset, so the extra sets reuse the generator.

## Before any recall is scored

`e00055_prerun.py` counts the veto's effect on alarms only. At experiment 43's thresholds the veto
(2.0 noise units) removes 14 of the 80 unexplained odd-month 24-hour D0 alarms and 29 of the 120
4-hour ones; at 1.5 noise units it removes 26 and 41, and at 3.0 it removes 6 and 15. The 24-hour
odd-month count of D2 reaches 80 at about 4.0 (2.0 units) or 3.75 (1.5 units), so the threshold is
expected to fall by 0.05 to 0.25.
