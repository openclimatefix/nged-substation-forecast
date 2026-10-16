---
id: "00059"
slug: sw2-13-several-windows-in-one-detector
status: planned
size: medium
hypothesis: >-
  One detector that combines the step statistic of experiment 43's detector D0 at windows of
  4, 12, 24, and 48 hours, after calibrating each window to the same false-alarm rate on odd
  months, finds more of the injected load transfers, weighted by the MW each moves, than the best
  single window at the same false-alarm rate on even months.
prediction: >-
  Of the headline-band injected transfers that start in even months, the best single window (24
  hours, from experiment 43) finds 0.30 to 0.40 of the moved MW when the transfers follow a constant
  MW or the receiver's profile. The maximum over the four calibrated windows adds +0.02 to +0.06 to
  that share, and the sum of the four calibrated statistics adds -0.01 to +0.04, because a sum lets
  weak evidence at several windows add up but also lets one noisy window raise the false-alarm
  rate. Most of the gain is in transfers whose step the 24-hour window sees only partly, such as
  those lasting 1 to 2 days.
decision_rule: >-
  Two planned contrasts, read at 97.5% (Bonferroni across C1 and C2) from 20,000 resamples of whole
  calendar months, over the headline-band (25% to 40% of the receiving load, over 1 day) injected
  transfers that start in even months, with the constant-MW and receiver-profile shapes pooled and
  each transfer weighted by the MW it moves. C1 is the maximum detector minus the best single
  window, in the MW-weighted share found. C2 is the sum detector minus the best single window. The
  best single window is the one of the four with the highest MW-weighted share found on the
  odd-month transfers of the same two shapes, chosen before any even-month score is read. Every
  detector and window is calibrated to the same count of unexplained alarms on odd months (80, the
  count of experiment 43's 24-hour D0). A contrast is `supported` when its interval lies wholly
  above zero, its point value is at least +0.02, and the detector's even-month unexplained-alarm
  count is no more than 10% above the best single window's; `refuted` when the upper end lies below
  +0.02; `inconclusive` otherwise. The threshold of +0.02 is set by cost: a combined detector adds
  three more shift computations and one calibration per window to the pipeline, with no new data
  and no model, so the gain must pay for four calibrations, and a smaller gain is under what 620
  transfers resolve. The experiment's verdict is `supported` if either contrast is `supported`,
  `refuted` if both are `refuted`, else `inconclusive`.
benchmark: >-
  The best single window of D0 at the same odd-month unexplained-alarm count, and a detector that
  raises no alarm.
row_set_id: >-
  none (this experiment scores detector alarms against injected transfers, as experiments 43, 55,
  and 58 do, not forecasts)
metric: >-
  The share of the moved MW whose transfer is found (onset within 2 hours and end within the larger
  of 6 hours and 10% of the duration), weighted by the MW the transfer removes from the donor over
  its first 24 hours; the unweighted share found is second; the even-month unexplained-alarm count.
code: ml-experiments/00059-sw2-13-several-windows-in-one-detector/
---

# Experiment 59: does one detector over several windows find more transfers than the best single window?

**Plan, written before any code runs.** Experiment 43 found that the better detector depends on the
window, and experiment 58 found that the detector misses about half of the transfers whose step is
4.0 noise units or more. This experiment asks whether combining windows finds more of them, and
classifies why the large transfers it still misses are missed.

## Design

- **Detector.** For each of four windows (8, 24, 48, and 96 half-hours, which are 4, 12, 24, and 48
  hours), compute experiment 43's level-shift statistic `z_w` per series, in units of the series'
  odd-month noise scale. The single-window detector at window `w` raises an alarm at each local
  peak of `|z_w|` above a threshold `T_w`, exactly as experiment 43's D0 does.
- **Calibration.** `T_w` is the lowest threshold on a grid of 0.05 whose odd-month unexplained
  alarms (alarms with no logged closure within 24 hours on the series) number at most 80. The same
  search sets the thresholds of the two combined detectors. The windows then have the same
  false-alarm rate by construction.
- **Maximum detector.** The score at a stamp is `max_w |z_w| / T_w`. An alarm is a local peak of the
  score within 24 hours above a threshold `lambda_max` calibrated to 80 odd-month unexplained alarms.
- **Sum detector.** The score is `|sum_w z_w / T_w|`, with the sign kept so that windows disagreeing
  in direction cancel. An alarm is a local peak within 24 hours above `lambda_sum`, calibrated the
  same way.
- **Interval.** The end of the transfer comes from the window that alarms: the window with the
  largest `|z_w| / T_w` at the alarm stamp. The interval builder is experiment 43's, run with that
  window. An alarm whose window is 24 hours or longer is dropped near a monthly refit, as in
  experiment 43.
- **Injected transfers.** The 1,900 transfers of experiment 58 (160 + 1,000 + 300 in the headline
  band, and the other eight bands), injected into the power with three shapes: a share of the
  donor's profile (the harness default, which experiment 58 scored), a constant MW, and a share of
  the receivers' profile. Each shape keeps the same donor, receivers, onset, end, and mean MW. The
  headline pools the constant-MW and receiver-profile shapes, which neither detector assumes; the
  donor-profile shape is the continuity check with experiment 58.
- **Labelled real transfers.** The labelled real transfers that last over 1 day are scored on the
  real residual for each detector, as counts without a verdict.
- **Even months only.** The odd months fix the noise scales, the thresholds, and the choice of the
  best single window. Every score read for a verdict uses transfers that start in even months.

## Reported beside the verdict

- **Recall by donor step in noise units, split by metered receiver**, for the best single window and
  both combined detectors, in the bins experiment 58 used.
- **The MW-weighted share found first and the unweighted share second**, with the share of the moved
  MW that each detector misses, stated in the take-home message.
- **Why the large transfers are missed.** For every transfer with a metered receiver whose donor
  step is 4.0 noise units or more (24-hour scale), the miss is assigned one reason, in this order:
  the end is outside tolerance (an alarm has a matching onset); the alarm was absorbed (the score
  exceeded the threshold within 2 hours of the onset on a series of the transfer, but a larger
  peak within the suppression window took the alarm); the onset is outside tolerance (an alarm on a
  series of the transfer lies within 24 hours of the onset but not within 2 hours); or no alarm
  (the score never exceeded the threshold near the onset, with the median score as a share of the
  threshold). This addresses experiment 58's finding that 134 of 306 such transfers were missed.
- **Each shape separately**, and the even-month unexplained-alarm counts.

## Sizing (the five triggers)

1. Stores: private parquet tables only, nothing in Delta. 2. Serving path: no. 3. Degradation rule:
no. 4. More than one defensible design: yes (maximum or sum, the choice of windows, the
calibration), so both combinations and all four windows are fixed here. 5. Spans code whose callers
cannot be named: no, the code is a copy of experiment 58's harness in this folder. The size is
medium: one Sonnet plan review, a Sonnet code review, and a Sonnet prose review.

## Earlier advice followed and departed from

- Follows: score on the metered-receiver and no-metered-receiver groups separately; report recall by
  donor step in noise units; match thresholds by recounting unexplained alarms on odd months;
  report even-month unexplained-alarm counts; use several hundred headline-band transfers.
- Departs from: nothing.
