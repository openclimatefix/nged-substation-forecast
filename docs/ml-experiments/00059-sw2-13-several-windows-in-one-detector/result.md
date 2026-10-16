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
  Of the headline-band injected transfers that start in even months, the best single window finds
  0.20 to 0.40 of the moved MW when the transfers follow a constant MW or the receivers' profile
  (earlier experiments measured only the donor-profile shape, so this range is wide). The best
  single window is the one odd months choose, and the prediction expects the 24-hour window.
  The maximum over the four calibrated windows adds +0.02 to +0.06 to that share, and the sum of
  the four calibrated statistics adds -0.01 to +0.04, because a sum lets weak evidence at several
  windows add up but also lets one noisy window raise the false-alarm rate. Most of the gain is in
  transfers whose step the 24-hour window sees only partly, such as those lasting 1 to 2 days.
decision_rule: >-
  Two planned contrasts, read at 97.5% (Bonferroni across C1 and C2) from 20,000 paired resamples
  of whole calendar months (both detectors use the same resampled months, and the MW-weighted share
  is a ratio of sums), over the headline-band injected transfers (25% to 40% of the receiving load,
  over 1 day) that start in even months (the UTC month of the onset stamp), with the constant-MW
  and receiver-profile shapes pooled and each transfer counted once and weighted by the MW it
  removes from the donor over its first 24 hours (the mean of the donor's injected loss over those
  48 half-hours). C1 is the maximum detector minus the best single window, in the MW-weighted share
  found. C2 is the sum detector minus the best single window. The best single window is the one of
  the four with the highest MW-weighted share found on the odd-month transfers of the same two
  shapes, chosen before any even-month score is read. Every detector and window is calibrated to
  the same count of unexplained alarms on odd months (at most 80, the count of experiment 43's
  24-hour detector); the achieved counts are reported. A contrast is `supported` when its interval
  lies wholly above zero, its point value is at least +0.02, and the detector's even-month
  unexplained-alarm count on the real residual is no more than 10 alarms above the best single
  window's (a count of 70 to 80 varies by about 9 from chance alone, so a percentage margin would
  fail a matched detector by chance); `refuted` when the upper end lies below +0.02; `inconclusive`
  otherwise. The threshold of +0.02 is set by the work of porting: a combined detector needs three
  more shift computations and four threshold searches, with no new data and no weather model, so a
  smaller gain is not worth four calibrations, and it is under what about 620 transfers resolve. The
  experiment's verdict is `supported` if either contrast is `supported`, `refuted` if both are
  `refuted`, else `inconclusive`.
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

**This experiment asks whether one detector that looks at several time spans at once finds more
injected load transfers than the best detector that looks at one time span.** The detector
(experiment 43's D0) raises an alarm when the average of one substation's residual (measured minus
predicted power) over a window steps up or down against the window before it. A transfer is load
moved from a donor substation to receiver substations. We inject known transfers into the power,
run the detector with windows of 4, 12, 24, and 48 hours and with two ways of combining them, and
count the share of the moved MW that each way finds. Every way is set to the same number of false
alarms on odd months, and the comparison is on even months. The plan was written before any code
ran.

## Why this experiment

Experiment 43 found that the better detector depends on the window, and experiment 58 found that
the 24-hour detector misses 134 of the 306 transfers that have a metered receiver and a donor step
of 4.0 noise units or more. This experiment asks whether combining windows finds more of them, and
classifies why each large transfer that is still missed is missed.

## Design

All windows below are given in hours, with half-hours in brackets: 4 hours (8), 12 hours (24), 24
hours (48), and 48 hours (96).

- **Single windows.** The level-shift statistic `z_w` of each window, per series, is in units of
  the series' odd-month noise scale (1.4826 times the median absolute deviation). The single-window
  detector at window `w` raises an alarm at each local peak of `|z_w|` above a threshold `T_w`,
  where a peak is the largest `|z_w|` within one window length either side, exactly as in
  experiment 43. A stamp where fewer than 75% of either half of the window is present has no
  statistic, and counts as zero in the combined scores.
- **Calibration.** `T_w` is the lowest threshold on a grid of 0.05 whose odd-month unexplained
  alarms (alarms with no logged closure within 24 hours on the series) number at most 80. The same
  search, on a grid of 0.01, sets the thresholds of the two combined detectors.
- **Maximum detector.** The score at a stamp is `max_w |z_w| / T_w`. An alarm is a local peak of
  the score within 24 hours (48 half-hours) either side, above a threshold calibrated to 80
  odd-month unexplained alarms. Single windows keep experiment 43's own peak radius, one window
  length.
- **Sum detector.** The score is `|sum_w z_w / T_w|`, with the sign kept so that windows disagreeing
  in direction cancel. An alarm is a local peak of the score within 24 hours either side, above a
  threshold calibrated the same way.
- **Interval.** The end of a transfer comes from the window that alarms: the window with the
  largest `|z_w| / T_w` at the alarm stamp, among the windows whose step has the sign of the
  combined score. The interval builder is experiment 43's, run with that window. For each alarm the
  order is: choose the peak, choose the window, drop the alarm if its window is 24 hours or longer
  and lies within one window length of a monthly refit, and only then count the alarm.
- **Injected transfers.** The 1,900 transfers of experiment 58, of which 1,460 are in the headline
  band (160 + 1,000 + 300), injected with three shapes: a drifting share of the donor's own power
  (experiment 58's shape), a constant MW equal to the mean MW of that share, and a drifting share of
  the metered receivers' summed power (half the donor's power when none is metered), capped at
  80% of the donor's power. Each shape keeps the same donor, receivers, onset, and end. The
  headline pools the constant-MW and receiver-profile shapes, which no detector here assumes; the
  donor-profile shape is the continuity check with experiment 58.
- **Labelled real transfers.** The labelled real transfers that last over 1 day are scored on the
  real residual for each detector, as counts without a verdict.
- **Even months only.** The odd months fix the noise scales, the thresholds, and the choice of the
  best single window. Every score read for a verdict uses transfers that start in even months.
- **Harness check.** Before any new result is read, the 4-hour and 24-hour single windows at
  experiment 43's thresholds must reproduce its alarm counts and the outcome of each of its 600
  injected transfers. This check passed when the code was written.

## Reported beside the verdict

- **Recall by donor step in noise units, split by metered receiver**, for the best single window and
  both combined detectors, in the bins experiment 58 used.
- **The MW-weighted share found first and the unweighted share second**, with the share of the moved
  MW that each detector misses, stated in the take-home message. The share whose onset alone is
  found is reported too, because a longer window changes the interval's end as well as the alarm.
- **Why the large transfers are missed.** For every transfer with a metered receiver whose donor
  step is 4.0 or more 24-hour noise units (measured on the injected power, in each shape), each
  detector gives one reason, in this order, from all the series of the transfer pooled: the end is
  outside tolerance (an alarm has a matching onset); the alarm was absorbed (the score exceeds the
  threshold within 2 hours of the onset, but a larger peak within the suppression radius took the
  alarm, or a monthly refit dropped it); the onset is outside tolerance (an alarm lies within 24
  hours of the onset but not within 2 hours); or no alarm (the median best score near the onset is
  reported as a share of the threshold).
- **Each shape separately**, and the achieved odd-month and even-month unexplained-alarm counts of
  every detector.

## Sizing (the five triggers)

1. **What gets stored:** private parquet tables only, nothing in Delta.
2. **The production serving path:** not touched.
3. **A degradation rule:** not touched.
4. **More than one defensible design:** yes (maximum or sum, the choice of windows, the
   calibration), so both combinations and all four windows are fixed here.
5. **Code whose callers cannot be named without searching:** none. The code is an adapted copy of
   experiment 58's harness in this folder, with five changes: the four windows, one window per
   alarm in the interval builder, two new injection shapes, the combined detectors, and the miss
   classification.

The size is medium: one Sonnet plan review (done), a Sonnet code review, and a Sonnet prose review.

## Earlier advice followed and departed from

- Follows: score the metered-receiver and no-metered-receiver groups separately; report recall by
  donor step in noise units; match thresholds by recounting unexplained alarms on odd months;
  report even-month unexplained-alarm counts; use several hundred headline-band transfers.
- Departs from: nothing.
