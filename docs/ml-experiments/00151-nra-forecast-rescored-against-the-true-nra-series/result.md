---
id: "00151"
slug: nra-forecast-rescored-against-the-true-nra-series
title: "NRA forecast rescored against the true NRA series, with the merged RO-NC intervals"
theme: switching-labels-in-the-forecast
also_relevant_to:
  - switching-detectors
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00151-nra-forecast-rescored-against-the-true-nra-series/
  only (per-row score tables). Serving path: not touched. Degradation rule: not touched. Designs:
  more than one defensible way to turn detected intervals into a forecast of the normal running
  arrangement (a regime switch in the filter, as experiment 141 did, or subtraction of the
  detected megawatts from the filter's input), and this plan runs both. Callers: this folder's
  scripts only; they copy the modules of experiment 141 they need and read the stored worlds of
  experiment 141 and the stored intervals of experiments 147 and 148 without changing them.
hypothesis: >-
  The detector-driven correction of experiment 141 was scored against observed power, which
  rewards a filter for following a load transfer. Aim 2 of the switching work is a forecast of the
  normal running arrangement (NRA), the power the substation would have drawn had no transfer
  happened. Scored against that true NRA series, a forecast whose filter subtracts the megawatts
  (MW) of each detected transfer from its input closes a useful share of the gap between the
  filter that ignores transfers and the filter fed the true NRA series, once the alarm has been
  raised. The two-regime filter of experiment 141 closes little, because it still follows the
  transfer.
prediction: >-
  Written before any fit. On forecast origins after the alarm is raised, with targets inside the
  transfer or in the 48 hours after its true end, the subtraction arm driven by detected intervals
  (SUB-DET) closes 30% to 60% of the gap in quantile score between the filter without correction
  (K0) and the filter fed the true NRA series (CLEAN). The reason for that range: experiment 88
  closed 43% (17% to 71%) with the detector's largest-step MW (post hoc) and 51% (9% to 74%) with
  the injected MW for a lag correction, and the intervals of experiments 147 and 148 fix the
  overlap and return-step faults that held experiment 88's pre-registered figure to 16%. The range
  is wide because the detected MW is a constant while the injected MW follows a daily shape. The
  two-regime filters (S-KF fed detected intervals, S-ORACLE fed true intervals) are controls that
  follow the observed level: their shares are at most 15% and may be negative. Subtraction of the
  true MW (SUB-TRUE-LAG, the daily shape; SUB-TRUE-CONST, a constant at the event mean), known
  after the same delays as a detection, closes 55% to 90% and 45% to 80%. A subtraction arm fed
  intervals shifted by whole weeks (SUB-PLACEBO) closes a negative share.
decision_rule: >-
  One decisive contrast, C1: the share of the gap closed by SUB-DET, (K0 - SUB-DET) / (K0 -
  CLEAN), in mean quantile score over the post-alarm row set (below). The numerator and the gap are
  resampled together from 20,000 resamples of whole calendar months of target time (the months, not
  the 44 worlds, are the independent units), and a resample whose gap is zero or negative is
  dropped and counted. The interval is read at 99%. `supported` when the lower end of C1 is at
  least 40%. `not supported` when C1's point value is under 20%, which the maintainer takes to
  mean the intervals are not good enough for aim 2. `partly supported` when the point value is at
  least 20% and the lower end is under 40%. `inconclusive` when the gap's own 99% interval
  includes zero or fewer than 8 calendar months have post-alarm rows. The 40% and 20% lines are
  the maintainer's, not derived from the cost of porting; the prediction (30% to 60%) straddles
  the 40% line, so a verdict of `supported` is not the expected one. All other contrasts (C2
  S-KF, C3 S-ORACLE, C4 SUB-TRUE-LAG, C5 SUB-TRUE-CONST, C6 SUB-PLACEBO, the shares inside the
  transfer and in the 48 hours after its end taken separately, the shares on the detection-lag
  rows, on the 48 hours after the detected end, and on the two periods and each month separately,
  and the same shares for the error of the median forecast) are read at 95% and decide nothing.
  C1 conditions on the transfer being detected, so the share of injected transfers that the
  detector found, and the K0-to-CLEAN gap on all rows after the onset including missed transfers,
  are reported beside it.
benchmark: >-
  K0 (GAM-K's Kalman filter run on the stream with the transfers in it, no correction) and CLEAN
  (the same filter run on the stream without the injected transfers, the true-MW arm). Absolute
  skill beside them as in experiment 141.
row_set_id: >-
  fold_a_daily_gap_v1 rows of experiment 141 (09:00 UTC initialisation, 2025-07-02 to 2026-06-29
  and the even months of 2024-07 to 2025-06), leads 0.5 to 168 h, six decision substations,
  injected worlds of experiment 141; rows outside the hand-filtered switching events; not
  comparable with the leaderboard
metric: >-
  Mean quantile score (twice the mean pinball loss over 11 quantiles) and absolute error of the
  median forecast, against the true NRA series, in points of effective capacity; the share of the
  K0-to-CLEAN gap closed; coverage and pinball loss at q01, q05, q95, and q99.
code: ml-experiments/00151-nra-forecast-rescored-against-the-true-nra-series/
---

**This page is the plan. The result sections are filled in after the run.**

## Question

Experiment 141 scored forecasts against observed power for 48 hours after an alarm. That rewards a
filter that adapts to a load transfer, so it did not test aim 2 of the switching work, a forecast of
the normal running arrangement (NRA). This experiment scores the same forecasts against the power
the substation would have drawn had the transfer not happened, which exists because we inject the
transfers into recorded power. The detected intervals are the merged intervals of
[experiment 147](../00147-causal-live-end-rule-for-ro-nc/result.md) with the 6-hour lookahead end
rule of [experiment 148](../00148-switching-end-rule-lookahead-ro-nc/result.md). The worlds, the
GAM-K filter, and the two-regime filter come from experiment 141
([PR 1453](https://github.com/openclimatefix/nged-substation-forecast/pull/1453)). The measure of
the share of the gap closed comes from
[experiment 88](../00088-sw2-21-nra-forecast-recovers-normal-arrangement-during-and-after-injected-switches/result.md),
which closed 83% of the gap with the true MW and 16% with the detector.

## Arms

All arms are GAM-K's member forecasts plus a correction from a filter run on a stream, with
experiment 141's K0 spread, so every contrast is a difference in location only. The stream is the
real residual stream plus the MW that the injected transfers add.

- **K0:** the filter on the stream, no correction.
- **CLEAN:** the filter on the stream without the injected MW (the true-MW arm, the ceiling; not
  deployable). The gap is K0 minus CLEAN.
- **S-KF:** experiment 141's two-regime filter, with the prior and variance multiplier that 141
  chose on its tuning months, fed the detected intervals.
- **S-ORACLE:** the same filter fed the true transfer intervals (not deployable).
- **SUB-DET:** at each forecast origin, the stream the filter reads has the detected step MW (the
  stored `step_mw`, a constant per interval) subtracted from the detected onset, over every
  detected interval known at that origin, including the false alarms and partner intervals of the
  real stream that a deployed arm would also act on. An interval is known from 96 half-hours after
  its onset (the 24-hour statistic plus a peak test, as in 141). Its end is known at the later of
  the stored `known_from` and 60 half-hours after the end (the median call lag of the 6-hour
  lookahead rule). Until the end is known the interval counts as still open. The filter is re-run
  once per distinct set of known intervals among the scored rows. A knowledge stamp is compared
  with the forecast's cutoff (the origin minus the 3.5-hour telemetry delay), the stamp of the
  last reading the filter may use.
- **SUB-TRUE-LAG:** the same with the true shaped MW and the true interval, known 96 half-hours
  after the true onset and 60 after the true end (isolates the detector's MW and interval errors
  from its delay; not deployable). **SUB-TRUE-CONST** is the same with a constant at the event's
  mean MW, which separates the cost of the detector's constant-MW shape from its other errors.
- **SUB-PLACEBO:** SUB-DET with the detected intervals shifted by a whole number of weeks, as 141's
  placebo (a control: it should close a negative share).

Experiment 88's booster arrangement is not run: it needs the 88 harness and labels, which is not
cheap, and the filter family already answers the question for the forecast that 141 built.

## Intervals

The 6-hour lookahead rule of experiment 148 is not stored for the injected worlds, so each world's
offline-end intervals (`R0_M48`) are used, with the end known at the later of their stored
`known_from` and 60 half-hours after the end. Experiment 148
found that L6 and R0 have the same onset on all 292 intervals of its base draw, the same end on
most (the median difference is 0 stamps; the mean is -1.3 stamps), and that L6 knows the end 60
half-hours after the end (median; 90th percentile 96). The check is printed in the report.

## Rows and windows

Rows: those experiment 141 scored (within an injected transfer or 72 hours after it, or 48 hours
after a causal alarm), for the six substations, in the replicate months and the scored year. Every
row carries three tags, computed from the injected truth and the detected intervals and not from any
arm:

- the window of its target time: inside the transfer, or in the 48 hours after the true end (the
  primary windows), and in the 48 hours after the detected end for detected transfers (descriptive);
- whether its forecast origin is before the onset (every arm forecasts the same, so these rows are
  dropped), between the onset and the stamp the alarm is known (the detection lag), or after the
  alarm is known;
- the period (replicate months or scored year).

Shares for the two target windows are also reported separately, because the pooled share can
hide a gain inside the transfer and a loss after it ends (the failure of experiment 141's
true-interval arm). The post-alarm row set of C1 is the rows whose origin is after the alarm and whose target is inside
the transfer or in the 48 hours after the true end. The detection-lag rows are reported separately.
Transfers the detector missed contribute their rows to the detection-lag set only.

## Descriptive check on real transfers

The labelled real transfers (the 35 over 1 day of experiment 146 and the shorter labelled ones)
whose donor is one of the six substations are scored on the real stream, with K0 and SUB-DET (the
real-stream intervals of experiment 147, with the same knowledge stamps). The NRA series is the
observed power plus the labelled `moved_mw_p50` over the labelled interval, because a donor's power
drops by that amount. The count of transfers scored is stated; no interval or verdict is drawn from
it, and it stays out of the headline.

## Checks and risks

- **Ceiling check:** CLEAN's correction equals K0's, to 1e-9, for every row whose cutoff precedes
  the first injected stamp at the site; the run script raises if not.
- **Identity check:** on the detection-lag rows S-KF and SUB-DET equal K0 except where an
  unrelated real-stream alarm is known; the report prints how many scored rows a false interval
  touches.
- **Leakage:** a forecast at origin `c` uses only intervals whose knowledge stamp is at most `c`,
  and a stream value is read only up to the cutoff `c` minus the 3.5-hour telemetry delay, as in
  141. A unit test-style assertion in the run script checks the knowledge stamps against the
  cutoff for every scored row.
- **Missing the reference:** CLEAN is rebuilt with the injected MW removed exactly, so the share of
  the gap is 100% for CLEAN by construction and a failure of the identity is a bug.
- **Leave one site out** is reported for C1, since five or six sites give few independent units.

## Results

To be written after the run.
