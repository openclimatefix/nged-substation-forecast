---
id: "00227"
slug: lsc-fewer-false-alarms-and-softer-use-part-2
title: >-
  Can an LSC alarm source with fewer false alarms per matched load shift, and a use of the alarm
  that keeps recent power, recover the gain without ordinary-day harm?
theme: lsc-detectors
also_relevant_to:
  - switching-labels-in-the-forecast
status: planned
size: >-
  medium. Stored: private score tables under
  data/ml-experiments/00227-lsc-fewer-false-alarms-and-softer-use-part-2/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: more than one is defensible for how an alarm is
  used (mask, soft down-weighting, pair-sum correction, hedge), so the arms are fixed below before
  any score is read. Callers: this folder's scripts only; they are copies of the modules of
  experiments 195, 203, and 208, and read the stored worlds of experiment 208 and the alarm tables
  of experiments 147, 156, 195, and 226.
hypothesis: >-
  Experiment 208 found that no real alarm source stays inside the ordinary-day harm limit of 0.05
  points, because each raises 104 to 242 alarms for 15 to 20 matched logged load shifts. (a) A
  source at a higher threshold, or a union of the pair alarm and the learned score that keeps the
  learned score only where the pair alarm also fires or its score is high, matches enough load
  shifts with few alarms per match. (b) A use of the alarm that keeps some recent power (soft
  down-weighting, a pair-sum correction, or a hedge between masked and unmasked forecasts) costs
  less per false alarm than a mask and keeps most of the gain after a load shift.
prediction: >-
  Written before any score is read. (1) At least one source and threshold in part (a) matches at
  least 12 of the 35 logged load shifts lasting over 1 day with at most 5 alarms per matched load
  shift, recovers at least 50% of the perfect-knowledge gain in the 48 hours after a load shift, and
  has an ordinary-day harm interval inside +/-0.05 points. It fails if every threshold that meets
  the harm limit matches under 12 load shifts, or if every source that recovers 50% breaks the
  harm limit; the page says so first, and gives the best trade-off point. (2) In part (b), the
  pair-sum correction or the soft down-weighting recovers at least 80% of the mask's gain in the 48
  hours after a load shift while cutting the harm per false alarm to at most half of the mask's.
  It fails if no arm does both; the hedge is reported as the cheap reference. If no arm beats the
  mask, the page says the harm is intrinsic to using any alarm and the fix is fewer alarms.
decision_rule: >-
  Pre-registered. Part (a), sources on experiment 208's real-record rows: the learned score of
  experiment 195 (re-ranker, alarms known 4 days after the step), the pair alarm gated by balance
  (experiment 166), the combination from experiment 226 that scored best on ordinary-day harm among
  those that match at least 20 of 35 (a U-rule fixed before scoring), and RO-NC for reference. Each
  source's threshold is swept over 5 to 8 values chosen on odd months and scored on even months
  (the learned score and RO-NC by the number of unexplained odd-month alarms, 10 to 200; the pair
  alarm by its balance gate, 0.4 to 0.9; each reported with its score threshold, in noise units for
  RO-NC and the pair alarm, experiment 217). Per source and threshold: alarms per matched logged
  load shift (alarms over the whole real record divided by the logged load shifts matched, using
  experiment 195's matching rule), the matched count of 35, the share recovered in the AFTER phase
  of the real record's logged load shifts (the learned mask's gain over no mask divided by the
  logged-oracle mask's gain, rows of experiment 163's after-load-shift set), and the ordinary-day
  harm with a 95% interval resampled by whole calendar month. Part (b), arms on experiment 208's
  harness with the same alarms and the same K0 baseline: MASK; SOFT (observation noise variance
  multiplied by 2, 4, or 16 inside the alarm); PAIRSUM (while a pair-balanced alarm is open, each
  Primary's correction uses the pair total, so the transferred share is subtracted using the
  expected-power ratio; the fallback is MASK where the partner is unmetered, and the coverage is
  reported); HALF (average of the K0 and MASK corrections); SUB (the detected MW subtracted, for
  sources that carry MW); NONE (K0). Shares recovered by phase (BEFORE_ALARM, DURING, AFTER,
  ORDINARY) on injected worlds for RO-NC and on the real record for the part (a) sources, and the
  harm per false alarm (loss difference on ordinary rows divided by the alarms, with its
  interval). Bar (from experiment 208): the best source with the best use recovers at least 50% of
  the AFTER gain, point estimate at or above 50%, and ordinary-day harm with a 95% interval inside
  +/-0.05. Verdict: `supported` if the combination meets the bar on the real record, `partly
  supported` if only part (a) or part (b) works, and `not supported` otherwise. The 50% bar and
  the 0.05 limit are set by the cost of porting an alarm source and its use, which adds a model and
  a pipeline step to the serving path. Positive controls: K0 and the pair-alarm mask reproduce
  experiment 203's 13.686 and 6.321 on the shared rows and the gated +0.030 on ordinary rows, and
  K0 and CLEAN agree before the first injected stamp on injected worlds. Reported beside (not
  bars): the median forecast minus the mean forecast per site, the mean-forecast error beside each
  contrast, leave-one-substation-out, and tail coverage and pinball loss at q01, q05, q95, q99.
benchmark: >-
  Experiment 208: perfect knowledge gains 1.50 points (1.08 to 1.96) in the 48 hours after a long,
  large simulated load shift, RO-NC recovers 53% (35% to 69%), and five real alarm sources raise the
  ordinary-day score by 0.03 to 0.14 points.
row_set_id: >-
  Experiment 208's injected-world rows and experiment 203's real-record rows (after-load-shift and
  ordinary); not comparable with the leaderboard
metric: >-
  Mean quantile score (twice the mean pinball loss over 11 quantiles), in points of effective
  capacity; gain is the K0 score minus the arm's score.
code: ml-experiments/00227-lsc-fewer-false-alarms-and-softer-use-part-2/
---

Planned. The write-up replaces this page when the experiment finishes.
