---
id: "00226"
slug: lsc-why-learned-score-mask-lost-part-1
title: >-
  Why did the learned LSC score find more load shifts but give a worse masked forecast, and does a
  combined mask fix it?
theme: switching-detectors
also_relevant_to:
  - switching-labels-in-the-forecast
status: planned
size: >-
  medium. Stored: private score tables under
  data/ml-experiments/00226-lsc-why-learned-score-mask-lost-part-1/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: more than one defensible (how the pair alarm and
  the learned score combine), so three combinations U1 to U3 are fixed below before any score is
  read. Callers: this folder's scripts only; they are copies of experiment 203's modules and read
  its stored tables and experiment 195's alarm tables.
hypothesis: >-
  Experiment 203's learned-score mask (7.31 points of effective capacity on the 72 hours after the
  7 logged load shifts, against 6.32 for the pair-alarm mask) loses because the learned alarm is
  known 2 to 4 days after the step, so its mask starts late on long load shifts. If that is the
  cause, a mask that takes the pair alarm's onset where the pair alarm fires, and the learned
  score's coverage elsewhere, keeps the learned score's extra matches without the loss.
prediction: >-
  Written before any forecast score is read. The best of U1 to U3 scores at most 6.32 on the 7
  transfers (matching the pair-alarm mask), has a pooled ordinary-day point estimate (mask minus no
  mask) at or below +0.02 on experiment 171's rows, and matches at least 20 of the 35 logged load
  shifts lasting over 1 day. The prediction fails if every combination scores above 6.32 on the 7.
  The page then says that the extra load shifts found are the wrong ones for the forecast (small,
  late, or low-value), and gives their size and timing.
decision_rule: >-
  Pre-registered. Same harness, rows, controls, and positive controls as experiment 203 (daily 09:00
  UTC cached runs, not 6-hourly, because the cached ensemble forecasts exist only for those runs;
  causal only). Combined masks, all built from experiment 195's R re-ranker alarms (known 192
  half-hours after the step, as in 203) and the pair alarm (every pair alarm, experiment 163's
  PAIR_NEW): U1, pair-alarm intervals plus learned-score intervals, merging intervals on one series
  that start within 24 hours, each interval keeping the earlier known time and the later end; U2,
  the learned-score intervals kept only where the candidate's opposite-step ratio folded to at most
  1 (min of ratio and 1/ratio, 0 where no opposite step) is at least 0.7, experiment 166's gate;
  U3, the learned-score intervals with the end known only by experiment 148's rule (the estimated
  end plus 60 half-hours, no experiment 147 stamp and no 96 half-hour shortcut) and the step stamp
  as the onset. Verdict by the brief's prediction on the best of U1 to U3 (chosen on the 7
  transfers, so the selection is reported as such): `supported` when the best combination scores at
  most 6.32 on the 7, its ordinary-day point estimate is at most +0.02, and it matches at least 20
  of 35; `partly supported` when two of the three hold; `not supported` when at most one holds.
  Intervals: transfers resampled whole for the gain (20,000 draws), rows resampled by whole calendar
  month for the harm. Descriptive, not bars: the per-transfer table, the gap decomposition, the harm
  split by alarm source (learned alarms that a pair alarm also raises within 48 half-hours, and
  learned-only alarms), by Primary substation, and by season; the same ordinary-day contrast with
  learned-only alarms removed that are balanced-opposite candidates (an opposite step of 0.4 to 2.5
  times the own step, experiment 195's definition, probable unlogged load shifts). The smallest gain
  worth porting is zero for a mask change that adds no pipeline step beyond the learned score
  already assumed; the harm limit is +/-0.05 points, from the brief.
benchmark: >-
  The pair-alarm mask (6.32 points on the 72 hours after the 7 transfers) and the unmasked forecast
  (13.69), experiment 163; the pair-alarm gated mask minus no mask on ordinary days (+0.030),
  experiment 171; the re-ranker mask (7.31, +0.031), experiment 203.
row_set_id: >-
  experiment 163's AFTER rows (7 and 11 transfers) and every cached run (2024-04-10 to 2026-06-29)
  of six Primary substations; not comparable with the leaderboard
metric: >-
  Mean quantile score (twice the mean pinball loss over 11 quantiles), in points of effective
  capacity.
code: ml-experiments/00226-lsc-why-learned-score-mask-lost-part-1/
---

Planned. The write-up replaces this page when the experiment finishes.
