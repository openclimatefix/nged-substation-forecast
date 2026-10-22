---
id: "00171"
slug: switching-gated-mask-no-harm-pooled-months
title: >-
  Does the gated mask on recent power do no harm on ordinary days, once 27 months of rows are pooled?
theme: switching-detectors
also_relevant_to:
  - switching-labels-in-the-forecast
status: planned
size: >-
  simple. Stored: private score tables under
  data/ml-experiments/00171-switching-gated-mask-no-harm-pooled-months/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: one defensible design, because the arms, the
  gate, the rows and the interval are those of experiment 166 with the scored year replaced by every
  cached run. Callers: this folder's scripts only; they are copies of experiment 166's modules and
  read experiment 166's stored balance scores and experiment 156's stored alarm tables.
hypothesis: >-
  Experiment 166 found that ignoring recent power (a "mask") only while a pair alarm with a balance
  score of at least 0.7 is active (the "gated mask") scores 0.009 points of effective capacity
  worse than no mask on the real rows outside logged transfers, with a 95% interval of -0.068 to
  +0.081 from one year of rows. That interval was too wide to show no harm. Pooling every cached
  forecast run, 27 months instead of 12, shows whether the gated mask does no harm on ordinary days.
prediction: >-
  Written before any result. On the pooled real rows outside logged transfers (six Primary
  substations, 2024-04 to 2026-06), the gated mask minus the unmasked stack has a 95% interval
  inside +/-0.03 points of effective capacity, and the same contrast on November to March rows
  has an interval inside +/-0.05. The prediction fails if the pooled point estimate is above
  +0.03, or if the pooled interval is still wider than +/-0.03; in that case no-harm cannot be
  shown from this data, and the page says so. Pooling 27 months instead of 12 shrinks a
  month-resampled interval by about 1.5 times (the square root of 27 over 12), so a half-width
  near 0.05 is the expected size and the interval inside +/-0.03 is not expected.
decision_rule: >-
  Pre-registered before any forecast score is read. Arms: K0 (the frozen generalised additive
  model with the Kalman filter and no mask), MASK_GATED (readings set missing from the moment a
  pair alarm with balance score 0.7 or more is known until the end rule of experiments 147 and 148
  fires), and, for reference only, MASK_NEW (every pair alarm) and MASK_UNION (pair or
  refit-offset neighbour-count alarm). Rows: every row of every cached run of the six Primary
  substations of experiment 166 (D5, D9, D10, D11, D14, D16; forecast origins 2024-04-10 to
  2026-06-29) with a reading, whose valid time lies outside every logged transfer from its onset
  to 72 hours after its end on that substation. Every series has more than 12 months of residual
  history before its first run, because the power panel starts in 2022-01. Scored with the same
  quantile score, the same frozen filter, spread and alarm tables as experiment 166. Interval:
  rows resampled by whole calendar month (year and month together), 20,000 draws, 95%. Support
  needs both G1 and G2. G1: pooled, MASK_GATED minus K0 has an interval inside -0.03 to +0.03. G2:
  the same on November to March rows, inside -0.05 to +0.05. The verdict is `supported` when both
  hold, `partly supported` when only G1 holds, `refuted` when the pooled point estimate is above
  +0.03, and `inconclusive` otherwise (an interval wider than the limit with the point estimate
  not above +0.03). The threshold is +/-0.03 because the mask adds alarm tables to the serving
  path, which is worth doing only if ordinary days do not get worse (experiment 166). The gain
  side stays as in experiment 166 (the 72 hours after a logged transfer ends, 7 transfers shared
  with best guess 2) and is not pooled with the no-harm rows. Reported beside the contrast: the
  number of gated alarms and masked half-hours (grid half-hours between an alarm's onset being
  known and its end being known) per substation and per calendar year, and the share of masked
  half-hours outside every logged transfer. Post hoc and labelled: the scored year alone and the
  earlier 15 months alone, to see whether the two agree; MASK_NEW and MASK_UNION minus K0;
  leaving one substation out; and the rows with a gated alarm active.
benchmark: >-
  The unmasked stack (K0) on the same rows; experiment 166's interval for the scored year alone
  (+0.009, -0.068 to +0.081).
row_set_id: >-
  every cached run (2024-04-10 to 2026-06-29) of six Primary substations, rows with a reading
  outside logged transfers; not comparable with the leaderboard
metric: >-
  Mean quantile score (twice the mean pinball loss over 11 quantiles), in points of effective
  capacity.
code: ml-experiments/00171-switching-gated-mask-no-harm-pooled-months/
production_status: not-for-production
adopted_in: none
---

Planned. The write-up replaces this text when the experiment has run.
