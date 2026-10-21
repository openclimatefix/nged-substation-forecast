---
id: "00166"
slug: switching-gated-mask-and-soft-mask-while-alarmed
title: >-
  A mask on recent power that trusts only balanced pair alarms, and a soft mask that inflates the
  filter's observation noise
theme: switching-detectors
also_relevant_to:
  - switching-labels-in-the-forecast
  - production-reference-and-best-guess
status: planned
size: >-
  medium. Stored: private score tables under
  data/ml-experiments/00166-switching-gated-mask-and-soft-mask-while-alarmed/ only. Serving path:
  not touched. Degradation rule: not touched. Designs: more than one defensible balance score,
  gate, and inflation rule, so the gate (0.7), the weight rule, and the three inflation factors are
  fixed here before any forecast score is read. Callers: this folder's scripts only; they copy the
  modules of experiments 165 and 156 and read experiment 156's stored alarm tables, experiment
  87's stored best guess 2 predictions, and experiment 155's arm C.
hypothesis: >-
  Experiment 165 found that setting recent power missing while a switching alarm is active lowers
  the error in the 72 hours after a logged transfer ends (13.69 to 6.31 points of effective
  capacity) but raises it by 0.102 points on rows outside logged transfers, because many alarms are
  not transfers. Two ways of using the alarm less bluntly keep the gain and remove the harm. (1) A
  gated mask: mask only when the alarm is a pair alarm of experiment 156 whose opposite steps
  balance at a metered neighbour with a balance score of at least 0.7, and never for a single-series
  refit-offset neighbour-count (RO-NC) alarm. (2) A soft mask: keep the readings in the Kalman
  filter but inflate their observation-noise variance by a factor that rises with the pair alarm's
  balance score, until the end rule fires.
prediction: >-
  Written before any result. Part 1: the gated mask scores within +/-0.03 points of the unmasked
  stack outside logged transfers (95% interval), and keeps at least 70% of the hard mask's gain in
  the 72 hours after a transfer ends (at most 8.5 points, against 13.69 unmasked and 6.31 for the
  hard mask from either alarm). Part 2: at the best of the three factors the soft mask beats the
  hard mask from either alarm outside logged transfers by at least 0.051 points (half of 165's
  +0.102), and keeps at least 80% of the hard mask's change in the 72 hours after a transfer ends
  and at most 120% of its harm while a transfer runs. Each part can fail: the gate can remove the
  harm only by also removing the gain, too few alarms can pass the gate to score, and no factor
  may keep both properties, which would show that the harm is intrinsic to using the alarm.
decision_rule: >-
  Pre-registered before any forecast score is read. Rows, row sets, score, intervals and the BG2
  comparison are those of experiment 165 (rows with a reading in the scored year of
  `fold_a_daily_gap_v1`, 292 runs, at D5, D9, D10, D11, D14, and D16; BEFORE, INSIDE, AFTER by
  logged transfer; OUTSIDE is every other row; winter is November to March; twice the mean pinball
  loss over 11 quantiles in points of effective capacity; outside rows resampled by calendar
  month, event rows by logged transfer, 20,000 draws, 95%). The row sets are defined by experiment
  165's earliest alarm of any kind, so every arm scores the same rows. AFTER and the BG2
  comparison use the rows shared with BG2 (7 transfers); INSIDE and the all-stack AFTER are
  descriptive. Alarm tables: the pair alarm's intervals from experiment 156 (window 8 half-hours,
  k 3.75) each carry the balance score of the alarm that opened them: r is minus the summed step of
  the opposite-signed metered neighbours over the alarm's own step, the score is min(r, 1/r), so it
  runs from 0.4 (the lowest experiment 156 keeps) to 1 (the steps cancel exactly). An interval with
  no stored alarm has score 0. The gate is a score of at least 0.7 (r from 0.7 to 1.43), fixed
  before scoring after counting only how many alarms pass: 113 of 282 real pair alarms. Hard
  masks are as in 165. The soft mask applies to every pair interval with weight w = (score - 0.4) /
  0.6 and inflates the observation-noise variance of its readings, from the moment the alarm is
  known until the end rule fires, by 1 + (F - 1) w, for F in 10, 100, 1000 (arms SOFT_F10,
  SOFT_F100, SOFT_F1000). RO-NC-only alarms do not act in either part. Arms: K0, GAM_ONLY, MASK_NEW,
  MASK_RONC, MASK_UNION (the hard mask of 165), MASK_GATED, the three SOFT_F arms, ORACLE (event
  rows), BG2 (experiment 87), the manual heuristic, and climatology. PART 1 (support needs both):
  G1, MASK_GATED minus K0 on OUTSIDE has a 95% interval inside -0.03 to +0.03; G2, on AFTER the
  point estimate of K0 minus MASK_GATED is at least 70% of K0 minus MASK_UNION (165: 13.69 -
  6.31 = 7.38, so at least 5.17). Part 1 is `inconclusive` when fewer than 5 of the 13
  transfer-substation pairs of 165's AFTER set have a gated alarm known before the transfer ends.
  Otherwise it is `not supported` when G1 or G2 fails, and `supported` when both hold. PART 2
  (support needs one factor to pass all three): S1, SOFT_F minus MASK_UNION on OUTSIDE has a point
  estimate of -0.051 or lower; S2, on AFTER, K0 minus SOFT_F is at least 80% of K0 minus
  MASK_UNION; S3, on INSIDE, SOFT_F minus K0 is at most 120% of MASK_UNION minus K0. The best
  factor is the one with the lowest SOFT_F minus K0 on OUTSIDE among factors passing S2 and S3, and
  its selection from three factors is noted in the write-up. Part 2 is `supported` when a factor
  passes S1, S2, and S3, and `not supported` otherwise. The experiment's verdict is `supported`
  when both parts are supported, `partly supported` when one is, `not supported` when neither is,
  and `inconclusive` when part 1 is inconclusive and part 2 is not supported. The threshold on G1
  is +/-0.03 because the mask adds alarm tables to the serving path, which is worth doing only if
  ordinary days do not get worse. The "inside-transfer gain" of the brief (6.32 against 13.69
  unmasked) is the AFTER set of 165; the INSIDE set, where observed power contains the moved
  load, is where 165 found the hard mask 5.32 points worse. Post hoc and labelled: gates of 0.6 and
  0.8, the same checks in winter and outside winter, and leave-one-substation-out.
benchmark: >-
  The unmasked stack (K0), the hard mask from either alarm (MASK_UNION), and BG2 (experiment 87,
  `p_s6_bg2`), on the same rows.
row_set_id: >-
  fold_a_daily_gap_v1 scored year, every row with a reading at six substations; not comparable with
  the leaderboard
metric: >-
  Mean quantile score (twice the mean pinball loss over 11 quantiles), in points of effective
  capacity; tail coverage and pinball loss at 0.01, 0.05, 0.95, and 0.99.
code: ml-experiments/00166-switching-gated-mask-and-soft-mask-while-alarmed/
---

Planned. The write-up replaces this text when the experiment finishes.
