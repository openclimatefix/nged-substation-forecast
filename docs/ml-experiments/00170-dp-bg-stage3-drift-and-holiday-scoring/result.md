---
id: '00170'
slug: dp-bg-stage3-drift-and-holiday-scoring
part: BG-S3
title: >-
  Does a drift term still lower the score of the stage 2 assembly (night offset and recipe R), and
  how do Christmas, Easter, and bank holidays score?
theme: differentiable-physics-forecast-engine
also_relevant_to:
- production-reference-and-best-guess
- differentiable-physics-long-history-and-drift
- calibration-and-uncertainty
status: planned
size: >-
  medium. Stored: private files under
  data/ml-experiments/00170-dp-bg-stage3-drift-and-holiday-scoring/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: more than one is defensible (whether the drift
  and the corrections are fitted in one pass or in sequence, the drift's training window, how an
  event day is compared with ordinary days), and this plan pre-registers one of each. Callers: this
  folder's scripts, which copy experiment 136's ensemble forecast and filter code, experiment 167's
  drift design code, and experiment 155's corrections and report code; they call the reviewed
  `studies.*` modules without changing them. No new modelling.
hypothesis: >-
  The convex demand model with a Kalman filter on recent errors, ensemble-mean weather, the heating
  split by day type, one shared filter noise setting, and the pinned solar capacity (experiment
  136's arm A2) improves further when it gets a drift term (experiment 167's D1) on top of the two
  post-forecast corrections of experiment 155's arm D (a winter night offset and recipe R's 14-day
  mean error profile). The mechanism is that the drift term removes a slow demand-level error that
  the monthly refit on 24 months of history leaves in, and neither the offset (winter nights only)
  nor recipe R (a 14-day profile held to 0.75 of its size) removes that error. Training on every
  year of history adds a further gain, because the drift term and the pinned capacity let the old
  years teach the heating and calendar terms.
prediction: >-
  Written before any score is read. Arm D plus drift at 24 months scores at least 0.20 points below
  arm D's 2.756, that is 2.556 or lower (C1, at most -0.20); training the same arm on all years
  scores a further 0.05 or more below the 24-month arm (C2, at most -0.05). The risk is that drift,
  the night offset, and recipe R remove the same slow level error, so that experiment 167's gain of
  0.40 points (drift on the 24-month GAM without ensemble weather, offset, or R) mostly vanishes;
  a combined gain under 0.20 is the fail, and the useful answer is then which part to drop. In the
  drop-one table, dropping the drift term costs more than dropping the offset or recipe R (the
  offset added +0.002 on top of R in experiment 155). On Christmas (24 December to 2 January), the
  score is at least 1.5 times the score on ordinary weekdays at the same half-hours of the day, in
  each Christmas and pooled, and recipe R raises the score on those days (the arm with R minus the
  arm without R is above zero), because its 14-day profile averages ordinary days. If only the
  first half holds, recipe R can stay on over Christmas only if the second half fails.
decision_rule: >-
  The headline contrast is C1, arm D plus drift at 24 months (`s24_full`) minus arm D as stored by
  experiment 155 (`d155`), over leads 0.5 to 168 hours at the five decision substations D5, D10,
  D11, D14, and D16 in the scored year (D9 beside them in the tables, in no contrast), on the rows
  every arm shares, outside the hand-filtered switching events, in points of effective capacity.
  Each planned contrast is read at the 99% level of the month-resampled paired bootstrap (the 12
  calendar months of the scored year, 20,000 resamples). `supported` needs C1 at most -0.20 with an
  interval wholly below zero, and C2 at most -0.05. `partly supported` is C1's interval wholly
  below zero with either C1 above -0.20 or C2 above -0.05. `refuted` is C1's interval wholly above
  zero. `inconclusive` is everything else. The 0.20 threshold is set by cost: the drift term adds
  48 fitted knots per substation to every refit and a held last level to the serving path, and a
  gain under 0.20 points (7% of the score) does not pay for that. The 0.05 threshold for all years
  is set by the extra storage, the three times larger refits, and the old capacity table. The drop-one
  table (C3 drop drift is C1; C4 drop the offset, `s24_full` minus `s24_r`; C5 drop recipe R,
  `s24_full` minus `s24_off`) carries no verdict. The winning arm is the arm with the lowest
  all-lead score whose contrast against arm D has an interval wholly below zero; arms within 0.005
  points tie, and the earlier in the order `s24_full`, `sall_full`, `s24_r`, `sall_r` wins. The
  verdict is capped at `inconclusive` if the reproduction control fails: the arm `a2_repro` (no
  drift, 24 months, this folder's fitting code) must reproduce experiment 136's stored A2 mean
  forecast to within 0.01 points at the median row, and the stage 2 code run on `a2_repro` must
  score within 0.005 points of arm D's 2.756. Part 2 is descriptive: the Christmas ratio
  (quantile score on 24 December to 2 January over the quantile score on ordinary weekdays at the
  same half-hours), the Easter and bank-holiday ratios against ordinary Sundays, and the with-minus-
  without recipe R and drift differences on those days, each by Christmas (replicate year 2024-25,
  scored year 2025-26) with the number of days and runs. Event days enter no verdict, because there
  are two Christmases. Experiment 143's holiday-aware filter update was not supported and is not
  an arm.
benchmark: >-
  Arm D as stored by experiment 155 (2.756 points over leads 0.5 to 168 hours at the five decision
  substations in the scored year), arm A (2.791), arm C (2.754), BG2 as stored (experiment 87,
  2.780), climatology and manual_heuristic as stored by experiment 111, for absolute skill
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over the 11 quantiles of experiment 87) in points of
  effective capacity, on rows outside the hand-filtered switching events, over leads 0.5 to 168
  hours, then under 24 hours, 24 to 72 hours, 72 to 168 hours, day D+1, winter (November to March),
  cold nights, Christmas (24 December to 2 January), Easter, and bank holidays; coverage and pinball
  loss at the 0.01, 0.05, 0.95, and 0.99 quantiles
code: ml-experiments/00170-dp-bg-stage3-drift-and-holiday-scoring/
---

# Experiment 170: stage 3 assembly, a drift term on arm D, and the holiday scores

This page is the plan. The finished write-up replaces it.

## Arms

- **Stage 1 forecasts** (T-all, ensemble mean, pinned solar capacity, POOL filter noise, monthly
  refits as in [experiment 136](../00136-dp-bg-stage1-assembly-on-the-ensemble-mean/result.md)):
  `a2_repro` (no drift, last 24 months; a control), `m24_d1` (drift D1, last 24 months), and
  `all_d1` (drift D1, all years).
- **Stage 2 corrections** ([experiment
  155](../00155-dp-bg-stage2-assembly-night-offset-and-error-correction/result.md)) on each
  drift forecast: `*_drift` (none), `*_off` (winter night offset), `*_r` (recipe R), and `*_full`
  (offset, then R on the offset's own errors, as arm D). `s24` is on `m24_d1` and `sall` on
  `all_d1`.
- **References:** arm A, C, and D as stored by experiment 155, BG2, climatology, and the manual
  heuristic.
