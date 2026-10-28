---
id: '00363'
slug: d8-segment-level-term-in-the-gam-stack
title: >-
  Can a penalised level term for each stretch between detected steps let the GAM stack explain D8's
  steps instead of masking them?
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - calibration-and-uncertainty
status: planned
kind: other
size: >-
  medium. Stored: private files under data/ml-experiments/00363-d8-segment-level-term-in-the-gam-stack/
  only. Serving path: not touched. Degradation rule: not touched (the segment detector is an R&D
  device here). Designs: more than one is defensible (random-effect intercepts, a step function of
  time, a latent state), so the plan fixes penalised intercepts at three strengths. Callers: the
  scripts import experiments 252, 354 and 361 and add one design column block. Asset graph: not
  touched.
hypothesis: >-
  D8 (Marsh Lane) is a Primary substation whose demand carries on/off steps of about 4 MW and
  multi-day drops of about 2.7 MW. Experiment 361 removed or masked the detected steps in the
  training target and the score did not move. Giving the stack one penalised level term for each
  stretch between detected steps, fitted jointly with the weather and calendar terms, lets those
  terms be estimated within stretches, so D8's quantile score falls against experiment 354's best
  arm.
prediction: >-
  Written before any score is read. (P1) Arm S at the middle penalty lowers D8's quantile score by
  at least 0.2 points against arm C, which reproduces 8.382. (P2) S beats the masking arm M (8.384)
  at the middle penalty. (P3) At D2 and D4 the score changes by less than 3%. (P4) S+ lowers the
  share of readings above q95 at D8 against S without raising the score.
decision_rule: >-
  Stop rule: if S at the middle penalty does not beat M, report and stop (no S+ and no weather-term
  comparison beyond the clean-Primary check). Otherwise two planned contrasts on the 96,986 rows
  of experiment 354, paired, month-resampled (20,000 resamples), 99% intervals (a Bonferroni
  correction for two). C1: S at the middle penalty minus C. C2: S at the middle penalty minus M.
  Threshold for C1: interval wholly below zero and the point estimate at most -0.2 points, because
  the term adds a detector and one design block per fit cut. `supported` when C1 clears the
  threshold, C2 is wholly below zero, and D2 and D4 move by under 3%. `partly supported` when C1
  clears the threshold and a placebo fails. `small gain` when an interval lies wholly below zero
  with a point estimate above -0.2. `not supported` otherwise. Control gate: arm C reproduces
  8.382 before any other arm is read.
benchmark: >-
  Experiment 354's fit (b) with the own-residual band by hour block and season (8.382 monthly);
  experiment 361's masking arm (8.384); BG2, climatology, and manual_heuristic as stored by
  experiment 87
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles) in points of D8's effective
  capacity (1 point is about 0.1 MW), with MAE, correlation, and bias, by lead band and by days
  with and without detected steps; q05 to q95 coverage; share below q01 and above q99
code: ml-experiments/00363-d8-segment-level-term-in-the-gam-stack/
---

**Planned, not yet run.** The five plan items below are fixed before any score is read.

1. **What is added.** Arms C, M, S, and S+ below, all on experiment 354's fit (b): monthly refits at
   the stored cuts, history from 2023-09, one level step per hour band at 2025-04-01. Arm S appends
   one column per stretch between detected steps (the boundaries are every start and end of the
   segments that experiment 361's step and multi-day-drop rules find on data up to the cut). Each
   column is 1 on the rows of its stretch. The columns carry a ridge penalty of lambda times the
   weather terms' ridge scale, with lambda at 0.1 (weak), 1 (middle), and 10 (strong). The global
   intercept stays unpenalised.
2. **Forecast time.** Rows after a cut take the level of the latest stretch before the cut, so no
   future data enter. The level is therefore up to one refit period old. Experiment 252's filter
   still assimilates recent power on top of the fitted stack.
3. **Arms.** C: experiment 354's fit (b), refitted here to reproduce 8.382. M: experiment 361's
   masking arm, read from its stored forecasts. S: the segment term at three penalties. S+: S at the
   middle penalty with the upper quantiles (q50 and above) widened by a two-component mixture. The
   mixture puts weight p on the forecast shifted up by the median detected depth, where p is the
   share of training rows inside detected falls, and applies only to rows whose latest stretch is a
   detected fall.
4. **Placebos.** C and S at the middle penalty at D2 and D4, which have no engine.
5. **Clean-Primary check.** At D2, compare the fitted weather terms (heating and solar responses)
   with and without the segment term, because the detector flags 171 rises against 193 falls and
   some are ordinary demand swings.

Headline caveat for every number: interim, one substation, one scored year; the detector recovers
55% of 2 MW and 91% of 4 MW injections and flags ordinary demand swings; the generator behind D8
is unproven.
