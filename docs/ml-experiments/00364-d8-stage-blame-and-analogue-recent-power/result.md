---
id: '00364'
slug: d8-stage-blame-and-analogue-recent-power
title: >-
  Which stage of the GAM stack loses skill at D8 against BG2's members, and does the analogue member
  win because it sees recent power?
theme: production-reference-and-best-guess
also_relevant_to:
  - calibration-and-uncertainty
status: planned
kind: other
size: >-
  medium. Stored: private files under data/ml-experiments/00364-d8-stage-blame-and-analogue-recent-power/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one is
  defensible for "recent power" in the analogue member, so the plan fixes two arms (no origin lags;
  no origin lags and no analogue within 7 days of the origin). Callers: the scripts import
  experiments 252, 354, 355, 361, and 87 and call their fit and scoring code.
hypothesis: >-
  At D8 (Marsh Lane), one stage of the GAM stack (the base GAM, the Kalman filter's level or
  transient, the winter night offset, recipe R, or the own-residual band) adds error that BG2's
  members do not carry, and the analogue member's lead over the GAM stack comes from its use of
  recent power.
prediction: >-
  Written before any score is read. (P1) One stage lowers the Pearson correlation of the median
  forecast with the actual demand at D8, and only on days with detected steps. (P2) The analogue
  member refitted with no origin lags and no analogue within 7 days of the origin scores above 8.0
  at D8 (it scores 7.472 with them). (P3) At D2 and D4, no stage moves the score by more than 0.1
  points.
decision_rule: >-
  Descriptive, no planned contrast that sets a verdict by threshold. A stage is blamed when its
  score rises by more than 0.1 points over the stage before, or its correlation falls (month-resampled
  95% interval wholly below zero). Stop rule: if no stage lowers correlation or raises the score by
  more than 0.1 points, report that the loss is spread across the stack, and stop. The analogue
  member's lead is attributed to recent power when the refit without it scores above 8.0 or is
  worse than the analogue member by at least 0.3 points (interval wholly above zero). Gate: the
  final stage reproduces 8.338 (weekly) or 8.382 (monthly), BG2 reproduces 7.593, on the same
  96,986 rows as experiment 354.
benchmark: >-
  BG2 (7.593), the analogue member alone (7.472), and the GAM stack (8.338 weekly, 8.382 monthly)
  from experiments 360, 355, and 354
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles) in points of effective capacity
  (1 point is about 0.1 MW at D8); MAE of the median in MW; bias in MW; Pearson correlation of the
  median with the actual demand, per lead band
code: ml-experiments/00364-d8-stage-blame-and-analogue-recent-power/
date: 2026-10-28
---

**Plan.** The GAM stack's stages are read from stored frames (no GAM refit): the plain GAM
(`mean - level - anomaly`), plus the Kalman level, plus the Kalman transient, plus the winter night
offset, plus recipe R, plus the own-residual band. Stages before the band take the stored band of
their arm. BG2's weekly XGBoost member, monthly XGBoost member, and analogue member come from
experiment 87's stored predictions. The analogue member is refitted twice at D8 only (monthly
refits, weekly analogue columns as in experiment 87): once with the three origin lags dropped, and
once with the origin lags dropped and every analogue half-hour within 7 days of the origin masked.
Every arm is scored by quantile score, MAE, bias, and correlation, by lead band (0 to 24, 24 to 72,
72 to 168 h), by days with and without detected steps (experiment 361's detector), and by month,
with month-resampled 95% intervals. D2 and D4 repeat the stage split as placebo Primaries where the
stages are stored. Figures follow the brief.

**Caveat.** Interim: one substation, one scored year; the generator behind D8 is unproven.
