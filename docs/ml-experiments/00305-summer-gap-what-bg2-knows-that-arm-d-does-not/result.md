---
id: '00305'
slug: summer-gap-what-bg2-knows-that-arm-d-does-not
title: >-
  Is the production-style XGBoost forecast better in April to June because of interactions between
  its inputs, or because of one input on its own?
theme: production-reference-and-best-guess
also_relevant_to:
  - differentiable-physics-forecast-engine
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00305-*/ only (refitted boosters and
  per-half-hour predictions). Serving path: not touched. Degradation rule: not touched. Designs: more
  than one is defensible (Shapley values only, permutation only, or refits with interactions
  forbidden); the plan pre-registers the refits as the decision and the other two as descriptions.
  Callers: this folder's scripts, which copy experiment 87's fitting helpers and call reviewed
  `studies.*` modules without changing them. CPU only, no downloads.
hypothesis: >-
  BG2 (experiment 87's production-style XGBoost forecast) beats the convex best-guess model (arm D)
  at the Primary substations D5 and D9 in April to June mainly through interactions of the sunshine
  inputs with temperature or with recent power, and not through the main effect of sunshine.
prediction: >-
  Written before any refit. Each line can fail. (P0, gate) A refit of BG2's monthly regime on the
  control ensemble member scores within 10% of the stored BG2 on the same half-hours (median
  absolute error, April to June, D5 and D9 pooled), and arm D is worse than the refit at D5 and D9.
  (P1, decision) Forbidding every interaction between the solar inputs (shortwave, solar zenith,
  solar azimuth) and the temperature inputs (air temperature, dew point) or the recent-power lags
  removes at least half of the refit's advantage over arm D: the share S = (QS_constrained -
  QS_full) / (QS_D - QS_full) is at least 0.5 on the quarter of April-to-June daylight half-hours at
  D5 and D9 where arm D's quantile score exceeds BG2's by the most. (P2) Forbidding solar-by-lag
  interactions alone gives S of at least 0.25, and forbidding solar-by-temperature alone gives S of
  at least 0.25. (P3) At D10 and D16 (placebos for the solar term) S is under 0.25. (P4) The
  sunshine family has the largest mean absolute Shapley value of the five families on the selected
  half-hours, but the largest Shapley interaction value between families is solar-by-lags or
  solar-by-temperature.
decision_rule: >-
  Intervals resample whole ISO weeks of forecast initialisation time (about 13 in April to June), the
  same blocks for every contrast. P1 is supported if the lower bound of the 95% interval of S is at
  least 0.5, and refuted if the upper bound is below 0.5; otherwise it is inconclusive. A refuted P1
  with S under 0.25 says the main effect of sunshine, not an interaction, carries the advantage,
  and experiments 3 and 5 of `dp_sunny_daylight_fix_design.md` are then re-run over spring rows.
  The threshold is set by the stated brief (half of the advantage), not by porting cost: nothing
  here is ported. P0 failing makes the whole experiment invalid for the interaction question.
benchmark: >-
  Arm D and BG2 on the rows of experiment 266 (row set fold_a_daily_gap_v1, six substations,
  2025-07-02 to 2026-06-30, 09:00 UTC runs, leads 0.5 to 168 hours), restricted to the control
  ensemble member for the refits.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles, in points of effective capacity).
code: ml-experiments/00305-summer-gap-what-bg2-knows-that-arm-d-does-not/
production_status: not-for-production
adopted_in: none
date: 2026-10-23
---

# Planned

Plan only. The refits use experiment 87's N1 recipe (11 quantile levels, 500 rounds, eta 0.05, depth
6, recency half-life 0.5 years), monthly refits on 2026-04-01, 2026-05-01, and 2026-06-01, on CPU
with 2 threads. The three variants differ in the XGBoost `interaction_constraints` only. The stored
BG2 is a blend of five models over 51 members, so the refit is a twin, not a copy, which is why P0
is a gate.
