---
id: '00228'
slug: dp-bg-rolling-origin-stack-against-bg2
title: >-
  Does the best-guess stack beat BG2 across 24 monthly origins, six substations, and every season?
theme: production-reference-and-best-guess
also_relevant_to:
- differentiable-physics-forecast-engine
- calibration-and-uncertainty
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00228-dp-bg-rolling-origin-stack-against-bg2/
  only. Serving path: not touched. Degradation rule: not touched. Designs: one defensible (re-score
  stored forecasts; retrain the network head and BG2 only where nothing is stored). Callers: this
  folder's scripts, which copy experiments 220 and 169 and call reviewed `studies.*` modules.
hypothesis: >-
  The best-guess stack (arm D with experiment 169's bands, plus the network head of experiment 201)
  beats BG2 in most monthly origins and at most substations, and its advantage is larger in winter
  than in summer.
prediction: >-
  Written before any score is read. (1) DH (D plus the network head) beats BG2 in at least 18 of 24
  monthly blocks and at 5 or more of the 6 substations, and the month-resampled 95% interval of
  DH minus BG2 excludes zero. (2) D alone beats BG2 in at least 13 of 24 monthly blocks and at 4 or
  more of 6 substations. (3) The advantage is smaller in summer than in winter. If (1) fails with
  wins only in winter, the stack fixes cold, not summer.
decision_rule: >-
  Predictions 1 and 2 as stated, on the quantile score (twice the mean pinball loss over 11
  quantiles, points of effective capacity), leads 0.5 to 168 hours, outside hand-filtered events.
  The gate must pass first: arm A within 0.005 of 2.7519 and BG2 within 0.005 of 2.780 on the five
  decision substations in the scored year. `supported` needs prediction 1; `partly supported` needs
  prediction 2 only or prediction 1's counts without the interval. The threshold is zero with an
  interval excluding zero, because the stack is already built and the question is robustness.
benchmark: >-
  BG2 (experiment 87 b_weekly for the scored year, a new production-style run for 2024-07 to
  2025-06), climatology, and manual heuristic.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score in points of effective capacity; coverage and pinball loss at q01, q05, q95, q99.
code: ml-experiments/00228-dp-bg-rolling-origin-stack-against-bg2/
date: 2026-10-22
---

Planned. Step 1 result: stored ensemble frames start 2024-04; arm D with 169's bands exists for 24
months (2024-07 to 2026-06); the network head and BG2 are stored for 2025-07 to 2026-06 only.
