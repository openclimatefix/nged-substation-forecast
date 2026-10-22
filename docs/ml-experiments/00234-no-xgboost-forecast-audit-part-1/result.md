---
id: '00234'
slug: no-xgboost-forecast-audit-part-1
title: >-
  Can demand, metered solar farms, and metered wind farms each be forecast without XGBoost, and
  where does the XGBoost-free forecast lose?
theme: production-reference-and-best-guess
also_relevant_to:
- generator-forecasting
- metered-solar-farms
- differentiable-physics-forecast-engine
status: planned
size: >-
  simple. Stored: private tables under data/ml-experiments/00234-no-xgboost-forecast-audit-part-1/
  only. Serving path: not touched (the code audit only reads it). Degradation rule: not touched.
  Designs: one defensible design, because every arm is a stored forecast and nothing is refitted.
  Callers: this folder's scripts, which read stored predictions and call reviewed `studies.*`
  modules without changing them. Asset graph: not touched.
hypothesis: >-
  For Primary substation demand, a metered solar farm, and a metered wind farm, the best forecast
  that uses no XGBoost anywhere scores within 2% of the quantile score of the best forecast that
  uses XGBoost, at 12 to 36 hour leads.
prediction: >-
  Written before any score is recomputed. Demand: yes, because experiment 228 found the
  XGBoost-free stack 5% better than BG2 over all leads. Solar: no, because experiment 224 found
  XGBoost 2.8% better. Wind: no at 12 to 36 hours (gap about 3% to 4%), because the physics core
  trails the XGBoost best guess by 0.27 points at days 0 to 2 in experiment 230.
decision_rule: >-
  Gap is the XGBoost-free arm's quantile score minus the XGBoost arm's, as a percentage of the
  XGBoost arm's, at 12 to 36 hour leads (all leads reported beside it). The quantile score is
  twice the mean pinball loss over the 11 levels. Within tolerance when the point estimate is at
  most 2%; unresolved when the point estimate is at most 2% but the 95% month-resampled interval's
  upper end is above 2%; over tolerance otherwise. The 2% is the maintainer's. A type that is over
  tolerance gets a Part 2 experiment (a separate PR).
benchmark: >-
  Demand: BG2 against arm D plus the correction network and the two-number solar summary (experiment
  228 rows). Solar: XGBoost (arm 5) against the plant fed the mean and spread of 51 members (arm 3)
  (experiment 224 rows). Wind: the XGBoost best guess refit with 11 levels (A0r) against the physics
  core (B1) (experiment 230 rows, read only).
row_set_id: >-
  The row set of each source experiment, unchanged: demand 228's six substations, solar 224's six
  farms in the scored year, wind 230's three farms on experiment 33's Tuesday runs
metric: >-
  Quantile score; MAE of the median; coverage and pinball loss at 0.01, 0.05, 0.95, 0.99;
  curtailment cost and threshold-weighted CRPS where a source experiment scored them
code: ml-experiments/00234-no-xgboost-forecast-audit-part-1/
production_status: not-for-production
adopted_in: none
---

**Planned; no result yet.** This page is the plan, committed before any score is recomputed.

## Question

Production runs XGBoost. We ask whether a forecast with no XGBoost anywhere could replace it, for
each of the three series types NGED needs: Primary substation demand, metered solar farms, and
metered wind farms. Part 1 recomputes the comparison from stored predictions and audits the code for
every remaining XGBoost dependency. Part 2, a separate experiment and PR, adds small XGBoost parts
to the XGBoost-free forecast wherever Part 1 finds a gap over 2%.

## Arms

- **Demand:** XGBoost-free is arm D (convex model with a Kalman filter) plus the correction network
  and the two-number solar summary (experiments [220](../00220-dp-bg-stage4-assembly-with-drop-one/result.md)
  and [228](../00228-dp-bg-rolling-origin-stack-against-bg2/result.md)); XGBoost is BG2.
- **Solar:** XGBoost-free is the plant fed the mean and spread of the 51 members; XGBoost is arm 5
  ([experiment 224](../00224-metered-farm-dayahead-two-number/result.md)).
- **Wind:** XGBoost-free is the physics core B1; XGBoost is A0r (experiment 230, read only, scores
  provisional until that experiment merges).

## Code audit

Lists each XGBoost dependency in `packages/xgboost_forecaster`, `packages/baseline_forecasters`,
`packages/ml_core`, and `src/`, with what would replace it.
