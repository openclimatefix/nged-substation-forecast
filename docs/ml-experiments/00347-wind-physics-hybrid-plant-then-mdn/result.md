---
id: "00347"
slug: wind-physics-hybrid-plant-then-mdn
title: >-
  Does a frozen physics plant model of a wind farm, fed each ensemble member in turn and followed by
  a mixture-density head on the summary of the resulting power ensemble, forecast the three metered
  wind farms better than Wind farms research default 1?
theme: generator-forecasting
also_relevant_to:
  - calibration-and-uncertainty
status: planned
date: 2026-10-24
kind: change
size: >-
  complex. Stored: private plant parameters, flagged-block tables, power-ensemble summaries, head
  outputs and predictions under data/ml-experiments/00347-wind-physics-hybrid-plant-then-mdn/ only.
  Serving path: not touched (the page says what serving each arm needs). Degradation rule: not
  touched. Designs: several are defensible (plant curve family, direction term, flagging rule, head
  inputs), so this plan fixes one of each before any Stage 4 score. Callers: this folder's scripts,
  which import experiments 230, 285, 303, and 339 and call no reviewed module beyond `studies.*`.
  Sonnet only, one code review and one prose review.
hypothesis: >-
  A physics plant model of each metered wind farm (a smooth cubic speed-to-power curve with
  cut-in, rated, and cut-out speeds, a direction-dependent loss, and priors that make the untrained
  model already sensible), fitted on ERA5 before 2024-04-01 and frozen, turns each of the 51
  ensemble members into a power value. A three-component Gaussian mixture-density network (MDN) fed
  the summary of those 51 values, the lead, and the farm then scores within 0.15 points of Wind
  farms research default 1 on the kept rows at 1 to 10 days, without a neural adapter. A second plant
  fitted after masking the blocks where the farm fell short of the first plant (a healthy-farm
  plant) lets the head learn the shortfall.
prediction: >-
  Written before any Stage 4 score. The Stage 3 direction term lowers the ERA5 hold-out score by
  0.02 to 0.1 relative to Stage 2, with a larger gain at W1 and W2 than at W3 (experiment 276's D6
  saw a tie of 0.052). y2 beats y1 by under 0.05. y2 is within 0.15 of the default on the kept rows
  and does not beat it by 0.03. y3 is within 0.03 of y2. A head fed only the plant's median does as
  well as the full head (experiment 339), so the head recalibrates. The share above q99 stays above
  1%. y4 (pass-2 plant) differs from y2 by under 0.05; if the placebo mask moves the fitted scale as
  much as the real mask, the flags are not hardware failures.
decision_rule: >-
  The five adoption conditions of the research-default page, read in the order y4, y3, y2, y1; the
  first arm that passes all five is adopted (at most one). Condition 1: the 99.75% month-block
  interval (level 1 - 0.01/4) of (arm minus default) on the kept-row quantile score at 1 to 10 days
  lies wholly below zero and the point estimate is at least 0.03 points (a new head). Condition 2:
  the one-sided 95% upper bound of the harm to the pooled 0.01 and 0.99 pinball losses is at most
  0.05 points each. Condition 3: the replicate (head trained on 2024-04 to 2024-12, scored on the
  daily runs of 2025-01 to 2025-06) has the same sign and a t test on monthly differences does not
  contradict it. Condition 4: reproduction gate (the default matches experiment 259 within 0.01),
  shuffled-summary placebo, five seeds, leakage check, known-answer tests pass, and no audit
  trigger is unresolved. Condition 5: the page says what serving needs. A winner on condition 1
  that fails another stays a candidate. Stages 1 to 3b are diagnostics and decide nothing.
benchmark: >-
  The default: Wind farms research default 1, the ERA5-trained XGBoost on the 51 members plus
  experiment 259's offset table (11.352 points on the kept rows). Also beside it: the frozen
  XGBoost reference, the XGBoost with the adapter column (experiment 326 arm c), experiment 324's
  W4, climatology, persistence on hours 0 to 6, the Stage 2 and 3 plants with the offset table and
  no MDN, and the Stage 3 plant with ERA5 (the Step 0 ceiling).
row_set_id: >-
  fold_a_tuesdays_v1 (53 Tuesday runs, 2025-07 to 2026-06; kept rows by experiment 210's mask are
  the headline, all half-hours beside). Hindcast for the head: 2024-04-01 to 2025-06-16.
metric: >-
  Quantile score over 11 levels in points of capacity, scored at 1 to 10 days (days 1 to 2 and 3 to
  10 separately); tolerance 0.02 for the headline, 0.05 for all half-hours and for the 0.01 and
  0.99 pinball losses.
code: ml-experiments/00347-wind-physics-hybrid-plant-then-mdn/
---

# Experiment 347 (planned)

The full spec is the worker brief `wind_physics_hybrid_plant_then_mdn_worker_brief.md` (private).
Stages 1 to 3b (data analysis, plant with priors, direction loss, fit-twice with flagged blocks) are
diagnostics; Stage 4 compares arms y1 to y4 with the default. The headline caveat is "scored at 1 to
10 days".
