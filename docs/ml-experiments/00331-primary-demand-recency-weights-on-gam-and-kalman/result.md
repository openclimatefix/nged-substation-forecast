---
id: '00331'
slug: primary-demand-recency-weights-on-gam-and-kalman
title: >-
  Does giving recent rows more weight improve next-month Primary demand forecasts?
theme: retraining-and-training-history
also_relevant_to:
  - lsc-detectors
status: planned
kind: other
size: >-
  Simple on the five triggers, with one Sonnet plan review and one Sonnet code review because the
  brief asks for them. Stored: private tables under data/ml-experiments/00331-*/ only. Serving
  path: not touched. Degradation rule: not touched. Designs: one defensible design, since the row
  weights are the only change and `fit_gam` already accepts weights. Callers: this folder's
  scripts, which call the reviewed `studies.dp_gam` functions and the code of experiments 295 and
  328 unchanged; no change under `packages/`.
hypothesis: >-
  When the additive (GAM) model with a Kalman filter on its daily residual is refitted monthly,
  giving older training rows exponentially less weight forecasts the next month better than
  equal weights on the full history, and no worse than training on the last 24 months only.
prediction: >-
  Written before any score is read; each can fail. The best exponential half-life (3, 6, 12 or 24
  months, chosen on odd months and scored on even months, then the roles swapped) beats equal
  weights on the full history (R0) by at least 0.02 points of the quantile score, with a 95%
  month-block interval wholly below zero, and is not worse than the 24-month window (W24) by more
  than 0.01 points. Secondary: at D8 in the three months after its 2023-08-15 break the best
  half-life moves q99 coverage towards nominal against R0. In the injected-step worlds of
  experiment 328 the 3-month half-life beats R0 in the first 90 days after the step. It fails if
  R0 or W24 matches the best half-life.
decision_rule: >-
  Five plan items, written before any score is read. (1) Family and base: Primary demand,
  monthly-refitted GAM_ONLY with the Kalman filter on past weather (ERA5 and CAMS, lab
  conditions), as in experiment 328, because the full Primary demand research default 3 chain
  (ensemble members, heads, solar summary) cannot be refitted per arm on CPU in the budget. This
  is why `kind` is `other`: a win is a candidate for a later `change` experiment that ports the
  weights onto Primary demand research default 3. (2) Arms: R0 equal weights on full history; W24
  the last 24 months (experiment 328's arm B); H3, H6, H12, H24 exponential weights with those
  half-lives in months; S12 the best half-life with a seasonal bonus (rows within 30 days of the
  forecast month's mid-date in earlier years get weight 2 times, before normalising). (3) Tier: a
  changed setting (tier 0 in the rule), but the manager's minimum gain of 0.02 points is used,
  because weights add a hyperparameter to every monthly refit. (4) Pre-registered contrasts: C1
  best H minus R0 (95% month-block interval, must lie below zero and reach -0.02); C2 best H minus
  W24 (one-sided 95% bound must stay under +0.01); C3 S12 minus best H, descriptive, and S12 is the
  one extra arm adopted only if C3 is below zero with an interval wholly below zero; C4 D8 q99
  coverage after the break, descriptive. Non-inferiority of the best H against R0 at every Primary
  except D8, margin 0.01 points, one-sided 95%. (5) Controls: a reproduction gate (R0 and W24 at
  ridge 0.1 and filter gain 0.10 must reproduce experiment 328's arms A and B per Primary within
  0.005 points); a known-answer test (injected 10% and 25% steps of experiment 328's report
  onsets, where H3 must beat R0 in the first 90 days); a placebo (a half-life of 1000 years must
  equal R0 within 0.001); and a leakage check (every weight uses only the row time and the cut,
  and `fit_gam` rejects rows after the cut). Hyperparameters per arm (ridge in 0.03, 0.1, 0.3 and
  Kalman gain in 0.05, 0.10, 0.20) are chosen on the selection half and scored on the other half.
  Verdict `supported` when the prediction holds, `not supported` when R0 or W24 matches the best
  half-life (two-sided interval of the difference includes values within 0.02), `small gain` when
  the interval excludes zero but the gain is under 0.02, `inconclusive` otherwise.
benchmark: >-
  R0 (equal weights, full history) and W24 (last 24 months), both with the same Kalman filter and
  bands; `manual_heuristic` and `climatology` as references.
row_set_id: >-
  Own rows (not comparable with the leaderboard): the Primaries of experiment 328, monthly
  forecast origins 2021-09 to 2026-06, the month after each origin on past weather, rows outside
  the hand-filtered load-shift spans.
metric: >-
  Quantile score over 11 levels in points of effective capacity, mean absolute error of the
  median, q95 and q99 pinball loss, q01 and q99 coverage.
code: ml-experiments/00331-primary-demand-recency-weights-on-gam-and-kalman/
date: 2026-10-24
production_status: not-for-production
adopted_in: none
---

**Take-home: not yet run.** This page is the plan, committed before any score is read.

We ask whether a Primary substation's demand forecast for the next month improves when the
monthly refit gives recent training rows more weight than old rows. Experiment 17 found a gain of
0.92 points for XGBoost on ERA5 weather, and this experiment tests the same idea on the additive
model with a Kalman filter that experiment 328 used.

- **Lab conditions:** weather is past ERA5 and CAMS, so the forecasts are an upper bound on what
  live ensemble weather allows, and the scored lead is the month after the origin, not a 1 to 10
  day range.
- **Not run:** the ensemble-scored variant and the curtailment cost, because both need a refit per
  ensemble member and the brief allows them only if cheap.
