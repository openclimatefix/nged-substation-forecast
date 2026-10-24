---
id: "00343"
slug: wind-research-default-1-train-on-ensemble-mean
title: >-
  Does training the wind XGBoost model on the ensemble mean beat Wind farms research default 1, and
  does it shrink the offset table?
theme: generator-forecasting
also_relevant_to:
  - weather-products-and-blending
status: planned
kind: change
date: 2026-10-24
size: >-
  medium. Stored: the arms' scored quantiles, private, under
  data/ml-experiments/00343-wind-research-default-1-train-on-ensemble-mean/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: the arms are fixed by the maintainer's brief;
  cross-fitting follows experiment 303. Callers: this folder's scripts, which import experiments
  340, 336, 326, 303, 285, 259, and 230.
hypothesis: >-
  Wind farms research default 1 trains XGBoost on ERA5 and runs it on each of the 51 ensemble
  members, then corrects with the offset table. The ensemble's 100 m wind speed at the grid cell is
  lower than at the turbines (a factor of 1.13 closed 78% of the gap to the ERA5 oracle in
  experiment 307). An XGBoost model trained on the 51-member mean of the ensemble's own history
  (arm M, 2024-04 to 2025-06, no calendar columns) would learn that bias and need less offset-table
  correction. Arm MS adds the member standard deviation of the 100 m and 10 m speeds. The three
  metered wind farms W1 to W3 are scored at 1 to 10 days.
prediction: >-
  Written before any score. P1: M trails the default by 0.2 to 1.0 points on the kept rows, because
  the ensemble history is about 15 months of runs against ERA5's 4.4 years. P2: M0 (M without the
  offset table) is closer to M than the default without its table is to the default. P3: MS beats M
  by at least 0.05 points on the 0.95 and 0.99 pinball losses and is no better on the median.
decision_rule: >-
  The five adoption conditions of the research-default page, read first for M and then for MS; the
  first arm that passes all five is adopted, at most one. Condition 1: the 99.5% month-block
  interval (level 1 - 0.01/2 for two arms) of (arm minus default) on the kept-row quantile score at
  1 to 10 days lies wholly below zero, and the gain is at least 0 points (a changed training
  setting that adds no pipeline step, so the minimum gain of the tier is 0). Condition 2: the
  one-sided 95% upper bound of the harm to the pooled 0.01 and 0.99 pinball losses is at most 0.05
  points each, and the upper bound of (arm minus default) on the rows below the ERA5 cut-in, on
  the experiment 210 idle rows, and on all half-hours is at most 0.1 points each. Condition 3:
  replication on the 2025-01 to 2025-06 hindcast runs gives the same sign with the table fitted on
  earlier valid times. Condition 4: the reproduction gate (the default within 0.01 points of
  experiment 326's 11.352 on the kept rows), a known-answer test, a placebo, a leakage check, and
  no unresolved audit trigger. Condition 5: only inputs available at forecast time.
benchmark: >-
  The default: Wind farms research default 1 (experiment 340's stored default, 11.352 points on the
  kept rows). Also reported: climatology and the persistence reference where experiment 340 scored
  them.
row_set_id: >-
  fold_a_tuesdays_v1 (53 Tuesday runs, 2025-07 to 2026-06; kept rows by experiment 210's mask are
  the headline). Replicate: the daily hindcast runs of 2025-01-01 to 2025-06-16.
metric: >-
  Quantile score over 11 levels in points of capacity at 1 to 10 days, with days 1 to 2 and days 3
  to 10 separately; tolerance 0.02 for the headline and 0.05 for the 0.01 and 0.99 pinball losses.
code: ml-experiments/00343-wind-research-default-1-train-on-ensemble-mean/
production_status: not-for-production
adopted_in: none
---

**Take-home: not yet run.** The result is written after the arms are scored.

## Plan

- **Arms.** D: the default (stored refit, reproduction gate). M: the same six weather columns as
  the default, on the 51-member mean of each column (sine and cosine of direction averaged, as the
  columns are), trained on the ensemble hindcast (2024-04-01 to 2025-06-16 runs, one row per run
  and valid time), same quantile loss, 500 rounds, eta 0.05, depth 6, binned-curve base margin.
  MS: M plus the member standard deviation of the 100 m speed and of the 10 m speed (the 10 m speed
  is the column other than the 100 m speed with the highest gain in the default, in all three
  farms). M0: M without the offset table, descriptive and never adopted. No calendar columns, no
  lead column, three seeds.
- **Cross-fitting.** The hindcast is both the training set and the set that fits the spread factor
  and the offset table, so M and MS are cross-fitted in three blocks of consecutive run dates as in
  experiment 303: the hindcast quantiles come from models fitted without the held-out block and
  the runs whose valid times overlap it. The scored year comes from a model fitted on all usable
  hindcast cases.
- **Reuse.** Scoring, the kept rows, the low-wind subsets, the replicate, and the spread factor and
  offset table follow experiments 340, 326, and 303 (imported, read only).
- **Predictions that can fail:** P1, P2, and P3 above.
- **Five triggers:** stored: private scored quantiles; serving path: no; degradation rule: no;
  designs: arms fixed by the brief; callers: this folder's scripts. Medium.
- **Ensemble-mean training is a lab condition** for serving: the mean of 51 members is available
  at forecast time, so condition 5 holds.
