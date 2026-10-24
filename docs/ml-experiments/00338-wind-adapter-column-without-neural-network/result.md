---
id: "00338"
slug: wind-adapter-column-without-neural-network
title: >-
  Can the adapter's effective wind speed be had as an XGBoost column without a neural network in
  the serving path?
theme: generator-forecasting
also_relevant_to:
  - calibration-and-uncertainty
status: planned
kind: change
date: 2026-10-24
size: >-
  complex by the research-default rule, because the experiment adds a fitted step to a family that
  adopts at most one change. Stored: private tables, boosted trees, and scored quantiles under
  data/ml-experiments/00338-wind-adapter-column-without-neural-network/ only, plus the default plus
  x forecasts of the adopted arm if there is one. Serving path: not touched by the experiment (a
  port of x1 would add a table file, a port of x2 a tree model). Degradation rule: not touched.
  Designs: several are defensible for the table bins and for the boosted tree's constraints, so
  this plan fixes one of each before any score is read. Callers: this folder's scripts, which import
  experiments 285, 303, and 326. One Sonnet code review and one Sonnet prose review (maintainer's
  brief), no Opus.
hypothesis: >-
  The gain of experiment 326's arm (c), the neural adapter's effective wind speed as an extra
  XGBoost column, can be kept without a neural network: a lookup table of speed scale by direction
  sector, speed ratio bin, and temperature bin (x1) keeps at least half of the gain, and a small
  monotone gradient-boosted tree (x2) keeps at least 80%.
prediction: >-
  Written before any score. x1 lowers the kept-row score by at least 0.05 points, which is at least
  half of (c)'s 0.099. x2 lowers it by at least 0.08. x3, a table that copies the network, is no
  better than x2. Dropping the direction sector or the speed ratio from the table removes most of
  the table's gain. The share of (c)'s gain that each non-neural column keeps is smaller at lead
  days 1 to 2 than at days 3 to 10.
decision_rule: >-
  The five adoption conditions of the research-default page, with Wind farms research default 1 as
  the default, read in the order x1, x2, x3, and the first arm that passes all five is adopted (at
  most one); x4 is a reference and cannot be adopted. Condition 1: the 99.75% month-block interval (level 1 - 0.01/4, as the brief sets for four arms) of (default plus x minus default) on the kept-row quantile score lies wholly below zero, and the point estimate is
  at least 0.02 points (a new fitted step, tier 2). Conditions 2 to 5 as in experiment 326. The
  share of (c)'s gain that x keeps is reported with a 95% month-resampled interval. A change that
  wins on condition 1 and fails another stays a candidate.
benchmark: >-
  The default: Wind farms research default 1 (experiment 303's P1, 11.352 points on the kept rows).
  Also reported beside it: climatology, the persistence on hours 0 to 6, the frozen XGBoost
  reference, and experiment 326's arm (c) as x4.
row_set_id: >-
  fold_a_tuesdays_v1 (53 Tuesday runs, 2025-07 to 2026-06; kept rows by experiment 210's mask are
  the headline, all rows beside). Replicate: the daily hindcast runs of 2025-01-01 to 2025-06-16.
metric: >-
  Quantile score over 11 levels in points of capacity, scored at 1 to 10 days, with days 1 to 2 and
  days 3 to 10 separately; tolerance 0.02 for the headline and 0.05 for the 0.01 and 0.99 pinball
  losses.
code: ml-experiments/00338-wind-adapter-column-without-neural-network/
production_status: not-for-production
adopted_in: none
---

**Plan, committed before any score is read.**

## Question

Experiment 326's arm (c) lowered the kept-row quantile score by 0.099 points (about 0.9%), but it
puts a neural network in the serving path. The serving path should stay close to "load a model, call
`predict`". If a non-neural column keeps most of the gain, the maintainer's decision becomes easy.

## Plan

1. **Family and version.** Wind farms research default 1 (W1 to W3), rows `fold_a_tuesdays_v1`,
   scored at 1 to 10 days, days 1 to 2 and 3 to 10 separately, metric and tolerances as in
   experiment 326. Single ERA5 fits (adapter and XGBoost on ERA5 before 2024-04-01), offset table
   on the hindcast, as in experiment 326.
2. **Arms.** The default (reproduction gate: within 0.01 of experiment 259's score on the kept rows)
   and default plus x, each x replacing experiment 326's column (c). Every x is fitted to the
   physics core's loss (mean absolute error of the expected power) on ERA5 before 2024-04-01, and
   enters XGBoost as the core's effective hub-height speed, so the column has the same definition
   as (c):
    - **x1:** a table of speed scale by 8 direction sectors, 6 speed-ratio (10 m over 100 m) bins,
      and 4 temperature bins. Bin edges are quantiles of the ERA5 training rows. The table is fitted
      jointly with the core by the same optimiser as the adapter, with the scale kept in the
      adapter's range of 0.8 to 1.25.
    - **x2:** a boosted tree (XGBoost, depth 3, 60 rounds, learning rate 0.1) that outputs the
      log speed scale from direction (sine, cosine), speed ratio, temperature, and hour of day
      (sine, cosine). It is trained end to end through the core's loss by a custom objective (pseudo-Huber
      loss on the core's expected power, Gauss-Newton Hessian), with the core's curve frozen at the
      fit without an adapter. Monotone constraints on the speed ratio and the temperature take the
      sign of the Spearman correlation between x1's fitted scale and each input on the ERA5 rows.
    - **x3:** a distillation: each cell of x1's table takes the median of the neural adapter's own
      scale (effective speed over the speed before the adapter) over the ERA5 rows in that cell.
    - **x4 (reference, not adoptable):** experiment 326's stored column (c).
3. **Tier.** A new fitted step (0.02 points), tier 2.
4. **Predictions that can fail.** In the front matter.
5. **Controls.** A shuffled-column placebo for x1 and x2; column subsampling off; a known-answer
   test (synthetic ensemble speeds with a known scale by sector, recovered by x1 within 3%); the
   leakage check that every table input is ensemble weather at forecast time. Drop-one arms of x1
   (without the direction sector, without the speed-ratio bin) test the prediction about what
   carries the gain.

## Five triggers

- **What gets stored:** private tables, trees, and scored quantiles only.
- **Serving path:** not touched by the experiment.
- **Degradation rule:** not touched.
- **More than one defensible design:** yes (bins, constraints), so the plan fixes them here.
- **Callers:** experiments 285, 303, and 326 are read-only imports.
