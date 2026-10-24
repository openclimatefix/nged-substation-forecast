---
id: "00342"
slug: wind-research-default-1-upper-tail-offsets-and-head
title: >-
  Can the upper quantiles of Wind farms research default 1 be made to cover the observed power,
  and does an ensemble-aware correction beat a plain recalibration?
theme: generator-forecasting
also_relevant_to:
  - calibration-and-uncertainty
  - curtailment-cost
status: planned
kind: change
date: 2026-10-24
size: >-
  medium. Stored: private shift tables, boosted trees, and scored quantiles under
  data/ml-experiments/00342-wind-research-default-1-upper-tail-offsets-and-head/ only; no
  forecasts are stored through write_predictions unless an arm is adopted. Serving path: not touched
  (a port of x1 adds a table, x2 a tree model, x3 one constant per farm). Degradation rule: not
  touched. Designs: the arms are fixed by the maintainer's brief, with one departure for x1 stated
  below. Callers: this folder's scripts, which import experiments 340, 336, 326, 303, 285, 259,
  and 230. One Sonnet code review and one Sonnet prose review, no Opus.
hypothesis: >-
  The upper quantiles of Wind farms research default 1 are too low: in experiment 324 the observed
  power exceeded q99 in 1.8% of half-hours (1% nominal) and q95 in 6.7% (5% nominal). Three
  corrections fitted on the hindcast can fix that without harming the headline score. x1 is a table
  of offsets for q90 to q99 that also depends on the ensemble-mean wind speed. x2 is a small
  boosted-tree head, fed the default's quantiles and summaries of the ensemble members' wind speed,
  that adds offsets to q90, q95, and q99. x3 lets the share of ensemble members above the farm's
  saturation speed set a probability mass at the farm's scale.
prediction: >-
  Written before any score. x1 brings the share above q99 from 1.8% to under 1.5% and lowers the
  0.99 pinball loss by at least 3%. x2 lowers the 0.99 pinball loss by at least 10%, and the head
  fed only the default's quantiles lowers it by at least half as much, so recalibration explains
  most of the gain. x3 matters only at W1 and W2, and at W3 (two turbines, which saturate often) it
  matches x1. No arm moves the headline score by more than 0.05 points. The share above q99 stays
  above 1% for every arm. A caution written before any score: the default's offset table already
  gives every farm and lead day its own offset at every level, fitted on the same hindcast, so an
  offset table keyed only by farm and lead day (the brief's literal x1, run as the descriptive arm
  x1lit) reproduces the default; experiment 340's per-level shifts found +1.83 points at q99 and
  nothing else, so a gain from x1 can only come from the wind-speed key.
decision_rule: >-
  The five adoption conditions of the research-default page, read in the order x1, x2, x3; the
  first arm that passes all five is adopted, at most one. Condition 1, gain on the targeted group:
  the 99.67% month-block interval (level 1 - 0.01/3) of (default plus x, minus default) on the mean
  of the 0.95 and 0.99 pinball losses, on the kept rows at 1 to 10 days, lies wholly below zero,
  and the point estimate gains at least 0 for x1 (a changed setting), 0.03 for x2 (a new head), and
  0.02 for x3 (a new fitted step), in points of capacity. Condition 2, no harm: the one-sided 95%
  upper bound of the change in the headline quantile score is at most 0.02, of the 0.01 pinball
  loss at most 0.05, and (maintainer's low-wind condition from experiment 340) of the score on
  the rows below the ERA5 cut-in, on the experiment 210 idle rows, and on all half-hours at most
  0.1. Condition 3, replication: the replicate period 2025-01 to 2025-06 (tables and heads fitted
  on hindcast valid times before 2025-01) gives the same sign, at least half of the smaller of the
  main estimate and the minimum gain, and a t test on the six monthly differences does not
  contradict it. Condition 4: the reproduction gate (default within 0.01 of experiment 303's P1), a
  known-answer test (the wind-speed-keyed fit recovers an injected offset at the right wind
  speeds), a placebo (x2 fed member summaries from other dates must keep under half of x2's gain),
  a leakage check (every fit reads hindcast runs only), and no unresolved audit trigger. Condition
  5: every fit reads hindcast runs only, which exist at forecast time. A change that wins
  condition 1 and fails another stays a candidate.
benchmark: >-
  The default: Wind farms research default 1 (experiment 303's P1 refitted in experiment 336,
  11.352 points on the kept rows). Reference rows, descriptive: the default plus experiment 340's
  drift shift, and experiment 339's mixture head (its arm x1). Also reported: the frozen XGBoost
  reference, climatology, and persistence where experiment 340 scored it.
row_set_id: >-
  fold_a_tuesdays_v1 (53 Tuesday runs, 2025-07 to 2026-06; kept rows by experiment 210's mask are
  the headline, all rows beside). Replicate: the daily hindcast runs of 2025-01-01 to 2025-06-16.
metric: >-
  Quantile score over 11 levels in points of capacity, scored at 1 to 10 days, with days 1 to 2 and
  days 3 to 10 separately; targeted metric the mean of the 0.95 and 0.99 pinball losses; tolerances
  0.02 for the headline and 0.05 for the 0.01 and 0.99 pinball losses.
code: ml-experiments/00342-wind-research-default-1-upper-tail-offsets-and-head/
---

**Planned, not yet run.** This page is the plan, committed before any score is read.

## Plan

- **Family and default.** Default plus x against Wind farms research default 1 (unchanged by
  experiments 338 to 340), on the three metered wind farms W1 to W3, scored at 1 to 10 days. All
  three arms are tier 2.
- **x1, upper-tail offsets keyed by wind speed.** Offsets for q90, q95, and q99 only, fitted as the
  residual quantile per farm, lead day, and bin of the ensemble-mean 100 m speed (edges 0, 5, 8,
  11, 14 m/s), on the default's final hindcast quantiles; a cell with fewer than 200 rows gets
  zero. Then every quantile is capped at the farm's scale, the 99.9th percentile of observed power
  on the hindcast rows whose ERA5 speed is at or above the farm's saturation speed.
  **Departure from the brief:** the brief's x1 (offsets per lead day and farm) is what the default's
  offset table already does, so it is run as the descriptive arm x1lit (default plus the cap) and is
  not eligible for adoption.
- **x2, tail head.** One boosted tree per level (q90, q95, q99), depth 3, 150 rounds, learning rate
  0.05, minimum child weight 50, subsample 0.8, pinball loss, base margin the default's quantile
  at that level. Inputs: the default's 11 quantiles, the ensemble mean, standard deviation, and
  share above the saturation speed of the members' 100 m speed, the lead, and the farm.
  Trained on the usable hindcast cases; three seeds (the default's seeds) in the main table, and
  five seeds on default seed 0 for the spread.
- **x3, saturation mass.** The forecast distribution becomes (1 - w) F + w * delta(top), with
  w = lambda_farm * (share of members above the saturation speed) and top the farm's scale.
  lambda_farm is one constant per farm fitted by grid on the hindcast quantile score of levels 0.9
  to 0.99.
- **Controls.** x2 fed only the default's quantiles and the lead (recalibration without ensemble
  information); x2 with the member summaries taken from a random other case of the same farm and
  lead day (shuffled-input placebo); x2 trained on two farms and scored on the third
  (leave-one-farm-out); a known-answer test on synthetic quantiles.
- **Reporting.** Tail table (shares below q01 and q05, above q95 and q99; pinball losses at q01,
  q05, q95, q99; coverage 5% to 95%), per farm and without any one farm; power-curve plots of
  the forecast q95 and q99 against the observed binned curve per farm and pooled, with a zoomed
  low-wind panel; the physics core plus adapter plus tail head named as the standard-architecture
  arm (experiment 339's x1 and x2) with its gap to the best family. The curtailment-cost metric is
  not run if its code needs a node with several generators; the page states which.
- **Cap.** Five hours; stop and report if the default fails to reproduce within 0.01.
- **Size (five triggers).** Stores private forecasts only; not on the serving path; no degradation
  rule; arms fixed by the brief; callers are this folder's scripts. Medium.
