---
id: "00340"
slug: wind-research-default-1-constant-shift-then-low-wind-drop
title: >-
  Does a constant upward shift of the wind quantiles reproduce the gain of dropping low-wind training
  rows in Wind farms research default 1?
theme: generator-forecasting
also_relevant_to:
  - calibration-and-uncertainty
  - curtailment-cost
status: planned
kind: change
date: 2026-10-24
size: >-
  medium. Stored: the shift constants and scored quantiles of four arms, private, under
  data/ml-experiments/00340-wind-research-default-1-constant-shift-then-low-wind-drop/ only, plus
  the adopted arm's forecasts if there is one. Serving path: not touched (a constant per quantile
  added to the final quantiles would need no serving-path change; the page says so). Degradation
  rule: not touched. Designs: the arms are fixed by the maintainer's brief, and the fitting rule for
  the shift is fixed below before any scored row is read. Callers: this folder's scripts, which
  import experiments 336, 326, 303, 285, 259, and 230. One Sonnet code review and one Sonnet prose
  review, no Opus.
hypothesis: >-
  Experiment 336 found that dropping the ERA5 training rows below the turbines' cut-in speed (x2)
  gains 0.117 points on the kept rows, and its audit traced about 85% of the gain to a constant
  upward shift of the quantiles. If that is right, a constant shift fitted on the 442-run hindcast
  (arm S) reproduces most of the gain without changing the training rows, and x2 adds little beyond
  the shift (arm S+X2). Wind farms research default 1 is the starting point, the three metered wind
  farms W1 to W3 are scored, and the lead range is 1 to 10 days.
prediction: >-
  Written before any score. P1: S gains 0.08 to 0.12 points on the kept rows, at least 80% of X2's
  gain. P2: S+X2 gains less than 0.03 points over S alone. P3: S does not worsen the score on the
  rows below the ERA5 cut-in by more than 0.2 points, while X2 does (+0.63 in experiment 336). P4:
  the shift S transfers to the replicate period with the same sign. A caution written before any
  score: the offset table is already fitted on the same hindcast, so a shift fitted on those runs
  alone may come out near zero (a hindcast-only probe before the plan was committed shows a
  quantile-score optimum at about +0.2 points of capacity, not +0.9); if so P1 fails, and the
  oracle shift (below) says what a drift-corrected shift would have done.
decision_rule: >-
  The five adoption conditions of the research-default page, read in the order S, S-q, S+X2, X2;
  the first arm that passes all five is adopted, at most one. Condition 1, gain: the 99.75%
  month-block interval of (default plus x, minus default) on the kept-row quantile score at 1 to 10
  days lies wholly below zero (level 1 - 0.01/4 for four arms), and the point estimate gains at
  least the minimum gain of the tier: 0.02 points for S, S-q, and S+X2 (each adds a fitted step,
  the shift), and 0 for X2 (a changed setting, as in experiment 336). Condition 2, no harm: the
  one-sided 95% upper bound of the harm to the pooled 0.01 and 0.99 pinball losses is at most 0.05
  points each, plus the maintainer's condition: the upper bound of (arm minus default) on the rows
  below the ERA5 cut-in, on the experiment 210 idle rows, and on all half-hours is at most 0.1
  points each. Condition 3, replication: the replicate period 2025-01 to 2025-06 gives the same
  sign (shifts and tables refitted on hindcast valid times before 2025-01), the replicate estimate
  is at least half of the smaller of the main estimate and the minimum gain, and a t test on the six
  monthly paired differences does not contradict it. Condition 4: the reproduction gate (default
  equals experiment 303's P1), a known-answer test (the shift fit recovers an injected shift), a
  placebo (the shift with its sign reversed must not gain), a leakage check (the fit reads no
  scored row), and no unresolved audit trigger. Condition 5: the shift is fitted on hindcast runs
  only, which are available at forecast time. The oracle shift (fitted on the scored year) is
  descriptive and is never adopted.
benchmark: >-
  The default: Wind farms research default 1 (experiment 336's refit, 11.352 points on the kept
  rows). Also reported: the frozen XGBoost reference (the default without the offset table),
  climatology, and the persistence reference where experiment 336 scored it.
row_set_id: >-
  fold_a_tuesdays_v1 (53 Tuesday runs, 2025-07 to 2026-06; kept rows by experiment 210's mask are
  the headline, all rows beside). Replicate: the daily hindcast runs of 2025-01-01 to 2025-06-16.
metric: >-
  Quantile score over 11 levels in points of capacity, scored at 1 to 10 days, with days 1 to 2 and
  days 3 to 10 separately; tolerance 0.02 for the headline and 0.05 for the 0.01 and 0.99 pinball
  losses.
code: ml-experiments/00340-wind-research-default-1-constant-shift-then-low-wind-drop/
---
## Plan

- **Family and default.** Wind farms research default 1 as reproduced in experiment 336
  (`default`: XGBoost fitted once on ERA5, the spread factor of experiment 33, then the
  (farm, lead day) offset table of experiment 259). Tier 2 for X2 (inside the fitted stage), tier 1
  for S and S-q (act on stored forecasts). Minimum gains as in the decision rule.
- **Arms.** S: one constant added to all 11 final quantiles, then clipped to 0 to 1 and sorted.
  S-q: one constant per quantile level. X2: experiment 336's x2, read from its stored mixtures.
  S+X2: x2's final quantiles plus a constant fitted on the hindcast made from the x2 model. Beside
  them: the default, the frozen XGBoost reference, and climatology.
- **Fitting rule, fixed before any scored row is read.** The constants are fitted on the usable
  hindcast runs (all farms, all rows, no mask) after the spread factor and the offset table, as the
  table was. S minimises the headline quantile score over a grid of 0.0001 (0.01 points of
  capacity). S-q sets constant k to the level-k quantile of the hindcast residuals (observed minus
  quantile k). Per-farm fits are reported but not applied. The replicate version refits table and
  constants on hindcast valid times before 2025-01.
- **Oracle shift (descriptive).** The same S fit on the scored year, labelled an oracle: it shows
  what a drift-corrected constant would do, and is never adopted.
- **Controls.** Reproduction gate; known-answer test (inject a known shift into synthetic
  quantiles, recover it); sign-reversed placebo; leakage check (the fit function receives no scored
  row, asserted by row count).
- **Power curves.** Per farm and pooled, power as a percentage of capacity against 100 m wind
  speed: observed binned medians with 5% to 95% bands, with each arm's median and its 5% and 95%
  quantiles (forecast x axis: ensemble-mean speed at forecast time; observed x axis: ERA5 speed at
  the valid time), a zoomed panel for 0 to 8 m/s, and forecast minus observed median per arm.
- **Compute.** No refits: every arm is built from experiment 336's stored mixtures. One process,
  `OMP_NUM_THREADS=1`.
