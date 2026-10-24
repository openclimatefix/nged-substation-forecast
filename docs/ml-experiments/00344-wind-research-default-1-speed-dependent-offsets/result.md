---
id: "00344"
slug: wind-research-default-1-speed-dependent-offsets
title: >-
  Do quantile offsets that depend on the forecast wind speed improve Wind farms research default 1?
theme: generator-forecasting
also_relevant_to:
  - calibration-and-uncertainty
  - curtailment-cost
status: planned
kind: change
date: 2026-10-24
size: >-
  medium. Stored: the fitted offsets and scored quantiles of the arms, private, under
  data/ml-experiments/00344-wind-research-default-1-speed-dependent-offsets/ only; an adopted arm is
  written once with experiment_store.write_predictions. Serving path: not touched. Degradation rule:
  not touched. Designs: three arms fixed by the maintainer's brief; the fitting rules are fixed in
  this plan before any score. Callers: this folder's scripts, which import experiments 336, 326,
  303, 285, 259, 230, and 342.
hypothesis: >-
  Experiment 340 found that a constant upward shift of the wind quantiles gains 0.09 to 0.12 points
  on the kept rows but worsens the rows below cut-in by 0.51 to 0.78, and that the default's median
  is too low at 8 to 15 m/s and too high below 5 m/s. An offset that depends on the ensemble-mean
  100 m wind speed at forecast time lifts the quantiles where the power curve is steep and leaves
  them alone where the forecast is near zero, so it keeps the gain without the low-wind harm. The
  starting point is Wind farms research default 1, the three metered wind farms W1 to W3 are
  scored, and the lead range is 1 to 10 days.
prediction: >-
  Written before any score. P1: x2 lowers the kept-row score by 0.05 to 0.12 points and worsens the
  rows below cut-in (the MEAN mask) by under 0.1. P2: x1 gains less than x2 on the replicate,
  because its cells are sparse. P3: x3 and x2 differ by under 0.03 points on the kept rows. P4: at
  least 60% of x2's kept-row gain comes from rows with forecast speed of 8 to 15 m/s. P5: no arm
  moves the 0.99 pinball loss by more than 0.05. A caution written before any score: the offset
  table already fits the hindcast per farm and lead day, so the speed structure that remains may be
  small, as the constant shift of experiment 340 was exactly zero on the hindcast.
decision_rule: >-
  The five adoption conditions of the research-default page, read in the order x2, x1, x3; the
  first arm that passes all five is adopted, at most one. Tier: each arm adds a fitted step, so the
  minimum gain is 0.02 points. Condition 1, gain: the 99.67% month-block interval (level 1 - 0.01/3,
  three arms) of (default plus x, minus default) on the kept-row quantile score at 1 to 10 days lies
  wholly below zero and the point estimate gains at least 0.02. Condition 2, no harm: the one-sided
  95% upper bound of the harm is at most 0.02 on all half-hours, at most 0.05 on the pooled 0.01
  and 0.99 pinball losses, and, as a guard carried from experiment 340, at most 0.1 on the rows
  below the ERA5 cut-in, the rows below the MEAN-mask cut-in, and the experiment 210 idle rows. Each
  farm and the rows above the rated speed are reported without a threshold. Condition 3,
  replication: the replicate period 2025-01 to 2025-06 (table and offsets fitted on hindcast valid
  times before 2025-01) gives the same sign, an estimate of at least half of the smaller of the main
  estimate and 0.02, and a t test on the six monthly paired differences that does not contradict it.
  Condition 4: the reproduction gate (default equals experiment 340's stored default and scores
  within 0.01 of experiment 303's P1), a known-answer test (a known speed-dependent bias applied to
  the default's quantiles is recovered by x2 within 20%), a shuffled-speed placebo (the arm's offsets
  applied with the speed of another date gain under a quarter of the arm's gain), a leakage check
  (the fits read hindcast rows that end before the scored year, and the key is the ensemble's
  forecast-time speed), and no unresolved audit trigger. Condition 5: the key and the fit use only
  data available at forecast time.
benchmark: >-
  The default: Wind farms research default 1 (experiment 336's refit, 11.352 points on the kept
  rows). Also reported: the default plus the out-of-sample constant shift of experiment 340
  (Sdrift), the frozen XGBoost reference, climatology, and the persistence reference where
  experiment 340 scored it.
row_set_id: >-
  fold_a_tuesdays_v1 (53 Tuesday runs, 2025-07 to 2026-06; kept rows by experiment 210's mask are
  the headline, all rows beside). Replicate: the daily hindcast runs of 2025-01-01 to 2025-06-16.
metric: >-
  Quantile score over 11 levels in points of capacity, scored at 1 to 10 days, with days 1 to 2 and
  days 3 to 10 separately; tolerance 0.02 for the headline and 0.05 for the 0.01 and 0.99 pinball
  losses.
code: ml-experiments/00344-wind-research-default-1-speed-dependent-offsets/
---

## Plan

- **Family and version.** Wind farms research default 1 as experiment 340 and 342 use it (the
  research-default log lists no wind version yet, so the name is a working name; no later version
  exists at plan time). Three farms, three seeds, rows `fold_a_tuesdays_v1`, lead range 1 to 10
  days. Tier 1: every arm acts on the default's stored quantiles (spread factor and offset table
  rebuilt from experiment 336's stored mixtures, as in experiment 340). A family comparison is not
  needed for a tier 1 change on stored forecasts.
- **The key.** The mean over the 51 ensemble members of the 100 m wind speed at the valid time
  (forecast-time data only), the same quantity as the MEAN mask of experiment 336.
- **x1.** An additive offset per farm, lead day, quantile (all 11), and speed bin (edges at 3, 6,
  9, 12, and 15 m/s), the pinball-optimal offset of the post-table residual in the cell, fitted on
  the hindcast; a cell with under 200 rows gets no offset. This equals the offset table keyed by the
  speed bin as well.
- **x2.** An additive correction per farm and quantile, c(v) = c_low + (c_high - c_low) s(v), where
  s is a smoothstep rising from 0 to 1 over a window of 6 m/s centred on a midpoint chosen from
  5, 6.5, 8, 9.5, and 11 m/s. The curve is monotone with three parameters (c_low, c_high, and the
  midpoint), fitted on the post-table hindcast residuals by minimising the pinball loss.
- **x3.** One shape per quantile (c_low, c_high, midpoint) fitted on the three farms pooled, times
  one scale per farm (a grid from 0 to 3) fitted on the farm's summed pinball loss over the levels.
- **Controls.** Shuffled-speed placebo (the speed of a scored row is replaced by that of a random
  row of the same farm and lead day, seed fixed, and the fitted offsets are applied); known-answer
  test; leakage check; reproduction gate.
- **Reported groups.** Each farm, the rows the MEAN mask sets, the rows above rated speed, the
  speed groups below 5, 5 to 8, 8 to 15, and 15 m/s or more, days 1 to 2, and days 3 to 10.
- **Replicate and cap.** As in experiment 340. Cap of 3 hours of work; stop and report if the
  reproduction gate fails.
- **Earlier advice followed.** Experiment 340: test a wind-speed-dependent correction against the
  low-wind rows, and do not fit a constant shift on the hindcast the table was fitted on. Experiment
  342 keyed only the upper quantiles q90 to q99 by speed (change of +1.0% in the 0.99 pinball loss);
  this experiment keys all 11 levels and so differs from it in the lower and middle quantiles.
