---
id: '00287'
slug: metered-farm-spread-binned-plant-heads
title: >-
  Which head gives the best day-ahead band on the spread-binned solar-farm power model?
theme: metered-solar-farms
also_relevant_to:
- calibration-and-uncertainty
status: planned
size: >-
  medium. Stored: private predictions and tables under
  data/ml-experiments/00287-metered-farm-spread-binned-plant-heads/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one defensible head (additive, neural,
  mixture-density, blend), so all are arms. Callers: this folder's scripts, copied from experiments
  270, 275, and 260, with no new reviewed module. Asset graph: not touched.
hypothesis: >-
  On top of the spread-binned plant (experiment 270's P2), an additive model of the residuals gives
  a band as good as a neural head with monotone quantile output, so the additive model is the
  explainable candidate and the neural head adds nothing worth porting.
prediction: >-
  The neural head does not beat the additive model: its quantile score is not at least 1.5% lower,
  with a 98.75% month-block interval excluding zero. This is the prediction that could fail. The
  per-quantile blend of the additive model, the neural head, and XGBoost beats the best single arm
  by under 0.5%.
decision_rule: >-
  Quantile score in points of capacity on experiment 270's rows, equal weight per farm, month-block
  resampling of the 12 scored months at 98.75% (Bonferroni across the four headline contrasts: neural
  head minus additive model, mixture-density head minus additive model, blend minus best single
  arm, additive model minus P2). The neural head is preferred only if its interval lies wholly
  below -1.5%. The threshold of 1.5% is set by the cost of porting a neural network and its
  training loop into the serving path. Stop rule: if the blend beats the best single arm by under
  0.5%, report the best single arm as the candidate and stop.
benchmark: >-
  P2 (experiment 270's six-by-three forecast-and-spread-binned plant), XGBoost with past weather and
  the era flag (experiment 275's x2, stored), and the manual_heuristic and climatology floors of
  experiment 224
row_set_id: >-
  Experiment 224's scored-year rows at 12 to 36 hour leads that every arm holds (daylight
  half-hours, sun more than 5 degrees above the horizon, farms A to F, 2025-09 to 2026-08)
metric: >-
  Quantile score in points of capacity, mean absolute error (MAE) of the median, pinball loss and
  coverage at 0.01, 0.05, 0.95, and 0.99, per farm, arm, and setting
code: ml-experiments/00287-metered-farm-spread-binned-plant-heads/
---

**Plan (before any fit).** National Grid Electricity Distribution (NGED) meters six solar farms,
labelled A to F. [Experiment 270](../00270-metered-farm-xgboost-vs-plant-diagnostic/result.md)
found that a plant whose forecast band is binned by its forecast and the ensemble spread (P2)
matches XGBoost's gain over the four-bin plant, and that an additive model of the residuals scores
lowest. [Experiment 257](../00257-metered-farm-fair-plant-arm/result.md) found that a linear fit
recovers about 90% of a neural head's gain, and [experiment
275](../00275-metered-farm-long-history-neural-head/result.md) that years of past weather add little
for the neural head. This experiment asks which head gives the best band on top of P2.

## Arms

- **P2:** control, from experiment 270's rule (6 forecast bins by 3 spread bins, 11 residual
  quantiles per cell from the tuning months). Also XGBoost with past weather and the era flag
  (experiment 275's x2, read from its stored predictions), `manual_heuristic`, and `climatology`.
- **GAM:** depth-1 boosted trees (an additive model) on experiment 270's rung-six inputs, trained
  with the quantile loss and P2's 11 quantiles as base margin, 400 rounds, rate 0.1.
- **NN:** a two-layer network whose input is the P2 forecast, spread, lagged power, clear-sky
  index, and lead. Its output is 11 quantiles built as P2's lowest quantile plus a free shift, then
  cumulative non-negative gaps (softplus of the inverse softplus of P2's gap plus the network
  output), so it is monotone by construction and equals P2 before training. Five seeds.
- **MDN:** the same inputs, a three-component Gaussian mixture on the residual from the plant
  forecast, trained by negative log-likelihood, read at the 11 quantiles, clamped to [0, ceiling].
  Five seeds.
- **Blend:** per quantile level, simplex weights over GAM, NN (seed mean), and XGBoost, fitted
  rolling-origin as in [experiment 260](../00260-metered-farm-quantile-combination/result.md).
- **Controls (own history):** NN with spread shuffled among rows, NN with lagged power shuffled, and
  NN without the P2 forecast column as an input (P2 still supplies the base quantiles).

All arms train on ensemble rows only (the five-seed experiment 275 shows past weather adds 0.47% to
the neural head). Settings: own history (the farm's earlier months) and zero history (the other five
farms). Weather at lag times is the ECMWF ensemble 00 UTC run (12 to 36 h leads) for forecast
inputs, and the farm's own power one or two days earlier for lagged power; both exist at forecast
time. Schedule: one fit on the tuning months (before 2025-09), no refits, 60 epochs, the last two
tuning months select the epoch. The cadence could change absolute scores, not the arm ranking
questions asked here.

## Pre-registered contrasts

- C1: NN minus GAM (prediction: not at least 1.5% lower).
- C2: MDN minus GAM.
- C3: blend minus the best single arm of GAM, NN, and XGBoost (chosen by point estimate).
- C4: GAM minus P2.
