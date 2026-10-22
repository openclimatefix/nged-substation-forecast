---
id: '00201'
slug: dp-gam-residual-mlp
theme: differentiable-physics-forecast-engine
also_relevant_to:
- calibration-and-uncertainty
- lagged-power-and-error-correction
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00201-dp-gam-residual-mlp/ only. Serving
  path: not touched. Degradation rule: not touched. Designs: more than one is defensible (network
  size, which inputs, additive or multiplicative output), and the plan pre-registers four input arms
  and one network shape. Callers: this folder's scripts, which copy experiments 136, 149 and 155
  and call the reviewed `studies.*` modules without changing them.
hypothesis: >-
  A small neural network that reads the convex model's own design features, and later the calendar,
  the ensemble-mean weather, and lagged power, can learn an interaction the additive convex model
  lacks, and so lowers the quantile score of the best-guess convex forecast (arm D of experiment 155)
  when it adds a correction to the median and rescales the band widths.
prediction: >-
  Written before any score is read. Arm 3 (design features, calendar, weather) or arm 4 (plus lagged
  power and a site embedding) lowers the quantile score over the scored year by at least 0.02
  points against arm D, with the blocked-month interval wholly below zero, and worsens none of the
  special-day groups (cold nights, sunny daylight, Christmas and bank holidays, Economy 7 nights) by
  more than 0.05 points. Arms 1 and 2 gain 0.00 to 0.01. If every arm fails, neural tweaking of the
  convex model leaves the queue.
decision_rule: >-
  Four planned contrasts (arm 1, 2, 3, 4 minus arm D), each read at the 97.5% level of the
  month-blocked paired bootstrap on the ENS-era scored year. The hypothesis is `supported` when the
  best of arms 3 and 4 has its interval wholly below -0.02 points and no special-day group worsens
  by more than 0.05. The threshold of -0.02 is set by cost: a network adds a trained model, a GPU
  step, and stored weights to the serving path. `partly supported` when the interval is wholly
  below zero but a condition fails. `refuted` when wholly above zero. `inconclusive` otherwise. The
  known-answer gate runs first: the network must recover at least 50% of an injected interaction on
  held-out months and change the score on uninjected rows by under 0.02 points. A failed gate caps
  the verdict at `inconclusive` and stops the experiment.
benchmark: >-
  Arm D of experiment 155 (the best-guess convex forecast with the winter night offset and recipe
  R), rebuilt for 2019-09 onward at six substations
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles) in points of effective capacity,
  over leads 0.5 to 168 hours, at day D+1, by lead band, on cold nights, sunny daylight, Christmas
  and bank holidays, and Economy 7 nights; coverage and pinball loss at q01, q05, q95, q99
code: ml-experiments/00201-dp-gam-residual-mlp/
production_status: not-for-production
adopted_in: none
---

# Experiment 201: a small neural network on top of the convex forecast

**Plan only; results to follow.** This experiment asks whether a small neural network, trained on
the forecasts of the convex demand model (GAM-K, arm D of experiment 155), can find an interaction
that the convex model's additive terms cannot express. It adds to
[experiment 142](../00142-differentiable-physics-sequence-model-on-gam-k-residuals/result.md),
whose network saw only residual history and about 1,100 rows.

## Stage 1: out-of-sample base forecasts (shared with the pooled correction head)

- Arm D at the six stored substations (D5, D9, D10, D11, D14, D16) exists only for 2024-07 to
  2026-06. The network needs more rows, so the experiment rebuilds arm D's recipe (monthly refit,
  24-month window, pinned solar capacity, T-all, winter night offset, recipe R) for 2019-09 onward.
- **The ensemble weather exists only from 2024-04.** Before that the rebuild uses past weather
  (ERA5 and CAMS, the analysis proxy), which has no forecast error. Each row carries an `era`
  column (`past_weather` or `ens`). The headline is scored on the `ens` era only. Early-era rows
  have smaller weather error than live rows, so a network trained on them may under-correct.
- The scored year of the rebuild must reproduce experiment 155's arm D; any gap is reported.
- Cost: up to 8 CPU cores for a few hours, then one GPU share for the network fits.
- The output is in the `dp_preds` layout, one parquet per (site, month), written as each month
  finishes, so the pooled correction head can start on partial output.

## Stage 2: the network

Arms 1 to 4 as pre-registered in the hypothesis; zero-initialised output layer, weight decay,
early stopping on held-out months, pinball loss over the 11 quantiles. Known-answer gate first.
