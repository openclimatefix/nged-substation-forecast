---
id: '00323'
slug: primary-demand-default-3-flexible-sunshine-response-in-the-convex-fit
title: >-
  Does a flexible sunshine response in the convex fit improve Primary demand research default 3 at
  0.5 to 240 hours?
theme: production-reference-and-best-guess
also_relevant_to:
  - weather-products-and-blending
status: planned
kind: change
size: >-
  medium. Stored: private files under data/ml-experiments/00323-*/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one is defensible (F1, F2, F3 are the arms
  fixed by the question). Callers: this folder's scripts, which import the folders of experiments
  184, 202, 220, 251, 317, 319, 320, and 322 and call reviewed `studies.*` modules without changing
  them. No change under `packages/`.
hypothesis: >-
  Primary demand research default 3 trails the depth-1 XGBoost refit on April-to-June daylight
  half-hours because its solar term (the pinned capacity times the derated unit shape, with the D5
  clip) is too high on clear spring middays. Experiment 322 grafted the refit's sunshine response
  onto the default and recovered 0.50 of the refit's advantage. A smooth correction in clear-sky
  index and sun elevation, fitted in the same bounded least-squares fit as the solar term, recovers
  a useful part of that gain with no model of other inputs.
prediction: >-
  Written before any score is read; each can fail. (PF1) F1 improves the April-to-June daylight
  half-hours at D5 and D9 together by at least 0.30 points at 0.5 to 168 hours, with a 99.5%
  interval below zero in the scored year and the replicate year's April and June agreeing in sign;
  the pooled six-Primary score over the year improves by at least 0.02 points at 0.5 to 240 hours;
  D10 and D16 move by under 0.02 points; sunny daylight is no worse than +0.05 points; cold winter
  nights and tails stay within their tolerances. If F1 gives under 0.15 at D5 and D9, the graft's
  recovery came from more than a single-input curve. (PF2) F2 improves D9 by at least 0.05 points
  more than F1 does, otherwise F2 is not adopted. (PF3) At 168 to 240 hours F1 improves the sunny
  daylight score by at least 1% beyond the default. (Stage 0) The fitted response differs from the
  physics term mostly at high clear-sky index and high sun, in the same direction at D5 and D9.
decision_rule: >-
  Family rule of research-default.md. F1 and F2 are new fitted steps, so the minimum gain is 0.02
  points at 0.5 to 240 hours. At most one of F1 and F2 is adopted. The no-harm bounds of the page
  hold at 0.5 to 240 hours and at 168 to 240 hours (pooled 0.02, q01 and q99 pinball 0.05, cold
  winter nights 0.05, sunny daylight 0.05, twCRPS 2% relative), and leaving one Primary out does
  not flip the sign. The reproduction gate comes first: the default equals experiment 322's
  default to within 0.01 points at 0.5 to 240 hours (2.727). F3 is kept only if it beats F1 by at
  least 0.02 points.
benchmark: >-
  Primary demand research default 3 (experiment 322's D1). The manual heuristic, climatology, and
  BG2 to 168 hours are references.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles, in points of effective capacity)
  per 24-hour lead band from day 1 to day 10, pooled over six Primaries, scored year 2025-07-02 to
  2026-06-30, rows outside the hand-filtered events, origin 09:00 UTC. Block bootstrap over the 12
  months of the forecast run's initialisation time (14-week blocks for April to June).
---

## Plan

This page is the plan and is rewritten as the write-up when the run finishes.

- **Arms.** Default (Primary demand research default 3). F1: per Primary, a tensor-product linear
  spline of the calibrated clear-sky index (6 knots, 0 to 1.2) and sun elevation (5 knots, 5 to 65
  degrees), 30 columns in points of effective capacity, added to demand in the T-all fit of
  experiment 184 (the model that produces the 51 members' solar responses), unconstrained in sign,
  with a second-difference penalty along both axes and a ridge, both set before any score is read.
  F2: one shape for all six Primaries and a per-Primary scale. F3 (optional): a small monotone
  network, run only if time allows.
- **Where F1 enters.** Each member's response is the F1 fit's forecast with that member's weather
  minus the F1 fit's forecast at the ensemble-mean weather, as in experiment 184, and the level
  difference between the F1 fit and the stored T-all fit at the ensemble-mean weather is added to
  the chain's mean. Experiment 319's x1 mixture, the daylight blend, and experiment 202's rolling
  replay are then rerun with these two inputs, with every factor refitted on completed history.
  This is a deviation from the brief's "same bounded least-squares fit as the other terms": the
  chain's own Kalman-filtered mean is not refitted, so the F1 effect on the level is the T-all
  fit's difference.
- **Stage 0.** The fitted F1 response against the physics term at four elevations, per Primary.
- **Reproduction gate.** Rerunning the replay with the stored responses reproduces experiment 322's
  default.
