---
id: '00325'
slug: primary-demand-default-3-sunshine-response-refit-and-restriction
title: >-
  Does refitting the stack around a sunshine response, and restricting the response to sunny
  half-hours, improve Primary demand research default 3 at 0.5 to 240 hours?
theme: production-reference-and-best-guess
also_relevant_to:
  - weather-products-and-blending
status: planned
kind: change
size: >-
  medium. Stored: private files under data/ml-experiments/00325-*/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one is defensible (the arms are fixed by the
  brief). Callers: this folder's scripts, which import the folders of experiments 136, 184, 201,
  202, 220, 251, 317, 319, 320, 322, and 323 and call reviewed `studies.*` modules without
  changing them. No change under `packages/`.
hypothesis: >-
  Experiment 323's sunshine response (F1) harmed all daylight half-hours because the Kalman
  filter's level, the winter night offset, and the median correction were estimated on the errors
  of a fit without a response and so counted the same error twice (double counting), or because the
  response acts too widely, at forecast clear-sky index where the damped ensemble-mean forecast
  index mis-applies it. Refitting those corrections on the response's own errors (F1r) removes the
  first cause, and restricting the response to clear-sky index above 0.6 and sun elevation above 15
  degrees (F1s) removes the second.
prediction: >-
  Written before any score is read; each can fail. (PR1, F1r) The all-daylight harm at 0.5 to 168
  hours falls from +0.031 to +0.015 or less, at least 70% of the sunny-daylight gain is kept (at
  least 0.064 points), and the pooled six-Primary score at 0.5 to 240 hours is at most 0 (from
  +0.011); a harm above +0.025 means the level shift was not double counting. (PR2, F1s) The
  pooled score improves by at least 0.02 points at 0.5 to 240 hours with a 99.5% interval below
  zero; sunny daylight improves by at least 0.08 points; all daylight at 0.5 to 168 hours is no
  worse than +0.005; the April-to-June gain at D5 and D9 at 0.5 to 168 hours is at least 0.10
  points with the replicate year's April and June agreeing in sign; sunny daylight at 168 to 240
  hours improves by at least 1.5%. (PR3, F1sel) D10 and D16 stay within 0.01 points of the default
  and at least 80% of F1s's gain at D5, D9, D11, and D14 is kept; a Primary whose switch flips
  more than 3 times in a year means F1s is reported without the switch. (Stage 0) At least 60% of
  F1's all-daylight harm sits at ensemble clear-sky index 0.3 to 0.9 and its sunny gain above 0.9;
  otherwise F1s is dropped and F1r alone runs.
decision_rule: >-
  Family rule of research-default.md. The response is a new fitted step, so the minimum gain is
  0.02 points at 0.5 to 240 hours. At most one of F1r, F1s, and F1sel is adopted, and a pass makes
  Primary demand research default 4. The no-harm bounds (pooled 0.02, q01 and q99 pinball 0.05,
  cold winter nights 0.05, sunny daylight 0.05, twCRPS 2% relative) hold at 0.5 to 240 hours and at
  168 to 240 hours, and leaving one Primary out does not flip the sign. The reproduction gate
  comes first: the rebuilt default equals experiment 322's default to within 0.01 points at 0.5 to
  240 hours (2.727).
benchmark: >-
  Primary demand research default 3 (experiment 322's D1). BG2 to 168 hours, the manual heuristic,
  and climatology are references.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles, in points of effective capacity)
  per 24-hour lead band from day 1 to day 10, pooled over six Primaries, scored year 2025-07-02 to
  2026-06-30, rows outside the hand-filtered events, origin 09:00 UTC. Block bootstrap over the 12
  months of the forecast run's initialisation time (14-week blocks for April to June).
---

## Plan

This page is the plan; the write-up replaces it when the run finishes.

- **Question.** Experiment 323 added a smooth sunshine response to the convex fit and left the
  Kalman filter, the winter night offset, recipe R, the residual-spread table, and the median
  correction as they were. This experiment refits all of them on the response's own errors (F1r),
  then restricts the response (F1s), then switches it per Primary as of each month (F1sel).
- **Arms.** Default: Primary demand research default 3. F1 as run (reference, experiment 323). F1r:
  the unrestricted response with the chain rebuilt. F1s: the restricted response (response weight
  rising linearly from 0 at forecast clear-sky index 0.6 to 1 at 0.7, and from 0 at sun elevation
  15 degrees to 1 at 20 degrees, fixed before any score), chain rebuilt. F1sel: F1s per Primary
  only where it beat the default by at least 0.02 points on sunny daylight over the previous 12
  months of origins. F2s only if F1s passes.
- **Rebuilt chain.** For each variant the convex fit with the response is refitted at every monthly
  cut (2019-09 to 2026-06), the Kalman filter runs on the variant's residuals against the analysis
  proxy, the winter night offset, recipe R, and the spread table are refitted on the variant's
  forecasts, the 51 members' responses come from the variant's fit, and experiment 201's network
  (arm 2, three seeds) is retrained on the variant's past-weather and ensemble-era rows to give the
  median correction. A control, `v0r`, runs the same code with no response and must reproduce the
  stored default (the gate).
- **Row set.** Experiment 322's: six Primaries, 798,116 rows, origin 09:00 UTC.
- **Retraining schedule.** Monthly refits of the convex fit, as in experiments 184 and 323; the
  contrasts could not depend on the schedule beyond what experiment 323 shows.
- **Size (five triggers).** Stores private forecasts and fits only; not on the serving path; no
  degradation rule; more than one defensible design but the arms are fixed by the brief; callers are
  this folder's scripts. Medium. Sonnet plan, code, and prose reviews; no Opus.
