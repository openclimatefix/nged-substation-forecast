---
id: '00350'
slug: primary-demand-capacity-free-irradiance-curve
title: >-
  Can a learned curve of irradiance, with no PV capacity estimate, replace the pinned capacity times
  the unit solar shape in Primary demand research default 3?
theme: solar-disaggregation
also_relevant_to:
  - production-reference-and-best-guess
status: planned
kind: change
size: >-
  medium. Stored: private files under data/ml-experiments/00350-*/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one is defensible for the curve (shared or
  per-Primary shape, monotone spline or small neural network), and the plan fixes two. Callers:
  this folder's scripts, which import the folders of experiments 111, 136, 155, and 184 and call
  reviewed `studies.*` modules (`dp_gam`, `dp_pinned_capacity`, `dp_features`, `dp_ens_weather`)
  without changing them. No change under `packages/`.
hypothesis: >-
  The convex model of Primary demand research default 3 subtracts a PV term, a pinned PV capacity
  (from the frozen weekly vintages of experiment 45) times a unit solar shape built from irradiance
  (35 degree tilt, south facing, temperature derated). Experiments 110 and 139 found that no
  fitted or as-of capacity improves on the pin, so the capacity estimate is the part of the stack
  most likely to be wrong. Replacing the pinned capacity and the fixed unit shape with a learned,
  monotone function of irradiance, scaled per Primary by a number the fit learns, keeps the
  forecast as good on sunny daylight half-hours and keeps it as good elsewhere, so the capacity
  estimate is not needed.
prediction: >-
  Written before any score is read; each can fail. (P1, headline) On sunny daylight half-hours
  (sun above 5 degrees and the ensemble-mean shortwave in the top quartile of the Primary's daylight
  half-hours), arm B (shared curve) changes the quantile score by between -0.10 and +0.05 points of
  capacity relative to arm A, at 0.5 to 240 hours and at 24 to 240 hours separately, and its
  month-resampled 95% interval upper bound is below +0.05. (P2) Arm C (per-Primary curve) scores
  within 0.03 points of arm B on sunny daylight half-hours, so a shared shape is not too stiff.
  (P3) Cold winter nights, tails (q01 and q99 pinball), and the 0.5 to 240 hour pooled score each
  change by less than 0.03 points for arms B and C. (P4) Gate: the reproduction of arm A matches the
  stored forecasts of the research default's convex stage within 0.005 points. If B is worse than A
  by more than 0.05 points on sunny daylight half-hours, the capacity estimate carries information
  the curve cannot learn, and the pin stays.
decision_rule: >-
  Family rule of research-default.md, two pre-registered arms (B, C) read at the 1 - 0.05/2 level
  for the headline sunny-daylight row at each of the two lead ranges. The research default's
  stored forecasts are the benchmark. The simplicity case is accepted for an arm when the upper
  bound of its month-resampled 95% interval on sunny daylight half-hours is below +0.05 points at
  both lead ranges and no no-harm bound of the research-default page is broken. The gain case
  needs an interval wholly below zero at both lead ranges. The threshold is set by the cost of
  porting: removing the capacity estimate deletes a pipeline input, so non-inferiority is enough
  and the minimum gain is zero. The reproduction gate (arm A equals the stored convex stage within
  0.005 points) comes first. Dropping one Primary at a time must not flip the sign of the
  headline contrast.
benchmark: >-
  Primary demand research default 3, as stored by experiment 322 (columns `d1_*`, 168 to 240
  hours) and experiments 220 and 251 (below 168 hours), and arm A, the convex stage of the stack
  refitted here with the pinned capacity as stored. The manual heuristic and climatology are
  references.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Change in the quantile score (twice the mean pinball loss over 11 quantiles, in points of
  effective capacity) on sunny daylight half-hours, with a month-resampled 95% interval, at 0.5 to
  240 hours and at 24 to 240 hours. No-harm groups: all half-hours, cold winter nights, q01 and
  q99 pinball, CRPS, and mean absolute error of the median. Scored year 2025-07-02 to 2026-06-30,
  six Primaries, origin 09:00 UTC.
---

## Plan

**Plan only: no score has been read.** The five items of the research-default page follow.

- **Family and version.** Primary demand research default 3.
- **Arms.** Default (stored forecasts); A (the convex stage refitted with the pinned capacity, the
  reproduction gate); B (the PV term replaced by one shared monotone smooth curve of irradiance
  with a learned per-Primary scale, fitted on each fold's training window); C (B with a
  per-Primary curve shape).
- **Tier.** Tier 2: the change acts inside the convex stage. The neural-network head, the
  solar summary, and the heuristic blend are not refitted (see "Scope"), so arms are compared at
  the convex-stage level, with the stored default beside them.
- **Scope.** The stack has eight dependent stages after the convex fit, and the member pass of the
  solar summary bakes in the current solar term, so nothing downstream can be reused. This
  experiment refits the convex fit, the Kalman filter, the offset, recipe R, and the spread table
  only, and says in the write-up that the head, summary, and blend were not refitted.
- **Lead range.** Scored 0.5 to 240 hours and 24 to 240 hours, the maintainer's priority being
  the sunny daylight row. NGED's range is 1 to 10 days.
- **Schedule.** Monthly refits on a 24-month window, as the default.

