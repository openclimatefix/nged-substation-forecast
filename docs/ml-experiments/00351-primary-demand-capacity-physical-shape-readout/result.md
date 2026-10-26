---
id: "00351"
slug: primary-demand-capacity-physical-shape-readout
title: >-
  Does reading the solar capacity out in the stack's own physical unit shape, monthly and as-of each
  origin, improve Primary demand research default 3 on sunny daylight half-hours?
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - production-reference-and-best-guess
  - solar-disaggregation
status: planned
date: 2026-10-26
code: ml-experiments/00351-primary-demand-capacity-physical-shape-readout/
kind: change
size: >-
  medium. Stored: private files under data/ml-experiments/00351-*/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one defensible read-out (with or without the
  injection divisor), and the plan fixes two. Callers: this folder's scripts, copied from
  experiment 349 and importing the folder of experiment 269 only for the arm B table; reviewed
  `studies.*` modules are called without changes. No change under `packages/`.
hypothesis: >-
  Experiment 349 read the solar capacity with the empirical plant shape of six metered farms
  (experiment 269) but multiplied it by the stack's physical unit shape, a mismatch its page lists
  as a lab shortcut. Reading the capacity out in the physical unit shape the stack itself uses, with
  the same monthly as-of refit and its own injection divisor, removes the mismatch. If the mismatch
  explained experiment 349's null result, the sunny-half-hour score improves.
prediction: >-
  Written before any fit. Arm P (physical-shape read-out, divided by its own divisor) minus arm A
  (pinned capacity) on the sunniest third of daylight half-hours is between -0.03 and +0.05 points
  pooled over six Primaries, with a point estimate that does not beat the pin by 0.05 points. Arm P
  minus arm B (experiment 349's read-out) is below -0.03 only if the shape mismatch mattered, and
  the plan predicts it does not: P minus B lies within +/-0.03. The reasons are experiment 139 (its
  physical-shape as-of capacity scored 0.110 points worse than the pin) and the finding that the
  sunny score is nearly flat in capacity between 0.8 and 1.2 times the pin. Arm R (the same read-out
  without the divisor, in the model's own units) is within 0.03 points of the pin. Cold winter
  nights and the tails move by under 0.02 points.
decision_rule: >-
  Contrast C1 is arm P minus arm A on sunny daylight half-hours (the sunniest third of each
  Primary's daylight half-hours by the unit shape), pooled over six Primaries, at 0.5 to 168 hours
  and at 24 to 168 hours, with a month-resampled 95% interval. C2 is arm P minus arm B on the same
  rows. C3 is arm R minus arm A. The hypothesis is `supported` when C1's interval lies wholly below
  zero and C2's point estimate is negative. The change is worth porting only at a gain of at least
  0.05 points on sunny daylight half-hours, because it adds a capacity fit per month and an
  injection divisor to a pipeline that pins one number per Primary. It is `small gain` when C1's
  interval lies below zero with a point estimate under 0.05 points, `refuted` when it lies wholly
  above zero, and `inconclusive` otherwise. A harm above 0.02 points on cold winter nights, all
  half-hours, or the tails caps the verdict at `partly supported`. Arm P2 (the read-out on the
  stack's own residual solar signal) is skipped, because that signal exists only after the later
  stages of the stack are refitted, and this experiment refits the first stage only.
benchmark: >-
  Primary demand research default 3 (arm A), tier 2 change, with the default's code imported from
  the experiment folders that hold it. Experiment 349's arm B is the reference row.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles, in points of effective capacity) at
  the six Primaries D5, D9, D10, D11, D14, and D16, scored year 2025-07-02 to 2026-06-30, rows
  outside the hand-filtered events, origin 09:00 UTC. Headline: sunny daylight half-hours, then all
  half-hours, daylight, night, cold winter nights, and the tails. The mean absolute error of the
  mean forecast beside each. Replicate year 2024-07 to 2025-06.
---

## Plan

1. **Arms.** A is the pinned capacity as stored. B is experiment 349's arm B as stored (the
   empirical shape read-out, the stack's physical unit shape applied). P is the capacity read out
   monthly in the physical unit shape, fitted on rows before each origin and divided by the median
   recovery of five injected farms read the same way. R is P without the divisor, which is the
   physical-shape read-out in the model's own units. Arm P2 is skipped (see the decision rule).
2. **Capacity tables.** Compare P with B and the pin per Primary and month, and list the
   Primary-months where P differs from the pin or from B by more than 10%.
3. **Scoring.** Row set `fold_a_daily_gap_v1`, six Primaries, scored year 2025-07-02 to 2026-06-30,
   0.5 to 168 hours and 24 to 168 hours. Headline on sunny daylight half-hours with month-resampled
   95% intervals; all half-hours, cold winter nights, and the tails as no-harm groups. Replicate
   year 2024-25.
4. **Reproduction gate.** Arms A and B, refitted by this folder's copy of the code, reproduce the
   stored scores of experiment 349 (largest gap in q01, q50, and q99 stated).
5. **Prediction.** As in the front matter.
