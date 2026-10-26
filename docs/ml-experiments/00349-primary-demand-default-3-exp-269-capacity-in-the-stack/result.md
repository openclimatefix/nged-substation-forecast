---
id: "00349"
slug: primary-demand-default-3-exp-269-capacity-in-the-stack
title: >-
  Does replacing the pinned solar capacity with the experiment 269 estimate improve Primary demand
  research default 3 on sunny daylight half-hours?
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - production-reference-and-best-guess
  - solar-disaggregation
status: planned
kind: change
size: >-
  medium. Stored: private files under data/ml-experiments/00349-*/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one way to read the estimate in as-of fashion,
  and the brief fixes three arms. Callers: this folder's scripts, which import the folders of
  experiments 139, 129, 269, and (if stage 2 runs) 136, 155, 317, 319, and 322, and call reviewed
  `studies.*` modules without changing them. No change under `packages/`.
hypothesis: >-
  Primary demand research default 3 subtracts a solar capacity pinned to experiment 45's `frozen`
  vintage (a hindsight growth prior) times a unit solar shape. Replacing that capacity with the
  experiment 269 estimate (the mean plant response of six metered farms, read at the Primary's
  satellite irradiance, fitted from rows before each fold's origin) lowers the quantile score on
  sunny daylight half-hours, because the experiment 269 read-out recovers 0.89 to 0.97 of an
  injected farm where the physical fleet shape recovers 0.78 to 0.83.
prediction: >-
  Written before any fit. On sunny daylight half-hours at 0.5 to 240 hours, arm B (as-of estimate)
  scores between 0.03 points better and 0.03 points worse than the default, with a point estimate
  that does not beat the default by 0.05 points. The reason is experiment 139: its as-of capacity
  from the physical shape scored 0.110 points worse than the pin on the sunniest third (interval
  0.038 to 0.181), and the experiment 269 shape lowers the fitted capacities by only 0.2 to 0.4 MW
  at the five decision Primaries. Arm C (the first as-of value held for the year) is no better than
  arm B by more than 0.02 points, so growth tracking is not what matters. The change on cold winter
  nights and at the tails is within 0.02 points.
decision_rule: >-
  Contrast C1 is arm B minus the default (Primary demand research default 3) on sunny daylight
  half-hours, pooled over the six Primaries, at 0.5 to 240 hours and at 24 to 240 hours reported
  separately, with a month-resampled 95% interval. Contrast C2 is arm C minus arm B on the same
  rows. The hypothesis is `supported` when C1's interval lies wholly below zero. The change is worth
  porting only at a gain of at least 0.05 points on sunny daylight half-hours, because it adds a
  capacity fit and a farm-response table to a pipeline that otherwise pins one number per Primary.
  It is `small gain` when the interval lies below zero but the point estimate is under 0.05 points,
  `refuted` when the interval lies wholly above zero, and `inconclusive` otherwise. The no-harm
  groups are cold winter nights, all daylight, all night, and the tails; a harm above 0.02 points
  on any of them caps the verdict at `partly supported`. Adoption into the research default needs
  the five conditions of `research-default.md`.
benchmark: >-
  Primary demand research default 3 (arm A), tier 2 change with the default's code imported from
  the experiment folders that hold it. Absolute scores of manual_heuristic and climatology are
  reported beside it.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles, in points of effective capacity)
  at the six Primaries D5, D9, D10, D11, D14, and D16, scored year 2025-07-02 to 2026-06-30, rows
  outside the hand-filtered events, origin 09:00 UTC. Headline: sunny daylight half-hours, then all
  half-hours, daylight, night, cold winter nights, and the tails (coverage and pinball loss at q01,
  q05, q95, and q99). MAE of the median forecast beside each. CRPS where the stored forecasts
  allow.
code: ml-experiments/00349-primary-demand-default-3-exp-269-capacity-in-the-stack/
---

**Plan only; no fit has run.** Experiment 269 found that a table learned from six metered solar
farms' output recovers 0.89 to 0.97 of an injected farm's output, against 0.78 to 0.83 for the
physical fleet shape. Primary demand research default 3 does not use either: it subtracts a
capacity pinned to [experiment 45](../00045-pvdis-4-weekly-capacity-asymmetric-l1/result.md)'s
`frozen` vintage. [Experiment
139](../00139-differentiable-physics-pv-as-of-capacity-in-gam-k/result.md) replaced that pin with
an as-of capacity and scored 0.110 points worse on the sunniest third. This experiment repeats the
swap with the better estimate and asks the sunny-half-hour question first.

## The five plan items

- **Family and version.** Primary demand research default 3, six Primaries, 0.5 to 240 hours. NGED
  cares about 1 to 10 days, so the 24 to 240 hour reading is reported beside the whole range.
- **Arms.** (A) the default as stored. (B) the default with the pinned capacity replaced by the
  experiment 269 estimate, fitted at each fold's origin from rows before the origin only (the
  as-of scheme of experiment 139: the latest 1 July origin at or before each monthly cut, the
  capacity divided by the recovery of five injected farms at the same origin). (C) B with the
  estimate held at its first as-of value for the whole scored year.
- **Tier.** Tier 2: the pinned capacity enters the convex model with the Kalman filter (experiment
  136's arm A2, built with experiment 139's adaptor), so the experiment refits that stage and every
  stage that reads its forecasts. Complexity tier: it adds a farm-response table and a capacity
  fit per origin, so the minimum gain is the 0.05 points stated in the decision rule.
- **Pre-registered arms and contrasts.** C1 (B minus A) and C2 (C minus B), with the no-harm groups
  of the decision rule.
- **Controls.** A wiring identity check (arm A refitted with the pin through experiment 139's
  `pin_adaptor` equals the stored forecasts), a known-answer check (the estimate of experiment 269
  at an injected-farm Primary recovers what experiment 269 reported), and a leakage check (every
  fit and divisor stops at its origin).

## Stages

1. **Level 1, the convex stage.** Build the as-of table with experiment 269's pooled empirical
   shape, refit the convex model with the Kalman filter at every monthly cut for arms B and C, and
   score against the pin. This is experiment 139's pipeline with one input changed.
2. **Level 2, the full default.** Rerun the stages that read the convex forecasts (experiments 155,
   160, 164, the neural-network head, the two-number solar summary, and the blend at 168 to 240
   hours) with each arm's forecasts, with each stage's reproduction gate passing for arm A. If a
   stage cannot be rerun inside the compute slot, the page reports level 1 only and says so.

## Checks and caveats

- **Lab shortcut.** The farm-response table is fitted once on the metered farms' history to
  2026-06-30, and only the capacity is as-of. The table is a plant shape, which changes slowly,
  and the page says so.
- **Past weather.** The capacity fit reads satellite irradiance at lag times (the analysis
  proxy), which is available at forecast time with the telemetry delay of experiment 139.
- **Lead range.** The six Primaries are forecast out to 240 hours, which covers NGED's 1 to 10 day
  range.
