---
id: '00337'
slug: primary-demand-standard-architecture-168-to-240-hours
title: >-
  Does the standard forecasting architecture (a physics core, a learned input adapter, and a
  mixture-density output head) match Primary demand research default 3 at 168 to 240 hours?
theme: production-reference-and-best-guess
also_relevant_to:
  - calibration-and-uncertainty
status: planned
kind: change
size: >-
  medium. Stored: private files under data/ml-experiments/00337-*/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one is defensible, but the arms are fixed by
  experiment 330. Callers: this folder's scripts, which import the folders of experiments 201, 278,
  311, 317, 330, and 335 and read stored files of experiments 111, 201, 317, and 335. No change
  under `packages/`.
hypothesis: >-
  Experiment 330 found that the standard architecture (STD: the frozen convex stack's mean, a
  two-layer input adapter, and a three-component mixture-density output) scores 0.053 points better
  than Primary demand research default 3 at 0.5 to 168 hours, but National Grid Electricity
  Distribution (NGED) needs 1 to 10 days. At 168 to 240 hours STD is again within 0.03 points of the
  default, so demand can share the standard architecture across the full range.
prediction: >-
  Written before any score is read; each can fail. (P1) Without D10, STD is within 0.03 points of
  Primary demand research default 3 at 168 to 240 hours (a difference of at most +0.03; experiment
  330 gave -0.009 without D10 at 0.5 to 168 hours). (P2) STD-A (the same network with a median-only
  output and the default's band construction) differs from the default by under 0.02 points. (P3)
  The gain of STD with D10 persists: STD is better than the default with D10 included, because 88%
  of the D10 gain in experiment 316 was narrowing of the night band, which a mixture output learns.
  (P4) Network seed noise is 0.01 to 0.02 points, so every contrast under 0.03 points is read on six
  seeds averaged. Stop rule: the reproduction gate fails (the default must score 3.077 points at 168
  to 240 hours), or STD is more than 0.10 points worse than the default at the first full run.
decision_rule: >-
  Primary demand research default 3 is the default, and "default" and "default plus x" are the two
  arms of the research-default rule, which owns the metric and tolerances. The experiment states the
  gap of STD to the default in absolute and relative terms, with and without D10, and does not
  decide adoption. Intervals are month-block, at the 1 - 0.01/2 level for the two planned contrasts
  (STD, STD-A against the default). STD is a candidate for the standard architecture at 168 to 240
  hours if its gap is at most +0.03 points with every no-harm tolerance held.
benchmark: >-
  Primary demand research default 3 at 168 to 240 hours (experiment 335's stored forecasts), the
  manual heuristic, and climatology. BG2 has no forecasts past 168 hours.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score over 11 levels (twice the mean pinball loss), in points of effective capacity;
  twCRPS at the 95th percentile; pinball loss and coverage at q01, q05, q95, q99.
code: ml-experiments/00337-primary-demand-standard-architecture-168-to-240-hours/
date: 2026-10-24
---

**Plan only; results follow.** This extends [experiment 330](../00330-primary-demand-standard-architecture-arm/result.md)
from 0.5 to 168 hours to 168 to 240 hours, with the same arms and decision rule.

## The five plan items

1. **Question.** Does the standard architecture (STD) score within 0.03 points of Primary demand
   research default 3 at 168 to 240 hours, with and without D10?
2. **Prediction that can fail.** P1 to P4 in the front matter.
3. **Rows and period.** Six Primaries, scored year 2025-07-02 to 2026-06-30, origin 09:00 UTC, rows
   outside hand-filtered events, valid times 168 to 240 hours after the origin. Rows come from
   experiment 317's stored chain (the ensemble era only), so the training pool grows from 12 months
   at the first refit to 24 months at the last.
4. **Metric.** As in experiment 330: the quantile score, the research-default page's protected
   groups and tolerances, tail calibration at q01, q05, q95, and q99, and month-block intervals.
5. **Stop rule.** The reproduction gate fails, or STD is more than 0.10 points worse than the
   default at the first full run.

## Design

- **Arms.** STD and STD-A as in experiment 330, with experiment 330's network code imported. Monthly
  refits on the P-roll schedule, six seeds averaged.
- **Core and inputs.** The core is arm D's mean from the chain (identical to the stored arm D below
  168 hours). The inputs are experiment 278's arm 2 columns. The level comes from the same run and
  hour band at 168 hours or less, the anomaly is zero past 168 hours (mean absolute value 0.009
  points at 144 to 168 hours), and the weather is the 51-member mean of experiment 111's store.
- **Training rows.** Only rows at 168 to 240 hours, from the chain's ensemble-era periods, ending
  before each refit.
- **Differences from experiment 330.** The replicate-year check is dropped, because the replicate
  months are in the training pool.
