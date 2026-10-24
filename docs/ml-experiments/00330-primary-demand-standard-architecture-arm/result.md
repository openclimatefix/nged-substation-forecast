---
id: '00330'
slug: primary-demand-standard-architecture-arm
title: >-
  Does the standard forecasting architecture (a physics core, a learned input adapter, and a
  mixture-density output head) match Primary demand research default 3?
theme: production-reference-and-best-guess
also_relevant_to:
  - calibration-and-uncertainty
status: planned
kind: change
size: >-
  medium. Stored: private files under data/ml-experiments/00330-*/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one is defensible, but the two arms are fixed
  by the question. Callers: this folder's scripts, which import the folders of experiments 278,
  311, and 314 and read the stored files of experiments 251, 311, 314, and 322. No change under
  `packages/`.
hypothesis: >-
  The maintainer prefers one standard architecture (a physics core, an optional neural input
  adapter, an optional mixture-density output head) across demand, wind, and solar forecasts, unless
  a different architecture gains strongly. Primary demand has the physics core already (the convex
  stack with a Kalman filter). One small network, trained jointly to correct the median and to give
  a mixture-density band, fed through a learned two-layer adapter of the ensemble summaries, scores
  within 0.03 points of Primary demand research default 3, so demand can share the standard
  architecture at little or no loss.
prediction: >-
  Written before any score is read; each can fail. (P1) The standard architecture (STD) is within
  0.03 points pooled of Primary demand research default 3 (a gap of at most +0.03). (P2) STD breaks
  none of the research-default page's no-harm tolerances (pooled 0.02, q01 and q99 pinball 0.05,
  cold winter nights 0.05, sunny daylight 0.05, twCRPS 2% relative); sunny daylight is read first.
  (P3) STD and STD-A (STD with the default's band construction in place of the mixture output)
  differ by under 0.01 points; a STD-A better than STD by 0.02 or more means the mixture output
  hurts the tails. (P4) Network seed noise is 0.01 to 0.02 points, so every contrast under 0.03
  points is read on at least six seed sets averaged. Stop rule: the reproduction gate fails, or STD
  is more than 0.10 points worse than the default at the first full run.
decision_rule: >-
  Primary demand research default 3 is the default, and "default" and "default plus x" are the two
  arms of the research-default rule; the rule owns the metric and tolerances. This experiment
  states the gap of STD to the default, to STD-A, and to the gated mixture head of experiment 314
  (arm G), by 24-hour lead band, and does not decide adoption. Read at the 1 - 0.01/2 level (two
  contrasts), month-block resampling. STD is a candidate for demand's standard architecture if its
  gap is at most +0.03 points with every tolerance held; it becomes Primary demand research default
  4 only if it is better than the default under the family's five conditions.
benchmark: >-
  Primary demand research default 3 (stored forecasts of experiments 251 below 168 hours and 322
  above), experiment 314's arm G, BG2 to 168 hours, the manual heuristic, and climatology.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score over 11 levels (twice the mean pinball loss), in points of effective capacity;
  twCRPS at the 95th percentile; pinball loss and coverage at q01, q05, q95, q99.
code: ml-experiments/00330-primary-demand-standard-architecture-arm/
---

## Plan

Primary demand research default 3 is our research demand forecast for six Primary substations. This
experiment asks whether a standard architecture shared across demand, wind, and solar forecasts
costs demand any accuracy. The architecture is a physics core (here the frozen convex stack), a
learned input adapter, and a mixture-density output head. The mixture head follows
[Bishop (1994)](https://publications.aston.ac.uk/id/eprint/373/).

### The five plan items

1. **Question.** Does the standard architecture (STD) score within 0.03 points of Primary demand
   research default 3 at 0.5 to 240 hours, and what is its gap at each lead band?
2. **Prediction that can fail.** P1 to P4 in the front matter.
3. **Rows and period.** `fold_a_daily_gap_v1`, six Primaries, scored year 2025-07-02 to 2026-06-30,
   replicate year 2024-25, origin 09:00 UTC, rows outside hand-filtered events.
4. **Metric.** The quantile score over 11 levels in points of effective capacity, with the
   research-default page's protected groups and tolerances and 12-month block bootstrap intervals,
   and tail calibration at q01, q05, q95, and q99.
5. **Stop rule.** The reproduction gate fails (the default equals experiment 322's default 3 to
   within 0.01 points), or STD is more than 0.10 points worse than the default at the first full
   run.

### Arms

- **STD:** one network trained jointly on a median correction and a three-component mixture on the
  residual of the frozen core's mean, replacing the median-correction head, the spread table, the
  night offset's width part, and the solar standard-deviation band step. A two-layer adapter maps
  the ensemble summaries (temperature and solar response, mean and standard deviation) into the
  trunk; the other inputs are those of experiments 311 and 220. Calibration is the as-of calibration
  of experiment 164 only.
- **STD-A:** STD's median correction and adapter, with the default's band construction.
- **References:** the default, arm G of experiment 314, BG2 to 168 hours, the manual heuristic, and
  climatology.

### Stages

- **Stage 1 (0.5 to 168 hours).** Experiment 311's rows, 24-month pool, and monthly refits (the
  schedule is P-roll, a contrast of 0.03 points is read on at least six seeds averaged). The core is
  arm D, which is the default's core below 168 hours.
- **Stage 2 (168 to 240 hours).** The stored feature rows end at 168 hours, so the core above 168
  hours is the default's stored median and width (experiment 322), and the pool is the stored
  168-to-240-hour rows that end before each refit (an expanding pool that starts with the six
  replicate months). The pooled 0.5-to-240-hour figure is reported only if stage 2 runs; otherwise
  every headline says "0.5 to 168 hours" and names the missing range as the result's main caveat.

### Checks to run

Median minus mean forecast per site, the mean-forecast error beside each quantile-score contrast,
leaving one Primary out (D10 carried experiment 311's gain), and the seven audit triggers.
