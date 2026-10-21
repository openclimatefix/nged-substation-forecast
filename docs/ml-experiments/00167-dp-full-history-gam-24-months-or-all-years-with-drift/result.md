---
id: '00167'
slug: dp-full-history-gam-24-months-or-all-years-with-drift
part: n/a
title: >-
  Does the GAM with explicit PV gain from training on every year of history back to late 2019, with
  or without a drift term, compared with the last 24 months?
theme: differentiable-physics-long-history-and-drift
also_relevant_to:
- retraining-and-training-history
- differentiable-physics-embedded-generation
status: planned
size: >-
  medium. Stored: private files under
  data/ml-experiments/00167-dp-full-history-gam-24-months-or-all-years-with-drift/ only. Serving
  path: not touched. Degradation rule: not touched. Designs: more than one defensible (the history
  rule for dropping a site, the spread, the treatment of D10's jumps), and this plan pre-registers
  one of each. Callers: this folder's scripts, which copy experiment 107's refit layout and
  experiment 136's T-all design and call the reviewed `studies.dp_*` modules without changing them.
hypothesis: >-
  With the PV capacity pinned to the as-of table (explicit PV) and the heating split by day type
  (T-all), the convex demand model (the GAM) trained on every year of history back to late 2019
  with a drift term scores better than the same model trained on the last 24 months, when both are
  refitted at 12 quarterly origins on past weather (ERA5 and CAMS). The mechanism is that explicit
  PV removes the changing solar capacity from the target, so the old years teach the heating and
  calendar terms without the capacity growth that made the all-years fits of experiments 104 and
  107 unstable, and the drift term absorbs the remaining slow change in the demand level.
prediction: >-
  Written before any fit. C1 (all years with drift, minus 24 months) is -0.10 to +0.05 points
  pooled, with 5 to 9 of 12 origins below zero, so the 0.05-point rule is likely to fail. C2 (all
  years without drift, minus 24 months) is 0.00 to +0.30, because experiments 104 and 107 found all
  years without drift worse than a shorter window at 8 of 12 origins. C3 (all years with drift,
  minus all years without) is -0.15 to -0.40, as experiment 107's C2 (-0.451, three-fifths of it one
  substation, which this experiment masks). Winter-quarter origins favour more years more than
  summer-quarter ones, as experiment 107's C3 did (-0.461 against +0.058).
decision_rule: >-
  The headline contrast C1 is all years with drift minus 24 months, both with T-all and the pinned
  capacity, as the quantile score (twice the mean pinball loss over experiment 87's 11 quantiles)
  in points of effective capacity, on the same rows, outside the hand-filtered switching events,
  pooled over the five decision substations (D5, D10, D11, D14, D16) and the 12 scored origins.
  The interval resamples whole calendar months at the 98.33% level (a Bonferroni correction over
  the three planned contrasts C1 to C3). `supported` needs a pooled C1 of at most -0.05 points, an
  interval wholly below zero, and C1 below zero at 8 or more of the 12 origins (the maintainer's
  rule of 2026-10-21). `partly supported` is an interval wholly below zero with either a size under
  0.05 or fewer than 8 negative origins. `refuted` is an interval wholly above zero. `inconclusive`
  is everything else. The 0.05 threshold is not zero because all years triples the rows of every
  refit and adds a drift term to fit, which a 24-month window does not need. Winter-quarter
  origins (cuts on 1 October and 1 January) are reported separately and never change the verdict.
  A leave-one-site-out table is part of the verdict check: a verdict that changes when one site is
  left out is capped at `inconclusive`.
benchmark: >-
  The 24-month T-all arm (m24) on the same rows; absolute skill of climatology and manual_heuristic
  as stored by experiment 107 (same origins and sites) for the reference row
row_set_id: >-
  dp_mw_v1 (experiment 103); MW sites; 12 quarterly origins 2023-07-01 to 2026-04-01, each scored on
  the following 3 months; a site-origin pair is scored only if the site has 24 months of usable rows
  before the origin
metric: >-
  Quantile score in points of effective capacity, on rows outside the hand-filtered switching
  events; tail coverage and pinball loss at q01, q05, q95, and q99
code: ml-experiments/00167-dp-full-history-gam-24-months-or-all-years-with-drift/
---

*Planned: the write-up replaces this page when the fits have been scored.*

## Question

**We ask whether the demand model can now use every year of history.** Experiments 104, 106, and 107
gave answers of changing sign for the last-12-months against all-years contrast. One suspected
cause is the solar capacity that grew over those years and that the model had to fit. Experiment
139 kept the pinned capacity, so the solar term is explicit in every arm here.

## Arms

All arms are the GAM of `studies.dp_gam` with experiment 136's T-all heating split, the PV capacity
pinned to experiment 45's real-time vintage, experiment 104's tuned penalties, and past weather
only (ERA5 and CAMS, no ensemble weather). Each arm is refitted at 16 quarterly cuts (2022-07-01 to
2026-04-01), frozen from its cut, and scored on the next 3 months; the 4 earliest cuts only feed the
spread pool of the first scored origins.

| Arm | Training window | Drift |
|---|---|---|
| m24 | the last 24 months | none |
| all | every year from the site's first usable row | none |
| all_d1 | every year | D1 |
| m24_d1 (exploratory) | the last 24 months | D1 |

## Row set and the history rule

- **Sites.** The five decision sites of experiment 107 (D5, D10, D11, D14, D16), plus D9 described
  beside them.
- **History rule.** A site is scored at an origin only if it has at least 24 months of usable rows
  before the origin, so the 24-month window is full and the arms differ in what they train on.
  A site with at least 12 months is still fitted at earlier cuts, only to build the spread pool.
  D9 starts on 2023-08-23, so it is scored at the three origins from 2025-10-01.
  Experiment 103 found that the first months of 13 of 16 Primary substations are cut, which the
  cache already applies.
- **D10.** Its two unlabelled-jump windows (2024-11-08 to 2024-12-01 and 2025-11-17 to 2025-12-14)
  are removed from training and scoring in every arm, because experiment 107 found they drove the
  sign of its history contrasts.
- **Spread.** Experiment 107's baseline spread (S0): each arm's own out-of-sample residuals of the
  four previous scored windows, per hour band and calendar quarter, centred.

## Contrasts

- **C1 (headline):** all_d1 minus m24.
- **C2:** all minus m24.
- **C3:** all_d1 minus all.
- **Exploratory:** m24_d1 minus m24 (drift on a short window), and the same contrasts without the
  D10 mask.

## Checks planned

Median minus mean forecast per site; the mean-absolute-error contrast beside every quantile-score
contrast; leave-one-site-out; per-origin intervals; tail coverage at q01, q05, q95, and q99.
The curtailment-cost saving is not scored, because past weather is not available at forecast time
and the experiment ranks training windows, not forecasts.

## Advice followed

From experiment 107: report every contrast per origin and split winter and summer origins; keep the
two day-of-year harmonics; do not judge a window on one site or origin year (the leave-one-site-out
table). From experiment 104: all years without drift is the arm to doubt.
