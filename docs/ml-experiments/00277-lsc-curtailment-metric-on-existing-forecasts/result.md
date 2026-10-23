---
id: '00277'
slug: lsc-curtailment-metric-on-existing-forecasts
title: >-
  What the curtailment metric says about climatology, the manual heuristic, XGBoost BG2 and the GAM
  stack
theme: curtailment-cost
also_relevant_to:
- lsc-detectors
- production-reference-and-best-guess
status: planned
size: >-
  simple. Stored: private score tables under
  data/ml-experiments/00277-lsc-curtailment-metric-on-existing-forecasts/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: more than one defensible way to define the row
  groups and the node, so both are fixed in this plan. Callers: the unchanged scoring entry point
  studies.node_curtailment.node_curtailment, which this experiment calls and does not change; no
  forecast is built or refitted. Asset graph: not touched.
hypothesis: >-
  The curtailment-cost metric of experiment 268 (studies.node_curtailment, sorted quantiles) ranks
  the four existing forecasts as the other metrics do, and can resolve the differences between
  them well enough to judge the load-shift questions of the decision program.
prediction: >-
  Written before any score is read. At the 0.5 scaled limit, on the six Primaries pooled and
  scored on 12 to 36 hour leads, avoidable cost orders the forecasts climatology > manual
  heuristic > BG2 > GAM stack (worst to best), and the 95% month-resampled interval of
  (climatology minus GAM stack) excludes zero. The interval of (BG2 minus GAM stack) is expected
  to include zero, because experiment 251 found the stack 0.136 points better than BG2 on the
  quantile score with a 95% interval that ended 0.005 below zero. If so, the curtailment part of
  the three questions cannot be answered on this record and steps 4 to 6 fall back on CRPS and
  twCRPS.
decision_rule: >-
  Row groups and nodes are fixed here. (1) Main rows: experiment 251's rows (six Primaries, 09 UTC
  runs, 2025-07-02 to 2026-06-30, outside the hand-filtered load shifts), one seed; the curtailment
  metric scores leads 12 to 36 hours (the metric's default), and MAE, CRPS and twCRPS are reported
  both on all leads and on 12 to 36 hours. (2) Load-shift groups: experiment 163's rows, which
  include the logged transfers and the rows within 72 hours after them; the GAM-stack arm there is
  the unmasked stack K0 (five stored quantiles; q95 is the only level the metric reads). (3) Bulk
  supply points D13 and D17: climatology and the manual heuristic (experiment 48) and the top-down
  and bottom-up forecasts of experiments 204 and 161; no BG2 or experiment 220 stack exists there.
  Metric: experiment 268's metric on the quantile path with sorting on, node-only (no generators),
  limit f times the node's largest flow before 2025-07-02, f = 0.5 pre-registered and 0.75 and 0.9
  as sweep, headline price GBP 2,000 per MWh-equivalent. Because the metric is node-only on a
  Primary's demand, Primary results are relative comparisons between forecasts and state no money
  saved. Intervals resample whole calendar months (12) for the main rows, and whole logged
  transfers for the shift groups. `supported`: the order holds in point estimates and the
  climatology-minus-stack interval excludes zero. `partly supported`: the order holds in point
  estimates but only part of the intervals exclude zero, or the interval of BG2 minus the stack
  includes zero. `not supported`: the order is broken. Detectability: for each metric, the smallest
  paired difference whose month-resampled 95% interval excludes zero, taken as 2.8 times the
  bootstrap standard error of the paired difference (80% power at 5% significance), compared with
  the load-shift effects of experiment 208 (1.50 points after a shift, 0.005 per year). The
  threshold is zero, because this is a measurement of existing forecasts and nothing is ported.
benchmark: >-
  Experiment 251's scores of the same forecasts on the quantile score and tail-weighted CRPS
row_set_id: >-
  fold_a_daily_gap_v1, scored year, experiment 251's six-Primary rows; experiment 163's rows for
  the load-shift groups
metric: >-
  Avoidable curtailment cost (GBP, node-only, scaled limit), MAE of the median and of the mean,
  CRPS (twice the mean pinball loss over 11 quantiles), twCRPS at the 95th and 99th percentile of
  earlier power
code: ml-experiments/00277-lsc-curtailment-metric-on-existing-forecasts/
production_status: not-for-production
adopted_in: none
---

Planned. This is step 3 of the load-shift decision program. It scores the forecasts that already
exist with the curtailment metric of experiment
[268](../00268-lsc-curtailment-metric-v2-member-wise-ens/result.md), and measures how small a
difference the metric can resolve. No forecast is built or refitted. The write-up replaces this
page when the scoring finishes.
