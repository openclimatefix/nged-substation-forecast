---
id: "00356"
slug: wind-default-2-blend-and-monthly-refit-retest
title: >-
  Does blending with Wind farms research default 1, or refitting every month, fix the upper tail of
  Wind farms research default 2?
theme: generator-forecasting
also_relevant_to:
  - calibration-and-uncertainty
  - retraining-and-training-history
status: planned
kind: change
date: 2026-10-27
size: >-
  medium. Stored: per-arm quantiles and refitted heads under
  data/ml-experiments/00356-wind-default-2-blend-and-monthly-refit-retest/ only; the adopted arm, if
  any, is written once with write_predictions. Serving path: not touched (the page says what each
  arm needs). Degradation rule: not touched. Designs: several are defensible (the four arms are
  fixed by the brief; the window and the lead bands are fixed here before any score). Callers: this
  folder's scripts, which import experiments 230, 259, 303, 310, 326, 336, 346, 347, and 348. One
  Sonnet code review and one Sonnet prose review, no Opus.
hypothesis: >-
  Wind farms research default 2 (experiment 347's arm y2) gains 0.234 points over Wind farms research
  default 1 at true days 1 to 10 but puts 2.28% of observations above q99. Either a per-quantile,
  per-lead-band blend of the two defaults (u2), experiment 348's arm z5 (u1), a monthly offset layer
  on default 2 (u4), or a monthly retrained head (u3) lowers the 0.99 pinball loss and the share above
  q99 without losing the gain.
prediction: >-
  Written before any score is read. u2 lowers the 0.99 pinball loss against default 2 by at least 10%
  and puts the share above q99 under 1.6%. u2 gains at least 0.05 points on the headline over true
  days 1 to 10. u1 matches its experiment 348 gain against default 1 (0.147, within 0.05), but its
  gain over default 2 is smaller than u2's. u4 gains under 0.05 and the share above q99 stays above
  2%. u3 gains under 0.05 and has a wider interval than u4.
decision_rule: >-
  The five adoption conditions of the research-default page, read in the order u2, u1, u4, u3, at
  most one adopted, every interval at level 1 - 0.01/4 (month-block bootstrap over the scored
  months). Condition 1: the interval of (arm minus default 2) on the quantile score of the kept rows
  at true days 1 to 10 lies wholly below zero and the point estimate reaches the tier's minimum
  gain (0.02 for u1 and u2, new fitted steps; 0 for u3 and u4, changed settings). Condition 2: the
  one-sided 95% upper bound of the harm to the headline is at most 0.02 and to the 0.01 and 0.99
  pinball losses at most 0.05, against default 2 and, for the cumulative harm budget (2 times the
  tolerance), against default 1. Condition 3: the replicate (Tuesday runs of 2026-07 to the last
  finished run, about 16 runs, fixed in advance, scored once) has the same sign and a gain of at
  least half of the smaller of the main estimate and the minimum gain. If the out-of-time period is
  too short to judge, the arm stays a candidate. Condition 4: the reproduction gate (default 2
  within 0.01 of experiment 347's score on the kept rows), the shuffled-weights placebo, five
  seeds for u3, the leakage check, and no audit trigger unresolved. Condition 5: the page says what
  serving needs.
benchmark: >-
  Wind farms research default 2 (experiment 347's y2, 10.518 points at true days 1 to 10 on the
  kept rows). Beside it, descriptive: Wind farms research default 1 (10.752), the frozen XGBoost
  reference (11.102), climatology (13.860), experiment 346's x1, and experiment 348's z5 against
  default 1.
row_set_id: >-
  fold_a_tuesdays_v1 (53 Tuesday runs, 2025-07 to 2026-06), kept rows by experiment 210's mask
  the headline, all half-hours beside. Replicate: the Tuesday runs of 2026-07 to the last finished
  run (kept rows end at the mask's end, 2026-09-13).
metric: >-
  Quantile score over 11 levels in points of capacity, headline scope true days 1 to 10 (lead 24 to
  240 hours), with days 1 to 2, days 3 to 10, and every lead beside; tolerance 0.02 for the headline
  and 0.05 for the 0.01 and 0.99 pinball losses.
code: ml-experiments/00356-wind-default-2-blend-and-monthly-refit-retest/
---

**Plan only: no score has been read.** The four arms, the controls, and the replicate are fixed in
advance. All arms use only forecasts, actuals, and heads available before each scored run's init
time; every refit happens on the first day of a month, from daily runs whose valid times are at
least 3.5 hours (the telemetry delay) before that day.

## Plan

- **Family and version.** Wind farms research default 2 against Wind farms research default 1 (its
  first version, for the cumulative harm budget). Rows, metric, and tolerances as in the front
  matter. The headline caveat is "scored at true days 1 to 10 (lead 24 to 240 hours)".
- **Arms.** u2: per quantile and lead band (under 24 h, 24 to 72 h, 72 to 240 h, over 240 h), a
  blend weight `w` in steps of 0.1 between default 2's quantile and default 1's, chosen by pinball
  loss on kept daily runs, refitted monthly on an expanding window. u1: experiment 348's z5 exactly
  (code imported). u4: default 2 plus an offset per farm, lead day, and quantile (experiment 259's
  table recipe), refitted monthly on an expanding window. u3: default 2's head retrained every month
  on the expanding window (same architecture and seeds, early stopping on the last 2 months of the
  window), five seeds.
- **Window.** The expanding window for u2, u4, and u3 starts at the head's training cutoff,
  2025-04-16 (u2 and u4), or 2024-04-01 (u3, which refits the head itself). The brief asked u2 and u4
  to start in 2024-04; the hindcast runs before 2025-04-16 are in-sample for default 2's head, so a
  weight or an offset fitted there would measure the head's training fit. This deviation is
  deliberate. The first scored months therefore have a short window (fewer than 500 kept rows falls
  back to weight 0.5 and a zero offset).
- **Tier.** u1 and u2 are new fitted steps (0.02), u4 and u3 changed settings (0). Tier 1 for u1, u2,
  and u4 (stored forecasts); tier 2 for u3.
- **Controls.** A shuffled-weights placebo for u2 (each quantile and lead band takes the weight
  chosen for a different, randomly drawn earlier month); the leakage check printing, for every
  refit, the latest valid time used and the month start; default 2 reproduced within 0.01 of
  experiment 347 (the gate), and u1 reproduced within 0.01 of experiment 348's z5 on the scored
  year; five seeds for u3; each family alone (default 2, default 1, the XGBoost reference) beside
  every blend, and each blend's added value over the better single family with an interval.
- **Replicate.** Tuesday runs from 2026-07 to the last finished run; weights, offsets, and heads use
  only runs finished before each. Scored once. About 16 runs give a wide interval, and no post hoc
  replicate may be used.
- **Sizing (five triggers).** Stored: private arrays only. Serving path: not touched. Degradation
  rule: not touched. More than one defensible design: yes for the window (fixed above). Callers
  not nameable without searching: no, this folder's scripts import the folders listed in the front
  matter.
- **Advice followed.** Experiment 348's advice (blend the plant's quantiles with the default's XGBoost
  quantiles per quantile and lead band; report the share above q99 beside the 0.99 pinball loss;
  treat a base-only head as the reference) is followed. Nothing departs from it.
