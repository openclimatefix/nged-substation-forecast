---
id: '00251'
slug: dp-final-best-guess-assembly-clip-and-coast-against-bg2
title: >-
  Does the full best-guess demand stack, with the D5 solar clip and the gap rule added, beat BG2 and
  the manual heuristic over one scored year, and where does it lose?
theme: production-reference-and-best-guess
also_relevant_to:
- differentiable-physics-forecast-engine
- calibration-and-uncertainty
- curtailment-cost
status: planned
size: >-
  medium. Stored: private files under
  data/ml-experiments/00251-dp-final-best-guess-assembly-clip-and-coast-against-bg2/ only. Serving
  path: not touched. Degradation rule: not touched (experiment 205's coast-and-widen rule is
  reported from experiment 205's stored scores, not changed). Designs: more than one is defensible
  for adding the D5 clip to a stack whose network head was trained without it; this plan
  pre-registers two (with and without the head) and picks the winner by a stated rule. Callers:
  this folder's scripts, which copy experiments 220 and 228 and call reviewed `studies.*` modules
  without changing them. No refits.
hypothesis: >-
  The full best-guess stack of experiment 220 (arm D, the network head, and the two-number solar
  summary) beats BG2, the manual heuristic, climatology, and a weak production-style XGBoost model
  over the scored year of 2025-07 to 2026-06. Adding the hand-set 60% clip at D5 lowers the D5 error
  on sunny half-hours when the network head is absent, and adds nothing measurable when the head is
  present, because experiment 232 found the head cancels most of the clip's gain.
prediction: >-
  Written before any score is read; each can fail. P1 and P4's BG2 clause reproduce experiment 228
  and are a replication check; the contest between arms is the D5 sunny-daylight cut. (P1) The gate
  passes and arm 1 (A+H+S) beats BG2 on the six-substation quantile score (D9 included in the six,
  not in the gate) by 0.10 to 0.17 points, as experiment 228 found (-0.136). (P2) Arm 2 (arm 1 with
  the clip) is identical to arm 1 at D9, D10, D11, D14, and D16, and at D5 its quantile score on
  sunny-daylight rows is less than 0.15 points below arm 1's, because experiment 232 found the head
  cancels most of the clip's gain. (P3) Arm 2b (A+S with the clip and no head) lowers the D5
  sunny-daylight median absolute error below arm 1's by at least 0.05 MW, with the 95% interval
  excluding zero. (P4) The winning arm beats the manual heuristic on all four headline metrics (mean
  absolute error of the median, CRPS, tail-weighted CRPS at the 95th percentile, and the
  scaled-limit curtailment cost at f = 0.9), and is worse than BG2 on the quantile score in at least
  3 of the 6 months April to September. (P5) Arm 3 is arm 2 with experiment 205's injected-gap
  scores attached, so it equals arm 2 on the scored year's rows; experiment 205's stored scores show
  coasting 0.25 to 0.44 points below the frozen model at gaps of 3, 6, 24, and 72 hours.
decision_rule: >-
  Gate first: arm A within 0.005 of 2.7519, BG2 within 0.005 of 2.780, and arm 1 within 0.005 of
  2.617 on the five decision substations (experiments 220 and 228), and an assertion that the row
  keys of every arm match exactly. The winning arm is arm 2 if its D5 sunny-daylight quantile score
  is at least 0.05 points below arm 1's with the 95% interval excluding zero; otherwise arm 1 (and
  arm 2b is reported as a post hoc variant). `supported` needs the winning arm to beat BG2 on the
  six-substation quantile score with the month-resampled 95% interval excluding zero and to beat
  the manual heuristic on all four headline metrics. `partly supported` needs the BG2 contrast but
  not all four heuristic contrasts. `not supported` if the BG2 interval includes zero. The threshold
  is zero because every part is already built and this is a final scoring, not a port decision.
benchmark: >-
  BG2 (experiment 87 p_s6_bg2), the manual heuristic, climatology, and the weak production-style
  XGBoost model (experiment 87 ref_xgboost_baseline), on the rows of experiment 228.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Mean absolute error of the median, CRPS (twice the mean pinball loss over 11 quantiles, in points
  of effective capacity), tail-weighted CRPS at the 95th percentile, and the scaled-limit curtailment
  cost; coverage and pinball loss at q01, q05, q95, q99.
code: ml-experiments/00251-dp-final-best-guess-assembly-clip-and-coast-against-bg2/
production_status: not-for-production
adopted_in: none
date: 2026-10-22
---

**Plan, written before any score is read.** The experiment assembles stored forecasts and fits
nothing. Its question is whether the stack of [experiment
220](../00220-dp-bg-stage4-assembly-with-drop-one/result.md) (arm D, a network head, and the
two-number solar summary), with the D5 export clip of [experiment
180](../00180-dp-sunny-daylight-fix-d5-solar-shape/result.md) and the coast-and-widen rule of
[experiment 205](../00205-dp-gamk-gap-robustness-coast-or-frozen-gam/result.md) added, beats BG2 and
the manual heuristic over the scored year, and which rows it loses on.

**The clip arms are an approximation.** The solar summary and the network head of arm 1 were built
on the unclipped solar term, and the head was trained on unclipped errors (experiment 232 found
that adding the clip after the head loses most of the clip's gain). The clip shift is a mean
difference per row, added to every quantile, so the bands keep their unclipped width; the write-up
reports D5 tail coverage for arm 2 to show whether that matters. [Experiment
237](../00237-dp-two-number-solar-summary-with-clip/result.md) scored the two-number summary with
the clip applied inside the member pass at D5 and D9 and is the closer test of that combination.

## Arms

- **Arm 1:** experiment 220's A+H+S.
- **Arm 2:** arm 1 plus the 60% clip at D5. Its mean forecast is arm 1's plus the clip's shift, the
  difference between experiment 214's clipped arm D and experiment 180's unclipped arm D
  (identical to experiment 220's arm D) on the same rows; every quantile moves by that shift. The
  other five substations are unchanged.
- **Arm 2b (post hoc, labelled):** the same shift added to A+S (no network head), because
  experiment 232 found the head cancels the clip's gain.
- **Arm 3:** arm 2 plus coast-and-widen. Coasting changes only rows after a reading gap, so on
  gap-free rows it equals arm 2; its gap-row scores come from experiment 205's stored scores.
- **Arm 4 (network before the filter):** not run. Experiment 222 did not support it.
- **Comparison arms:** BG2, the manual heuristic, climatology, and the weak production-style XGBoost
  model, all stored by experiment 87.

## Plan for the write-up

- Tail coverage and pinball loss at q01, q05, q95, and q99 by four UTC hour bands (00-06, 06-12,
  12-18, 18-24). The curtailment metric is the scaled-limit node cost of
  [`docs/roadmap/cost-savings-metrics.md`](../../roadmap/cost-savings-metrics.md), Metric 2, at f =
  0.9 with 0.75 and 0.5 as a sweep, copied from experiment 228. Arm 3 reports the share of scored
  rows after a reading gap beside experiment 205's gap scores.
- Figures first: MAE against lead time for every stage and comparison arm, then the contrasts by
  month, season, lead band, and substation, tail coverage by hour band, and the curtailment saving.
- Cuts: cold nights below 2 C, Christmas and bank holidays, sunny daylight.
- Files for the substation-cards session: `dp_preds` for arm 1 and the winning arm, a table of
  per-substation fitted properties, the score tables, and a README.
