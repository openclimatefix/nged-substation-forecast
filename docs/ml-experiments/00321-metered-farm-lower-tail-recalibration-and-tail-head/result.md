---
id: '00321'
slug: metered-farm-lower-tail-recalibration-and-tail-head
title: >-
  Which single change brings the 0.01 quantile of the metered solar-farm forecast to its nominal
  coverage at leads of 1 to 10 days without costing the quantile score or the 0.99 gain?
theme: metered-solar-farms
also_relevant_to:
- calibration-and-uncertainty
- curtailment-cost
status: planned
kind: change
size: >-
  medium. Stored: private predictions and tables under
  data/ml-experiments/00321-metered-farm-lower-tail-recalibration-and-tail-head/ only. Serving
  path: not touched. Degradation rule: not touched. Designs: several are defensible, but the arms
  are fixed by the question. Callers: this folder's scripts, which import the folders of
  experiments 310 and 318 and call reviewed `studies.*` modules without changing them. No change
  under `packages/`. Asset graph: not touched. One Sonnet code review and one Sonnet prose review,
  no Opus; an audit runs (Sonnet) if a trigger fires.
hypothesis: >-
  Metered solar farms research default 1 (the blend assembled by experiment 300, scored to 10 days
  by experiment 310) lets the actual output fall below its 0.01 quantile 3.3% to 3.7% of the time,
  and no mixture-density head tried in experiment 318 changes that. Default plus a recalibration
  table, a zero-inflated lower tail, or a tail-targeted head brings the share of actual values
  strictly below the 0.01 quantile close to 1% at leads of 1 to 10 days, without raising the
  quantile score and without losing experiment 318's 0.99 gain.
prediction: >-
  Written before any score is read; each can fail. (P1, arm 1, recalibration table) The share of
  actual values strictly below the 0.01 quantile lies within 0.5 points of 1% (0.5% to 1.5%) at
  every day band, in own history and in zero history, and the quantile score changes by under 0.02
  points. Can fail: the misses are outages and zeros, which a table fitted on earlier months may
  not predict. (P2, arm 2, zero-inflated lower tail) The 0.01 pinball loss falls by at least 5%
  against research default 1. Can fail: outage half-hours are rare (experiment 301). (P3, arm 3,
  tail-targeted head) The 0.99 pinball loss keeps the 4.8% to 5.6% gain of experiment 318 at days 3
  to 7 and gains a further 2% at days 1 to 2. Can fail: the gain came from inputs that help a head
  nowhere else. (P4, arm 4, one pooled head with no plant) The quantile score is within 1% of
  research default 1 at every day band. Can fail: experiment 313 found a gap of 3.4% at days 8 to
  10, which experiment 318 closed by 3.0%. (P5, controls) The shuffled-input tail head, the
  farm-swapped table, and the one-week-stale table change the 0.01 coverage by under 0.3 points
  (shuffled head) or leave it near 3.3% to 3.7% (swapped table).
decision_rule: >-
  Family rule of research-default.md for Metered solar farms research default 1, with the headline
  the farm-equal quantile score in points of capacity, six planned contrasts per day band read at
  the 1 - 0.01/6 level (99.83%) with month-block resampling of the 12 scored months: recalibration
  table, shrunk table, zero-inflated blend, tail-head blend, pooled no-plant head, each minus
  research default 1, and arm 5 minus research default 1 (run only if arms 1 and 3 each pass their
  own prediction). The recalibration table is a recalibration (minimum gain 0 points); the
  zero-inflated head and the tail-targeted head are new heads (minimum gain 0.03 points); the
  pooled head is a changed setting of an existing head. Tolerances: 0.02 points on the pooled
  headline and 0.05 on the 0.01 and 0.99 pinball tails, one-sided 95%. The replicate is the second
  half of the scored year (2026-03 to 2026-08) with the same sign. Because no admissible subgroup
  is defined for this family, a change that helps only a tail is recorded as a candidate and
  not adopted. At most one arm is adopted. Stop rule: if no arm brings the 0.01 coverage under 2%
  at any band, report that the lower tail is outages that no input predicts and add no arms.
benchmark: >-
  Metered solar farms research default 1 (experiment 310's `blend_s0` stored forecasts), each model
  family alone (GAM, mixture-density network, plant-fed XGBoost), experiment 318's head, and the
  `climatology` and `manual_heuristic` floors of experiment 310
row_set_id: >-
  Experiment 310's rows (farms A to F, scored year 2025-09 to 2026-08, day bands 1 to 10)
metric: >-
  Quantile score (twice the mean pinball loss over 11 levels, in points of capacity), pinball loss
  at 0.01, 0.05, 0.95, and 0.99, and coverage at those levels
code: ml-experiments/00321-metered-farm-lower-tail-recalibration-and-tail-head/
---

## Plan (committed before any score is read)

The five items of the research-default rule, in order:

- **Family and version.** Metered solar farms, starting from Metered solar farms research default 1
  (the blend of experiment 300; stored forecasts `blend_s0` of experiment 310).
- **Arms.** "default" is the blend. "default plus x" has five pre-registered forms. Arm 1: a
  recalibration table of quantile offsets per farm, lead day (1 to 10), and quantile level, fitted
  rolling-origin on the stored blend forecasts of earlier months (the offset is the empirical
  quantile at the nominal level of actual minus forecast), applied after sorting and clipped to
  zero or more; arm 1s is the same table multiplied by k/(k+3) for k earlier months. Arm 2: a
  zero-inflated lower tail, where the mixture-density head gets one extra output, the probability
  of an exact-zero half-hour (output of 0.002 or less of the scale) from the inputs of the default
  head plus the hour as a sine and cosine pair, and the head's quantiles are those of
  pi times a point mass at zero plus (1 - pi) times the mixture; the blend then uses this head in
  place of the mixture head (the weights are refitted by the same rolling simplex). Arm 3: a
  tail-targeted head, a small network with experiment 318's full inputs and the 0.01, 0.02, 0.05,
  0.95, 0.98, and 0.99 quantiles as outputs, trained with the pinball loss plus a non-crossing
  penalty on all bands pooled; the blend's 0.01, 0.05, 0.95, and 0.99 quantiles are replaced by the
  head's and the rest stay. Arm 4: experiment 318's `pure_lead_pooled` (the pure mixture-density
  network with lead features, no plant column, one head for all bands), with a zero-history fit
  added. Arm 5: arm 1's table applied to arm 3's output, run only if arm 1 and arm 3 each pass their
  own prediction.
- **Tier.** Arms 1 and 5 act on stored forecasts (tier 1); arms 2, 3, and 4 act inside a fitted
  stage (tier 2, importing the code of experiments 310 and 318). Complexity: arm 1 a recalibration
  (minimum gain 0), arms 2 and 3 new heads (0.03 points), arm 4 a changed setting.
- **Controls.** The shuffled-input tail head (lead and clear-sky index shuffled across rows, own
  history), the farm-swapped table (each farm uses the next farm's table), the one-week-stale table
  (fitted on rows at least 7 days before the month starts), `climatology`, and `manual_heuristic`.
  A reproduction gate: the five-seed refit of experiment 318's `default` head must score within
  0.02 points of experiment 318's stored value.
- **Coverage definition.** The headline coverage is the share of actual values strictly below the
  quantile, because an exact-zero actual sits on a point mass and the "at or below" rule counts it
  as covered by every non-negative quantile; the "at or below" share used in experiments 310 and
  318 is reported beside it with the share of exact-zero actuals.
- **Schedule.** Quarterly refits on all earlier months (the screening schedule), five seeds for
  neural arms in own history and three in zero history, rows exactly as experiment 310. Absolute
  scores could shift under weekly refits and the contrasts were not tested for dependence on the
  schedule.

