---
id: '00220'
slug: dp-bg-stage4-assembly-with-drop-one
part: BG-S4
title: >-
  Do the solar summary, the pooled correction network, and the gap rule stack on the best-guess
  demand forecast, or do they remove the same error?
theme: differentiable-physics-forecast-engine
also_relevant_to:
- production-reference-and-best-guess
- calibration-and-uncertainty
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00220-dp-bg-stage4-assembly-with-drop-one/
  only. Serving path: not touched. Degradation rule: not touched (experiment 205's coast-and-widen
  rule is scored on injected gaps, not changed). Designs: more than one is defensible (how the
  network's correction and the solar summary combine with the bands, and how a part is removed when
  the network was trained with it in); this plan pre-registers one of each. Callers: this folder's
  scripts, which copy experiment 169's loaders and scoring and experiment 205's gap code; they call
  the reviewed `studies.*` modules without changing them. No new modelling and no refits.
hypothesis: >-
  The three parts added to the best-guess demand forecast since experiment 170 remove mostly
  different errors, so their gains add. The parts are the two-number solar summary of the 51
  ensemble members (experiments 193 and 202), the pooled correction network on arm D's mean
  (experiments 200, 201, and 209), and the coast-and-widen rule for missing readings (experiment
  205). The solar summary changes only daylight bands, and the network corrects arm D's median from
  its own output, the calendar, and temperature, so the two act on different rows and different
  error. The network may remove what the winter night offset and recipe R removed (both correct arm
  D's mean by time of day), and it may make the bands' cold-night widening unnecessary.
prediction: >-
  Written before any score is read. (1) The stack with the network and the solar summary (A+H+S)
  scores at least 0.10 points below arm A (2.7519) on all rows at the five decision substations,
  with the 99% interval wholly below zero, and its gain is at least 85% of the sum of the gains of
  A+H and A+S. (2) The network's own 11 quantiles (H-own), with none of experiment 160's or 164's
  band fixes, reach cold-night q95 coverage of at least 0.90 and score within 0.02 points of A+H
  on all rows. (3) In the drop-one table over A+H+S, removing the network costs more than removing
  recipe R or the winter night offset. A fail on (1) names which parts overlap; the write-up says
  so first.
decision_rule: >-
  Headline contrast C1 is A+H+S minus A, over leads 0.5 to 168 hours at the five decision
  substations D5, D10, D11, D14, and D16 in the scored year (D9 in the six-site tables only),
  outside the hand-filtered switching events, in points of effective capacity. Intervals come from
  the paired month-resampled bootstrap with 20,000 resamples, at the 95% level for every table and
  the 99% level for C1. C2 is the stacking ratio C1 / (C(A+H) + C(A+S)), reported with the
  interaction C1 - C(A+H) - C(A+S) and its 95% interval. Prediction 1 passes when C1 is at most
  -0.10 with its 99% interval wholly below zero and the ratio is at least 0.85. Prediction 2 passes
  when H-own's cold-night q95 coverage is at least 0.90 and its all-row score is within 0.02 of
  A+H. Prediction 3 passes when the cost of removing the network (A+S minus A+H+S) exceeds the
  cost of removing recipe R and the cost of removing the winter night offset (the point
  estimates; the intervals are reported). `supported` needs predictions 1 and 2 to pass,
  `partly supported` one of them, and `not supported` neither; prediction 3 is descriptive and
  enters no verdict. The 0.10 threshold is set by the added complexity: the network adds a trained
  neural network to the serving path, and a gain under 0.10 points (3.6% of the score) is too small
  to justify that; the solar summary costs two stored columns, so any gain whose interval excludes
  zero would justify it. The verdict is capped at `inconclusive` if the reproduction gate fails:
  arm A must score 2.7519 within 0.005 on the five-substation rows before any contrast is read.
benchmark: >-
  Arm A (experiment 169's arm A: experiment 155's arm D with experiment 160's smooth spread table
  and experiment 164's cold-day widening, 2.7519 over leads 0.5 to 168 hours at the five decision
  substations in the scored year). Absolute reference rows: experiment 169's BG2 (2.7804), manual
  heuristic (3.4968), and climatology (3.8038).
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over the 11 quantiles of experiment 87) in points of
  effective capacity, on rows outside the hand-filtered switching events; coverage and pinball loss
  at the 0.01, 0.05, 0.95, and 0.99 quantiles on all rows, cold nights, and daylight; leads 0.5 to
  168 hours and leads 15.5 to 39 hours
code: ml-experiments/00220-dp-bg-stage4-assembly-with-drop-one/
date: 2026-10-22
---

**Plan, written before any score is read.** This page is the plan; the write-up replaces it when
the run finishes.

## Question

Three parts have been added to the best-guess demand forecast since experiment 170, each tested
alone. We ask whether their gains add when they are combined, or remove the same error.
Experiment 167's drift term gained 0.50 points alone and -0.03 points inside the full stack
(experiment 170), so stacking cannot be assumed.

- **Solar summary (S).** The mean and the standard deviation of the solar response of the 51
  ensemble members, as in
  [experiment 193](../00193-dp-ens-two-number-solar-summary/result.md) and its rolling replay
  [experiment 202](../00202-dp-ens-two-number-rolling-replay/result.md) (R2, daylight quantile
  score -2.3%).
- **Pooled correction network (H).** The two-hidden-layer network of
  [experiment 201](../00201-dp-gam-residual-mlp/result.md) (arm 2) that corrects arm D's mean,
  with [experiment 200](../00200-dp-pooled-correction-head-on-kalman-state/result.md) and
  [experiment 209](../00209-dp-pinball-table-vs-network-head-on-arm-d/result.md) as the evidence
  that the network, not a binned table, carries the gain.
- **Coast-and-widen rule (C).** [Experiment
  205](../00205-dp-gamk-gap-robustness-coast-or-frozen-gam/result.md)'s rule for missing readings.

## Arms

All arms share arm A's rows. Arm A is the reference of
[experiment 169](../00169-calibration-combined-fixes-and-ensemble-spread/result.md): arm D of
[experiment 155](../00155-dp-bg-stage2-assembly-night-offset-and-error-correction/result.md) with
[experiment 160](../00160-calibration-band-width-fix-on-stage-2-winner/result.md)'s spread table
and [experiment 164](../00164-calibration-cold-night-tail-under-coverage/result.md)'s cold-day
widening.

- **A.** Reference.
- **A+S.** A with experiment 202's R2 quantiles on every row (R2 changes daylight rows only).
- **A+H.** A with the network's median correction (`delta`, the mean over the three seeds) added
  to all 11 quantiles. The band offsets from the mean are A's.
- **A+H+S.** R2's quantiles plus `delta`.
- **H-own.** The network's own 11 quantiles (the mean over the three seeds), with none of
  experiment 160's or 164's band fixes.
- **A+H+S+C.** A+H+S plus the effect of the coast-and-widen rule on injected gaps of 3, 6, 24, and
  72 hours, on the gap rows only (post hoc composition, below).

**Drop-one table over A+H+S.** Each row removes one part and keeps the rest:

| Removed | Construction |
|---|---|
| S | A+H |
| H | A+S |
| Cold-day widening (experiment 164) | experiment 160's table alone (the `ref` arm of experiment 169) with `delta` and R2's daylight change added |
| Recipe R | experiment 155's arm B (the offset without R) as the mean, with A's offsets from the mean, `delta`, and R2's daylight change added |
| Winter night offset | experiment 155's arm C (R without the offset) as the mean, with the same additions |

## Composition rules

Two rules are fixed here, before any score is read.

- **The network is not refitted.** `delta` is the correction experiment 201 learned for arm D. When
  an arm removes recipe R or the winter night offset, `delta` is added unchanged to the changed
  mean, so the "remove R" and "remove offset" rows measure those parts with the network frozen. The
  same two rows are also reported without the network (on A+S) so the reader can compare.
- **R2 is added as a change.** Where an arm's base differs from A, R2's change in each quantile
  (R2 minus A, on the same row) is added to the base. This applies only to the drop-one rows that
  change the mean; for A+S and A+H+S the R2 quantiles are used directly.
- **Coast-and-widen composition (post hoc).** Experiment 205's code is re-run for rule (b)
  (coasting with wider bands) and for the no-gap forecast on the event runs that fall in the scored
  year. Each quantile's difference between the two is added to A+H+S's quantiles on the same rows,
  with `delta` and R2 unchanged. This ignores any interaction between a gap and the network or the
  solar summary, which is why the arm is reported separately and enters no verdict.

## Scope and departures from the brief

- **Row sets.** Five decision substations (D5, D10, D11, D14, D16) for every headline number; D9 in
  six-site tables. Leads 0.5 to 168 hours, and leads 15.5 to 39 hours (the rows on which experiment
  201 scored the network); each table says which rows it uses. A+H is also compared with arm D on
  the leads 15.5 to 39 hours.
- **Scored year only.** The network (experiment 201) and the solar summary (experiment 202) exist
  only for the scored year (2025-07-02 to 2026-06-29). The replicate year the brief names cannot be
  built without refitting both, which this experiment does not do. The write-up says so.
- **Seeds.** `delta` and the H-own quantiles are the mean over experiment 201's three seeds. The
  per-seed H-own scores are also reported.
- **Subsets.** Cold nights as in [experiment 169](../00169-calibration-combined-fixes-and-ensemble-spread/result.md):
  November to March, night and evening bands, ERA5 daily mean temperature below 2 °C. Daylight
  as experiment 202 defines it (`kind` is `daylight`).

## Gates (blocking)

1. Arm A scores 2.7519 within 0.005 on the five-substation rows.
2. Arm A's quantiles equal experiment 202's stored arm A on every shared row, and experiment 201's
   `actual` equals arm A's `actual`.
3. Experiment 155's arm D mean equals arm A's mean; arms B and C cover the same rows.
4. R2 alone reproduces experiment 202's printed daylight change (-2.30%, interval -2.96% to -1.61%)
   within 0.05 percentage points.
5. Arm 2 of experiment 201 reproduces its printed contrast against arm D (-0.104 on six
   substations) within 0.01.

## Planned contrasts

C1: A+H+S minus A. C2: the stacking ratio and interaction above. C3: H-own minus A+H. C4: A+H
minus A, A+S minus A. C5: each drop-one row minus A+H+S. All on all rows and leads 0.5 to 168
hours, then cold nights, daylight, and leads 15.5 to 39 hours. The curtailment proxy of experiment
169 is scored for A, A+H+S, and H-own if the scaled-limit code runs on the new arms without change;
otherwise the page says it was not scored.

## Checks the write-up carries

The median forecast minus the mean forecast per site, the error of the mean beside every quantile
score contrast, and which of the seven audit triggers fired. Triggers 3 and 5 are the likely ones:
A+H shifts the median away from the mean, and H-own is a different band construction.

## Amendments after the plan review (before any contrast is read)

- **Prediction 1 needs both clauses.** C1 at most -0.10 with its 99% interval below zero, and a
  stacking ratio of at least 0.85. The ratio gets a month-resampled interval. If either single
  gain (A+H or A+S) is under 0.02 points, the ratio is reported as undefined and only the
  interaction decides. A ratio above 1.15 is reported as super-additive, not as a quiet pass.
  Leads 0.5 to 168 hours at the five decision substations decide prediction 1; the 15.5 to 39
  hour rows are descriptive. If prediction 1 fails, the write-up opens with the pairwise
  interactions and the drop-one differences.
- **`delta` re-centres only.** A+H adds `delta` to all 11 quantiles and keeps A's offsets. The
  write-up reports A+H coverage at q05 and q95 on all rows, cold nights, and daylight. A refit of
  the offsets is not run unless A+H's coverage is off by more than 0.02.
- **Frozen-network bias.** `delta` was learned on arm D, which contains recipe R and the offset, so
  the frozen-network rows understate the cost of removing R or the offset. Prediction 3 is
  therefore judged on both versions (with and without the network) and a pass on the frozen
  version alone is weak evidence.
- **Quantile order.** Every arm is sorted per row after construction; the number of rows that
  needed it is reported.
- **Self-consistency gate 6.** A+H+S built directly (R2 plus `delta`) equals A plus (R2 minus A)
  plus `delta`; the two are compared to 1e-6.
- **Row counts and keys.** Joins use `(site, run, valid_time)` with a one-to-one check; row counts
  after each join are in the gates file. Every arm covers the same rows.
- **Tables.** One tail-calibration table per arm (q01, q05, q95, q99 coverage and pinball) on all
  rows, cold nights, and daylight. Cold nights are experiment 169's definition, which is
  experiment 160's and 164's (ERA5 daily mean below 2 °C, November to March, night and evening).
- **Gap arm.** Sorted after the addition, with the gap-row count per gap length and the fraction of
  rows rearranged; it cannot show any interaction between a gap and the network or solar summary.
- **Citations and the unused network.** The experiment also builds on
  [experiment 187](../00187-dp-ens-members-in-stack/result.md) and [experiment
  170](../00170-dp-bg-stage3-drift-and-holiday-scoring/result.md). Experiment 211's blind reference
  network is not used. The replicate year is dropped for every arm.
