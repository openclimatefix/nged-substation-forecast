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
status: done
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
verdict: refuted
production_status: not-for-production
adopted_in: none
headline: >-
  On the sunniest third of daylight half-hours (0.5 to 168 hours, scored year, six Primaries),
  reading the capacity in the physical unit shape raised the quantile score by 0.065 points (95%
  month-resampled interval +0.017 to +0.126), from 4.371 to 4.436. The shape mismatch of experiment
  349 does not explain its null result.
best_arm: >-
  the default's pinned capacity (arm A): 4.371 on the sunniest third and 2.773 on all half-hours at
  0.5 to 168 hours, against 4.436 and 2.775 for arm P
take_home_message: >-
  Reading the solar capacity in the same physical unit shape the forecast uses did not make the
  demand forecast better on sunny half-hours: the score was 0.065 points worse than with the
  pinned capacity (+0.017 to +0.126), the same direction as the earlier read-out in the empirical
  shape. So the mismatch of shapes in experiment 349 was not the reason its null result, and the
  capacity estimate is not the limit on sunny half-hours. Dropping one Primary, D16 or D10, widens
  the interval to include zero, and an audit has not yet been run.
lessons:
  - >-
    The empirical-shape read-out (arm B) and the physical-shape read-out (arm P) score within 0.018
    points of each other on sunny half-hours (-0.020 to +0.054), so the shape mismatch is not what
    holds the estimate back.
  - >-
    The capacity read out in the physical shape and divided by the injection recovery is 5 to 13%
    above experiment 349's capacity at every Primary, and above the pin at D5, D10, D11, and D14 in
    July 2025. Leaving the divisor out (arm R) gives +0.048 (-0.077 to +0.173) against the pin.
  - >-
    Arm P lowers the score on cold winter nights (-0.015, -0.019 to -0.007) and on the darkest third
    of daylight (-0.041, -0.064 to -0.019), so the estimate carries some information the pin lacks,
    away from the sunny half-hours.
advice:
  - >-
    Keep the pinned capacity in the first stage of Primary demand research default 3: neither the
    empirical-shape read-out nor the physical-shape read-out lowered the sunny-half-hour score
    (figure 1).
  - >-
    Do not repeat a capacity read-out in another shape to explain experiment 349: the two shapes
    differ by 0.018 points (-0.020 to +0.054) on sunny half-hours (P minus B).
  - >-
    Test the capacity's effect on cold winter nights and the darkest daylight third, where arm P
    scored better than the pin, before concluding that the read-out carries no information (figure
    1).
kind: change
---

## Take-home message

**Reading the solar capacity in the physical unit shape the forecast already uses did not lower the
score on sunny half-hours, so the shape mismatch is not why experiment 349's read-out failed.** The
research forecast of demand at six Primary substations (anonymised as D5, D9, D10, D11, D14, and
D16) subtracts the output of unmetered rooftop solar from each substation's demand, using a solar
capacity pinned to the values of
[experiment 45](../00045-pvdis-4-weekly-capacity-asymmetric-l1/result.md).
[Experiment 349](../00349-primary-demand-default-3-exp-269-capacity-in-the-stack/result.md) read the
capacity with the empirical plant shape of six metered farms from
[experiment 269](../00269-pv-capacity-gain-decomposition/result.md) and then multiplied it by the
forecast's physical shape, which its page lists as a shortcut that could explain its null result.

We asked whether a capacity read out in the physical shape, which the forecast multiplies it by,
scores better. We refitted the first stage of the forecast (the convex demand model with a Kalman
filter of [experiment 139](../00139-differentiable-physics-pv-as-of-capacity-in-gam-k/result.md))
at every monthly origin, using only rows before the origin. Each read-out was divided by the median
share of five injected metered farms that the same fit recovered (0.74 to 0.86).

Scores are quantile scores in points of the substation's effective capacity, smaller is better, and
a positive change is worse. On the sunniest third of daylight half-hours (the third of each
Primary's daylight half-hours with the highest solar output per unit of capacity), arm P scored
4.436 against 4.371 for the pin, a change of +0.065 (95% interval from resampling whole months,
+0.017 to +0.126). **Verdict: refuted**, by the pre-registered rule that the interval lie wholly
above zero. Leaving out D16 or D10 widens the interval to include zero, which fires an audit
trigger (see Checks). The first stage only was refitted, so the full stack and the 168 to 240 hour
range are untested.

## Figures

![Quantile-score contrasts, 0.5 to 168 hours](figure_1_contrasts_0p5_168.svg)

![Quantile-score contrasts, 24 to 168 hours](figure_2_contrasts_24_168.svg)

![Pinned capacity and the read-outs of experiments 349 and 351 by Primary](figure_3_capacities.svg)

## Results

- **Arms.** This is a tier 2 change: the capacity is read inside the fitted stage, with the
  default's code imported from the experiment folders that hold it. A is the pinned capacity
  (experiment 45's `frozen` vintage). B is experiment 349's arm B, refitted here from its stored
  as-of table. P is the monthly as-of read-out in the physical unit shape, divided by its own
  injection recovery (a table of 27 origins, 2024-04 to 2026-06, at six Primaries). R is P without
  the divisor. Arm P2, a read-out on the stack's own residual solar signal, was skipped because
  that signal exists only after the later stages are refitted.
- **Code reused.** `ml-experiments/00351-primary-demand-capacity-physical-shape-readout/` copies
  the scripts of experiment 349 and drops the empirical shape from the table builder
  (`e00351_asof_table.py`); it reads experiment 349's private as-of table for arm B.
- **Row set and range.** `fold_a_daily_gap_v1`, scored year 2025-07-02 to 2026-06-30, six
  Primaries, rows outside the hand-filtered events, origin 09:00 UTC, lead 0.5 to 168 hours and 24
  to 168 hours. The 168 to 240 hour range was not rerun, so the result covers 7 of the 10 forecast
  days that NGED (the distribution network operator) cares about.
- **Schedule.** Monthly refits of the first stage on a 24-month window, as in experiment 349. The
  contrast could depend on the schedule only through the capacity path, which changes monthly.
- **Reproduction gate.** Arm A equals experiment 139's stored pin forecasts at all six Primaries
  (largest gap in q01, q50, and q99 0.0). Arms A and B reproduce experiment 349's stored scores on
  the sunniest third (4.371 and 4.418) and on all half-hours (2.773 and 2.777).
- **Headline, sunniest third (scored year, 0.5 to 168 hours).** C1, P minus A: +0.065 (+0.017 to
  +0.126). C2, P minus B: +0.018 (-0.020 to +0.054). C3, R minus A: +0.048 (-0.077 to +0.173).
  B minus A: +0.047 (-0.029 to +0.135). At 24 to 168 hours C1 is +0.068 (+0.018 to +0.131), C2 is
  +0.025 (-0.014 to +0.063), and C3 is +0.035 (-0.099 to +0.160). Error of the mean forecast, P minus
  A, 0.5 to 168 hours: +0.142 (+0.070 to +0.249).
- **Absolute scores, 0.5 to 168 hours (A, B, P, R).** Sunniest third 4.371, 4.418, 4.436, 4.419.
  All half-hours 2.773, 2.777, 2.775, 2.790. At 24 to 168 hours the sunniest third scores 4.511,
  4.555, 4.580, 4.546.
- **Replicate year (2024-07 to 2025-06), P minus A.** Sunniest third +0.066 (+0.016 to +0.088) at
  0.5 to 168 hours and +0.073 (+0.018 to +0.099) at 24 to 168 hours, so both years point the same
  way. P minus B is +0.026 (-0.039 to +0.071).
- **The other groups, P minus A at 0.5 to 168 hours.** Gains of more than 0.01 points: cold winter
  nights (December to February, after dark) -0.015 (-0.019 to -0.007) and the darkest third of
  daylight -0.041 (-0.064 to -0.019). No change beyond 0.01 points in all half-hours (+0.002), night
  (-0.000), or all daylight (+0.006).
- **By Primary, sunniest third, P minus A.** D5 +0.033, D10 +0.182, D11 +0.029, D14 -0.056, D16
  +0.213, D9 -0.009. Leaving one Primary out gives +0.036 (-0.006 to +0.091) without D16, +0.041
  (-0.016 to +0.111) without D10, +0.071 (+0.040 to +0.113) without D5, +0.073 without D11, +0.081
  without D9, and +0.090 without D14.
- **Capacities (figure 3, the capacity each fit applies, in points of effective capacity).** At the
  2025-07-01 origin P exceeds the pin by 10% at D10, 12% at D11, and 26% at D14, and is below it by
  24% at D16 and 7% at D9; B differs from the pin by under 3% at D10, D11, and D5 and by
  -29% at D16. Over the 27 origins, P differs from the pin by more than 10% in 18 (D10), 18 (D11),
  20 (D14), 23 (D16), 8 (D5), and 1 (D9) origins. P differs from B by more than 10% in 20 (D10), 27
  (D11), 2 (D14), 4 (D16), 1 (D5), and 8 (D9) origins; P is above B at every Primary because the
  divisor (0.74 to 0.86) scales a capacity that is already in the physical shape's units.
- **Tails, six Primaries, 0.5 to 168 hours (A, B, P, R).** Coverage below q01 is 2.20%, 2.17%,
  2.12%, and 2.19%; below q05 6.68%, 6.67%, 6.61%, and 6.70%; below q95 94.74%, 94.68%, 94.71%, and
  94.65%; below q99 98.35%, 98.30%, 98.34%, and 98.26%. Pinball loss at q01 is 0.215, 0.216, 0.216,
  and 0.215; at q05 0.721, 0.725, 0.727, and 0.725; at q95 0.720, 0.715, 0.716, and 0.716; at q99
  0.220, 0.220, 0.220, and 0.219. No tail moves by more than 0.006 points.
- **Reference rows.** The first stage alone is not Primary demand research default 3, so the
  manual heuristic and climatology are not placed beside it. The research default's own scores
  (2.727 at 0.5 to 240 hours, against 3.414 for the manual heuristic and 3.664 for climatology) are
  in the [research default log](../research-default-log.md).
- **Model family.** One model family and no blend.
- **Prediction scorecard.** Predicted P minus A within -0.03 to +0.05: missed (+0.065). P minus B
  within +/-0.03: held (+0.018). R minus A within 0.03: missed (+0.048). Cold winter nights and tails
  within 0.02: the winter-night change is -0.015 and the tails hold.

## Checks

- **Median minus mean forecast, per Primary and arm:** 0.0 points for all four arms.
- **Mean-forecast error beside each contrast:** P minus A on the sunniest third is +0.142 (+0.070 to
  +0.249), the same direction as the quantile score. On cold winter nights the mean-forecast error
  moves the same way as the quantile score (-0.037, -0.050 to -0.015).
- **Triggers.** Trigger 4 fired: leaving out D16 or D10 turns the verdict of C1 from `refuted`
  into `inconclusive` (+0.036, -0.006 to +0.091; +0.041, -0.016 to +0.111), although the sign holds.
  Trigger 2 is borderline: C1 contradicts its prediction by 0.015 points, and the session tested an
  explanation (the capacity tables above: P sits 5 to 13% above B because the divisor scales a
  capacity already in the physical shape's units, and arm R without the divisor is nearer to the
  pin but its interval is wide). No Opus audit has been run, because the brief allows Sonnet only.
  The experiment is not merged and waits for the manager's audit.

## Caveats and lab shortcuts

- **The plant-response table of arm B is not as-of.** Arm B reads experiment 349's table, which was
  fitted once to the six farms' history to 2026-06-30. Arm P uses no plant-response table: only the
  capacity fit and its injection recoveries are used, both with rows before each origin.
- **The first stage only.** The neural-network head, the solar summary, the blend, and the 168 to
  240 hour range were not rerun.
- **Divisor choice.** P divides by the injection recovery, which makes its capacity larger than the
  physical-shape units want; R is the unit-consistent version, with a wide interval.

## What to try next

- Test the capacity's effect on cold winter nights and the darkest daylight third, where P scored
  better than the pin, in the full stack.
- Read the capacity out in the empirical shape and apply it in the empirical shape, in the first
  stage and in the ensemble's solar response, if the mismatch is still suspected.
