---
id: '00361'
slug: d8-step-catalogue-detector-injection-and-gam-stack
title: >-
  Can a plain step detector separate generator-like steps from D8's demand, and does removing them
  from the training target improve the GAM stack?
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - calibration-and-uncertainty
  - lsc-detectors
status: done
kind: other
size: >-
  medium. Stored: private files under
  data/ml-experiments/00361-d8-step-catalogue-detector-injection-and-gam-stack/ only. Serving path:
  not touched. Degradation rule: not touched (a detector run over the history at each fit cut is an
  R&D device here; a port would need its own plan). Designs: more than one is defensible (a
  changepoint rule or the load-shift detectors; add the removed level back or mask the rows), so
  the plan fixes one plain rule and both removal choices as arms. Callers: the scripts import
  experiments 252 and 264 and copy experiment 354's fit code with one change, the training target.
  Asset graph: not touched.
hypothesis: >-
  D8 (Marsh Lane) is a Primary substation whose demand carries on/off steps of about 4 MW and
  multi-day drops of about 2.7 MW that look like a dispatchable generator. A plain, as-of step
  detector finds such components, and a GAM stack fitted on a training target with the detected
  steps removed scores better at D8 than the same stack fitted on the raw target, with the removed
  component returning through the own-residual band as the chance that the generator is running.
prediction: >-
  Written before any score is read; the catalogue is not yet built. (P1) The detector recovers at
  least 80% of injected components of 2 MW or more when one component runs at a time, and the
  recovery falls below 80% when three components overlap. (P2) The detector finds 25 to 40 negative
  segments of at least 6 hours at D8 since 2023-08, and 20 to 30 of at least 2 days (experiment 159
  found 29 with a different rule). (P3) Adding the removed level back to the training target
  lowers D8's quantile score by at least 0.2 points against the control, which reproduces 8.382.
  (P4) Masking the rows instead lowers the score by less than adding the level back does.
decision_rule: >-
  Stop rule: if the injection test recovers under 50% of injected components of 2 MW or more with
  one component at a time, stop before the forecast step and report. Otherwise two planned
  contrasts on the 96,986 rows of experiment 354, paired, month-resampled (20,000 resamples), 99%
  intervals (a Bonferroni correction for two). C1: T1 (level added back) minus the control. C2: T0
  (rows masked) minus the control. Threshold for C1: the interval wholly below zero and the point
  estimate at most -0.2 points, because the detector and the target edit are one new step per fit
  cut at D8 only. `supported` when C1 clears the threshold and P1 holds. `partly supported` when C1
  clears the threshold and P1 fails, or C2 alone clears. `small gain` when an interval lies wholly
  below zero with a point estimate above -0.2. `not supported` otherwise. Control gate: the control
  reproduces experiment 354's fit (b) with arm 3's band, 8.382, before any treatment arm is read.
benchmark: >-
  Experiment 354's fit (b) with the own-residual band by hour block and season (8.382 monthly);
  BG2, climatology, and manual_heuristic as stored by experiment 87
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles) in points of D8's effective
  capacity (1 point is about 0.1 MW); q05 to q95 coverage; share below q01. For the detector: share
  of injected components recovered, timing error, and size error
code: ml-experiments/00361-d8-step-catalogue-detector-injection-and-gam-stack/
verdict: not supported
best_arm: >-
  None. The control (experiment 354's fit (b) with arm 3's band) scores 8.382 on 96,986 rows; the
  level-added-back arm scores 8.375 and the masked arm 8.384.
headline: >-
  A plain detector recovers 75% of injected spells of 2 to 4 MW when one runs at a time (2 MW 55%,
  3 MW 83%, 4 MW 91%), and 59% when three overlap. Adding the detected levels back to D8's training
  target changes the quantile score by -0.007 points (99% interval -0.097 to +0.053), and masking
  the rows by +0.001 (-0.071 to +0.054), against the pre-registered -0.2.
lessons:
  - >-
    The detector flags 9% to 13% of D8's training rows and moves the median forecast by 0.12 MW on
    average, yet the score does not move, so the edit changes the forecast without bringing it
    closer to the readings.
  - >-
    The rule finds 171 rises of at least 1.5 MW for 193 falls, so a large share of its falls are
    ordinary demand swings, not generator spells.
  - >-
    The multi-day drops (6 found, median 2.7 MW and 2.5 days) match the maintainer's reading, but
    they cover too few rows to move a monthly fit.
production_status: not-for-production
adopted_in: none
take_home_message: >-
  A plain step detector finds injected 3 MW and 4 MW generator-like spells most of the time and 2
  MW spells about half the time. Removing the steps it finds from D8's training target does not
  improve D8's forecast, so a particle filter that separates the generator is not supported by
  this test. A detector that rejects ordinary demand swings, and a longer record of the generator's
  own output, would be the next things to try.
advice:
  - >-
    Do not remove steps found by a plain residual-median rule from D8's training target; the score
    moved by -0.007 points (99% interval -0.097 to +0.053) (C1, Figure 2).
  - >-
    Expect a residual-median rule with a 1.5 MW threshold to miss about half of injected 2 MW
    spells and to flag as many rises as falls at D8, so test any detector against injected spells
    and against rises before using its output (Figures 1 and 3).
date: 2026-10-28
---

**A plain step detector separates 3 MW and 4 MW generator-like spells from D8's demand most of the
time, but removing the steps it finds from the training target does not improve D8's forecast.**

The test does not support building a particle filter on the evidence of this detector. The next
step is a detector that rejects ordinary demand swings, since the rule flags as many rises as falls.
The verdict is `not supported`.

D8 is a primary substation (Primary) whose power seems to include a dispatchable generator that
switches on and off. We asked whether a simple rule can find such steps in D8's power, and whether
taking them out of the training target helps the forecast. We added simulated on/off spells of 2, 3,
and 4 MW to three Primaries with no known generator, D2, D3, and D4, and counted how many spells
the rule found again. We then refitted D8's generalised additive model (GAM) stack with the
detected steps removed. The quantile score is twice the mean pinball loss over 11 quantiles, in
points of D8's capacity (1 point is about 0.1 MW), and lower is better. The residual-median rule
extends experiment 159's rule for multi-day drops (see
[experiment 159](../00159-switching-d8-multi-day-drops-recurrence-and-report/result.md)). The stack,
its refits, and the own-residual band come from
[experiment 354](../00354-d8-own-residual-bands-and-longer-fit-history/result.md), and the
gas-engine background from [experiment 264](../00264-dp-gas-engine-asymmetric-bands-at-d8/result.md),
[experiment 265](../00265-dp-gas-engine-bands-prior-and-visibility-switch/result.md), and
[experiment 291](../00291-d8-drops-engine-spells-load-shift/result.md).

**Figure 1: the detector finds 4 MW spells 91% of the time and 2 MW spells 55% of the time.**

![Injection recovery](figure_1_injection_recovery.svg)

**Figure 2: neither removal arm moves the score by more than 0.1 points.**

![Planned contrasts](figure_2_contrasts.svg)

**Figure 3: the rule finds falls in every month.**

![Falls per month](figure_3_falls_per_month.svg)

## Results

- **Detector.** The baseline of each half-hour is the median of the same half-hour on the 2 to 8
  days before and after. A fall is a run of at least 6 hours in which the 6-hour centred median of
  the residual is at or below -1.5 MW, with gaps under 6 hours merged. A multi-day drop uses
  experiment 159's rule (24-hour median, gaps under 12 hours merged, at least 2 days). At a fit
  cut only data before the cut is seen. The drop pass was added after the first catalogue run
  found only 4 drops of 2 days or longer with the 6-hour rule, and before any forecast was scored.
- **Catalogue at D8 since 2023-08-15.** 193 falls of at least 6 hours (median level -3.0 MW, median
  dwell 0.38 days) and 171 rises of the same size; 6 multi-day drops (median level -2.7 MW, median
  dwell 2.5 days). At 1.0 MW the counts are 273 falls and 11 drops; at 2.0 MW, 126 falls and 1
  drop. Prediction P2 (25 to 40 falls and 20 to 30 drops) failed on both counts: experiment 159
  found 29 drops with a different baseline.
- **Injection test.** 20 draws on each of D2, D3, and D4, 2024-07 to 2025-06, with spells and gaps
  resampled from D8's catalogue. A spell is recovered when detected segments cover at least half of
  it and the segment's level is within 30% of the injected total. One component at a time: 75%
  (3,986 spells; 95% Wilson interval 73% to 76%); 2 MW 55%, 3 MW 83%, 4 MW 91%. Two at once: 70%.
  Three at once: 59%, and 43% at 2 MW. Largest overlap with at least 80% recovery: none across all
  sizes; one component for 3 MW and above. The median start error is 0.5 hours and the median
  level error is within 6% in every cell. On the uninjected hosts the same windows are recovered
  by chance 0.4% of the time, and the rule finds 29, 5, and 6 falls a year at D2, D3, and D4.
  P1 failed on its first half (75% against 80%) and held on its second.
- **Rows.** 96,986 half-hourly rows at D8 in 2025-07-02 to 2026-06-30, leads 0.5 to 168 hours
  (the stored ensemble forecasts stop at 168 hours), `fold_a_daily_gap_v1`. Intervals resample whole
  months at the 99% level, the Bonferroni level for two planned contrasts.
- **Arms.** Monthly refits (12 fits) from 2023-09 with a level step per hour band at 2025-04-01,
  arm 3's own-residual band by hour block and season, the night offset and recipe R. Control: the
  raw target. T1: each detected segment's level added back to its rows. T0: the segment's rows
  masked from the fit. The detector flags 9% to 13% of training rows at the cuts checked
  (2024-09 and 2026-06), at a mean level of about -3.0 MW. Fit code is copied from experiment 354's
  `e00354_fit.py` with the target edit; the scripts import experiments 252, 264, and 354.
- **Gate.** The control scores 8.382, as in experiment 354's fit (b).
- **Absolute scores, all rows.** Control 8.382, T1 8.375, T0 8.384, BG2 7.593, manual heuristic
  8.469, climatology 9.757. All three arms trail BG2.
- **C1, T1 minus control.** -0.007 (99% -0.097 to +0.053), so P3 (at least -0.2) failed.
- **C2, T0 minus control.** +0.001 (-0.071 to +0.054). P4 held trivially, since neither arm helped.
- **Post hoc slices, descriptive.** On sunny daylight half-hours, T1 minus control is -0.197 (-0.699
  to -0.015) and T0 minus control -0.105 (-0.526 to +0.079). On cold nights, T1 minus control is
  -0.100 (-0.206 to +0.274). These slices were not planned, so they do not change the verdict.
- **Tail calibration, arm 3 band.** q05 to q95 coverage 0.846 (control), 0.847 (T1), 0.845 (T0).
  Share of readings below q01 0.039, 0.040, 0.040. Share above q99 0.029 in all three.
- **Scope.** Dispatchable on/off components only; solar and wind follow the weather. Only the sum of
  demand and generators is observed, so components of similar size may be unidentifiable. The
  generator behind D8 is unproven (experiments 264, 265, 291). The result is for D8 at leads of 0.5
  to 168 hours, shorter than the 10 days NGED care about.
- **Not run.** A particle filter, a production port, and a mixture band with an explicit
  on-probability. The removed component re-enters through the own-residual band, which is fitted on
  the actual power.

## Checks

- **Median minus mean forecast, points of capacity.** Control -1.44, T1 -1.31, T0 -1.30.
- **Error of the mean forecast beside the quantile score.** Mean absolute error 1.505 MW (control),
  1.499 MW (T1), 1.502 MW (T0), so both measures point the same way and the difference is small.
- **Triggers.** Trigger 2 fired in part: C1 contradicts its prediction of -0.2. The explanation
  test shows the edit moved the median forecast by 1.17 points (0.12 MW) on average for T1, and by
  1.03 points for T0, so the fit changed without improving. This fits a detector whose falls are
  partly ordinary demand swings (171 rises for 193 falls). No Opus audit was run, because the
  brief excludes Opus, and no later experiment depends on this result's sign. Triggers 1, 3, 4, 5,
  6, and 7 did not fire.

## What to try next

- Reject ordinary demand swings by requiring a fall to be matched by a rise of similar size at the
  end of its dwell, then repeat the injection test with that rule.
- Test the 6 multi-day drops alone as a mask on the training target, since they are the segments
  most likely to be the generator.
