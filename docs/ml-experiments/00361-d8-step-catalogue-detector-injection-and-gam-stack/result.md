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
status: planned
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
---

**Planned, not yet run.** This page is the plan; the write-up replaces it.

## Plan

1. **Step catalogue at D8.** A plain as-of rule. For each half-hour, the baseline is the median of
   the same half-hour of the day over the 2 to 8 days before and after (at a fit cut, only the
   days before the cut exist). The residual is D8's power minus the baseline. A negative segment is
   a run in which the 6-hour centred rolling median of the residual stays at or below -1.5 MW
   (experiment 159's threshold), with gaps under 6 hours merged. A segment of at least 6 hours is a
   step, and a segment of at least 2 days is a multi-day drop. The catalogue lists each segment's
   start, end, level (the median residual in MW), and dwell, with counts per month and the dwell
   distribution. Positive segments are counted and are not removed. At each fit cut the rule sees
   only data before the cut. Thresholds 1.0 and 2.0 MW are a sensitivity.
2. **Injection test.** On D2, D3, and D4 (Primaries with no engine, experiment 354's placebos),
   subtract simulated on/off components of 2, 3, or 4 MW from the power, with on-spells and off-gaps
   resampled from D8's catalogue. Run with 1, 2, and 3 overlapping components, 20 draws each, and
   report the share recovered (a detected segment overlapping at least half of the injected
   spell, level within 30%), the start and end errors, the level errors, and the false-segment rate
   on the uninjected series. State the largest number of overlapping components with recovery
   of at least 80%. The only recovery check, since the real D8 generator has no ground truth.
3. **Effect on the forecast.** Refit D8's monthly GAM stack as in experiment 354's fit (b)
   (fit from 2023-09, level step at 2025-04-01), with arm 3's own-residual band. Control: the raw
   target. T1: each detected segment's level added back to the training target. T0: the segment's
   rows masked from the training target. The band is fitted on the residual of the actual power
   against the median, so the generator's chance of running re-enters as the lower tail. The
   removed component lowers actual power, so the widening is in the lower quantiles, not the
   upper.
4. **Reuse.** Experiment 252's stack and forecast engine, experiment 264's as-of residual code,
   and experiment 354's fit, band, and scoring code (copied, with the target edit).
5. **Scope.** Dispatchable on/off components only. Only the sum of demand and generators is
   observed, so components of similar size may be unidentifiable. The generator behind D8 is
   unproven (experiments 264, 265, 291). The result bears on D8 alone and on the 1 to 7 day lead
   range of the stored forecasts (168 hours).
