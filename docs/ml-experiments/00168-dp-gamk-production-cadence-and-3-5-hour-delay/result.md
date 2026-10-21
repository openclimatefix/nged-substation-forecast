---
id: '00168'
slug: dp-gamk-production-cadence-and-3-5-hour-delay
part: BG-S2
theme: differentiable-physics-forecast-engine
also_relevant_to:
- production-reference-and-best-guess
- lagged-power-and-error-correction
status: planned
size: >-
  simple, by the five triggers. Stored: private files under
  data/ml-experiments/00168-dp-gamk-production-cadence-and-3-5-hour-delay/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: one defensible design (re-issue the stored
  forecasts of experiments 136 and 155 at the four production origins), with the 3.5-hour delay
  removed as a labelled diagnostic arm. Callers: this folder's scripts, which copy the filter and
  T-all code of experiment 136 and call the reviewed `studies.*` modules without changing them. No
  new modelling.
hypothesis: >-
  The GAM-K forecast of experiment 155's arm D (the convex demand model with a Kalman filter on
  recent errors, the winter night offset, and recipe R) keeps its score when it is issued the way
  production would issue it. Production refreshes the forecast every 6 hours from the newest
  ECMWF run, so the weather forecast is 12 to 30 hours old, and the filter reads only power
  readings that are at least 3.5 hours old. Experiments 136 and 155 issued one forecast a day, 9
  hours after the run. The mechanism is that a fresher filter state makes up for an older weather
  forecast, so the score at a given lead changes little.
prediction: >-
  Written before any score is read. C1 (arm D issued at the four production origins, minus arm D
  issued once a day, at the same lead and run, pooled over leads 0.5 to 24 hours) is +0.00 to +0.03
  points of effective capacity. The older weather forecast costs little at short leads, and the
  filter gains nothing from the later origins at the same lead. C3 (arm D at the production origins
  with the 3.5-hour delay, minus the same forecasts with no delay) is +0.01 to +0.04 pooled at
  leads under 24 hours, and larger than +0.04 at leads under 3.5 hours. C4 (production origins
  minus once-a-day origins at the same valid time) is -0.02 to -0.06 at leads under 24 hours,
  because the later origin has a shorter lead and a fresher filter. The 06:00 origin, whose
  weather forecast is 30 hours old, has the largest C1 of the four origins. On cold nights (daily
  mean temperature below 2 degrees Celsius) C1 is at most +0.10.
decision_rule: >-
  The headline contrast is C1, at the five decision substations D5, D10, D11, D14, and D16 in the
  scored year, pooled over the four production origins and over leads 0.5 to 24 hours (D9 beside
  them in the tables and in no contrast). Each contrast is read at the 99% level of the
  month-resampled paired bootstrap (the 12 calendar months of the scored year, 20,000 resamples).
  The hypothesis is `supported` when C1's point estimate is below +0.05 points and its interval's
  upper bound is below +0.05. It is `partly supported` when the point estimate is below +0.05 and
  the upper bound is not. It is `refuted` when the point estimate is +0.05 or more. The threshold
  of +0.05 is set by cost: it is larger than the 0.037 points that recipe R gained in experiment
  155, so a delay cost above it would remove the gain of the stage 2 corrections, and a cost below
  it is within what the production cadence can carry without a redesign. Where C1 is +0.05 or more,
  the write-up says which origin, lead band, and subset carry it. The lead check of experiment 155
  is repeated: the score at leads of 3.5 to 9.5 hours minus the score at 0.5 to 3.5 hours, for the
  production origins minus the same difference for once-a-day origins, is reported with a 99%
  interval, and passes when its upper bound is at most +0.02 points. The verdict is capped at
  `inconclusive` if the reproduction control fails (an origin shift of zero hours must reproduce
  the stored means to within 0.001 points and the stored quantiles to within 0.001 points).
benchmark: >-
  Arm D and arm A of experiment 155, as stored, issued once a day (2.756 and 2.791 points over
  leads 0.5 to 168 hours at the five decision substations in the scored year)
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over the 11 quantiles of experiment 87) in points of
  effective capacity, on rows outside the hand-filtered switching events, by lead from the
  forecast's own origin, over leads 0.5 to 147 hours at all four origins (leads up to 165 hours at
  fewer origins), then under 24 hours, by origin, and on cold nights; coverage and pinball loss at
  the 0.01, 0.05, 0.95, and 0.99 quantiles
code: ml-experiments/00168-dp-gamk-production-cadence-and-3-5-hour-delay/
---

# Experiment 168: GAM-K at the production cadence with the 3.5-hour delay

**Planned.** This page holds the plan only. The write-up replaces it when the experiment finishes.

**Question.** Experiments 136 and 155 issued one forecast a day, 9 hours after the 00:00 UTC
ECMWF run. Production would issue one every 6 hours from the newest run, so the weather forecast is
12 to 30 hours old, and the Kalman filter would read only power readings at least 3.5 hours old.
Does the stage 2 forecast (arm D of [experiment
155](../00155-dp-bg-stage2-assembly-night-offset-and-error-correction/result.md)) lose more than
0.05 points of score at leads under 24 hours when issued that way?

**Method.** Each stored run is re-issued at 12:00, 18:00, 00:00, and 06:00 UTC, which is 3, 9, 15,
and 21 hours after the original origin. The weather input is the same run's ensemble mean and the
GAM fit is the same, so only the filter's state (read 3.5 hours before the new origin) and the lead
change. The filter, the shift of arm D, and the spread table come from [experiment
136](../00136-dp-bg-stage1-assembly-on-the-ensemble-mean/result.md) and experiment 155.
