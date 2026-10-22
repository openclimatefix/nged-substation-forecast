---
id: '00175'
slug: dp-unlabelled-day-shapes
part: BG-EB3
title: >-
  What shape does the forecast's error take on the unlabelled worst days: a sharp-edged block, a
  smooth offset, a ramp, or a spike?
theme: differentiable-physics-forecast-engine
also_relevant_to:
- switching-detectors
- switching-labels-in-the-forecast
status: planned
size: >-
  medium. Stored: private tables under data/ml-experiments/00175-dp-unlabelled-day-shapes/ only.
  Serving path: not touched. Degradation rule: not touched. Designs: more than one is defensible
  (the step test, the spike test, the window of the neighbour check), so this plan fixes each rule
  in numbers before any day is looked at. Callers: this folder's scripts, which read experiment
  174's day tables, experiment 155's arm D forecasts as copied into experiment 174's folder,
  experiment 87's BG2 forecasts, experiment 147's and experiment 156's stored detector intervals, and
  NGED's interconnection table, and call reviewed `studies.*` modules without changing them. No
  forecast model is fitted.
hypothesis: >-
  At least half of the 37 unlabelled worst substation-days of experiment 174 are sharp-edged blocks
  in the forecast's error (a switching-like change in the power a substation draws), and in at
  least a third of those a substation interconnected with the first shows a step of opposite sign
  within 2 hours (a load transfer between the two).
prediction: >-
  Written before any day is looked at. Of the 37 days, 19 to 28 are sharp-edged blocks (class a),
  5 to 10 are smooth offsets (class b), 2 to 6 are ramps or midday-only errors (class c), and 0 to 3
  are spikes (class d). Of the class (a) days with a metered neighbour, 30% to 60% have an
  opposite-signed neighbour step within 2 hours, against a chance rate of 10% to 25% from shifting
  the neighbour's residuals by whole days. The detector of experiments 141 and 147 fires on 25% to
  50% of the class (a) days, and the step persists over the following 7 days on 15% to 35%.
decision_rule: >-
  Both halves are read from counts with 95% Wilson intervals. The first half passes when class (a)
  holds at least 19 of the 37 days (at least half). The second half passes when at least a third of
  the class (a) days that have a metered neighbour show an opposite-signed neighbour step within 2
  hours. `supported` when both pass, `partly supported` when exactly one passes, `refuted` when
  neither does. If the first half fails, the take-home message is that the lever is the Kalman
  filter's level adaptation, not switching detection. If the first half passes and the second
  fails, it is that the steps are real but the interconnection table cannot explain them, a direct
  case for a latent normal-running-arrangement node. The second half is read next to its chance
  rate from the shifted-neighbour control: the second half also needs the observed rate to exceed
  the mean chance rate by at least 10 points, otherwise it counts as failed (an observed rate near
  chance is evidence against a load transfer). The second half is "not assessed" when fewer than 10
  class (a) days have a metered neighbour, and the verdict is then `partly supported` or `refuted`
  on the first half alone. The first half is also read against the class (a) rate of ordinary days
  (below): if ordinary days reach class (a) as often as the unlabelled worst days, class (a) says
  nothing about worst days and the verdict is `inconclusive`. If the first half fails, the
  take-home message names the Kalman filter's level adaptation only when class (b) holds more days
  than any other class; otherwise it says the shape is mixed.
benchmark: >-
  Ordinary days (for each substation, the days whose excess loss lies in the middle half of its
  scored days, 40 drawn at random per substation) given the same classifier; a known-answer control
  that injects a block, an offset, a ramp, and a spike into ordinary days; and a shifted-neighbour
  control (each neighbour's residual series rolled by a whole number of days, separately per
  neighbour, 1,000 draws) for the opposite-signed-step rate. Experiment 172's known-answer control
  (89.2% of 10-point injections found in the worst 5%, under its 90% bar) is quoted, so that every
  share here is read as approximate
row_set_id: fold_a_daily_gap_v1
metric: >-
  Counts of days per shape class, and the share of class (a) days with a neighbour step of
  opposite sign. Residual is observed minus forecast of experiment 155's arm D (ensemble-weather
  forecast), D+1 band (lead 15 to 39 hours), outside the hand-filtered switching masks,
  2025-07-03 to 2026-06-30, as in experiments 172 and 174
code: ml-experiments/00175-dp-unlabelled-day-shapes/
date: 2026-10-22
---

**Planned; the write-up replaces this page when the run finishes.**
[Experiment 174](../00174-dp-unlabelled-worst-days-weather-shared-or-single-site/result.md) left
37 worst substation-days whose loss is mostly not weather error and not shared with other Primaries.
This experiment asks what the forecast's error looks like on each of those days, so that the next
experiment can target the right lever. It is descriptive: it fits no new model.

## Rules fixed before any day is looked at

The residual of a day is arm D's observed minus forecast value at each half-hour of the D+1 band
(lead 15 to 39 hours). Arm D is experiment 155's convex demand model with a Kalman filter, driven
by the ensemble-mean weather forecast. The residual series is continuous across midnight, so a
window can cross a day boundary, and a window counts clock positions (half-hours), so a window
holds fewer valid readings where rows are missing. Missing half-hours (the hand-filtered switching
rows) are skipped, not filled. The **day-before spread** `s` is 1.4826 times the median absolute
deviation of the previous local day's residual (at least 24 valid readings), floored at the 10th
percentile of that substation's daily spreads over the scored year; when the previous day has fewer
than 24 valid readings, the nearest earlier day within 7 days that has enough is used, then the
floor. A day is classified by the first rule that holds, in the order a, d, b, c; a day matching
none is "mixed".

- **Edge test.** At half-hour `t`, the jump is the median of the 2 readings from `t` minus the
  median of the 2 readings before `t` (at least 1 valid reading each side), and the shift is the
  same difference for 8 readings each side (at least 4 valid each side). An edge needs `|jump| > 3
  s`, `|shift| > 3 s`, the same sign for both, and `|jump| >= |shift| / 2` (the step is sharp, not
  a ramp). Consecutive qualifying half-hours within 1 hour merge into the one with the largest
  `|shift|`. The day's primary edge is the merged edge with the largest `|shift| / s`.
- **(a) Sharp-edged block.** The day has at least one edge. The block "returns" when a later edge in
  the same local day has the opposite sign and `|shift|` at least half the primary edge's;
  otherwise it "stays shifted". A block that starts before the day's first half-hour is not found,
  and a block that straddles midnight is counted as returning only if the return falls in the same
  local day as the primary edge.
- **(d) Spike.** Not (a), and the number of readings in the whole day lying more than 4 s from the
  day's median residual is between 1 and 3, and those readings account for at least half of the
  day's summed absolute deviation from that median.
- **(b) Smooth same-sign offset.** Not (a) or (d), the day's median residual exceeds 1 s in
  magnitude, and at least 75% of the day's valid readings share its sign.
- **(c) Ramp or midday-only error.** Not (a), (d), or (b), and either the readings from 10:00 to
  16:00 local time account for at least half of the day's summed squared residual, or the day's
  residual after a 6-reading rolling median never moves against its overall direction by more than
  0.5 s between neighbouring readings and its range exceeds 2 s. The two parts are reported
  separately within class (c).
- **Neighbour check, class (a) days only.** The neighbours are the Primaries joined to the
  substation's Primary by NGED's interconnection table that have a forecast from BG2 (the
  production-style XGBoost blend of experiment 87), using its median forecast. A neighbour's
  residual is observed minus that median, with the neighbour's own day-before spread `s_n`. A
  neighbour shows an opposite-signed step when its 8-reading shift, at any half-hour within 2 hours
  of any edge of the substation's day, exceeds 2 `s_n` with the sign opposite to that edge's shift.
  A class (a) day with no neighbour that has a BG2 forecast is left out of the second half's
  denominator. The chance rate rolls each neighbour's residual and spread series by a random whole
  number of days from 1 to 90 (circular, separately per neighbour), 1,000 draws, and compares the
  observed rate with the mean over draws. The same rate is also computed on ordinary days that
  have an edge, as a second null.
- **Detectors.** The detector "fired" on a class (a) day when a stored onset for that series lies
  within 3 hours either side of the primary edge. Three tables are read: the refit-offset
  neighbour-count detector (RO-NC, called XL in experiment 82; the detector of experiments 141
  and 147) with the causal-live end rule R2 and 48-hour merging, the same detector with the
  hindsight end rule R0 and 48-hour merging (both from experiment 147), and the pair alarm of
  experiment 156 (4-hour window). The share of ordinary days with an onset within 3 hours of
  any half-hour is reported as the base rate.
- **Persistence.** The step "persists" when the median of the daily median residuals over the 7
  local days after the primary edge's day, minus the same median over the 7 days before it, has the
  sign of the step and at least half its size. The days need at least 24 valid readings each, and
  at least 3 of the 7 on each side must qualify, otherwise the result is "unknown". The residual
  comes from a forecast whose Kalman filter adapts to a level shift, so persistence of the real
  step is understated.
- **Ordinary days.** For each of the five substations, 40 days drawn at random (seed 175) from the
  days whose excess loss is between the 25th and 75th percentile of its scored days, with at least
  24 valid readings.
- **Known-answer control.** A synthetic error is added to the arm D residual of 40 ordinary days
  per class: a block of 3 `s` lasting 6 hours with a sharp edge (class a), a constant offset of 2
  `s` over the day (b), a linear ramp from 0 to 4 `s` over the day (c), and a spike of 3 readings
  of 6 `s` (d). The control reports the share of injected days classified as intended; the plan
  does not set a bar, and the write-up reads the real counts as approximate by the recovery rate.

## Outputs

- Class counts with Wilson intervals, the same split by substation, and the same split for
  ordinary days.
- The neighbour-step share next to the shifted-neighbour chance rate and the ordinary-day rate.
- Detector, pair-alarm, return, and persistence tallies for the class (a) days.
- A compact multi-panel SVG set, one panel per day for all 37 days (residual, hand-filtered
  readings, detector intervals, and the neighbours' residual), using D-labels only.
- Experiment 172's control recovery (89.2%), quoted so that all shares read as approximate.

## Audit and sizing

- **Audit triggers:** the known-answer control above can fail. Trigger 2 (a result contradicting
  its own prediction or an earlier experiment, without an explanation tested in the write-up)
  fires if class (a) holds fewer than 19 days, because experiments 141 and 147 expected switching
  to be common; the session records an explanation tested in the write-up, and asks for an audit
  only if the shifted-neighbour rate contradicts the observed rate.
- **Reviews:** one Sonnet plan review (done), one Sonnet code review, one Sonnet review of the
  write-up against the printed report, one Sonnet prose review. No Opus.
