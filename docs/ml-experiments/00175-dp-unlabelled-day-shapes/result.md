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
  model is fitted.
hypothesis: >-
  At least half of the 37 unlabelled worst substation-days of experiment 174 are sharp-edged blocks
  in the forecast's error (a switching-like change in the power a substation draws), and in at
  least a third of those a substation interconnected with the first shows a step of opposite sign
  within 2 hours (a load transfer between the two).
prediction: >-
  Written before any day is looked at. 19 to 28 of the 37 days are sharp-edged blocks (class a),
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
  rate from the shifted-neighbour control, and the verdict is capped at `inconclusive` if the
  shifted-neighbour rate is within 10 points of the observed rate and the observed rate is under a
  half.
benchmark: >-
  Experiment 172's known-answer control (89.2% of 10-point injections found in the worst 5%, under
  its 90% bar), so every share here is approximate; and a shifted-neighbour control (the
  neighbours' residual series moved by whole days, 1,000 draws) for the opposite-signed-step rate
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

The residual of a day is arm D's observed minus forecast value at each half-hour of the D+1 band.
The residual series is continuous across midnight, so a window can cross a day boundary. Missing
half-hours (the hand-filtered switching rows) are skipped, not filled. The **day-before spread** `s`
is 1.4826 times the median absolute deviation of the previous local day's residual, floored at the
10th percentile of that substation's daily spreads over the scored year. A day is classified by the
first rule that holds, in the order a, d, b, c; a day matching none is "mixed".

- **(a) Sharp-edged block.** Some half-hour `t` in the day has both `|median(r[t..t+1]) -
  median(r[t-2..t-1])| > 3 s` (the jump within 1 hour) and `|median(r[t..t+7]) - median(r[t-8..t-1])|
  > 3 s` (the shift lasts at least 4 hours, so a spike of 3 readings or fewer cannot pass). The
  block "returns" if an opposite edge of at least half the size follows within the same local day,
  and otherwise "stays shifted". Each side of both windows needs at least 4 valid readings.
- **(d) Spike.** Not (a), and 1 to 3 readings lie more than 4 s from the day's median residual,
  contributing at least half of the day's summed absolute deviation from that median.
- **(b) Smooth same-sign offset.** Not (a) or (d), the day's median residual exceeds 1 s in
  magnitude, and at least 75% of the day's valid readings share its sign.
- **(c) Ramp or midday-only error.** Not (a), (d), or (b), and the readings from 10:00 to 16:00
  local time carry at least half of the day's summed squared residual, or the day's residual
  crosses zero once with a slope of the same sign throughout the day.
- **Neighbour check, class (a) days only.** The neighbours are the Primaries joined to the
  substation's Primary by NGED's interconnection table that have a BG2 forecast (experiment 87,
  median forecast `q50`). Their residual is observed minus `q50`, with their own day-before spread
  `s_n`. A neighbour shows an opposite-signed step when the step statistic above, computed on its
  residual, exceeds 2 `s_n` with the sign opposite to the substation's step, at a time within 2
  hours of the substation's edge. A class (a) day with no metered neighbour is left out of the
  second half's denominator. The chance rate moves each neighbour's residual by a random whole number
  of days from 1 to 90, 1,000 draws.
- **Detectors.** The detector "fired" on a class (a) day when a stored interval of the
  refit-offset neighbour-count detector (RO-NC, called XL in experiment 82; experiments 141 and 147)
  has its onset within 3 hours of the edge. Two readings are reported: the causal-live rule R2 with
  48-hour merging, and the hindsight rule R0 with the same merging, both from experiment 147. The
  pair alarm of experiment 156 (4-hour window, `chosen_k` 4.5) is reported the same way.
- **Persistence.** The step "persists" when the median of the daily median residuals over the 7
  local days after the edge's day, minus the same median over the 7 days before it, has the sign
  of the step and at least half its size; otherwise it "reverts".

## Outputs

- Class counts with Wilson intervals, and the same split by substation.
- The neighbour-step share next to the shifted-neighbour chance rate.
- Detector, pair-alarm, and persistence tallies for the class (a) days.
- A compact multi-panel SVG set, one panel per day for all 37 days (residual, hand-filtered
  readings, detector intervals, and the neighbours' residual), using D-labels only.
- Experiment 172's control recovery (89.2%), quoted so that all shares read as approximate.

## Audit and sizing

- **Audit triggers:** a control cannot fail here, since the run injects nothing. Trigger 2 fires if
  class (a) holds fewer than 19 days, because experiments 141 and 147 expected switching to be
  common; the session records an explanation tested in the write-up, and asks for an audit only if
  the shifted-neighbour rate contradicts the observed rate.
- **Reviews:** one Sonnet plan review, one Sonnet code review, one Sonnet review of the write-up
  against the printed report, one Sonnet prose review. No Opus.
