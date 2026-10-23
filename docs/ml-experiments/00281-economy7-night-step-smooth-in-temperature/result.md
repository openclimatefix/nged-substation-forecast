---
id: '00281'
slug: economy7-night-step-smooth-in-temperature
title: >-
  Does an Economy 7 night step that grows with the cold lower the forecast error on winter nights
  when the weather is an ensemble forecast?
theme: differentiable-physics-demand-calendar-and-weather
also_relevant_to:
  - production-reference-and-best-guess
  - calibration-and-uncertainty
status: planned
size: >-
  medium. Stored: private files under
  data/ml-experiments/00281-economy7-night-step-smooth-in-temperature/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: more than one is defensible (free night levels
  alone, or levels plus a step whose amplitude is a monotone function of temperature, with the
  step start fixed at the card's value or fitted), and this plan pre-registers both arms and the
  fixed start. Callers: this folder's scripts, which copy experiment 136's fitting code and
  experiment 155's winter night offset and call reviewed `studies.*` modules without changing them.
  No download, no GPU for the first three arms.
hypothesis: >-
  A forecast of primary substation demand whose night level is a set of free levels for each UTC
  night half-hour (experiment 121's NS-U), and then a step whose size is a monotone function of the
  cold, scores better in the Economy 7 step hours and on cold winter nights than the same forecast
  with only the winter night offset (experiment 155), when the weather input is the ensemble-mean
  forecast and not past weather. Economy 7 is the tariff that switches storage heaters on shortly
  after 00:30 UTC. Experiment 121 found a gain of 0.233 points on winter nights on past weather and
  asked for a test on weather forecasts.
prediction: >-
  Written before any score is read. (C1) A1 minus A0 is -0.05 to -0.15 points on step-hour rows
  and -0.02 to -0.06 on all winter-night rows; ensemble weather is noisier than the 0.233 found on
  past weather. (C2) A2 minus A1 is -0.01 to -0.04 on cold-night rows, with a step amplitude that
  is larger below 3 C at 4 of 5 substations. (C3) A2 minus A0 is the sum of the two within 0.02.
  The cold-night q95 coverage rises by 0 to 3 percentage points over A0. A3 (a small neural
  network for the ramp shape) is not run unless A1 or A2 meets the decision rule; its prediction is
  A3 minus A2 within +/-0.01. On the question from experiment 266: at D10 the night gap of +0.776
  points in April to June is not explained by the Economy 7 term, because the step amplitude is
  close to zero in warm months and the stored residual is flat across the night half-hours (about
  -2 points from 21:30 to 09:00 UTC) and not a step at 00:30; A2 minus A0 on D10's April to June
  night rows is within +/-0.1 points.
decision_rule: >-
  Contrasts C1 (A1 minus A0), C2 (A2 minus A1), and C3 (A2 minus A0) are read at the 99% level of
  the paired bootstrap that resamples whole calendar months of the scored year, on the five
  decision substations D5, D10, D11, D14, and D16, with D9 in the tables only. The hypothesis is
  `supported` when A1 or A2 has a step-hour interval wholly below -0.02 points against A0, its
  all-row interval's upper bound is at most +0.01, and it improves the step-hour score at 4 of 5
  substations. The threshold of -0.02 is set by cost: the step adds a fitted time and amplitude
  per substation to the stored model. The verdict is capped at `inconclusive` when the start of
  the step, fitted on the 24 months before the scored year at each substation, differs from the
  card's start of 00:30 UTC by more than 30 minutes at 2 or more of the 5 substations (experiment
  121's unresolved caveat). It is `partly supported` when an interval lies wholly below zero but
  one of the other conditions fails, and `refuted` when the step-hour interval lies wholly above
  zero. Step-hour rows and winter-night rows are reported separately. The replicate year (2024-07
  to 2025-06, even months) must show the same sign as the scored year for `supported`.
benchmark: >-
  A0, experiment 155's arm B (experiment 136's arm A2 plus the winter night offset), rebuilt here
  from a refit that matches 136's stored forecasts exactly, plus BG2 (experiment 87) and A0 without
  the offset as references.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles, in points of a substation's rated
  demand) at D+1 (15 to 39 hours) and over all leads to 168 hours; mean absolute error of the mean
  on step-hour rows; cold-night q95 coverage; tail pinball loss at q01, q05, q95, q99. Outside the
  hand-filtered load shifts.
code: ml-experiments/00281-economy7-night-step-smooth-in-temperature/
---

**Plan only: no arm has been scored.** The code to fit and forecast the arms had been run (the
control reproduced experiment 136's stored forecasts exactly) before this page was committed, but
no quantile score of arm A1 or A2 had been computed or read.

## Question

[Experiment 121](../00121-differentiable-physics-mh-multi-hinge-heating-curve-and-night-step/result.md)
added free night levels on the UTC clock to GAM-K and lowered the winter-night score by 0.233
points, but only on past weather.
[Experiment 137](../00137-differentiable-physics-e7-economy-7-night-step-term-in-the-forecast/result.md)
and [experiment 155](../00155-dp-bg-stage2-assembly-night-offset-and-error-correction/result.md)
found that a winter night offset learned from the forecast's own past errors does as well as a
temperature-dependent term. This experiment asks whether the night step, fitted inside the model
and given a size that grows with the cold, beats that offset on ensemble weather.
[Experiment 123](../00123-differentiable-physics-e7-storage-heating-night-step-estimation/result.md)
put the start of the step at the half-hour starting 00:30 UTC at all 14 series.

## Arms

- **A0.** Experiment 136's A2 (the heating split by day type, one filter noise setting, the pinned
  solar capacity), plus the winter night offset of experiment 155. Rebuilt from a refit.
- **A1.** A0's model with 20 free night levels, one for each half-hour starting 22:00 to 07:30 UTC,
  added to the fit; the winter night offset is then applied to A1's own errors.
- **A2.** A1 with four hinge terms, `max(0, knot - T)` at knots 3, 7, 11, and 15.5 C of the 24-hour
  smoothed temperature, each multiplied by the step window (00:30 to 08:00 UTC) and constrained to
  a non-negative coefficient, so the step is a monotone piecewise-linear function of the cold.
- **A3.** Not run unless A1 or A2 meets the decision rule: a small neural network for the ramp
  shape (the start offset and the width within the hour).

The raw arms without the offset (K0, E1, E2) are reported as references.

## Rows

- **Step-hour rows.** The half-hours starting 00:30 to 02:00 UTC in December to February.
- **Winter-night rows and cold-night rows.** Experiment 155's definitions (winter nights in
  November to March; cold nights with a daily mean temperature below 2 C).
- **Row set.** `fold_a_daily_gap_v1`, 292 runs of 2025-07-02 to 2026-06-29, with the tuning year
  (odd months of 2024-07 to 2025-06) fitting each arm's spread and the replicate year (even months)
  giving the sign.

## Audits and the experiment 266 question

- **Start audit.** At each decision substation, refit A2 on the 24 months before 2025-07-01 with the
  step window starting at each half-hour from 23:30 to 02:00 UTC, and take the start with the lowest
  training loss. Compare it with 00:30.
- **D10 night gap.** From experiment 266's stored forecasts, print the night residual of the
  stack and of BG2 by half-hour at D10 in April to June, and the A0, A1, and A2 scores on D10's
  April to June night rows. This cut was chosen after reading experiment 266's stored forecasts.

## Plan notes

- Compute: CPU only, 4 cores, the fits take under 10 minutes and the forecasts under 5.
- Weather: the ensemble-mean forecast at forecast time, as in experiment 136. The weather in the
  start audit is past weather, because the audit fits on history.
- Retraining schedule: monthly refits with a 24-month window, as experiment 136's A2. The contrast
  could depend on the schedule, which weekly refits would test.
