---
id: "00200"
slug: dp-pooled-correction-head-on-kalman-state
title: "Differentiable physics: a pooled learned correction head on arm D's forecast and recent errors"
theme: differentiable-physics-forecast-engine
also_relevant_to: [lagged-power-and-error-correction, calibration-and-uncertainty, differentiable-physics-pooling-across-substations]
status: done
size: >-
  medium. Stored: private files under data/ml-experiments/00200-dp-pooled-correction-head-on-kalman-state/
  only. Serving path: not touched. Degradation rule: not touched. Designs: several are defensible
  (network type, which state inputs, training window), and this plan pre-registers one. Callers: this
  folder's scripts, which copy experiment 142's network and gate code and read the shared arm D
  forecasts. A new method, so a Sonnet plan review and a Sonnet code review; no Opus.
hypothesis: >-
  Experiment 142's small network failed its known-answer gate with about 1,100 samples from one year
  at five substations. The same kind of network, trained on rows pooled across many Primary
  substations and years, learns to correct arm D's forecast from the recent errors, the calendar, the
  lead, and the ensemble-mean weather, and so beats recipe R (subtract 0.75 times the mean error of
  the last 14 days) at day D+1.
prediction: >-
  Written before any fit. Stage 1 (gate, 6 stored substations, 2024-07 to 2026-06): the network
  recovers 10% to 40% of an injected +3-point 8-week step (gate: at least 30%). Stage 2 (pooled rows,
  scored after the shared forecasts exist): the head minus recipe R at D+1 is -0.04 to +0.02 points;
  the likeliest outcome is a null. Tail pinball at q01 and q99 improves by less than 3%.
decision_rule: >-
  Stage 1 gate: at least 30% of an injected +3-point step recovered from the third day, and at least
  15% of a +3-point Saturday-evening shift (experiment 142's gates), on held-out blocks. If the gate
  fails, the neural error corrector leaves the queue, because recipe R and the filter do the job.
  Stage 2: supported if head minus recipe R at D+1 in the scored year is at most -0.02 points with
  the 99% month-resampled paired interval excluding zero. The threshold is 0.02 because the head is a
  new trained component on the serving path, which a smaller gain does not pay for. If it passes,
  compare with the level-adaptation arms of experiment 175's follow-up on the same rows.
benchmark: "Recipe R on arm D (experiment 155), 2.436 points at D+1 on the five decision substations"
row_set_id: "to be fixed once the shared arm D forecasts exist; stage 1 uses experiment 155's stored rows"
metric: "Quantile score over the 11 quantiles in points of effective capacity, D+1, outside switching events"
date: 2026-10-22
verdict: partly supported
best_arm: >-
  head minus arm D -0.072 points (99% interval -0.156 to -0.016) at D+1; the blind head (no error inputs) gets -0.062
production_status: not-for-production
adopted_in: none
headline: >-
  Gate 29% of an injected 3-point step (gate 30%, recipe R 67%) and 21% of a Saturday-evening shift (gate 15%). On the scored year the head beats arm D by 0.072 points, but a head given no error inputs gets 0.062 of that.
take_home_message: >-
  A small neural network trained on one year of arm D's own errors lowers the day-ahead score by 0.07 points at five substations, but nearly all of that gain comes from the calendar, lead and substation alone, not from reading recent errors. So the network does not do the Kalman filter's job better; it corrects a systematic bias in arm D that a simple recalibration could also remove.
lessons:
  - >-
    Reading recent errors added -0.011 points (interval -0.106 to +0.045) beyond a head with the error inputs zeroed.
  - >-
    The head recovers a third of an injected step against two thirds for recipe R, so it is a weaker adapter than the hand-built profile.
advice:
  - >-
    Test a plain calendar, lead and substation recalibration of arm D's median before another neural error corrector (blind-head control, C2).
  - >-
    Give any learned corrector a same-weekday error input; without it a Saturday-evening shift is not recovered (gate, -1.5% against 21%).
  - >-
    Rerun a learned corrector on the 16-Primary forecasts of the sibling experiment before concluding it cannot read errors (stage 2 uses 6 substations and one training year).
code: ml-experiments/00200-dp-pooled-correction-head-on-kalman-state/
---

**Take-home: a small neural network lowers arm D's day-ahead score by 0.07 points at five
substations, but a copy of it that cannot see any recent error gets 0.06 of that.** Arm D is our
convex demand model with a Kalman filter (a method that updates a forecast from its recent errors),
the night offset, and recipe R (subtract 0.75 times the mean error of the last 14 days). We trained
one network, shared by 6 substations, to correct arm D's median and rescale its spread. Most of the
gain therefore comes from the calendar, the lead time, and the substation, which fix a bias in arm D
without reading the errors. Reading the errors adds -0.011 points (99% interval -0.106 to +0.045),
which is not distinguishable from zero. This is a weaker test than the brief planned: it uses the
6 substations with stored forecasts and one training year, not 16 Primaries since 2019, and it has no
weather input. The verdict is `partly supported`, because the headline contrast passes its rule
but the known-answer gate does not (29% against 30%) and the mechanism is not supported.

## Figures

![Contrasts](figure_1_contrasts.svg)

![Gate](figure_2_gate.svg)

![Sites](figure_3_sites.svg)

## Results

- **Method.** One network (two hidden layers of 128 units, 4-dimensional substation embedding, a
  linear skip) trained with the pinball loss at the 11 quantiles on arm D's rows at leads 12 to 48
  hours. Quantiles are `median + correction + exp(scale) x (arm D quantile - arm D median)`, so they
  cannot cross. Inputs: the last 12 half-hours of errors known at the origin (3.5 hours old), the
  mean error of the last 14 days at the row's half-hour, the mean error at the same weekday and
  half-hour in the last 4 weeks, the lead, half-hour, day of week, and arm D's mean. The network
  follows the pooled-network design of [experiment
  142](../00142-differentiable-physics-sequence-model-on-gam-k-residuals/result.md) and uses the
  pinball loss of [Koenker and Bassett (1978)](https://doi.org/10.2307/1913643).
- **Rows.** 287,000 rows (site, run, valid time) from 3,900 site-runs, against about 3,200 samples
  in experiment 142: about 90 times the rows but only 1.2 times the independent history. Training
  for stage 2 uses 125,000 rows from runs before 2025-06-28.
- **Gate (6 substations, 3 seeds, an 8-week injection on held-out blocks).** Level step of +3
  points: head 29% (22% to 34%, gate 30%), recipe R 67%. Saturday-evening shift of +3 points: head
  21% (20% to 23%, gate 15%), recipe R 11%. A first version without the same-weekday input
  recovered -1.5% of the shift. The input was added before any scored read.
- **Scored year, five decision substations, leads 15.5 to 39 hours, outside switching events.**
  Arm D 2.432, head 2.360, blind head 2.370 points (C1 head minus arm D -0.072, interval -0.156 to
  -0.016; C2 blind minus arm D -0.062, -0.096 to -0.032; C3 head minus blind -0.011, -0.106 to
  +0.045). Experiment 169's calibrated-spread arm D scores 2.430 on the same rows, so spread
  calibration is not the source.
- **Per substation, head minus arm D:** D5 -0.086, D10 -0.069, D11 +0.041, D14 -0.116, D16 -0.137,
  D9 +0.025.
- **Tails (decision substations).** Pinball at q99: arm D 0.411, head 0.391. At q01: 0.386, 0.375.
  Coverage at q95: arm D 94.5%, head 93.9%; q99: 98.1%, 97.8%. The head gives no tail calibration
  gain.
- **Caveats.** No ensemble weather input (the stored forecasts hold none), a single training year
  (2024-07 to 2025-06), 6 substations, and a head tuned on nothing. The gate is on one 8-week block
  per injection. The blind head is trained on the same year, so its gain is a year-specific bias
  correction that may not hold in a different year.

## Checks

- **Median minus mean forecast per site (head):** D10 -0.55, D11 +0.15, D14 -0.23, D16 +0.08, D5
  -0.02, D9 -0.03 points.
- **Error of the median forecast beside the score:** mean absolute error arm D 4.191, head 4.091,
  blind head 4.121; the contrasts point the same way.
- **Triggers that fired:** (1) the known-answer gate marginally failed, 29% against 30%; (5) the
  blind head, nested in the head, scores almost as well; (7) the sibling experiment's design may
  depend on this result. Leaving one substation out never flips the sign of C1 (range -0.039 to
  -0.076). An audit by a fresh Opus agent has been requested from the ML manager and has not run.

## What to try next

- Replace the network with a calendar, lead, and substation recalibration of arm D's median and see
  whether it matches the blind head.
- Rerun on the sibling session's forecasts for 16 Primaries before 2025-07, which give more than
  one training year, with the ensemble-mean weather inputs.
