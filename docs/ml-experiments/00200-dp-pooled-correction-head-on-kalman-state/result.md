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
benchmark: >-
  Arm D of experiment 155 (it includes recipe R), 2.432 points at D+1 on the five decision substations
row_set_id: >-
  experiment 155 arm D rows of the scored year, 6 substations, leads 15.5 to 39 hours (stage 3 reads the same rows from experiment 201's shared file)
metric: >-
  Quantile score over the 11 quantiles in points of effective capacity, D+1, outside switching events
date: 2026-10-22
verdict: partly supported
best_arm: >-
  head minus arm D -0.125 points (99% interval -0.209 to -0.060) at D+1 with 2020-onward training; the
  blind head (no error inputs) gets -0.100
production_status: not-for-production
adopted_in: none
headline: >-
  Gate 29% of an injected 3-point step (gate 30%, recipe R 67%) and 21% of a Saturday-evening shift
  (gate 15%). Scored year: the head beats arm D by 0.071 points trained on one year and 0.125 points
  trained on 2020 onward, but a head given no error inputs gets 0.054 and 0.100 of that.
take_home_message: >-
  A small neural network trained on arm D's own past errors lowers the day-ahead score by 0.07 to
  0.12 points at five substations, but a copy that cannot see any recent error gets four fifths of
  that gain. So the network mostly corrects a systematic bias in arm D by calendar, lead, and
  substation, and it does not read recent errors better than the Kalman filter and recipe R do.
lessons:
  - >-
    Reading recent errors and the Kalman state added -0.017 points (one year) and -0.025 points (2020
    onward) beyond a head with those inputs zeroed; both intervals include zero.
  - >-
    Four times the training rows lifted the headline gain from 0.071 to 0.125 points and every
    substation gained, so the pooled head does learn something; the gate still marginally failed.
advice:
  - >-
    Test a plain calendar, lead, and substation recalibration of arm D's median before another neural
    error corrector (blind-head control, C2).
  - >-
    Give any learned corrector a same-weekday error input; without it a Saturday-evening shift is not
    recovered (gate: -1.5% against 21%).
  - >-
    Train a learned corrector on the 2020-onward past-weather rows as well as the ensemble year; it
    doubled the gain over one year (C1, stage 3).
code: ml-experiments/00200-dp-pooled-correction-head-on-kalman-state/
---

**Take-home: a small neural network lowers arm D's day-ahead score by 0.07 to 0.12 points at five
substations, but a copy of it that cannot see any recent error gets four fifths of that.** Arm D is
our convex demand model with a Kalman filter (a method that updates a forecast from its recent
errors), the night offset, and recipe R (subtract 0.75 times the mean error of the last 14 days). We
trained one network, shared by 6 substations, to correct arm D's median and rescale its spread. Most
of the gain therefore comes from the calendar, the lead time, and the substation, which fix a bias in
arm D without reading errors. Reading the errors and the filter's state adds -0.025 points (99%
interval -0.106 to +0.029) when the network trains on 2020 onward, which is not distinguishable from
zero. This is a weaker test than the brief planned: 6 substations, not 16 Primaries, because the
other Primaries lack the inputs arm D needs, and no weather input. The verdict is `partly supported`:
the headline contrast passes its rule, but the known-answer gate marginally fails (29% against 30%)
and the mechanism, reading recent errors, is not supported.

## Figures

![Contrasts](figure_1_contrasts.svg)

![Gate](figure_2_gate.svg)

![Sites](figure_3_sites.svg)

## Results

- **Method.** One network (two hidden layers of 128 units, a 4-dimensional substation embedding, a
  linear skip) trained with the pinball loss at the 11 quantiles ([Koenker and Bassett
  (1978)](https://doi.org/10.2307/1913643)) on arm D's rows at leads 12 to 48 hours, following the
  pooled-network design of [experiment
  142](../00142-differentiable-physics-sequence-model-on-gam-k-residuals/result.md). Quantiles are
  `median + correction + exp(scale) x (arm D quantile - arm D median)`, so they cannot cross.
  Inputs: the last 12 half-hours of errors known at the origin (3.5 hours old), the mean error of the
  last 14 days at the row's half-hour, the mean error at the same weekday and half-hour in the last
  4 weeks, the lead, half-hour, day of week, and arm D's mean. Stage 3 adds the Kalman level and
  anomaly and a flag for ensemble weather.
- **Two training sets.** Stage 2: the 6 substations' stored ensemble year (2024-07 to 2025-06),
  125,000 rows. Stage 3: the shared file of the sibling experiment, which adds forecasts on past
  weather from 2020 onward, 592,000 rows (4.7 times as many). Both score only the ensemble-weather
  scored year (2025-07-02 to 2026-06-29).
- **Rows against experiment 142.** The gate fit has 287,000 rows from 3,900 site-runs, against about
  3,200 samples in experiment 142: about 90 times the rows, but only 1.2 times the independent
  history.
- **Gate (6 substations, 3 seeds, 8-week injection on held-out blocks).** Level step of +3 points:
  head 29% (22% to 34%, gate 30%), recipe R 67%. Saturday-evening shift of +3 points: head 21% (20%
  to 23%, gate 15%), recipe R 11%. A first version without the same-weekday input recovered -1.5% of
  the shift. The input was added before any scored read.
- **Scored year, five decision substations, leads 15.5 to 39 hours, outside switching events.** Arm
  D scores 2.432 points. Stage 2: head 2.361, blind head 2.377 (C1 head minus arm D -0.071, interval
  -0.154 to -0.014; C2 blind minus arm D -0.054, -0.089 to -0.026; C3 head minus blind -0.017, -0.109
  to +0.039). Stage 3: head 2.307, blind head 2.332 (C1 -0.125, -0.209 to -0.060; C2 -0.100, -0.140
  to -0.064; C3 -0.025, -0.106 to +0.029). Experiment 169's calibrated-spread arm D scores 2.430 on
  the same rows, so spread calibration is not the source.
- **Per substation, stage 3 head minus arm D:** D5 -0.179, D10 -0.075, D11 -0.008, D14 -0.133, D16
  -0.243, D9 -0.104. Stage 2: D5 -0.081, D10 -0.066, D11 +0.040, D14 -0.112, D16 -0.142, D9 +0.014.
- **Tails (decision substations, stage 3 head against arm D).** Pinball at q99: 0.391 against 0.411.
  At q01: 0.347 against 0.386. Coverage at q95: 93.1% against 94.5%; at q99: 97.6% against 98.1%. The
  head lowers tail pinball loss but lowers coverage in the upper tail.
- **Curtailment cost and the manual heuristic.** Not scored: the stored rows are single-substation
  demand, not node and generator forecasts.
- **Caveats.** No ensemble weather input. Stage 3 trains on 4 years of forecasts built on past
  weather, whose errors are smaller than live ensemble errors, and it flags the era. The gate is one
  8-week block per injection. The prediction that the stage 2 gain would be -0.04 to +0.02 was too
  pessimistic about the sign; the bound on reading errors held.

## Checks

- **Median minus mean forecast per site (stage 3 head):** D10 -0.69, D11 -0.01, D14 -0.79, D16
  -0.33, D5 -0.13, D9 -0.11 points. These are arm D's own offsets plus the head's correction.
- **Error of the median forecast beside the score:** mean absolute error arm D 4.191, stage 3 head
  4.006, blind head 4.063; the contrasts point the same way.
- **Triggers that fired:** (1) the known-answer gate marginally failed, 29% against 30%; (5) the
  blind head, nested in the head, scores almost as well; (7) the sibling experiment's design may
  depend on this result. Leaving one substation out never changes the sign of C1 (range -0.097 to
  -0.145 in stage 3). An audit by a fresh Opus agent was requested from the ML manager and has not
  run.

## What to try next

- Replace the network with a calendar, lead, and substation recalibration of arm D's median and see
  whether it matches the blind head.
- Add the ensemble-mean temperature and shortwave inputs, which the stored forecasts lack.
