---
id: "00200"
slug: dp-pooled-correction-head-on-kalman-state
title: "Differentiable physics: a pooled learned correction head on arm D's forecast and recent errors"
theme: differentiable-physics-forecast-engine
also_relevant_to: [lagged-power-and-error-correction, calibration-and-uncertainty, differentiable-physics-pooling-across-substations]
status: planned
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
code: ml-experiments/00200-dp-pooled-correction-head-on-kalman-state/
---

**Planned, not run.** Experiment 142's network failed its known-answer gate with too few rows. This
experiment asks whether pooling many more rows (all Primaries, all years the shared forecasts cover)
fixes that, and stops at the gate if it does not.

## Plan

- **Stage 1, gate (now).** Train on the 6 stored substations (experiment 155, arm D, 2024-07 to
  2026-06) with the head's inputs; inject experiment 142's step and Saturday-evening shift into the
  recent-error inputs and targets; report recovery.
- **Stage 2, contrast (after the shared forecasts exist).** Pool all Primaries and years, train with
  the pinball loss at the 11 quantiles, score against recipe R at D+1 with tail calibration.
- **Inputs.** Recent errors (actual minus arm D mean, by hour band, last 6 hours), the lead, day of
  week, half-hour, holiday flag, ensemble-mean temperature and shortwave, a site embedding. The
  Kalman level and anomaly states are not stored for experiment 155; the lagged errors stand in for
  them in stage 1.
- **Output.** A median correction and a spread scale. Rebuilt quantiles cannot cross.
- **Compute.** One GPU share, one fit at a time, at most 4 CPU cores, 16 GB memory cap.

## Checks

To be filled when the experiment has run.
