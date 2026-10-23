---
id: "00303"
slug: wind-adapter-in-production-setting
title: >-
  Does a small neural correction of the wind speed beat XGBoost plus a lookup table of offsets when
  the weather is the 51-member ensemble forecast?
theme: generator-forecasting
also_relevant_to:
  - calibration-and-uncertainty
status: planned
date: 2026-10-23
size: >-
  complex by one trigger, run with Sonnet only as the maintainer instructed. Stored: private tables
  under data/ml-experiments/00303-wind-adapter-in-production-setting/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: more than one defensible way to train an adapter
  on 442 ensemble runs (from scratch, pre-trained on ERA5 then fine-tuned, cross-fitted for the
  spread factor), so this plan fixes one. Callers: this folder's scripts, which copy experiments
  285 and 259, and no new reviewed module. One Sonnet code review and one Sonnet write-up review.
hypothesis: >-
  Experiment 285 found that, with ERA5 as the weather, a small neural adapter in front of the
  physics core of a wind farm scored better than XGBoost. On the 51 ensemble members in the scored
  year, an adapter fine-tuned on the ensemble hindcast beats XGBoost plus experiment 259's offset
  table (the benchmark that matters), because the adapter corrects the wind speed itself and the
  table corrects only the power.
prediction: >-
  Written before any run, on the kept rows. P2 (adapter trained on ERA5, run on members) beats P0
  (XGBoost on members) by 0.05 to 0.2 points, but P3 (P2 plus the offset table) does not beat P1
  (P0 plus the offset table) by more than 0.1. P5 (ERA5 pre-training, then fine-tuning on the
  hindcast) beats P3 by at least 0.05, and P4 (trained on the hindcast alone) is worse than P5. P6
  (P5 or the best of P2 to P5, plus a neural head of 11 quantile offsets) lowers the 0.99 pinball
  loss by at least 10% against P3 and stays within 0.1 points on the quantile score. P8 (XGBoost
  with the adapter's effective wind speed as an extra column, plus the offset table) is at least as
  good as P3. The blind control P7 closes under a third of the gap that P5 closes.
decision_rule: >-
  Supported if P5 beats P1 by at least 0.10 points on the kept rows, with the Bonferroni interval
  (8 planned contrasts, level 99.375%) wholly above zero. The threshold is 0.10 points because
  porting the adapter adds a neural network and a fine-tuning step to the weekly retraining, which
  a gain under 0.10 points does not pay for. Partly supported if P5 or P8 beats P1 with an interval
  above zero but under 0.10 points. Not supported if no arm beats P1. Stop and check the
  ERA5-to-ensemble mismatch if P2 is worse than P0 by more than 0.3 points. Intervals resample
  whole months of the ensemble run date and the three seeds (50,000 resamples); farms are not
  resampled. Planned contrasts, each first arm minus second arm so that a negative number means the
  first arm is better; C1 P2 - P0, C2 P3 - P1, C3 P5 - P3, C4 P4 - P5, C5 P6 - P3 (quantile score
  and 0.99 pinball loss), C6 P8 - P3, C7 P7 vs P5 as a share of the gap closed against P3, C8 P5 -
  P1. The 95% intervals the brief asks for are reported beside the Bonferroni ones.
benchmark: >-
  P1, experiment 285's xgb_B4 (XGBoost run on members, then the lead-day, quantile, and farm offset
  table of experiment 259). P0 is experiment 285's xgb_B0.
row_set_id: >-
  experiment 33's wind cases (`fold_a_tuesdays_v1`: 53 Tuesday runs from 2025-07-01 to 2026-06-23,
  W1 to W3, three seeds). Fitting rows are the daily ensemble hindcast runs of 2024-04-01 to
  2025-06-16 (442 runs), as in experiments 259 and 285.
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over nine levels, the estimate of the
  CRPS), in points of capacity; also the 0.01 and 0.99 pinball loss, the 5% to 95% coverage, and the
  share of observed power below the median forecast
code: ml-experiments/00303-wind-adapter-in-production-setting/
---

**Plan (committed before any run).** This page is the plan until the write-up replaces it.

## Question

Experiment [285](../00285-wind-effective-wind-adapters/result.md) found two things. With ERA5 (the
ECMWF reanalysis of past weather) as the weather, a small neural adapter on the wind speed that the
physics core reads closed more than the whole gap between the core and XGBoost. On the ensemble, a
lookup table of power offsets ([experiment 259](../00259-wind-plant-plus-quantile-offset-head/result.md))
beat every wind-speed mapping. This experiment tests the adapter where it matters: on the 51 ensemble
members, in the scored year, against XGBoost plus the offset table.

## Design

- **Weather and fits.** The physics core, the adapter, and XGBoost fit to ERA5 before 2024-04-01.
  The ensemble hindcast (442 daily runs, 2024-04-01 to 2025-06-16) supplies the fitting rows for the
  offset table, the spread factor, and the fine-tuned arms. Each arm runs on every member, mixes the
  51 members with equal weights, applies experiment 33's spread factor fitted month by month on
  earlier cases, and is scored on the 53 Tuesday runs.
- **Arms.** P0 is experiment 285's xgb_B0 and P1 its xgb_B4, read from stored mixtures and
  re-finalised by this folder's code as a reproduction check. P2 is the adapter A4 of experiment 285
  (inputs: direction, the 10 m to 100 m speed ratio, 2 m temperature, hour of day), fitted to ERA5
  and run on members. P3 adds the offset table to P2. P4 trains the whole core and an adapter on the
  hindcast members directly, with two extra adapter inputs (lead day and the members' standard
  deviation of the 100 m speed). P5 starts from P2 and fine-tunes on the hindcast at a learning rate
  one tenth of the pre-training rate, stopping on the last 3 hindcast months. P6 adds experiment
  259's neural head H5c (11 quantile offsets, inputs the 11 base quantiles, the member summaries,
  lead, and farm) to the best of P2 to P5, chosen on the hindcast score and never on the scored
  year. P7 is P5 with the adapter given only the 100 m speed (experiment 285's blind adapter). P8 is
  XGBoost with the adapter's effective hub-height speed as an extra column, plus the offset table.
- **Training rows for P4, P5, and P7.** Each hindcast case contributes 6 randomly chosen members, and
  the loss is the L1 error of the core's expected power against the observed fraction. Because a
  fitted arm is in sample on the hindcast, P4, P5, and P7 are cross-fitted: three contiguous
  blocks of the hindcast each get predictions from a model fitted on the other two blocks, and the
  spread factor and the residual quantiles use only those out-of-fold predictions. The scored
  predictions come from a model fitted on the whole hindcast (stopping on its last 3 months).
- **Residual quantiles for P4, P5, and P7.** The offsets from the core's expected power are fitted to
  the case-mean expected power (not to each member), so the 51-member mixture does not count the
  ensemble spread twice. The spread factor then calibrates the mixture.
- **Checks.** Leave-one-farm-out for P3 and P5 (an adapter and core fitted on two farms, scored on the
  third, with offsets from the other two farms), the share of observed power below the median, the
  median forecast minus the mean forecast, and the mean-forecast error beside every quantile-score
  contrast.
- **Not done.** Experiment 284's tail head is a solar-farm model with a tail-only loss and not
  tested on wind; P6 uses experiment 259's head, which was tested on these farms. The curtailment
  cost metric of experiment 277 is not used unless its code is ready.

## Earlier advice followed and departed from

- Followed: offset table first as the benchmark (experiment 259, D3), and the stack of a
  lead-day mapping on the table (experiment 285, B5).
- Followed: keep the hour of day in the adapter (experiment 285, A4 against A5).
- Departed from: experiment 259 advises against porting a neural head; P6 re-tests the head only
  after the adapter has corrected the speed.

## Compute

Sonnet only. 4 CPU cores (threads capped at 1 to 2), one GPU job at a time under 3 GPU hours in
total, under `systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue`. Cap: 8 hours.
