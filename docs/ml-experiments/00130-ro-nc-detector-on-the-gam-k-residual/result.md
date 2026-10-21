---
id: "00130"
slug: ro-nc-detector-on-the-gam-k-residual
theme: switching-detectors
also_relevant_to:
  - differentiable-physics-forecast-engine
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00130-ro-nc-detector-on-the-gam-k-residual/
  only (monthly fits of the convex model, three residual panels, alarm and outcome tables), no
  prediction set. Serving path: not touched. Degradation rule: not touched. Designs: two defensible
  residuals (the frozen convex model, and the filter's one-step innovations), and the plan
  pre-registers which one is the headline. Callers: this folder's scripts only, which copy
  experiment 82's detector code and call the reviewed `studies.dp_gam` and `studies.dp_assimilation`
  modules without changing them.
hypothesis: >-
  The refit-offset neighbour-count detector (RO-NC) finds more of the moved megawatts (MW) of
  injected load transfers when its residual is the one-step innovation of the convex demand model
  with the Kalman filter (GAM-K), than when it is the residual of the lagless XGBoost model of
  experiment 82. A lower residual noise should lift more of the donor steps of 2 to 4 noise units
  over the 4.0 threshold, and those steps cause 401 of the 725 transfers that RO-NC misses.
prediction: >-
  Headline C1, GAM-K innovations (arm C) minus lagless XGBoost (arm A, 0.770 in experiment 82), in
  MW-weighted recall at 80 unexplained odd-month alarms: +0.02 to +0.08. Two effects pull against
  each other. Experiment 122 found GAM-K 3.1 points of capacity better than lagless XGBoost one day
  ahead, mostly from tracking the level, which lowers the noise. The filter also absorbs a step over
  about a day, which shrinks the 24-hour step statistic of a real transfer. The frozen convex model
  (arm B) is predicted to gain nothing, -0.02 to +0.02, because experiment 122 found the plain
  convex model no better than lagless XGBoost (GAM-K minus plain -3.06, -1.83, and -1.90 points,
  against -3.10, -2.00, and -2.17 points for GAM-K minus lagless XGBoost).
decision_rule: >-
  One planned contrast, C1 = arm C minus arm A, the paired difference in MW-weighted recall pooled
  with equal weight over shapes S2, S3, and S4, on experiment 82's fresh headline-band draw, for
  transfers that start in even months. The 95% interval resamples whole calendar months 20,000
  times. Every arm is held to 80 unexplained odd-month alarms, and every arm's noise unit is
  estimated from its own odd-month residual. The verdict is `supported` when the lower end of C1 is
  above +0.03; `small gain` when the lower end is above zero but under +0.03; `refuted` when the
  upper end is under +0.02; and `inconclusive` otherwise. The threshold of +0.03 pays for porting
  the convex model's monthly fits for every Primary into the detector, which is a new pipeline step,
  and is larger than the +0.02 that experiment 82 used for a change of the step statistic. Arm B
  minus arm A (C2), arm D minus arm B, recall by donor step split by metered receiver, the 35
  labelled real transfers found, the share of misses with a donor step of 2 to 4 noise units, and
  the long draw are descriptive and carry no verdict. The gate rests on the injected transfers, and
  the real-event scores are descriptive because only about 14 usable events exist.
benchmark: >-
  Arm A, RO-NC on the lagless XGBoost residual of experiment 82, re-scored here on the rows of the
  GAM-K arms (0.770 in experiment 82, 98.33% interval 0.665 to 0.857).
row_set_id: >-
  none (this experiment scores detector alarms against injected and labelled real transfers, as
  experiments 82 and 101 do, not forecasts)
metric: >-
  MW-weighted recall, pooled over S2, S3, and S4 on even months at 80 odd-month unexplained alarms.
  Descriptive: recall by donor step in noise units split by metered receiver, the number of the 35
  labelled real transfers over 1 day found, and the share of misses with a donor step of 2 to 4
  noise units.
code: ml-experiments/00130-ro-nc-detector-on-the-gam-k-residual/
production_status: not-for-production
adopted_in: none
---

# Experiment 130: does the detector find more load transfers on the GAM-K residual?

**Plan only.** This page holds the pre-registered plan. The write-up replaces it after the fits.

## Question and design

We ask whether the refit-offset neighbour-count detector (RO-NC, [experiment
82](../00082-sw2-18-19-combined-refit-offset-and-same-sign-count/result.md), checked on real events
in [experiment 101](../00101-ro-nc-detector-on-real-hand-filtered-switching-events/result.md))
finds more of the moved megawatts of a load transfer when its residual comes from the convex demand
model with the Kalman filter (GAM-K, [experiment
122](../00122-differentiable-physics-three-winter-backtest-gam-k/result.md)) than from the lagless
XGBoost model. One variable changes: the residual. The injected-transfer draw (same seeds, shapes
S2, S3, and S4, even-month starts), the 35 labelled real transfers over 1 day, the 80 unexplained
odd-month alarms, the first-return end rule, and the resampling by whole calendar months are
experiment 82's.

- **Arm A:** RO-NC on the lagless XGBoost residual, as in experiment 82. It is the sanity arm and
  must reproduce 0.770 on experiment 82's rows.
- **Arm B:** RO-NC on the frozen convex model's residual (observed power minus the plain convex
  model's forecast, with no filter). The two-model refit offset uses the convex model fitted on the
  1st and on the 16th of each month.
- **Arm C (the pre-registered headline):** RO-NC on the one-step innovations of the GAM-K filter
  (observed power minus the convex forecast and the filter's level and anomaly before the reading),
  with the same two-model refit offset. The filter is re-run on each injected power series, because
  the injected transfer is an observed reading. The filter's hyperparameters are those of
  experiment 111.
- **Arm D:** arm B with one model, with no refit offset (the PF14 form on the convex residual).
  Descriptive.
- **Why arm C is the headline.** Experiment 122 found the plain convex model no better than
  lagless XGBoost, so arm B is expected to leave the noise unchanged. Only the filter lowers it. The
  design note that proposed this experiment names the filter's innovations as one of two residuals
  that do not adapt to a step, because an innovation is computed before the filter sees its own
  reading. The risk is that the filter absorbs a transfer over about a day and so shrinks the
  24-hour step. Arm B is in the experiment so that this risk can be read off.

## Checks that guard the comparison

- **No injected rows in the training window.** The convex model is fitted on uninjected power only,
  at each cut, with the fit's training window ending at the cut. The injection is added to power
  after the fit, as in experiment 82, where the lagless model also does not retrain on it. A unit
  check asserts that the arm A residual of an injected series equals the uninjected residual plus
  the injected delta.
- **Noise units per arm.** Each arm's noise unit is the robust scale of its own odd-month 24-hour
  step statistic, and its threshold is the one that gives 80 unexplained odd-month alarms.
- **Known-answer check.** On a series with no injection, the arm C innovations must have a mean near
  zero and no step at the cuts; the plan reports their standard deviation against arm A's.
- **Weather.** The convex model uses past weather (ERA5) at fit and forecast times, as experiments
  82 and 122 do. This is a lab condition: ERA5 arrives about 5 days late, so a live service cannot
  build these residuals in real time. [Experiment
  101](../00101-ro-nc-detector-on-real-hand-filtered-switching-events/result.md) made the same
  choice.

## Compute

CPU only, at most 2 cores and 4 GB, under `systemd-run --user --scope -p MemoryMax=4G -p
OOMPolicy=continue`. The convex model needs fits for 16 Primaries at the 1st and the 16th of each
month from 2022-01, which is about 1,700 fits. A pilot first times one fit. If the full set would
take more than 12 hours on 2 cores, the fits start where the pinned-capacity vintages begin
(2023-08), and arm A is re-scored on the same shortened rows, so that the contrast stays paired.
The plan's compute fallback is recorded in the write-up either way.

## Sizing

- **Stored data:** private tables only; nothing enters a Delta table or a prediction set.
- **Production serving path:** not touched.
- **Degradation rule:** not touched.
- **Defensible designs:** two residuals (arms B and C), resolved by pre-registering arm C.
- **Callers:** the experiment's own scripts, which import only their folder, `studies.*`, and
  reviewed packages. Moving the detector code or the residual builder into `packages/` needs a
  review and is not part of this experiment.

## Earlier advice followed and departed from

- **Followed:** the advice of experiment 82 to look for the remaining misses among sharp steps of 2
  to 4 noise units with no metered receiver (this experiment's descriptive table).
- **Followed:** the advice of experiment 101 to quote recall on real transfers as a count out of 35.
- **Departed from:** none.
