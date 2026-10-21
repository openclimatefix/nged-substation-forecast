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
  Headline C1, arm B (frozen convex model residual) minus arm A (lagless XGBoost residual), in
  MW-weighted recall at 80 unexplained odd-month alarms: +0.02 to +0.06. The noise diagnostic (see
  "Pre-run deviations") found that the 24-hour step statistic of arm B has a median of 0.8 times
  arm A's noise, so a given step is about 1.25 times as many noise units. Arm C (filter
  innovations) has 0.2 times arm A's noise but keeps only 0.2 of a 3-day step, so its steps in
  noise units are about as large as arm A's: C2 (arm C minus arm A) is predicted at -0.03 to
  +0.03. Experiment 122's gain of GAM-K over lagless XGBoost is not a prediction for either arm,
  because experiment 122 compared with models frozen for a year, and arm A's XGBoost refits twice a
  month.
decision_rule: >-
  One planned contrast, C1 = arm B minus arm A, the paired difference in MW-weighted recall pooled
  with equal weight over shapes S2, S3, and S4, on experiment 82's fresh headline-band draw, for
  transfers that start in even months, on the 15 series that have convex-model residuals. The 95%
  interval resamples whole calendar months 20,000 times, and is expected to be about 0.08 wide,
  from experiment 82's interval for C1. Every arm is held to 80 unexplained odd-month alarms, every
  arm's noise unit is estimated from its own odd-month residual, and every arm fits its own logistic
  score on its own candidates. The verdict is `supported` when the lower end of C1 is above +0.03;
  `small gain` when the lower end is above zero but under +0.03; `refuted` when the upper end is
  under +0.02; and `inconclusive` otherwise. The threshold of +0.03 is set by the cost of the
  change: a detector that needs a convex-model fit for every Primary, twice a month, is a new
  pipeline step, and the residual needs past weather that arrives about 5 days late, so the gain
  must exceed what a change of the step statistic alone is worth (+0.02 in experiment 82). C2 (arm C
  minus arm A, reported with the same interval), arm D minus arm B, recall by donor step split by
  metered receiver, the 35 labelled real transfers found, the share of misses with a donor step of
  2 to 4 noise units, and the long draw are descriptive and carry no verdict. The gate rests on the
  injected transfers, and the real-event scores are descriptive because only about 14 usable events
  exist.
benchmark: >-
  Arm A, RO-NC on the lagless XGBoost residual of experiment 82, re-scored here on the same 15
  series as arms B and C (experiment 82's all-16-series 0.770, 98.33% interval 0.665 to 0.857, is
  re-scored as a sanity check).
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

We ask whether a detector of load transfers finds more of the power moved when it is built on a
better demand forecast. A load transfer is a switching operation that moves part of one Primary
substation's demand onto neighbouring Primary substations for hours or days. The detector,
[experiment 82](../00082-sw2-18-19-combined-refit-offset-and-same-sign-count/result.md)'s
refit-offset neighbour-count detector (RO-NC), subtracts a forecast from the measured power and
raises an alarm when the 24-hour average of the difference steps by 4.0 noise units or more. It
found 0.770 of the megawatts (MW) moved by transfers that we added to the real power history of 16
Primaries, where the forecast was a lagless XGBoost model. We swap that forecast for the convex
demand model of [experiment 122](../00122-differentiable-physics-three-winter-backtest-gam-k/result.md),
with and without its Kalman filter (GAM-K), and count the share of the moved MW found again.

## Design

One variable changes: the residual. The injected-transfer draw (same seeds, shapes S2, S3, and S4,
even-month starts), the 35 labelled real transfers over 1 day, the 80 unexplained odd-month
alarms, the first-return end rule, and the resampling by whole calendar months are experiment 82's.

- **Arm A:** RO-NC on the lagless XGBoost residual, as in experiment 82.
- **Arm B (headline):** RO-NC on the frozen convex model's residual (observed power minus the
  convex model's forecast, no filter). The convex model is fitted on a 24-month window on the 1st
  and on the 16th of each month, without drift and with a fitted PV capacity, and the two fits give
  the two-model refit offset.
- **Arm C:** RO-NC on the one-step innovations of the Kalman filter on the frozen residual, with the
  same two fits. The filter is re-run on each injected power series. The end rule's level tracker
  reads the frozen residual, because the innovations return to zero after a step.
- **Arm D:** arm B with one model and no refit offset. Descriptive.

## Pre-run deviations and diagnostics

These were found before any detector was scored. The diagnostic reads only residual noise and step
retention, not recall.

- **Headline moved from arm C to arm B.** The first draft of this plan named arm C. A plan review
  asked for a positive control on the filter. The control adds a constant 4-noise-unit, 3-day step
  to the frozen residual, re-runs the filter, and reads the peak 24-hour statistic. The filter keeps
  0.13 to 0.29 of the step (median 0.22 over the 15 series), because the transient's half-life is
  2 to 4 hours and the level tracks within hours. The statistic's noise falls to 0.11 to 0.29 of arm
  A's, so arm C's steps in noise units are about as large as arm A's. Arm B's noise is 0.6 to 1.2
  times arm A's (median 0.8). Arm B is therefore the arm that should gain, and arm C stays as a
  descriptive arm.
- **Fifteen series, not 16.** Series 12 (D15) is a generator case with no usable history for the
  convex model, so it has no residual. Arm A is re-scored with series 12 blanked and with transfers
  that start at series 12 left out, so that every arm sees the same 15 series. Experiment 82's
  16-series 0.770 is reproduced once as a sanity check.
- **Fitted PV capacity, not pinned.** The real-time capacity vintages that experiment 111 pinned
  exist for 8 of the 16 series, so every arm uses the fitted capacity. Experiments 111 and 122 used
  the pinned capacity at the sites they scored.
- **Own filter settings.** Experiment 111's filter settings exist for 6 series and were fitted on a
  year that overlaps this experiment's odd months. The four filter settings of each series are fitted
  here on its odd-month frozen residual with experiment 106's prior.
- **Hindsight in the convex model's training rows.** The convex model leaves out the hand-filtered
  switching rows when it trains. These include the 35 labelled real transfers, so the real-transfer
  count favours arms B and C, and arm A's XGBoost trained on all rows. The injected-transfer
  contrast is unaffected, because the injection is added after the fit.
- **Time axis and units.** The convex model's cache holds end-of-half-hour times and points of
  effective capacity. Converted to MW at the series' effective capacity, the observed power equals
  experiment 82's panel to the last digit.

## Checks that guard the comparison

- **No injected rows in the training window.** Each fit sees uninjected power only, and the
  injection is added after the fit, as for arm A. A unit check asserts that arm B's residual of an
  injected series equals its uninjected residual plus the injected delta.
- **Noise units per arm.** Each arm's noise unit is the robust scale of its own odd-month 24-hour
  step statistic, and its threshold gives 80 unexplained odd-month alarms.
- **Weather.** The convex model uses past weather (ERA5) at fit and forecast times, as experiments
  82 and 122 do. ERA5 arrives about 5 days late, so a live service could not build these residuals.
  [Experiment 101](../00101-ro-nc-detector-on-real-hand-filtered-switching-events/result.md) made
  the same choice. The convex model is `not-for-production` (experiment 122).

## Compute

CPU only, 6 to 10 cores and 8 GB under `systemd-run --user --scope -p MemoryMax=8G -p
OOMPolicy=continue`. The 1,800 convex-model fits take about 3 minutes. The detector scoring over
the injected draw dominates the run time.

## Sizing

- **Stored data:** private tables only; nothing enters a Delta table or a prediction set.
- **Production serving path:** not touched.
- **Degradation rule:** not touched.
- **Defensible designs:** two residuals (arms B and C), resolved by the positive control above.
- **Callers:** the experiment's own scripts, which import only their folder, `studies.*`, and
  reviewed packages. Moving the detector code or the residual builder into `packages/` needs a
  review and is not part of this experiment.

## Earlier advice followed and departed from

- **Followed:** experiment 82's advice to look for the remaining misses among sharp steps of 2 to 4
  noise units with no metered receiver (the descriptive donor-step table).
- **Followed:** experiment 101's advice to quote recall on real transfers as a count out of 35.
- **Followed:** experiment 122's caveat that GAM-K's gain over frozen models is mostly level
  tracking, which is why arm A's twice-monthly refits and arm B's frozen residual are compared and
  the filter's gain is not assumed.
