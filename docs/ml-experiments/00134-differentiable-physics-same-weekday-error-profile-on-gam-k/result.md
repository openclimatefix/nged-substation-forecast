---
id: "00134"
slug: differentiable-physics-same-weekday-error-profile-on-gam-k
title: "Differentiable physics: a same-weekday error profile on top of GAM-K"
theme: differentiable-physics-forecast-engine
also_relevant_to: [lagged-power-and-error-correction]
status: planned
size: >-
  simple to medium. Stored: private files under data/ml-experiments/00134-<slug>/ only. Serving
  path: not touched. Degradation rule: not touched. Designs: more than one is defensible (how
  the weekday and all-days profiles combine, how many weeks, how the settings are chosen), and
  this plan pre-registers one of each. Callers: this folder's scripts only; the folder copies the
  modules of experiment 115 it needs and reads experiment 111's and 115's stored outputs.
hypothesis: >-
  Adding a same-weekday component to the 14-day all-days error profile (recipe R of experiment 115)
  lowers GAM-K's day-ahead quantile score, because demand errors repeat from week to week
  (Saturdays differ from Tuesdays) and the all-days profile averages that structure away.
prediction: >-
  Written before any fit, see "Predictions". Headline: RB minus R is -0.01 to -0.04 points at D+1;
  RW minus R is 0.00 to +0.05; COPY7 is worse than K0 overall.
decision_rule: see "Decision rule"
benchmark: >-
  Recipe R of experiment 115 (kappa 0.75 on the 14-day mean D+1 error profile, 2.436 points at D+1
  in the scored year at the five decision substations), K0 (GAM-K as stored, 2.525), a naive
  copy-7-days-ago forecast, and L1 (last week's same-weekday error)
row_set_id: >-
  experiment 115's rows: tuning on the odd months and replicate on the even months of 2024-07 to
  2025-06, scored on the 292 runs of fold_a_daily_gap_v1 (2025-07-02 to 2026-06-29), at D5, D10,
  D11, D14, and D16, with D9 beside them
metric: >-
  Quantile score (twice the mean pinball loss over experiment 87's 11 quantiles) in points of
  effective capacity, on rows outside the hand-filtered switching events, at day D+1 (leads 15.5 to
  39 hours from the 09:00 UTC power_fcst_init_time); every lead band beside it
code: ml-experiments/00134-differentiable-physics-same-weekday-error-profile-on-gam-k/
---

**Plan, committed before any fit.** This experiment asks whether the forecast's recent-error
correction should look at the same weekday as the day being forecast, rather than at the last 14
days alike. It reuses experiment 111's stored GAM-K forecasts and experiment 115's K0 and recipe R,
and refits no demand model. It follows [experiment
115](../00115-differentiable-physics-w1-diurnal-shape-state-on-ens/result.md), whose recipe R is a
plain average of the last 14 days' errors by half-hour; no weekday-specific version has been tested.
The idea of shrinking a weekday profile towards an all-days profile is a standard partial-pooling
step ([Gelman and Hill (2006)](https://doi.org/10.1017/CBO9780511790942)), and the naive baseline
of repeating the value seven days ago is the usual seasonal-naive forecast
([Hyndman and Athanasopoulos (2021)](https://otexts.com/fpp3/simple-methods.html)).

## Arms

- **K0.** GAM-K as stored (control).
- **R.** Recipe R reproduced exactly as experiment 115 stored it: K0 minus kappa times the mean of
  the last 14 UTC days' D+1 errors by local half-hour, each day's mean removed, kappa 0.75, no
  holiday rule.
- **Rh.** R with the holiday rule below, kappa re-chosen on the tuning months. It separates the
  effect of the holiday rule from the effect of the weekday.
- **COPY7.** The actual power at the same half-hour 7 days earlier as a point forecast (all 11
  quantiles equal), reported beside the arms and never corrected.
- **L1.** K0 minus kappa times last week's same-weekday D+1 error profile (one sample).
- **RW.** K0 minus kappa times the mean same-weekday error over the last n weeks; n from 2, 3, 4.
- **RB (lead arm).** Profile = A + lambda (W - A), where A is Rh's 14-day all-days profile, W is
  the n-week same-weekday profile, and lambda lies in 0 to 1. Where W has no usable day, W is A.
- **RB-day-types.** As RB, with W built from days of the same day type: Monday, Tuesday to
  Thursday, Friday, Saturday, Sunday.

**No lookahead.** At the 09:00 UTC origin on day D the target is D+1. Each source day is a UTC day
fully observed by D 00:00 UTC, so the same weekday of the target last counts on day D-6. A day
enters a profile when at least 40 of its 48 half-hours have a reading, as in experiment 115.

**Holiday rule (Rh, RW, L1, RB, RB-day-types).** A holiday-like day is an English bank holiday,
Easter Saturday or Sunday, or 24 or 27 to 31 December. Holiday-like days are left out of every
profile as source days, and a holiday-like target gets no correction. Holiday-like targets are
scored separately.

## Settings and how they are chosen

- **Grids.** kappa from 0.5, 0.75, 1.0; lambda from 0, 0.25, 0.5, 0.75, 1; n from 2, 3, 4.
- **Rule.** Fleet-wide settings (one set for all five decision substations), because each
  substation has about 3,000 tuning-month D+1 rows and the weekday profiles use 2 to 4 samples; a
  per-substation choice would mostly fit noise. The setting with the lowest mean absolute error of
  the corrected median at D+1 on the tuning rows, pooled over the five decision substations, wins.
  Each arm's spread is then refitted on the tuning rows and its factors chosen per lead band, as
  for every arm of experiment 115.
- **Sensible limits.** If lambda = 0 wins, RB equals Rh and the verdict says so.

## Known-answer gate (reported before any score)

**On real tuning-month residuals, inject a +3-point shift into the readings of Saturdays
17:00 to 21:00 local for 8 consecutive weeks and measure the share of it each arm's correction
recovers at the later Saturdays of the window.** Pass conditions: RB recovers at least twice what
R recovers and at least 0.3 of the shift; the placebo (the weekday labels of the profile shifted by
3 days, same settings) recovers no more than R plus 0.05; and on the tuning months the placebo
RB's gain over Rh is under half the real RB's gain. A failed condition caps the verdict at
`inconclusive`.

## Predictions

- C1 (RB minus R) is -0.01 to -0.04 points at D+1; C2 (RW minus R) is 0.00 to +0.05, because 2 to 4
  samples per weekday are noisy; COPY7 is worse than K0 overall; the best lambda lies between 0.25
  and 0.75; RB-day-types is within 0.01 of RB (C5).

## Contrasts and decision rule

- **C1** RB minus R (decides the verdict), **C2** RW minus R, **C3** RB minus K0, **C4** RB minus L1,
  **C5** RB-day-types minus RB. **C6** (diagnostic) RB minus Rh.
- **Supported** if C1's scored-year estimate is at most -0.02 points and its interval lies wholly
  below zero; the replicate year has the same sign and at least half the size; at least 3 of 5
  substations improve in both periods; the winter breakdown (November to March, and winter evening)
  shows no loss; and the gate passes. **Small gain** if the interval lies below zero but the
  estimate is smaller than 0.02 points. **Not supported** if the scored interval includes zero or
  lies above it. A gate failure caps the verdict at `inconclusive`.
- **Threshold, with the reason.** 0.02 points (about 1% of the score), because the change adds one
  profile of the same size as recipe R to the serving path.
- **Intervals.** 99% intervals, month blocks resampled, paired across arms
  (`studies.bootstrap.bootstrap_difference_at_level`).

## Breakdowns and checks

Each weekday separately; winter (November to March) and the winter evening peak; cold nights (daily
mean under about 3 C, as in experiment 122) with q95 coverage; holidays, Christmas windows, and
Easter; non-winter; per substation (5 of 5 counts); tails (coverage and pinball loss at q01, q05,
q95, q99); and the "Checks" list the `ml-experiment` skill requires.

## Compute and schedule

CPU only, at most 4 cores and 8 GB under `systemd-run --user --scope -p MemoryMax=8G -p
OOMPolicy=continue`. Each arm is a correction on stored member forecasts, so a fit takes seconds.
No retraining happens: experiment 111's monthly refits are reused.
