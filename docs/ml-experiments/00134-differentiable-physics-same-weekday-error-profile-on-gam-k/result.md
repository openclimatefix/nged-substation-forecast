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
  quantiles equal), reported beside the arms and never corrected, on its own subset: the rows where
  that reading exists, with K0 and RB rescored on the same subset.
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
  the corrected median at D+1 on the tuning rows, in points of effective capacity, pooled with equal
  weight per row over the five decision substations, wins. Holiday-like targets stay in the tuning
  rows (they carry zero correction in every arm). Weeks `n` is chosen separately for RW and RB, and
  the full kappa-by-lambda-by-n grid is reported.
  Each arm's spread is then refitted on the tuning rows and its factors chosen per lead band, as
  for every arm of experiment 115.
- **Sensible limits.** If lambda = 0 wins, RB equals Rh and the verdict says so.

## Known-answer gate (reported before any score)

**On real tuning-month residuals, add +3 points to the readings of Saturdays 17:00 to 21:00 local
for 8 consecutive Saturdays (2 November to 21 December 2024), run every arm's full profile code on
the changed readings, and measure the share of the shift each correction recovers at the third to
eighth Saturday.** The Saturday is the UTC day of the valid time, and the half-hours are local. The
injection goes through the same 40-of-48 reading filter, holiday rule, and fallback as the real
scoring, so the gate can fail for a real reason. The share is linear in the weights: with kappa 0.75,
R recovers about 0.75 x 2/14 = 0.11, and RB at lambda 0.5 and n 4 recovers about 0.19.

- **Settings.** The gate runs at the tuned settings and at a fixed reference (kappa 0.75, lambda 0.5,
  n 4), so the result does not depend on where tuning lands.
- **Pass conditions, at the reference.** RB recovers at least 0.15 of the shift and at least twice
  what R recovers; the placebo (the profile reads the weekday that falls 3 days after the target's;
  same settings, not re-tuned, because the placebo only has to show the gain needs the right
  weekday) recovers at most what R recovers plus 0.05; and on the tuning months the placebo's mean
  absolute error at D+1 lies less than halfway from Rh's to RB's.
- **Tuned settings.** Reported beside the reference. If tuning picks lambda 0, RB equals Rh and the
  verdict says so.
- A failed condition caps the verdict at `inconclusive`. The gate runs before any replicate or
  scored forecast is read.

## Predictions

- C1 (RB minus R) is -0.01 to -0.04 points at D+1; C2 (RW minus R) is 0.00 to +0.05, because 2 to 4
  samples per weekday are noisy; COPY7 is worse than K0 overall; the best lambda lies between 0.25
  and 0.75; RB-day-types is within 0.01 of RB (C5).

## Contrasts and decision rule

- **C1** RB minus R (decides the verdict), **C2** RW minus R, **C3** RB minus K0, **C4** RB minus L1,
  **C5** RB-day-types minus RB. **C6** (diagnostic) RB minus Rh.
- **Supported** if C1's scored-year interval has its upper end at most -0.02 points; the replicate
  year has the same sign and an estimate at least half the scored estimate; at least 3 of 5
  substations improve in both periods; the winter C1 estimate is at most 0 with its upper end under
  +0.02; and the gate passes. **Small gain** if the scored interval lies wholly below zero but its
  upper end is above -0.02. **Not supported** if the interval includes zero. **Refuted** if the
  interval lies wholly above zero. A gate failure caps every verdict at `inconclusive`.
- **C1 and the holiday rule.** C1 is RB minus R as the brief states, so it includes the holiday rule.
  The write-up reports C1 on non-holiday rows and C6 (RB minus Rh) beside it, and says so when the
  weekday effect and the holiday rule point in different directions.
- **Threshold, with the reason.** 0.02 points (about 1% of the score), because the change adds one
  profile of the same size as recipe R to the serving path.
- **Intervals.** 99% intervals, month blocks resampled, paired across arms
  (`studies.bootstrap.bootstrap_difference_at_level`).

## Breakdowns and checks

The winter evening peak is 16:00 to 20:00 local from November to March. "Last n weeks" counts the
last n same-weekday dates, and a date left out by the holiday rule or by the reading filter is not
replaced. Charts and tables use D-labels only. `result.md` opens with the take-home message, then one
headline chart with intervals, then the weekday-by-weekday chart.

Each weekday separately; winter (November to March) and the winter evening peak; cold nights (daily
mean under about 3 C, as in experiment 122) with q95 coverage; holidays, Christmas windows, and
Easter; non-winter; per substation (5 of 5 counts); tails (coverage and pinball loss at q01, q05,
q95, q99); and the "Checks" list the `ml-experiment` skill requires.

## Compute and schedule

CPU only, at most 4 cores and 8 GB under `systemd-run --user --scope -p MemoryMax=8G -p
OOMPolicy=continue`. Each arm is a correction on stored member forecasts, so a fit takes seconds.
No retraining happens: experiment 111's monthly refits are reused.
