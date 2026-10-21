---
id: "00143"
slug: differentiable-physics-holiday-aware-updating-of-the-gam-k-filter
title: "Differentiable physics: keeping holiday readings out of the GAM-K filter's update"
theme: differentiable-physics-forecast-engine
also_relevant_to:
  - differentiable-physics-demand-calendar-and-weather
  - lagged-power-and-error-correction
status: planned
size: >-
  simple to medium. Stored: private files under data/ml-experiments/00143-<slug>/ only. Serving
  path: not touched. Degradation rule: not touched. Designs: more than one is defensible (which
  days count as holidays, whether the day after a holiday is also skipped, what a holiday target
  gets), and the plan pre-registers one holiday set and four arms. Callers: this folder's scripts
  only; the folder copies modules of experiment 134, and no change to packages/ or src/.
hypothesis: >-
  The Kalman filter of GAM-K lets a holiday's reading error teach it a wrong level, which carries
  into the days after the holiday; leaving holiday readings out of the filter's update (and nothing
  else) lowers the day-ahead score on those following days, and does not hurt the holiday targets
  or the rest of the year.
prediction: >-
  Written before any fit, see "Predictions" below. Headline: K-skip minus K0 on the days 1 to 5
  after a holiday cluster is -0.05 to -0.30 points at D+1 in the scored year, with the largest
  gain on the Christmas window; on holiday days it is -0.05 to +0.10; on all other days it is
  within 0.01.
decision_rule: see "Decision rule"
benchmark: >-
  K0, GAM-K as experiment 111 stored it and experiment 115 rebuilt it (2.524 points at D+1 in the
  scored year at the five decision substations); K-zero, the drop-and-zero idea of experiment 118's
  KEZ-DT, as a reference
row_set_id: >-
  experiment 115's rows: tuning on the odd months and replicate on the even months of 2024-07 to
  2025-06, scored on the 292 runs of fold_a_daily_gap_v1 (2025-07-02 to 2026-06-29), at D5, D10,
  D11, D14, and D16, with D9 beside them
metric: >-
  Quantile score (twice the mean pinball loss over experiment 87's 11 quantiles) in points of
  effective capacity, on rows outside the hand-filtered switching events, at day D+1 (leads 15.5 to
  39 hours from the 09:00 UTC power_fcst_init_time); every lead band beside it
---

**This experiment asks whether the recent-error filter in the day-ahead forecast should ignore
readings taken on holidays.** The filter (a Kalman filter that tracks a slowly moving level error
and a quickly decaying error in the physics-based demand model's forecast) reads each new half-hour
of demand and moves its level towards the error it sees. A holiday's demand differs from the model's
forecast for reasons that do not repeat the next day, so the filter may learn a level error that
is wrong on the days after the holiday. This forecast is judged over this Christmas, so the days
after Christmas matter most. We keep the demand model, its filter settings, and its spread fixed,
and change only which readings the filter updates on. The plan below is committed before any fit.

## Background and credit

**GAM-K is the physics-based demand model (a generalised additive model) with the Kalman filter on
its recent errors, as built in [experiment
111](../00111-differentiable-physics-6-ens-forecast-engine-against-bg2/result.md),** following the
Kalman-on-additive-model idea of [de Vilmarest and Goude
(2022)](https://doi.org/10.48550/arXiv.2110.00334). This experiment reuses experiment 111's stored
fits, calibrations, and filter settings; experiment 115's cached ensemble member forecasts, K0
spread, and scoring rows; and the scoring and breakdown code of [experiment
134](../00134-differentiable-physics-same-weekday-error-profile-on-gam-k/result.md). [Experiment
118](../00118-differentiable-physics-kfe-kalman-filter-on-the-gam-effects-on-ens/result.md) tried
two holiday actions at once in a different filter (KEZ-DT): it dropped holiday readings from the
update and zeroed the correction on holiday targets, and it scored +1.24 points worse on 17 holiday
days (+0.62 to +1.83). Halving the correction (KEZ-DTs) was +0.09 on those days. Only the second
action is known to hurt. No GAM-K breakdown of Christmas exists. Experiment 134 found that the
holiday rule on the recent-error profile raised the score by 0.004 points (an interval including
zero), and that its Christmas-window contrast was +0.07; the holiday rule's own effect in those
windows was not printed.

## Arms

All four arms are K0's member forecasts plus a correction, with K0's filter hyperparameters, spread
tables, and spread factors (experiment 115's stored values), so the arms differ only in the
correction.

- **K0 (control).** GAM-K's filter reads every reading, as in experiment 111.
- **K-skip.** A reading on a holiday day is treated as missing for the update only. A holiday
  target still gets the carried level and transient.
- **K-skip+1.** K-skip, and also the first non-holiday day after each holiday day is skipped, to
  test slow recovery.
- **K-zero (reference only).** K-skip's filter, and the level and transient are set to zero on
  holiday targets. This is experiment 118's drop-and-zero idea on GAM-K.

**The holiday set is fixed here before any fit:** every England and Wales bank holiday in the
`holidays` package's list for England (including the substitute days it marks as observed), and
every day from 24 December to 2 January. Days are Europe/London dates; a reading belongs to the date
of its half-hour's start.

**Day categories (disjoint, by the target's local date).** *Christmas window* (24 December to 2
January); *bank holiday* (a bank holiday outside that window); *after Christmas* and *after bank
holiday* (days 1 to 5 after the last day of a holiday cluster, where a cluster joins holiday days at
most 3 days apart, so Good Friday and Easter Monday are one cluster); *other* (everything else).
"Leakage days" are the two "after" categories. 27 to 30 December lie inside the Christmas window and
are also scored on their own.

## Settings

**No setting is fitted.** The filter hyperparameters, spread tables, and spread factors are those
of K0 as experiment 111 and 115 tuned them. The odd months of 2024-07 to 2025-06 (tuning months) are
therefore a first read, with the known-answer gate run on them; the even months (replicate months)
and the scored year (2025-07-02 to 2026-06-29) are read after the predictions are fixed.
Christmas 2024 lies in the replicate months and Christmas 2025 in the scored year. The past-weather
winters of [experiment
122](../00122-differentiable-physics-three-winter-backtest-gam-k/result.md) cannot be added cheaply:
that experiment stored quantile predictions only, not residual streams, so rerunning the filter on
earlier Christmases would need new past-weather fits and is left to a later experiment.

## Planned contrasts

All at day D+1, quantile score in points of effective capacity, K-arm minus K0 unless stated
(negative favours the first-named arm), pooled over the five decision substations, with 99%
intervals from a day-block bootstrap (below).

- **C1 (decides).** K-skip minus K0 on the leakage days.
- **C2 (non-inferiority).** K-skip minus K0 on holiday days (Christmas window and bank holidays).
- **C3 (non-inferiority).** K-skip minus K0 on the other days.
- **C4.** K-skip+1 minus K-skip on the leakage days.
- **C5 (reference).** K-zero minus K-skip on holiday days.
- **C6.** K-skip minus K0 over all D+1 rows (month-block bootstrap, `studies.bootstrap`).

**Required breakdowns for C1 to C3** (K-skip against K0, and K-skip+1 against K0): winter
(November to March); winter evening 16:00 to 20:00; cold nights (days with a daily mean under 3 C,
local 22:00 to 06:00), with q95 coverage; each holiday cluster separately, with a figure per
Christmas (20 December to 5 January, daily mean D+1 score per arm); 27 to 30 December; every lead
band beside D+1. Tail calibration (coverage and pinball loss at q01, q05, q95, q99) is reported for
each arm.

**Day-block bootstrap.** A block is one local date, with all decision substations' rows on that date
(substations move together on a day). Dates are resampled with replacement within the category and
period, 5,000 resamples, a fixed seed, percentile interval. The write-up states how many dates each
category holds. Christmas occurs once a year, so the Christmas findings rest on two Christmases
(2024 and 2025) and are not generalised.

## Predictions

- **P1.** C1 is between -0.05 and -0.30 in the scored year, and negative in the replicate months.
- **P2.** On 27 to 30 December, K-skip minus K0 is at most -0.10 in each of the two Christmases.
- **P3.** C2 is between -0.05 and +0.10.
- **P4.** C3 is within ±0.01, and its upper interval end is at most +0.01.
- **P5.** C4 is within ±0.03 (skipping the day after a holiday adds nothing measurable).
- **P6.** C5 is above +0.30 (zeroing the correction on holiday targets hurts), the direction of
  experiment 118.

## Decision rule

**The verdict follows C1, C2, and C3 in the scored year, and the replicate months must agree in the
sign of C1.** Skipping holiday readings costs nothing in production, because it is a mask on the
update, so the threshold for a gain is zero.

- **supported:** C1's interval lies wholly below zero; C2's upper end is at most +0.10 (about 4% of
  the score; holiday days are about 5% of the year); C3's upper end is at most +0.01; and C1 is
  negative in the replicate months.
- **partly supported:** C1's interval lies wholly below zero but C2 or C3 fails its margin.
- **not supported:** C1's interval includes zero or lies above it.
- A failed known-answer gate caps the verdict at `inconclusive`.

## Known-answer gate (before any scored fit)

**On real tuning-month residuals, add +3 points to every reading of one holiday day, run each arm's
filter on the changed stream, and measure how much of the step reaches the correction on the
following days.** The holiday days injected are 1 January 2025, 5 May 2025, and 26 May 2025 (all in
tuning months); the control day is Wednesday 15 January 2025 (not a holiday, not adjacent to one).
The change in the correction (changed stream minus real stream) is averaged over D+1 rows whose
target is 1 to 5 days after the injected day, as a share of 3 points.

- **Pass conditions.** On each holiday day, K0 carries a share of at least 0.10 into the second day
  after the injected day (the first day whose forecast sees the whole injected day), and K-skip and
  K-skip+1 carry 0.00 (to 1e-9). On the control day, K-skip's share is within 0.02 of K0's, so that
  skipping holidays does not damage the filter elsewhere. K-zero's change on the holiday targets
  themselves is exactly zero.
- **K0 reproduces experiment 115.** The K0 frame's q01, q05, q50, q95, and q99 match experiment 115's
  stored K0 frame to 1e-3 points at every site.
- A failed condition is reported and capped as stated in the decision rule.

## Compute and process

CPU only, at most 4 cores. Each filter run over a site's grid takes about 1 second, and a frame
build for four arms and six substations takes a few minutes. Every script runs under
`systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue`. Reviews: one Sonnet plan
review, one Sonnet code review, one Sonnet numbers review, and one Sonnet prose review. No Opus
review unless the ML manager asks.
