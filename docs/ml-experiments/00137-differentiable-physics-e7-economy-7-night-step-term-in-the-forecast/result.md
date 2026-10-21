---
id: "00137"
slug: differentiable-physics-e7-economy-7-night-step-term-in-the-forecast
title: >-
  Differentiable physics, E7 forecast: does an Economy 7 night-step term lower the day-ahead error
  on winter nights?
part: E7F
theme: differentiable-physics-demand-calendar-and-weather
also_relevant_to: [differentiable-physics-forecast-engine, production-reference-and-best-guess]
status: planned
size: >-
  medium. Stored: private files under
  data/ml-experiments/00137-differentiable-physics-e7-economy-7-night-step-term-in-the-forecast/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one is
  defensible (the term's form, whether sites share coefficients, how the forecast temperature
  enters), and the plan pre-registers one of each. Callers: this folder's scripts only, which read
  experiment 133's stored prediction frames and call the reviewed `studies.bootstrap` and
  `studies.tail_calibration` modules.
hypothesis: >-
  An additive night term, fitted on the base forecast's own past errors as a function of the
  ENS-mean daily temperature and the UTC clock time, lowers the day-ahead quantile score on the
  half-hours from 01:00 to 03:00 UTC on cold winter nights, without raising the score elsewhere.
  The mechanism is that the base GAM-K forecast (experiment 133's shared-noise arm) runs below
  actual demand from about 01:00 on cold nights (experiment 123), so a correction learned from
  earlier winters removes a bias the forecast repeats.
prediction: >-
  C1 (the per-substation term minus the base, quantile score on the E7 window rows of cold winter
  nights at leads 12 to 36 hours, scored year, six decision substations) is -0.5 to -2.0 points.
  The term recovers at least half of the oracle's gain (the oracle is the same term fitted on the
  scored year itself). The mean forecast's error on the same rows (actual minus mean) shrinks by
  at least a third. The all-rows quantile score changes by -0.03 to 0.00 points.
decision_rule: >-
  Five planned contrasts, each read at the 99% level of the month-resampled paired bootstrap (the
  12 calendar months of the scored year, 20,000 resamples). C1 is the headline and is `supported`
  when its interval lies wholly below -0.3 points, the non-inferiority row C4 has an upper bound
  of at most +0.01 points, and at least 4 of the 6 substations improve. The threshold of -0.3
  points is set by cost: the term adds a fitted component with 72 coefficients per substation
  and a clock rule fixed to UTC, so it must beat noise by a margin an operator could see.
  It is `small gain` when the interval lies wholly below zero but not below -0.3 points, or fewer
  than 4 of 6 substations improve. It is `refuted` when the interval lies wholly above zero. It is
  `inconclusive` otherwise. The verdict is capped at `inconclusive` if the injected-step control
  or the shuffled-night placebo fails (see "Controls"). C2 (pooled-coefficient term minus base),
  C3 (experiment 123's fixed step minus base), C4 (per-substation term minus base on every row
  of the scored year, all leads) and C5 (per-substation term minus pooled term on the C1 rows) are
  read by the same intervals; C3 is a reading of the upper bound and carries no verdict.
benchmark: >-
  The base arm: experiment 133's `pool` arm (GAM-K on the ENS ensemble mean with the shared
  noise setting), as stored; climatology and manual_heuristic as in experiment 111 for reference
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over experiment 87's 11 quantiles) in points of
  effective capacity, on rows outside the hand-filtered switching events, at the six decision
  substations D5, D9, D10, D11, D14, and D16, at leads 12 to 36 hours, then winter nights and
  the E7 window rows; coverage and pinball loss at the 0.01, 0.05, 0.95, and 0.99 quantiles
code: ml-experiments/00137-differentiable-physics-e7-economy-7-night-step-term-in-the-forecast/
---

**Status: planned. No code has run.** The plan below was committed before any fit.

## Question

Experiment 123 found that demand at 13 of 14 Primary substations steps up around 01:00 UTC on
winter nights, by about 10 points at a daily mean of 3 C, and that the day-ahead forecast of
experiment 114 runs below actual demand from about 01:00 on cold nights. An operator sees that step
and the forecast's miss. This experiment asks whether an explicit night term, learned from the base
forecast's own past errors, closes the miss.

## Design

- **Base.** Experiment 133's `pool` arm: the convex GAM with a Kalman filter (GAM-K) on the ENS
  ensemble mean, with one shared noise setting. Its prediction frames are read from experiment
  133's private folder (tuning months 2024-07 to 2025-06, odd and even, and the scored year).
  No file of the best-guess assembly is edited.
- **The term.** For a half-hour at UTC clock time `h` in the night set (21:00 to 08:30 UTC, 24
  half-hours) on the night of evening date `D`, the correction to every quantile is
  `a[h] + b[h] * cold + c[h] * weekend`, where `cold = max(0, 15.5 - T)`, `T` is the ENS-mean
  temperature averaged over the half-hours of UTC day `D` that follow the forecast origin, and
  `weekend` flags Saturday and Sunday evenings. Coefficients are a ridge regression of the base
  forecast's error (actual minus the mean forecast) on those 72 columns, fitted per substation
  (`T_SITE`) or shared across the six substations (`T_POOL`), on all rows whose valid time is
  before the first of the month being forecast (an expanding window, refitted monthly, starting
  from 2024-07). The ridge penalty is chosen on the odd tuning months from a fixed grid. The
  spread around the shifted median is unchanged.
- **The clock is UTC.** Experiment 123 found the step follows UTC on British Summer Time nights
  (a post hoc finding at 13 of 14 substations). The term therefore uses UTC and does not test
  that choice.
- **Two reference arms.** `FIXED123` inserts experiment 123's estimated step (the pooled
  `fit_headline` step at the forecast temperature, with its estimated ramp) minus the step the
  base already forecasts on the same night. `ORACLE` is `T_SITE` fitted on the scored year's own
  rows, so it bounds what any term of this form could gain.
- **Weather at forecast time.** `T` is the ENS-mean forecast available at the origin. No ERA5
  value enters any score. Experiment 123's winters are used only by `FIXED123`, and end before the
  scored year.
- **Limits stated now.** The expanding window holds one training winter (2024-25) for the first
  scored winter months and grows through winter 2025-26. The scored year holds one winter, so the
  cold-night rows are few and the month-resampled interval will be wide.

## Rows and subsets

- Rows: `fold_a_daily_gap_v1`, 09:00 UTC init, six decision series, scored year (the runs
  stored by experiment 133).
- **E7 window rows**: valid time 01:00 to 03:00 UTC (the half-hours ending 01:30 to 03:00).
- **Cold nights**: ENS-mean daily temperature below 3 C on the evening date.
- **Winter**: November to March.
- Headline subset: E7 window rows, cold winter nights, leads 12 to 36 hours. Beside it: all
  winter nights, all E7 window rows, all rows, and a figure of the step against the forecast on the
  coldest nights.

## Controls (run before any scoring)

- **Injected step.** Add a synthetic step of known size to the `actual` of every night in the
  stored frames (a ramp from 0 at 00:30 to 8 points at 01:30 UTC, held to 08:30, plus 0.5 points
  per degree of `cold`), refit `T_SITE` and `T_POOL` with the same code, and read the fitted
  correction at 3 C. Pass: the fitted step at 01:30 to 03:00 UTC is within 25% of the truth
  (8 + 0.5 * 12.5 = 14.25 points) at 5 of 6 substations, from the first scored winter month on.
- **Shuffled-night placebo.** Permute `cold` across the nights of each substation and refit.
  Pass: the C1 contrast on the permuted term is within +/-0.1 points of zero.
- **Reproduction.** The stored base rows reproduce experiment 133's `pool` score at 12 to 36 hours
  to within 1e-4 points.
- **Stop rule.** If the injected-step control fails at 3 or more substations, stop, record the
  verdict `failed`, and write what was found. No scored contrast is read after a failed control.

## Compute

CPU only. The ridge fits are small: a run is expected to take minutes on 4 cores, well inside the
16 GB memory cap. Predictions are written under `data/ml-experiments/00137-*/predictions/` with
`q01` to `q99`, `actual`, `lead_hours`, and `time_series_id`, so `studies.tail_calibration` can
score them.

## Prior work followed

- Experiment 123's estimated step, timing, and cold slope (read first, not re-estimated).
- Experiment 133's stored frames, scoring and month-resampled intervals, copied by pattern.
- `advice` from experiment 123: judge on cold winter nights from 01:00; try the UTC clock.
  Departure: the term is fitted on the forecast's errors, not on the observed night mean, so that
  the part of the step the base already models is not added twice.
