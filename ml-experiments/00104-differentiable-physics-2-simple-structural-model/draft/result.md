---
id: "00104"
slug: differentiable-physics-2-simple-structural-model
title: "Differentiable physics, part 2: a simple structural model of demand and solar"
part: 2
theme: differentiable-physics-demand-calendar-and-weather
also_relevant_to: [retraining-and-training-history, solar-disaggregation]
status: planned
size: >-
  complex for the framework module (new tested code in packages/studies, reused by Parts 3 to 6);
  medium for the experiment scripts. Stored: private files under
  data/ml-experiments/00104-differentiable-physics-2-simple-structural-model/ and a feature cache
  beside them; no contract or production table. Serving path: not touched. Degradation rule: not
  touched. Designs: more than one is defensible for the module (loss, spread model, drift form,
  operator interface), and this plan pre-registers one. Callers: the module's callers are this
  part's scripts and the later parts of the series only.
hypothesis: >-
  A convex demand-minus-solar model, trained on every usable year with a slowly drifting level per
  hour band, forecasts Primary demand better than the same model trained on the last 12 months.
prediction: see "Predictions"
decision_rule: see "Decision rule"
benchmark: >-
  Lagless XGBoost on the same ERA5 and CAMS inputs, with and without experiment 64's solar offset
  (am), each with a matching zero-start arm; climatology; manual_heuristic
row_set_id: dp_mw_v1 (experiment 103), six MW long-history sites, scored 2025-07-01 to 2026-06-30
metric: >-
  Quantile score (twice the mean pinball loss over experiment 87's 11 quantiles) in points of
  effective capacity, on rows outside the hand-filtered switching events at the scored site
verdict: <supported | partly supported | small gain | not supported | refuted | inconclusive>
prod: no
take_home_message: <written after the results>
date: <date>
pr: <pr>
advice: <written after the results>
---

# Differentiable physics, part 2: a simple structural model of demand and solar

This page is the plan. It is written before any fit, and the sections that need results are left
blank. The series brief (`briefs/dp_series.md`) and the part brief (`briefs/dp_part2.md`) hold the
reasoning; this page records what Part 2 will do, with the three points the manager settled on
2026-10-19.

## Settled before the first fit

- **Capacity source.** Experiment 45's `capacity_weekly_vintages.parquet`, variant `frozen`, taken
  from the latest `origin_monday` on or before each training cut. It has no `recommended == True`
  rows. The write-up states whether `frozen` equals the season-year constant of the hindsight file
  `capacity_weekly.parquet`, checked site by site. The vintage covers every training week back to
  2019, so no growth-prior scaling is needed where the vintage reaches; the write-up counts the
  training rows that needed it. The unobserved weeks of D14 take the last observed value.
- **Verdict contrast.** C1 (years) answers Part 2's own question. C2 and C3 (gaps to lagless
  XGBoost with and without the solar offset) are reported beside it.
- **Lagged reference.** Descriptive only: XGBoost with the three power lags of experiment 87,
  trained on the same 12 months, scored at leads of 24 to 48 hours (initialisation at 00:00 UTC on
  the day before the valid day) and, where rows allow, at 7 days (leads of 144 to 168 hours,
  initialisation at 00:00 UTC six days before the valid day).
  The no-lag arm X-12am is scored on the same rows. SIMP1's roughly 0.9 point penalty for removing
  the lags is quoted beside it.

## What Part 2 builds

- **The module** `packages/studies/src/studies/dp_*.py`, in its own reviewed pull request: a
  feature-tensor cache (calendar, temperature and its four smoothings, solar elevation, unit PV
  shape, season-year, band, a row mask, a per-row node list), a per-row observation operator
  (`identity`, `apparent`, and a typed `sum_of_nodes` stub), the convex GAM (design in chunks,
  Gram accumulation, a bound-constrained penalised least-squares solver, drift D1, steps D3, the
  forward cross-fitted residual-quantile spread), a pooled PyTorch configuration with the
  interaction network for the GPU benchmark, switches for every later module, and a leading batch
  dimension.
- **The six unit tests** of the part brief (closed-form ridge, constraints, known answer,
  observation operator, leakage, device agreement). Scoring does not start until all six pass.
- **The experiment scripts** `ml-experiments/00104-*/e00104_*.py`: tuning, the four GAM arms, the
  XGBoost arms, the cold block, the report and charts, and the GPU benchmark.

## Arms

Every arm is screening tier: one frozen fit on data before 2025-07-01, scored through 2025-07-01 to
2026-06-30, one seed, no refits. The lockdown window and the masked rows are left out of training
(`default_rows`). Scored sites: D5, D9, D10, D11, D14, and D16. The decision count uses D5, D10,
D11, D14, and D16, since D9 holds 9.0 months at 2024-07-01.

| Arm | Model | Window | Drift | Start |
|---|---|---|---|---|
| G1 | GAM | 2024-07-01 to 2025-06-30 | none | |
| G-all | GAM | first usable date to 2025-06-30 | none | |
| G-all+D1 | GAM | first usable date to 2025-06-30 | D1 | |
| G-all+D1+steps | GAM | first usable date to 2025-06-30 | D1 plus D3 | |
| X-12 | XGBoost | 12 months | | estimated base score |
| X-12z | XGBoost | 12 months | | zero |
| X-12am | XGBoost | 12 months | | zero plus solar offset |
| X-allz | XGBoost | all years | | zero |
| X-allam | XGBoost | all years | | zero plus solar offset |

Exploratory extras, labelled as such: the two lagged references above (L-12am-d1 and L-12am-d6).
The cross-fits that fit each site's solar weight `w` train a median-only XGBoost model with the same
parameters as the arm, since only the median residual is used. Climatology and
`manual_heuristic` are the reference rows.

## Design choices fixed here

- **GAM terms** as in the part brief: day type × 48 half-hours; one coefficient per hour band for
  each named holiday, Easter Saturday and Sunday, bridge days, and the Christmas run (24 and 27 to
  31 December); a tent and a signed tent for days to Christmas inside ±14 days; two day-of-year
  harmonics per band; non-negative slopes on relu(15.5 − T_k) for k = 6, 24, 72, 168 hours per
  band (trailing means of hourly ERA5 temperature); four saturating piecewise-linear ramps of
  darkness in solar elevation (knots −6°, 0°, 6°, 12°); and one non-negative, non-decreasing
  capacity per season-year times the unit PV shape (tilt 35°, azimuth 180°, DC to AC 1.15, Erbs
  split of CAMS GHI, derating 0.4% per °C of cell temperature). Four hour bands: night 00:00 to
  05:59, morning 06:00 to 10:59, midday 11:00 to 15:59, evening 16:00 to 23:59 local.
- **Fit.** Penalised least squares in points of effective capacity, solved on the Gram matrix with a
  projected-Newton active-set method for the sign constraints (heating slopes, monotone capacity
  increments, and the two halves of the step path). The scoring year uses the last trained
  season-year's capacity.
- **Spread.** The 11 quantiles are the mean plus empirical quantiles of forward cross-fitted
  residuals per hour band and calendar quarter. The training window's last 12 months are split into
  4 quarters; each quarter is predicted by a fit on the data before it, with the same window rule
  as the arm (G1 uses the 12 months ending at the quarter's start) and drift held.
- **Tuning** on 2024-07-01 to 2025-06-30, fitting on data before 2024-07-01, by mean absolute
  error of the mean: ridge penalty {0.1, 1, 10} on G-all (the grid first read {1, 10, 100}; a synthetic
  check before any scored fit showed a ridge of 10 shrinking heating slopes by 30 to 45%, so the grid moved down); drift step {3×10³, 3×10⁴, 3×10⁵} (scaled
  by training site-years) on G-all+D1; step penalty {100, 300, 1000} (scaled the same way) on
  G-all+D1+steps. D9 is left out of the tuning score. The scoring year is never used to tune.
- **XGBoost arms.** Experiment 87's Primary parameters (eta 0.05, depth 6, subsample 0.8, 500
  rounds, 11 quantiles), no lags, uniform row weights inside each window, the inputs the part
  brief lists. The offset is w · s · a(t) with `w` fitted per site on the cross-fitted residuals of
  the matching zero-start arm over April to September.
- **Scoring rows.** All present rows in the scoring year outside `switching_mask` at the scored
  site (headline), and all present rows (beside it). No valid time on or after 2026-07-01.
  Predictions are written once per valid time, against a day-ahead initialisation at 09:00 UTC on
  the previous day.

## Changes after the plan was committed

- **The ridge grid** moved down before any scored fit, as the design choices above record.
- **The spread** is fitted on a replica of the scored forecast (one fit 12 months before the cut,
  predicting the following 12 months) and not on four quarter-by-quarter cross-fits: an Opus review
  showed the quarters were only 0 to 3 months stale while the scored fit is 0 to 12 months stale.
- **Post-hoc arms**, added after a first look at the scored arms and labelled post hoc in the
  write-up: `x_all` (XGBoost, all years, default start),
  the `amm` arms (training median plus the solar offset), and the `ng` arms (no irradiance input).
  The first look showed the zero start scoring worse than
  the default start and the solar offset adding nothing when CAMS irradiance is an input. The
  planned contrasts C1 to C3 and the verdict are unchanged.
- **The lagged references** use the `amm` arm as their lagless counterpart.

## Planned contrasts

Three contrasts, Bonferroni-corrected (98.33% intervals from `studies.bootstrap`, whole calendar
months, paired across arms), each with the count of sites moving each way and every arm's absolute
score:

- **C1, years (the verdict contrast):** G-all+D1 minus G1.
- **C2, GAM against XGBoost on 12 months:** G1 minus X-12am.
- **C3, GAM against XGBoost on all years:** G-all+D1 minus X-allam.

A gain over XGBoost is claimed only from C2 or C3. Everything else is exploratory at 95%.

## Predictions

- **Framework:** unit tests 1 to 6 pass before any scored fit.
- **Old years without drift hurt:** G-all minus G1 is +0.3 to +0.8 points, interval wholly above
  zero. If this fails, C1 is not read.
- **Solar offset:** X-12am minus X-12z is −0.10 to −0.40 points with the interval below zero, and
  each site's `w` lies between −1.3 and −0.6.
- **Zero start:** X-12z minus X-12 is between −0.10 and +0.02 points.
- **Baselines:** climatology and `manual_heuristic` score worse than every model arm at every site.
- **C1:** +0.0 to +0.4 points, interval spanning zero, so `inconclusive`.
- **C2:** G1 is 0.1 to 0.5 points worse than X-12am. **C3:** G-all+D1 is 0.0 to 0.5 points worse
  than X-allam.
- **Steps:** G-all+D1+steps within ±0.1 points of G-all+D1.
- **Tails:** every GAM arm's q01 and q99 coverage within 1 percentage point of nominal, and q05 and
  q95 within 2 points.
- **Capacity:** the GAM's 2024-25 capacities lie within ±20% of experiment 45's vintages at D10,
  D11, D14, and D16.
- **Run time:** each arm under 10 minutes; tuning, arms, and cold block under 2 hours of CPU in
  slot C.

## Decision rule

The verdict comes from C1 under the series brief's Phase A rule, with eligibility counted at
2024-07-01. `helps` (provisional) if the C1 estimate is below zero and at least 4 of the 5 eligible
sites improve. `hurts` if the interval lies wholly above zero. `inconclusive` otherwise, including
any failed positive control. Under the series framing, `hurts` does not count against the
long-history line. C2 and C3 get the same labels in the bullets, not in `verdict`. A GAM more than 1
point worse than X-12am at 3 or more sites is reported to the manager before Part 3 starts. A
failed framework test stops scoring until it is fixed.

## GPU benchmark

About 15 minutes with other GPU work paused (the manager is asked first). Same code and tensors,
float32 where it agrees. (1) G-all+D1 on the six sites, CPU against GPU, wall time from cached
tensors to stored predictions. (2) 20 month-block bootstrap weightings of the same fit as one
batched model. (3) The pooled configuration with the interaction network (two hidden layers of
64) on all MW sites plus the MVA dark rows, 200 optimiser steps. The DP uses the GPU from Part 3 on
if workload 3 runs more than 3 times faster end to end; workloads 1 and 2 must agree with the CPU
within 0.01 points. Peak GPU memory is recorded.

## Results

To be written after the fits.
