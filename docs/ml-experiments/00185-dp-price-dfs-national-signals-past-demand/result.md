---
id: '00185'
slug: dp-price-dfs-national-signals-past-demand
part: BG-EB5
title: >-
  Do national system prices, Demand Flexibility Service events, or other national signals help the
  model understand past demand?
theme: differentiable-physics-forecast-engine
also_relevant_to:
- production-reference-and-best-guess
status: planned
size: >-
  medium. Stored: private files under
  data/ml-experiments/00185-dp-price-dfs-national-signals-past-demand/ only (public data, under 100
  MB). Serving path: not touched, because the signals are used in hindsight only. Degradation rule:
  not touched. Designs: more than one is defensible (which price summary, how ordinary days are
  matched, how the covariate enters the Kalman filter), and this plan fixes one of each before any
  result is read. Callers: this folder's scripts, which read experiment 155's arm D predictions and
  experiments 172 and 175's day tables, and replay experiment 136's arm A2 and experiment 155's
  arm D with the reviewed `studies.dp_*` modules unchanged. No new modelling module.
hypothesis: >-
  A national signal that experiment 177 found on many of the 37 unlabelled worst days (a high
  national system price, or a Demand Flexibility Service (DFS) event) explains part of the
  day-to-day variation in the best-guess forecast's loss over all days of the scored year, not only
  on the 37 worst days. Such a signal cannot be forecast, but a hindsight covariate may help the
  model interpret the past.
prediction: >-
  Written before any result is read. A price summary or a DFS flag explains at least 10% of the
  variance of arm D's daily excess loss beyond temperature and day type (cross-validated by
  month block). This prediction can fail, and I expect it to: experiment 177 found the price
  covering the worst days only 3 days more than ordinary days. The matched-ordinary-day comparison
  keeps a price excess of 1 to 3 days over 37, with an interval that includes zero. No other
  national signal beats price by more than 3 points of variance explained. The hindsight covariate
  moves the five decision substations' score by under 0.02 points.
decision_rule: >-
  Part 1. The unit is a substation-day (the six substations of experiment 172, D9 included in the
  pooled fit and shown apart). The outcome is arm D's daily excess loss (experiment 172's `excess`,
  D+1 band, rows outside hand-filtered switching events). The baseline model has site fixed effects,
  a natural cubic spline of the ERA5 day-mean temperature (4 degrees of freedom), and day type
  (weekday, Saturday, Sunday, bank holiday). The test model adds, fixed in advance, (a) the three
  price summaries (daily mean, daily maximum, evening-peak mean over 16:00 to 19:00 local), (b) the
  DFS flag, and (c) each of those multiplied by the day-mean temperature. The gain is the
  out-of-fold R-squared increment, 12 month blocks as folds, with a 95% interval from resampling
  the month blocks (2,000 resamples). The prediction holds when the increment of the full price +
  DFS model is at least 0.10 and its interval lies above 0.05. Otherwise the prediction fails, and
  the page opens by saying so. The 18-of-37 price excess survives when its p-value over 1,000
  draws of 37 matched ordinary days (same substation, same day of week, same season, day-mean
  temperature within 2 degrees, widened to 4 degrees if no day matches) is under 0.05 and the excess
  is at least 15 points (5.6 days). The same is run for the DFS flag (9 of 37).
  Part 2. The same out-of-fold increment for each national signal alone (NESO national demand,
  embedded wind and solar generation, wind generation, and net interconnector flow, each as a daily
  mean and an evening-peak mean), compared with the price's increment. A signal explains more than
  price when the paired difference of increments has a 95% interval above zero.
  Part 3. The best signal by out-of-fold increment in the tuning months (July 2024 to June 2025) is
  added as a day-level covariate to arm A2's Kalman filter (one coefficient per hour band, fitted
  on the tuning months, applied to the target day in hindsight) and arm D is rebuilt from it. A
  control with the coefficient set to zero must reproduce the stored arm D to within 1e-4 points.
  The contrast is the quantile score of the rebuilt arm D minus the stored arm D at the five
  decision substations over the scored year, D+1 band, with a 95% interval from the month-block
  bootstrap. The threshold is zero, since a hindsight covariate cannot go to production, and the
  result is labelled "not forecastable". Verdict: `supported` if the Part 1 prediction holds;
  `partly supported` if it fails but a Part 1 or Part 3 interval excludes zero in the predicted
  direction; `refuted` if nothing is distinguishable from zero.
benchmark: >-
  Experiment 155's arm D as stored (the best-guess forecast), the 37 matched ordinary-day draws of
  experiment 177, and the no-covariate baseline model of Part 1.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Out-of-fold R-squared increment on the daily excess loss (Parts 1 and 2); the quantile score (twice
  the mean pinball loss over the 11 quantiles, points of effective capacity) and the coverage and
  pinball loss of the 0.01, 0.05, 0.95, and 0.99 quantiles (Part 3).
code: ml-experiments/00185-dp-price-dfs-national-signals-past-demand/
date: 2026-10-22
---

**This page is the plan, written before any result was read.** The best-guess forecast (arm D of
[experiment 155](../00155-dp-bg-stage2-assembly-night-offset-and-error-correction/result.md)) loses
most on a few days. [Experiment 177](../00177-dp-worst-days-other-data-sources/result.md) found that
a high national system price covered 18 of the 37 unlabelled worst days (14.9 on matched ordinary
days, p = 0.075), and that Demand Flexibility Service (DFS) events covered 9 (8.5 on ordinary
days). The maintainer was never expecting one source to explain all 37 days. This experiment asks
whether those signals, and other national signals, explain the forecast's loss across every day of
the scored year, and whether giving the model the signal in hindsight lowers its score. System
price cannot be forecast, so the hindsight test only shows whether the signal helps the model read
the past.

## Pre-registration

- **Data (public, under 100 MB).** Elexon settlement system prices (the cached 2025-07-02 to
  2026-07-01 of experiment 177, plus 2024-07-01 to 2025-07-01 for the tuning months); the NESO
  DFS service-requirement archive; the NESO historic demand data for 2024 to 2026 (national
  demand, transmission system demand, embedded wind and solar generation, wind generation, and the
  interconnector flows).
- **Day definition.** A local (Europe/London) calendar day. Price and national signals are
  aggregated over the local day's settlement periods.
- **Weather at lag times.** The temperature in the Part 1 baseline is the ERA5 day-mean
  temperature of experiment 172. ERA5 arrives about 5 days late, so Part 1 is a diagnosis of past
  days and not a forecast-time model.
- **Part 3 design.** The filter's residual stream is reduced by the coefficient times the
  standardised covariate of the day, per hour band, and each forecast adds back the coefficient times
  the covariate of the target day. The covariate is known for the target day only in hindsight.
  Coefficients are fitted on the tuning months and fixed for the scored year.
- **Chosen before results.** The signal for Part 3 is chosen on the tuning months only.
- **Anonymisation.** Demand series only; the sites keep their D labels.
