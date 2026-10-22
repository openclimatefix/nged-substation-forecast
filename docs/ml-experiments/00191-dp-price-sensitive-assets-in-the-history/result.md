---
id: '00191'
slug: dp-price-sensitive-assets-in-the-history
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - switching-labels-in-the-forecast
  - differentiable-physics-forecast-engine
status: planned
size: >-
  medium. Stored: private files under
  data/ml-experiments/00191-dp-price-sensitive-assets-in-the-history/ only (public price files are
  reused from experiments 185 and 186; nothing is downloaded). Serving path: not touched. Degradation
  rule: not touched. Designs: more than one is defensible (which price features, how a masked row is
  chosen, what stands in for the unknown price), so this plan fixes one of each before any score is
  read. Callers: this folder's scripts, which copy the ensemble-driven GAM-K engine of experiment 152
  and call the reviewed `studies.dp_*` modules unchanged. No new modelling module.
hypothesis: >-
  Some of D8's multi-day drops (experiment 159) and the matching shortfalls at its neighbour D14 are
  output of price-following gas generation (experiment 186), which no weather input explains and no
  forecast of the price can reach at a lead of days. Letting the convex demand model (GAM-K) see
  the half-hourly price in its training history, gated by the registered connection date of the
  6 MW gas engine, lets it learn a cleaner baseline for the rest of the demand. The forecast then
  improves even when the price is unknown at forecast time.
prediction: >-
  Written before any score is read. H (hindsight: true price at forecast time, a ceiling that
  cannot be forecast) lowers D8's mean absolute error over all rows by at least 5% against A0, and
  more on high-spread half-hours. M (the same fit, with the price features replaced by their
  expected value given the hour, day type and season) or T (the price-driven half-hours masked out
  of training) gains under 1.5% on D8 in all rows, so the pre-registered threshold for a useful
  change is probably missed. I expect the cleaner baseline to give a small gain at most, because
  D8's series has only 12 monthly refits from 3 to 14 months of history. D14, with a pre-connection
  history to learn the break from, should show a clearer gain from T than D8. P (last settled day's
  price as the input) gains less than M. No decision Primary's score rises by 0.02 points.
decision_rule: >-
  Reference: A0, the ensemble-driven GAM-K (fitted solar capacity, monthly refits on a 24-month
  window, experiment 133's shared filter cell, spread fitted on tuning months) with no price input.
  Contrasts, all on the quantile score (twice the mean pinball loss over 11 quantiles, in points of
  effective capacity) and on the mean absolute error of the mean forecast: C1 = H minus A0 at D8,
  all rows. C2 = M minus A0, and C3 = T minus A0, at D8, all rows. C4 = P minus A0 at D8. The same
  four at D14. Each interval is the 95% month-block bootstrap (12 calendar months of the scored
  year, 20,000 resamples), Bonferroni-corrected to the 99.4% level across the eight contrasts per
  site pair. `supported`: C1 is at or below -5% of A0's mean absolute error at D8 with the interval
  wholly below zero (the upper bound), and at least one of C2 or C3 is at or below -1.5% with its
  interval wholly below zero and the no-harm check passes (no decision Primary's quantile score rises
  by 0.02 points or more). `partly supported`: the first half holds and the second does not.
  `not supported` otherwise. `inconclusive` when the interval of every contrast includes zero.
  The 1.5% threshold is set by cost: the fix adds an asset register, a price feed and a second
  fit to the weather layer, which a smaller gain does not pay for.
benchmark: >-
  A0 at the same site, the same rows and leads; climatology and the manual heuristic for absolute
  skill; where the site has them, experiment 136's stored arm A2 and experiment 155's arm D at D14,
  as a check on A0 (A0 omits T-all, the night offset and recipe R).
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score and mean absolute error of the mean forecast in points of effective capacity, over
  leads 0.5 to 168 hours, outside the hand-filtered switching events, in the scored year; also on
  half-hours with a positive spark spread at 40% efficiency; q05-to-q95 coverage; lead bands.
code: ml-experiments/00191-dp-price-sensitive-assets-in-the-history/
date: 2026-10-22
---

# Experiment 191: can price-following assets be modelled in the history so the forecast improves?

This page is the plan. The results are added after the scripts have run.

## Plan

- **Question.** Experiment 186 found that D8's multi-day drops look partly like a gas engine that
  follows the spark spread (the system price minus the cost of the gas burned at 40% efficiency).
  The August 2026 Embedded Capacity Register lists a 6 MW gas engine, not combined heat and power,
  connected on 2023-06-13, 0.4 km from the Primary labelled D14 and about 0.8 km from D8. The price
  cannot be forecast days ahead, so the question is whether the model can use the price to understand
  the past and forecast the rest better.
- **Sites.** D8 first (apparent power only, series valid from 2025-04-01, 12 monthly refits from
  2025-07). D14 as a second test site, because the engine is registered under it and its history
  runs back to 2020, so the connection date is a structural break the fit can learn. D5 (a 1 MW
  gas engine connected in 2021, gate always on) and D10, D11 and D16 (no price-following asset
  connected, gate always off) as the no-harm check.
- **Arms.** A0 (the reference). H (hindsight price at forecast time). M (price features replaced at
  forecast time by their training-period expected value given the half-hour of the day, the day type
  and the quarter). T (the half-hours that H's fitted price term explains are dropped from training;
  the forecast model has no price term). P (the last settled day's price at the same half-hour of
  the day, the only price known at the 09:00 UTC origin; the day-ahead auction price is published
  after the origin and is not used). N, a learned monotone price-response curve, runs only if the
  above leave time and GPU budget.
- **Price features.** Four non-negative indicators of the half-hour, multiplied by the asset's
  gate (1 from the connection date): system price above its 50th, 75th and 90th percentile (fixed
  thresholds from 2023-06 to 2026-06), and a positive spark spread at 40% efficiency with a 5 pound
  per MWh operating cost (experiment 186's assumptions; days with a gas price above 15 pence per
  kWh are data errors and are excluded). Each enters the design with a minus sign and a
  non-negative coefficient, so a price-following asset can only lower net load.
- **No-harm check.** Where the gate is always off, the price columns are zero and every arm equals
  A0 by construction. The plan also runs M with the gate switched on for the whole history at the
  sites with no asset, as the stricter test.
- **Stated simplification.** A0 is the ensemble-driven GAM-K with fitted solar capacity (experiment
  152's engine), not experiment 155's arm D. D8 has no pinned PV capacity vintage, and its series
  restarted in April 2025. T-all, the winter night offset and the 14-day error profile of recipe R
  are added to every arm at the end only if the score of A0 differs from arm D at D14 by more than
  the contrasts under test.
- **Past weather.** The price at lag times is public and settles within the hour. The ERA5
  temperature and CAMS irradiance used in training exist only after the fact, as in every earlier
  experiment of this series; the ensemble mean drives the forecast.
