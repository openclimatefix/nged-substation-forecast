---
id: '00191'
slug: dp-price-sensitive-assets-in-the-history
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - switching-labels-in-the-forecast
  - differentiable-physics-forecast-engine
status: done
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
verdict: not supported
production_status: not-for-production
adopted_in: none
best_arm: >-
  None. At D8 the hindsight arm H (true price at forecast time) scores -0.64% of the reference's mean
  absolute error (95% interval -2.51% to +1.12%), against a pre-registered bar of -5%.
headline: >-
  Letting the model see the price in its history does not improve the forecast at D8 or D14. The
  ceiling that uses the true price at forecast time gains 0.6% at D8, and every arm that could run
  live is no better than a model with no price input.
lessons:
  - >-
    At D8 the price columns explain 2.7% to 4.8% of the error on the rows they are fitted to, so even
    a perfect price forecast could not have reached the pre-registered 5% bar.
  - >-
    The fitted price response at D8 is 1.2 to 1.9 MW, the size of a gas engine's block, but it does
    not carry forward: out of sample the best arm gains 0.6%, with an interval that includes zero.
  - >-
    At D14, where the register places the 6 MW gas engine, the fitted response is 0 to 0.14 MW. The
    engine's footprint appears at D8, not D14.
  - >-
    Removing the price-driven half-hours from training (T) and replacing the price by its expected
    value (M) do not help, and persisting yesterday's price (P) hurts D8 by 4.4%.
take_home_message: >-
  Modelling a price-following asset in the history does not make the forecast better at D8 or D14.
  A model that is given the true price at forecast time, which no live service has, gains only 0.6% at
  D8, and the models that use a price available at forecast time lose 0.9% to 4.4%. The gas engine's
  response is real enough to fit (1.2 to 1.9 MW at D8 within the training rows), so the remaining
  D8 error comes from something the price does not explain, and NGED's half-hourly output for the
  engine is the next thing to ask for.
advice:
  - >-
    Do not add half-hourly price indicators to the demand model for D8 or D14: the hindsight ceiling
    gains 0.6% at D8 and loses 0.5% at D14 (Figure 1).
  - >-
    Ask NGED for the half-hourly output of the gas engine registered near D8, because the price
    columns reach at most 4.8% of the error even on their own training rows (Results, in-sample gain).
  - >-
    Fit D8's spread on D8's own residuals before adding another input: the q05 to q95 interval covers
    42% of readings and the q05 quantile sits above 31% of them (Results, tails).
  - >-
    Treat D14 as the site where the register lists a gas engine but the data shows no price response;
    do not gate a D14 model on the connection date (Figure 2).
---

# Experiment 191: can price-following assets be modelled in the history so the forecast improves?

**Take-home message: modelling a price-following asset in the history does not improve the forecast
at D8 or D14.** D8 and D14 are two substations of National Grid Electricity Distribution (NGED) in
the Boston area. [Experiment 186](../00186-switching-d8-gas-peaker-spark-spread-and-triads/result.md)
found that D8's multi-day drops in demand look partly like a gas engine that runs when electricity is
dear relative to gas. A price cannot be forecast days ahead, so we asked a narrower question: if the
demand model sees the price in its training history, does it learn a cleaner picture of the rest of
the demand, and does the forecast improve even when the price is unknown at forecast time? We added
four price indicators to a convex demand model, each lowering net load by a fitted, non-negative
amount, switched on from the connection date of a registered 6 MW gas engine. We tested four ways of
using them at forecast time and scored the forecasts on a year that the fits had not seen. The best
arm, which uses the true price at forecast time and so cannot run live, lowers D8's mean absolute
error by 0.6%, with an interval that includes zero. The pre-registered bar was 5%, so the verdict is
`not supported`. The model reproduces earlier work to numerical precision (see Checks), the gas
engine's response is visible in the fitted coefficients, and it still does not carry forward.

![Mean absolute error of each price-aware arm minus the reference's, at D8 and D14](figure_1_error_against_reference.svg)

![Fitted price response in MW at each monthly fit, D8 and D14](figure_2_fitted_price_response.svg)

## What was built

- **The register.** NGED's Embedded Capacity Register (August 2026) lists one connected gas engine
  that is not combined heat and power under the Primaries with metered series: 6 MW, connected on
  2023-06-13, 0.4 km from D14's coordinates and about 0.8 km from D8's. D8 itself lists only solar and
  a 1.35 MW row of unknown type. D5 has a 1 MW gas engine connected in 2021. Batteries of 0.2 to 0.25
  MW are connected at D2, D9 and D11 (2024 to 2026), and two more are accepted but not connected, at
  D3 and D16. The demand-side-response sheet holds no rows. The eight other Primaries that the
  register covers list no price-following asset (`printed_ecr.txt`).
- **Reference (A0).** The ensemble-driven convex demand model of
  [experiment 152](../00152-dp-mva-fold-aware-gam-k-at-mva-only-primaries/result.md) (fitted solar
  capacity, monthly refits on a 24-month window, shared Kalman filter, spread fitted on tuning months),
  with no price input. At D8 it has 12 monthly fits from 3 to 14 months of history, because the series
  is valid only from 2025-04-01. At D14 it has 27 fits. It is not the best-guess stack of
  [experiment 155](../00155-dp-bg-stage2-assembly-night-offset-and-error-correction/result.md): the
  winter night offset and the 14-day error profile are absent, because D8 has no tuning months to fit
  the offset on. At D14 the reference's mean absolute error is 3.84 points against 3.69 for the
  best-guess stack.
- **Arms.** H uses the true price at forecast time (a ceiling). M uses the expected indicator given
  the half-hour of the day, day type and quarter. P uses the last settled day's indicator at the same
  half-hour, the only price known at the 09:00 UTC origin. T removes the half-hours that the fitted
  price term explains from training and forecasts without a price term. A diagnostic arm (JB) is the
  price-aware fit's base model with its price term switched off at forecast time. The learned
  monotone price curve (arm N) was not run.
- **Price indicators.** System price above its 50th, 75th and 90th percentile, and a positive spark
  spread at 40% efficiency (system price minus gas cost per MWh of electricity minus 5 pounds per
  MWh), from public data, 2023-06 to 2026-06. The gate is 1 from the engine's connection date.
- **Masking rule (fixed after one coefficient check, before any score was read).** T drops a training
  half-hour when the fitted price output there is at least 25% of the largest fitted output and that
  largest output is at least 1 point (1% of the substation's effective capacity). Without the
  1-point floor the rule dropped half of D14's training rows for a response of 0.04 MW.

## Results

- **Absolute scores (D8, scored year, all rows, leads 0.5 to 168 hours).** Reference: mean absolute
  error 15.18 points (1.6 MW), quantile score 10.95, q05-to-q95 coverage 0.42. D14: 3.84, 2.16, 0.89.
- **Mean absolute error against the reference, D8.** H -0.64% (-2.51% to +1.12%), M +0.88% (-0.39%
  to +2.27%), P +4.41% (+3.02% to +5.82%), T +3.44% (+0.82% to +6.40%), JB +5.73% (+2.57% to
  +8.65%). The quantile score gives the same signs: H -1.04%, M +1.00%, P +5.36%, T +4.56%.
- **Mean absolute error against the reference, D14.** H +0.49% (+0.23% to +0.78%), M +0.02%, P
  +0.16%, T +0.14%, JB -0.13% (-0.54% to +0.24%).
- **High-spread half-hours (spark spread above zero), D8.** The reference's error is 16.37 points
  against 13.96 on other half-hours. H -0.74%, M +1.08%, P +3.73%, T +5.27%. No arm is better where
  the price signal should be strongest.
- **By lead, D8, H.** The quantile score falls 2.2% (interval -4.3% to -0.1%, not corrected for the
  number of cuts) at 0 to 24 hours, 1.8% at 24 to 72 hours and 0.4% at 72 to 168 hours.
- **In-sample gain (the ceiling).** On each fit's own training rows the price columns lower D8's mean
  absolute error by 4.1% on average over the 12 fits (2.7% to 4.8%). The fitted response, with all
  four indicators on, is 1.9 MW at the first fit and 1.2 MW at the last. At D14 the in-sample gain is
  0.0% and the response is 0 to 0.14 MW. At D5 it is 0.0%.
- **No-harm check.** The largest rise in quantile score at any of the five decision Primaries (D5,
  D10, D11, D14, D16) is 0.011 points, at D14 for H, against the 0.02 limit. With the gate switched on
  for the whole history at sites with no asset (placebo), D10 rises by 0.009 points for H.
- **Tails.** At D8 the reference's q05 quantile lies above 30.8% of readings (nominal 5%), q95 above
  73.1% (nominal 95%), q01 above 20.6% (1%) and q99 above 83.9% (99%). H changes none of them by more
  than 0.5 points. D14 is close to nominal (q05 6.4%, q95 95.1%). The D8 spread comes from six other
  Primaries and is far too narrow for D8, so D8's quantile scores are weaker evidence than its mean
  absolute errors.
- **Post hoc: settlement-day indicators.** Replacing the half-hourly indicators by the settlement
  day's mean price and spread lowers nothing: D8 H +2.23% (+0.41% to +4.29%), M +2.40%, P +3.83%, T
  +5.75%.
- **Not computed.** The curtailment-cost metric (it needs a node and its generators) and the
  climatology and manual-heuristic baselines.
- **Schedule.** Monthly refits on a 24-month window, as in experiment 152, not weekly refits, so the
  absolute scores are not production-setting scores. The contrasts could depend on the schedule at D8,
  where a longer history would help the price columns.

## Checks

- **Control.** The reference reproduces the stored experiment 152 arm N at D8 (mean absolute gap
  1e-12 on 97,012 rows) and `studies.dp_gam.fit_gam` at one cut (largest coefficient gap 2e-10).
  Where the gate is off (D10, D11, D16) every arm equals the reference.
- **Median minus mean forecast.** 0.000 points at every site and arm, because the spread is
  symmetric about the corrected mean.
- **Mean-forecast error beside the quantile score.** Both are reported in Results and have the same
  sign in every contrast of the main tables.
- **Audit triggers.** Trigger 2 fired: H gained 0.6% against a pre-registered 5%. The explanation was
  tested and is the in-sample figure: even on the rows the price columns are fitted to they explain at
  most 4.8%. Trigger 5 fired: at D14 H, which contains the reference, is 0.49% worse than the
  reference. JB, which keeps the base fit and drops the price term, is within 0.13% of the
  reference, so the loss comes from the price term, whose coefficients move between 0 and 0.9 points
  from fit to fit. No Opus audit was run, because neither result changes the next step. Triggers 1, 3,
  4, 6 and 7 did not fire.

## What the next experiment should answer

- **Does a spread fitted on D8's own residuals change the picture?** The reference covers 42% of
  readings where it should cover 90%, so D8's interval scores are weak evidence.
- **Does a wider band on spark-spread-positive days help?** D8's error is 17% larger on those
  half-hours. This was not tested.
- **What does the engine's own output show?** Ask NGED for its half-hourly output, because the
  price indicators reach at most 4.8% of the error on their own training rows.
