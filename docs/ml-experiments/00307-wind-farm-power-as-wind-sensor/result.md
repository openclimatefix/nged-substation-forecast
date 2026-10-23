---
id: "00307"
slug: wind-farm-power-as-wind-sensor
title: >-
  Which property of the ensemble's wind speed is wrong, judged against the wind that the farms' own
  power implies?
theme: generator-forecasting
also_relevant_to:
  - weather-products-and-blending
status: finished
date: 2026-10-23
size: >-
  medium. Stored: private tables under data/ml-experiments/00307-wind-farm-power-as-wind-sensor/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one defensible
  way to correct a speed series, so each correction is a registered arm. Callers: this folder's
  scripts, which copy experiments 261 and 285. One Sonnet code review and one Sonnet write-up and
  prose review.
hypothesis: >-
  The wind that the three metered wind farms' own power implies (the fitted physics core's power
  curve run backwards) is closer to a high-resolution analysis than to ERA5 at 9 to 14 m/s, and at
  least half of the ensemble's shortfall (observed power below the forecast's median on 34.1% of
  rows instead of 50%) is in the spread and timing of the members, not in their mean speed.
prediction: >-
  Written before any run. (1) The implied wind agrees with ERA5 to within 1 m/s on average at the
  invertible speeds. (2) The mean-speed correction (a) explains under 50% of the 18.7-point gap
  between the ensemble mixture's 34.1% and the ERA5 oracle's 52.8% of kept rows below the median,
  and the spread (b) and timing (c) corrections together explain at least 50%. (3) Where
  ICON-D2 is available, its first forecast hours are closer to the implied wind than ERA5 is at 9 to
  14 m/s.
decision_rule: >-
  Descriptive attribution, with no hypothesis test on a score. Supported when (1) holds, (2) holds
  and the first-hours ICON-D2 or UKV wind is closer to the implied wind than ERA5 is at 9 to 14 m/s.
  Partly supported when (2) holds alone or (3) holds alone. Not supported when the mean explains over
  70%. Stop after step 2 if the implied wind differs from ERA5 by more than 1 m/s on average at the
  invertible bins (the curve fit is then the issue). Shares carry 95% intervals that resample the 12
  months of the scored year; farms are not resampled. Nothing here goes to production, so there is
  no porting threshold. Cap at 5 hours.
benchmark: >-
  Experiment 261's stage shares for the same rows: ERA5 oracle 52.8%, ensemble-mean wind 36.6%,
  51-member mixture 34.1% of kept rows below the median; experiment 285's quantile mapping by lead
  day, 37.5%
row_set_id: >-
  experiment 33's wind cases (`fold_a_tuesdays_v1`: 53 Tuesday runs from 2025-07-01 to 2026-06-23,
  W1 to W3); corrections are fitted on experiment 259's 442 daily hindcast runs of 2024-04-01 to
  2025-06-16. Headline on the rows kept by experiment 210's mask, all rows beside it.
metric: >-
  Share of rows where observed power is below the forecast's median (calibrated: 50%), in points of
  PIT share explained out of the 18.7-point gap; implied-minus-reference wind speed in m/s by speed
  bin; quantile score in points of capacity as a secondary check
code: ml-experiments/00307-wind-farm-power-as-wind-sensor/
verdict: not supported
best_arm: >-
  Not an arm contrast. One speed factor of 1.13 fitted to the implied wind raised the share of
  observed power below the ensemble's median from 34.1% to 48.6% of kept rows, closing 77.6% (66.4%
  to 92.4%) of the 18.7-point gap to the ERA5 oracle; the per-lead-day factor closed 69.8%
headline: >-
  The ensemble's shortfall is a speed-level bias, not a spread or timing problem. A speed factor
  closes 70% to 79% of the gap; widening the members closes none (-0.9%) and shifting them by up to
  3 hours closes 0% to 5%.
production_status: not-for-production
adopted_in: none
lessons: >-
  The ensemble's 100 m wind is about 1.1 m/s slower than the wind that the farms' own power implies,
  from day 0 and growing with lead. A plain speed factor fixes most of that, and experiment 285's
  quantile mapping did not, so the form of the mapping, not the level it was fitted at, was the weak
  step.
take_home_message: >-
  The ensemble's low wind-power forecast comes mostly from wind speeds that are too slow, by about
  1.1 m/s on average and more at longer leads. Multiplying every member's wind speed by a factor
  fitted on the previous 14 months of the farms' own power closed about 78% of the shortfall, while
  widening the members or shifting them in time closed almost none. The factor works better when
  fitted to the wind that the farms' power implies than when fitted to ERA5.
advice:
  - >-
    Correct the ensemble's wind speed with a multiplicative factor per lead day (1.06 at day 0 to
    1.17 at day 9) before the physics core, because it closed 69.8% of the gap to the ERA5 oracle
    where a single factor of 1.13 over-corrected days 0 to 2 to 60.8% (e_lead, a_mean).
  - >-
    Fit a wind-speed correction to the wind that the farms' power implies rather than to ERA5: the
    contrast is 4.2 points of share below the median (3.8 to 4.5) at the same factor form (a_mean
    against a_mean_era5).
  - >-
    Do not spend effort widening the ensemble members or shifting them in time to fix the low
    median: the spread factor closed -0.9% and the shifts 0% to 5.5% of the gap (b_spread, c_shift).
  - >-
    Replace experiment 285's quantile mapping of member speeds with a plain factor: a factor fitted
    to ERA5 closed 55.4% of the gap where the mapping closed about 18% (a_mean_era5, and 285's B1 to
    B3).
  - >-
    Quote the share of rows below the median beside the score, because the factor fitted to ERA5 had
    a slightly better quantile score (11.402) than the factor fitted to the implied wind (11.439)
    while its median was 4 points less calibrated (a_mean_era5 against a_mean).
---

**Take-home: the ensemble's low wind-power forecast is a wind-speed level error, not a spread or
timing error.** The three metered wind farms' observed power was below the forecast's median on only
34.1% of rows (50% is calibrated). We ran each farm's physics core, the fitted model that turns wind
speed into power, backwards to get the wind speed that the farm's own power implies. The 51 ensemble
members (separate forecast scenarios) are on average 1.1 m/s slower than that wind, from lead day 0,
and the gap grows with lead. Multiplying every member's wind speed by one factor of 1.13, fitted on
the previous 14 months, raised the share below the median to 48.6%, which closes 78% of the gap to
the ERA5 oracle (the core fed ERA5, our estimate of past weather, which gives 52.8%). Widening the
members closed none of the gap, and shifting them by up to 3 hours closed at most 5.5%. The verdict
is `not supported`, because the hypothesis said spread and timing would carry at least half of the
gap.

We asked which property of the ensemble's wind is wrong. The reference is the implied wind, which
neither ERA5 nor the ensemble supplies, because it comes from the turbines' own output. The question
follows [experiment 261](../00261-wind-low-bias-decomposition/result.md), which put 89% of the low
median in the ensemble's weather, and [experiment 285](https://github.com/openclimatefix/nged-substation-forecast/pull/1602),
whose quantile mapping of member speeds to ERA5's closed about 18% of the gap. The physics core is
from [experiment 230](../00230-wind-physics-core-part-1/result.md), and the idle-row mask from
[experiment 210](../00210-wind-idle-noise-mva-artefact/result.md).

**The wind speed that the farms imply is a better target than ERA5.** A factor fitted to ERA5 closed
55.4% (47.2% to 66.4%) of the gap, and the same factor form fitted to the implied wind closed 77.6%.
The paired difference is 4.2 points of share below the median (3.8 to 4.5). The quantile score
tells a smaller story: the ERA5-fitted factor scored 11.402 and the implied-wind factor 11.439
(lower is better), so the implied wind calibrates the median better without improving the whole
distribution.

**The high-resolution check gives a mixed answer.** The first forecast hour of the Met Office UKV
(from AWS, the live feed) is closer to the implied wind than ERA5 is at 9 to 14 m/s (root-mean-square
error 1.41 and 1.55 m/s against ERA5's 1.58 and 1.74). ICON-D2's freshest run is no closer than ERA5
(1.61 and 1.73 m/s).

![Share of the gap that each correction closes](figure_1.svg)

![Mean speed minus implied wind by speed bin](figure_2.svg)

![Ensemble minus implied wind by lead day](figure_3.svg)

![Time shift of the ensemble mean](figure_4.svg)

![Distance to the implied wind of ERA5, ICON-D2, and UKV](figure_5.svg)

## Design and results

- **Implied wind (step 1).** The core (experiment 261's R1, fitted to ERA5 before 2024-04-01) is run
  backwards on half-hours outside experiment 210's mask and outside runs of 48 or more half-hours
  below 1% of capacity, where observed power lies between 5% and 90% of the curve's plateau. That
  covers hub-height speeds from about 3 to 11.4 m/s (W1), 12.4 m/s (W2), and 15.8 m/s (W3), and
  79,000 to 82,000 rows per farm. The share of rows that can be inverted is 75% at ERA5 speeds of 5
  to 7 m/s, 95% to 99% at 7 to 11 m/s, 91% at 11 to 14 m/s, 60% at 14 to 18 m/s, and 42% above 18
  m/s.
- **Stop rule.** The implied 100 m wind minus ERA5's is +0.04 m/s on average out of sample (84,785
  rows, 2024-04 to 2026-06; root-mean-square error 1.61 m/s, correlation 0.80) and +0.16 m/s in
  sample. By farm it is +0.17, +0.11, and -0.14 m/s (W1, W2, W3). By season it is -0.36 (winter),
  +0.29, +0.27, and -0.10 m/s (autumn). Per direction sector it ranges from -0.14 to +0.37 m/s. The
  1 m/s stop rule did not fire.
- **Selection effect.** Binned by ERA5 speed, the implied wind is +1.5 m/s above ERA5 at 3 to 5 m/s
  and -1.9 m/s below it at 14 to 18 m/s (figure 2, blue). Selecting rows by observed power pulls
  every product's bin means towards the middle, so only differences between products on the same
  rows are meaningful.
- **Ensemble against implied wind (step 2, scored year, 79,461 rows).** The ensemble mean is 1.11 m/s
  slower than the implied wind (ERA5: +0.04 m/s), with root-mean-square error 2.60 m/s against ERA5's
  1.56. By lead day the mean gap is -0.42, -0.37, -0.27, -0.69, -1.07, -0.99, -1.08, -1.28, -1.67, and
  -1.55 m/s (days 0 to 9). The implied wind is below the members' 10th percentile on 4.3% of rows and
  above the 90th percentile on 21.7% (10% each if calibrated). The members' mean spread (10th to 90th
  percentile range over 2.56) is 2.38 m/s, against a spread of 2.35 m/s of the implied wind about
  the ensemble mean. At day 0 the member spread is 0.94 m/s against 1.61 m/s.
- **Timing.** The root-mean-square error of the ensemble mean against the implied wind is lowest at
  0 to +0.5 h (2.599 and 2.600 m/s) and rises to 2.67 m/s at -3 h and 2.64 m/s at +3 h.
- **Corrections (step 3).** Each factor multiplies the 10 m and 100 m speeds alike and was fitted on
  the 442 hindcast runs of 2024-04 to 2025-06. One factor is 1.13 (ERA5-fitted: 1.09). The factor per
  lead day runs from 1.06 at day 0 to 1.17 at day 9. The factor per direction sector (ensemble-mean
  direction) is 1.09 to 1.21, largest in sectors 0 to 3.
- **Share of kept rows below the median, scored year (95% interval).** Raw 34.1% (31.5 to 37.1);
  one factor 48.6% (45.8 to 51.6); factor per lead day 47.1% (44.4 to 50.1); factor per direction
  sector 48.9% (46.1 to 51.9); spread widened 33.9% (31.3 to 37.0); lead day, direction and spread
  together 47.1% (44.5 to 50.3); ERA5-fitted one factor 44.4% (41.7 to 47.5); ERA5-fitted per lead
  day 43.3% (40.7 to 46.3); ERA5 oracle 52.8% (50.3 to 55.1). Shifts: 35.1% at -3 h, 34.7% at -2 h,
  34.3% at -1 h, 34.1% at +1 h, 34.2% at +2 h, 34.3% at +3 h.
- **Share of the 18.7-point gap closed.** One factor 77.6% (66.4 to 92.4); per lead day 69.8% (60.1
  to 82.9); per direction sector 79.0% (66.8 to 94.6); spread -0.9% (-1.4 to -0.2); shifts 5.5%
  (-3 h), 3.3%, 1.1%, 0.2%, 0.8%, and 1.3% (+3 h); ERA5-fitted 55.4% and 49.4%. Direction sector
  against one factor: +0.3 points (-0.6 to +1.4), so the direction effect is the mean effect.
- **The single factor over-corrects early leads.** At days 0 to 2 it gives 60.8% of rows below the
  median, against 46.9% for the factor per lead day (days 3 to 6: 49.0% and 46.2%; days 7 and later:
  44.2% and 47.7%).
- **Quantile score (nine levels, points of capacity, kept rows, no spread factor).** Raw 11.823; one
  factor 11.439; per lead day 11.361; per direction sector 11.462; spread widened 11.993; all
  together 11.484; ERA5-fitted 11.402 and 11.361; shifts 11.84 to 12.18. Widening the members and
  shifting them both raise the score.
- **By farm (share of kept rows below the median, raw, one factor, ERA5 oracle).** W1 32.9%, 47.8%,
  50.3%; W2 28.0%, 42.0%, 50.9%; W3 41.8%, 56.5%, 57.5%. W2 stays furthest from its oracle.
- **High-resolution check (step 4, 51,797 half-hours of the scored year, 46,580 kept).** ICON-D2
  (Open-Meteo previous-runs archive, freshest value of each hour, converted from km/h) and UKV (AWS
  lead 0, 100 m level) cover 2024-01 to 2026-09 and 2024-10 to 2026-10, so both cover the scored
  year. Share of kept rows below the core's median with each product as the weather: ERA5 52.9%,
  ICON-D2 41.7%, UKV lead 0 34.1%, UKV lead 2 36.2%. Quantile score: ERA5 6.19, ICON-D2 5.74, UKV
  lead 0 6.21. The shares are not comparable in level: the core's hub height and shear exponent were
  fitted to ERA5's 10 m and 100 m speeds, and another product's 10 m to 100 m ratio changes the
  exponent. The comparison with the implied wind is the one that does not depend on that fit.
- **Speed bins against the implied wind (root-mean-square error, m/s; ERA5, ICON-D2, UKV).** 7 to 9
  m/s: 1.38, 1.41, 1.30. 9 to 11 m/s: 1.58, 1.61, 1.41. 11 to 14 m/s: 1.74, 1.73, 1.55. 14 m/s and
  above: 2.62, 2.51, 2.11 (38,303 rows).
- **Schedule.** One core fit before 2024-04-01 and one set of corrections fitted before 2025-06-17,
  with no refits, so the contrasts say nothing about retraining cadence.

## Checks

- **Median minus mean forecast.** This experiment does not store a mean forecast, so the per-farm
  median-minus-mean check is not repeated. Experiment 261 gives -4.7, -4.5, and -3.8 points for the
  raw mixture at W1, W2, and W3, and the mean of the median forecast moves from 17.7 points (raw)
  to 24.2 (one factor), against 29.2 for the oracle and an observed mean of 28.9.
- **Mean-forecast error beside the quantile score.** Not computed; the verdict rests on the share
  below the median, and the quantile score points the same way for every correction that moves the
  median.
- **Tail calibration and curtailment cost.** Not run: no arm is a production candidate, and the
  scores here are medians only.
- **Control.** The recomputed raw mixture median equals experiment 259's stored R1 median to within
  0.0000 on every kept row, so the refit reproduces the earlier forecasts.
- **Triggers.** Trigger 2 (a result contradicting an earlier experiment's finding) fired against
  experiment 285. The control above shows that a plain factor fitted to ERA5 closes 55.4% where 285's
  quantile mapping closed about 18%, but this experiment has not tested why the mapping fell short.
  The manager has been told, and no audit has run. No other trigger fired: the one factor raises
  the share by 14.9, 14.0, and 14.7 points at W1, W2, and W3, so no farm carries the result, and no
  nested arm beat its container.

## Caveats

- **The implied wind exists only where observed power is between 5% and 90% of the plateau.** No row
  above about 12 to 16 m/s hub-height speed is invertible, so the finding says nothing about the
  highest speeds, which matter for curtailment.
- **The implied wind inherits the core fitted to ERA5.** The core's hub height absorbs ERA5's own
  bias, so the implied wind is expressed in the same effective units as ERA5. The result says the
  ensemble is slower than that effective wind, not that either matches the wind at the turbines.
- **The corrections are fitted on observed power and so are a calibration, not a physics fix.** The
  fit uses only the hindcast before 2025-06-17, so the scored year is out of sample, but a factor
  fitted to observed power would be expected to calibrate the median.
- **Direction sectors use the ensemble-mean direction,** and the 10 m and 100 m speeds are scaled
  together so the shear exponent is unchanged.
- **Three farms and one weather year.** Intervals resample the 12 months and not the farms.

## What the next experiment should answer

- **Does a per-lead-day speed factor survive in the production chain?** Apply it before the core and
  XGBoost on the same rows, score with the offset table of experiment 259 on top, and include the
  tail levels (0.01 and 0.99).
- **Why did experiment 285's quantile mapping fall short?** Re-run its mapping next to a plain factor
  on identical rows, and check whether mapping to the marginal ERA5 distribution flattens the members.
