---
id: "00006"
slug: stage1a-demand-input-feature-sweep
status: planned
size: medium
hypothesis: >-
  For a weather-to-power XGBoost model trained on ERA5 for substation demand, the local clock
  (hour of day and day of week) is the input that lowers the held-out error most, and the other
  non-lag inputs tested here (holiday encodings, time of year, trailing temperature means, sun
  position beyond zenith and azimuth, UTC offset, windchill, and a COVID indicator) each lower it
  by less.
prediction: >-
  Against the weather block alone, the clock lowers the mean absolute error of the median by at
  least 1.5 points of capacity (experiment 00003 found about 2). Against the weather block plus the
  clock, each of the other non-lag inputs lowers it by less than 0.5 points, and windchill by less
  than 0.05 points.
decision_rule: >-
  A contrast is material if its 95% month-resampled interval (Bonferroni-widened across the nine
  planned contrasts) excludes zero and the point estimate is at least 0.1 points of capacity in
  the direction stated. The hypothesis is supported if P1 is material and its size exceeds that of
  every other input's add-one-in contrast against the weather block.
benchmark: >-
  The weather block W alone, and a month-by-half-hour climatology row.
row_set_id: era5_val_2024_07_to_2025_06_v1
metric: mean absolute error of the median, points of capacity (power over effective capacity, times 100)
verdict: pending
code: ml-experiments/00006-stage1a-demand-input-feature-sweep/
pull_request: pending
---

# Which inputs work best for the ERA5-trained demand model (stage 1a)?

**Status: planned.** This page holds the plan until the first scored fit, and the results are added
below it.

## Plan

**Question.** This experiment does not forecast. It trains on ERA5 weather and scores on held-out
ERA5 months, predicting the observed power at the same valid time. The question is which non-lag
inputs make the best weather-to-power model. Every input must also be computable at forecast time,
for the next 15 days, from the calendar and from forecast weather. No observed-power lag is used.

**Series.** The 20 series of `TimeSeriesMetadata` whose `substation_type` is in
`SUBSTATION_TYPES`: 16 Primary substations (P1 to P16), 2 bulk supply points (B1, B2), and 2 grid
supply points (G1, G2). Generators, batteries, and the HV and EHV customer series are excluded.
Power is the cleaned power (`scan_cleaned_power`, unflagged rows), divided by the series' effective
capacity from the fold A frames. Labels come from a seeded permutation inside the experiment; no
identifier, name, or coordinate is published.

**Weather.** ERA5 from the 121-cell stage 1 box, at the cell nearest each series, resampled to the
half-hourly period-ending grid with the same convention as experiment 00003 (`ssrd` and `fdir`
clipped at zero; accumulations are totals over the hour ending at the stamp). The script copies
that convention because an experiment may not import another study's folder, and checks its
output against 00003's stored ERA5 features for the same series. One arm replaces the nearest cell
by the mean of the 3 by 3 block of cells around it.

**Weather block W (9 columns):** `shortwave`, `longwave`, `temperature_2m`,
`dew_point_temperature_2m`, `pressure_surface`, `precipitation`, `wind_speed_10m`, `solar_zenith`,
and `solar_azimuth`. Every column is a function of forecast weather and the clock, so it is
available at forecast time.

**Folds.** The experiment feeds the lag experiment 00005, which validates on 2025-07 onwards, so
no choice may be influenced by that year. **Selection fold:** validate on 2024-07-01 to
2025-06-30, train on ERA5 from 2019-09-01 to one week before validation (and, for the recent-start
variants, from 2023-09). A second selection fold validates on 2023-07-01 to 2024-06-30 if time
allows. Every add-one-in, greedy, and drop-one-out decision uses the selection folds only. The
chosen feature set is then frozen and written into this page before anything is scored on
2025-07-01 to 2026-06-30. **Confirmation fold:** the frozen set, the weather block W, and W plus the
clock are scored once on 2025-07-01 to 2026-06-30, training up to one week before it, and reported
as confirmation and not as selection. No valid time on or after 2026-07-01 is scored.

**Models.** Two configurations, each run for every arm.

1. One XGBoost model per series.
2. One XGBoost model per substation type (Primary, BSP, GSP) with the series identifier as an
   input, both as a categorical column and as an integer code. The two encodings are run for W and
   for the best set only.

XGBoost settings are identical in every arm: `eta` 0.05, `max_depth` 6, 500 rounds, `subsample` 0.8,
`colsample_bytree` 1.0, `colsample_bynode` 1.0 (column subsampling would give a wider set a free
win). The first pass uses the absolute-error objective, whose fitted value is the median. The 9
quantile levels (0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95) are run for the finalists, and
the pinball loss is reported then. The first pass uses one seed; the finalists use three.

**Arms** (each adds the listed columns to W, one at a time, then in groups):

- **Clock:** `hour_local`, `day_of_week_local`.
- **Time of year:** sine and cosine of the local day of the year, with one harmonic and with two.
  The encoding is periodic, so a new year reuses the same values.
- **Holiday, five arms:** the holiday name as a categorical (England and Wales, from the `holidays`
  package); the holiday flag with day-before and day-after flags; the signed days to the nearest
  holiday, clipped at 7; the signed days to 25 December, clipped at 14; a bridge-day flag (a working
  day between a holiday and a weekend). Holidays are known years ahead.
- **Trailing temperature, four arms:** the mean temperature over the previous 24 hours; 72 hours; 7
  days; and exponentially weighted means with half-lives of 12 hours, 72 hours, and 7 days. The
  window ends at the valid time. At forecast time the window straddles analysis and forecast
  weather and would be built from one member's trajectory.
- **Heating degree-hours:** the mean over 24 and 72 hours of the excess of 15.5 °C over the
  temperature.
- **Sun position:** day length and hours since sunrise (local); and the sine and cosine of the
  azimuth. Sun elevation is 90 degrees minus the zenith, a monotone function a tree treats as the
  same column, so it is not tested.
- **UTC offset:** 0 or 1 hour.
- **Windchill:** the Environment Canada formula, equal to the temperature above 10 °C or below 4.8
  km/h, and the product of the wind speed and the shortfall of temperature below 15.5 °C. Both are
  implemented inside the experiment script only.
- **COVID:** a 0/1 indicator for 2020-03-23 to 2021-07-19, reading 0 in every validation month; an
  arm that excludes those training rows; and an arm with recency sample weights (half-life 2 years).
- **Spatial mean:** the 3 by 3 block mean replaces the nearest cell for every weather column.

**Extra inputs from an Opus consultation.** Taken: the exponentially weighted temperatures, the
heating degree-hours, the spatial block mean, the bridge-day flag, and the replacement for
windchill. Not taken, and listed in the write-up as untested: an embedded-solar proxy (needs an
outside capacity series), school terms (dates must be collected by hand), cooling degree-hours,
temperature change over 24 hours, snow, humidity, days since the clock change, sporting events, and
the substation-type fields.

**Planned contrasts** (mean absolute error of the median, points of capacity, one model per series;
the interval resamples whole validation months). P1 to P7 are scored on the selection fold. P8 and
P9 are scored on the confirmation fold, and P8 is the one contrast the frozen set must pass.

- **P1:** W plus clock minus W.
- **P2:** W plus clock plus one harmonic minus W plus clock.
- **P3:** W plus clock plus the holiday flags minus W plus clock.
- **P4:** W plus clock plus the 24-hour trailing mean minus W plus clock.
- **P5:** W plus windchill (the formula) minus W.
- **P6:** W plus clock plus the COVID indicator minus W plus clock.
- **P7:** W plus clock with recency weights minus W plus clock.
- **P8:** the frozen set selected on the selection fold minus W plus clock.
- **P9:** the selected set trained from 2023-09 minus the same set trained from 2019-09.

Every other number is exploratory and labelled so: the add-one-in arms against W and against W plus
clock, the drop-one-out, the pooled configurations, the second selection fold, and the breakdowns by type,
hour, and season. A month-by-half-hour climatology (`ClimatologyForecaster`) is the reference row.
Absolute scores are reported for every arm.

**Not tested here.** Lagged power, forecasting from ENS weather, and downloaded COVID series
(Oxford stringency, Google mobility).
