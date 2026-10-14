---
id: "00006"
slug: stage1a-demand-input-feature-sweep
status: done
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
verdict: supported
best_arm: >-
  W + hour + weekday + COVID indicator + days to Christmas, trained from 2023-09 (Primary
  substations, confirmation year: 6.71 points)
headline: >-
  P1, hour and weekday against the weather block alone, Primary substations: -0.55 (-0.71 to -0.41)
  on the selection year and -0.46 (95%: -0.57 to -0.35) on the confirmation year; the largest input
  contrast, but a third of the predicted size. The training-history choices (COVID indicator,
  recency weights, a 2023 start) are as large or larger.
lessons: >-
  The weather block already carries the season, so time-of-year columns add nothing; windchill is
  useless; the training history matters more than any added column.
code: ml-experiments/00006-stage1a-demand-input-feature-sweep/
pull_request: pending
---

# Which inputs work best for the ERA5-trained demand model (stage 1a)?

**Verdict: supported on the decision rule, with the predicted size wrong.** The clock (hour of day
and day of week) is the largest input contrast for the 13 Primary substations, and no other input
beats it. It lowers the error by about 0.5 points of capacity, a third of the 1.5 points the
hypothesis predicted. Every other non-lag input lowers the error by less than the clock does, as
predicted, and windchill changes the error by less than 0.05 points, as predicted.

**Primary substations are the headline, because NGED cares most about them.** Every score below
is split by substation type, and no pooled mean over all types is quoted as a result, because the
2 grid supply points (GSP) have errors about four times those of the Primary substations and would
dominate it. A point is 1% of the series' effective capacity, and every score is the mean absolute
error of the median.

![Error on the confirmation year by arm and substation type](figure_1.svg)

## What to use for each substation type

**The Primary set is the weather block, the clock, a COVID indicator, and the days to Christmas.**
Greedy forward selection on the selection year (2024-07 to 2025-06) was repeated for each type
and stopped when the next addition gained less than 0.1 points. The sets differ by type, with
only 2 series each for BSP (bulk supply points) and GSP:

| Substation type | Selection-year set (all start from the weather block W and the clock) | Series |
| --- | --- | --- |
| Primary | COVID indicator, then days to 25 December (-0.25, then -0.14 points) | 13 |
| BSP | holiday flags: the day, the day before, and the day after (-0.16 points) | 2 |
| GSP | COVID indicator, days to the nearest holiday, heating degree-hours (-1.67, -0.24, -0.14 points) | 2 |

**The Primary set was frozen before anything was scored on 2025-07 to 2026-06.** The first
all-types selection chose the days to the nearest holiday where the Primary-only selection chose
the days to Christmas, and the Primary-only set replaced it before the confirmation year was
scored. Its columns are `hour_local`, `day_of_week_local`, `covid_indicator` (1 from 2020-03-23 to
2021-07-19, and 0 in all later data, so it reads 0 at forecast time), and `days_to_christmas` (days
to 25 December, negative before, clipped to 14 days either way). Only `days_to_christmas` is outside
the set of derived columns the stage 1 pilot already had. Each column is computable at forecast
time from the calendar alone.

**Dropping one group at a time from the Primary set shows the clock and the COVID indicator matter
most.** On the selection year, removing the clock raised the Primary error by 0.59 points (95%
interval +0.49 to +0.69), removing the COVID indicator by 0.30 (+0.20 to +0.40), and removing the
days to Christmas by 0.14 (-0.06 to +0.35), so the Christmas column is the least certain member of
the set.

## Absolute scores on the confirmation year (2025-07 to 2026-06)

Mean absolute error of the median in points of capacity, with 95% intervals that resample whole
months, averaged over three fitting seeds, one XGBoost model per series. The confirmation year
scores 14 Primary, 2 BSP, and 2 GSP series; the 2 series with under a year of history before the
fold are left out of it.

| Arm | Primary (14) | BSP (2) | GSP (2) |
| --- | --- | --- | --- |
| Climatology (month, half-hour, weekend) | 8.93 (8.61 to 9.28) | 13.11 (12.02 to 14.27) | 36.62 (33.86 to 39.24) |
| Weather block W | 8.31 (8.00 to 8.61) | 9.90 (9.01 to 10.85) | 30.05 (27.83 to 32.22) |
| W plus clock | 7.85 (7.48 to 8.23) | 9.60 (8.71 to 10.58) | 29.97 (27.83 to 32.02) |
| Primary set (W, clock, COVID, Christmas) | 7.57 (7.29 to 7.85) | 9.41 (8.57 to 10.26) | 26.47 (25.07 to 27.84) |
| Primary set, recent years weighted more | 7.11 (6.84 to 7.39) | 9.04 (8.36 to 9.74) | 23.98 (22.56 to 25.16) |
| Primary set, trained from 2023-09 | 6.71 (6.37 to 7.10) | 8.99 (8.44 to 9.52) | 21.73 (19.35 to 23.87) |
| Primary set, weighted, from 2023-09 | 6.68 (6.36 to 7.05) | 8.98 (8.44 to 9.50) | 21.51 (19.12 to 23.68) |
| BSP set (W, clock, holiday flags) | | 9.53 (8.57 to 10.59) | |
| GSP set (W, clock, COVID, nearest holiday, heating) | | | 27.71 (25.75 to 29.56) |

**Every arm beats the climatology for every type.** The Primary set is 1.36 points (95% interval
-1.67 to -1.08) better than the climatology for Primary substations.

## Planned contrasts

![Paired contrasts on the confirmation year](figure_2.svg)

The contrasts P1 to P9 were fixed in the plan below before any fit. The intervals for the nine
planned contrasts are Bonferroni-widened to 99.4%. All numbers are changes in the mean absolute
error, so negative is better, for Primary substations with 3 fitting seeds.

| Contrast | Year | Change | Interval | Material? |
| --- | --- | --- | --- | --- |
| P1: clock against W | selection | -0.55 | -0.71 to -0.41 | yes |
| P1: clock against W | confirmation | -0.46 | -0.57 to -0.35 (95%) | yes |
| P2: one harmonic of time of year, against W plus clock | selection | +0.04 | -0.40 to +0.37 | no |
| P3: holiday flags, against W plus clock | selection | -0.11 | -0.21 to -0.02 | borderline: the interval excludes zero, the size is at the 0.1 floor |
| P4: previous-24-hour mean temperature | selection | +0.04 | -0.01 to +0.08 | no (slightly worse) |
| P5: guarded windchill, against W | selection | +0.003 | -0.011 to +0.016 | no effect |
| P6: COVID indicator | selection | -0.26 | -0.40 to -0.10 | yes |
| P7: recency weights | selection | -0.32 | -0.49 to -0.13 | yes |
| P8: Primary set against W plus clock | confirmation | -0.29 | -0.58 to +0.02 (99.4%); -0.50 to -0.07 (95%) | not at the planned level |
| P9: Primary set trained from 2023-09, against the full history | confirmation | -0.86 | -1.19 to -0.48 | yes |

**P8, the step from the clock to the Primary set, is supported at 95% but not at the planned
99.4% level.** The COVID and Christmas columns together lower the confirmation-year error by 0.29
points, and the interval touches zero at the planned level. The two columns passed on the selection
year (-0.26 for the COVID indicator alone at 99.4%).

**P2, P4, and P5 predicted correctly that these columns do nothing.** P5 stays within 0.01 points of
zero, and P2 and P4 move the error up by 0.04 points. The two replication contrasts on a second
selection year (2023-07 to 2024-06, 13 Primary series) give the same signs: P1 -0.50 (-0.66 to
-0.34), P5 -0.005 (-0.021 to +0.013), P6 -0.11 (-0.27 to +0.06), and P7 -0.42 (-0.61 to -0.24).

The BSP and GSP selection-year contrasts follow the same signs, with wider intervals from 2 series
each: P1 is -0.37 (-0.59 to -0.16) for BSP and -0.35 (-0.68 to +0.03) for GSP, and P6 is -1.74
(-2.76 to -0.69) for GSP.

## Which columns help and which do not

![Change in error from adding one set of columns, Primary substations, selection year](figure_3.svg)

**The training-history choices are as large as the best added column.** For Primary substations on
the selection year, weighting recent years more (half-life 2 years) lowers the error by 0.32 points,
the COVID indicator by 0.25, and dropping the lockdown months from training by 0.21. The graded
lockdown indicator (1 for the national lockdowns, 0.5 for the restrictions between them, from
approximate dates) scores 7.130 against 7.142 for the 0/1 indicator, so grading adds nothing. On the
confirmation year, training from 2023-09 lowers the Primary error by 0.86 points more than the full
history, and weighting recent years lowers it by 0.46 more than the unweighted chosen set. Recency
weights on top of a 2023-09 start lower it by a further 0.43 points. The GSP error drops by 4.74
points (95% interval -7.93 to -2.28) with the 2023-09 start, so drift dominates the GSP series.

**Some holiday columns help a little.** For Primary substations, holiday flags lower the error by
0.10 points (-0.17 to -0.04), the days to Christmas by 0.10 (-0.33 to +0.12), the holiday name by
0.07 (-0.13 to -0.02), and the days to the nearest holiday by 0.07 (-0.24 to +0.09). These
intervals are from one fitting seed and 12 months. The bridge-day flag changes the error by
-0.004 points.

**No weather-derived column helps Primary substations.** The 24-hour, 72-hour, and 7-day trailing
mean temperatures add +0.04, +0.03, and +0.07 points, the weighted temperatures (12 hours, 72 hours,
7 days) +0.21 (0.00 to +0.43), heating degree-hours +0.03, the day length and hours since sunrise
+0.01, the UTC offset +0.02, and wind-driven heat loss +0.01. The mean of the 3 by 3 block of
cells around the nearest cell, in place of the nearest cell, changes the error by -0.01 points.
Sun azimuth as sine and cosine lowers it by 0.04 points (-0.08 to -0.02), too small to matter.

**Time of year does not help, because the weather block already carries the season.** One
harmonic of the day of the year adds +0.04 points (-0.28 to +0.29), two harmonics +0.10, and the
month as a categorical +0.09 against the weather block plus clock.

![Time of year added to weather blocks with columns removed](figure_4.svg)

The masking ablation removes the columns that could carry the season. Time of year lowers the
Primary error only when the weather block loses sun position and temperature together: -0.56
points (-0.94 to -0.20) with one harmonic, -0.49 with two, and -0.54 (-0.81 to -0.27) with the
month as a categorical. Removing only the sun position gives -0.28 (-0.67 to +0.04), and removing
only the temperature and dew point gives +0.09 (-0.16 to +0.32). A coarser encoding (one harmonic,
or month) does no better than a finer one, which does not support the suspicion that a fine
day-of-year encoding lets the trees memorise individual weeks of the training years. In the full
weather block, the permutation importance of the sun's position and the temperature is large
(below), which fits the explanation that these columns carry the season.

![Error by month of the year, with and without time of year](figure_6.svg)

**The error is highest in October to December whether or not time of year is a column.** The Primary
error on the selection year ranges from about 6.1 in August to about 9.2 in December for the weather
block plus clock, and the time-of-year columns change single months by up to 1.3 points (December,
9.17 to 7.89) and worsen others by up to 0.5 (October), without removing the seasonal shape.

## Windchill is useless, and its production code can go

**Neither windchill formula changes the error.** The guarded Environment Canada formula changes the
Primary error by +0.003 points (-0.011 to +0.016, 99.4%) against W. Production's expression, which
has no guard on its valid range, changes it by -0.002 (95% -0.016 to +0.010) against W and by -0.007
(-0.022 to +0.005) against W plus clock. The wind-driven heat loss (wind speed times the shortfall
of temperature below 15.5 °C) changes it by +0.007 points. [Issue #1199 (Remove the windchill
feature from the production feature
code)](https://github.com/openclimatefix/nged-substation-forecast/issues/1199) asks for the
production windchill code to be removed.

## Pooled models against one model per series

**One model per series beats one model per substation type on Primary substations.** A pooled model
needs the series identifier as an input. Without it, the pooled Primary model scores 11.79 on the
confirmation year against 7.57 for the per-series model, both with the Primary set, because the
series have different levels of power over capacity. With a categorical identifier and the same
columns plus recency weights, the pooled model scores 7.34 against 7.11 for the per-series model. A
categorical identifier scores 0.2 to 0.3 points lower than an integer code for Primary (7.34 against
7.57, with recency weights), and the two are identical for BSP and GSP, which have 2 series each.
The selection-year comparison agrees (7.11 pooled against 6.92 per series). The best feature set
does not change with the configuration.

## What the model uses

![Importance of each column, Primary substations](figure_5.svg)

**Hour of day, temperature, and the sunlight columns carry most of the Primary model.** Across the
14 per-series models of the Primary set with recency weights, shuffling `hour_local` raises the
validation error by a median of 6.2 points, `temperature_2m` by 3.3, `solar_azimuth` by 2.8,
`shortwave` by 2.4, `solar_zenith` by 1.7, `day_of_week_local` by 0.6, and `days_to_christmas` by
0.2. Gain, which favours columns with many distinct values, puts `hour_local` (median share 0.27)
and `shortwave` (0.21) first. `covid_indicator` scores 0.0 on permutation because it reads 0 in
every validation month; its value is in quarantining the lockdown months during training. For the
pooled Primary model (categorical identifier, recency weights), the series identifier has the
largest permutation importance (7.9), followed by `hour_local` (5.1), `temperature_2m` (3.0),
`shortwave` (2.9), `solar_azimuth` (2.3), and `solar_zenith` (1.6).

## Probabilistic scores

**The 9 quantile levels (0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95) give the same ranking as
the median.** On the confirmation year the mean pinball loss over the 9 levels, in points of
capacity, falls from 2.80 (W) to 2.68 (W plus clock) to 2.42 (Primary set with recency weights) for
Primary, from 3.34 to 3.27 to 3.11 for BSP, and from 9.50 to 9.48 to 7.76 for GSP. The interval from
the 0.05 to the 0.95 level covers 85%, 84%, and 82% of the Primary observations for the three arms,
and 79%, 78%, and 81% for GSP, against a nominal 90%, so the quantiles are too narrow. Calibration
is the job of stage 2, not of this experiment.

## Failures and limits

- **The predicted size of the clock effect was wrong.** The experiment predicted at least 1.5 points
  from experiment 00003 and measured about 0.5. Experiment 00003 compared ENS-trained and
  ERA5-trained models over 22 series including battery and grid-flow series, so the two are not
  directly comparable.
- **P8 fails at the planned level.** The COVID and Christmas columns together are supported at 95%
  only on the confirmation year.
- **BSP and GSP rest on 2 series each**, so their selected sets, and the BSP and GSP intervals, are
  noisy. The GSP error is dominated by drift, so a GSP set tuned on one year may not carry over.
- **The sweep used one fitting seed and 12 validation months**; the add-one-in intervals do not
  include seed variation, and 12 months is a short span for the holiday columns, which see 1 to 2
  occurrences of each holiday per year.
- **Series with under a year of history before a fold's purge are left out of that fold**, which
  removes 3 of the 20 series from the selection year and 2 from the confirmation year.
- **No hyperparameter was tuned per arm.** Every XGBoost model has 500 rounds, a learning rate of
  0.05, depth 6, subsample 0.8, and no column subsampling.
- **The first greedy run was scored on a partial results folder** (the add-one-in run had not
  finished) and was discarded and rerun on the complete folder.
- **The ERA5 weather and the power timestamps share one convention copied from experiment 00003**,
  and the copy reproduces 00003's ERA5 features to 0.0 on every weather column.

## Untested ideas

From an Opus consultation, not tested because each needs outside data or hand-collected dates:
an embedded-solar proxy built from the clear-sky irradiance times the embedded PV capacity of each
grid supply point; school-term flags by nation; cooling degree-hours; the temperature change over
24 hours; a snow flag; relative humidity or apparent temperature; days since the last clock change;
sporting and broadcast events; and the substation-type and customer-mix fields. A follow-up could
also test a measured lockdown size (the Oxford stringency index or Google mobility) against the
recency weights, since recency weights beat the 0/1 indicator on Primary here.

## Lessons for later experiments

- **Choose the training window before adding columns.** A 2023 start, recency weights, and a COVID
  indicator each lower the Primary error as much as the clock does.
- **Use per-substation sets, and report each type separately.** The best set differs between
  Primary, BSP, and GSP, and the GSP error would swamp a pooled mean.
- **Add time-of-year columns only to a weather block that cannot already carry the season.**
- **Remove windchill from the production feature set.**
- **Report a Bonferroni-widened interval for each pre-registered contrast.** The P8 contrast reads
  as supported at 95% and not at the planned level.

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
clock, the drop-one-out, the pooled configurations, the second selection fold, and the breakdowns by
type, hour, and season. A month-by-half-hour climatology (`ClimatologyForecaster`) is the reference
row. Absolute scores are reported for every arm.

**Not tested here.** Lagged power, forecasting from ENS weather, and downloaded COVID series
(Oxford stringency, Google mobility).

## Selection on 2024-07 to 2025-06, and the frozen set

**Written before anything was scored on 2025-07-01 to 2026-06-30.** The selection fold scores 17
series (13 Primary, 2 BSP, 2 GSP): series without a year of observed training power before the
fold are left out of it (3 series). Mean absolute error of the median, points of capacity, one
model per series, one seed for the sweep, three for the planned contrasts.

- **Greedy forward selection from W plus the clock** added the COVID indicator (-0.39 points) and
  then the signed days to the nearest holiday (-0.13 points), and stopped when the next best
  addition (azimuth sine and cosine) gained 0.04 points, under the 0.1-point minimum.
- **Frozen feature set:** W plus `hour_local`, `day_of_week_local`, `covid_indicator`, and
  `days_to_nearest_holiday`. Training from the whole history with recency weights (half-life 2
  years) scored 0.3 points lower than unweighted training on the selection fold, so both the
  weighted and the unweighted fit are scored on the confirmation fold.
- **Planned contrasts P1 to P7 on the selection fold** (Bonferroni-widened 99.4% intervals): P1
  -0.51 (-0.66 to -0.37), P2 +0.14 (-0.56 to +0.81), P3 -0.10 (-0.21 to -0.01), P4 +0.05 (-0.03 to
  +0.12), P5 -0.003 (-0.024 to +0.013), P6 -0.40 (-0.60 to -0.19), P7 -0.46 (-0.63 to -0.24).

### Revision before any confirmation-fold scoring: select on the Primary series

**The planning session asked for the list to serve the Primary substations.** Greedy selection was
repeated on the 13 Primary series alone, on the same selection fold. It adds the COVID indicator
(-0.25 points) and then the signed days to 25 December (-0.14 points), and stops when the next
addition (the holiday flags) gains 0.05 points. The all-series selection chose the COVID indicator
and the signed days to the nearest holiday. **The frozen set is now W plus `hour_local`,
`day_of_week_local`, `covid_indicator`, and `days_to_christmas`.** Nothing has been scored on
2025-07-01 or later. The earlier frozen set is kept as an exploratory arm.
