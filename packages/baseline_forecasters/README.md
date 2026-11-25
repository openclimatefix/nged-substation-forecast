# Baseline forecasters

This package holds the naive baselines that a trained forecasting model has to beat. A baseline
subclasses `BaseForecaster` and rides the same cross-validation chain as `XGBoostForecaster`, so the
leaderboard scores it the same way.

## What each baseline emits

**`ManualHeuristicForecaster` emits the 13 quantiles of its power lags.** The manual heuristic is the
analogue-ensemble method that was, until recently, the normal forecasting approach among
distribution network operators. It reads the power observed at the same weekday and time of day on
earlier weeks: 6 weekly analogues from the last 6 weeks, and 7 annual analogues from 49 to 55 weeks
back. `conf/model/manual_heuristic.yaml` lists the 13 power lags. For each forecast row, `predict`
takes the analogue values that are not null and passes them to `ml_core.quantiles.to_stored_quantiles`,
which stores their linear empirical quantiles at the 13 `DELIVERY_QUANTILES`. `nwp_init_time` is null
on every row, because the forecaster consumes no weather.

**`ClimatologyForecaster` emits the 13 delivery quantiles of past power.** A calendar cell
is a series, a local month, a local half-hour of day, and a local weekday-or-weekend flag, all
derived from `valid_time` in Europe/London time. Each training sample counts towards nine cells: its
own, and the cells one month and one half-hour either side, with the same weekend flag. December and
January are neighbours, and so are half-hours 47 and 0. `train` stores the quantiles of each cell's
pooled samples at the 13 `DELIVERY_QUANTILES`, using linear interpolation, in the same
`power_fcst_p1` … `power_fcst_p99` columns that `PowerForecast` stores. `predict` looks up the cell
of each forecast row. `nwp_init_time` is null on every row, because the forecaster consumes no
weather. Bank holidays are ordinary days.

## Why the baselines have their own feature engineer

**`NwpRunRowsWithoutWeatherFeatureEngineer` gives each baseline the same forecast runs as every
other forecaster, without reading any weather.** A leaderboard compares forecasters on the same
`(power_fcst_init_time, valid_time)` rows. Those rows come from the numerical weather prediction
(NWP) archive's list of runs. The engineer reads from the NWP frame only the run, cell, and
valid-time keys, and passes that key-only frame to `TabularFeatureEngineer`. No weather value and no
ensemble member is read. An NWP run missing from the archive therefore removes that run's rows for
both baselines and for every other forecaster alike. Both baselines use the engineer. Climatology
requests no power lag, so its rows carry `power` and no lag column.

## Caveats

**The weekly analogues shed with lead time.** A power lag no longer than the forecast lead time
would not yet have been observed when the forecast was issued, so the feature pipeline nulls that
lag. The weekly group holds 6 analogues below 168 h of lead, 5 analogues at leads in [168 h, 336 h),
and 4 analogues from 336 h. An operator issuing a forecast at time T has only the weeks before T, so
the operator loses the same analogues.

**A series short of history, or with flagged readings, has fewer analogues.** A series with less than
55 weeks of history has no annual analogue for the earliest forecasts. Every pipeline stage that
reads observed power reads only the rows the power cleaning left unflagged, so the analogue a
flagged reading would have supplied is null, and that analogue is shed. A row with no analogue left
is dropped.

**The lags are fixed numbers of hours in UTC, so a clock change moves an analogue by an hour.** A
lag of a multiple of 168 h is a whole number of weeks in UTC. Across a clock change, the analogue
sits an hour away from the target in local clock time, whereas the operator's method matches local
weekday and time of day. About one analogue value in ten is affected.

**Climatology pools at least 66 samples per cell, with a median of 156, for a series with a full
training history.** The pooled samples come from at least 22 distinct days in a weekend cell (median
26) and at least 49 in a weekday cell (median 66), because neighbouring half-hours of one day are
strongly correlated (median lag-1 autocorrelation of raw power 0.97). Without pooling,
such a series holds 8 to 19 samples in a weekend cell and 20 to 45 in a weekday cell. Pooling over
nine cells multiplies the typical count by about eight. The tail quantiles of a cell with few pooled
samples sit at the observed extremes of those samples, because linear interpolation never
extrapolates. A cell with one pooled sample gives 13 equal quantiles. The training run logs the
minimum and median pooled samples per cell.

**The climatology is tuned in-sample on the leaderboard fold.** The cell keying, pooling width,
and holiday handling were chosen by scoring on the same validation year the
leaderboard fold uses, so the climatology is a slightly stronger reference than an untuned one.
Series with under 7 months of history (two of the 31) are forecast only on the cells they have, and
a CRPSS against climatology is not meaningful for them.

**Climatology drops a forecast row whose cell has no training sample anywhere in its
neighbourhood.** `predict` logs one warning with the count and the series. A series whose history is
shorter than the training window loses the most: on the leaderboard fold, two such series lose about
a third to two fifths of their validation rows. Compare climatology with other forecasters on the
rows that all of them forecast.

**Most of the remaining tail miscalibration is year-to-year variation and sampling error that no
choice of quantile levels fixes.** On the leaderboard fold, the 18 full-history substations exceed
the top quantile on 3.1% of rows, against 1.0% for calibrated quantiles. Part of the excess is
year-to-year variation, which no choice of quantile levels fixes: the median's monthly bias ranges
from −10% to
+14% of mean power. Part is sampling error in the tail quantiles, which are estimated from few
distinct days per cell: across the 27 full-history substation, PV, and wind series, weekend cells
exceed their top quantile on 3.4% of rows and weekday cells on 2.3%.

**For a battery and a biofuel generator, pooling worsens the lower-tail pinball losses.** Against
unpooled cells, the pinball loss at the 1st percentile worsens by 18% and at the 5th percentile by
14%, because the power of both series is bimodal.

**In March and October, a photovoltaic cell mixes days an hour apart in solar time.** The cell keys
use local time, so a March or October cell mixes days on Greenwich Mean Time with days on British
Summer Time. Without pooling, that mixing raised the CRPS of photovoltaic series by
about 2.6% in those two months.
