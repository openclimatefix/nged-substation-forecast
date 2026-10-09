# Baseline forecasters

This package holds the naive baselines that a trained forecasting model has to beat. A baseline
subclasses `BaseForecaster` and rides the same cross-validation chain as `XGBoostForecaster`, so the
leaderboard scores it the same way.

## What each baseline emits

**`ManualHeuristicForecaster` emits one ensemble member per power lag.** The manual heuristic is the
analogue-ensemble method that was, until recently, the normal forecasting approach among
distribution network operators. It reads the power observed at the same weekday and time of day on
earlier weeks: 6 weekly analogues from the last 6 weeks, and 7 annual analogues from 49 to 55 weeks
back. `conf/model/manual_heuristic.yaml` lists the 13 power lags. The member index is the rank of
the lag among the selected power lags, shortest lag first, so members 0 to 5 are the weekly
analogues and members 6 to 12 are the annual analogues. What the member index means across
forecasters is described on `PowerForecast.ensemble_member`. `nwp_init_time` is null on every row,
because the forecaster consumes no weather.

**`ClimatologyForecaster` emits 51 quantiles of past power as 51 ensemble members.** A calendar cell
is a series, a local month, a local half-hour of day, and a local weekday-or-weekend flag, all
derived from `valid_time` in Europe/London time. Each training sample counts towards nine cells: its
own, and the cells one month and one half-hour either side, with the same weekend flag. December and
January are neighbours, and so are half-hours 47 and 0. `train` stores the quantiles of each cell's
pooled samples at the equiprobable levels `(k + 0.5)/51` for k = 0 to 50, using linear
interpolation. `predict` looks up the cell of each forecast row. Member 0 is the lowest quantile and
member 25 is the median. `nwp_init_time` is null on every row, because the forecaster consumes no
weather. Bank holidays are ordinary days.

**The quantiles are equiprobable, and there are 51 of them, for two reasons.** The metrics layer
treats the members as an equiprobable sample, so members at the tail-heavy delivery levels would be
read as a wider-tailed distribution than the climatology they represent. The 51 members match the 51
members of the ECMWF ensemble, so climatology and a 51-member XGBoost ensemble compare at equal
member count on the size-dependent metrics. Equal count does not equalise member type, because
quantile members and random draws have different calibrated exceedance rates.

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
lag. The weekly group holds 6 members below 168 h of lead, 5 members at leads in [168 h, 336 h), and
4 members from 336 h. An operator issuing a forecast at time T has only the weeks before T, so the
operator loses the same analogues.

**A series short of history, or with flagged readings, has fewer members.** A series with less than
55 weeks of history has no annual analogue for the earliest forecasts. Every pipeline stage that
reads observed power reads only the rows the power cleaning left unflagged, so the analogue a
flagged reading would have supplied is null, and that member is shed. A row with no member left is
dropped.

**The lags are fixed numbers of hours in UTC, so a clock change moves an analogue by an hour.** A
lag of a multiple of 168 h is a whole number of weeks in UTC. Across a clock change, the analogue
sits an hour away from the target in local clock time, whereas the operator's method matches local
weekday and time of day. About one member value in ten is affected.

**Climatology pools at least 66 samples per cell, with a median of 156, for a series with a full
training history.** Those samples come from 22 to 66 distinct days, because neighbouring half-hours
of one day are almost perfectly correlated (median lag-1 autocorrelation 0.97). Without pooling,
such a series holds 8 to 19 samples in a weekend cell and 20 to 45 in a weekday cell. Pooling over
nine cells multiplies the typical count by about eight. The tail members of a cell with few pooled
samples sit at the observed extremes of those samples, because linear interpolation never
extrapolates. A cell with one pooled sample gives 51 equal members. The training run logs the
minimum and median pooled samples per cell.

**The climatology is tuned in-sample on the leaderboard fold, and 15 months of history make it
close to last year's distribution per cell.** The cell keying, pooling width, member count, and
holiday handling were chosen by scoring on the same validation year the leaderboard fold uses, so
the climatology is a slightly stronger reference than an untuned one. Series with under 7 months of
history (two of the 31) are forecast only on the cells they have, and a CRPSS against climatology is
not meaningful for them.

**Climatology drops a forecast row whose cell has no training sample anywhere in its
neighbourhood.** `predict` logs one warning with the count and the series. A series whose history is
shorter than the training window loses the most: on the leaderboard fold, two such series lose about
a third to two fifths of their validation rows. Compare climatology with other forecasters on the
rows that all of them forecast.

**A set of quantiles reads lower under the fair CRPS the fewer members it has, so the fair CRPS
cannot compare member counts.** The fair CRPS corrects its spread term for members that are
independent draws, and a set of equally spaced quantiles is not independent draws. On the
leaderboard fold, 13 unpooled climatology members read about 4% to 7% below the plain CRPS of 101
members, and 51 members read about 2% below (1.6% to 2.3% across the 27 full-history series). The
51-member pooled climatology therefore reads no better on the fair CRPS than a 13-member unpooled
climatology would, though its tails are far better. The same effect flatters climatology by about
2% against the manual heuristic and XGBoost, which does not change any ranking, because the gaps are
9% and 29%. It also gives climatology a small structural edge in a CRPS comparison with the
weather ensemble ([Ferro (2014)](https://doi.org/10.1002/qj.2270)): read a near-tie at extended
range as "the weather ensemble adds little out here", not as climatology winning.

**Most of the remaining tail miscalibration is year-to-year variation and sampling error that no
choice of members fixes.** On the leaderboard fold, the 18 full-history substations exceed the top
member on 3.1% of rows, against 1.0% for calibrated quantiles. Part of the excess is year-to-year
variation, which no choice of members fixes: the median member's monthly bias ranges from −10% to
+14% of mean power. Part is sampling error in the tail quantiles, which are estimated from only 22
to 66 distinct days per cell: weekend cells exceed their top member on 3.4% of rows and weekday
cells on 2.3%.

**For a battery and a biofuel generator, 51 pooled members worsen the lower-tail pinball losses.**
Against 13 unpooled members, the pinball loss at the 1st percentile worsens by 18% and at the 5th
percentile by 14%, because the power of both series is bimodal. The improvement in the pinball loss
at the 99th percentile from 13 to 51 members is partly mechanical, since the derived 99th percentile
of 13 equiprobable members sits at level about 0.93 and of 51 members at about 0.98.

**In March and October, a photovoltaic cell mixes days an hour apart in solar time.** The cell keys
use local time, so a March or October cell mixes days on Greenwich Mean Time with days on British
Summer Time. At 13 unpooled members that mixing cost about 2.6% of the CRPS of photovoltaic series
in those two months.

**The 51-member climatology compares with XGBoost's 51 members at equal member count, and with the
13-member manual heuristic at unequal member count.** The prediction interval coverage, pinball
loss, interval width, and exceedance rate all depend partly on member count and member type, so a
difference from the manual heuristic on those metrics is partly a difference in member count and
member type.