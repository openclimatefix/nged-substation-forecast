# Baseline forecasters

This package holds the naive baselines that a trained model has to beat. A baseline subclasses
`BaseForecaster` and rides the same cross-validation chain as `XGBoostForecaster`, so the
leaderboard scores it the same way. This package does not own the feature pipeline, which stays in
`ml_core`, nor the XGBoost model, which stays in `xgboost_forecaster`.

## What each baseline emits

**`ManualHeuristicForecaster` emits one ensemble member per power lag.** The manual heuristic is the
analogue-ensemble method that distribution network operators use today. It reads the power observed
at the same weekday and time of day on earlier weeks: 6 weekly analogues from the last 6 weeks, and
7 annual analogues from 49 to 55 weeks back. `conf/model/manual_heuristic.yaml` lists the 13 power
lags. The member index is the rank of the lag among the selected power lags, shortest lag first, so
members 0 to 5 are the weekly analogues and members 6 to 12 are the annual analogues. What the
member index means across models is described on `PowerForecast.ensemble_member`. `nwp_init_time` is
null on every row, because the forecaster consumes no weather.

## Why the forecaster has its own feature engineer

**`PowerLagsPerNwpRunFeatureEngineer` gives the baseline the same forecast runs as every other
model, without reading any weather.** A leaderboard compares models on the same
`(power_fcst_init_time, valid_time)` rows. Those rows come from the numerical weather prediction
(NWP) archive's list of runs. The engineer reads from the NWP frame only the run, cell, and
valid-time keys, and passes that key-only frame to `TabularFeatureEngineer`. No weather value and no
ensemble member is read. An NWP run missing from the archive therefore removes that run's rows for
the manual heuristic, as it does for every other model.

## Caveats

**The weekly analogues shed with lead time.** A power lag no longer than the forecast lead time
would not yet have been observed when the forecast was issued, so the feature pipeline nulls it. The
weekly group holds 6 members below 168 h of lead, 5 members at leads in [168 h, 336 h), and 4
members from 336 h. An operator issuing a forecast at time T has only the weeks before T, so the
operator loses the same analogues. A member keeps its index as others shed.

**A series short of history, or with flagged readings, has fewer members.** A series with less than
55 weeks of history has no annual analogue for the earliest forecasts. Every reader of power skips a
reading that the power cleaning flags, so the analogue that reading would have supplied is null and
that member is shed. A row with no member left is dropped.

**The lags are fixed numbers of hours in UTC, so a clock change moves an analogue by an hour.** A
lag of a multiple of 168 h is a whole number of weeks in UTC. Across a clock change, the analogue
sits an hour away from the target in local clock time, whereas the operator's method matches local
weekday and time of day. About one member value in ten is affected.

## Not served live

**The baselines are research baselines.** `PowerLagsPerNwpRunFeatureEngineer.engineer` raises
`NotImplementedError` in single-run mode, which is the mode the live service uses.
