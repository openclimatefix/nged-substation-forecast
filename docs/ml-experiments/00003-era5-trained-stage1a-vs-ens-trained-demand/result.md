---
id: "00003"
slug: era5-trained-stage1a-vs-ens-trained-demand
status: planned
size: complex
hypothesis: >-
  A weather-to-power XGBoost model trained on ERA5 (2019-09 onward) and applied to the 51 ENS
  members, with the member forecasts averaged, has a lower mean absolute error on electricity
  demand series than the same columns trained on ENS weather.
prediction: >-
  The mean absolute error of the ensemble-mean forecast falls by at least 0.15 points of capacity
  against the ENS-trained model at most lead bands. Without a calendar column the effect is
  small, because weather alone does not carry the hour of day or the day of week.
decision_rule: >-
  Material if the mean paired difference is at most -0.15 points, the 95% month-resampled interval
  is wholly below zero, and at least 9 of 12 validation months agree.
benchmark: >-
  ens_trained (the same columns trained on the ENS control member, 2024-04 to 2025-06), with two
  climatology rows as references.
row_set_id: fold_a_every5d_v1
metric: mean absolute error of the ensemble-mean forecast, points of capacity
verdict: pending
best_arm: pending
headline: pending
code: studies/stage1_pilot/
pull_request: pending
---

# Does an ERA5-trained weather-to-power model beat the same model trained on ENS, for demand?

**Status: planned. This section was written before the first fit.** The experiment repeats
experiment 00001 for the 22 series of the fold A frames that are not wind or solar farms. The
write-up below the plan is filled in after the run.

## Plan

**Question.** Does stage 1a (one XGBoost model per series, trained on ERA5, applied to each of
the 51 ENS members' weather and averaged) forecast demand better than the same model trained on
ENS weather, and does the answer change when the calendar and lagged power are added to both?

**Series.** The 22 series of fold A that are not wind or solar farms, labelled D1 to D22: 16
disaggregated primary-substation demand series, 4 bulk-supply and grid-supply point flows, and 2
others (a battery and a biofuel generator). The results report the three groups separately. Each
series takes ERA5 weather from the nearest cell of the 121-cell stage 1 box.

**Row set and folds.** Validation 2025-07-01 to 2026-06-30, on the 73 runs of the row set
`fold_a_every5d_v1`: every fifth day at 00:00 UTC from 2025-07-01. The fold A frames of the
quick-wins study hold Tuesday runs only (row set `fold_a_tuesdays_v1`), which puts every case at
lead 0 to 1 day on a Tuesday or Wednesday. The five-day step cycles the run's weekday, so every
lead band has about 14% of its cases on each weekday of the valid time. The ENS-trained arms train
on the control member of every daily run from 2024-04-01 to 2025-06-30, keeping valid times up to
2025-06-30. Both sets of rows come from `demand_build_ens.py`, which reads the ENS archive and uses
production's half-hourly resampling; for the 2025-07-01 run it reproduces the fold A frame's
weather and lag columns exactly. Stage 1a trains on valid times from 2019-09 to 14 days before the
first run. Seeds 0, 1, and 2. A first fit on the Tuesday runs was run before this row set was
chosen and has not been scored; it is kept as a labelled reference.

**Weather columns, the same in every arm** (ENS serves each; ERA5 gives the same quantity):
`temperature_2m`, `dew_point_temperature_2m`, `shortwave`, `longwave`, `pressure_surface`,
`pressure_msl`, `precipitation`, `wind_speed_10m`, `wind_speed_100m`, and the precipitation-type
code (an integer 0 to 8). ERA5 shortwave and `fdir` are clipped at zero before use. No 500 hPa
geopotential height is used, because it is not on disk for ERA5.

**Variants.** Each variant gives every arm the same extra columns.

- **V0 (main):** the weather columns only. No calendar column and no power lag.
- **V1:** V0 plus the local hour of day and the local day of week.
- **V2:** V1 plus the power lags at 24, 48, 168, and 336 hours, null where the lag is no longer
  than the lead. The ENS-trained arm reads the lags from the fold A frames. For the ERA5-trained
  arm each training row draws one lead uniformly from 0.5 to 351 hours and nulls the lags that
  lead rules out.

**Arms.**

- **ens_trained:** the columns trained on the ENS control member, applied to each of the 51
  members and averaged (the benchmark).
- **s1_memberavg:** the columns trained on ERA5, applied to each member and averaged.
- **s1_short:** as s1_memberavg, trained on ERA5 from 2024-04 only (isolates the training weather
  from the training length).
- **ens_trained_dedup (V1 and V2 only):** the ENS-trained arm on one row per valid time (the
  shortest lead). The ENS training frame repeats each valid time about 15 times, once per lead,
  which lets a model with calendar columns memorise the calendar position, as in 00001. The
  dedup arm is the guard.
- **s1_recency, s1_nocovid (V0 and V1; exploratory):** stage 1a with weights that halve every 2
  years of age, and stage 1a without 2020-03 to 2021-07.
- **climatology** (mean by local month and half-hour of the day) and **climatology_dow** (the
  same, separately for weekdays, Saturdays, and Sundays), 2019-09 to 2025-06.

**Planned contrasts** (mean absolute error, points of capacity; ensemble-mean forecast):

- **D1:** s1_memberavg minus ens_trained, V0.
- **D2:** s1_memberavg minus ens_trained_dedup, V1.
- **D3:** s1_memberavg minus ens_trained_dedup, V2.

Every other number is exploratory: the contrasts against `ens_trained` in V1 and V2, the
short-training and regime arms, the contrasts by series group, the top-decile contrasts, and the
climatology rows. The top decile is the cases where observed power exceeds the series' 90th
percentile of cleaned power in stage 1a's training window. Intervals resample whole validation
months and the three fitting seeds, as in 00001. Absolute scores are reported for every arm.

**Check before any fit.** The diurnal profile of ERA5's `ssrd` against the solar zenith is
compared with ERA5's hour-ending convention, and the result is reported in the write-up, in case
the label is shifted. The weather columns of ENS member 0 at short lead are compared with ERA5 at
the same valid time, as in 00001.

**To be added.** A CAMS arm: stage 1a with CAMS shortwave irradiance in place of ERA5's in
training only, as in 00001, at the 22 locations. It needs a CAMS download at those locations and
is scoped separately.

**Not tested here.** Where lagged power should enter a two-stage forecast, several weather
models, quantile forecasts, and a second fold.

## Input columns

Filled in with the results.

## Results

Filled in after the run.
