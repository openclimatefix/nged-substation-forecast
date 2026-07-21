# NGED substation baseline forecasts — export for switching-event detection

These three parquet files are an **early weather/calendar-only XGBoost baseline** for the 32-series
trial area (28 series had enough data to be forecast), produced for switching-event detection work.
The model has **no power-lag features**, so its residual (observed − expected) is meant to expose
switching events rather than absorb them. See the roadmap's
[shared baseline](https://openclimatefix.github.io/nged-substation-forecast/roadmap/switching-events/#the-baseline-shared-foundation).

> **⚠️ Please read the "How good are these?" section at the bottom before using these for serious
> analysis.** These predictions are not yet skilful, and it is probably worth waiting for the
> improved XGBoost (expected ~mid-August 2026) before relying on them.

## The files

All three cover the **same 28 time series** over the **same period**, and all power values are in
**physical units — MW (active power) or MVA (apparent power) per series** (you already hold the
`time_series_id → unit` mapping). Positive = power exported to NGED's grid; negative = drawn from
the grid.

For each `(time_series_id, valid_time)` we kept **only the freshest forecast run** — the most recent
run whose forecast covers that timestamp (median lead ≈ 12 hours) — so there is exactly one expected
value per series per half-hour.

| File | One row per | Columns |
|---|---|---|
| `*_full_ensemble.parquet` | series × timestep × ensemble member (51) | `time_series_id`, `valid_time`, `power_fcst_init_time`, `nwp_init_time`, `ensemble_member`, `power_fcst`, `observed_power` |
| `*_ensemble_mean.parquet` | series × timestep | `time_series_id`, `valid_time`, `power_fcst_init_time`, `power_fcst_mean`, `observed_power` |
| `*_quantiles.parquet` | series × timestep | `time_series_id`, `valid_time`, `power_fcst_init_time`, `power_fcst_p10`, `power_fcst_p50`, `power_fcst_p90`, `observed_power` |

Column notes:

- `valid_time` — the half-hour being forecast (UTC).
- `power_fcst_init_time` — when that forecast run was issued (= NWP run time + 6 h publication delay).
- `nwp_init_time` — the ECMWF run the forecast used (full-ensemble file only).
- `ensemble_member` — ECMWF ENS member index, 0–50 (0 is the control run).
- `power_fcst` / `power_fcst_mean` / `power_fcst_p10|p50|p90` — the forecast (ensemble member value,
  ensemble mean, or across-member quantiles).
- `observed_power` — the metered value at that `valid_time`, same units, for computing the residual
  directly. Present for ~96.5% of rows (nulls where telemetry is missing).

The obvious residual to work from is `observed_power − power_fcst_mean` (or `− power_fcst_p50`).

## Dates

- **Forecasts cover:** half-hourly from **2025-07-01 00:00 UTC to 2026-06-30 21:00 UTC** (the
  cross-validation *validation* year).
- **The model was trained on:** **2024-04-01 to 2025-06-30** (15 months), and never saw the forecast
  period above.

## What the model sees (XGBoost inputs)

Weather and calendar only — **no power-lag features by design** (a power lag would let switching
contamination leak back in through the residual). 20 features:

- **Calendar / time (8):** `local_time_of_day_sin/cos`, `local_time_of_year_sin/cos`,
  `local_day_of_week_sin/cos`, `local_day_of_week`, `local_utc_offset` (all in local wall-clock time).
- **ECMWF ENS weather (11):** `temperature_2m`, `dew_point_temperature_2m`, `wind_speed_10m`,
  `wind_direction_10m`, `wind_speed_100m`, `wind_direction_100m`, `pressure_surface`,
  `pressure_reduced_to_mean_sea_level`, `geopotential_height_500hpa`,
  `downward_short_wave_radiation_flux_surface`, `categorical_precipitation_type_surface`.
- **Derived (1):** `windchill`.

One model is trained per `time_series_id`.

## ⚠️ The ensemble is under-dispersed (don't trust the spread)

The 51-member spread **badly understates real uncertainty**. On this run the spread-skill ratio is
~0.12 (≈1 would be well-calibrated) and only ~12% of observations fall inside the nominal-**80%**
p10–p90 band. So the ensemble is heavily over-confident: treat `power_fcst_mean`/`power_fcst_p50` as
a point estimate, and **do not read the p10–p90 band as a real 80% interval** — it is far too narrow.

## How good are these? (please read)

Not very, yet — and that is expected. This is a deliberately minimal baseline: the XGBoost
improvements that would make it a *good* forecaster (better features, a proper probabilistic
objective, etc.) haven't been done yet. For reference, the overall normalised MAE is ~0.13, ranging
by asset type from ~0.02 to ~0.23.

**Recommendation:** these are fine for prototyping the residual/detection pipeline, but for any
serious analysis it is probably worth **waiting until we improve XGBoost (expected ~mid-August
2026)** and re-exporting, rather than building conclusions on this early baseline.
