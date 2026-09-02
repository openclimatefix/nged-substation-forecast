# Model configuration

How to set hyperparameters and choose features for a forecasting experiment.

---

## Config files

Each model family has a base YAML in `conf/model/`. The only one today is
`conf/model/xgboost.yaml`. The file has two required top-level keys:

```yaml
# Identifies the BaseForecaster subclass to instantiate.
_target_: xgboost_forecaster.forecaster.XGBoostForecaster

model_params:
  selected_features:
    - "power_lag_24h"
    - "temperature_2m"
    - ...

  # Model-specific hyperparameters:
  n_estimators: 500
  learning_rate: 0.05
  ...
```

`_target_` is a fully-qualified Python class path, resolved by
`contracts.config_schemas.import_class` at registration time. You should not change it unless you
are wiring up a new model family. The class that validates `model_params` is not named here: it is
the forecaster's `CONFIG_CLASS`, the same class its `load` rebuilds a saved config with, so the two
cannot drift apart.

---

## Features (`selected_features`)

`selected_features` is a set of strings, written in the YAML as a list. Registration only checks
that it is a list of strings coercible to that set — a typo'd top-level key (e.g.
`selected_featuers`) is rejected there, by pydantic's `extra="forbid"` on `BaseForecasterConfig`.
The individual strings inside the list are not parsed until training runs. The feature
engineering pipeline (`ml_core.features._parsed_features.ParsedFeatures.from_strings`) parses
each one into a typed descriptor and raises `ValueError` on any unrecognised or forbidden name,
so a typo'd feature name (e.g. `tempurature_2m`) surfaces only then, not at registration.

### Power lags

| Pattern | Example | Notes |
|---|---|---|
| `power_lag_{N}h` | `power_lag_24h` | Observed power at `valid_time − N hours`. N must be a positive integer ≤ 17,520 (2 years). |

Power lags shorter than or equal to the forecast lead time are automatically nullified at
engineering time to prevent lookahead bias — see [Lookahead-bias guardrails](#lookahead-bias-guardrails) below.

### Raw weather variables

These are the NWP variables available directly from ECMWF ENS at `valid_time`.

| Feature name | Variable |
|---|---|
| `temperature_2m` | 2 m temperature (°C) |
| `dew_point_temperature_2m` | 2 m dew point (°C) |
| `wind_speed_10m` | Wind speed at 10 m (m/s) |
| `wind_direction_10m` | Wind direction at 10 m (°) |
| `wind_speed_100m` | Wind speed at 100 m (m/s) |
| `wind_direction_100m` | Wind direction at 100 m (°) |
| `pressure_surface` | Surface pressure (Pa) |
| `pressure_reduced_to_mean_sea_level` | MSLP (Pa) |
| `geopotential_height_500hpa` | 500 hPa geopotential height (m) |
| `downward_long_wave_radiation_flux_surface` | Downward LW radiation (W/m²) |
| `downward_short_wave_radiation_flux_surface` | Downward SW radiation (W/m²) |
| `precipitation_surface` | Total precipitation (kg/m²) |
| `categorical_precipitation_type_surface` | Precipitation type (categorical) |

### Weather lags and rolling means

Any raw weather variable can be lagged or smoothed:

| Pattern | Example | Notes |
|---|---|---|
| `{weather_var}_lag_{N}h` | `temperature_2m_lag_6h` | NWP value at `valid_time − N hours`. Never leaky (NWP forecasts are available for future times). |
| `{weather_var}_rolling_mean_{N}h` | `temperature_2m_rolling_mean_6h` | Mean of the weather variable over the N-hour window ending at `valid_time`. Never leaky. |

N must be a positive integer ≤ 17,520 (2 years).

### Time features

Cyclical and calendar encodings computed from `valid_time` in local wall-clock time (respecting
DST). Electricity demand follows human behaviour, which tracks local time not UTC.

| Feature name | Encoding |
|---|---|
| `local_time_of_day_sin` | Sine of fraction of day (24 h period) |
| `local_time_of_day_cos` | Cosine of fraction of day (24 h period) |
| `local_time_of_year_sin` | Sine of fraction of year (ordinal day of year divided by 366, so a given calendar date's fraction shifts by one day's worth after 29 February in a leap year) |
| `local_time_of_year_cos` | Cosine of fraction of year (see `local_time_of_year_sin`) |
| `local_day_of_week_sin` | Sine of day-of-week (7 day period) |
| `local_day_of_week_cos` | Cosine of day-of-week (7 day period) |
| `local_day_of_week` | Day-of-week name as a categorical (`Monday`–`Sunday`) |
| `local_utc_offset_minutes` | UTC offset in minutes (0 or 60 for GB; 330 for India's +5:30) |

### Static derived features

Derived from weather variables via a fixed formula; no time-shifting.

| Feature name | Definition |
|---|---|
| `windchill` | Wind-chill temperature from `temperature_2m` and `wind_speed_10m` (°C) |

### Pass-through base columns

These columns from the `AllFeatures` frame can be passed directly to the model as-is (e.g. to
let the model learn horizon-dependent biases).

| Feature name | Description |
|---|---|
| `nwp_lead_time_hours` | Hours between `nwp_init_time` and `valid_time` |
| `ensemble_member` | ECMWF ensemble member index (0–50) |
| `time_series_id` | Substation identifier (integer) |
| `time_series_type` | Asset category, e.g. `PV`, `Wind`, `Disaggregated Demand` (categorical string) |
| `power_fcst_init_time` | When the forecast was issued |
| `nwp_init_time` | When the NWP model ran |

---

## Lookahead-bias guardrails

Two feature names are **forbidden** and raise `ValueError` at parse time:

- **`power`** — requesting the raw target variable as an input would let the model learn a
  trivial identity function, useless at inference time.
- **`valid_time`** — an index column; use time features (e.g. `local_time_of_day_sin`) instead.

**Power lags** are automatically nullified by the feature engineering pipeline when the lag is
shorter than or equal to the forecast lead time. For example, `power_lag_1h` would leak
observed power into a 1-hour-ahead forecast, so its value is set to `null` for those rows. The
model sees a null and treats it as a missing value (XGBoost handles nulls natively). This
nullification happens per-row in `_nullify_leaky_lags()`, not at config time — the feature name
is still valid; the pipeline just makes it safe.

Weather lags and rolling means are **never** nullified: NWP forecasts cover future `valid_time`s,
so a weather feature is always available at inference time regardless of lead time.

---

## XGBoost hyperparameters

These fields live in `XGBoostConfig` (which inherits the universal fields from
`BaseForecasterConfig`).

### Universal fields (all model families)

| Field | Default | Description |
|---|---|---|
| `selected_features` | (required) | List of feature name strings — see above. |
| `weather_source` | `""` | Leaderboard tag, e.g. `"ecmwf_control"` or `"ecmwf_ens"`. |
| `training_strategy` | `""` | Leaderboard tag, e.g. `"horizon_as_feature"`. |
| `random_seed` | `0` | Passed to XGBoost's `seed` param; makes re-training a fold reproduce the same model. |
| `experiment_name` | `""` | Set from the job's own `experiment_name`; cannot be overridden. |
| `ml_flow_experiment_id` | `None` | Stamped onto every `PowerForecast` row. Nothing sets it automatically. |

### XGBoost-specific fields

| Field | Default | Description |
|---|---|---|
| `n_estimators` | `1000` | Number of boosting rounds (`num_boost_round`). |
| `learning_rate` | `0.05` | Step size shrinkage (`eta`). |
| `max_depth` | `6` | Maximum tree depth. |
| `min_child_weight` | `1` | Minimum sum of instance weights in a child. |
| `subsample` | `0.8` | Fraction of rows sampled per tree. |
| `colsample_bytree` | `0.8` | Fraction of features sampled per tree. |
| `device` | `"cpu"` | `"cpu"` or `"cuda"`. |
| `objective` | `"reg:squarederror"` | Loss function. |

---

## Tweaking a config for an experiment

You never edit a YAML file per experiment. Instead, pass `config_overrides` to
`register_experiment_job`. Each override is applied to `model_params` before the config object is
constructed.

**Every key must name a field the config class declares** — the two tables above. Write
`n_estimtors` instead of `n_estimators` and registration fails, before a single fold is scheduled,
with a `ValidationError` naming the key. This matters more than a typo usually would, because the
searches that drive most registrations are unattended. The LLM auto-research agent registers,
materialises, and reads the leaderboard with nobody in the loop, and the variant grid sweeps
several dimensions at once. A key that was quietly dropped would give you a grid of *identical*
runs, each scoring plausibly, each landing on the leaderboard, and nothing to distinguish that
grid from a genuine null result. That is [principle
8](../design-philosophy/design-principles.md#8-every-experiment-is-scored-identically).

Two further keys are refused for their own reasons. `_target_` names the forecaster class, and the
config class follows from it: to use a different one, point `base_model_config` at a different
YAML. `experiment_name` comes from the job's own `experiment_name` parameter, which would overwrite
an override of it.

**Example — reduce tree depth and add a feature:**

```json
{
  "max_depth": 4,
  "selected_features": [
    "power_lag_24h",
    "power_lag_48h",
    "temperature_2m",
    "local_time_of_day_sin",
    "local_time_of_day_cos"
  ]
}
```

Every override is a **whole-value replacement**, not a merge. `selected_features` is the case
you will meet first: to add one feature to the baseline set you must list all the features you
want, not just the new one. The same holds when a `model_params` value is itself a mapping —
an override replaces the whole mapping, dropping the base's other keys, so restate every key
you want to keep.

The resolved config (the YAML defaults with your overrides applied) is frozen as a JSON tag on the
MLflow experiment at registration time. That frozen record is what `trained_cv_model` reads back
at train time — so changing the YAML after registering an experiment has no effect on it.
