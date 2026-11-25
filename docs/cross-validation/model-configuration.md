# Model configuration

How to set hyperparameters and choose features for a forecasting CV run.

---

## Config files

Each model family has a base YAML in `conf/model/`. Two exist today:
`conf/model/manual_heuristic.yaml` and `conf/model/climatology.yaml`. Each file has two required
top-level keys, shown here abridged from `conf/model/manual_heuristic.yaml`:

```yaml
# Identifies the BaseForecaster subclass to instantiate.
_target_: baseline_forecasters.manual_heuristic.ManualHeuristicForecaster

model_params:
  selected_features:
    - "power_lag_168h" # 1 week
    - "power_lag_336h" # 2 weeks
    - ...

  # MLflow experiment tags (used for leaderboard grouping)
  weather_source: "none"
  training_strategy: "none"
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
The individual strings inside the list are not parsed until training runs. The feature engineering
pipeline (`ml_core.features._parsed_features.ParsedFeatures.from_strings`) parses each one into a
typed descriptor and raises `ValueError` on any unrecognised or forbidden name, so a typo'd feature
name (e.g. `tempurature_2m`) surfaces only then, not at registration.

**Each forecaster reads only part of the catalogue below.** The manual heuristic reads power lags
and nothing else, and climatology reads no feature at all, so `conf/model/climatology.yaml` sets
`selected_features: []`. Both forecasters raise when `trained_cv_model` runs if the list holds a
feature they do not read, and the manual heuristic also raises on a list that holds no power lag.
The weather, time, and pass-through features are what the default
`TabularFeatureEngineer` can build for a forecaster that reads them.

### Power lags

| Pattern | Example | Notes |
|---|---|---|
| `power_lag_{N}h` | `power_lag_24h` | Observed power at `valid_time − N hours`. N must be a positive integer ≤ 17,520 (2 years). |

Power lags shorter than or equal to the forecast lead time are automatically nullified at
engineering time to prevent lookahead bias — see [Lookahead-bias
guardrails](#lookahead-bias-guardrails) below.

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
| `precipitation_surface` | Total precipitation rate (kg/m²/s) |
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
| `local_day_of_week` | Day-of-week name (`Monday`–`Sunday`), as a Polars `Enum` in that order |
| `local_utc_offset_minutes` | UTC offset in minutes (0 or 60 for GB; 330 for India's +5:30) |

### Static derived features

Derived from weather variables via a fixed formula; no time-shifting.

| Feature name | Definition |
|---|---|
| `windchill` | Wind-chill temperature from `temperature_2m` and `wind_speed_10m` (°C) |

### Pass-through base columns

These columns from the `AllFeatures` frame can be passed directly to the model as-is (e.g. to let
the model learn horizon-dependent biases).

| Feature name | Description |
|---|---|
| `nwp_lead_time_hours` | Hours between `nwp_init_time` and `valid_time` |
| `ensemble_member` | ECMWF ensemble member index (0–50) |
| `time_series_id` | Substation identifier (integer) |
| `time_series_type` | Asset category, e.g. `PV`, `Wind`, `Disaggregated Demand` (one of `LIST_OF_TIME_SERIES_TYPES`) |
| `power_fcst_init_time` | When the forecast was issued |
| `nwp_init_time` | When the NWP model ran |

---

## Lookahead-bias guardrails

Two feature names are **forbidden** and raise `ValueError` at parse time:

- **`power`** — requesting the raw target variable as an input would let the model learn a trivial
  identity function, useless at inference time.
- **`valid_time`** — an index column; use time features (e.g. `local_time_of_day_sin`) instead.

**Power lags** are automatically nullified by the feature engineering pipeline when the lag is
shorter than or equal to the forecast lead time. For example, `power_lag_1h` would leak observed
power into a 1-hour-ahead forecast, so its value is set to `null` for those rows. The forecaster
sees a null: the manual heuristic leaves a null lag out of that row's quantiles, and drops a row
whose lags are all null. This nullification happens
per-row in `_nullify_leaky_lags()`, not at config time — the feature name is still valid; the
pipeline just makes it safe.

Weather lags and rolling means are **never** nullified: NWP forecasts cover future `valid_time`s, so
a weather feature is always available at inference time regardless of lead time.

---

## Config fields

These fields live in `BaseForecasterConfig`. Both baseline forecasters validate `model_params` with
`BaseForecasterConfig` itself, so neither has a field beyond this table. A model family with
hyperparameters of its own declares them on a subclass of `BaseForecasterConfig`.

| Field | Default | Description |
|---|---|---|
| `selected_features` | (required) | List of feature name strings — see above. |
| `weather_source` | `""` | Leaderboard tag, e.g. `"ecmwf_control"` or `"ecmwf_ens"`. |
| `training_strategy` | `""` | Leaderboard tag, e.g. `"horizon_as_feature"`. |
| `random_seed` | `0` | Seed for a forecaster's random draws, so re-training a fold reproduces the same model. Neither baseline forecaster draws at random, so for them the field changes only the CV run's identity. |
| `model_release` | `0` | Integer stamped on every forecast row as `power_fcst_model_version`. `0` is the default, for a run that is not a numbered release. Range 0 to 32767. |
| `experiment_name` | `""` | Set from the job's own `experiment_name`; cannot be overridden. |
| `ml_flow_experiment_id` | `None` | Stamped onto every `PowerForecast` row. Nothing sets it automatically. |

---

## Tweaking a config for a CV run

You never edit a YAML file per CV run. Instead, pass `config_overrides` to
`register_experiment_job`. Each override is applied to `model_params` before the config object is
constructed.

**Every key must name a field the config class declares** — the table above. Write
`random_sede` instead of `random_seed` and registration fails, before a single fold is scheduled,
with a `ValidationError` naming the key. This matters more than a typo usually would, because the
searches planned to drive most registrations will run unattended. The
[large language model (LLM) agent](../roadmap/auto-research.md) planned for automated
ML experiments will train and score screening runs with nobody in the loop. The
[variant grid](../roadmap/training-history.md) planned for the training-history CV runs will
sweep several dimensions at once. A key that was quietly dropped would give you a grid of
*identical* runs, each scoring plausibly, each landing on the leaderboard, and nothing to
distinguish that grid from a genuine null result. That is
[principle 8](../design-philosophy/design-principles.md#8-every-cv-run-is-scored-identically).

Two further keys are refused for their own reasons. `_target_` names the forecaster class, and the
config class follows from it: to use a different one, point `base_model_config` at a different YAML.
`experiment_name` comes from the job's own `experiment_name` parameter, which would overwrite an
override of it.

**Example — run the manual heuristic on the weekly analogues alone:**

```json
{
  "selected_features": [
    "power_lag_168h",
    "power_lag_336h",
    "power_lag_504h",
    "power_lag_672h",
    "power_lag_840h",
    "power_lag_1008h"
  ]
}
```

Every override is a **whole-value replacement**, not a merge. `selected_features` is the case you
will meet first: to add one feature to the base set or drop one from it, you must list all the
features you want, not just the one that changes. The same holds when a `model_params` value is
itself a mapping — an override replaces the whole mapping, dropping the base's other keys, so
restate every key you want to keep.

The resolved config (the YAML defaults with your overrides applied) is frozen as a JSON tag on the
MLflow experiment at registration time. That frozen record is what `trained_cv_model` reads back at
train time — so changing the YAML after registering a CV run has no effect on it.
