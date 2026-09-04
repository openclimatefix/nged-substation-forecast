# XGBoost Substation Forecaster

This package implements an XGBoost-based model to forecast power flows at NGED primary substations using numerical weather prediction (NWP) forecasts. It implements the `BaseForecaster` interface defined in `ml_core`.

## How it works

One `xgb.Booster` is trained per `time_series_id`, so each substation's model can learn its own relationship between weather and power. Features are passed via the `AllFeatures` schema (see `contracts`), which joins NWP variables, power lag/rolling features, and static metadata. Categorical and string columns are encoded as integer codes before being handed to XGBoost; all features are cast to `Float32`, and missing values are left as `NaN` so XGBoost handles them natively. The model is deterministic, and an ensemble forecast still comes out of it — the `XGBoostForecaster` class below says how.

Both `train()` and `predict()` collect their input once, so keeping that collect bounded is the **caller's** job: the dominant cost is the multi-tens-of-GB NWP scan, which has to be pruned at the *inputs* and streamed, because filtering the engineered *output* cannot prune it. The `train` and `predict` docstrings below give the mechanics. For the dataset sizes and the table of which predicates actually prune the NWP scan, see [Bounding feature-engineering memory](https://openclimatefix.github.io/nged-substation-forecast/architecture/performance/#bounding-feature-engineering-memory-prune-the-inputs-not-the-output).

## Save format

`XGBoostForecaster.save(path)` writes:

- `{time_series_id}.ubj` — one XGBoost native binary model per trained substation
- `meta.json` — the full `XGBoostConfig` serialised via Pydantic, so `load()` is completely self-contained

## Configuration

`XGBoostConfig` extends `BaseForecasterConfig` with XGBoost hyperparameters (`n_estimators`, `learning_rate`, `max_depth`, etc.). `BaseForecasterConfig` contributes `selected_features`, `random_seed` (threaded into XGBoost's own `seed` parameter for deterministic training), the experiment-identity fields `experiment_name` and `ml_flow_experiment_id`, and the leaderboard tag fields `weather_source` and `training_strategy`. Model-family identity — `MODEL_NAME` ("xgboost") and `MODEL_VERSION` — lives on the `XGBoostForecaster` class itself, not in the config; both a config's experiment identity and the class's model-family identity are stamped onto every row of the `PowerForecast` output, so the Delta Lake table is self-describing. `XGBoostConfig` inherits `extra="forbid"` from `BaseForecasterConfig`, so a key that names neither an `XGBoostConfig` field nor an inherited `BaseForecasterConfig` field raises `ValidationError` rather than being silently ignored.
