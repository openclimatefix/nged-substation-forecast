# XGBoost Substation Forecaster

This package implements an XGBoost-based model to forecast power flows at NGED primary substations using numerical weather prediction (NWP) forecasts. It implements the `BaseForecaster` interface defined in `ml_core`.

## How it works

One `xgb.Booster` is trained per `time_series_id`, so each substation's model can learn its own relationship between weather and power. Features are passed via the `AllFeatures` schema (see `contracts`), which joins NWP variables, power lag/rolling features, and static metadata. Categorical and string columns are encoded as integer codes before being handed to XGBoost; all features are cast to `Float32`, and missing values are left as `NaN` so XGBoost handles them natively. The model is deterministic; the ensemble spread comes from each NWP ensemble member's weather flowing through the relevant booster — `predict()` scores every ensemble member present in its input in one call.

`train()` collects its input once and groups in memory by `time_series_id`, feeding each series to an `xgb.QuantileDMatrix` (8-bit quantile bins, not an uncompressed `Float32` copy). Keeping that collect bounded is the **caller's** job — the dominant cost is the multi-tens-of-GB NWP scan, which must be pruned at the *inputs* (control member, the relevant H3 cells, the window's `init_time` partitions) and streamed; filtering the engineered *output* cannot prune the upstream join/upsample. `predict()` likewise collects once and groups by `time_series_id`; at validation the full ~51-member ensemble is too large to collect whole, so the caller (`cv_power_forecasts`) predicts **one `init_time` chunk at a time**, writing to Delta incrementally. See [Bounding feature-engineering memory](../../docs/architecture/overview.md#bounding-feature-engineering-memory-prune-the-inputs-not-the-output) for the dataset sizes and the table of which predicates actually prune the NWP scan.

`_data_iter.py` (`LazyFrameBatchIter`) is **not currently used**. It is kept for the future experiment of training across many NWP ensemble members, where a single booster's data will not fit in memory and must be streamed — fed a per-`ensemble_member` predicate-filtered frame (or a parquet/Delta scan), *not* row slices of the full join.

## Save format

`XGBoostForecaster.save(path)` writes:
- `{time_series_id}.ubj` — one XGBoost native binary model per trained substation
- `meta.json` — the full `XGBoostConfig` serialised via Pydantic, so `load()` is completely self-contained

## Configuration

`XGBoostConfig` extends `BaseForecasterConfig` with XGBoost hyperparameters (`n_estimators`, `learning_rate`, `max_depth`, etc.) and the model identity fields (`power_fcst_model_name`, `power_fcst_model_version`, `ml_flow_experiment_id`). Identity fields are stamped onto every row of the `PowerForecast` output so the Delta Lake table is self-describing.
