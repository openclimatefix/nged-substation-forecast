# XGBoost Substation Forecaster

This package implements an XGBoost-based model to forecast power flows at NGED primary substations using numerical weather prediction (NWP) forecasts. It implements the `BaseForecaster` interface defined in `ml_core`.

## How it works

One `xgb.Booster` is trained per `time_series_id`, so each substation's model can learn its own relationship between weather and power. Features are passed via the `AllFeatures` schema (see `contracts`), which joins NWP variables, power lag/rolling features, and static metadata. Categorical and string columns are encoded as integer codes before being handed to XGBoost; all features are cast to `Float32`, and missing values are left as `NaN` so XGBoost handles them natively. Ensemble power forecasts are produced by calling `predict()` once per NWP ensemble member — the model itself is deterministic.

Training is **memory-bounded**: rather than collecting the whole `AllFeatures` frame, `train()` streams each series' rows to an `xgb.QuantileDMatrix` in fixed-size batches (`train_batch_size`) via `LazyFrameBatchIter`, collecting one batch at a time off the lazy Polars plan. The full `Float32` dataset is never resident and XGBoost never builds a second uncompressed `DMatrix` copy, so peak memory stays bounded as the dataset scales. The iterator is grouping-agnostic — it batches whatever `LazyFrame` it is given — so a future model that trains one booster across many series feeds the same iterator a frame spanning multiple `time_series_id`s.

## Save format

`XGBoostForecaster.save(path)` writes:
- `{time_series_id}.ubj` — one XGBoost native binary model per trained substation
- `meta.json` — the full `XGBoostConfig` serialised via Pydantic, so `load()` is completely self-contained

## Configuration

`XGBoostConfig` extends `BaseForecasterConfig` with XGBoost hyperparameters (`n_estimators`, `learning_rate`, `max_depth`, etc.) and the model identity fields (`power_fcst_model_name`, `power_fcst_model_version`, `ml_flow_experiment_id`). Identity fields are stamped onto every row of the `PowerForecast` output so the Delta Lake table is self-describing.
