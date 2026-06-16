Today, let's discuss a big new feature for this codebase: Implementing a simple XGBoost power
forecasting model.

- `XGBoostForecaster` code could live in packages/xgboost_forecaster/
- The XGBoost forecaster should subclass `BaseForecaster` (in `ml_core`), and should use
  `engineer_features` from `ml_core`).
- For today, let's keep the XGBoost code as simple as possible. For example, today we are _not_
  going to use the XGBoost iterator; we'll implement that later so we can train on more data than
  can fit into RAM.
- The XGBoost model will only see one NWP ensemble member at a time.
- We'll create an ensemble of power forecasts by passing each NWP ensemble member one by one through the XGBoost model.
- We'll train a different XGBoost model per `time_series_id`. That single XGBoost model per
  `time_series_id` will handle all lead times (at a later date we'll experiment with training
  multiple XGBoost models per `time_series_id`, for different buckets of lead times).
- We'll produce half-hourly forecasts for the next 14 days.
- We should implement a simple Pydantic class to capture the basic config options for
  `XGBoostForecaster`.
- Regarding `PowerForecast` metadata fields: PowerForecast has ml_flow_experiment_id, power_fcst_model_name, and power_fcst_model_version. These can't come from AllFeatures — they'd need to live in XGBoostConfig (or be stamped on at predict time via a separate argument). We could put these in the Pydantic config, or should predict() accept them as extra kwargs? I'd maybe lean towards predict() accepting them as extra, optional args. But I'd be keen to hear your thoughts. Note that I'd like to run the pipeline without `ml_flow` (e.g. to run a very simple integration test, using real data).
- Regarding save/load with one model per time_series_id. We'd end up with a dict[int, xgb.Booster]. The natural save layout would be a directory with one .ubj file per time_series_id. Alternatively, we could have a single serialised pickle/joblib for simplicity? I think I'd go for the first option (one file per time_series_id) but, again, I'd like to hear your thoughts, please.

What's already in place:
- packages/xgboost_forecaster/ exists with the right deps (xgboost, ml_core, contracts, mlflow-skinny) but is empty.
- BaseForecaster already has the selected_features / model_params pattern.
- engineer_features() already handles one-ensemble-member-at-a-time input vs. the full ensemble, and AllFeatures already carries ensemble_member, time_series_id, etc.

Later (maybe tomorrow), we'll implement some more of the pipeline:
- two simple Dagster assets: a `train` asset, and a `predict` asset.
  These assets should be completely agnostic to exactly which ML model they're using. Instead, all
  they care about is that they read a Hydra config, which specifies which subclass of
  `BaseForecaster` to instantiate.
- We'll also want to use Dagster to orchestrate expanding-window
  cross-fold validation (e.g. where the first fold trains on 2020 and validates on 2021, then the
  next fold trains on 2020 and 2021, and validates on 2022, etc.).
- And we'll implement another Dagster asset that
  takes a `PowerForecast` Delta table and plots predictions using Altair.
