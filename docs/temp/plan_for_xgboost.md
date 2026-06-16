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
- We will also have to implement some code to up-sample the NWP data to half-hourly (to match the
  half-hourly power data). This upsampling should live in `features.py`.
- We should implement a simple Pydantic class to capture the basic config options for
  `XGBoostForecaster`.

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
