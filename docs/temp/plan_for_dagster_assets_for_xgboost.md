Today, let's discuss a new feature: Implementing Dagster assets to train and evaluate our new
XGBoost energy forecasting model.

- Let's discuss the design.
- I'm imagining we'll build two simple Dagster assets: a `train` asset, and a `predict` asset.
  These assets should be completely agnostic to exactly which ML model they're using. Instead, all
  they care about is that they read a Hydra config, which specifies which subclass of
  `BaseForecaster` to instantiate, and the params for that model.
- We'll also want to use Dagster to orchestrate expanding-window
  cross-fold validation (e.g. where the first fold trains on 2020 and validates on 2021, then the
  next fold trains on 2020 and 2021, and validates on 2022, etc.).
- And we'll implement another Dagster asset that
  takes a `PowerForecast` Delta table and plots predictions using Altair.
- In general, I'm imagining that Dagster will be responsible for lazily loading the NWP data and
  power data, and filtering to include only the appropriate time windows, and then passing these
  Polars LazyFrames to the ML model.

What's already in place:
- packages/xgboost_forecaster/ contains a functional, minimal XGBoostForecaster. It eagerly loads
  all the data it's given into RAM (so we have to be careful not to give it more data than will fit
  into RAM!). Each XGBoost model only sees a single NWP ensemble member at a time. We create an
  ensemble of _power_ forecasts by passing each NWP ensemble member through the XGBoost model. The
  system trains a different XGBoost model per `time_series_id`.
- throughout this project, we are making 14-day forecasts, at half-hourly resolution, and we
  (almost) always want to produce an ensemble of power forecasts.

