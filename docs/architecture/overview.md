# Architecture Overview

The architecture prioritises developer velocity, idempotent re-runs, and strict **Training-Serving Symmetry**.

The primary aim is to develop novel, ambitious, state-of-the-art ML approaches to forecasting. We are simultaneously building a "test-harness" production service so that ML research runs in a production-like environment from day one.

The aim is to manage the *entire* data pipeline in Dagster: download data, validate data, train ML models, run inference, perform backtests. MLflow tracks every experiment. Re-running a backtest should be as easy as clicking a button in Dagster. Swapping a new model into production should require minimal friction.

The system is designed as a modular monorepo using [uv workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/), with [Dagster](https://dagster.io/) orchestrating the data pipeline and [MLflow](https://mlflow.org/) tracking experiments.

## Core Components

* **Environment & Modularity**: `uv` workspace (Monorepo). Python 3.14. Individual components must be pip-installable with expressive type hints.
* **Data Processing**: **Polars**. Chosen for extreme speed and its native `join_asof` functionality to guarantee no future-data leakage during feature engineering.
    * **Centralized Data Preparation**: All data entering ML models passes through a centralized preparation step to enforce strict data contracts, handle missing entities, and ensure consistency between training and inference.
* **Storage**: **Delta Lake** on cloud object storage for both power data and NWP weather data. Delta Lake provides ACID transactions, time-travel, and efficient partitioning. NWP data is quantised to 12-bit precision and stored as `Int16` (roughly halving the size versus `float32`), with `zstd` level-14 compression — compressing the full ECMWF ENS dataset for Great Britain to approximately 40 GB per year. A single day's NWP data takes about one minute to download and convert.
* **Orchestration**: **Dagster**. Manages the pipeline via Software-Defined Assets (SDAs). Partitioned by NWP init time, not substation, allowing models to train globally across all substations (if they want to). Dagster is responsible for detecting bad data.
* **Configuration Management**: **Hydra** combined with `pydantic-settings`. Dagster passes string overrides (e.g., model=xgboost_global) to trigger Hydra's Config Groups, swapping massive architectural parameter trees. The resolved YAML is logged to MLflow.
* **Experiment Tracking**: **MLflow**.
* **Visualisation**: Altair for plotting, Marimo for interactive data exploration and web apps.

## Lazy Evaluation Strategy

**Rule: everything stays lazy until the model boundary.**

The entire feature-engineering pipeline operates on Polars `LazyFrame`s. No code between reading from storage and calling `BaseForecaster.train()` or `predict()` is allowed to call `.collect()`. The pipeline is:

```
NwpOnDisk.scan_delta(path)               # lazy scan
  → NwpOnDisk.to_nwp_in_memory()        # lazy expression; no collect
  → FeatureEngineer.engineer()           # spatial join + feature pipeline;
                                         #   returns pt.LazyFrame[AllFeatures]; no collect
  → BaseForecaster.train/predict()       # subclass calls .collect() exactly once, right here
```

**Why it matters:**

- **Memory**: Polars only materialises the data once, at the last moment. Intermediate representations (individual join outputs, lag columns before filtering) are never written to RAM.
- **Query optimisation**: Polars can see the full plan end-to-end and push down filters (e.g. date ranges, `time_series_id` subsets) into the Delta Lake scan before any data crosses the wire.
- **Clean boundaries**: The collect site is the model boundary — the one place where a third-party library (XGBoost, PyTorch, …) needs a concrete in-memory array. Keeping it there makes the boundary explicit and easy to find.

**The one exception** is a `limit(1).collect()` guard in `_build_historical_weather` (inside `ml_core`) that cheaply checks for the NWP control member before building the lazy plan, so the pipeline fails loudly instead of silently returning an empty frame.

**Contract for `BaseForecaster` subclasses**: call `.collect()` exactly once, as late as possible — typically the line immediately before constructing the model's input matrix. Never ask callers to collect before passing data in.

## The Universal Model Interface

All forecasting models subclass `BaseForecaster` (defined in `ml_core`), which provides a common `train` / `predict` / `save` / `load` interface. The model wrapper encapsulates the model weights and all translation logic, keeping Dagster assets completely agnostic to the underlying implementation. Each subclass of `BaseForecaster` is responsible for defining:

- _Feature engineering_: Each subclass carries a `feature_engineer: ClassVar[FeatureEngineer]` strategy (composition, not inheritance) that owns the full preparation pipeline — from raw inputs (observed power, gridded NWP, time-series metadata) to an `AllFeatures` frame. The default `TabularFeatureEngineer` does the nearest-cell NWP spatial join then runs the tabular feature pipeline. A future model that needs a different data view (e.g. a CNN wanting a spatial NWP crop per time series) overrides `feature_engineer` with a different `FeatureEngineer` subclass without touching `BaseForecaster` or any other model. `FeatureEngineer` and `TabularFeatureEngineer` live in `packages/ml_core/src/ml_core/features/`.
- _Input translation_: Transforms the canonical `AllFeatures` Polars LazyFrame into the required model shape.
- _Output translation_: Converts native model outputs into the strict `PowerForecast` schema.
- _Persistence_: Each subclass owns its own save/load format. `XGBoostForecaster` writes one `.ubj` file per `time_series_id` plus a `meta.json` containing the full serialised `XGBoostConfig`. (This may change later. We may switch to saving models using native MLflow flavors (e.g., `mlflow.xgboost.log_model`), which serialize the raw model object directly.)
- _Identity_: Model name, version, and optional MLflow experiment ID travel with the config, so every `PowerForecast` row is self-describing.
