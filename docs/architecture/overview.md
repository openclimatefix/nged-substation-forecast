# Architecture Overview

The system is designed as a modular monorepo using [uv workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/), with [Dagster](https://dagster.io/) orchestrating the data pipeline and [MLflow](https://mlflow.org/) tracking experiments.

## Core Components

* **Environment & Modularity**: `uv` workspace (Monorepo). Python 3.14. Individual components must be pip-installable with expressive type hints.
* **Data Processing**: **Polars**. Chosen for extreme speed and its native `join_asof` functionality to guarantee no future-data leakage during feature engineering.
    * **Centralized Data Preparation**: All data entering ML models passes through a centralized preparation step to enforce strict data contracts, handle missing entities, and ensure consistency between training and inference.
* **Storage**: **Delta Lake** on cloud object storage for the power data and forecasts. Upgraded from raw Parquet to provide ACID transactions, time-travel, and efficient partitioning (by year/month) for K-fold cross-validation. The weather data will use raw Parquet with well-defined naming conventions (we use raw Parquet for NWP data because Delta Lake doesn't support `uint8`).
* **Orchestration**: **Dagster**. Manages the pipeline via Software-Defined Assets (SDAs). Partitioned by NWP init time, not substation, allowing models to train globally across all substations (if they want to). Dagster is responsible for detecting bad data.
* **Configuration Management**: **Hydra** combined with `pydantic-settings`. Dagster passes string overrides (e.g., model=xgboost_global) to trigger Hydra's Config Groups, swapping massive architectural parameter trees. The resolved YAML is logged to MLflow.
* **Experiment Tracking**: **MLflow**.
* **Visualisation**: Altair for plotting, Marimo for interactive data exploration and web apps.

## The Universal Model Interface

All forecasting models subclass `BaseForecaster` (defined in `ml_core`), which provides a common `train` / `predict` / `save` / `load` interface. The model wrapper encapsulates the model weights and all translation logic, keeping Dagster assets completely agnostic to the underlying implementation. Each subclass of `BaseForecaster` is responsible for defining:

- _Input translation_: Transforms the canonical `AllFeatures` Polars LazyFrame into the required model shape.
- _Output translation_: Converts native model outputs into the strict `PowerForecast` schema.
- _Persistence_: Each subclass owns its own save/load format. `XGBoostForecaster` writes one `.ubj` file per `time_series_id` plus a `meta.json` containing the full serialised `XGBoostConfig`. (This may change later. We may switch to saving models using native MLflow flavors (e.g., `mlflow.xgboost.log_model`), which serialize the raw model object directly.)
- _Identity_: Model name, version, and optional MLflow experiment ID travel with the config, so every `PowerForecast` row is self-describing.
