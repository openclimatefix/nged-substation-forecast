# Architecture Overview

This page and the rest of `architecture/` describe **what is actually built** — the components,
their responsibilities, and the local design rationale recorded next to each. The transferable
*why* — the [design principles](../design-philosophy/design-principles.md) that govern these
decisions, and the [engineering hypotheses](../design-philosophy/engineering-hypotheses.md) they
serve — lives in the [Design Philosophy](../design-philosophy/index.md) section.

The architecture prioritises developer velocity, idempotent re-runs, and strict **training–serving symmetry** — [one execution path from research to production](../design-philosophy/design-principles.md#3-one-execution-path-from-research-to-production), so [nothing gets rewritten on the way to production](../ml_experimentation/index.md#nothing-gets-rewritten-on-the-way-to-production). The primary aim is to develop novel, ambitious, state-of-the-art machine-learning (ML) approaches to forecasting. We are simultaneously building a "test-harness" production service so that ML research runs in a production-like environment from day one.

The aim is to manage the *entire* data pipeline in Dagster: download data, validate data, train ML models, run inference, perform backtests. MLflow tracks every experiment. Re-running a backtest should be as easy as clicking a button in Dagster. Swapping a new model into production should require minimal friction.

The system is designed as a modular monorepo using [uv workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/), with [Dagster](https://dagster.io/) orchestrating the data pipeline and [MLflow](https://mlflow.org/) tracking experiments. The measured performance engineering behind these choices — storage formats, lazy evaluation, memory bounds, and Polars' row-index ceiling — lives on [Performance and Scale](performance.md).

## Core Components

* **Environment & Modularity**: `uv` workspace (Monorepo). Python 3.14. Individual components must be pip-installable with expressive type hints.
* **Data Processing**: **Polars**. Chosen for extreme speed and its native `join_asof` functionality to guarantee no future-data leakage during feature engineering.
    * **Centralised data preparation**: All data entering ML models passes through a centralised preparation step to enforce strict data contracts, handle missing entities, and ensure consistency between training and inference.
* **Storage**: **Delta Lake** on cloud object storage for both power data and numerical weather prediction (NWP) data. Delta Lake provides ACID transactions, time-travel, and efficient partitioning. The same technology is also how forecasts are [delivered to NGED](forecast-delivery.md). Each table's physical layout is owned by the `delta_store` package (as `contracts` owns its logical shape); the layouts were chosen by measurement rather than assumption, and the measured numbers are on [Performance and Scale → Storage formats](performance.md#storage-formats-measured-not-assumed).
* **Orchestration**: **Dagster**. Manages the pipeline via Software-Defined Assets (SDAs). Partitioned by NWP init time, not substation, allowing models to train globally across all substations (if they want to). Dagster is responsible for detecting bad data.
    * **Every asset says whether the live service needs it**, as a `layer` tag valued `production` or `research` (`defs/_tags.py`). The four `production` assets — `power_time_series_and_metadata`, `h3_grid_weights`, `ecmwf_ens`, and `live_forecasts` — are everything the deployment runs to produce forecasts. The `research` assets are everything else: the cross-validation assets, plus `promotable_model_runs` and `promoted_model`, which need an MLflow tracking server the deployment does not reach. The tag says whether the service needs an asset, not where the asset runs: the `research` assets all run on a researcher's laptop today, and some may move to the cloud later, but never onto the VM that serves the forecasts. Filter either side in the Dagster UI's selection box, or on the command line, with `tag:layer=production` / `tag:layer=research`.
* **Configuration Management**: plain YAML for model configs, `pydantic` for validating them, and `pydantic-settings` for environment-derived settings. A model YAML in `conf/model/` names its `BaseForecaster` subclass as a `_target_` import path, which `contracts.config_schemas.import_class` resolves. `register_experiment_job` applies the run's `config_overrides` on top and constructs that class's `CONFIG_CLASS`, so pydantic validates every hyperparameter at registration time — its name as well as its value. The resolved config is logged to MLflow.
* **Experiment Tracking**: **MLflow**.
* **Visualisation**: Altair for plotting, Marimo for interactive data exploration and web apps.

## The Universal Model Interface

All forecasting models subclass `BaseForecaster` (defined in `ml_core`), which provides a common `train` / `predict` / `save` / `load` interface. The model wrapper encapsulates the model weights and all translation logic, keeping Dagster assets completely agnostic to the underlying implementation. Each subclass of `BaseForecaster` is responsible for defining:

* _Feature engineering_: Each subclass carries a `feature_engineer: ClassVar[FeatureEngineer]` strategy (composition, not inheritance) that owns the full preparation pipeline — from raw inputs (observed power, gridded NWP, time-series metadata) to an `AllFeatures` frame. The default `TabularFeatureEngineer` does the nearest-cell NWP spatial join then runs the tabular feature pipeline. A future model that needs a different data view (e.g. a CNN wanting a spatial NWP crop per time series) overrides `feature_engineer` with a different `FeatureEngineer` subclass without touching `BaseForecaster` or any other model. `FeatureEngineer` and `TabularFeatureEngineer` live in `packages/ml_core/src/ml_core/features/`.
* _Input translation_: Transforms the canonical `AllFeatures` Polars LazyFrame into the required model shape.
* _Output translation_: Converts native model outputs into the strict `PowerForecast` schema.
* _Persistence_: Each subclass owns its own save/load format. `XGBoostForecaster` writes one `.ubj` file per `time_series_id` plus a `meta.json` containing the full serialised `XGBoostConfig`. (This may change later. We may switch to saving models using native MLflow flavors (e.g., `mlflow.xgboost.log_model`), which serialize the raw model object directly.)
* _Identity_: `MODEL_NAME` and `MODEL_VERSION` are class-level constants on each `BaseForecaster` subclass, separate from `BaseForecasterConfig`. `experiment_name` and `ml_flow_experiment_id` travel in the config instead. Both levels are stamped onto every `PowerForecast` row at predict time, so each row is self-describing.
