# Architecture Overview

The system is designed as a modular monorepo using [uv workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/), with [Dagster](https://dagster.io/) orchestrating the data pipeline and [MLflow](https://mlflow.org/) tracking experiments.

## Core Components

* **Environment & Modularity**: `uv` workspace (Monorepo). Python 3.14. Individual components must be pip-installable with expressive type hints.
* **Data Processing**: **Polars**. Chosen for extreme speed and its native `join_asof` functionality to guarantee no future-data leakage during feature engineering.
    * **Centralized Data Preparation**: All data entering ML models passes through a centralized preparation step to enforce strict data contracts, handle missing entities, and ensure consistency between training and inference.
    * **Strategy Pattern for Ingestion**: CSV parsing uses a Strategy pattern to handle diverse and evolving data formats (power data, metadata, switching events) without modifying core ingestion logic.
* **Storage**: **Delta Lake** on cloud object storage for the power data and forecasts. Upgraded from raw Parquet to provide ACID transactions, time-travel, and efficient partitioning (by year/month) for K-fold cross-validation. The weather data will use raw Parquet with well-defined naming conventions (we use raw Parquet for NWP data because Delta Lake doesn't support `uint8`).
* **Orchestration**: **Dagster**. Manages the pipeline via Software-Defined Assets (SDAs). Partitioned by NWP init time, not substation, allowing models to train globally across all substations (if they want to). Dagster is responsible for detecting bad data.
* **Configuration Management**: **Hydra** combined with `pydantic-settings`. Dagster passes string overrides (e.g., model=xgboost_global) to trigger Hydra's Config Groups, swapping massive architectural parameter trees. The resolved YAML is logged to MLflow.
* **Experiment Tracking**: **MLflow**.
* **Visualisation**: Altair for plotting, Marimo for interactive data exploration and web apps.

## The Universal Model Architecture

To maximize developer velocity and model robustness we use a **Single Universal Model** architecture (see [ADR-016](../architecture_decision_records/016-universal-model-architecture.md)).

**Key Characteristics:**
- **Horizon-Agnostic Training**: A single model is trained on a "long-format" dataset where each row represents a unique combination of (substation, initialization time, valid time).
- **Lead Time as a Feature**: The model explicitly receives `lead_time_hours` as a primary feature.
- **Global Substation Training**: The model is trained across all substations simultaneously, using `substation_number` as a categorical feature.
- **Ensemble Support**: The architecture natively supports probabilistic forecasting by running inference over each NWP ensemble member individually.

## The Universal Model Interface

To decouple the Dagster data pipeline from the ML code, all models are saved using native MLflow flavors (e.g., `mlflow.xgboost.log_model`), which serialize the raw model object directly (see [ADR-005](../architecture_decision_records/005-zero-magic-ml-pipeline.md)).

**The Adapter Pattern**: The model wrapper encapsulates the model weights and all translation logic.

- _Input translation_: Transforms the canonical Polars DataFrame into the required model shape.
- _Output translation_: Converts native model outputs into the strict target quantile schema.
