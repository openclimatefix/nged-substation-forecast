# `ml_core`

Unified ML model interface and shared utilities for substation forecasting.

## Purpose

This package provides the abstract base classes and shared logic required to implement the "Write-Once" ML pipeline. It strictly separates the **Trainer** (heavy training logic) from the **Model** (lightweight inference artifact).

## Why this package exists

To enable rapid experimentation with different model architectures (XGBoost, GNNs, etc.), we need a common interface that shields the model developer from the underlying data engineering (Dagster, Polars, Hydra).

## Dependency Isolation

This package is separate from `packages/contracts` to ensure that the data schemas remain lightweight. `ml_core` depends on `mlflow-skinny` and other ML-related libraries. By keeping this logic out of `contracts`, we prevent "dependency bloat" in non-ML components of the system.

## Key Components

- **`assets.py`**: The "Dagster Vocabulary" (FeatureAsset Enum).
- **`trainer.py`**: Base classes for trainers and data requirements.
- **`model.py`**: Base classes for inference artifacts (MLflow pyfunc wrappers).
- **`features.py`**: Shared feature engineering logic (e.g., cyclical time features).
- **`data.py`**: Shared data splitting and loading logic.
- **`scaling.py`**: Shared normalization and scaling utilities.
