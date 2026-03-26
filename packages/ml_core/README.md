# `ml_core`

Unified ML model interface and shared utilities for substation forecasting.

## Purpose

This package provides the abstract base classes and shared logic required to implement the "Zero Magic" ML pipeline. It prioritizes readability, explicit dependencies, and native framework integrations.

## Why this package exists

To enable rapid experimentation with different model architectures (XGBoost, GNNs, etc.), we need a common interface that shields the model developer from the underlying data engineering.

## Key Components

- **`assets.py`**: The "Dagster Vocabulary" (FeatureAsset Enum).
- **`model.py`**: Base classes for forecasters (`BaseForecaster`).
- **`features.py`**: Shared feature engineering logic (e.g., cyclical time features).
- **`data.py`**: Shared data splitting and loading logic.
- **`scaling.py`**: Shared normalization and scaling utilities.
