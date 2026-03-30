# `ml_core`

Unified ML model interface and shared utilities for substation forecasting.

## Purpose

This package provides the abstract base classes and shared logic required to implement the "Tiny Wrapper" ML pipeline. It prioritizes readability, explicit dependencies, and native framework integrations.

## Why this package exists

To enable rapid experimentation with different model architectures (XGBoost, GNNs, etc.), we need a common interface that shields the model developer from the underlying data engineering.

## Key Components

- **`model.py`**: Base classes for forecasters (`BaseForecaster`). The `BaseForecaster` protocol defines a standard interface for all forecasting models, requiring them to implement `train()` and `predict()` methods. This allows Dagster to orchestrate any model uniformly.
- **`utils.py`**: Shared MLOps utilities for training (`train_and_log_model`) and evaluation (`evaluate_and_save_model`).
- **`features.py`**: Shared feature engineering logic (e.g., cyclical time features).
- **`data.py`**: Shared data splitting and loading logic.
- **`scaling.py`**: Shared normalization and scaling utilities.

## Advanced Forecasting Features

The `BaseForecaster` protocol supports several advanced forecasting features:

1. **Multi-NWP Support**: Models can ingest forecasts from multiple Numerical Weather Prediction (NWP) providers simultaneously. Secondary NWP features are prefixed with their model name (e.g., `gfs_temperature_2m`), and all NWPs are joined using a 3-hour availability delay.
2. **Dynamic Seasonal Lags**: Prevents lookahead bias by calculating autoregressive lags dynamically based on the forecast lead time. The model always uses the most recent *available* historical data for a given lead time (e.g., `lag_days = max(1, ceil(lead_time_days / 7)) * 7`).
3. **Rigorous Backtesting**: Supports simulating real-time inference via the `collapse_lead_times` parameter. When enabled, it filters NWP data to keep only the latest available forecast for each valid time, enforcing the 3-hour availability delay.
