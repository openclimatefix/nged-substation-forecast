# XGBoost Substation Forecaster

This package implements an XGBoost-based model to forecast power flows at NGED primary substations using numerical weather prediction (NWP) forecasts. It implements the `BaseForecaster` protocol defined in `ml_core`.

## Features

- **Unified ML Interface**: Implements the `BaseForecaster` protocol, allowing seamless integration with the Dagster orchestration pipeline.
- **Multi-NWP Support**: Ingests forecasts from multiple NWP providers simultaneously. Secondary NWP features are prefixed with their model name (e.g., `gfs_temperature_2m`), and all NWPs are joined using a 3-hour availability delay.
- **Dynamic Seasonal Lags**: Prevents lookahead bias by calculating autoregressive lags dynamically based on the forecast lead time. The model always uses the most recent *available* historical data for a given lead time (e.g., `lag_days = max(1, ceil(lead_time_days / 7)) * 7`).
- **Rigorous Backtesting**: Supports simulating real-time inference via the `collapse_lead_times` parameter. When enabled, it filters NWP data to keep only the latest available forecast for each valid time, enforcing the 3-hour availability delay.
- **H3-based Weather Matching**: Automatically matches substation coordinates to H3 resolution 5 cells used in the weather data.
- **Ensemble Averaging**: Averages weather variables across ensemble members for robust feature engineering.
- **Temporal Features**: Includes cyclical temporal features (sine/cosine for hour and day of year) and day of week.
- **Long-Range Horizon Handling**: Supports 14-day (336h) forecasts at 30-minute resolution. The `lead_time_hours` is passed as a feature to the XGBoost model, allowing it to learn the decay in NWP skill over time.
- **Physical Wind Logic**: Wind speed and direction are interpolated using Cartesian `u` and `v` components instead of circular interpolation. This avoids "phantom high wind" artifacts during rapid direction shifts and ensures physical correctness.

## Installation

This package is part of the `uv` workspace. Install all dependencies from the root:

```bash
uv sync
```

## Usage

This package is intended to be used as part of the Dagster pipeline. The `XGBoostForecaster` class handles the full lifecycle of the model, including training and inference.
