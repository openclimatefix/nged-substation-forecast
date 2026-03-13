# XGBoost Substation Forecaster

This package implements a simple XGBoost-based model to forecast power flows at NGED primary substations using ECMWF weather forecasts.

## Features

- **H3-based Weather Matching**: Automatically matches substation coordinates to H3 resolution 5 cells used in the weather data.
- **Ensemble Averaging**: Averages weather variables across ensemble members for robust feature engineering.
- **Automated Metadata Matching**: Joins CKAN substation locations with local Parquet flow files.
- **Temporal Features**: Includes hour of day, day of week, and month as features.

## Installation

This package is part of the `uv` workspace. Install all dependencies from the root:

```bash
uv sync
```

## Usage

### Generating Plots

You can also generate a plot of the forecast vs actual ground truth for a specific substation:

```bash
uv run python packages/xgboost_forecaster/examples/plot_forecast.py
```

This will save a PNG file (e.g., `forecast_Lawford_33_11kv_S_Stn.png`) in the current directory.

### Programmatic Usage

```python
from datetime import date
from xgboost_forecaster import get_substation_metadata, prepare_training_data, XGBoostForecaster, EnsembleSelection

# 1. Get metadata
metadata = get_substation_metadata()

# 2. Prepare data for a substation
data = prepare_training_data(
    substation_numbers=[12345], 
    metadata=metadata,
    start_date=date(2026, 2, 1),
    end_date=date(2026, 2, 28),
    selection=EnsembleSelection.MEAN
)

# 3. Train model
forecaster = XGBoostForecaster()
forecaster.train(data)
```

## Data Sources

- **Power Data**: Expected in `data/NGED/parquet/live_primary_flows/`.
- **Weather Data**: Expected in `packages/dynamical_data/data/` as H3-indexed Parquet files.
