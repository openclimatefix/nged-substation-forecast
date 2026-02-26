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

### Running the Demo

The demo script trains models for 5 random substations and prints sanity checks and performance metrics:

```bash
uv run python packages/xgboost_forecaster/examples/train_demo.py
```

### Programmatic Usage

```python
from xgboost_forecaster.data import get_substation_metadata, prepare_training_data
from xgboost_forecaster.model import train_model

# 1. Get metadata
metadata = get_substation_metadata()

# 2. Prepare data for a substation
data = prepare_training_data("Hillmorton 33 11kv S Stn", metadata)

# 3. Train model
model, metrics = train_model(data)
```

## Data Sources

- **Power Data**: Expected in `data/NGED/parquet/live_primary_flows/`.
- **Weather Data**: Expected in `packages/dynamical_data/data/` as H3-indexed Parquet files.
