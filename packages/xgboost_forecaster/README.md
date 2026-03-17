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

This package is intended to be used as part of the Dagster pipeline.
