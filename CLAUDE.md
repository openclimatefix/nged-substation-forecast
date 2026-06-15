# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Linting & formatting
uv run ruff check .            # check
uv run ruff check . --fix      # fix
uv run ruff format .           # format
uv run ty check                # type checking

# Testing
uv run pytest                                # all tests
uv run pytest path/to/test_foo.py::test_bar  # single test

# Run Dagster UI
uv run dg dev                  # open http://localhost:3000

# Marimo notebooks
uv run marimo edit packages/notebooks/some_notebook.py
```

## Architecture

This is a `uv` workspace monorepo. The root `src/nged_substation_forecast/` is the Dagster application; all reusable logic lives in `packages/`.

### Packages

| Package | Purpose |
|---|---|
| `contracts` | Patito data schemas (the single source of truth for all data shapes) |
| `ml_core` | Feature engineering and `BaseForecaster` abstract class |
| `nged_data` | Reading NGED JSON files from S3 and writing to Delta Lake |
| `dynamical_data` | Downloading ECMWF ensemble NWP from Dynamical.org |
| `geo` | H3 spatial indexing utilities |
| `xgboost_forecaster` | Concrete `BaseForecaster` implementation using XGBoost |
| `dashboard` | Marimo web app for visualisation |
| `notebooks` | Marimo exploration notebooks |

### Dagster Assets (`src/nged_substation_forecast/defs/assets.py`)

Three main assets:
- `power_time_series_and_metadata` — pulls NGED telemetry from S3, appends to Delta Lake, upserts metadata parquet
- `h3_grid_weights` — computes fractional H3 cell overlap with the GB boundary for spatial NWP aggregation
- `ecmwf_ens` — daily-partitioned asset that downloads ECMWF ENS NWP, scales to `Int16`, writes to Delta Lake

### Data Contracts (`packages/contracts/`)

All tabular data flowing through the system is validated with **Patito** models. Key schemas:

- `PowerTimeSeries` — half-hourly power observations (MW/MVA) per `time_series_id`
- `TimeSeriesMetadata` — substation metadata including lat/lon, H3 index, substation type
- `NwpInMemory` / `NwpOnDisk` — NWP weather data. Stored on disk as `Int16` (quantised to 12-bit range per `NwpScalingParams`) and converted back to `Float32` physical units in memory
- `AllFeatures` — the final joined dataset handed to ML models; primary key is `(time_series_id, power_fcst_init_time, valid_time[, ensemble_member])`
- `PowerForecast` — model output schema

### Feature Engineering (`packages/ml_core/src/ml_core/features.py`)

`engineer_features()` is the central function: given a `set[str]` of requested feature names, it joins power observations with NWP and metadata, then applies features. Feature names are parsed by `ParsedFeatures.from_strings()` into typed `LagFeature`, `RollingFeature`, `StaticFeature`, `TimeFeature`, or `WeatherFeature` objects.

**Critical design invariant — no lookahead bias:** `power_fcst_init_time` (when we make the forecast) is distinct from `nwp_init_time` (when the NWP model ran). Power lag features are nullified via `nullify_leaky_lags()` when the lag is shorter than or equal to the forecast lead time. Weather lags use a dual-strategy join: same NWP run for future target times, freshest NWP run for past target times.

Two operating modes:
- **Bulk training and multi-run backtesting** (recommended for most callers): `power_fcst_init_time` is `None`; it is derived per-row as `nwp_init_time + nwp_publication_delay_hours`.
- **Single-run inference or backfilling**: `power_fcst_init_time` is provided; NWP is joined on `(time_series_id, valid_time, nwp_init_time)` for the one matching NWP run.

### ML Model Interface (`packages/ml_core/src/ml_core/base_forecaster.py`)

All forecasting models subclass `BaseForecaster`, which defines `train(AllFeatures)` and `predict(AllFeatures) -> PowerForecast`. Models are saved/loaded using native MLflow flavors and must encapsulate all input/output translation logic.

## Code Style

- **Python 3.14+** required.
- **Polars only** — pandas is strictly forbidden. Use `pl.LazyFrame` and only `.collect()` when necessary.
- **Patito** for all DataFrame schema definitions and validation. Use Patito type annotations in **public** function signatures only — private helpers (`_foo`) use plain `pl.DataFrame` / `pl.LazyFrame`.
- **Ruff**: 100-char line length, double quotes, Google-style docstrings.
- `snake_case` for variables/functions, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- All function signatures must have complete type hints including return types.
- Never relax an existing test to make it pass.
- `docs/v2_design/` is excluded from ruff and ty checks.
