# NGED substation forecast

[![ease of contribution: hard](https://img.shields.io/badge/ease%20of%20contribution:%20hard-bb2629)](https://github.com/openclimatefix#how-easy-is-it-to-get-involved)

To external contributors: Please note that this repo holds the very-early-stage research code for a new
project, and there will be a lot of code churn over the next few months. As such, this repo isn't
suitable for external contributions at the moment, sorry.

---

TODO(Jack): Adapt the OCF template README for this project :)

## Development

This repo is a `uv` [workspace](https://docs.astral.sh/uv/concepts/projects/workspaces): A single
repo which contains multiple Python packages.

### Sub-packages

- **`packages/contracts`**: Lightweight package defining strict data schemas (using Patito/Polars) and project settings. This package has minimal dependencies to ensure it can be used by any component without bringing in heavy ML libraries. Key contracts include `SubstationFeatures` and `PowerForecast`.
- **`packages/ml_core`**: Unified ML model interface and shared utilities. This package contains the `BaseForecaster` protocol, which standardizes model training and inference, and shared logic like feature engineering and data splitting. It depends on `mlflow-skinny`.
- **`packages/xgboost_forecaster`**: Implementation of the substation forecast using XGBoost, following the `BaseForecaster` interface. It includes advanced features like multi-NWP support, dynamic seasonal lags to prevent lookahead bias, and rigorous backtesting capabilities.
- **`packages/nged_data`**: Data ingestion and processing for NGED datasets.
- **`packages/geo`**: Generic geospatial utilities, including H3 grid mapping and spatial operations.
- **`packages/dynamical_data`**: Handling of NWP and other time-varying datasets.

### Dependency Isolation

We maintain a strict separation between `contracts` and `ml_core`. `contracts` defines the *shape* of the data, while `ml_core` defines the *machinery* for ML. By keeping them separate, we ensure that a component that only needs to validate a schema (like a data ingestion script) doesn't need to install heavy ML dependencies like MLflow.

### Unified ML Model Interface

The project uses a unified `BaseForecaster` protocol (defined in `ml_core.model`) to standardize how machine learning models are trained and evaluated. This allows the orchestration system (Dagster) to interact uniformly with any model type (e.g., `XGBoostForecaster`), making it easy to swap implementations or add new model architectures (like PyTorch GNNs) without changing the orchestration logic.


### ML Core Utilities

The `ml_core` package provides shared utilities for training and evaluating models:
- **`train_and_log_model`**: Handles temporal slicing of data based on the `TrainingConfig` and logs the trained model, parameters, and metrics to MLflow.
- **`evaluate_model`**: Handles temporal slicing of data for inference, generates predictions using the `BaseForecaster.predict` method, and calculates evaluation metrics (e.g., MAE, RMSE) which are logged to MLflow.


### Data Engineering & Quality

To ensure robust and reliable machine learning pipelines, we employ several data engineering patterns:
- **Centralized Data Preparation**: All data entering the ML models goes through a centralized `prepare_data` step (e.g., in `ml_core.data` or model-specific data modules). This ensures consistent handling of missing substations, strict type enforcement, and uniform target filtering across both training and inference.
- **Strategy Pattern for CSV Parsing**: The ingestion of various NGED datasets (power data, metadata, switching events) uses a Strategy pattern. This allows the pipeline to easily adapt to different CSV formats and schemas without modifying the core ingestion logic, adhering to the Open/Closed Principle.

### Advanced Forecasting Features

The forecasting models implement several advanced features to ensure robustness and accuracy:

1. **Multi-NWP Support**: Models can ingest forecasts from multiple Numerical Weather Prediction (NWP) providers simultaneously. Secondary NWP features are prefixed with their model name (e.g., `gfs_temperature_2m`), and all NWPs are joined using a 3-hour availability delay to simulate real-world data availability.
2. **Dynamic Seasonal Lags**: To strictly prevent lookahead bias, autoregressive lags are calculated dynamically based on the forecast lead time. The model always uses the most recent *available* historical data for a given lead time (e.g., `lag_days = max(1, ceil(lead_time_days / 7)) * 7`).
3. **Rigorous Backtesting**: The `predict` method includes a `collapse_lead_times` parameter. When simulating real-time inference, it filters NWP data to keep only the latest available forecast for each valid time, enforcing the 3-hour availability delay. For rigorous backtesting, it evaluates all available lead times up to the cutoff.
4. **Physical Wind Logic**: Wind speed and direction are interpolated using Cartesian `u` and `v` components instead of circular interpolation. This avoids "phantom high wind" artifacts during rapid direction shifts and ensures physical correctness.
5. **Long-Range Horizon Handling**: The model supports 14-day (336h) forecasts at 30-minute resolution. The `lead_time_hours` is passed as a feature to the XGBoost model, allowing it to learn the decay in NWP skill over time.

### Power Forecast Storage

Power forecasts generated by the ML models are stored in a single Delta Lake table located at `data/evaluation_results.delta`. To ensure high performance and avoid the "small files problem", the table is partitioned by:
1.  **`power_fcst_model_name`**: The name of the ML model (e.g., `xgboost`).
2.  **`power_fcst_init_year_month`**: The year and month the forecast was generated (e.g., `2026-03`).

This partitioning strategy allows for efficient querying of specific model runs while keeping the underlying Parquet files large and optimized.


### Manual Integration Test

To verify the XGBoost forecaster and the ML core utilities without running the full Dagster pipeline, you can run the manual integration test. This script trains and evaluates the model on a small subset of 5 substations:

```bash
uv run pytest tests/test_xgboost_dagster_integration.py -v -m manual
```

This test executes the actual Dagster assets in-process using an in-memory I/O manager, ensuring the production pipeline logic is fully tested without requiring a heavy infrastructure setup.

### Downstream Analysis

The `metrics` and `plot` assets are model-agnostic and partitioned by `model_partitions`. They use an `AutoMaterializePolicy.eager()` to automatically run whenever a new model partition is added to the system (e.g., when an `evaluate_*` asset finishes and calls `context.instance.add_dynamic_partitions`).

## 🛠 Build, Lint, and Test Commands

This project uses `uv` for dependency management and task execution.

### Modernization & Testing Practices

Recent improvements to the pipeline have modernized our dependencies and testing practices:
- **Python 3.14+**: Leveraging the latest Python features.
- **Polars & Patito**: Using Polars for fast, memory-efficient data processing and Patito for strict data contracts and schema validation.
- **Testing with Pytest & Hypothesis**: Comprehensive testing using `pytest` (v9+) and property-based testing with `hypothesis`.
- **Linting & Formatting**: Enforcing code quality with `ruff` and strict type checking with `ty`.
- **Dagster Orchestration**: Managing the ML pipeline and data ingestion with Dagster.

### Setup

1. Ensure [`uv`](https://docs.astral.sh/uv/) is installed following their [official documentation](https://docs.astral.sh/uv/getting-started/installation/).
2. **Install dependencies**: `uv sync`
3. **Install pre-commit hooks**: `uv run pre-commit install`

### Linting & Formatting

- **Check linting**: `uv run ruff check .`
- **Fix linting**: `uv run ruff check . --fix`
- **Format code**: `uv run ruff format .`
- **Type checking**: `uv run ty check`

### Testing

- **Run all tests**: `uv run pytest`
- **Run a single test file**: `uv run pytest tests/test_placeholder.py`
- **Run a single test function**: `uv run pytest tests/test_placeholder.py::test_placeholder`
- **Run tests with coverage**: `uv run pytest --cov`

### Development

- **Run Dagster UI**: `uv run dagster dev`
- Open http://localhost:3000 in your browser to see the project.
- **Run Marimo notebooks**: `uv run marimo edit packages/notebooks/some_notebook.py`

Optional: To allow Dagster to remember its state after you shut it down:
1. `mkdir ~/dagster_home/`
2. Put the following into `~/dagster_home/dagster.yaml`:
    ```yaml
    storage:
      sqlite:
        base_dir: "dagster_history"

    # Limit concurrency for heavy processing steps like downloading ECMWF ENS:
    concurrency:
      pools:
        default_limit: 1
    ```
3. Add `export DAGSTER_HOME=<dagster_home_path>` to your `.bashrc` file, and restart your terminal.

## Environment variables

This code uses `pydantic-settings` to manage configuration. You can set these variables in your environment or in a `.env` file.

For a full list of available settings, their types, and their defaults, please see the `Settings` class in:
`packages/contracts/src/contracts/config.py`

### NGED CKAN API token:
1. Log in to [NGED's Connected Data](https://connecteddata.nationalgrid.co.uk) platform.
1. Click on your username (top right), and go to "User Profile" -> API Tokens -> Create API token
   -> Copy your API token. If you need more help then see [NGED's docs for getting an API
   token](https://connecteddata.nationalgrid.co.uk/api-guidance#api-tokens).
1. Paste your API token into `.env` after `NGED_CKAN_TOKEN=`.

---

*Part of the [Open Climate Fix](https://github.com/orgs/openclimatefix/people) community.*

[![OCF Logo](https://cdn.prod.website-files.com/62d92550f6774db58d441cca/6324a2038936ecda71599a8b_OCF_Logo_black_trans.png)](https://openclimatefix.org)
