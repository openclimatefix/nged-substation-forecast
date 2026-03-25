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

- **`packages/contracts`**: Lightweight package defining data schemas (using Patito/Polars) and project settings. This package has minimal dependencies to ensure it can be used by any component without bringing in heavy ML libraries.
- **`packages/ml_core`**: Unified ML model interface and shared utilities. This package contains the base classes for trainers and models, and shared logic like feature engineering and data splitting. It depends on `mlflow-skinny`.
- **`packages/xgboost_forecaster`**: Implementation of the substation forecast using XGBoost, following the `ml_core` interface.
- **`packages/nged_data`**: Data ingestion and processing for NGED datasets.
- **`packages/dynamical_data`**: Handling of NWP and other time-varying datasets.

### Dependency Isolation

We maintain a strict separation between `contracts` and `ml_core`. `contracts` defines the *shape* of the data, while `ml_core` defines the *machinery* for ML. By keeping them separate, we ensure that a component that only needs to validate a schema (like a data ingestion script) doesn't need to install heavy ML dependencies like MLflow.

1. Ensure [`uv`](https://docs.astral.sh/uv/) is installed following their [official documentation](https://docs.astral.sh/uv/getting-started/installation/).
1. `uv sync --all-packages` (The `--all-packages` flag ensures that dependencies for all workspace members, like the dashboard and notebooks, are installed into the shared virtual environment, which is necessary for IDE support and type checking).
1. `uv run pre-commit install`

To run linting and type checking:
1. `uv run ruff check .`
2. `uv run --all-packages ty check`

To run tests:
1. `uv run --all-packages pytest`

To run Dagster:
1. `uv run dg dev`
1. Open http://localhost:3000 in your browser to see the project.

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
