# NGED substation forecast

[![ease of contribution: hard](https://img.shields.io/badge/ease%20of%20contribution:%20hard-bb2629)](https://github.com/openclimatefix#how-easy-is-it-to-get-involved)

To external contributors: Please note that this repo holds the very-early-stage research code for a new project, and there will be a lot of code churn over the next few months. As such, this repo isn't suitable for external contributions at the moment, sorry.

---

## Documentation

For detailed information about the project, including architecture, design philosophy, and user guides, please visit our [documentation site](https://openclimatefix.github.io/nged-substation-forecast/).

## Development

This repo is a `uv` [workspace](https://docs.astral.sh/uv/concepts/projects/workspaces): A single repo which contains multiple Python packages.

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

- **Run only the fast tests**: `uv run pytest -m "not slow"`
- **Run only the slow tests**: `uv run pytest -m "slow"`
- **Run all tests**: `uv run pytest`
- **Run tests with coverage**: `uv run pytest --cov`
- **View generated plots**: To view the generated HTML plots (e.g., `tests/xgboost_dagster_integration_plot.html`), start a local web server in the project root: `python -m http.server 8000`, then open `http://localhost:8000/tests/xgboost_dagster_integration_plot.html` in your browser.

### Development

- **Run Dagster UI**: `uv run dagster dev`
- Open http://localhost:3000 in your browser to see the project.
- **Run Marimo notebooks**: `uv run marimo edit packages/notebooks/some_notebook.py`

---

*Part of the [Open Climate Fix](https://github.com/orgs/openclimatefix/people) community.*

[![OCF Logo](https://cdn.prod.website-files.com/62d92550f6774db58d441cca/6324a2038936ecda71599a8b_OCF_Logo_black_trans.png)](https://openclimatefix.org)
