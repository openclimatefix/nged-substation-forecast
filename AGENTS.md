# Agent Guide: NGED Substation Forecast

This repository contains the research and production code for forecasting net demand at National Grid Electricity Distribution (NGED) substations, disaggregating it into gross demand, solar, and wind components.

## 🛠 Build, Lint, and Test Commands

This project uses `uv` for dependency management and task execution.

### Setup
- **Install dependencies**: `uv sync`
- **Install pre-commit hooks**: `uv run pre-commit install`

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
- **Run Marimo notebooks**: `uv run marimo edit packages/notebooks/some_notebook.py`

---

## 🎨 Code Style Guidelines

### General Principles
- **Python Version**: Use Python 3.14+.
- **Type Hints**: All function signatures **must** use expressive type hints for all arguments and return types. Use `typing` and `collections.abc` as needed.
- **Modularity**: Keep logic in small, focused packages under `packages/`. The main app in `src/` should primarily handle orchestration.
- **Small functions**: Prefer small function bodies that do one, well-defined thing.
- **Minimalism**: Re-use existing tools (Polars, Xarray, Dagster) instead of reinventing logic.
- **Comments**: Do not remove existing comments unless they are misleading or out of date. Only add
new comments if you're doing something that isn't obvious from the code. Write self-documenting
code, and assume the reader is fluent in Python. I repeat: never delete existing comments unless you
are 100% certain they are wrong! This is the law!
- **Tests**: Unit tests should each be a short, simple function. For each function in the main code,
  there should be at least one test function that tests the "happy path", and one test function for
  each of the main "unhappy" paths. Never relax an existing test just to get it to pass!

### Formatting & Linting (Ruff)
- **Line Length**: 100 characters.
- **Quotes**: Use **double quotes** (`"`) for strings.
- **Docstrings**: Use **Google convention** for docstrings.
- **Imports**: Sorted automatically by `ruff` (isort rules).
- **Naming**:
    - Variables/Functions: `snake_case`
    - Classes: `PascalCase`
    - Constants: `UPPER_SNAKE_CASE`

### Data Handling
- **Tabular Data**: Use **Polars** (`import polars as pl`) for dataframes. Avoid Pandas unless strictly necessary for library compatibility.
- **Gridded/NWP Data**: Use **Xarray** and **Zarr**.
- **Data Contracts**: Use **Patito** for defining and validating data schemas.
- **Persistence**: Prefer partitioned Parquet files for tabular data.

### Machine Learning (PyTorch)
- Use **PyTorch** for differentiable physics models.
- Use **PyTorch Geometric** for GNN implementations.
- Use **MLFlow** for tracking experiments.
- Follow the "test-harness" pattern: separate research logic from production orchestration but ensure they use the same data contracts.

### Error Handling
- Use specific exceptions.
- Leverage Sentry for observability in production-like code.
- Validate data at boundaries using data contracts.

---

## 📂 Repository Structure

- `packages/`: Modular, pip-installable components (contracts, data loading, models, etc.).
- `src/nged_substation_forecast/`: Main Dagster application and orchestration logic.
- `tests/`: Integration tests for the main application. Each package in `packages/` should have its own `tests/` directory.
- `pyproject.toml`: Root configuration for the `uv` workspace and dev tools.

## 🤖 AI / Agent Specific Instructions

- Always run `uv run ruff check . --fix` and `uv run ruff format .` before submitting changes.
- Ensure `uv run ty check` passes for any new code.
- When adding new functionality, consider if it belongs in a new or existing package within `packages/`.
- Refer to `DESIGN.md` for the long-term architectural vision and ultimate aims of the project.
- Refer to `README.md` and `DESIGN.md` files within each child `packages/*` for package-specific
  guidance.
