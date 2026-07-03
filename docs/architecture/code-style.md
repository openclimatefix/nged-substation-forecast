# Code Style Guidelines

(This is mostly written for LLM coding agents!)

## General Principles

- **Python Version**: Use Python 3.14+.
- **Type Hints**: All function signatures **must** use expressive type hints for all arguments and return types. Use `typing` and `collections.abc` as needed.
- **Modularity**: Keep logic in small, focused packages under `packages/`. The main app in `src/` should primarily handle orchestration.
- **Small functions**: Prefer small function bodies that do one, well-defined thing.
- **Minimalism**: Re-use existing tools (Polars, Xarray, Dagster) instead of reinventing logic.
- **Comments**: Do not remove existing comments unless they are misleading or out of date. Only add new comments if you're doing something that isn't obvious from the code. Write self-documenting code, and assume the reader is fluent in Python.
- **Tests**: Unit tests should each be a short, simple function. For each function in the main code, there should be at least one test function that tests the "happy path", and one test function for each of the main "unhappy" paths. Never relax an existing test just to get it to pass!

## Formatting & Linting (Ruff)

- **Line Length**: 100 characters.
- **Quotes**: Use **double quotes** (`"`) for strings.
- **Docstrings**: Use **Google convention** for docstrings.
- **Imports**: Sorted automatically by `ruff` (isort rules).
- **Naming**:
  - Variables/Functions: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`

## Data Handling

- **Tabular Data**: Use **Polars** (`import polars as pl`) for dataframes. Pandas is strictly forbidden. Use Polars for all tabular data.
- **Lazy evaluation**: Use `pl.LazyFrame` throughout the pipeline. **Do not call `.collect()` before the model boundary.** See [Lazy Evaluation Strategy](overview.md#lazy-evaluation-strategy) for the full contract.
- **Gridded/NWP Data**: Use **Xarray** and **Zarr**.
- **Data Contracts**: Use **Patito** for defining and validating data schemas. Use Patito type annotations (`pt.DataFrame[MySchema]`, `pt.LazyFrame[MySchema]`) whenever a function consumes or returns data that conforms to an existing schema — whether the function is public or private. Don't invent a new schema just to annotate a private helper; if no existing schema fits, use plain `pl.DataFrame` / `pl.LazyFrame`.
- **Patito friction budget**: CLAUDE.md documents four Patito-vs-Polars gotchas (cross-model LazyFrame joins, dict-`.cast` on model-bearing frames, `.filter()` dropping the Patito subclass, and Delta dictionary-encoded columns). Four workarounds is an acceptable price for schema validation — but if a fifth becomes necessary, revisit the approach: either validate only at I/O boundaries (typed annotations everywhere, `.validate()` only at persistence edges) or evaluate an alternative such as `dataframely`.
- **Persistence**: Prefer partitioned Parquet files for tabular data.

## Machine Learning (PyTorch)

- Use **PyTorch** for differentiable physics models.
- Use **MLFlow** for tracking experiments.
- Follow the "test-harness" pattern: separate research logic from production orchestration but ensure they use the same data contracts.

## Error Handling

- Use specific exceptions.
- Leverage Sentry for observability in production-like code.
- Validate data at boundaries using data contracts.
