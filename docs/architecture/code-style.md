# Code Style Guidelines

(This is mostly written for LLM coding agents!)

## General Principles

- **Python Version**: Use Python 3.14+.
- **Type Hints**: All function signatures **must** use expressive type hints for all arguments and return types. Use `typing` and `collections.abc` as needed.
- **Modularity**: Keep logic in small, focused packages under `packages/`. The main app in `src/` should primarily handle orchestration.
- **Small functions**: Prefer small function bodies that do one, well-defined thing.
- **Minimalism**: Re-use existing tools (Polars, Xarray, Dagster) instead of reinventing logic.
- **Comments**: Do not remove existing comments unless they are misleading or out of date. Only add new comments if you're doing something that isn't obvious from the code. Write self-documenting code, and assume the reader is fluent in Python.
- **Tests**: Unit tests should each be a short, simple function. For each function in the main code, there should be at least one test function that tests the "happy path", and one test function for each of the main "unhappy" paths. Never relax an existing test just to get it to pass! See the [Testing](#testing) section below for where tests live, how they are wired, mocking, and the assertion house style.

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

## Testing

Conventions for where tests live, how they are wired up, and the house style for assertions. Two
testing *gotchas* are documented alongside the other gotchas in CLAUDE.md rather than here: the
moto S3 backend being process-global (reset it per test), and Polars row counts wrapping past
2³² rows.

### Where tests and their dependencies live

- **Test tooling is declared once, at the workspace root.** `pytest`, `moto`, and `numpy` live in
  the root `pyproject.toml` `[dependency-groups] dev`, and every workspace package inherits them.
  A package that gains a `tests/` directory does **not** re-declare `pytest` in its own
  `pyproject.toml` — we run the whole suite from the repo root with `uv run pytest`, against the
  root environment. (`packages/geo` declares its own `pytest`/`pytest-cov`; treat that as a
  historical exception, not the pattern to copy.)
- **Discovery is automatic.** The only pytest configuration is the root
  `[tool.pytest.ini_options]` block; there is no `testpaths` setting, so pytest collects both the
  top-level `tests/` directory and every `packages/*/tests/` directory. A brand-new
  `packages/<pkg>/tests/` directory is picked up with no configuration change.
- **`--import-mode=importlib` is set deliberately** so that identically-named test modules in
  different packages (for example, two `test_storage.py` files) do not collide during collection.
  Because of this, test directories do not need `__init__.py` files.
- **Test data files go in a `tests/data/` subdirectory** and are loaded relative to the test
  module with `Path(__file__).parent / "data" / filename`. `packages/nged_data/tests/` is the
  canonical example (it also keeps a small script documenting how the fixtures were trimmed down).

### Fixtures and mocking

- **Define fixtures inline in the test module by default.** When a fixture — or a fixture factory
  — is shared across more than one test module *within a single package*, put it in a
  package-level `tests/conftest.py`. `packages/dynamical_data/tests/conftest.py` is the example:
  it builds synthetic Xarray datasets that two test modules share. Keep such a conftest scoped to
  its own package; there is no repo-wide conftest.
- **Mock with pytest's `monkeypatch` fixture, not `unittest.mock`.** Patch environment variables
  (`monkeypatch.setenv`), object attributes, and module-level functions
  (`monkeypatch.setattr(some_module, "open", fake_open)`) through the built-in fixture. For S3,
  drive the in-process `moto` server instead of mocking — `tests/test_s3_data_paths.py` is the
  canonical pattern.

### Network-gated tests

Most tests run fully offline, mocking any network call (for example, patching
`dynamical_catalog.open` to return a synthetic `xr.Dataset`). A handful of tests are worth running
against a **real** external service — chiefly to catch the *shared-convention blind spot*, where a
synthetic fixture and the code under test share the same wrong assumption about the real data's
shape (dimension order, latitude orientation, longitude range, dtypes, units) and both pass.

Mark such a test `@pytest.mark.network`. The root `conftest.py` skips every `network`-marked test
unless the caller passes `--run-network`, so a plain `uv run pytest` (local dev and the per-PR CI)
never touches the network. Run them explicitly — nightly CI or on demand — with:

```bash
uv run pytest --run-network              # whole suite, network tests included
uv run pytest --run-network -m network   # only the network tests
```

`packages/dynamical_data/tests/test_ecmwf_ens_network.py` is the canonical example: it drives the
real `open → download → convert` pipeline against the Dynamical.org ECMWF ENS catalog and asserts
the conventions the offline fixtures merely assume.

The gate is a collection hook, **not** an `addopts = "... -m 'not network'"`. pytest keeps only the
*last* `-m` it is given, so any caller-supplied marker expression (e.g. `-m "not integration"`) would
silently replace an `addopts` `-m "not network"` and re-include the network tests. A skip applied
during collection cannot be overridden that way — the gate holds whatever `-m` the caller passes, and
even `-m network` alone stays skipped until `--run-network` is added.

### NWP grid → H3 orientation coverage

The NWP-grid-to-H3 mapping is the classic place for a silent orientation bug — a vertically or
horizontally flipped weather grid, a transpose (`np.meshgrid` `indexing="ij"` vs `"xy"`), or a
lat/lon swap. Three tests guard it in layers, from cheap-and-synthetic to real-and-networked. Each
was checked by *mutation*: introducing the bug into the production code and confirming the test goes
red. The table records which mutation each layer catches (✓ = the test fails when that bug is
present):

| Mutation in production code | synthetic `convert` test¹ | cached real-slice test² | geo landmark test³ |
| --- | :---: | :---: | :---: |
| `np.meshgrid` `indexing="ij"` → `"xy"` (transpose) | ✓ | ✓ | n/a |
| lat/lon swap in the value-join keys | ✓ | ✓ | ✓ |
| reversed latitude ravel (vertical flip) | ✓ | ✓ | n/a |
| swap `cell_to_lat`/`cell_to_lng` when snapping | n/a | n/a | ✓ |

- ¹ `dynamical_data/tests/test_convert_to_polars.py::test_convert_maps_each_grid_point_to_its_own_lat_lon`
  — a 2×2 synthetic grid with a distinct value at every corner. Guards the ravel-alignment step
  *inside* `convert`; the value-join itself is positional-agnostic, so which hexagon owns a given
  (lat, lon) is delegated to the upstream `h3_grid_weights` asset (the geo test below).
- ² `dynamical_data/tests/test_ecmwf_ens_cached.py::test_cached_real_slice_conventions_and_orientation`
  — the same orientation check on a committed real ECMWF ENS slice, so the conventions the synthetic
  fixture only *assumes* (descending latitude, °C, dimension order) are exercised on genuine bytes.
- ³ `geo/tests/test_h3.py::test_grid_weights_preserve_geographic_orientation` — proves
  `compute_h3_grid_weights` labels each H3 cell with grid points at the cell's own (lat, lon), using
  two well-separated GB landmarks. This is what fixes the hexagon↔(lat, lon) geography the `convert`
  tests delegate.

`test_ecmwf_ens_network.py` (network-gated, above) composes all three against the live catalog and is
the only layer that can catch *future upstream drift* — a change in Dynamical.org's own conventions
that the committed slice, frozen at capture time, cannot.

### Assertion style for Patito frames

Build a frame, attach the model, cast, and validate for the happy path:

```python
df = pt.DataFrame({...}).set_model(MySchema).cast()
df.validate()                       # happy path: raises nothing
MySchema.validate(existing_df)      # or validate a frame produced elsewhere
```

For the unhappy path, assert that validation raises:

```python
from patito.exceptions import DataFrameValidationError

with pytest.raises(DataFrameValidationError):
    bad_df.cast().validate()
```

`packages/contracts/tests/test_geo_schemas.py` is the shortest end-to-end example.
