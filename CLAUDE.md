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

### Feature Engineering (`packages/ml_core/src/ml_core/features/`)

`_engineer_features()` (in `tabular_feature_engineer.py`) is the central tabular pipeline function: given a `set[str]` of requested feature names, it joins power observations with NWP and metadata, then applies features. Feature names are parsed by `ParsedFeatures.from_strings()` (in `_parsed_features.py`) into typed `LagFeature`, `RollingFeature`, `StaticFeature`, `TimeFeature`, or `WeatherFeature` objects. Callers reach this via `FeatureEngineer.engineer()` — see the ML Model Interface section below.

**Critical design invariant — no lookahead bias:** `power_fcst_init_time` (when we make the forecast) is distinct from `nwp_init_time` (when the NWP model ran). Power lag features are nullified via `_nullify_leaky_lags()` when the lag is shorter than or equal to the forecast lead time. Weather lags use a dual-strategy join: same NWP run for future target times, freshest NWP run for past target times.

Two operating modes:
- **Bulk training and multi-run backtesting** (recommended for most callers): `power_fcst_init_time` is `None`; it is derived per-row as `nwp_init_time + nwp_publication_delay_hours`.
- **Single-run inference or backfilling**: `power_fcst_init_time` is provided; NWP is joined on `(time_series_id, valid_time, nwp_init_time)` for the one matching NWP run.

### ML Model Interface (`packages/ml_core/src/ml_core/base_forecaster.py`)

All forecasting models subclass `BaseForecaster`, which defines `train(AllFeatures)`, `predict(AllFeatures) -> PowerForecast`, `save(Path)`, and `load(Path) -> Self`. Each subclass owns its own persistence format; `XGBoostForecaster` writes one `.ubj` file per `time_series_id` plus a `meta.json` with the full `XGBoostConfig`.

Identity is split across two levels. **Model-family identity** — `MODEL_NAME` and `MODEL_VERSION` — are class-level constants on each `BaseForecaster` subclass (properties of the implementation; bumping `MODEL_VERSION` is a deliberate code change). **Experiment identity** — `experiment_name` and `ml_flow_experiment_id` — lives in `BaseForecasterConfig` so it travels with the saved model. Both levels are stamped onto every `PowerForecast` row at predict time: `power_fcst_model_name`/`power_fcst_model_version` from the class, and the dedicated `experiment_name`/`ml_flow_experiment_id` columns from the config. Do not collapse experiment identity into `power_fcst_model_name`.

Each `BaseForecaster` also carries a `feature_engineer: ClassVar[FeatureEngineer]` — a strategy object (composition, not inheritance) that owns the full data-preparation pipeline from raw inputs to an `AllFeatures` frame, including the NWP spatial join. The default `TabularFeatureEngineer` maps each gridded NWP H3 cell to the nearest time series then runs the tabular `_engineer_features` pipeline. A future model needing a different view of the data (e.g. a CNN wanting a spatial NWP crop) overrides `feature_engineer` with a different `FeatureEngineer` subclass — it does not change `_engineer_features` or `BaseForecaster`. Both classes live in `packages/ml_core/src/ml_core/features/`.

## Code Style

- **Python 3.14+** required.
- **Polars only** — pandas is strictly forbidden. Use `pl.LazyFrame` and only `.collect()` when necessary.
- **Patito** for all DataFrame schema definitions and validation. Use Patito type annotations (`pt.DataFrame[Schema]`, `pt.LazyFrame[Schema]`) whenever a function consumes or returns data that conforms to an existing schema — whether the function is public or private. Don't invent a new schema just to annotate a private helper; if no existing schema fits, use plain `pl.DataFrame` / `pl.LazyFrame`.
- **Ruff**: 100-char line length, double quotes, Google-style docstrings.
- **Comments must reflect current state only** — never reference previous iterations of the
  code or deleted files.
- **Never reference temporary implementation plans** (in `plans/`) from code *or* docs — neither
  the files themselves nor their sections (e.g. "§4.3.1"). These plans are deleted once the work
  lands, so any reference to them rots. Linking *between code and the permanent docs under `docs/`*
  is encouraged — e.g. a docstring pointing at a page in `docs/architecture/`.
- **MkDocs-compatible constant docs** — document module-level constants with a string literal
  immediately after the assignment, not with Sphinx-style `#:` comments. This is correct:
  ```python
  MY_CONST: Final[str] = "value"
  """One-line summary.

  Optional further detail.
  """
  ```
- `snake_case` for variables/functions, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- All function signatures must have complete type hints including return types.
- All consts must be marked with the maximally "constant" type.
  e.g. `CONST_SEQ: Final[tuple[str, ...]] = ("a", "b")` or `FOO: Final[str] = "bar"`
- Never relax an existing test to make it pass.

### Polars Style

These rules are all about making Polars code easy to read.

- When casting, prefer using the `cast` method like this: `df.cast({"foo": pl.Int8})`, in favour of
  using `df.with_columns(pl.col("foo").cast(pl.Int8))`. **Caveat:** this is only safe on a plain
  Polars frame — passing a `{column: dtype}` mapping to a *model-bearing* Patito frame silently does
  the wrong thing. See "Patito + Polars Gotcha: `.cast({...})` on a model-bearing frame" below.
- When using `.with_columns`, prefer specifying the destination column name as a key word argument
  like this: `df.with_columns(bar=pl.col("foo").expression())` instead of using `alias` like this:
  `df.with_columns(pl.col("foo").expression().alias("bar"))`

### Patito + Polars Gotcha: cross-model LazyFrame joins

Patito creates a unique Python subclass for each model (e.g. `PowerTimeSeriesLazyFrame`,
`PowerForecastLazyFrame`). Polars' `assert_same_type` check inside `.join()` rejects joining
two differently-typed Patito LazyFrames with a `TypeError`.

Workaround: strip the Patito subclass from the right-hand operand before joining:

```python
# Strip Patito model annotation so Polars' cross-subclass type check doesn't reject the join
plain_lf = pl.LazyFrame._from_pyldf(patito_lf._ldf)
left_patito_lf.join(plain_lf.select(...), on=..., how="inner")
```

`pl.LazyFrame._from_pyldf` constructs a plain `pl.LazyFrame` from the same underlying Rust
object — zero-copy, no data movement. The check passes because `type(left_lf)` is a subclass
of `pl.LazyFrame`, so `isinstance(left_lf, type(plain_lf))` is `True`.

### Patito + Polars Gotcha: `.cast({...})` on a model-bearing frame

Patito **overrides** `.cast`: its signature is `cast(self, strict=False, columns=None)` and, on a
frame that carries a model (set via `.set_model(...)` or a typed `pt.DataFrame[Schema]`), it casts
every column to the *model's* declared dtypes. So `df.cast({"foo": pl.Int8})` on such a frame does
**not** apply your mapping — Polars' `{column: dtype}` dict is swallowed as the `strict` argument
and your `foo` cast is silently ignored while unrelated columns are reverted to model dtypes. The
result usually only surfaces later as a confusing `validate()` dtype error.

The trap fires only when the model is still attached. Many Polars ops **drop** the model
(`group_by(...).agg(...)`, `.collect()`, `.unpivot()`, `.as_polars()`), so a dict-`.cast` after
them is plain Polars and fine. But **iterating** `group_by` (`for k, g in df.group_by(...)`) yields
sub-frames that **keep** the model, and `pl.concat` keeps it too — so a dict-`.cast` on the
concatenated result hits the trap.

Workaround: strip the Patito model before a `{column: dtype}` cast (mirrors the join gotcha above):

```python
# Strip the Patito model so the dict-cast uses plain Polars semantics (zero-copy)
result = pl.DataFrame._from_pydf(patito_df._df).cast({"foo": pl.Categorical})
```

(No-arg `df.cast()` — casting a model-bearing frame to its declared dtypes — *is* the intended
Patito use and is correct. Expression/Series casts like `pl.col("foo").cast(pl.Int8)` are always
plain Polars and unaffected.)

## This is a young project

The project is a new, green-field project. No one else is using this code yet. Which means:

- It's 100% fine to make breaking changes, if doing so improves the code. (And as long as we update
  all the downstream code.)
- Our aim is to make the code well-organised and easy to use.
- None of this code is "written in stone" or battle-tested.
- If you see a design mistake _anywhere_ in the code, then please flag that design mistake to me.
  I'd much rather end up with a project that's well engineered. (That said, if we're working on
  feature X, and you spot a mistake in some code that isn't obviously in scope for X, then please
  discuss the change with me first. Definitely don't make out-of-scope changes with asking me!)
