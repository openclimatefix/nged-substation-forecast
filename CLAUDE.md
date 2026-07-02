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
uv run pymarkdown scan -r docs README.md CLAUDE.md metadata/README.md packages/*/README.md  # markdown lint

# Testing
uv run pytest                                # all tests
uv run pytest path/to/test_foo.py::test_bar  # single test

# Run Dagster UI
uv run dg dev                  # open http://localhost:3000

# Marimo notebooks
uv run marimo edit packages/notebooks/some_notebook.py
```

Markdown (README.md files, docs/*.md, and Python docstrings) is linted automatically by the
pre-commit hook, but when developing code or docs it's a good idea to run the markdown lint
command above yourself before committing, for faster feedback than waiting on the commit-time
hook.

## Docs

`docs/` contains a lot of useful information beyond API reference: forward-looking plans and
their ordering (`docs/roadmap/`), durable explainers of solution methods (`docs/techniques/`),
background/requirements context (`docs/background/`), and architecture notes
(`docs/architecture/`). When planning new features, check `docs/` for relevant prior discussion
before proposing an approach.

The docs are published at <https://openclimatefix.github.io/nged-substation-forecast>. When
linking to a docs page from anywhere outside `docs/` itself (GitHub issues, PR bodies, code
docstrings), link to that rendered site (e.g.
`https://openclimatefix.github.io/nged-substation-forecast/roadmap/live-service/#anchor`),
never to a `github.com/.../blob/main/docs/...` path.

## How planning works

Full description and a "which place do I use?" table: `docs/roadmap/index.md`. In brief:

- **GitHub** (issues + the OCF Project board) is the *complete, ordered* task list — task-level
  priority lives only there. When current priorities matter, query it with `gh` (epics map 1:1
  to roadmap milestones; dependencies are `blocked by` issue links).
- **`docs/roadmap/`** holds design, dependencies, and the milestone arc. Step-by-step mechanics
  sit inside each page under an "Implementation details (deleted when this ships)" section.
- **`plans/`** holds at most one file: the in-flight PR's mechanical checklist, deleted on
  merge. Usually empty.

**Creating GitHub issues** — whenever you create an issue, also set:

- **Labels** and **Type** (org issue type: Task / Bug / Feature / Spike / Epic / …) — pick
  whatever fits the issue.
- Add it to the **OCF project** (org project 33, `gh project item-add 33 --owner
  openclimatefix --url <issue-url>`) and set the project fields **Status = Todo**,
  **Project = NGED**, **Area = ML**.
- If it is a sub-issue, attach it to its parent epic **and position it appropriately in the
  parent's sub-issue order** (execution order, respecting `blocked by` chains) — the
  `reprioritizeSubIssue` GraphQL mutation with `afterId`/`beforeId`.
- When the body links to a docs page, link to the **rendered site**
  (`https://openclimatefix.github.io/nged-substation-forecast/...`), never a `github.com`
  blob path.

`gh issue create` can't set any of these: use `gh issue edit --add-label` for labels, the
`updateIssueIssueType` GraphQL mutation for Type, and `gh project item-edit` (or the
`updateProjectV2ItemFieldValue` mutation) for the project fields.

**Ship-time triage** — when a PR lands a roadmap item, that PR (or an immediate follow-up)
must also:

1. Promote surviving design decisions to their permanent home (`docs/architecture/`,
   `docs/ml_experimentation/`, …).
2. Delete the item's "Implementation details" section (and any `plans/` file), pasting it (or
   a summary) into the PR body. When a roadmap page's last 🚧 item ships, delete the page
   (nav entry, inbound doc links).
3. Close the GitHub issue; update the status banner on the roadmap page (and the milestone
   section in `docs/roadmap/index.md` if the arc changed).

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
- **Prefer small functions.** Extract private helpers (`_name`) rather than letting a function body grow long, even if that means more parameters. A well-named helper with a clear docstring beats a long inline block. Eight parameters is acceptable when each is distinct and the division of labour is clear.
- **Ruff**: 100-char line length, double quotes, Google-style docstrings.
- **Comments must reflect current state only** — never reference previous iterations of the
  code or deleted files.
- **Code links only to durable docs** — `docs/background/`, `docs/techniques/`,
  `docs/architecture/`, `docs/ml_experimentation/`. Never link from code *or* docs to `plans/`
  files, and never from code to `docs/roadmap/` pages or to any
  "Implementation details (deleted when this ships)" section — all of those are deleted when
  the work lands, so the reference rots. (Docs-to-docs links into `docs/roadmap/` are fine;
  retargeting them is part of ship-time triage.) Linking from a docstring to a durable page —
  e.g. `docs/architecture/` — is encouraged.
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

- **`Literal` type aliases — use a `Type` suffix** to distinguish them from the runtime tuples
  that drive Polars `Enum` declarations. Example:

  ```python
  EVALUATION_SCOPES: Final[tuple[str, ...]] = ("leaderboard", "production_monitoring", "ad_hoc")
  """Runtime tuple — used as pl.Enum(EVALUATION_SCOPES)."""

  EvalScopeType = Literal["leaderboard", "ad_hoc"]
  """Type annotation — currently-implemented subset; update when adding a new scope."""
  ```

  The `Type`-suffixed alias is what goes in function signatures; the `UPPER_SNAKE_CASE` tuple is
  what goes into `pl.Enum(...)`. They serve different purposes and should both exist.

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

### Delta Lake dictionary-encoded columns: declare Delta filter/partition columns as `String`

delta-rs stores all Arrow dictionary-encoded columns (`Categorical`, `Enum`) as plain `String` in
Parquet (this is the write-path gotcha documented in `_write_metrics_to_delta`, which casts the
remaining `Enum` columns to `String` before writing). Two consequences:

1. **A contract column you filter or partition on in Delta should be `String`, not `Categorical`.**
   If the schema declared it `Categorical`, every read would need a `String → Categorical` cast to
   satisfy the model — and a cast placed between `pl.scan_delta(...)` and a `.filter()` on that
   column **blocks predicate pushdown** (Polars can no longer prune Delta partitions or skip row
   groups, so it reads the *whole* table even when the filter names one partition). Declaring the
   column `String` matches what is on disk, so the scan is typed by `set_model` with no cast, the
   filter pushes straight down, and there is no dtype tension at the write boundary either.
   `PowerForecast.experiment_name` / `fold_id` (the `power_forecasts` partition columns) and
   `power_fcst_model_name` are `String` for exactly this reason; `PopulationFilter.apply` therefore
   takes and returns a typed `pt.LazyFrame[PowerForecast]`. Confirm pushdown with `.explain()` — it
   should list only the matching `partition=value` paths.

2. **For a genuinely low-cardinality column you only *read* (never filter on), cast `String →
   Enum`/`Categorical` lazily** — in the `pl.scan_delta(...)` result, before `set_model` — so the
   scan is typed from the start and the cast stays zero-cost until `.collect()`:

   ```python
   typed_scan = pt.LazyFrame.from_existing(
       pl.scan_delta(str(path)).with_columns(
           metric_name=pl.col("metric_name").cast(pl.Enum(METRIC_NAMES)),
       )
   ).set_model(MetricsSchema)
   ```

### Patito + Polars Gotcha: `pt.LazyFrame.filter()` drops the Patito subclass

Most Polars operations on a `pt.LazyFrame` return a plain `pl.LazyFrame`, including `.filter()`.
Reassigning `scan = scan.filter(...)` where `scan: pt.LazyFrame[Schema]` therefore fails `ty`'s
assignment check.

Workaround: rebind to a plain `pl.LazyFrame` local for the filter accumulation, then re-wrap before
returning:

```python
def apply(self, scan: pt.LazyFrame[MySchema]) -> pt.LazyFrame[MySchema]:
    lf: pl.LazyFrame = scan  # .filter() drops the pt subclass; accumulate on a plain LazyFrame
    if self.foo is not None:
        lf = lf.filter(pl.col("foo") == self.foo)
    return pt.LazyFrame.from_existing(lf).set_model(MySchema)  # zero-copy re-wrap
```

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
