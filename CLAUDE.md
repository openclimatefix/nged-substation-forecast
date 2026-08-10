# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Linting & formatting
uv run ruff check .            # check
uv run ruff check . --fix      # fix (never over a marimo notebook - see the Marimo section)
uv run ruff format .           # format
uv run ty check                # type checking
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md  # markdown lint

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

**Testing conventions** — where test dependencies and fixtures live, how discovery works, mocking
with `monkeypatch`, network-gated tests, the moto S3 reset-per-test rule, and the Patito assertion
house style — are documented on the **[Testing](docs/architecture/testing.md)** page.

## Skills

Detail that only matters while you are touching one specific thing lives in
`.claude/skills/<name>/SKILL.md` and is loaded on demand. **Load the relevant skill *before* you
start** — most of what they hold are traps that fail silently, so by the time you notice you
needed one, the mistake is already written.

| Skill | Load it before… |
|---|---|
| `polars-patito-gotchas` | writing Polars/Patito code that joins, casts, filters a `pt.LazyFrame`, declares a Patito field, or reads/writes Delta |
| `mkdocs-authoring` | editing any page under `docs/` — especially nested lists, list items containing code blocks, or wrapped links |
| `marimo-notebooks` | creating or editing a Marimo notebook (`packages/dashboard/*.py`, `packages/notebooks/*.py`) |
| `ty-workarounds` | acting on a `ty` error in Altair chart code or numpy `.view()` code, or adding any `# ty: ignore` |
| `plan-issue` | deciding what to build for a GitHub issue (`/plan-issue <N>`) — writes a reviewed plan, no code |
| `implement-issue` | writing code for an approved plan: worktree, verify set, PR, adversarial review, stop |
| `github-issue-pr-workflow` | `gh issue create`, `gh pr create`, `gh pr merge`, or ship-time triage |
| `github-graphql` | any `gh api graphql` call — sub-issue attach/reorder, issue Type, project fields |

## Docs

`docs/` contains a lot of useful information beyond API reference: forward-looking plans and
their ordering (`docs/roadmap/`), the portable design principles, engineering hypotheses and
inherent-stability argument (`docs/design-philosophy/`), durable explainers of solution methods
(`docs/techniques/`), background/requirements context (`docs/background/`), design rationale for
what's already built (`docs/architecture/`), and step-by-step operational how-to for what's already built
(`docs/ml_experimentation/`, `docs/live_service/` — design and how-to are deliberately separate
pages, cross-linked via "See also"). When planning new features, check `docs/` for relevant prior
discussion before proposing an approach.

The docs are published at <https://openclimatefix.github.io/nged-substation-forecast>. When
linking to a docs page from anywhere outside `docs/` itself (GitHub issues, PR bodies, code
docstrings), link to that rendered site (e.g.
`https://openclimatefix.github.io/nged-substation-forecast/roadmap/live-service/#anchor`),
never to a `github.com/.../blob/main/docs/...` path.

**Chart images — optimise an SVG before committing it.** A Vega/Altair chart exported straight to
SVG carries one path point per reading, at more decimal places than the viewport can express, so
the file is far larger than it needs to be. Run the export through

```bash
npx svgo@4 --multipass --precision=1
```

which took `docs/example_power_forecast.svg` from 571 KB to 296 KB with no visible change (verified
by rendering both to PNG at 2× and comparing pixel by pixel). Unoptimised exports tend to trip
`check-added-large-files`' 500 KB limit, which is the signal that this step was skipped.

**Prose style — write full sentences, don't drop the subject.** Don't clip words for terseness
if it leaves a sentence without a clear subject/verb. Prefer "We split storage across two
buckets so that..." over "Two buckets, not one — split so that...". The full form is more
readable and no less concise in practice.

**Write about the present, not the past.** The docs describe how the code works *now*. Don't write
about how it used to work, what a change replaced, or which issue changed it — that history lives
in git, in the PR and in the issue tracker, and repeating it here turns every page into a running
changelog and makes the docs unreadable. When a change invalidates a passage, rewrite the passage
to describe the new behaviour rather than appending a note about what changed. This is the
"comments and docs must reflect current state only" rule under Code Style, applied to prose.

## How planning works

Full description and a "which place do I use?" table: `docs/documentation-guide.md`. In brief:

- **GitHub** (issues + the OCF Project board) is the *complete, ordered* task list — task-level
  priority lives only there. When current priorities matter, query it with `gh` (epics map 1:1
  to roadmap milestones; dependencies are `blocked by` issue links).
- **`docs/roadmap/`** holds design, dependencies, and the milestone arc. Step-by-step mechanics
  sit inside each page under an "Implementation details (deleted when this ships)" section.
- **`docs/design-philosophy/engineering-hypotheses.md`** holds the falsifiable claims the
  engineering is meant to deliver. Cite them by label (`H1`, `T1.2`); labels are append-only —
  never renumber.
- **`plans/`** holds at most one file: the in-flight branch's implementation plan, written by the
  `plan-issue` skill before any code is touched and deleted on merge. One worktree per branch is
  what keeps it to one file, so parallel sessions never collide. Usually empty on `main`.

**Creating an issue or a PR has a checklist** — labels, org issue Type, OCF project membership
and its fields, sub-issue ordering, the `JackKelly` assignee — and none of it can be set by `gh
issue create` / `gh pr create`. It lives in the `github-issue-pr-workflow` skill, along with the
never-squash-merge rule and ship-time triage. Load it before you run either command.

## How work gets done

Two skills, in order, and they are deliberately separate so that Jack approves a design before
any code moves:

1. **`plan-issue`** (`/plan-issue <N>`) reads the issue, decides whether it is worth implementing
   at all, writes `plans/<branch-name>.md`, has a fresh sub-agent adversarially review that plan,
   and stops for Jack. It writes no code.
2. **`implement-issue`** picks up an approved plan: worktree, implement, the green-before-push
   verification set, PR with labels and assignee, a *second and independent* adversarial review
   of the diff, triage, stop. **Never merge.**

Stay inside the issue's scope; report unrelated design mistakes rather than fixing them.

**Why:** Jack reviews diffs in GitHub's UI and wants a PR to already have survived an
adversarial pass by the time he looks at it, so his review is the last line of defence rather
than the first. The fresh-reviewer requirement exists so the reviewer cannot be anchored by the
implementer's rationale; the triage step exists because reviewer findings are often wrong and
must not be applied uncritically.

## Architecture

This is a `uv` workspace monorepo. The root `src/nged_substation_forecast/` is the Dagster application; all reusable logic lives in `packages/`.

**A short list of design principles** governs architectural decisions:
[`docs/design-philosophy/design-principles.md`](docs/design-philosophy/design-principles.md).
Read them before proposing a structural change. If a change violates one, that is not a veto, but
say which principle is being traded away and what is bought in return.

### Inherent stability (production code)

**In production, never raise because an input is absent or stale — degrade, widen the uncertainty
bands, and record the degradation on the row.** Raising is reserved for states that are our own bug
(an empty promoted model, a contract violation), not for the outside world misbehaving. Corollaries
that come up constantly when editing `defs/`:

- **Liberal about missing inputs, strict about malformed ones.** Absent data routes into the
  always-output path; malformed data is rejected at the Patito boundary. Detectably-*wrong* input
  (a stuck meter) is treated as missing, not as data.
- **Asset checks warn, they do not block** — `AssetCheckSeverity.WARN` with `blocking=False`. There
  is deliberately no `ERROR`-severity check anywhere in the repo. A warning function must never be
  able to raise, or fail-open silently becomes fail-closed.
- **Measure degradation in missed NWP runs, never in hours of age.** We ingest one ECMWF run per
  day, so healthy NWP is 12–30 hours old depending on the 6-hourly slot.
- **R&D is the opposite**: the CV, training and metrics assets fail fast, because a quietly-degraded
  training run poisons every comparison built on it.
- **When a capability could live in the training loop or in the production service, put it in the
  training loop.** Keep the serving path close to "load a model, call `predict`".

Full rationale, the degradation ladder and the numbered rules:
[`docs/design-philosophy/inherent-stability.md`](docs/design-philosophy/inherent-stability.md). The
falsifiable claims it is meant to deliver — cite them as `H1`/`T1.2` and never renumber them — are
in [`docs/design-philosophy/engineering-hypotheses.md`](docs/design-philosophy/engineering-hypotheses.md).

### Packages

| Package | Purpose |
|---|---|
| `contracts` | Patito data schemas (the single source of truth for all data shapes) |
| `delta_store` | Physical storage policy for Delta tables: parquet writer properties, sort orders, significand rounding, write helpers |
| `ml_core` | Feature engineering and `BaseForecaster` abstract class |
| `nged_data` | Reading NGED JSON files from S3 and writing to Delta Lake |
| `dynamical_data` | Downloading ECMWF ensemble NWP from Dynamical.org |
| `geo` | H3 spatial indexing utilities |
| `xgboost_forecaster` | Concrete `BaseForecaster` implementation using XGBoost |
| `dashboard` | Marimo web apps for visualisation (`view_forecasts.py`, `map_and_timeseries.py`) plus their shared helpers in `src/dashboard/` |
| `notebooks` | Marimo exploration notebooks |

### Dagster Assets (`src/nged_substation_forecast/defs/assets.py`)

Three main assets:

- `power_time_series_and_metadata` — pulls NGED telemetry from S3, appends to Delta Lake, upserts metadata parquet
- `h3_grid_weights` — computes fractional H3 cell overlap with the GB boundary for spatial NWP aggregation
- `ecmwf_ens` — daily-partitioned asset that downloads ECMWF ENS NWP and appends it to Delta Lake via `delta_store.nwp.write_nwp`

### Data Contracts (`packages/contracts/`)

All tabular data flowing through the system is validated with **Patito** models. Key schemas:

- `PowerTimeSeries` — half-hourly power observations (MW/MVA) per `time_series_id`
- `TimeSeriesMetadata` — substation metadata including lat/lon, H3 index, substation type
- `Nwp` — NWP weather data in physical-unit `Float32`, on disk and in memory alike (rounded to a 13-bit significand at write time by `delta_store.nwp`)
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

The rules that come up in almost every edit are below. The fuller write-up — the ruff rule
selection and its `per-file-ignores` traps, error handling, the Patito friction budget — is
[`docs/architecture/code-style.md`](docs/architecture/code-style.md); keep the two consistent when
you change either.

- **Python 3.14+** required.
- **Polars only** — pandas is strictly forbidden. Use `pl.LazyFrame` throughout the pipeline and
  **do not call `.collect()` before the model boundary**; the full contract is
  [Lazy evaluation strategy](docs/architecture/performance.md#lazy-evaluation-strategy).
- **Patito** for all DataFrame schema definitions and validation. Use Patito type annotations (`pt.DataFrame[Schema]`, `pt.LazyFrame[Schema]`) whenever a function consumes or returns data that conforms to an existing schema — whether the function is public or private. Don't invent a new schema just to annotate a private helper; if no existing schema fits, use plain `pl.DataFrame` / `pl.LazyFrame`.
- **Prefer small functions.** Extract private helpers (`_name`) rather than letting a function body grow long, even if that means more parameters. A well-named helper with a clear docstring beats a long inline block. Eight parameters is acceptable when each is distinct and the division of labour is clear.
- **Ruff**: 100-char line length, double quotes, Google-style docstrings.
- **Comments and docs must reflect current state only** — never reference previous iterations of
  the code or deleted files. See "Write about the present, not the past" under Docs.
- **Code links only to durable docs** — `docs/design-philosophy/`, `docs/background/`, `docs/techniques/`,
  `docs/architecture/`, `docs/ml_experimentation/`, `docs/live_service/`. Never link from code *or* docs to `plans/`
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
- **Prefer self-documenting type hints over bare containers — a signature is documentation.**
  Jack strongly prefers expressive signatures and is happy to spend a few extra lines of code to
  get them, as long as complexity stays low. Whenever you would write `dict[str, str]` (or a bare
  `str` for a value from a fixed set, or a tuple of positional values), stop and ask whether a more
  self-documenting type is practical. Reach for: a `Type`-suffixed `Literal` alias for a closed set
  of string values (`StageType = Literal["register", "train", "predict", "metrics"]`); a named
  alias for a recurring shape (`MlflowTags = dict[str, str]`) so the intent is stated once and
  reused; a `TypedDict` for a structured mapping with known keys (e.g. `ObjectStoreOptions`) —
  taking the `TypedDict` in the signature and widening to a plain dict at the call boundary.
  Constraining `dict` *keys* to a `Literal` alias (`dict[TableNameType, str]`) is worthwhile for a
  closed vocabulary and works with bidirectional inference when callers pass dict literals.
  `packages/ml_core/src/ml_core/_repro.py` is the worked example. Don't force it where no honest
  stricter type exists — a genuinely heterogeneous or open-ended dict stays `dict[str, str]`.
- All consts must be marked with the maximally "constant" type.
  e.g. `CONST_SEQ: Final[tuple[str, ...]] = ("a", "b")` or `FOO: Final[str] = "bar"`
- Never relax an existing test to make it pass.

### Polars Style

These rules are all about making Polars code easy to read.

- When casting, prefer using the `cast` method like this: `df.cast({"foo": pl.Int8})`, in favour of
  using `df.with_columns(pl.col("foo").cast(pl.Int8))`. **Caveat:** this is only safe on a plain
  Polars frame — passing a `{column: dtype}` mapping to a *model-bearing* Patito frame silently does
  the wrong thing. See the `polars-patito-gotchas` skill.
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

### Gotchas that fail silently

Patito's model machinery collides with Polars and with delta-rs in five ways, and **none of them
raise at the point of the mistake**: a cross-model `.join()` that has to have its right-hand
operand stripped, a `{column: dtype}` `.cast` that gets swallowed on a model-bearing frame,
`ge`/`le` doing nothing at all on a datetime field, `.filter()` dropping the Patito subclass, and
a dictionary-encoded column blocking Delta predicate pushdown so a partition-filtered query reads
the whole table. They are written up, with the workaround for each, in the
**`polars-patito-gotchas`** skill. Load it before writing the code, not after the confusing
`validate()` error.

Two more live in their own skills: **`marimo-notebooks`** (leading underscores are cell-local,
imports belong in `app.setup`, never `ruff check --fix` a notebook) and **`ty-workarounds`**
(known upstream `ty` bugs on Altair and numpy, where the code is correct and the checker is not).

### Polars Gotcha: row counts silently wrap past 2³² rows (32-bit `IdxSize`)

Default Polars builds use a 32-bit row index (`IdxSize`), capping any single materialised frame,
row count, or row index at 2³² (~4.29 billion) rows. Past the cap there is **no error** — counts
wrap modulo 2³², streaming engine included.

- **Never row-count a table that can exceed 2³² rows with Polars.** Use the Delta log instead —
  `DeltaTable(path).count()`, or sum `num_records` over `get_add_actions(flatten=True)` — both
  metadata-only and exact.
- Filtered/partition-pruned queries whose *result* stays under 2³² rows are correct even when the
  underlying scan is bigger, and value aggregations (`sum`, `min`/`max`, quantiles) over >2³² rows
  are unaffected — only row counts and row indices wrap.
- Tables past the cap today: NWP (~5.9B rows). `power_forecasts` will pass it at V2 scale.

The measurements behind all of that, and the reasoning:
[Performance and Scale → The other hard ceiling](https://openclimatefix.github.io/nged-substation-forecast/architecture/performance/#the-other-hard-ceiling-polars-32-bit-row-index).

## This is a young project

The project is a new, green-field project. No one else is using this code yet. Which means:

- It's 100% fine to make breaking changes, if doing so improves the code. (And as long as we update
  all the downstream code.)
- Our aim is to make the code well-organised and easy to use.
- None of this code is "written in stone" or battle-tested.
- We haven't trained any "serious" ML models yet, so a change that invalidates an existing trained
  model or its saved config costs us a retrain, not a migration path. Don't design for backwards
  compatibility with models we've already trained.
- If you see a design mistake _anywhere_ in the code, then please flag that design mistake to me.
  I'd much rather end up with a project that's well engineered. (That said, if we're working on
  feature X, and you spot a mistake in some code that isn't obviously in scope for X, then please
  discuss the change with me first. Definitely don't make out-of-scope changes with asking me!)
