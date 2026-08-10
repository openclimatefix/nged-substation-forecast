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
with `monkeypatch`, network-gated tests, and the Patito assertion house style — are documented on the
**[Testing](docs/architecture/testing.md)** page. (The moto and Polars row-count testing *gotchas*
live further down this file, with the other gotchas.)

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
- **Body** — if (and only if) the docs already contain a plan for the issue (e.g. a
  `docs/roadmap/` section), the body may be *just* a link to that rendered docs section and
  nothing more; don't duplicate the plan. Otherwise, write a self-contained body.
- When the body links to a docs page, link to the **rendered site**
  (`https://openclimatefix.github.io/nged-substation-forecast/...`), never a `github.com`
  blob path.

`gh issue create` can't set any of these: use `gh issue edit --add-label` for labels, the
`updateIssueIssueType` GraphQL mutation for Type, and `gh project item-edit` (or the
`updateProjectV2ItemFieldValue` mutation) for the project fields.

**Creating pull requests** — whenever you create a PR, also set:

- **Labels** — pick whatever fits (e.g. `documentation`, `enhancement`, `bug`), same label set as
  issues.
- **Assignees = JackKelly**.

`gh pr create` can't set either: use `gh pr edit --add-label` and `gh pr edit --add-assignee
JackKelly` right after creating the PR.

**Merging pull requests** — never squash-merge. Jack wants the full commit history preserved in
`main`, so use a merge commit (`gh pr merge --merge`) or rebase (`gh pr merge --rebase`), not
`gh pr merge --squash`.

**GitHub GraphQL calls** (attaching/reordering sub-issues, setting an issue's Type, setting a
project field) — see the `github-graphql` skill (`.claude/skills/github-graphql/`) for exact
`gh api graphql` invocations and how to obtain the node IDs they need.

**Ship-time triage** — when a PR lands a roadmap item, that PR (or an immediate follow-up)
must also:

1. Promote surviving design decisions to their permanent home (`docs/architecture/`,
   `docs/ml_experimentation/`, …).
2. Delete the item's "Implementation details" section (and any `plans/` file), pasting it (or
   a summary) into the PR body. When a roadmap page's last 🚧 item ships, delete the page
   (nav entry, inbound doc links).
3. Close the GitHub issue; update the status banner on the roadmap page (and the milestone
   section in `docs/roadmap/index.md` if the arc changed).

## Sub-agent routine

When dispatching a sub-agent (or a fresh Claude Code/Desktop session) to solve a GitHub issue in
this repo, give it these steps up front — a report back after step 1 is not finished work:

1. **Set up an isolated worktree** so concurrent sessions don't collide:

   ```bash
   git worktree add .claude/worktrees/<branch-name> -b <branch-name>
   cd .claude/worktrees/<branch-name>
   ln -s /home/jack/dev/python/nged-substation-forecast/.env .env   # if it exists and isn't already there
   ```

   If several sub-agents run concurrently, also give each its own scratchpad subdirectory —
   a shared scratchpad root means two agents writing e.g. `pr_body.md` can collide and one
   agent's output gets briefly published under another's PR.

2. **Implement**, following every convention in this file — including leaving every doc page
   that touches the change consistent with the code as it now stands, describing only how the
   code works *now* (see "Write about the present, not the past" above). For a docs change that
   touches a link, run `uv run mkdocs build --strict` and read the rendered HTML under `site/`,
   not just the linter — Python-Markdown has rendering gotchas that neither `pymarkdown scan`
   nor a successful `mkdocs build` catches on their own.

3. **Verify, all green before pushing**: `uv run ruff check .`, `uv run ruff format .`,
   `uv run --all-packages ty check`, `uv run pytest`, plus (if docs were touched)
   `uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md` and
   `uv run mkdocs build --strict`.

4. **Push and open the PR** against `main`, with labels and `JackKelly` as assignee (`gh pr
   create` can't set either — follow with `gh pr edit --add-label <label>` and `gh pr edit
   --add-assignee JackKelly`), linking the issue so it closes on merge. Commit messages end
   with `Co-Authored-By: Claude <noreply@anthropic.com>`.

5. **Spawn a *new*, independent sub-agent to adversarially review the PR** — give it only the
   PR number, not the implementer's reasoning, so it isn't anchored by it. Tailor the reviewer
   brief to the issue: name the failure modes most worth attacking (the risky claim, a
   behaviour change hiding inside a refactor, whether a new test would actually have failed on
   `main`) rather than asking for a generic review.

6. **Triage the review's findings** — verify each against the code rather than accepting it,
   fix genuine defects, push, and record why any finding was rejected.

7. **Stop and wait for Jack's review. Never merge.**

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

- **Python 3.14+** required.
- **Polars only** — pandas is strictly forbidden. Use `pl.LazyFrame` and only `.collect()` when necessary.
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

### Patito Gotcha: `ge`/`le` are silently ignored on a datetime field

`pt.Field(ge=..., le=...)` enforces nothing on a `datetime` column. Patito builds its bounds checks
by reading the `minimum`/`maximum` keywords out of the Pydantic JSON schema, and JSON Schema
defines those keywords for numbers only — so a datetime field's `Ge`/`Le` metadata never reaches
the JSON schema, Patito finds no keyword to turn into a filter, and `validate()` accepts every
year. There is no warning and no error; the constraint simply does not exist. (`ge`/`le` on a
numeric field works exactly as documented, which is what makes this so easy to miss.)

**How to apply:** bound a datetime column from the model's `validate` override, not from the field.
`contracts.common.check_datetime_bounds` is the shared helper, and `MIN_PLAUSIBLE_DATETIME` /
`MAX_PLAUSIBLE_DATETIME` are the shared bounds; `PowerTimeSeries.validate` and `Nwp.validate` are
the worked examples. A `constraints=` Polars expression on the field also works, but its failure
message is the generic "1 row does not match custom constraints", so prefer the explicit check when
you want the error to say which bound was broken.

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

### Polars Gotcha: row counts silently wrap past 2³² rows (32-bit `IdxSize`)

Default Polars builds use a 32-bit row index (`IdxSize`), capping any single materialised frame,
row count, or row index at 2³² (~4.29 billion) rows. Past the cap there is **no error** — counts
wrap modulo 2³²: `pl.len()` over the 5.9-billion-row NWP dev table returns 1,652,180,189
(= 5,947,147,485 mod 2³²), and `group_by(...).agg(pl.len())` wraps identically for any single
group past the cap, streaming engine included. Full analysis:
[Performance and Scale → The other hard ceiling](https://openclimatefix.github.io/nged-substation-forecast/architecture/performance/#the-other-hard-ceiling-polars-32-bit-row-index).

- **Never row-count a table that can exceed 2³² rows with Polars.** Use the Delta log instead —
  `DeltaTable(path).count()`, or sum `num_records` over `get_add_actions(flatten=True)` — both
  metadata-only and exact.
- Filtered/partition-pruned queries whose *result* stays under 2³² rows are correct even when the
  underlying scan is bigger, and value aggregations (`sum`, `min`/`max`, quantiles) over >2³² rows
  are unaffected — only row counts and row indices wrap. Both verified empirically.
- Tables past the cap today: NWP (~5.9B rows). `power_forecasts` will pass it at V2 scale.

### Testing Gotcha: moto's S3 backend is process-global — reset it per test

The in-process `moto` server used for the S3 tests keeps its bucket contents in a **process-global
backend that outlives the `ThreadedMotoServer` object**, so a module-scoped server does not hand
each test a clean slate. A test whose write path runs twice against that server — a re-run, or
state left behind by an earlier test — reads stale data: an appended Delta table returns double the
rows, and an `object_exists` precondition sees a leftover parquet. Keep the *server* module-scoped
for speed, but give each test a **function-scoped** fixture that `POST`s to `/moto-api/reset` and
recreates the bucket before the test body runs, so every test starts pristine and independent of
execution order. `tests/test_s3_data_paths.py` is the canonical pattern.

### Altair Gotcha: `ty` loses the chart type after a `mark_*()` call

Altair decorates every `mark_*` method with `@use_signature`, whose return type is expressed
through a hand-written generic `TypeAliasType` over `Concatenate`. Since ty 0.0.64, ty resolves
that alias but never solves its type variable, so `alt.Chart(df).mark_line()` infers as the bare
`T@__call__` and the next call in the chain fails with
`unresolved-attribute: Object of type 'T@__call__' has no attribute 'encode'`. The code is
correct — pyright infers `Chart` — and this is upstream ty bug
[astral-sh/ty#2520](https://github.com/astral-sh/ty/issues/2520).

**How to apply:** put `# ty: ignore[unresolved-attribute]` on the `.encode(` line of each chart
chain. Restructuring does not help: annotating an intermediate variable as `alt.Chart` instead
raises `invalid-assignment`, and calling `.encode()` before `.mark_*()` just moves the unsolved
type variable to the function's return. When ty fixes the bug, every suppression turns into an
`unused-ignore-comment` warning, which is the signal to delete them all.

Ruff's lint rule *selection* is written out in `pyproject.toml` (`[tool.ruff.lint] select`) for
the same class of reason: ruff's defaults are a curated menu with no stability promise, so an
inherited selection lets a `uv lock` refresh change which rules the repo enforces. `select` names
whole families; `ignore` names each family member we decline, with the reason on the line above
it. **When a rule fires somewhere it should not, add an `ignore` entry (or a `per-file-ignores`
entry) with its justification — do not drop the whole family.** Note that ruff's defaults are not
a superset of the old `E4`/`E7`/`E9`/`F` gate: of pycodestyle-E they keep only `E722` and `E902`,
which is why `E4`/`E7`/`E9` are listed explicitly.

Two traps when editing `[tool.ruff.lint.per-file-ignores]`:

- **`*` crosses `/`.** A key of `packages/dashboard/*.py` also silences
  `packages/dashboard/src/dashboard/*.py`. Name individual files when you mean "just the ones in
  this directory"; `**/tests/**` is the right way to say "every tests directory, root included".
- **A `# noqa` cannot live inside a docstring**, because it would become part of the string. An
  over-long docstring line has to be reworded or re-wrapped, never suppressed.

### numpy Gotcha: `ty` mis-types `.view(np.uint32)` — pass `np.dtype(np.uint32)` instead

Since ty 0.0.67, `arr.view(np.uint32)` infers as
`ndarray[_AnyShape, type[unsignedinteger[_32Bit]]]` instead of
`ndarray[_AnyShape, dtype[unsignedinteger[_32Bit]]]`, so every subsequent operation on that array
fails — a bit-mask check reports `unsupported-operator: Unsupported & operation`. `ndarray.view`
is overloaded, and the inferred type looks like the overload taking
`DTypeT | _HasDType[DTypeT]` (with `DTypeT` bound to `np.dtype`) matched with `DTypeT` solved as
`type[np.uint32]`, in violation of that bound. The code is correct at runtime — pyright infers
`dtype[unsignedinteger[_32Bit]]` — and this is upstream ty bug
[astral-sh/ty#4208](https://github.com/astral-sh/ty/issues/4208).

**How to apply:** pass a real dtype object — `arr.view(np.dtype(np.uint32))` — which is the same
call at runtime and which ty resolves to the correct `ndarray[..., dtype[uint32]]`. Prefer this
over a `# ty: ignore` comment: the suppression would have to sit on the line that *uses* the
array, which can be several lines away from the `.view()` call that actually causes it.
Annotating the intermediate as `npt.NDArray[np.uint32]` does not work — it raises
`invalid-assignment` instead. The significand-rounding tests in `packages/delta_store/tests/`
are the worked examples. Nothing warns when the upstream bug is fixed, so the signal to delete
this section is astral-sh/ty#4208 closing; the `np.dtype(...)` calls themselves can stay, because
they are correct either way.

### Marimo Notebooks

Marimo notebooks (`packages/dashboard/*.py`, `packages/notebooks/*.py`) are reactive: each
`@app.cell` function is a separate cell, and the `with app.setup:` block holds names shared by
every cell. Two authoring rules follow from how Marimo scopes names and how ruff sees them.

- **Never give a leading underscore to anything you want to reuse across cells.** Marimo treats
  any name starting with `_` (a variable *or* a function) as *cell-local* — it is not exported to
  other cells, so a `_helper()` defined in one cell (or in `app.setup`) is invisible everywhere
  else and the call fails at runtime. A helper that multiple cells call must have a public name
  (no leading underscore). This is the opposite of the usual `_private` convention, so it is easy
  to get wrong; the leading-underscore-means-private habit does not apply inside a Marimo
  notebook.

- **Put every import in the `with app.setup:` block, never at module top level and never let
  Marimo thread them through cell signatures.** When imports live in `app.setup`, they are real
  `import` statements that ruff analyses, so ruff flags a missing or unused import. If you instead
  scatter imports into individual cells, Marimo passes the imported names into the cells that use
  them as function parameters (`def _(pl, mo): ...`), and ruff treats a parameter as always
  defined — so a genuinely missing import is invisible to the linter and only blows up at runtime.
  Keeping all imports in `app.setup` keeps them statically checkable and available to every cell.

- **Never run `ruff check --fix` over a Marimo notebook.** When an autofix needs a name the file
  does not import yet, ruff inserts the import into the file's *top-level* import block — outside
  `app.setup`, where no cell can see it. The notebook then dies with a `NameError` the next time it
  is opened, while `ruff check` reports success, because what ruff produced is valid Python. This
  is a whole class, not one rule: any fix that adds an import does it, and ruff has no per-file
  fixability setting to prevent it (`unfixable` is global, and `per-file-ignores` would silence the
  check as well as the fix). The pre-commit hook is split so notebooks are checked but never
  auto-fixed; a bare `uv run ruff check . --fix` typed by hand is *not* covered, so after running
  one, check `git diff` for an import that landed above `import marimo` and move it into
  `app.setup`. `marimo check --fix` does not rescue it: that deletes the module-level import and
  rewrites the cell that used the name as `def _(name)`, leaving the name as a cell input nothing
  defines. Both shapes are caught by `scripts/check_marimo_notebooks.py` — a pre-commit hook, and
  run over every notebook by `tests/test_marimo_notebooks.py` — so a broken notebook fails the
  commit or CI rather than surviving to whoever next opens it.

### MkDocs Gotcha: a list item needs a blank line before it if it follows an indented continuation

Python-Markdown (MkDocs' renderer) doesn't let a list item interrupt a paragraph the way
GitHub-flavored Markdown does. If a bullet's continuation content ends with an indented
paragraph (e.g. a clarifying sentence after a fenced code block inside the item) and the next
sibling bullet immediately follows with no blank line in between, Python-Markdown treats the new
list-marker line as more paragraph text rather than a new list item — the marker renders as a
literal hyphen, merged into the previous sentence's prose. `pymarkdown scan` does **not** catch
this: a markdown source with the missing blank line lints clean.

**How to apply:** always put a blank line between a list item's continuation content (paragraphs,
fenced code blocks) and the next sibling item. For any non-trivial list item — one that embeds a
code block or multiple paragraphs — spot-check with `uv run mkdocs build --strict` and inspect
the rendered HTML rather than trusting the linter alone. See also the nested-sub-bullet indent
gotcha (4 spaces, not 2) tracked in memory — same root cause class: Python-Markdown's list
parsing is stricter than CommonMark and stricter than `pymarkdown`'s default checks.

### MkDocs Gotcha: a wrapped link whose continuation line starts with `#` renders as a heading

CommonMark requires a space after `#` for a line to start an ATX heading (`#5` is just text,
`# 5` is a heading). Python-Markdown does not enforce that space, so any line that happens to
start with `#` — for any reason — is parsed as a heading. A markdown link wrapped across the
80-ish-character line length this repo's prose otherwise isn't held to can put the `#123](url)`
half of `[issue #123](url)` at the start of a line, and Python-Markdown reads it as a heading
rather than as the second half of a link. The rendered page gets a stray `<h1>`/`<h2>` containing
the raw URL, the link text before the wrap point left dangling as plain text, and the paragraph
split in two. Neither `pymarkdown scan` nor `mkdocs build --strict` catches this — both pass on a
source file that renders visibly broken.

**How to apply:** when a link's markdown source wraps across a line break, make sure the
continuation line does not begin with `#`; keep `[text](url)` together on one line, or wrap
before `[` rather than inside it. This is the same root cause as the list-continuation gotcha
above — Python-Markdown is stricter/weirder than CommonMark in ways the linters don't catch — so
it generalises to a standing rule, not just this one wrapping case: **any docs PR that touches
links should run `uv run mkdocs build --strict` and then actually read the generated HTML under
`site/`**, not just trust a clean `pymarkdown scan` plus a successful `mkdocs build`. Both of
those can pass on rendering that is visibly wrong; only reading the HTML catches it.

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
