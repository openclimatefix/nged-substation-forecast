# Plan: a cleaned copy of NGED's power telemetry (#1019)

**The problem.** Cleaning rules for NGED's power telemetry are being written now, and the pipeline
has no place to run them. Every asset that reads observed power — training, CV prediction, live
forecasting, eligibility, and effective capacity — reads the raw `power_time_series` Delta table
directly. There is nowhere to drop the first three months of a substation's data, a substation's
false zeros, a physically implausible reading, or a generator's spurious zeros, and nothing to
report what was dropped.

**The planned solution.** A new Dagster asset, `clean_nged_power_data`, runs in the hourly ingest
job straight after `power_time_series_and_metadata`. It reads the whole raw power table and the
`TimeSeriesMetadata` roster, passes both to one function, `nged_data.cleaning.flag_nged_power`, and
writes the result over a new Delta table, `cleaned_power_time_series`. That table holds every raw
row plus a nullable `drop_reason` column: null for a row that passed, otherwise the name of the rule
that rejected it. Today `flag_nged_power` flags nothing. The asset reports, per drop reason, how
many rows and series were flagged and the range of the flagged values, as Dagster output metadata.
Every reader of power except the ingest and its freshness check — training, CV prediction, live
forecasting, eligibility, effective capacity, the leaderboard's `metrics`, and the two dashboards —
switches to the cleaned table's unflagged rows. The raw table is never modified.

## Verdict, size and departures

**Verdict: worth implementing.** No cleaning step exists anywhere: `docs/roadmap/data-cleaning.md`
says versions 0.1 to 0.3 train on uncleaned telemetry apart from the timestamp repair at ingest.

**Size: complex.** The five triggers:

- What gets stored: **yes** — a new Delta table and a new Patito contract,
  `CleanedPowerTimeSeries`.
- The production serving path: **yes** — `live_forecasts` reads the cleaned table, and the new
  asset joins the hourly production ingest job.
- A degradation rule: **yes** — `live_forecasts` now depends on a second hop from the raw data.
- More than one defensible design: **yes** — the alternatives are recorded below.
- Callers not nameable without searching: no — the readers are named below.

Complex buys both plan reviews and both diff reviews. The design was settled in discussion with the
maintainer after two earlier designs (read-time cleaning with a list of rule functions) were each
reviewed; this design gets a fresh review.

**Departures from the issue body.**

- The issue imagines a function that returns the lazy frame it receives, unchanged. Here the
  function returns the same rows plus a `drop_reason` column that is null everywhere today, so
  readers see exactly the rows they see now.
- The issue asks for "a" cleaning asset; the plan's asset materialises a cleaned table rather than
  passing a frame through.

## Why the whole cleaned table is written eagerly, not cleaned lazily at read time

**The cleaning rules need context that the reader's request does not carry, so the cleaning has to
run once over the whole table, and the result has to be stored.** Two of the planned rules show it:

- "Drop the first three months of each substation's data" needs each series' first reading. A
  reader asking for a 15-day live window, or for a training window starting after the series began,
  sees a different "first reading" and would drop the wrong rows.
- A rule comparing a PV site with its neighbours needs the neighbours' readings. A reader asking for
  one site would get a different cleaning result from a reader asking for all of them.

Lazy cleaning could only give consistent answers by reading every series' full history on every
read. That removes laziness's only advantage, and at V2 scale costs up to about 5 GB in memory per
read (2,500 series, about 16 bytes per row in memory as measured on V1, 4 to 7 years of
half-hours).

**A stored table also makes the failure policy simple.** If cleaning fails, the asset fails loudly
(the job's Sentry failure hook) and every reader carries on with the last good cleaned table.
Production degrades through data at rest (principle 14) with no guard in any reader, and research
reads a table that is stale rather than silently uncleaned.

**Storage is cheap.** The raw table measured 36 MB for 3.3M rows across 33 series on 2026-10-05,
about 11 bytes per row. At V2 (about 2,500 series), 4 years is about 175M rows, 1 to 2 GB on disk;
the cleaned copy roughly doubles that.

**What it costs** is a full rewrite of the cleaned table on every run — trivial at V1, where
rewriting the 36 MB table took 0.08 s, and a few GB per run at V2, to be re-measured before V2 —
and one more hop between NGED's data and `live_forecasts`. Appending only new rows instead of
rewriting is deferred to its own issue under the v0.9 epic, because appending needs a key
comparison, a context window, a re-cleaned recent tail, and a version-triggered rebuild.

## What changes, file by file

### `packages/contracts/src/contracts/power_schemas.py` — new contract (maintainer-approved)

- `DROP_REASONS: Final[tuple[str, ...]] = ()` — the vocabulary of drop reasons, empty today. The
  docstring says a rule's author adds the rule's reason here with the rule.
- `CleanedPowerTimeSeries(PowerTimeSeries)` — the same three fields plus `drop_reason: str | None`,
  stored as `pl.String` with the constraint `drop_reason.is_null() |
  drop_reason.is_in(DROP_REASONS)`. Not a `pl.Enum`: the reviewer showed that an Enum column written
  through `write_deltalake` makes every later read raise a `SchemaError` (gotcha 5 in
  `polars-patito-gotchas`), and the readers filter on this column. Subclassing keeps
  `PowerTimeSeries.validate`'s datetime-bound, sort-order, and key-uniqueness checks (the reviewer
  confirmed all three run on the subclass).

### `packages/nged_data/src/nged_data/cleaning.py` (new)

- `flag_nged_power(power: pt.LazyFrame[PowerTimeSeries], metadata:
  pt.DataFrame[TimeSeriesMetadata]) -> pt.LazyFrame[CleanedPowerTimeSeries]` — the function the
  rules go into. `power` is always the whole raw table. Today the function adds a null `drop_reason`
  column (`pl.lit(None, dtype=pl.String)`; a bare `pl.lit(None)` has the `Null` dtype) and nothing
  else. The docstring states the contract a rule must keep: return every input
  row exactly once, never change `time_series_id` or `time`, flag rather than delete, and when two
  rules match one row, record the first in the function's order.
- No `first_times` argument. Because `power` is the whole table, a rule computes each series' first
  reading itself (`pl.col("time").min().over("time_series_id")`). The append design will see only a
  window, so it will need the first readings passed in; that issue adds the argument.

### `packages/nged_data/src/nged_data/storage.py`

- `scan_cleaned_power(path, storage_options) -> pt.LazyFrame[PowerTimeSeries]` — scans the cleaned
  table, keeps rows where `drop_reason` is null, and selects the three `PowerTimeSeries` columns.
  It is the one read path for every consumer, so no consumer can forget the filter.
- `coverage_from_power(power: pt.LazyFrame[PowerTimeSeries]) -> pt.DataFrame[TimeSeriesCoverage]` —
  split out of `time_series_coverage(path)`, which keeps its absent-table check and calls it. The
  other two callers of `time_series_coverage` (`power_data_is_fresh` and the ingest's
  `select_new_rows`) keep reading raw coverage, because freshness and de-duplication are about what
  NGED delivered.

### `packages/delta_store/src/delta_store/cleaned_power_time_series.py` (new)

- `write_cleaned_power_time_series(df: pt.DataFrame[CleanedPowerTimeSeries], table_uri, *,
  raw_version: int, storage_options, retention_hours: int = 2)` — `write_deltalake(mode="overwrite",
  partition_by=["time_series_id"])`, one atomic Delta commit (principle 10), then
  `vacuum(retention_hours=retention_hours, dry_run=False, enforce_retention_duration=False)`. Both
  flags matter: `dry_run` defaults to `True`, which lists files and deletes none, and delta-rs
  refuses a retention under 168 hours without `enforce_retention_duration=False`. Without the vacuum
  every hourly overwrite leaves the previous copy on disk forever. The retention bounds how long the
  longest reader scan may take: a scan resolves its file list when it starts, and only files
  tombstoned more than 2 hours ago are deleted, so at steady state about three copies sit on disk
  (about 110 MB at V1).
- The write commit's `custom_metadata` records `raw_version`, the raw table's Delta version the
  cleaning run read, for provenance and for the keeping-up check. The vacuum adds two commits of
  its own (`VACUUM START` and `VACUUM END`) after the write, so a reader of `raw_version` walks
  `DeltaTable.history()` back to the newest `WRITE` commit rather than taking the latest commit.
- `write_deltalake` does not keep row order within a partition (the reviewer found 13 of 33
  partitions out of `time` order after one write), so the table on disk is not sorted. That does
  not matter at V1; at V2 it weakens row-group pruning on `time`.
- `delta_store/__init__.py` lists every module in its docstring and `__all__`; both gain the new
  module.

### `packages/contracts/src/contracts/settings.py`

- `cleaned_power_time_series_data_path`, derived beside `power_time_series_data_path` in
  `_derive_unset_paths`.

### `src/nged_substation_forecast/defs/cleaning_assets.py` (new)

The asset gets its own module rather than joining `defs/assets.py`, which holds the ingest assets.

- `clean_nged_power_data` — `deps=["power_time_series_and_metadata"]`, production-layer tags. A
  minimal wrapper:
    - If the raw table does not exist yet, log and return with `n_rows: 0` metadata rather than
      raising, because an absent input degrades (inherent-stability).
    - Scan the raw table, read the whole roster (with `allow_superfluous_columns=True`, as
      `metrics` does, because the parquet carries extra geo columns), call `flag_nged_power`,
      collect with the streaming engine, sort by `(time_series_id, time)` —
      `PowerTimeSeries.validate` rejects unsorted rows, and Delta file order is not sorted, which
      the reviewer confirmed fails on the real V1 data — then `CleanedPowerTimeSeries.validate`,
      then write, passing the raw table's version read at the start of the run.
    - Metadata: `n_rows`, `n_rows_kept`, `n_time_series`, and per drop reason
      `drop_reason/<reason>/n_rows`, `.../n_time_series`, `.../min_power`, and `.../max_power`,
      computed from the collected frame with one `group_by("drop_reason")`.
    - No `try` around the cleaning: a failing rule or a contract violation is our bug, so the asset
      raises and the readers keep the last good table.
    - The vacuum is housekeeping, and a vacuum failure (for example a transient object-store
      error) comes from the outside world after the fresh table is already written. The writer
      therefore runs the vacuum in its own `try`; the asset catches a vacuum failure, reports it
      with `report_asset_degradation(asset_name="clean_nged_power_data", ...)`, and records
      `vacuum_failed: True` in the metadata rather than failing the run.
- Registered in `src/nged_substation_forecast/definitions.py`'s `load_assets_from_modules` list.
  The new asset check below goes in the same file's explicit `asset_checks=[...]` list.

### `src/nged_substation_forecast/defs/checks.py`

- `cleaned_power_keeps_up_with_raw` — a `WARN`, `blocking=False` asset check on
  `power_time_series_and_metadata`, beside `power_data_is_fresh`. It compares the raw table's
  current Delta version with the `raw_version` recorded in the cleaned table's newest `WRITE`
  commit, and warns when the cleaned table is 3 or more raw commits behind, when the cleaned table
  is absent, or when the key is missing. Counting raw commits rather than hours of reading time
  counts missed runs, which is the unit inherent-stability asks for. A lag of 0 or 1 commits is
  healthy: the check runs in parallel with `clean_nged_power_data` in the same job, so it may see
  the cleaning from the previous run. Reading time would false-alarm after an NGED backlog, when one
  ingest catches up several hours of readings at once. The body sits inside the same `BaseException`
  guard as the other checks and reports with `report_check_degradation`, so it cannot raise. It is
  the only signal for a cleaning asset that silently stopped running, because `power_data_is_fresh`
  reads raw coverage and the Sentry failure hook fires only on a run that failed. The roster is a
  new way for that to happen: `live_forecasts` deliberately avoids the roster, but a bad roster now
  stops cleaning and so makes live power go stale.

### `src/nged_substation_forecast/defs/schedules.py`

- Add `"clean_nged_power_data"` to `power_time_series_and_metadata_job`'s selection, so the asset
  runs hourly, straight after ingest and five minutes before each 6-hourly `live_forecasts` slot.
  That is well over the four runs a day the maintainer requires. The job's `description` string
  and the comment above the job, which both say the job only ingests, are updated to name the
  cleaning step too. #1020 also edits this file (the
  `ecmwf_ens` schedule), but the maintainer will merge this PR before work on #1020 starts, and a
  comment on #1020 says so.

### Readers

- `_engineering_inputs.py`, `load_engineering_inputs`: read `scan_cleaned_power` instead of the raw
  path. This covers `trained_cv_model`, `cv_power_forecasts`, and `live_forecasts`.
- `cv_assets.py`, `effective_capacity`: read `scan_cleaned_power`.
- `cv_assets.py`, `metrics`: the actuals it scores against come from `scan_cleaned_power` instead
  of the raw path (cv_assets.py:986–989). The maintainer decided that the leaderboard scores
  against cleaned actuals. `metrics` is also being edited for #958, so this one-line change may need
  a trivial rebase in whichever PR merges second.
- `cv_assets.py`, `eligible_time_series`: `coverage_from_power(scan_cleaned_power(...))`, behind a
  `delta_table_exists` check on the cleaned table, so an absent table still yields an empty
  population.
- The two dashboard notebooks (`packages/dashboard/view_forecasts.py:332` and
  `map_and_timeseries.py:122`) read `scan_cleaned_power`, following the `marimo-notebooks` skill.
- The `deps` of `eligible_time_series`, `effective_capacity`, `trained_cv_model`, `metrics`, and
  `live_forecasts` change from `power_time_series_and_metadata` to `clean_nged_power_data`.
- Provenance: `trained_cv_model`, `cv_power_forecasts`, and `metrics` stamp the
  `power_time_series` Delta version in MLflow so a run can be replayed with
  `scan_delta(version=N)`. The cleaned table's own versions are vacuumed after 2 hours, so they are
  not replayable. The three assets instead stamp a new key, `cleaned_power_time_series_source`: the
  `raw_version` from the cleaned table's newest `WRITE` commit, read at the start of the asset, or
  `ABSENT` when the table or the key is missing. Replaying means re-running the cleaning over that
  raw version at the run's git SHA. `ml_core/repro.py` gains the `TableNameType` value and a small
  lookup function beside `get_delta_versions`, which only reads `DeltaTable.version()`. The stamp
  is approximate in one way: `cv_power_forecasts` re-scans power for every `init_time` chunk, so one
  long run can span several hourly cleaned tables.
- `tests/test_asset_layer_tags.py` pins the production-layer assets; `clean_nged_power_data` is
  added there, and that test requires `docs/architecture/overview.md` to change with it.
- Still reading raw power: the ingest itself and its `power_data_is_fresh` check, which measure
  what NGED delivered, and the cleaning asset. Nothing else reads the raw table.

## How the cleaned table could drive `asset_health_history`

**The cleaned table already holds most of what the delivery table `asset_health_history` needs.**
That table has one row per `(time_series_id, time)` and a `warning_type` from a fixed vocabulary
(`HEALTHY`, `MISSING VALUE`, `STUCK TIMESERIES`, `INVALID TIMESERIES VALUE`, `GENERATOR OR CIRCUIT
FAULT`, and others). A later asset could build it from the cleaned table by mapping each
`drop_reason` to a `warning_type` — `implausible_value` to `INVALID TIMESERIES VALUE`,
`generator_spurious_zero` to `GENERATOR OR CIRCUIT FAULT`, and a null to `HEALTHY` — and by adding
`MISSING VALUE` rows for half-hours absent from the table. Some reasons describe data policy rather
than asset health (`substation_first_three_months`) and would map to `HEALTHY` or be left out.
Nothing in this issue builds that table; the mapping is a decision for the issue that does.

## Design-philosophy check

- **Production degrades, research fails fast.** The asset raises on our own bugs; readers in
  production carry on with the last good cleaned table, as `live_forecasts` already does when the
  ingest misses an hour. Research reads the same table.
- **One `WARN`, `blocking=False` asset check is added** (`cleaned_power_keeps_up_with_raw`), whose
  body cannot raise. The cleaning stats themselves go to output metadata, and a failed cleaning run
  also alerts through the job's Sentry failure hook.
- **First deploy**: the cleaned table is absent until the asset first runs, and `live_forecasts`
  raises reading it, exactly as it raises today when the raw table is absent. The deploy step in
  `operations.md` closes that window.
- **Principle 7 (strict contracts)**: a new Patito model whose `drop_reason` is constrained to a
  declared vocabulary.
- **Principle 11**: the asset materialises the whole table by design; readers stay lazy.
- **Principle 14**: the asset and `live_forecasts` couple through the cleaned table at rest.
- **Principle 15**: the raw table is untouched, so changing a rule means re-running one asset over
  data already on disk, never re-downloading.

## Tests

- `packages/contracts/tests`: `CleanedPowerTimeSeries.validate` accepts a null `drop_reason`,
  rejects an unknown string, rejects a duplicate key, and rejects unsorted rows. `test_settings.py`
  gains a case for the derived `cleaned_power_time_series_data_path`.
- `packages/nged_data/tests/test_cleaning.py`: `flag_nged_power` returns every input row with a null
  `drop_reason`. Fails on `main`, where the function does not exist.
- `packages/nged_data/tests/test_storage.py`: `scan_cleaned_power` drops flagged rows and returns
  the `PowerTimeSeries` columns only. The existing `time_series_coverage` tests stay green, which is
  the evidence the coverage split is a pure refactor.
- `packages/delta_store/tests`: the writer overwrites rather than appends (two writes; only the
  second write's rows remain), read back through `pl.scan_delta` so a dtype that breaks Delta reads
  fails the test; and with `retention_hours=0` the superseded parquet files are gone after the
  second write, which fails if `dry_run=False` is missing. The `raw_version` is readable from the
  newest `WRITE` commit after the vacuum's two commits.
- `tests/test_cleaning_assets.py` (new):
    - the asset writes every raw row with null reasons and reports `n_rows`;
    - with `flag_nged_power` monkeypatched to flag some rows, the per-reason metadata counts and
      min/max are right;
    - an absent raw table yields `n_rows: 0` and no exception;
    - every assertion reads the written table back through `pl.scan_delta`.
    - a vacuum that raises still leaves the run successful, with `vacuum_failed: True`.
- `tests/test_checks.py`: `cleaned_power_keeps_up_with_raw` passes at a lag of 0 and 1 raw commits,
  warns at 3, warns when the cleaned table or its `raw_version` key is absent, and degrades rather
  than raising when a table is unreadable.
- `ml_core` tests: the new provenance lookup returns the newest `WRITE` commit's `raw_version`, and
  `ABSENT` for a missing table or key.
- Readers: one test per read path proves flagged rows are excluded — `effective_capacity`,
  `eligible_time_series` (flagging series 1, the only series eligible for the fixture's `FOLD_ID`),
  `metrics` (a flagged actual is not scored), and `live_forecasts` (a spy on
  `build_live_power_frame`, because the test model's weather-only features forecast identically with
  or without power). `trained_cv_model` and `cv_power_forecasts` share `load_engineering_inputs`
  with `live_forecasts`, so the live test covers that path.
- **Fixture churn**: `test_cv_assets.py`, `test_trained_cv_model.py`, `test_cv_power_forecasts.py`,
  `test_live_forecasts.py`, and `test_metrics.py` (which materialises `effective_capacity`) write a
  raw power table; each must also write the cleaned table. A shared helper in `tests/` writes an
  all-unflagged cleaned copy of a raw fixture.

## Docs to update

- `docs/roadmap/data-cleaning.md` — where rules go (`flag_nged_power`, `DROP_REASONS`), the
  contract a rule keeps, and that a failed cleaning run leaves readers on the last good table.
- `docs/design-philosophy/inherent-stability.md` — the degradation table gains the cleaned-table
  hop.
- `docs/live_service/operations.md` — what an operator does when `clean_nged_power_data` fails or
  `cleaned_power_keeps_up_with_raw` warns, and a deploy step: materialise `clean_nged_power_data`
  once before the first `live_forecasts` slot, because until then the cleaned table does not exist
  and the slot fails reading it.
- `docs/architecture/overview.md` — the new production asset (required by
  `test_asset_layer_tags.py`).
- `ml_core/repro.py`'s module docstring, which promises replay with `scan_delta(version=N)`.
- `packages/nged_data/README.md`, `packages/delta_store/README.md`, and
  `packages/contracts/README.md` — the new functions and contract.
- `CLAUDE.md` — the Dagster assets list and the contracts list.
- The docstrings of every touched asset, which are operator docs in the Dagster UI.

## Verification commands

The `implement-issue` green-before-push set (`ruff check`, `ruff format`, `ty check`, `pytest`, and
the `pymarkdown` scan), plus `pydoclint` and the docs-link checker as CI runs them, plus `uv run
mkdocs build --strict`, plus one local materialisation of `clean_nged_power_data` against the real
V1 data to check the run time and the on-disk size after vacuum.

## Risks and open questions

1. **V2 rewrite cost.** A few GB per run, 24 runs a day. Recommendation: measure before V2; the
   append issue is the answer if the rewrite is too slow.

## Review log

Earlier versions of this plan cleaned power at read time, first with a list of rule functions and
then with per-step logging functions. Each version had a simplicity review and a correctness review;
the git history of this file holds them. The maintainer then chose a stored cleaned table, for the
reasons in "Why the whole cleaned table is written eagerly". Findings from those reviews that carry
over are already folded in: the coverage split, the `delta_table_exists` check in
`eligible_time_series`, the series-1 eligibility fixture, and the `build_live_power_frame` spy.

### Review of the stored-table plan (fresh Opus sub-agent, with experiments on a copy of V1 data)

Accepted:

- `drop_reason` is a `String` with an `is_in` constraint, not a `pl.Enum`: an Enum written to Delta
  makes every read raise. `DROP_REASONS` therefore starts empty, and the open question about seeding
  four names is gone.
- `vacuum` needs `enforce_retention_duration=False` below 168 hours; the retention is now 2 hours,
  and the plan states what the retention bounds.
- Sort before `validate`, which rejects the real V1 data in Delta file order.
- Missing files added: `definitions.py`, `test_asset_layer_tags.py`,
  `docs/architecture/overview.md`, `ml_core/repro.py`, `test_metrics.py`, and `test_settings.py`.
- Provenance stamps the raw version the cleaning read, recorded in the cleaned commit's metadata,
  because vacuumed cleaned versions are not replayable.
- The roster is read with `allow_superfluous_columns=True`, and the WARN check comparing the two
  tables moves from a follow-up into this PR, because a bad roster now makes live power go stale
  with nothing else to report it.
- A first-deploy step in `operations.md`.
- The writer and asset tests read back through `pl.scan_delta`.

Not acted on: skipping the rewrite in hours when ingest found no new data (0.07 s at V1).

### Review of the fixes (fresh Opus sub-agent, with experiments)

Accepted, all ten findings:

- The vacuum call needs `dry_run=False`; without it nothing is deleted. The retention is now a
  writer keyword so a test can pass 0 and assert the old files are gone.
- The provenance stamp reads the newest `WRITE` commit, not the latest commit (the vacuum adds two),
  uses a new key rather than redefining `power_time_series`, and maps a missing table or key to
  `ABSENT`. The claim that the stamp stops being one version ahead was false and is gone.
- The keeping-up check counts raw commits, not hours of reading time, which avoids a false alarm
  after an NGED backlog, and is registered in `definitions.py`'s `asset_checks` list.
- The monkeypatched-`DROP_REASONS` test case is dropped: Patito captures the tuple when the class is
  defined.
- The table on disk is not sorted within a partition; the plan now says so.
- A vacuum failure degrades rather than failing a run whose table is already written.
- `delta_store/__init__.py` and the typed null literal are named.

### Maintainer decisions after that review

- This PR edits `defs/schedules.py` itself; the maintainer will merge it before work on #1020
  starts, and a comment on #1020 explains the edit.
- Every reader except the ingest and its freshness check uses cleaned power, including `metrics`
  and the two dashboards.
