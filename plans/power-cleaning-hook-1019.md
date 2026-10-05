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
The power readers switch from the raw table to the cleaned table's unflagged rows. The raw table is
never modified.

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

- `DROP_REASONS: Final[tuple[str, ...]]` — the vocabulary of drop reasons. An empty `pl.Enum` is
  unusable, so the tuple starts with the four planned rules' names
  (`"substation_first_three_months"`, `"substation_zero"`, `"implausible_value"`, and
  `"generator_spurious_zero"`), and the docstring says a rule's author adds a value here with the
  rule.
- `CleanedPowerTimeSeries(PowerTimeSeries)` — the same three fields plus `drop_reason: str | None`
  with `dtype=pl.Enum(DROP_REASONS)`. Subclassing should keep `PowerTimeSeries.validate`'s
  datetime-bound and key-uniqueness checks; confirm that while implementing.

### `packages/nged_data/src/nged_data/cleaning.py` (new)

- `flag_nged_power(power: pt.LazyFrame[PowerTimeSeries], metadata:
  pt.DataFrame[TimeSeriesMetadata]) -> pt.LazyFrame[CleanedPowerTimeSeries]` — the function the
  rules go into. `power` is always the whole raw table. Today the function adds a null `drop_reason`
  column and nothing else. The docstring states the contract a rule must keep: return every input
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
  storage_options)` — `write_deltalake(mode="overwrite", partition_by=["time_series_id"])`, one
  atomic Delta commit (principle 10), then `vacuum` with a 24-hour retention. Without the vacuum
  every hourly overwrite leaves the previous copy on disk: 24 copies a day, about 0.9 GB a day at
  V1 and tens of GB a day at V2.

### `packages/contracts/src/contracts/settings.py`

- `cleaned_power_time_series_data_path`, derived beside `power_time_series_data_path` in
  `_derive_unset_paths`.

### `src/nged_substation_forecast/defs/cleaning_assets.py` (new)

`defs/assets.py` belongs to #1020, so the asset gets its own module.

- `clean_nged_power_data` — `deps=["power_time_series_and_metadata"]`, production-layer tags. A
  minimal wrapper:
    - If the raw table does not exist yet, log and return with `n_rows: 0` metadata rather than
      raising, because an absent input degrades (inherent-stability).
    - Scan the raw table, read the whole roster, call `flag_nged_power`, collect with the streaming
      engine, `CleanedPowerTimeSeries.validate`, write.
    - Metadata: `n_rows`, `n_rows_kept`, `n_time_series`, and per drop reason
      `drop_reason/<reason>/n_rows`, `.../n_time_series`, `.../min_power`, and `.../max_power`,
      computed from the collected frame with one `group_by("drop_reason")`.
    - No `try`: a failing rule or a contract violation is our bug, so the asset raises and the
      readers keep the last good table.
- Registered wherever the asset modules are collected (confirm while implementing).

### `src/nged_substation_forecast/defs/schedules.py` — out of bounds; needs coordination

- Add `"clean_nged_power_data"` to `power_time_series_and_metadata_job`'s selection, so the asset
  runs hourly, straight after ingest and five minutes before each 6-hourly `live_forecasts` slot.
  That is well over the four runs a day the maintainer requires. #1020 owns this file; see open
  question 1.

### Readers

- `_engineering_inputs.py`, `load_engineering_inputs`: read `scan_cleaned_power` instead of the raw
  path. This covers `trained_cv_model`, `cv_power_forecasts`, and `live_forecasts`.
- `cv_assets.py`, `effective_capacity`: read `scan_cleaned_power`.
- `cv_assets.py`, `eligible_time_series`: `coverage_from_power(scan_cleaned_power(...))`, behind a
  `delta_table_exists` check on the cleaned table, so an absent table still yields an empty
  population.
- The `deps` of `eligible_time_series`, `effective_capacity`, `trained_cv_model`, and
  `live_forecasts` change from `power_time_series_and_metadata` to `clean_nged_power_data`.
- The provenance tags `trained_cv_model` and `cv_power_forecasts` write to MLflow (the Delta
  version of `power_time_series`) record the cleaned table's version as well.
- Untouched: `metrics` (owned by #958, still reads raw actuals), `power_data_is_fresh`, the ingest,
  and the two dashboard notebooks.

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
- **No asset check is added.** The stats go to output metadata, and a failed cleaning run alerts
  through the job's Sentry failure hook. A WARN check comparing the cleaned table's latest reading
  with the raw table's would catch a cleaning run that silently stopped running; see open question
  3.
- **Principle 7 (strict contracts)**: a new Patito model with an Enum of drop reasons.
- **Principle 11**: the asset materialises the whole table by design; readers stay lazy.
- **Principle 14**: the asset and `live_forecasts` couple through the cleaned table at rest.
- **Principle 15**: the raw table is untouched, so changing a rule means re-running one asset over
  data already on disk, never re-downloading.

## Tests

- `packages/contracts/tests`: `CleanedPowerTimeSeries.validate` accepts a null and a known
  `drop_reason`, rejects an unknown one, and rejects a duplicate key.
- `packages/nged_data/tests/test_cleaning.py`: `flag_nged_power` returns every input row with a null
  `drop_reason`. Fails on `main`, where the function does not exist.
- `packages/nged_data/tests/test_storage.py`: `scan_cleaned_power` drops flagged rows and returns
  the `PowerTimeSeries` columns only. The existing `time_series_coverage` tests stay green, which is
  the evidence the coverage split is a pure refactor.
- `packages/delta_store/tests`: the writer overwrites rather than appends (two writes; only the
  second write's rows remain).
- `tests/test_cleaning_assets.py` (new):
    - the asset writes every raw row with null reasons and reports `n_rows`;
    - with `flag_nged_power` monkeypatched to flag some rows, the per-reason metadata counts and
      min/max are right;
    - an absent raw table yields `n_rows: 0` and no exception.
- Readers: one test per read path proves flagged rows are excluded — `effective_capacity`,
  `eligible_time_series` (flagging series 1, the only series eligible for the fixture's `FOLD_ID`),
  and `live_forecasts` (a spy on `build_live_power_frame`, because the test model's weather-only
  features forecast identically with or without power). `trained_cv_model` and
  `cv_power_forecasts` share `load_engineering_inputs` with `live_forecasts`, so the live test
  covers that path.
- **Fixture churn**: `test_cv_assets.py`, `test_trained_cv_model.py`, `test_cv_power_forecasts.py`,
  and `test_live_forecasts.py` write a raw power table; each must also write the cleaned table. A
  shared helper in `tests/` writes an all-unflagged cleaned copy of a raw fixture.

## Docs to update

- `docs/roadmap/data-cleaning.md` — where rules go (`flag_nged_power`, `DROP_REASONS`), the
  contract a rule keeps, and that a failed cleaning run leaves readers on the last good table.
- `docs/design-philosophy/inherent-stability.md` — the degradation table gains the cleaned-table
  hop.
- `docs/live_service/operations.md` — what an operator does when `clean_nged_power_data` fails.
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

1. **`defs/schedules.py` belongs to #1020.** Without the one-line selection change, the cleaned
   table is never refreshed in production and `live_forecasts` reads stale power. Recommendation:
   implement everything else now, and add the line after #1020 merges (or ask that session to add
   it). This PR must not merge before the line is in.
2. **Should `metrics` score against cleaned actuals?** Probably yes for rows describing a plant that
   no longer exists, but `metrics` belongs to #958. Recommendation: raise it there.
3. **A WARN check that the cleaned table keeps up with the raw one?** Recommendation: add it in a
   follow-up if the Sentry failure alert proves insufficient.
4. **Starting `DROP_REASONS` with the four planned names** commits names before the rules exist.
   Recommendation: accept; renaming a value is a one-line contract edit plus a re-run.
5. **V2 rewrite cost.** A few GB per run, 24 runs a day. Recommendation: measure before V2; the
   append issue is the answer if the rewrite is too slow.

## Review log

Earlier versions of this plan cleaned power at read time, first with a list of rule functions and
then with per-step logging functions. Each version had a simplicity review and a correctness review;
the git history of this file holds them. The maintainer then chose a stored cleaned table, for the
reasons in "Why the whole cleaned table is written eagerly". Findings from those reviews that carry
over are already folded in: the coverage split, the `delta_table_exists` check in
`eligible_time_series`, the series-1 eligibility fixture, and the `build_live_power_frame` spy.
