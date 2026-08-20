# Move the four remaining Delta writes into `delta_store` (#637)

## Problem

`delta_store` exists so a Dagster asset writes a table's physical format once, in one place,
rather than every caller open-coding `write_deltalake` settings. Only two of the six tables go
through it today (`nwp`, `power_forecasts`). The other four — `power_time_series`,
`eligible_time_series`, `effective_capacity`, `forecast_metrics` — are written directly from
`src/nged_substation_forecast/defs/`, each with its own ad-hoc call and no `delta_store` module.
`forecast_metrics`'s write is the sharpest case: its Enum→String cast is exactly the kind of
on-disk-format knowledge `delta_store` exists to hold, and it currently lives in a private
Dagster-module helper instead.

## Solution

Add one `delta_store` module per remaining table — `power_time_series.py`,
`eligible_time_series.py`, `effective_capacity.py`, `forecast_metrics.py` — each exporting a
`write_*` function that wraps `deltalake.write_deltalake` with exactly the settings the current
call site already uses. Move `_write_metrics_to_delta`'s Enum→String cast into
`forecast_metrics.write_forecast_metrics` unchanged. Update the four call sites in `defs/` to
import and call the new functions. This is a **pure move**: no call site's `mode`, `predicate`,
`partition_by` or `storage_options` behaviour changes, so the bytes each table produces today are
byte-for-byte what it produces after. No new `WriterProperties`, sort order or significand
rounding is added for these four tables in this issue — see "Departures from the issue" below.

## Verdict, size, departures

**Verdict: worth it, as described.** The organisational problem is real and verified against the
code (see file/line list below); the fix the issue proposes — one module per table, matching the
`nwp.py`/`power_forecasts.py` shape — is the obvious one and needs no re-scoping.

**Size: complex**, per the issue's own sizing (confirmed against `CLAUDE.md`'s rule: this touches
what gets stored, for four tables, in the production ingest path and three research assets). Gets
the plan and all four adversarial reviews (this plan: simplicity then correctness/testability;
diff: correctness/cut-it-down then mutation-testing).

**Departure from the issue body:** the issue's "practical consequence" paragraph frames this as
unlocking compression work that has "gone into" `delta_store`'s writer properties/sort
orders/rounding for the other four tables. This plan does not add any of that for
`power_time_series`, `eligible_time_series` or `effective_capacity` — the issue's own fix
paragraph asks only to "move each table's write logic and physical-format knowledge… into it",
and for these three tables there is no physical-format knowledge today (no rounding, no custom
`WriterProperties`) to move; inventing new tuning here would be undocumented, unmeasured
guesswork of exactly the kind `nwp.py`'s docstring warns against (the `BYTE_STREAM_SPLIT` choice
that helps `power_forecasts` measured *worse* for `nwp`). `forecast_metrics.metric_value` is a
`Float32` column that could plausibly benefit from the same rounding treatment, but doing that
without a measurement is the same guesswork. Recommendation in "Risks and open questions" below:
file a follow-up issue for profiling these four tables' compression, once real data volume exists
to measure against.

## What changes, file by file

### New: `packages/delta_store/src/delta_store/power_time_series.py`

```python
def write_power_time_series(
    power_ts: pt.DataFrame[PowerTimeSeries],
    table_uri: str | Path,
    storage_options: ObjectStoreOptions | None = None,
) -> None: ...
```

Replaces `assets.py:192`'s `new_power_ts_deduped.write_delta(delta_path, mode="append",
storage_options=typeddict_to_dict(storage_options), delta_write_options={"partition_by":
"time_series_id"})`. Ports the polars `.write_delta()` call to `deltalake.write_deltalake` (arrow
input, `mode="append"`, `partition_by=["time_series_id"]`) to match the `nwp.py`/
`power_forecasts.py` pattern the issue asks for — functionally identical to the polars method,
confirmed by reading the installed `polars` source: `pl.DataFrame.write_delta`'s non-merge branch
is exactly `write_deltalake(table_or_uri=target, data=self, mode=mode,
storage_options=storage_options, **delta_write_options)`, so today's
`delta_write_options={"partition_by": "time_series_id"}` is already just
`write_deltalake(partition_by="time_series_id")` (a bare string, not a list — `write_deltalake`
accepts both; the new module passes `["time_series_id"]` to match the list form the other three
modules use, which is cosmetic).

### New: `packages/delta_store/src/delta_store/eligible_time_series.py`

```python
def write_eligible_time_series(
    eligible: pt.DataFrame[EligibleTimeSeries],
    table_uri: str | Path,
    fold_id: str,
    storage_options: ObjectStoreOptions | None = None,
) -> None: ...
```

Replaces `cv_assets.py:144`. `fold_id` is an explicit parameter rather than read off the frame
(`eligible.item(0, "fold_id")`) for the same reason `write_power_forecasts` takes
`replace_partition` explicitly: a fold with zero eligible series must still clear its Delta
partition, and an empty frame has no row to read a fold id from.

### New: `packages/delta_store/src/delta_store/effective_capacity.py`

```python
def write_effective_capacity(
    capacity: pt.DataFrame[EffectiveCapacity],
    table_uri: str | Path,
    storage_options: ObjectStoreOptions | None = None,
) -> None: ...
```

Replaces `cv_assets.py:195`. Whole-table overwrite, no partition/predicate — matches the current
call exactly (this table has no `partition_by` today; not adding one here is in scope for the
same "pure move" reason as above).

### New: `packages/delta_store/src/delta_store/forecast_metrics.py`

```python
def write_forecast_metrics(
    metrics: pt.DataFrame[Metrics],
    table_uri: str | Path,
    experiment_name: str,
    fold_id: str,
    storage_options: ObjectStoreOptions | None = None,
) -> None: ...
```

Moves the whole body of `cv_assets.py`'s `_write_metrics_to_delta` (~line 691-724) verbatim,
including its docstring's Enum→String rationale and the `pl.Enum` cast loop. `cv_assets.py` loses
this private helper; its single call site (~line 895) becomes
`write_forecast_metrics(metrics_path, enriched, exp_name, fold_id, storage_options)` →
`write_forecast_metrics(enriched, metrics_path, exp_name, fold_id, storage_options)` (positional
argument order matches the new module's signature, table_uri second like the other three).

### `packages/delta_store/src/delta_store/__init__.py`

Add the four new `write_*` imports and `__all__` entries; update the module docstring's "One
module per table (currently `power_forecasts` and `nwp`)" line to name all six.

### `packages/delta_store/README.md`

Add a "Contents" bullet for each of the four new modules (and for `nwp`, which the file omits
today — a pre-existing gap, fixed here since this plan touches the same "Contents" list).

### `src/nged_substation_forecast/defs/assets.py`

`power_time_series_and_metadata` (~line 189-197): replace the `.write_delta(...)` call with
`write_power_time_series(new_power_ts_deduped, delta_path, storage_options=storage_options)`,
keeping the surrounding `if not new_power_ts_deduped.is_empty(): if_local_path_then_make_parent_dir(...)`
guard exactly as-is (unchanged control flow, so the asset's retry/degradation boundaries — the
outer `try/except` ends before this write, at line 158 — are untouched). Add
`from delta_store.power_time_series import write_power_time_series` to the import block
(~line 43, alongside the existing `from delta_store.nwp import write_nwp`).

### `src/nged_substation_forecast/defs/cv_assets.py`

- `eligible_time_series` (~line 111-151): replace the `write_deltalake(...)` call with
  `write_eligible_time_series(eligible_df, settings.eligible_time_series_data_path, fold_id,
  storage_options=storage_options)`.
- `effective_capacity` (~line 167-200): replace the `write_deltalake(...)` call with
  `write_effective_capacity(capacity_df, settings.effective_capacity_data_path,
  storage_options=storage_options)`.
- `_write_metrics_to_delta` (~line 691-724): delete the function; its one call site (~line 895)
  calls `write_forecast_metrics` from `delta_store` instead.
- Import block (~line 38): add the three new `delta_store` imports; drop `from deltalake import
  write_deltalake` if nothing else in the file still calls it (confirm at implementation time —
  `trained_cv_model` and `cv_power_forecasts`, owned by the concurrent #638 session, do not write
  Delta directly as far as this plan's read of the file shows, so the import likely becomes
  unused and should be removed, not left dangling).

**Territory check (wave fencing):** this plan touches `eligible_time_series`,
`effective_capacity` and `_write_metrics_to_delta`/its call site in `cv_assets.py`, and the
`power_time_series_and_metadata` asset body in `assets.py` — none of which overlaps #638's
territory (`trained_cv_model`, `cv_power_forecasts`) or #639's (the `_BaseSummary`/
`_FileListingSummary`/`_PowerTimeSeriesSummary` classes). No spill expected; if the `write_deltalake`
import removal in `cv_assets.py` turns out to still be needed by #638's functions, leave the
import in place and note it rather than editing those functions.

## Design-philosophy check

This is a mechanical extraction with no behaviour change: every `write_*` function is a direct
port of the write call it replaces, with the same `mode`, `predicate`, `partition_by` and
`storage_options`, and (matching `write_nwp`/`write_power_forecasts`) no internal `try/except` —
whatever the underlying `write_deltalake` raises still propagates to the caller exactly as before.

- `power_time_series_and_metadata` runs in production (`PRODUCTION_LAYER_TAGS`). Its write call
  sits **outside** the asset's NGED-S3 retry guard (that `try/except` ends at line 158, well
  before the write at line 189) precisely so a write fault is treated as ours, not the outside
  world's — this plan does not move the write inside that guard or add a new one. No degradation
  path applies to a write failure here; it should keep raising and failing the Dagster run, which
  is what moving the call unchanged preserves.
- `eligible_time_series`, `effective_capacity` and `forecast_metrics` run in R&D
  (`RESEARCH_LAYER_TAGS`), where CLAUDE.md's rule is fail-fast, not degrade — again, no change to
  raise-vs-degrade behaviour, since nothing here adds error handling.
- No asset check is added or edited by this plan.
- No engineering hypothesis (`H*`/`T*`) is targeted; this is an internal code-organisation change
  with no observable behaviour difference, so there is nothing to falsify.
- No `design-principles.md` principle is traded away — if anything this plan strengthens the
  "explicit is better" boundary already established by `nwp.py`/`power_forecasts.py`, by making
  `delta_store` genuinely own all six tables as its own docstring already claims.

## Tests

New `packages/delta_store/tests/` files, one per new module, in the same style as
`test_nwp.py`/`test_power_forecasts.py` — small synthetic frames written to a `tmp_path` Delta
table, assertions against the real on-disk result:

- `test_power_time_series.py`: `test_append_partitions_by_time_series_id` — write two frames with
  different `time_series_id`s, assert the table has two `time_series_id=…` Hive partition
  directories and both frames' rows survive (append semantics). Fails on `main` today only in the
  sense that `write_power_time_series` doesn't exist yet — this pins the ported behaviour so a
  future accidental `mode="overwrite"` regression is caught.
- `test_eligible_time_series.py`: `test_overwrite_is_partition_scoped_by_fold_id` — write fold A,
  then fold B, then re-write fold A with different rows; assert fold B's rows are untouched and
  fold A's rows are the new set, not the union (mirrors `test_eligible_time_series_overwrite_is_
  partition_scoped` already in `tests/test_cv_assets.py`, but exercising the new function
  directly rather than through the Dagster asset). `test_empty_frame_still_clears_partition` —
  write fold A with rows, then call `write_eligible_time_series` again with an empty frame and the
  same `fold_id`; assert the partition is now empty (this is exactly why `fold_id` is a required
  parameter rather than read off the frame — a test that only ever passes a non-empty frame would
  not catch a regression to `eligible.item(0, "fold_id")`).
- `test_effective_capacity.py`: `test_overwrite_replaces_whole_table` — write once, write again
  with a different row set, assert the table holds only the second write's rows (no
  `partition_by`, so this is a whole-table overwrite — pins that there is no accidental
  partitioning added).
- `test_forecast_metrics.py`: `test_enum_columns_cast_to_string_on_disk` — write a `Metrics` frame
  with `metric_name`/`horizon_slice` populated (`pl.Enum` in the Patito model), read the Parquet
  back with `pyarrow.parquet` directly (not through `Metrics.scan_delta`, which would re-cast) and
  assert the physical column type is `string`, not dictionary-encoded — this is the property
  `_write_metrics_to_delta`'s docstring exists to guarantee, and there is no existing test for it
  today (grep of `tests/test_metrics.py` and `tests/test_cv_assets.py` finds none). `
  test_overwrite_is_partition_scoped_by_experiment_and_fold` — same shape as the
  `eligible_time_series` partition-scoping test, for the `(experiment_name, fold_id)` predicate.

Existing tests that exercise these write paths **indirectly**, through the real Dagster assets —
`test_eligible_time_series_overwrite_is_partition_scoped`,
`test_effective_capacity_is_idempotent`, `test_metrics_leaderboard_writes_forecast_metrics_delta`,
`test_power_time_series_and_metadata_ingests_and_writes`, and others in `tests/test_cv_assets.py`,
`tests/test_metrics.py`, `tests/test_assets.py` — are left unchanged. Because this plan preserves
behaviour exactly, they should stay green with no edits; if any fails, that is a signal the "pure
move" claim above is wrong somewhere and the diff needs re-checking against this plan, not the
test relaxed to fit.

## Docs to update

- `packages/delta_store/README.md` — "Contents" section, as described above.
- `packages/delta_store/src/delta_store/__init__.py` module docstring — table count, as described
  above.
- No `docs/roadmap/` page references this issue directly (checked: the only `delta_store` hit
  outside `docs/architecture/` is `docs/roadmap/xgboost-improvements.md:95`, which cites
  `delta_store.nwp.write_nwp` specifically and is unaffected); no roadmap status banner to update.
- `docs/architecture/overview.md:20` and `docs/architecture/performance.md` describe `delta_store`
  owning "each table's physical layout" and detail the `nwp`/`power_forecasts` formats
  specifically — neither makes a "only two tables" claim that this plan would falsify, so no edit
  needed there (checked by grep across `docs/architecture/`).

## Verification commands

Standard green-before-push set: `uv run ruff check .`, `uv run ruff format .`, `uv run
--all-packages ty check`, `uv run pytest`. Add `uv run pymarkdown scan -r docs README.md
CLAUDE.md packages/*/README.md` since `packages/delta_store/README.md` is touched. No network
tests apply (no NWP conversion or S3 code touched); no `mkdocs build` needed (no link changes, just
a package README table addition).

## Risks and open questions

1. **Should `power_time_series`, `eligible_time_series`, `effective_capacity` and
   `forecast_metrics` get their own writer-properties/rounding tuning, now or as a follow-up?**
   This plan's recommendation: follow-up issue, after implementation, once there's real production
   volume to measure against — `power_forecasts` and `nwp`'s tuning was each backed by a measured
   before/after on real data (6.33 GB → 0.73 GB; a specific row-group pruning benchmark), and
   guessing at settings without that measurement risks repeating the `nwp` module's own cautionary
   tale (`BYTE_STREAM_SPLIT` helped one table and hurt the other). Flag for human confirmation.
2. **The `write_deltalake` import removal in `cv_assets.py`**: `grep -n write_deltalake
   cv_assets.py` finds exactly the three call sites this plan removes and nothing else, so the
   import should become unused and get removed. #638 is editing `trained_cv_model`/
   `cv_power_forecasts` concurrently in the same file; the implementer should re-run that grep at
   diff time in case #638 lands a fourth call site first, rather than trusting this snapshot.
3. **`write_power_time_series`'s move from polars' `.write_delta()` to `deltalake.write_deltalake`
   is confirmed a mechanical API swap, not a behaviour change** — verified by reading the
   installed `polars` package's `DataFrame.write_delta` source directly (see the module section
   above): its non-merge branch calls `write_deltalake` with the exact same `mode`,
   `storage_options` and `delta_write_options` this plan ports across. Still worth the
   regression test (`test_append_partitions_by_time_series_id` above) as a durable guard, not as
   a substitute for this verification.
