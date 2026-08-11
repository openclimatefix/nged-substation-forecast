# Plan — #508: convert the last bare-parquet tables to Delta, and stop an unusable roster wedging the ingest

Issue: <https://github.com/openclimatefix/nged-substation-forecast/issues/508>
Branch: `claude/plan-issue-508-c54a06`

## Verdict: worth implementing — but by converting the storage, not by hardening the parquet path

The bug is real and reproduces. The issue offers three fixes, all of which keep `metadata.parquet` a
bare parquet file. **Jack's decision, after the discussion recorded below: convert the roster and the
H3 grid weights to Delta tables instead, restructure `upsert_metadata` around Delta `MERGE`, keep the
recovery path, and close #508 with that.** After the conversion, no managed table in the project is
a bare parquet file, so there is no bespoke atomic-write machinery to write, own or test.

### What reproduces on `main`

Verified in this worktree:

- A junk `metadata.parquet` makes `pl.read_parquet` raise
  `polars.exceptions.ComputeError: parquet: File out of specification: The file must end with PAR1`
  — exactly the message the issue quotes, from inside `upsert_metadata`
  ([storage.py:405](../packages/nged_data/src/nged_data/storage.py)). A 0-byte file gives the same
  error, which is the shape a full disk produces.
- The corruption mechanism is real and specific to a **local** path: a subprocess writing a large
  frame with `write_parquet` straight to the live path, killed with `SIGKILL` once it had started
  emitting bytes, left a 256 KB file that no longer parses.

There are **three** raisers on that code path, not one, and Delta only removes one of them by itself:

| # | Raiser | Fixed by Delta alone? |
|---|---|---|
| 1 | Torn write from a kill or `ENOSPC` (`storage.py:438`) | **Yes** — a commit is atomic, and delta-rs does the temp-file-then-rename itself. On S3 it commits via conditional put (`docs/live_service/aws.md:84-86`), so concurrent writers are safe too |
| 2 | `TimeSeriesMetadata.validate(existing_metadata)` (`storage.py:408`) raises `DataFrameValidationError` when the stored roster no longer satisfies the contract — verified with a roster missing `substation_type` | No. Delta adds its own schema-enforcement error alongside |
| 3 | `pl.concat([new_metadata, existing_metadata])` (`storage.py:436`) raises `ShapeError` on a width or column-order mismatch | Only via `MERGE`, which this plan uses |

Raiser 3 is live today rather than hypothetical. Four `TimeSeriesMetadata` fields are
`allow_missing=True` (`information`, `area_wkt`, `area_center_lat`, `area_center_lon` —
`packages/contracts/src/contracts/power_schemas.py:245-279`), so a narrower frame validates cleanly
and then `pl.concat` raises `ShapeError: unable to append to a DataFrame of width 14 with a
DataFrame of width 12`; with the same columns reordered it raises `unable to vstack, column names
don't match`. It is reachable because `_extract_time_series_metadata` derives its columns from each
JSON file's own keys (`read_nged_json.py:44-47`), which is exactly why `download_and_parse_files`
has to union them with `how="diagonal"` (`storage.py:198`). The same asymmetry makes the *diff*
unreliable: `hash_rows` is column-order sensitive (verified), so a reordered roster reports every
row as changed and gets rewritten every run.

So **the recovery path is needed whether or not we convert the storage** — raisers 2 and 3 are our
contract boundary and our merge, not the file format. What the conversion buys is that raisers 1 and
3 stop existing at all, and that we own no atomic-write code.

### An unusable roster also breaks `live_forecasts` — out of scope

`_load_engineering_inputs` reads the roster unguarded
(`src/nged_substation_forecast/defs/cv_assets.py:322`), so an unusable roster takes the forecast off
the degradation ladder entirely — not even rung 4. Fixing that means separating the production caller
from the fail-fast CV caller, which is a different change. **Filed separately as [#528](https://github.com/openclimatefix/nged-substation-forecast/issues/528); see D4.**

This is an [H1](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/engineering-hypotheses/#h1-a-service-that-mostly-runs-itself)
bug in the most direct sense. Today's recovery is "someone deletes the file by hand" — a **T1.1**
intervention that is not an upstream format change — and two of the five scripted **T1.4** game-day
failures, "disk full" and "daemon killed", are precisely the events that produce a half-written
roster.

### Departures from the issue body

- **All three of the issue's proposed fixes are superseded.** Option 1 (write atomically) is what
  Delta does for us; options 2 and 3 (rebuild / quarantine) survive in a narrower form, below.
- **Quarantine largely disappears, because the transaction log *is* the quarantine.** For the two
  likely faults — a stored roster that fails our Patito contract, or a Delta schema mismatch —
  recovery is a single overwrite commit, and the previous version stays in the log, readable by time
  travel. Nothing needs copying aside. The one fault that would still need a copy-aside is a
  genuinely corrupt `_delta_log`, and there the recovery is **not** automatic: see D2.
- **The issue says "every id NGED is currently publishing is in `new_metadata`", so "the only loss
  is ids that have gone quiet". That is wrong.** `download_and_parse_files` is fed
  `list_of_new_json_files`, so the snapshot covers only ids whose files were new *this run*. A
  rebuild from it can thin the roster, and a thin roster is not merely lossy — it can make
  `live_forecasts` **fail** (D3). Hence the repair pass below.
- **The asset keeps its current order** (roster upsert, then the power write). A guard around the
  upsert delivers what the issue asks for without reordering.

## What changes, file by file

### `packages/delta_store/` — two new modules

The package already owns physical layout plus the write helpers that apply it
(`packages/delta_store/README.md:6-12`), so both tables get a module each, matching the existing
`nwp.py` / `power_forecasts.py` convention.

- **`metadata.py`** — `METADATA_WRITER_PROPERTIES` (ZSTD-3; no per-column encodings, and the
  docstring should say why: at 14 columns and ~2,500 rows the table is tens of KB, so the
  encoding work that pays for itself on `power_forecasts` and `nwp` buys nothing measurable here)
  and one function:

  `merge_time_series_metadata(snapshot, table_uri, storage_options) -> dict[str, int]` — creates
  the table if `delta_table_exists` is false, otherwise `DeltaTable.merge` on
  `target.time_series_id = source.time_series_id`, with `when_matched_update_all(predicate=…)` and
  `when_not_matched_insert_all()`, returning delta-rs' merge metrics. Everything about *how this
  table is written* lives here; the roster's *semantics* stay in `nged_data` (below).

  Three verified facts belong in its docstring, because each is a trap:

    1. The matched-clause predicate is a chain of `(source.<col> IS DISTINCT FROM target.<col>)`
       over the non-key columns, and **each comparison must be parenthesised**. Without the
       parentheses delta-rs fails with `DeltaError: Generic DeltaTable error: type_coercion … Cannot
       infer common argument type for logical boolean operation Float32 OR Boolean`.
    2. With that predicate, only genuinely-changed rows are rewritten (measured: `updated=1,
       copied=2` for a three-row target with one changed row), and **a merge that changes nothing
       commits no new version at all** (measured: history stayed at 2 versions). So the roster's
       Delta history grows only when NGED's metadata actually changes — which is what keeps the
       vacuum burden negligible for this table.
    3. The predicate is built from the contract's non-key columns, which is safe **only because
       the caller aligns the snapshot to the full schema first** (D1). Handed a snapshot that omits
       an `allow_missing` column, the predicate would reference a column that does not exist — so
       the function asserts the source carries every contract column rather than silently building
       a narrower predicate.

- **`h3_grid_weights.py`** — `H3_GRID_WEIGHTS_WRITER_PROPERTIES` and
  `write_h3_grid_weights(weights, table_uri, storage_options)`, a plain `mode="overwrite"` commit.
  The table is write-once-per-boundary-change and ~30 KB.

### `packages/contracts/src/contracts/`

- **`power_schemas.py`** — add `TimeSeriesMetadata.scan_delta(path, storage_options) ->
  pt.LazyFrame[Self]`, mirroring [`Nwp.scan_delta`](../packages/contracts/src/contracts/weather_schemas.py)
  (`weather_schemas.py:461`) so every roster reader gets a typed, cast scan from one place.
- **`geo_schemas.py`** — the same for `H3GridWeights`.
- **`settings.py`** — the derived defaults become `metadata.delta` (`settings.py:394`) and
  `h3_grid_weights.delta` (`settings.py:378`), matching `power_time_series.delta`. The field *names*
  (`metadata_path`, `h3_grid_weights_path`) do not change — they are format-agnostic already.
- **`_uri.py` gains nothing.** This is the point of the conversion: no `write_parquet_atomically`, no
  `copy_object`, no bespoke IO helpers. `object_exists` gives way to the existing
  `delta_table_exists` at the roster's call sites.

### `packages/nged_data/src/nged_data/storage.py`

`upsert_metadata` keeps its name and its place — the roster's *policy* (which id wins, what counts
as an update, what to do when the stored table is unusable) is NGED-data policy, not physical
layout. `packages/nged_data/pyproject.toml` gains a `delta_store` dependency, exactly as
`ecmwf_ens` already depends on `delta_store.nwp.write_nwp`.

- **Deleted**: the `COMPRESSION` constant (`storage.py:384`), the `pl.read_parquet` +
  `TimeSeriesMetadata.validate(existing_metadata)` pair (405–408) as a *gate*, the `hash_rows` diff
  as a *gate*, and the `concat` + `unique(keep="first")` merge (436). Raisers 1 and 3 go with them.
- **New `_align_to_contract(df) -> pt.DataFrame[TimeSeriesMetadata]`** — selects
  `TimeSeriesMetadata.columns` in declared order, supplying
  `pl.lit(None, dtype=TimeSeriesMetadata.dtypes[name])` for any column the snapshot lacks. Only the
  four `allow_missing=True` fields can be absent from a frame that validated, and all four are
  declared `| None`, so the filled nulls are contract-legal (verified). This is D1's mechanism: it
  is what makes a field NGED stops sending get *cleared* rather than silently retained, which is
  today's behaviour, and it is what lets the merge predicate be a fixed list of columns.
- **`upsert_metadata`'s new shape**: validate the snapshot, align it; if the table does not exist,
  create it; otherwise compute the informational diff (below) and call
  `merge_time_series_metadata`. If the read or the merge raises because the *stored table* is
  unusable — our contract rejects it, or Delta rejects the schema — log at ERROR and rebuild it with
  an overwrite commit (`schema_mode="overwrite"`), reporting the rebuild in the stats. The previous
  version stays in the log, so nothing is destroyed and no file is copied aside.
- **If the rebuild commit *itself* fails, degrade rather than trying harder** (D2): report and let
  the asset-level guard carry the run. This is the discrimination the corrupt-`_delta_log` case
  needs, and doing it by "did the recovery work?" rather than by inspecting exception types is what
  keeps it robust — there is no reliable taxonomy separating "delta-rs cannot open this table" from
  "our contract rejects its contents", and guessing wrong in either direction is worse than trying
  the cheap recovery and reporting when it does not take.
- **The diff is demoted from a gate to an annotation.** `UpsertMetadataStats` currently publishes
  `metadata_updated_TimeSeriesIDs`, a *list* of ids, and delta-rs' merge metrics give counts only —
  verified: `num_target_rows_inserted`, `num_target_rows_updated`, `num_target_rows_copied`, no ids.
  So the read stays, but only to name the changed ids for the Dagster UI, and it runs inside its own
  guard: if it fails, the merge still happens and the stats simply omit the id list. That is a
  strictly better structure than today's, where the same read is load-bearing for correctness.
- **`UpsertMetadataStats`** keeps its existing keys, sourced from the merge metrics
  (`metadata_n_new_TimeSeriesIDs` ← `num_target_rows_inserted`,
  `metadata_n_updated_TimeSeriesIDs` ← `num_target_rows_updated`), and gains two optional ones:
  `metadata_roster_rebuilt_reason: str` and `metadata_upsert_failed: str` (the latter set by the
  asset).
- **`_read_existing_roster`** — reads via `TimeSeriesMetadata.scan_delta` and validates, returning
  `None` plus a reason on `Exception` rather than raising. Catching `Exception` and **not**
  `BaseException` is deliberate, and differs from the `checks.py` guards: a pyo3 panic is not
  evidence about the *table*, and overwriting a table on that evidence is worse than skipping the
  update, so a panic falls through to the asset-level guard, which degrades without rebuilding.

### `src/nged_substation_forecast/defs/assets.py`

- **`power_time_series_and_metadata`** — the `upsert_metadata` call (line 125) goes under a
  `try`/`except BaseException` that re-raises `KeyboardInterrupt | SystemExit |
  DagsterExecutionInterruptedError` (the `checks.py:318-337` idiom, and the same reasoning: a pyo3
  `PanicException` from polars/delta-rs/obstore does not derive from `Exception`, and each compiled
  extension defines its own class). The handler logs the traceback, calls
  `report_asset_degradation("power_time_series_and_metadata", exc)`, and substitutes
  `UpsertMetadataStats(metadata_upsert_failed=repr(exc))` so the power write below still runs. Needs
  a `DagsterExecutionInterruptedError` import.
- **Repair a rebuilt roster in the same run.** Where the stats carry
  `metadata_roster_rebuilt_reason`, log at ERROR, call `report_asset_degradation` with the reason,
  then re-derive the full roster: take the newest JSON file per `time_series_id` from
  `list_of_large_json_files` (the *all-files* listing the asset already holds, before
  `select_new_rows` narrows it), run it back through `download_and_parse_files`, and
  `upsert_metadata` the result — which now merges into the valid table just written. The returned
  power frame is discarded; `select_new_rows` already rejects those rows and re-parsing them is
  CPU-only. The whole block sits in its own guard: if it fails, the thin-but-valid roster stands and
  the run still succeeds, so the repair can only improve on the bare rebuild. Cost is one full pass
  over NGED's files (32 today, ~2,500 at V2 scale — the same work a first-ever backfill does) on a
  path that should fire once in the project's life.
- **The consequence worth a comment**, because it is a genuine trade that holds whichever order the
  two writes go in: the power Delta table is what `select_new_rows` uses to decide which JSON files
  are new, so once the power rows land, a *failed* roster update is not retried — that run's
  metadata change is lost until NGED republishes those series (~5 h). Losing one refresh of derived,
  re-delivered data is much cheaper than blocking the power stream, and it is the second reason the
  failure must reach Sentry rather than only the logs.
- **`h3_grid_weights`** — `weights.write_parquet(...)` (line 167) becomes
  `write_h3_grid_weights(...)`; `if_local_path_then_make_parent_dir` stays (a Delta write needs the
  parent too).
- **`ecmwf_ens`** — the weights read at line 260 becomes `H3GridWeights.scan_delta(...)`.

### Remaining read sites

Mechanical, all from `pl.read_parquet` to the new `scan_delta` classmethods:

- `src/nged_substation_forecast/defs/checks.py:216-223` — `_read_roster_ids`: `object_exists` →
  `delta_table_exists`, `scan_parquet` → `TimeSeriesMetadata.scan_delta`. Still inside the check's
  catch-all, so it cannot raise into the run.
- `src/nged_substation_forecast/defs/cv_assets.py:322` (`_load_engineering_inputs`) and `:998`
  (`forecast_metrics`).
- `packages/dashboard/view_forecasts.py:63` and `packages/dashboard/map_and_timeseries.py:48-50`.
- `packages/notebooks/view_baseline_export.py:42`.

### `src/nged_substation_forecast/_sentry.py`

- **New `report_asset_degradation(asset_name: str, detail: BaseException | str) -> None`** — tags
  `degraded_asset` on a forked scope, sending `capture_exception` for an exception or
  `capture_message(…, level="error")` for a string, never raising, exactly like
  `report_check_degradation` (`_sentry.py:133-161`). The union is what lets one function serve both
  call sites: the asset-level guard has a live exception; the rebuild path has only a reason, because
  the exception was handled a layer down inside `nged_data`, which must not depend on Sentry or
  Dagster. The shared `new_scope`/`try` body is extracted into a module-private helper so there are
  not two copies; `report_check_degradation`'s signature and behaviour are unchanged. See D5.
- `init_sentry`'s docstring enumerates "the three explicit senders" (`_sentry.py:98-99`); it becomes
  four.

### Migrating the two existing files

Both are re-derivable, and there is one deployment, so this is a documented manual step rather than
code: for each table, `pl.read_parquet(<old>).write_delta(<new>)`, verify the row count, then delete
the `.parquet`. The weights can equally be re-materialised from the asset, since they are
deterministic. Put the exact commands in the PR body and in `docs/live_service/operations.md`; do not
add a migration code path (CLAUDE.md: no backwards compatibility with data we can re-derive).

## Design-philosophy check

This path is **production** — the hourly `power_time_series_and_metadata_job`, which carries
`sentry_capture_failure` (`defs/schedules.py:19`) — so it degrades rather than raises.

- **Principle 10** ("every write is atomic and idempotent",
  `docs/design-philosophy/design-principles.md:387`) is the one this change *delivers*. Its *Decided*
  paragraph (`design-principles.md:411-419`) credits Delta Lake for atomicity; the roster and the
  weights were the two live exceptions, and after this change there are none.
- **Rule 1** (never raise because an input is absent or stale): the ingest keeps running and records
  the degradation instead of stopping.
- **Rule 2** (liberal about missing, strict about malformed): an unusable stored roster is still
  rejected at the Patito boundary — never merged, never trusted. What changes is only that rejecting
  it no longer rejects the power data with it, which is the exact boundary the issue draws.
- **Rule 3** (treat detectably-wrong input as missing): the clean instantiation. A roster we cannot
  read or cannot merge is treated as *absent*, routing it into the same create branch a first-ever
  run takes.
- **Rule 7** (a warning path may never fail the thing it warns about): respected and unchanged. No
  asset check is added or edited; `power_data_is_fresh` keeps `WARN`/`blocking=False` and its
  catch-all. The new Sentry helper cannot raise, and both the informational diff and the repair pass
  are individually guarded, so the recovery path contains no raiser.
- **What the asset-level guard trades away, stated rather than glossed:** it wraps the whole
  `upsert_metadata` call, whose first statement validates the *snapshot* (`storage.py:386`), so in
  principle a contract violation — which rule 1 says should raise, being our own bug — is now
  degraded to a Sentry event. In practice the strict-contract boundary on incoming data is untouched:
  `_extract_time_series_metadata` already validates every file's metadata and raises at
  `read_nged_json.py:58`, inside `download_and_parse_files`, which is **outside** this guard, so a
  genuine NGED contract break still fails the run there. What the guard really absorbs is a bug in
  our own upsert code, and absorbing that is the price of the property the issue asks for. Narrowing
  the guard would reintroduce the wedge for anything outside it.
- **Rules 6, 11**: untouched — no new check, no new cross-job run-status dependency.
- **H1**: removes a hand-intervention class from **T1.1** and covers two **T1.4** game-day failures
  (disk full, daemon killed) with automatic recovery.
- **The one property the conversion gives up**: automatic self-heal from a *corrupt store*. On
  parquet, "the file will not parse" and "the file is off-contract" were the same recovery. On Delta,
  a corrupt `_delta_log` cannot be recovered by committing over it, and moving a whole table prefix
  aside on S3 is a list-copy-delete loop — bespoke machinery of exactly the kind this conversion is
  meant to avoid. So that case degrades and pages, with a documented manual step, and Delta makes it
  much less likely than the parquet failure it replaces (atomic commits; data files written before
  the commit that references them). D2 records that as a deliberate choice.

## Tests

Every assertion below fails on `main` today.

`packages/delta_store/tests/test_metadata.py` (new):

1. `test_merge_updates_only_the_rows_that_changed` — three-row table, snapshot with one changed row.
   Asserts `num_target_rows_updated == 1` and `num_target_rows_copied == 2`. On `main`: the function
   does not exist, and today's `unique(keep="first")` rewrites the whole table.
2. `test_merge_commits_no_new_version_when_nothing_changed` — merge an identical snapshot; assert
   `DeltaTable(uri).history()` is unchanged in length. This is the property that keeps the vacuum
   burden negligible, and it is the one a careless `when_matched_update_all()` (no predicate) would
   silently break — so it is also a regression test on the predicate.
3. `test_merge_accepts_a_snapshot_missing_the_optional_columns` — a snapshot without
   `information`/`area_wkt`/`area_center_*` against a full-schema table. Asserts the insert succeeds
   and the new row's absent fields are null. On `main`: `ShapeError` from the `concat` (verified).
4. `test_merge_creates_the_table_when_absent` — first-ever call against a non-existent URI.

`packages/nged_data/tests/test_storage.py` (rewritten for Delta):

5. `test_upsert_metadata_rebuilds_when_the_stored_roster_fails_its_contract` — write a Delta table
   missing `substation_type` (a required field), then upsert. Asserts a rebuild rather than a
   `DataFrameValidationError`, that the stats carry `metadata_roster_rebuilt_reason`, and that the
   **pre-rebuild version is still readable via time travel** — the property that replaces the
   quarantine copy. On `main`: `DataFrameValidationError` (verified).
6. `test_upsert_metadata_reports_the_changed_ids` — unchanged behaviour, pinned because the diff
   moves from gate to annotation: assert `metadata_updated_TimeSeriesIDs` still names exactly the
   changed ids. Fails on `main` only in that the table is parquet there; it is the guard against the
   demotion quietly dropping the field.
7. `test_upsert_metadata_still_merges_when_the_informational_diff_fails` — monkeypatch the diff read
   to raise; assert the merge happened and the stats omit the id list without raising. On `main` the
   equivalent read is load-bearing, so a failure loses the write entirely.
8. `test_upsert_metadata_merges_a_stored_roster_with_reordered_columns` — same data, different
   column order. Asserts a no-op merge and no new version. On `main`: `hash_rows` is order-sensitive
   (verified), so it reports a spurious update or raises `ShapeError` on the vstack.
9. `test_upsert_metadata_clears_a_field_the_snapshot_no_longer_carries` — the D1 semantics, and the
   one test that would catch (B) being implemented by accident. A stored row has
   `information = "note"`; this run's snapshot omits the `information` column entirely. Asserts the
   stored value ends up **null**, not `"note"`. On `main`: `ShapeError` from the `concat` (verified).
   Without `_align_to_contract`, `when_matched_update_all` leaves the old value in place — verified,
   and it is silent.

`tests/test_s3_data_paths.py` (moto, `integration`-marked; the per-test reset fixture at
`tests/test_s3_data_paths.py:85-101` handles moto's process-global backend):

10. `test_metadata_delta_round_trip_over_s3` — replaces the existing
   `test_metadata_parquet_round_trip_over_s3` (`:237`): create, merge, read back over S3, exercising
   delta-rs' conditional-put commit path against the object store.

`tests/test_assets.py` (the existing `_FakeS3Store` + `env` harness):

11. `test_power_time_series_and_metadata_writes_power_when_the_roster_upsert_fails` — monkeypatch
    `assets.upsert_metadata` to raise `RuntimeError`; assert `result.success`, that the
    `power_time_series` Delta table holds the fixture rows, and that `metadata_upsert_failed` appears
    in the materialisation metadata. This is the issue's headline property. On `main`: the run fails.
12. `test_power_time_series_and_metadata_repairs_a_rebuilt_roster` — with the fake store serving two
    series, ingest once, then make the stored roster off-contract and arrange for only *one* series
    to have a new file. Asserts success and that the roster ends up holding **both** ids — the repair
    ran and the roster was not left thin — plus `metadata_roster_rebuilt_reason` in the metadata.
13. `test_power_time_series_and_metadata_survives_a_failed_roster_repair` — as above with the
    repair's `download_and_parse_files` raising; assert the run still succeeds and the thin roster
    stands.
14. `test_h3_grid_weights_materialises_and_writes_a_delta_table` — the existing
    `test_h3_grid_weights_materialises_and_writes_parquet` (`:236`) converted, asserting
    `delta_table_exists` and the row count. Its six sibling `_write_h3_grid_weights` fixture calls
    (`:275`–`:496`) move to the Delta writer.

`tests/test_sentry.py`:

15. Mirror the two existing `report_check_degradation` tests (`:176`, `:220`) for
    `report_asset_degradation` — the built event carries `{"degraded_asset": …}`, no tag leaks into
    the current or isolation scope, and a raising `capture_*` is swallowed and logged. Add the
    message-shaped call as a third case. On `main`: the function does not exist.

`tests/test_checks.py`:

16. `test_power_data_is_fresh_degrades_on_a_corrupt_metadata_parquet` — rename, and rewrite both the
    corruption it sets up (a Delta table, not junk bytes at a file path) and its docstring, which
    currently states that "`upsert_metadata` reads the same file first and fails the asset outright"
    (line ~336) — no longer true.

`packages/contracts/tests/test_settings.py`:

17. `:49` asserts the derived `h3_grid_weights_path` ends in `.parquet`; update, along with the
    path-suffix set at `:82`.

## Docs to update

Written to describe how the code works now, per CLAUDE.md's "write about the present".

- **`CLAUDE.md:194`** — the asset "upserts metadata parquet" becomes a Delta upsert. Check the
  package table at `:179` still reads correctly for `delta_store` (it will, but it now owns four
  tables).
- **`docs/design-philosophy/design-principles.md:411-419`** — principle 10's *Decided* paragraph:
  every managed table is now Delta, so the property holds project-wide with no exception.
- **`docs/design-philosophy/inherent-stability.md`** — a sentence in "Missing versus wrong" giving
  the roster as the second worked example of rule 3, and a new **rule 12**: *a derived artifact we
  cannot read is absent, not fatal — rebuild it and keep going.* Appended, so nothing is renumbered.
  Note the wording deliberately drops any atomicity clause: atomicity is principle 10's job and is
  now delivered by the store, so rule 12 carries only the part rules 1–11 do not imply and needs no
  "restates a principle" marking.
- **`docs/architecture/production-deployment.md`** — three places: the "half-written
  `metadata.parquet`" example at `:86` (a state that no longer exists), the Sentry section at
  `:174-179` whose "the only fault the hook cannot see is a standalone `@asset_check` … which, by
  design, is both of them" is falsified by an *asset* now catching its own exception (and by the new
  `degraded_asset` tag), and a short new subsection for the roster's policy: a Delta table, merged
  rather than rewritten, rebuilt in place if it fails our contract, with the power write no longer
  coupled to any of it.
- **`src/nged_substation_forecast/defs/checks.py:311-313`** — `power_data_is_fresh`'s own docstring
  repeats the "half-written `metadata.parquet`" example; fix it with the other two.
- **`docs/live_service/operations.md`** — the same phrase at `:171`, plus a new operator paragraph:
  what `metadata_roster_rebuilt_reason` / `metadata_upsert_failed` mean, the
  `degraded_asset:power_time_series_and_metadata` Sentry filter, how to read the pre-rebuild roster
  back with time travel, the manual step for a corrupt `_delta_log` (D2), and the two migration
  commands. Also tell the operator to append to the intervention log, as the rest of that section
  does.
- **`docs/live_service/setup.md:69`** — the derived default is now `metadata.delta`.
- **`docs/ml_experimentation/dagster-workflow.md:17,26`** — both name a `.parquet` artifact.
- **`packages/nged_data/README.md`** — the `upsert_metadata` line gains the Delta merge and the
  rebuild.
- **`packages/delta_store/README.md`** — two new modules in the charter and the table list.
- **`packages/contracts/README.md`** — if it lists the schemas' helpers, the two new `scan_delta`
  classmethods belong there.
- No roadmap page completes here, so there is no "Implementation details" section to delete and no
  status banner to move. `#508` is referenced from no `docs/` page.

## Verification commands

```bash
uv run ruff format . && uv run ruff check . && uv run --all-packages ty check && uv run pytest
```

```bash
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
```

```bash
uv run mkdocs build --strict
```

Plus, specific to this change:

```bash
uv run pytest tests/test_s3_data_paths.py tests/test_assets.py tests/test_checks.py tests/test_sentry.py packages/delta_store/tests packages/nged_data/tests packages/contracts/tests -q
```

The marimo notebook check matters here because two dashboards change:

```bash
uv run pytest tests/test_marimo_notebooks.py -q
```

`mkdocs build --strict` earns its place because eight prose files change — read the rendered HTML
for the edited sections, since the nested-list and wrapped-link traps pass both linters
(`mkdocs-authoring` skill). No `--run-network` run is needed: nothing touches the Dynamical catalog
or NWP conversion conventions.

## Decisions, and remaining risks

Jack approved all five recommendations below; they are recorded as decisions rather than questions so
the implementer does not have to re-derive them, and so a reviewer can see what was weighed.

**D1 — A field the snapshot no longer carries is cleared, not retained.** The two options differed
only in which columns the merge source carries. **(A, chosen) Align the snapshot to all 14 columns**,
typed nulls for absent ones: a field NGED stops sending is cleared, which is exactly today's
`unique(keep="first")` semantics, so the conversion stays a storage change rather than a semantics
change and the roster keeps meaning "the latest snapshot of what NGED published". **(B, rejected)
Merge only the columns the snapshot has**: the stored value survives, so the roster mixes current and
stale field values in one row and a field NGED genuinely clears sticks around forever. The mechanism
is `_align_to_contract`, and it needs a *test* rather than a comment, because (B) is what Delta does
by default — `when_matched_update_all` on a narrow source silently implements it (verified). Test 9
is that test.

**D2 — A corrupt `_delta_log` degrades and pages; it does not self-heal.** Recovering it would mean
moving the whole table prefix aside (on S3, a list-copy-delete loop) before creating a fresh table —
bespoke IO machinery of exactly the kind this conversion removes. The manual move-aside goes into the
operations runbook instead. This is the one property the conversion gives up versus the parquet plan,
and it is the case Delta makes very unlikely; the two faults that *are* likely — off-contract and
schema mismatch — self-heal in place with the old version retained by the log. Implemented as "try
the cheap recovery, report if it does not take", not by inspecting exception types.

**D3 — Keep the roster-repair pass.** A thin roster is not benignly lossy.
`_load_engineering_inputs` derives the NWP cell filter from the roster (`cv_assets.py:328`, feeding
`h3_index.is_in(cells)` at `:336-342`) while the population comes from the promoted model's
`trained_ids` (`production_assets.py:238`), and `live_forecasts` raises outright on an empty result
(`production_assets.py:285-290`) — so a thin roster costs, at best, forecasts for the missing series
and, at worst, the whole slot. `forecast_metrics` (`cv_assets.py:997-1000`) and `cv_power_forecasts`
read the roster too, where thinning would silently shrink a training or scoring population, the
opposite of the fail-fast posture R&D is meant to have. The pass is ~5 lines over existing functions,
guarded so it cannot make things worse, and it turns "degrade and wait ~5 h" into an actual repair.

**D4 — The `_load_engineering_inputs` hazard is filed separately, as
[#528](https://github.com/openclimatefix/nged-substation-forecast/issues/528).** Its unguarded roster
read means an unusable *or thin* roster fails `live_forecasts` outright — off the ladder entirely —
for as long as the fault lasts. This change shortens that window to at most one hourly ingest but
does not remove it, and the real fix is a design question (the issue argues the live path should not
depend on the roster at all for a series whose model it already holds), so it does not belong here.

**D5 — `report_asset_degradation(name, detail: BaseException | str)` keeps the union signature.** One
function, one tag, one docstring covering both shapes, versus two near-identical six-line functions.
The union exists only because the rebuild path has no live exception at the asset boundary.

**Vacuum and checkpoints — already tracked.**
[#357 "Implement auto vacuuming for Delta Lake"](https://github.com/openclimatefix/nged-substation-forecast/issues/357)
is open and covers this; it has been fleshed out and attached to the v0.8 epic rather than a
duplicate being filed. It matters less for these two tables than the raw numbers suggest, because a
no-op merge commits nothing (measured), so the roster's history grows only when NGED's metadata
actually changes, and the weights change only when the boundary or H3 resolution does.

**Measured cost of the conversion**, so it is on the record rather than assumed. Roster-sized tables,
local SSD, and S3 request counts against a moto server:

| | read latency | S3 requests | on disk |
|---|---|---|---|
| bare parquet | 0.6–1.0 ms | 3 | 5 KB, 1 file |
| Delta, fresh | 4–5 ms | 6 | 7.6 KB, 2 files |
| Delta, 26 versions, unvacuumed | — | 56 | — |
| Delta, 101 versions, unvacuumed | 9.4 ms | — | 670 KB, 204 files |

Absolute latencies are trivial; what matters is that request count grows with commit history, which
is what #357 exists to bound. The two dashboards pay +3 requests per interactive read.

**Delta is stricter about dtypes than parquet**: `write_delta` refuses a `Null`-dtype column outright
(verified — it raised on the first attempt at the benchmark above), where parquet stores one happily.
Any all-null column must carry a concrete dtype, which Patito's `.cast()` already ensures on every
path that goes through the contract.

## History of this plan

### Jack's decision: Delta rather than a hardened parquet path

The first version of this plan implemented the issue's options 1 and 3 directly: a
`write_parquet_atomically` helper (local temp file + `os.replace`, remote written directly because an
S3 `PUT` is already atomic), a `copy_object` helper for the quarantine, and an `_align_to_contract`
step to stop the `concat` raising. Jack asked whether converting every bare-parquet file to Delta
was not the better fix — no bespoke atomic-write machinery, and one storage idiom.

It is, and for a reason the first plan undersold: `MERGE` **deletes** the machinery rather than
moving it. The read → validate → `hash_rows` diff → `concat` → `unique` → write dance collapses into
one transactional upsert, which takes raisers 1 and 3 out by construction along with the
read-modify-write race. `_align_to_contract` survives the rewrite, but shrunken: the parquet plan
needed it on *both* frames to stop the `concat` raising, whereas here it applies to the snapshot
alone, and for a different reason — D1's "a field NGED stops sending is cleared" semantics, which
Delta would otherwise silently reverse. Delta's version history also
preserves metadata changes we currently overwrite and lose, which matters for reproducing past CV
runs, and it subsumes the quarantine copy. Against that, the measured costs above and the corrupt-log
case in D2. The conversion covers the roster and the H3 grid weights;
`scripts/export_forecasts_for_alex.py`'s exports stay parquet, being files for a human outside the
project.

### What the adversarial review of the first version found

A fresh sub-agent reviewed the parquet version with no knowledge of the reasoning behind it. Its
findings that still apply are folded in above; recorded here so nothing is silently dropped:

1. **The `pl.concat` `ShapeError`** — the draft asserted it "cannot raise" after a successful
   validate. False, because of the four `allow_missing` fields, and a recovery keyed only on the read
   never reaches it. The review's headline finding; it is now raiser 3, and `MERGE` is what disposes
   of it.
2. **The freshness-check consequence was backwards**, inherited from the issue body and destined for
   the operator runbook: `stale` is computed from coverage and deliberately *not* restricted to the
   roster (`checks.py:167-173`), so a dropped id keeps being flagged stale. What a thin roster really
   does is stop never-seen ids being reported at all (`checks.py:180-183`) and shrink
   `n_series_total` (`:198-201`).
3. **A thin roster can fail `live_forecasts` outright**, not merely thin it — now D3, and the reason
   the repair pass exists.
4. **Rule 12 restated principle 10**, which `inherent-stability.md:155-159` requires to be marked and
   paired. Resolved differently here: rule 12 now drops the atomicity clause entirely, because the
   store delivers it.
5. **A fixed `.tmp` suffix was unsafe for two writers** (nothing serialises this asset: no `pool=`,
   and `default_limit: 1` governs pooled ops only). Moot now — delta-rs commits via conditional put
   on S3 and an atomic rename locally, so concurrency safety comes from the store.
6. **Docstrings and pages the change falsifies** — `_sentry.py:98-99`, `production-deployment.md:174-179`,
   `checks.py:311-313`, and `_uri.py:7-11`. The first three are in the docs list above; the `_uri.py`
   one is moot, since the conversion adds nothing to that module.
7. **The R&D readers of the roster** (`cv_assets.py:997-1000`, `cv_power_forecasts`) were missing from
   the impact list. Folded into D3.
8. **Line-reference drift** — `validate(existing_metadata)` is at 408, not 407. Corrected.
9. **`os.replace` changes the inode**, so a symlinked roster path would be replaced by a plain file.
   Moot with Delta.

Considered and **partly rejected**: the review read the asset-level `except BaseException` as newly
swallowing the strict-contract boundary on *incoming* metadata. Rejected as stated —
`_extract_time_series_metadata` validates and raises at `read_nged_json.py:58`, outside the guard, so
an NGED contract break still fails the run. The narrower point (a bug in our own upsert code is now
degraded rather than raised) is real and is stated in the design-philosophy check above.
