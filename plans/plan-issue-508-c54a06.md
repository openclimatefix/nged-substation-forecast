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
| 3 | `pl.concat([new_metadata, existing_metadata])` (`storage.py:431`) raises `ShapeError` on a width or column-order mismatch | Only via `MERGE`, which this plan uses |

Raiser 3 is live today rather than hypothetical. Four `TimeSeriesMetadata` fields are
`allow_missing=True` (`information`, `area_wkt`, `area_center_lat`, `area_center_lon` —
`packages/contracts/src/contracts/power_schemas.py:247-283`), so a narrower frame validates cleanly
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
(`src/nged_substation_forecast/defs/cv_assets.py:324`), so an unusable roster takes the forecast off
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

  Five verified facts have to be written down, because each is a trap. **Where they go is now
  constrained by `main`'s "one home per argument" rule** (`docs/architecture/code-style.md`): a
  sentence of "because" may sit in a docstring, but a paragraph of it belongs on a docs page with the
  docstring linking to it. These five are paragraphs. So they go in a new subsection of
  `docs/architecture/performance.md` — the page that already exists to record *why* each table is
  written the way it is — and `merge_time_series_metadata`'s docstring states the guarantees and the
  one-line prohibitions, linking to that subsection as a rendered URL. The five facts, wherever they
  land:

    1. **Every `pl.Enum` column must be cast to `pl.String` before any write.**
       `TimeSeriesMetadata` has four (`time_series_type`, `units`, `licence_area`,
       `substation_type` — `power_schemas.py:188,196,203,218`), and handing a validated,
       Enum-typed roster to `write_delta` aborts a rust worker: `DeltaError: Generic DeltaTable
       error: writer join error: task 28 panicked with message "internal error: entered
       unreachable code: cannot downcast Utf8View dictionary value to byte array"` (reproduced).
       This is the write-path gotcha the `polars-patito-gotchas` skill documents at
       `SKILL.md:100-121`, and `_write_metrics_to_delta` already carries the fix
       (`cv_assets.py:750`) — `enum_cols = [c for c, d in df.schema.items() if isinstance(d,
       pl.Enum)]`, then cast each to `pl.String`. It must be applied on **all three** write paths
       here (create, rebuild-overwrite, and the merge source), and doing it inside this module is
       what stops every caller having to remember. Delta stores these as `String`, so reads must
       cast back — see the `scan_delta` note below. `MERGE` alone happens to tolerate an
       Enum-typed source, because datafusion casts to the target schema; that is exactly why a
       test exercising only the merge path would pass while production's create path panics.
    2. The matched-clause predicate is a chain of `(source.<col> IS DISTINCT FROM target.<col>)`
       over the non-key columns, and **each comparison must be parenthesised**. Without the
       parentheses delta-rs fails with `DeltaError: Generic DeltaTable error: type_coercion … Cannot
       infer common argument type for logical boolean operation Float32 OR Boolean`.
    3. With that predicate, **a merge that changes nothing commits no new version at all**
       (measured: history stayed at 2 versions), and only changed rows are *reported* as updated
       (measured on a ten-row target with two changed rows: `updated=2, copied=8`). So the
       roster's Delta history grows only when NGED's metadata actually changes — which is what
       keeps the vacuum burden negligible for this table. Note what this does *not* buy: the same
       measurement shows `files_scanned=1, files_removed=1, files_added=1`, so any content change
       rewrites the single file holding the whole table, exactly as today's `unique(keep="first")`
       does. The win is the skipped commit, not fewer bytes.
    4. The predicate is built from the contract's non-key columns, which is safe **only because
       the caller aligns the snapshot to the full schema first** (D1). Handed a snapshot that omits
       an `allow_missing` column, the predicate names a column that does not exist and delta-rs
       raises `DeltaError: … Schema error: No field named source.information` (reproduced) — so
       the function asserts the source carries every contract column rather than silently building
       a narrower predicate.
    5. **The same applies to the *target*, and it is not hypothetical.** A predicate naming
       `target.<col>` for a column the stored table lacks raises the same catchable `DeltaError`.
       Two ways in: a table migrated from a parquet file that never carried the four
       `allow_missing` columns, and the day someone adds a field to `TimeSeriesMetadata`. Both
       would otherwise be absorbed by the rebuild path — a routine schema addition would silently
       thin the roster and trigger a full re-download. So the function compares the stored
       schema against the contract *before* merging and reports a mismatch as its own condition:
       a contract addition is a deliberate `schema_mode="merge"` migration, not a rebuild. This is
       also why the migration below must write all 14 columns.

  **Neither module declares a sort order, and that is a departure worth stating** rather than
  leaving as an omission, because `delta_store`'s charter is "compression-friendly sort orders"
  (`packages/delta_store/README.md:8`). Two reasons. At tens of KB there is no compression lever to
  pull. And `MERGE` **destroys row order** and offers no way to restore it: measured on a table
  stored as ids 1…10 with rows 3 and 7 changed, the stored order afterwards is
  `[3, 7, 1, 2, 4, 5, 6, 8, 9, 10]` — updated rows are rewritten first, copied rows follow. Today
  `upsert_metadata` sorts by `time_series_id` on every write (`storage.py:433`), so this is a real
  behaviour change and not merely a storage swap. It is safe: `TimeSeriesMetadata` declares no
  `columns_to_sort_by` and `validate` has no sortedness check (unlike `PowerTimeSeries.validate`
  at `power_schemas.py:83-92`), and the dashboards sort for themselves
  (`packages/dashboard/view_forecasts.py:73`). Both module docstrings should say the table is
  unordered on disk so nobody later "fixes" it with a sort that a merge would immediately undo.

- **`h3_grid_weights.py`** — `H3_GRID_WEIGHTS_WRITER_PROPERTIES` and
  `write_h3_grid_weights(weights, table_uri, storage_options)`, a plain `mode="overwrite"` commit.
  The table is write-once-per-boundary-change and ~30 KB. No Enum columns here, so fact 1 does not
  bite — but `H3GridWeights.h3_index` is `pl.UInt64` (`geo_schemas.py:16`) and the Delta protocol
  has no unsigned integer types, so it comes back `Int64`. Every read must cast; see below.

### `packages/contracts/src/contracts/`

- **`power_schemas.py`** — add `TimeSeriesMetadata.scan_delta(path, storage_options) ->
  pt.LazyFrame[Self]`, mirroring [`Nwp.scan_delta`](../packages/contracts/src/contracts/weather_schemas.py)
  (`weather_schemas.py:442`) so every roster reader gets a typed, cast scan from one place.
- **`geo_schemas.py`** — the same for `H3GridWeights`.
- **The `.cast()` in those classmethods is load-bearing, not decorative**, and this is the reason
  the classmethods exist rather than each caller writing `pl.read_delta`. Delta stores the roster's
  four Enums as `String` and `h3_res_5`/`h3_index` as `Int64`, so a bare
  `TimeSeriesMetadata.validate(pl.read_delta(uri))` fails with five dtype errors, and
  `H3GridWeights.validate` fails on `h3_index`. Both reproduced; both pass once the scan is
  `pt.LazyFrame.from_existing(pl.scan_delta(...)).set_model(cls).cast()`, exactly as `Nwp.scan_delta`
  does it. Per the `polars-patito-gotchas` skill (`SKILL.md:100-121`) the contracts keep their Enum
  dtypes: that guidance only pushes a column to `String` when it is *filtered or partitioned* on in
  Delta, which none of these are, so casting on read costs nothing and keeps the in-memory contract
  expressive.
- **`settings.py`** — the derived defaults become `metadata.delta` (`settings.py:395`) and
  `h3_grid_weights.delta` (`settings.py:379`), matching `power_time_series.delta`. The field *names*
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
  as a *gate*, and the `concat` + `unique(keep="first")` merge (431). Raisers 1 and 3 go with them.
- **New `_align_to_contract(df) -> pt.DataFrame[TimeSeriesMetadata]`** — selects
  `TimeSeriesMetadata.columns` in declared order, supplying
  `pl.lit(None, dtype=TimeSeriesMetadata.dtypes[name])` for any column the snapshot lacks. Only the
  four `allow_missing=True` fields can be absent from a frame that validated, and all four are
  declared `| None`, so the filled nulls are contract-legal (verified). This is D1's mechanism: it
  is what makes a field NGED stops sending get *cleared* rather than silently retained, which is
  today's behaviour, and it is what lets the merge predicate be a fixed list of columns.
- **`upsert_metadata`'s new shape**: validate the snapshot, align it; if the table does not exist,
  create it; otherwise **read-and-validate the stored roster as a gate**, then merge. If that gated
  read fails, or the merge raises, because the *stored table* is unusable — our contract rejects it,
  Delta rejects the schema, or its columns do not match the contract — log at ERROR and rebuild it
  with an overwrite commit (`schema_mode="overwrite"`), reporting the rebuild in the stats. The
  previous version stays in the log, so nothing is destroyed and no file is copied aside.
- **Two different things read the stored roster, and the plan must not conflate them** — an earlier
  draft did, which is how "the read is a gate that triggers a rebuild" and "the read is a guarded
  annotation that can fail harmlessly" both ended up in this section. They are separate steps with
  opposite failure policies:
    1. **`_read_existing_roster` is the gate.** It reads and validates. Failure means the stored
       table is unusable, and the response is a rebuild. Without this, an off-contract roster would
       sit there until some later merge happened to trip over it.
    2. **`_changed_ids(stored, snapshot)` is the annotation.** It runs *only* on a roster the gate
       already accepted, and computes the id list purely to label the materialisation. Its own guard
       covers a bug in the diff itself (`hash_rows` on an unexpected shape, say) — not a bad table,
       which the gate has already excluded. If it fails, the merge still happens and the stats omit
       the id list.
- **If the rebuild commit *itself* fails, degrade rather than trying harder** (D2): report and let
  the asset-level guard carry the run. This is the discrimination the corrupt-`_delta_log` case
  needs, and doing it by "did the recovery work?" rather than by inspecting exception types is what
  keeps it robust — there is no reliable taxonomy separating "delta-rs cannot open this table" from
  "our contract rejects its contents", and guessing wrong in either direction is worse than trying
  the cheap recovery and reporting when it does not take.
  The reason the diff cannot simply come from the merge is that `UpsertMetadataStats` publishes
  `metadata_updated_TimeSeriesIDs`, a *list* of ids, and delta-rs' merge metrics give counts only —
  verified: `num_target_rows_inserted`, `num_target_rows_updated`, `num_target_rows_copied`, no ids.
  Demoting it is still a strictly better structure than today's, where the same read is load-bearing
  for correctness.
- **`UpsertMetadataStats`** keeps its existing keys, sourced from the merge metrics
  (`metadata_n_new_TimeSeriesIDs` ← `num_target_rows_inserted`,
  `metadata_n_updated_TimeSeriesIDs` ← `num_target_rows_updated`), and gains two optional ones:
  `metadata_roster_rebuilt_reason: str` and `metadata_upsert_failed: str` (the latter set by the
  asset).
- **The rebuild and create paths must not reuse those two count keys**, which is a trap the merge
  sourcing walks straight into. A rebuild is an *overwrite*, so there are no merge metrics, and the
  obvious fallback — "every row we wrote is new" — makes `metadata_n_new_TimeSeriesIDs` report the
  whole snapshot height at precisely the moment the roster was *thinned*. The stat would be at its
  most reassuring when the operator most needs alarm. So the rebuild and create paths report
  `metadata_n_rows_written` alongside their reason and leave the merge counts unset; a key that means
  "inserted by a merge" is only ever set by a merge. Relatedly, the repair pass calls
  `upsert_metadata` a second time in the same run, so the asset must **merge the two stats dicts
  under distinct keys** rather than calling `add_output_metadata` twice with the same ones — verified
  that duplicate keys do not raise, the later value silently wins, which would hide the rebuild
  behind the repair's own numbers.
- **`_read_existing_roster`** — reads via `TimeSeriesMetadata.scan_delta` and validates, returning
  `None` plus a reason on `Exception` rather than raising. Catching `Exception` and **not**
  `BaseException` is deliberate, and differs from the `checks.py` guards: a pyo3 panic is not
  evidence about the *table*, and overwriting a table on that evidence is worse than skipping the
  update, so a panic falls through to the asset-level guard, which degrades without rebuilding.
  It must also catch `OSError`: `delta_table_exists` does **not** return `False` for a plain file
  sitting at the table path, it raises `OSError: Generic LocalFileSystem error ↳ Unable to walk dir`
  (reproduced). That is exactly the state a deployment left mid-migration is in — see the migration
  section, which is where the real fix for it lives.

### `src/nged_substation_forecast/defs/assets.py`

- **`power_time_series_and_metadata`** — the `upsert_metadata` call (line 125) goes under a
  `try`/`except BaseException` that re-raises `KeyboardInterrupt | SystemExit |
  DagsterExecutionInterruptedError` (the `checks.py:347-361` idiom, and the same reasoning: a pyo3
  `PanicException` from polars/delta-rs/obstore does not derive from `Exception`, and each compiled
  extension defines its own class). The handler logs the traceback, calls
  `report_asset_degradation("power_time_series_and_metadata", exc)`, and substitutes
  `UpsertMetadataStats(metadata_upsert_failed=repr(exc))` so the power write below still runs. Needs
  a `DagsterExecutionInterruptedError` import.
- **Repair a rebuilt *or newly created* roster in the same run.** Where the stats carry
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

    **The create branch needs the same treatment, and this is the plan's own Q3 argument turned
    against it.** A create is structurally identical to a rebuild: the table does not exist, so the
    roster ends up holding only this run's snapshot. Since the create branch sets no rebuild reason,
    the repair would never fire and nothing would be reported — an automatic repair for the rebuild
    path and none for the path most likely to actually happen (a migration that was skipped, or an
    env var still pointing at the old `.parquet`). The discriminator is not "created versus rebuilt"
    but **"was this run's file listing narrowed?"**: fire the repair whenever the roster was created
    or rebuilt *and* `list_of_new_json_files` is shorter than `list_of_large_json_files`. On a
    genuine first-ever backfill the two are equal, so the repair correctly does not fire; on every
    other create it does.
- **The consequence worth a comment**, because it is a genuine trade that holds whichever order the
  two writes go in: the power Delta table is what `select_new_rows` uses to decide which JSON files
  are new, so once the power rows land, a *failed* roster update is not retried — that run's
  metadata change is lost until NGED republishes those series (~5 h). Losing one refresh of derived,
  re-delivered data is much cheaper than blocking the power stream, and it is the second reason the
  failure must reach Sentry rather than only the logs.
- **On the rebuild path, though, "lost until NGED republishes" understates it: for a series that has
  stopped publishing, the loss is permanent.** `select_new_rows` keeps only rows whose time exceeds
  the stored `last_time` per series (`storage.py:343`), so files already represented in the power
  table are never parsed again. If the roster is rebuilt thin *and* the repair pass fails (test 15
  pins that as an acceptable outcome), a quiet series' metadata row does not come back on its own —
  ever. Nothing re-reads its file, and the repair only runs in the same run as the rebuild. So the
  runbook needs an explicit manual re-derivation step (re-run `download_and_parse_files` over the
  full listing, or read the pre-rebuild row back by time travel), and it is not optional garnish:
  it is the only route back.
- **`h3_grid_weights`** — `weights.write_parquet(...)` (line 167) becomes
  `write_h3_grid_weights(...)`; `if_local_path_then_make_parent_dir` stays (a Delta write needs the
  parent too).
- **`ecmwf_ens`** — the weights read at line 259 becomes `H3GridWeights.scan_delta(...)`.

### Remaining read sites

All move from `pl.read_parquet` to the new `scan_delta` classmethods. **Not purely mechanical**,
because three of them build an *eager* `pt.DataFrame` today and `scan_delta` returns a
`pt.LazyFrame` — each needs a `.collect()` and a re-wrap, and two of them get their dtypes fixed as a
side effect rather than by accident:

- `src/nged_substation_forecast/defs/checks.py:234-243` — `_read_roster_ids`: `object_exists` →
  `delta_table_exists`, `scan_parquet` → `TimeSeriesMetadata.scan_delta`. Still inside the check's
  catch-all, so it cannot raise into the run.
- `src/nged_substation_forecast/defs/cv_assets.py:324` (`_load_engineering_inputs`) and `:1000`
  (`forecast_metrics`) — both eager.
- `src/nged_substation_forecast/defs/assets.py:258-262` (`ecmwf_ens`, listed above) deserves its own
  mention: it does `set_model(H3GridWeights)` with **no `.cast()`** today, which is harmless against
  parquet and silently wrong against Delta — `h3_index` would arrive `Int64` where the contract says
  `UInt64`, and it feeds the NWP spatial join. `H3GridWeights.scan_delta` casting is what fixes it.
- `packages/dashboard/view_forecasts.py:63` and `packages/dashboard/map_and_timeseries.py:48-50`
  both do `validate(read_parquet(...))`, which **fails outright** against a Delta table (verified:
  five dtype errors on the roster's four Enums plus `h3_res_5`). They must go through `scan_delta`,
  not merely swap the reader. Both are marimo notebooks, so the `marimo-notebooks` skill's import and
  cell rules apply — load it before editing them.
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
code. **The naive one-liner does not work**, and an earlier draft of this plan specified it: run
verbatim, `pl.read_parquet(<old>).write_delta(<new>)` fails on the roster with the Enum panic from
fact 1, because parquet *preserves* the Enum dtype on round-trip where Delta cannot store it at all
(reproduced end-to-end through a `metadata.parquet` written by today's `upsert_metadata`). Three
requirements the migration therefore has:

- **Go through the new writer, not `write_delta` directly.** `pl.read_parquet(<old>)` → align to the
  contract → `delta_store.metadata.merge_time_series_metadata(<new>)`. That picks up the Enum cast and
  the 14-column alignment for free, and alignment is not optional: a table migrated with only the
  columns the live parquet happens to carry hits fact 5 on the very first merge afterwards.
- **Verify with `validate`, not a row count.** The stated check ("verify the row count") would pass on
  a table that no reader can use. `TimeSeriesMetadata.scan_delta(<new>).collect()` then `validate` is
  the check that actually proves the migration, and for the weights it is the one that catches
  `h3_index` coming back `Int64`.
- **Update `METADATA_PATH` / `H3_GRID_WEIGHTS_PATH` if either is set explicitly**, and say so
  loudly. Only the *derived defaults* change, and `docs/live_service/setup.md:64-72` invites setting
  these by hand. Left pointing at the old `.parquet`, every run afterwards hits the `OSError` from
  `delta_table_exists` walking a plain file as a directory. It does at least page rather than pass
  silently — the asset-level guard sends a Sentry event, and the freshness check drops to `WARN` — but
  nothing self-heals, so it stays broken until someone reads the alert. The runbook step is: migrate,
  update the env var, delete the `.parquet` last.

Put the exact commands in the PR body and in `docs/live_service/operations.md`; do not add a migration
code path (CLAUDE.md: no backwards compatibility with data we can re-derive). The weights can equally
be re-materialised from the asset, since they are deterministic — which is the simpler route for that
table, and worth stating as the recommended one.

**The test suite migrates too, and it is most of the mechanical work.** Five helpers write the roster
as parquet, all needing the same treatment, and none were in the earlier draft's list:

- `tests/test_checks.py:216` `_write_metadata_roster` (`write_parquet` at `:233`), called at `:251`,
  `:282`, `:343`, plus the corrupt-roster test at `:388`
- `tests/test_live_forecasts.py:119` `_write_metadata`, called at `:166`
- `tests/test_cv_power_forecasts.py:94` `_write_metadata`, called at `:132`
- `tests/test_metrics.py:133` `_write_metadata`, called at `:196`
- `tests/test_trained_cv_model.py:101` `_write_metadata`, called at `:148`
- `tests/test_assets.py:157` (`pl.read_parquet(env / "NGED" / "metadata.parquet")`), `:229`
  (`assert not (...).exists()`) and `:250` (the weights read)

The *source* read/write inventory in the sections above is complete: nothing else reads or writes
either table, nothing globs for them by name, and `.env.example`, the `Dockerfile`, `.dockerignore`
and `conf/` reference neither filename. `packages/contracts/tests/test_uri.py:27-28,48` mention
`metadata.parquet` only as an arbitrary string for `uri_join`/`object_exists`, so they stay as they
are.

## Design-philosophy check

This path is **production** — the hourly `power_time_series_and_metadata_job`, which carries
`sentry_capture_failure` (`defs/schedules.py:19`) — so it degrades rather than raises.

- **Principle 10** ("every write is atomic and idempotent",
  `docs/design-philosophy/design-principles.md:387`) is the one this change *delivers*. Its *Decided*
  paragraph (`design-principles.md:410-416`) credits Delta Lake for atomicity; the roster and the
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

Most of these assert behaviour that does not exist on `main`. Where a test is new only because the
module it exercises is new, that is said rather than dressed up as a bug being fixed — and **one item
below already exists and passes on `main`** (item 6), which an earlier draft of this plan wrongly
claimed as a new failing assertion.

`packages/delta_store/tests/test_metadata.py` (new module, so these are coverage for new code rather
than regressions):

1. `test_merge_updates_only_the_rows_that_changed` — three-row table, snapshot with one changed row.
   Asserts `num_target_rows_updated == 1` and `num_target_rows_copied == 2`. Note what not to assert:
   `files_added`/`files_removed` are both 1 either way, because the table is one file (measured), so
   an assertion on bytes or files written would be testing a property this change does not deliver.
2. `test_merge_commits_no_new_version_when_nothing_changed` — merge an identical snapshot; assert
   `DeltaTable(uri).history()` is unchanged in length. This is the property that keeps the vacuum
   burden negligible, and it is the one a careless `when_matched_update_all()` (no predicate) would
   silently break — so it is also a regression test on the predicate.
3. `test_merge_accepts_a_snapshot_missing_the_optional_columns` — a snapshot without
   `information`/`area_wkt`/`area_center_*` against a full-schema table. Asserts the insert succeeds
   and the new row's absent fields are null. On `main`: `ShapeError` from the `concat` (verified).
4. `test_merge_creates_the_table_when_absent` — first-ever call against a non-existent URI. **This is
   the test that must be written against a fully Enum-typed roster**, because the create path is where
   fact 1's rust panic lands, and a suite that only ever merges into a pre-existing table would pass
   while production's first write aborts. Assert the round trip too: read back through
   `TimeSeriesMetadata.scan_delta` and `validate`, which is what pins the `String → Enum` cast at the
   other end.
5. `test_merge_rejects_a_target_whose_schema_does_not_match_the_contract` — a stored table lacking the
   four `allow_missing` columns (i.e. one migrated from a narrow parquet file). Assert a clear,
   catchable error naming the mismatch, *not* a rebuild: fact 5's point is that a schema difference is
   a migration to perform, not corruption to overwrite. Without this the failure surfaces as
   `DeltaError: … Schema error: No field named target.information` (reproduced) from deep inside the
   merge, and the rebuild path swallows it.
6. `test_merge_does_not_preserve_row_order` — pin the *documented* behaviour, not an aspiration:
   after a merge that changes two rows of ten, the stored order is no longer the id order (measured:
   `[3, 7, 1, 2, 4, 5, 6, 8, 9, 10]`). This exists so the next person to notice the unsorted table
   finds a test explaining why, rather than adding a `.sort()` the following merge silently undoes.

`packages/nged_data/tests/test_storage.py` (rewritten for Delta):

7. `test_upsert_metadata_rebuilds_when_the_stored_roster_fails_its_contract` — write a Delta table
   missing `substation_type` (a required field), then upsert. Asserts a rebuild rather than a
   `DataFrameValidationError`, that the stats carry `metadata_roster_rebuilt_reason`, and that the
   **pre-rebuild version is still readable via time travel** — the property that replaces the
   quarantine copy. On `main`: `DataFrameValidationError` (verified).
8. **`test_upsert_metadata_returns_diff` already exists and passes on `main`**
   (`packages/nged_data/tests/test_storage.py:145-232`, asserting exactly
   `metadata_n_new_TimeSeriesIDs == 1`, `metadata_n_updated_TimeSeriesIDs == 1` and
   `set(metadata_updated_TimeSeriesIDs) == {2}` — confirmed by running it). So this is a **port, not a
   new test**: change its writes and read-backs to Delta and leave the assertions alone. That is the
   point — it is the regression guard that the diff's demotion from gate to annotation does not
   quietly drop the field, and it is worth more for being untouched.
9. `test_upsert_metadata_still_merges_when_the_informational_diff_fails` — monkeypatch the diff read
   to raise; assert the merge happened and the stats omit the id list without raising. On `main` the
   equivalent read is load-bearing, so a failure loses the write entirely.
10. `test_upsert_metadata_merges_a_stored_roster_with_reordered_columns` — same data, different
   column order. Asserts a no-op merge and no new version. On `main`: `hash_rows` is order-sensitive
   (verified), so it reports a spurious update or raises `ShapeError` on the vstack.
11. `test_upsert_metadata_clears_a_field_the_snapshot_no_longer_carries` — the D1 semantics, and the
   one test that would catch (B) being implemented by accident. A stored row has
   `information = "note"`; this run's snapshot omits the `information` column entirely. Asserts the
   stored value ends up **null**, not `"note"`. On `main`: `ShapeError` from the `concat` (verified).
   Without `_align_to_contract`, `when_matched_update_all` leaves the old value in place — verified,
   and it is silent.

`tests/test_s3_data_paths.py` (moto, `integration`-marked; the per-test reset fixture at
`tests/test_s3_data_paths.py:85-101` handles moto's process-global backend):

12. `test_metadata_delta_round_trip_over_s3` — replaces the existing
   `test_metadata_parquet_round_trip_over_s3` (`:237`): create, merge, read back over S3, exercising
   delta-rs' conditional-put commit path against the object store.

`tests/test_assets.py` (the existing `_FakeS3Store` + `env` harness):

13. `test_power_time_series_and_metadata_writes_power_when_the_roster_upsert_fails` — monkeypatch
    `assets.upsert_metadata` to raise `RuntimeError`; assert `result.success`, that the
    `power_time_series` Delta table holds the fixture rows, and that `metadata_upsert_failed` appears
    in the materialisation metadata. This is the issue's headline property. On `main`: the run fails.
14. `test_power_time_series_and_metadata_repairs_a_rebuilt_roster` — with the fake store serving two
    series, ingest once, then make the stored roster off-contract and arrange for only *one* series
    to have a new file. Asserts success and that the roster ends up holding **both** ids — the repair
    ran and the roster was not left thin — plus `metadata_roster_rebuilt_reason` in the metadata.
15. `test_power_time_series_and_metadata_survives_a_failed_roster_repair` — as above with the
    repair's `download_and_parse_files` raising; assert the run still succeeds and the thin roster
    stands.
16. `test_h3_grid_weights_materialises_and_writes_a_delta_table` — the existing
    `test_h3_grid_weights_materialises_and_writes_parquet` (`:236`) converted, asserting
    `delta_table_exists` and the row count. Its six sibling `_write_h3_grid_weights` fixture calls
    (`:275`–`:502`) move to the Delta writer.

`tests/test_sentry.py`:

17. Mirror the two existing `report_check_degradation` tests (`:176`, `:220`) for
    `report_asset_degradation` — the built event carries `{"degraded_asset": …}`, no tag leaks into
    the current or isolation scope, and a raising `capture_*` is swallowed and logged. Add the
    message-shaped call as a third case. On `main`: the function does not exist.

`tests/test_checks.py`:

18. `test_power_data_is_fresh_degrades_on_a_corrupt_metadata_parquet` (`:388`) — rename, and rewrite
    both the corruption it sets up and its docstring, which currently states that "`upsert_metadata`
    reads the same file first and fails the asset outright" (`:395`) — no longer true.
    **Specify the replacement corruption, because two obvious candidates do not work.** Junk bytes at
    the path (today's setup, `:402`) now make `delta_table_exists` raise `OSError` walking a file as a
    directory — still caught by the check's catch-all, so the test would pass, but it would be
    asserting on a filesystem-shape error rather than on a corrupt roster, which is not the state the
    docstring describes. And a merely *off-contract* Delta table does not degrade this check at all:
    `_read_roster_ids` (`checks.py:234-243`) only does `.select("time_series_id")` and never validates.
    The state that genuinely reproduces it is **a Delta table with a corrupt `_delta_log`** — truncate
    `_delta_log/00000000000000000000.json` — where `delta_table_exists` returns `True` and the scan
    raises. That is also the D2 fault, so the test doubles as the check's half of it.

`packages/contracts/tests/test_settings.py`:

19. `:49` asserts `settings.metadata_path == "/srv/data/NGED/metadata.parquet"` and `:49` the same for
    `h3_grid_weights_path` — **both** need updating; an earlier draft cited only one of them. The set at
    `:84` needs **no** change: it is a set of *field names*
    (`{"metadata_path", "h3_grid_weights_path"}`) used to decide which paths derive from the internal
    versus delivery root, not a set of path suffixes, and this plan deliberately keeps those field
    names.

## Docs to update

Written to describe how the code works now, per CLAUDE.md's "write about the present".

- **`CLAUDE.md:219`** — the asset "upserts metadata parquet" becomes a Delta upsert. Check the
  package table at `:204` still reads correctly for `delta_store` (it will, but it now owns four
  tables).
- **`docs/design-philosophy/design-principles.md:410-416`** — principle 10's *Decided* paragraph:
  every managed table is now Delta, so the property holds project-wide with no exception.
- **`docs/design-philosophy/inherent-stability.md`** — a sentence in "Missing versus wrong" giving
  the roster as the second worked example of rule 3, and a new **rule 12**: *a derived artifact we
  cannot read is absent, not fatal — rebuild it and keep going.* Appended, so nothing is renumbered.
  Note the wording deliberately drops any atomicity clause: atomicity is principle 10's job and is
  now delivered by the store, so rule 12 carries only the part rules 1–11 do not imply and needs no
  "restates a principle" marking.
- **`docs/architecture/production-deployment.md`** — three places: the "half-written
  `metadata.parquet`" example at `:91` (a state that no longer exists), the Sentry section at
  `:179` whose "the only fault the hook cannot see is a standalone `@asset_check` … which, by
  design, is both of them" is falsified by an *asset* now catching its own exception (and by the new
  `degraded_asset` tag), and a short new subsection for the roster's policy: a Delta table, merged
  rather than rewritten, rebuilt in place if it fails our contract, with the power write no longer
  coupled to any of it.
- **`src/nged_substation_forecast/defs/checks.py:340-341`** — `power_data_is_fresh`'s own docstring
  repeats the "half-written `metadata.parquet`" example; fix it with the other two.
- **`docs/live_service/operations.md`** — the same phrase at `:182`, plus a new operator paragraph:
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
- **`docs/architecture/performance.md:16-37`** — "Storage formats: measured, not assumed" is the page
  that records each table's writer-properties choice *and why it was chosen*, currently for `nwp` and
  `power_forecasts`. Two new `*_WRITER_PROPERTIES` constants belong there, with the honest
  justification: plain ZSTD-3, no per-column encodings, because at tens of KB there is nothing to
  measure. Saying that explicitly is what stops the page implying the omission was an oversight.
- **`packages/contracts/src/contracts/settings.py:278`** — the `nged_data_path` docstring says
  "Directory holding the NGED power_time_series Delta table and metadata parquet."
- **`docs/architecture/production-deployment.md:75`** — "a roster series (present in the
  `TimeSeriesMetadata` parquet)". A fourth spot in the same file, on top of the three above.
- **`docs/architecture/ecmwf-ens-known-issues.md:269`** — "the `h3_grid_weights` parquet it has
  already loaded".
- **`packages/contracts/README.md`** — if it lists the schemas' helpers, the two new `scan_delta`
  classmethods belong there.
- No roadmap page completes here, so there is no "Implementation details" section to delete and no
  status banner to move. `#508` is referenced from no `docs/` page.

## House rules `main` added after this plan was written

`main` moved 30 commits while this plan sat on the branch, and two of the rules it landed change what
the implementation must do — not just where the line numbers point:

- **A docs link in code is spelled as its rendered URL, never a repo path**
  (`docs/architecture/code-style.md`). Every docstring and `#` comment this change adds must follow it,
  and `storage.py`'s own doc link was converted on `main` for the same reason.
- **"One home per argument"** — rationale worth a paragraph lives on one docs page, and the docstring
  links to it. This plan was written with five paragraph-length traps in a module docstring, which the
  rule forbids; they move to `docs/architecture/performance.md`, as noted above. The rule cuts both
  ways, and the second half matters here too: rationale worth a paragraph must not live *only* in a
  docstring, where nobody browsing the docs will find it.
- **The prose rules in `CLAUDE.md`** (concrete and skim-readable; concise by cutting whole sentences
  rather than clipping words; full sentences with subjects; present tense only) now apply to all
  fourteen documentation edits and to the PR body. Note the last one especially: several of the
  passages this change rewrites are exactly the "what it used to do" shape the rule bans.
- **Never hard-wrap a GitHub issue or PR body.** Relevant because the migration commands and the
  stats-key explanation go in the PR body.

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
`_load_engineering_inputs` derives the NWP cell filter from the roster (`cv_assets.py:330`, feeding
`h3_index.is_in(cells)` at `:338-344`) while the population comes from the promoted model's
`trained_ids` (`production_assets.py:238`), and `live_forecasts` raises outright on an empty result
(`production_assets.py:285-290`) — so a thin roster costs, at best, forecasts for the missing series
and, at worst, the whole slot. `forecast_metrics` (`cv_assets.py:1000`) and `cv_power_forecasts`
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

**Delta is stricter about dtypes than parquet, and this is the conversion's largest hidden cost** —
three separate ways, all verified, and the reason the write and read paths both belong behind helpers
rather than being open-coded per caller:

- **`pl.Enum` cannot be written at all.** Not a rejection but a rust panic surfaced as a `DeltaError`
  (fact 1). Parquet round-trips Enums perfectly, so this bites only on conversion, and it bites the
  create path hardest.
- **A `Null`-dtype column is refused outright** (it raised on the first attempt at the benchmark
  above), where parquet stores one happily. Any all-null column must carry a concrete dtype, which
  Patito's `.cast()` already ensures on every path that goes through the contract.
- **Unsigned integers do not exist in the Delta protocol.** `UInt64` comes back `Int64`, silently, so
  `H3GridWeights` and `TimeSeriesMetadata.h3_res_5` need a cast on every read.

The pattern across all three: the round trip is lossy in the *type* domain while being exact in the
value domain, so nothing fails loudly at the boundary unless a contract validates there. Which is the
argument for `scan_delta` classmethods over `pl.read_delta` at each call site.

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
   roster (`checks.py:196-204`), so a dropped id keeps being flagged stale. What a thin roster really
   does is stop never-seen ids being reported at all (`checks.py:205-208`) and shrink
   `n_series_total` (`:221-226`).
3. **A thin roster can fail `live_forecasts` outright**, not merely thin it — now D3, and the reason
   the repair pass exists.
4. **Rule 12 restated principle 10**, which `inherent-stability.md:153-159` requires to be marked and
   paired. Resolved differently here: rule 12 now drops the atomicity clause entirely, because the
   store delivers it.
5. **A fixed `.tmp` suffix was unsafe for two writers** (nothing serialises this asset: no `pool=`,
   and `default_limit: 1` governs pooled ops only). Moot now — delta-rs commits via conditional put
   on S3 and an atomic rename locally, so concurrency safety comes from the store.
6. **Docstrings and pages the change falsifies** — `_sentry.py:98-99`, `production-deployment.md:179`,
   `checks.py:340-341`, and `_uri.py:7-11`. The first three are in the docs list above; the `_uri.py`
   one is moot, since the conversion adds nothing to that module.
7. **The R&D readers of the roster** (`cv_assets.py:1000`, `cv_power_forecasts`) were missing from
   the impact list. Folded into D3.
8. **Line-reference drift** — `validate(existing_metadata)` is at 408, not 407. Corrected.
9. **`os.replace` changes the inode**, so a symlinked roster path would be replaced by a plain file.
   Moot with Delta.

Considered and **partly rejected**: the review read the asset-level `except BaseException` as newly
swallowing the strict-contract boundary on *incoming* metadata. Rejected as stated —
`_extract_time_series_metadata` validates and raises at `read_nged_json.py:58`, outside the guard, so
an NGED contract break still fails the run. The narrower point (a bug in our own upsert code is now
degraded rather than raised) is real and is stated in the design-philosophy check above.

### What the adversarial review of the Delta version found

A second fresh sub-agent reviewed the rewrite, again with no knowledge of the reasoning behind it, and
was asked specifically to re-run the Delta measurements rather than trust them. **Every one of its 16
findings was verified as real** and is folded in above; the load-bearing ones were reproduced
independently before being accepted. It also re-ran every empirical claim the plan makes — the
parenthesised predicate, the no-op-commits-nothing property, the narrow-source insert, the
counts-not-ids metrics, the `Null`-dtype refusal, D1's "(B) happens by accident" — and all of them
held, along with four the plan depended on without stating: `merge` does respect `writer_properties`,
`IS DISTINCT FROM` is correct in both null directions, a `UInt64`-vs-`Int64` predicate comparison is
safe, and time travel does survive `schema_mode="overwrite"`.

The findings that changed the design rather than the prose:

1. **`write_delta` cannot store the roster at all.** `TimeSeriesMetadata` has four `pl.Enum` columns,
   and delta-rs aborts a rust worker on them. Blocking, and the plan had no mention of it despite the
   repo already carrying both the workaround (`cv_assets.py:750`) and a skill documenting the trap.
   Now fact 1 of `metadata.py`'s docstring, and the reason test 4 must use a fully-typed roster:
   `MERGE` tolerates an Enum source, so a suite that only merges would have passed while the create
   path panicked in production. Reproduced independently.
2. **The documented migration one-liner fails**, for that reason — parquet preserves the Enum dtype on
   round-trip. The migration section is rewritten to go through the new writer, to verify with
   `validate` rather than a row count, and to update the env vars.
3. **`MERGE` destroys row order**, and the plan had silently dropped today's `.sort("time_series_id")`
   while presenting the change as a pure storage conversion. Reproduced. Now stated, documented in both
   module docstrings, and pinned by test 6.
4. **The create branch had no repair pass**, though it is structurally identical to the rebuild and far
   more likely to fire (a skipped migration). Fixed with a sharper discriminator than "created or
   rebuilt": whether the run's file listing was narrowed.
5. **A predicate naming a column the *target* lacks raises**, which would have turned a routine
   contract addition — or a migration from a narrow parquet file — into a silent roster rebuild plus a
   full re-download. Now fact 5 and test 5. Reproduced by accident while checking finding 3.
6. **"Every assertion below fails on `main` today" was false**: `test_upsert_metadata_returns_diff`
   already exists and passes. Item 8 is now honestly labelled a port.
7. **The rebuild path's data loss is permanent, not "~5 h"** — `select_new_rows` never re-reads a file
   already represented in the power table, so a quiet series' metadata row cannot come back on its own.
   The runbook needs a manual re-derivation step, which is now stated as the only route back.
8. **`metadata_n_new_TimeSeriesIDs` would have lied on exactly the paths that matter**, reporting the
   whole snapshot as "new" at the moment the roster was thinned. The rebuild and create paths now report
   their own key.
9. **The roster read had two incompatible jobs** — gate and annotation — in adjacent bullets. Separated
   explicitly, since an implementer would otherwise have picked one at random.
10. **Missed sites**: five roster-writing test helpers, four documentation spots (including
    `performance.md`, the page that exists to record writer-properties rationale), and three read sites
    that are eager and so not the mechanical swap the plan claimed. All added.

Two sub-claims were **narrowed rather than accepted as stated**. The review called the stale-env-var
failure "silent and permanent" — it is permanent, but not silent: the asset guard sends a Sentry event
and the freshness check drops to `WARN`, so it pages, and the migration runbook is the fix. And it
listed items 4, 9 and 14 as further instances of the false "fails on `main`" claim; they are new tests
for new modules, which the Tests section now says plainly rather than implying a regression.
