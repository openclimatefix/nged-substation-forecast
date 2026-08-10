# Plan — #508: an unusable `metadata.parquet` must not wedge the hourly ingest

Issue: <https://github.com/openclimatefix/nged-substation-forecast/issues/508>
Branch: `claude/plan-issue-508-c54a06`

## Verdict: worth implementing, roughly as described

Both halves of the issue's premise reproduce on `main`, and the issue is right that this needs a
decision rather than a `try`. Verified in this worktree:

- A junk `metadata.parquet` makes `pl.read_parquet` raise
  `polars.exceptions.ComputeError: parquet: File out of specification: The file must end with PAR1`
  — exactly the message the issue quotes, and it comes out of `upsert_metadata`
  (`packages/nged_data/src/nged_data/storage.py:405`), inside the asset op. A 0-byte file gives the
  same error, which is the shape a full disk produces.
- The corruption mechanism is real and specific to a **local** path: a subprocess writing a large
  frame with `write_parquet` straight to the live path, killed with `SIGKILL` once it had started
  emitting bytes, left a 256 KB file that no longer parses. The same experiment writing to a temp
  file in the same directory and then `os.replace`-ing left the original file **intact and
  readable**, with the part-written temp file beside it.

Three things the issue does not say, all found while verifying it, and all of which the fix must
cover because they wedge the ingest in exactly the same way:

1. **`TimeSeriesMetadata.validate(existing_metadata)` (`storage.py:408`) is a second, independent
   raiser on the same code path.** An existing roster that no longer satisfies the contract raises
   `patito.exceptions.DataFrameValidationError` — verified with a roster missing `substation_type`
   ("Missing column"), and Patito rejects superfluous columns too. So the next time we *rename* a
   `TimeSeriesMetadata` field, every deployment's roster wedges the hourly ingest with a different
   exception and the same symptom. "The existing roster is unusable" is one condition with several
   causes, and it wants one recovery path.
2. **`pl.concat([new_metadata, existing_metadata])` (`storage.py:436`) is a *third* raiser, and this
   one is live today rather than hypothetical.** Four `TimeSeriesMetadata` fields are
   `allow_missing=True` (`information`, `area_wkt`, `area_center_lat`, `area_center_lon` —
   `packages/contracts/src/contracts/power_schemas.py:245-279`), so a frame that omits them
   validates cleanly. Verified: a 10-column roster and a 14-column snapshot both pass `validate`,
   and then `pl.concat` raises `ShapeError: unable to append to a DataFrame of width 14 with a
   DataFrame of width 12`; with the same columns in a different order it raises `unable to vstack,
   column names don't match`. This is reachable now, because `_extract_time_series_metadata` derives
   its columns from each JSON file's own keys (`read_nged_json.py:44-47`) and
   `download_and_parse_files` therefore has to union them with `how="diagonal"`
   (`storage.py:198`) — so this run's snapshot and the roster on disk can legitimately differ in
   width. It also means the *diff* is unreliable: `hash_rows` is column-order sensitive (verified),
   so a roster whose column order differs from the snapshot's shows every row as changed and gets
   rewritten every run. **A recovery keyed only on the read would never reach this raiser**, so the
   plan fixes the merge itself as well.
3. **An unusable roster also breaks `live_forecasts`**, via the unguarded
   `pl.read_parquet(settings.metadata_path)` in `_load_engineering_inputs`
   (`src/nged_substation_forecast/defs/cv_assets.py:322`). That is *off* the degradation ladder — no
   forecast at all, not even rung 4 — but fixing it properly means separating the production caller
   from the fail-fast CV caller, which is a different change. **Out of scope; see Q6.**

This is an [H1](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/engineering-hypotheses/#h1-a-service-that-mostly-runs-itself)
bug in the most direct sense. Today's recovery is "someone deletes the file by hand", which is a
**T1.1** intervention that is not an upstream format change, and two of the five scripted **T1.4**
game-day failures — "disk full" and "daemon killed" — are precisely the events that produce a
half-written roster. `ENOSPC` mid-write is the likelier cause than a kill, incidentally: the roster
is small (50 KB on the dev box), so the kill window is microseconds, whereas a full disk truncates
the write deterministically.

### Departures from the issue body

- **Do options 1 and 3 together, not one of the three.** Option 1 (atomic write) fixes the cause but
  cannot un-wedge a file that is already corrupt, and does nothing about the two contract/merge
  raisers above; options 2/3 fix the symptom but leave the cause in place. Prevention plus
  automatic, noisy recovery is the pair that makes the failure survivable *and* rare.
- **Option 3 over option 2** for the recovery half: copy the unusable file aside before rebuilding,
  so the bad bytes are available for diagnosis instead of being destroyed by the very write that
  recovers from them.
- **Atomic write for local paths only, and deliberately not for remote URIs.** The issue notes that
  "rename is not atomic on S3" and treats that as a cost of option 1. It is instead a reason not to
  do it there: an S3 `PUT` publishes the whole object or nothing, and a multipart upload only
  materialises on `CompleteMultipartUpload`, so no half-written object is ever visible to a reader.
  A temp key plus copy-and-delete would *add* a failure mode. (Our roster is ~50 KB, far below the
  5 MB multipart threshold, so it is a single `PUT`.)
- **A rebuild re-derives the *whole* roster rather than writing this run's snapshot.** The issue
  says "every id NGED is currently publishing is in `new_metadata`", so "the only loss is ids that
  have gone quiet". That is wrong: `download_and_parse_files` is fed `list_of_new_json_files`, so
  `new_metadata` covers only the ids whose JSON files were new *this run*. Writing that as the whole
  roster can thin it to a handful of ids, and a thin roster is not merely lossy — it can make
  `live_forecasts` **fail** (see Q5). The asset therefore follows a rebuild by re-deriving the
  roster from the most recent JSON file per series. Details below.
- **The asset keeps its current order** (roster upsert, then the power write). The issue asks that
  "the power time series write should not be blocked by a roster failure", and a guard around the
  upsert call delivers that without reordering; see the asset section for the one consequence that
  ordering has either way.

## What changes, file by file

### `packages/contracts/src/contracts/_uri.py`

The module already owns the local-or-remote-aware helpers that wrap every Delta/parquet write
(`object_exists`, `if_local_path_then_make_parent_dir`), so both new helpers go here. Adds imports
of `os`, `shutil` and `polars as pl`, and the module docstring's "the two things it does around
every Delta/parquet write" (`_uri.py:7-11`) becomes an accurate list of four.

`delta_store`'s charter says it owns *physical* layout while `contracts` owns *logical* shape
(`packages/delta_store/README.md:6-12`), so a parquet codec landing in `contracts` needs a word:
`delta_store` is Delta-specific by name and design, `nged_data` does not depend on it
(`packages/nged_data/pyproject.toml:6-12`), and these two helpers are about *where* a file is
(local vs. remote), which is `_uri.py`'s whole subject. Add the bare-parquet exception to
`delta_store/README.md`'s charter sentence so the split stays honest.

- **`write_parquet_atomically(df: pl.DataFrame, uri: str, storage_options: ObjectStoreOptions | None = None) -> None`**
  — for a local `uri`, write to `<uri>.<pid>.tmp` and then `os.replace` onto `uri` (same directory,
  so the same filesystem, so the rename is atomic); for a remote URI, write straight to it. Fixes
  `compression="zstd"` inside, making this the single place the physical write policy for our bare
  parquet files lives. (`zstd` is also Polars' default — verified on polars 1.43.2 — so
  `h3_grid_weights` sees no change in what it writes.) The temp name carries the **pid** rather than
  a fixed suffix because nothing serialises this asset: it has no `pool=`
  (`src/nged_substation_forecast/defs/assets.py:65`, contrast `pool="ECMWF"` at 243) and the
  deployment sets only `concurrency.pools.default_limit: 1` (`docs/getting-started.md:84-86`), which
  governs pooled ops — so the hourly schedule overlapping a manual materialisation gives two
  writers, and a *shared* temp name would let one publish the other's half-written file, through the
  very mechanism meant to prevent that. With per-pid temp files, two writers race only on the
  `os.replace`, where last-writer-wins and no reader sees a torn file. The docstring must carry:
  why remote needs no temp object; why there is deliberately no `fsync` (the failure modes in scope
  are `SIGKILL`, an OOM kill, `ENOSPC` and a process crash, all of which the page cache survives —
  not power loss, and a derived 50 KB roster does not justify the durability barrier); that debris
  is bounded to one file per pid that ever died mid-write; and that `os.replace` makes a new inode,
  so a symlinked or specially-permissioned `uri` is replaced by a plain file with default mode.
- **`copy_object(src_uri: str, dst_uri: str, storage_options: ObjectStoreOptions | None = None) -> None`**
  — `shutil.copyfile` for a local path, `obstore.copy(store, from_, to, overwrite=True)` for a
  remote one (signature verified on obstore 0.11.0). Copy rather than move so one code shape serves
  both, and because the caller overwrites the source immediately afterwards anyway.

### `packages/nged_data/src/nged_data/storage.py`

- **`UpsertMetadataStats`** gains three optional keys (it is already `total=False`), all published
  straight into the asset's Dagster metadata: `metadata_roster_rebuilt_reason: str`,
  `metadata_unusable_roster_copied_to: str`, and `metadata_upsert_failed: str` (set by the asset,
  not by this function).
- **New `_align_to_contract(df: pl.DataFrame) -> pt.DataFrame[TimeSeriesMetadata]`** — selects
  `TimeSeriesMetadata.columns` in declared order, supplying
  `pl.lit(None, dtype=TimeSeriesMetadata.dtypes[name])` for any column the frame lacks. Only the
  four `allow_missing=True` fields can be absent from a frame that validated, and all four are
  declared `| None`, so the filled nulls are contract-legal (verified). Applied to **both**
  `new_metadata` and the existing roster, which is what fixes raiser 2: equal widths and identical
  column order make the `pl.concat` unable to raise `ShapeError`, and make `hash_rows` an honest
  row comparison instead of an order-sensitive one that rewrites the file every run.
- **New `_ExistingRoster` `NamedTuple`** — `frame: pt.DataFrame[TimeSeriesMetadata] | None`,
  `unusable_reason: str | None`, `copied_to: str | None`. Matches the module's existing
  `DownloadAndParseResult` shape (`storage.py:134`).
- **New `_read_existing_roster(metadata_path, storage_options) -> _ExistingRoster`** — reads,
  validates, and aligns, returning the frame on success. On `Exception` it logs the traceback with
  `log.exception`, calls `_copy_unusable_roster_aside`, and returns a frame of `None` with the
  reason (`repr(exc)`) filled in. Read, validate *and* align live behind this one call because "is
  the existing roster usable?" is one question: corrupt bytes, contract drift and an unmergeable
  shape all mean no, and all want the same recovery. Catching `Exception` and **not**
  `BaseException` is deliberate and differs from the `checks.py` guards: a pyo3 panic is not
  evidence about *the file*, and overwriting the roster on that evidence is worse than skipping the
  update, so a panic falls through to the asset-level guard below, which degrades the upsert without
  rebuilding.
- **New `_copy_unusable_roster_aside(metadata_path, storage_options) -> str | None`** — copies to
  `<metadata_path>.unusable-<YYYYmmddTHHMMSSZ>` via `copy_object` and returns that path. Wrapped in
  its own `try`/`except Exception` that logs and returns `None`: quarantine is a diagnostic
  convenience and must never block the recovery it precedes.
- **`upsert_metadata`** — the bare `pl.read_parquet` + `validate` pair at 405–408 becomes a
  `_read_existing_roster` call. When the frame is `None`, skip the diff and the merge entirely and
  write the aligned `new_metadata` as the whole roster, returning
  `metadata_n_new_TimeSeriesIDs=new_metadata.height`, `metadata_n_updated_TimeSeriesIDs=0`, the
  reason and the quarantine path. Both write sites (the create branch at 394 and the merge write at
  438) become `write_parquet_atomically`, and the local `COMPRESSION: Final[str] = "zstd"` constant
  at 384 moves into that helper.

  What a rebuild costs, for the docstring: the roster is written from this run's snapshot alone, and
  `new_metadata` covers only the ids whose JSON files were new this run. The asset repairs that
  immediately (below); if the repair itself fails, the roster stays thin until the remaining series
  publish (~5 h). The consequence while it is thin is *not* the one the issue describes: `stale` is
  computed from the Delta table's coverage and is deliberately not restricted to the roster
  (`src/nged_substation_forecast/defs/checks.py:167-173`), so a dropped id keeps being flagged
  stale exactly as before. What actually changes is that an id with **no** power data at all
  disappears from the roster and so stops being reported as "never" at all
  (`checks.py:180-183`), and `n_series_total` shrinks (`checks.py:198-201`) — the rebuild removes
  signal rather than reclassifying it, which is why the repair below exists.

### `src/nged_substation_forecast/defs/assets.py`

- **`power_time_series_and_metadata`** — the `upsert_metadata` call at line 125 goes under a
  `try`/`except BaseException` that re-raises `KeyboardInterrupt | SystemExit |
  DagsterExecutionInterruptedError` (the `checks.py:318-337` idiom, and the same reasoning: a pyo3
  `PanicException` from polars/delta-rs/obstore does not derive from `Exception`, and each extension
  compiles its own class). The handler logs the traceback, calls
  `report_asset_degradation("power_time_series_and_metadata", exc)`, and substitutes
  `UpsertMetadataStats(metadata_upsert_failed=repr(exc))` so the power write below still runs. Needs
  a `DagsterExecutionInterruptedError` import.
- **New: repair a rebuilt roster in the same run.** Where the returned stats carry
  `metadata_roster_rebuilt_reason`, log at ERROR, call `report_asset_degradation` with the reason,
  and then re-derive the full roster: take the newest JSON file per `time_series_id` from
  `list_of_large_json_files` (the *all-files* listing the asset already has, before
  `select_new_rows` narrows it), run it back through `download_and_parse_files`, and
  `upsert_metadata` the resulting metadata — which now merges into the valid file just written. The
  power frame that comes back is discarded; `select_new_rows` on the Delta table already rejects
  those rows, and re-parsing them is CPU-only. This whole block sits inside its own guard: if it
  fails, the thin-but-valid roster from the first upsert stands and the run still succeeds, so the
  repair can only improve on the simple rebuild. It costs one full pass over NGED's files (32 today,
  ~2,500 at V2 scale — the same work a first-ever backfill does) on a path that should fire once in
  the project's life.
- **The consequence worth writing down in a comment**, because it is a genuine trade and holds
  whichever order the two writes go in: the power Delta table is what `select_new_rows` uses to
  decide which JSON files are new, so once the power rows land, a *failed* roster update is not
  retried — that run's metadata change is lost until NGED republishes those series (~5 h). Losing
  one refresh of derived, re-delivered data is much cheaper than blocking the power stream, and it
  is the second reason the failure must reach Sentry rather than only the logs.
- **`h3_grid_weights`** — `weights.write_parquet(...)` at line 167 becomes
  `write_parquet_atomically(...)`. Same fault class, same file, one line; see Q1.

### `src/nged_substation_forecast/_sentry.py`

- **New `report_asset_degradation(asset_name: str, detail: BaseException | str) -> None`** — tags
  `degraded_asset` on a forked scope and sends `capture_exception` for an exception or
  `capture_message(..., level="error")` for a string, never raising, exactly like
  `report_check_degradation` (`_sentry.py:133-161`). The union is what lets one function serve both
  call sites: the asset-level guard has a live exception, the rebuild path has only a reason (the
  exception was handled a layer down, inside `nged_data`, which must not depend on Sentry or
  Dagster). The shared `new_scope`/`try` body is extracted into a module-private helper so there are
  not two copies of it; `report_check_degradation`'s public signature and behaviour are unchanged.
  See Q3.
- `init_sentry`'s docstring enumerates "the three explicit senders" (`_sentry.py:98-99`); it becomes
  four.

## Design-philosophy check

This code path is **production** — the hourly `power_time_series_and_metadata_job`, which carries
`sentry_capture_failure` (`defs/schedules.py:19`) — so it degrades rather than raises.

- **Rule 1** (never raise because an input is absent or stale): the roster is our own derived
  artifact rather than an outside input, but the shape is the rule's: the ingest now keeps running
  and records the degradation instead of stopping.
- **Rule 2** (liberal about missing, strict about malformed): the unusable roster is still rejected
  at the Patito boundary — never merged, never trusted, and copied aside as evidence. What changes
  is only that rejecting it no longer rejects the power data with it. That is the exact boundary the
  issue draws.
- **Rule 3** (treat detectably-wrong input as missing): the clean instantiation. An unreadable,
  off-contract or unmergeable roster is detectably wrong, so it is treated as *absent*, which routes
  it into the same rebuild branch a first-ever run takes.
- **Rule 7** (a warning path may never fail the thing it warns about): unchanged and respected. No
  asset check is added or edited; `power_data_is_fresh` keeps `WARN`/`blocking=False` and its
  existing catch-all. The new reporting helper cannot raise, and both the quarantine step and the
  roster repair are individually guarded, so the recovery path has no raiser in it either.
- **What the asset-level guard trades away, stated rather than glossed:** it wraps the whole
  `upsert_metadata` call, whose first statement is `TimeSeriesMetadata.validate(new_metadata…)`
  (`storage.py:386`), so in principle a contract violation — which rule 1 says *should* raise, being
  our own bug — is now degraded to a Sentry event. In practice the strict-contract boundary on
  incoming data is untouched: `_extract_time_series_metadata` already validates every file's
  metadata and raises at `read_nged_json.py:58`, inside `download_and_parse_files`, which is
  **outside** this guard, so a genuine NGED contract break still fails the run there. What the
  guard really absorbs is a bug in our own diff/merge/stats code, and absorbing that is the price of
  the property the issue asks for. Narrowing the guard to the roster read/merge/write alone would
  reintroduce the wedge for anything outside it, which is not a good trade.
- **Rules 6, 11**: untouched — no new check, no new cross-job run-status dependency.
- **[H1](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/engineering-hypotheses/#h1-a-service-that-mostly-runs-itself)**:
  removes a hand-intervention class from **T1.1** and covers two of the **T1.4** game-day failures
  (disk full, daemon killed) with automatic recovery.
- **Principle 10** ("every write is atomic and idempotent",
  `docs/design-philosophy/design-principles.md:387`) is the principle this *serves* rather than
  trades away — it is currently satisfied only where Delta Lake provides it, and the roster is the
  one live counter-example. Its *Decided* paragraph (`design-principles.md:411-419`) credits Delta
  alone and needs the bare-parquet case added. Nothing moves into the serving path (rule 8): the
  recovery lives in the ingest, which is where the write is.

## Tests

Every assertion below fails on `main` today.

`packages/nged_data/tests/test_storage.py`:

1. `test_upsert_metadata_rebuilds_from_an_unreadable_existing_roster` — junk bytes at
   `metadata_path`, then `upsert_metadata(new_metadata, path)`. Asserts it returns rather than
   raising, the file now reads back as exactly the aligned `new_metadata`, the stats carry
   `metadata_roster_rebuilt_reason`, and no `.tmp` debris is left in the directory. On `main`:
   `ComputeError`.
2. `test_upsert_metadata_copies_the_unusable_roster_aside` — same setup; asserts the path named in
   `metadata_unusable_roster_copied_to` exists and its bytes are the original junk. On `main`:
   `ComputeError`.
3. `test_upsert_metadata_rebuilds_when_the_existing_roster_fails_its_contract` — the existing file is
   a *valid* parquet that is missing `substation_type` (a required field). Asserts a rebuild, not a
   `DataFrameValidationError`. On `main`: `DataFrameValidationError` (verified).
4. `test_upsert_metadata_merges_a_roster_that_omits_the_optional_columns` — the existing file is a
   valid 10-column roster (no `information`/`area_wkt`/`area_center_*`), the snapshot is 14-column.
   Asserts a normal *merge* — not a rebuild — with the optional columns null-filled for the old
   rows, and that `metadata_roster_rebuilt_reason` is absent. On `main`: `ShapeError` (verified).
5. `test_upsert_metadata_does_not_rewrite_when_only_the_column_order_differs` — the existing file
   holds the same data with columns in a different order. Asserts
   `metadata_n_updated_TimeSeriesIDs == 0` and that the file's mtime/bytes are unchanged. On `main`:
   `hash_rows` is order-sensitive (verified), so either it reports a spurious update or raises
   `ShapeError` on the vstack.
6. `test_upsert_metadata_leaves_the_existing_roster_intact_when_the_write_fails` — the atomicity
   test, made deterministic instead of racing a `SIGKILL`: monkeypatch `pl.DataFrame.write_parquet`
   with a fake that writes junk to whatever path it is handed and then raises `RuntimeError`.
   (`pt.DataFrame` inherits the method, so patching `pl.DataFrame` catches the patito frame too.)
   Asserts the `RuntimeError` propagates (the asset-level guard, not this function, is what absorbs
   it) *and* that `pl.read_parquet(metadata_path)` still returns the old roster. On `main` the fake
   junks the live path, so the read raises.
7. `test_upsert_metadata_rebuilds_even_if_the_quarantine_copy_fails` — monkeypatch
   `nged_data.storage.copy_object` to raise; asserts the rebuild still happens and the stats carry
   the reason with no quarantine path. On `main`: `ComputeError` before any of this is reached.

`tests/test_s3_data_paths.py` (moto, `integration`-marked; the per-test reset fixture at
`tests/test_s3_data_paths.py:85-101` already handles moto's process-global backend):

8. `test_upsert_metadata_rebuilds_from_an_unreadable_object_on_s3` — `put` junk at the roster key,
   upsert, assert the rebuild and that the quarantine key exists. This is the only test that
   exercises `copy_object`'s remote branch and the remote (no-temp-object) write path. On `main`:
   `ComputeError`.

`tests/test_assets.py` (the existing `_FakeS3Store` + `env` harness):

9. `test_power_time_series_and_metadata_writes_power_when_the_roster_upsert_fails` — monkeypatch
   `assets.upsert_metadata` to raise `RuntimeError`; assert `result.success`, that the
   `power_time_series` Delta table has the fixture rows, and that `metadata_upsert_failed` is in the
   materialisation metadata. This is the issue's headline property. On `main`: the run fails.
10. `test_power_time_series_and_metadata_repairs_a_rebuilt_roster` — with the fake store serving two
    series, ingest once, then corrupt the roster and arrange for only *one* series to have a new
    file. Asserts success, and that the roster ends up holding **both** ids — i.e. the repair pass
    ran and the roster was not left thin — plus `metadata_roster_rebuilt_reason` in the metadata. On
    `main`: the run fails.
11. `test_power_time_series_and_metadata_survives_a_failed_roster_repair` — as above but with the
    repair's `download_and_parse_files` raising. Asserts the run still succeeds and the thin roster
    from the first upsert is intact. On `main`: the run fails.

`tests/test_sentry.py`:

12. Mirror the two existing `report_check_degradation` tests (`tests/test_sentry.py:176`, `:220`) for
    `report_asset_degradation` — the built event carries `{"degraded_asset": …}` and no tag leaks
    into the current or isolation scope; and a `capture_*` that raises is swallowed and logged,
    because it is called from inside the asset's own `except` handler. Add the message-shaped call
    as a third case. On `main`: the function does not exist.

`tests/test_checks.py`:

13. Not a new test — the docstring of
    `test_power_data_is_fresh_degrades_on_a_corrupt_metadata_parquet` (line ~336) states that
    "`upsert_metadata` reads the same file first and fails the asset outright", which this change
    makes false. Rewrite it to describe what now happens (the ingest rebuilds the roster, so the
    corrupt state is transient, and this test pins the check's own half of the guard).

## Docs to update

Written to describe how the code works now, per CLAUDE.md's "write about the present".

- **`docs/architecture/production-deployment.md`** — three places. The passage at ~86 that offers "a
  half-written `metadata.parquet`" as the motivating example for the check's catch-all now describes
  a state the ingest prevents and self-heals; reword to an object-store error or an unusable roster.
  The Sentry section at ~174-179 says the only fault the failure hook cannot see is "a standalone
  `@asset_check` that caught its own exception … which, by design, is both of them" — an *asset* now
  catches its own exception too, so that paragraph needs the `degraded_asset` tag alongside
  `asset_check`. And add a short subsection near the freshness-check one for the roster's own
  policy: written atomically, rebuilt and repaired if unusable, bad file copied aside, and the power
  write no longer coupled to any of it.
- **`src/nged_substation_forecast/defs/checks.py:311-313`** — `power_data_is_fresh`'s own docstring
  uses the same "a half-written `metadata.parquet`" example; correct it with the other two rather
  than leaving one behind.
- **`docs/live_service/operations.md`** — in "Degraded input data", the same "left half-written by a
  killed process" phrase at ~171 needs the same correction, plus a new operator paragraph: what
  `metadata_roster_rebuilt_reason` / `metadata_upsert_failed` in the asset's metadata mean, the
  `degraded_asset:power_time_series_and_metadata` Sentry filter, where the
  `metadata.parquet.unusable-*` file is and that it is what you restore from, and that a rebuilt
  roster whose repair pass failed can stop reporting never-seen series in the freshness check until
  they publish again. Both are next-business-day, not emergencies. It should also say to append to
  the intervention log, as the rest of that section does.
- **`packages/nged_data/README.md`** — the one-line `upsert_metadata` entry gains the rebuild
  behaviour.
- **`packages/delta_store/README.md:6-12`** — the charter sentence gains the bare-parquet exception
  (see the `_uri.py` section above).
- **`packages/contracts/src/contracts/_uri.py:7-11`** and
  **`src/nged_substation_forecast/_sentry.py:98-99`** — both module docstrings enumerate things this
  change adds to ("the two things it does", "the three explicit senders").
- **`docs/design-philosophy/inherent-stability.md`** — one sentence in "Missing versus wrong" giving
  the roster as the second worked example of rule 3 (an unusable derived artifact is detectably
  wrong, so it is treated as absent and rebuilt), and, subject to Q2, a new **rule 12**: write a
  derived artifact atomically, and treat an unusable one as absent. Appended, so no existing rule is
  renumbered. Its first clause restates
  [principle 10](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/design-principles/#10-every-write-is-atomic-and-idempotent-and-every-failure-is-confined-to-one-partition),
  so it must be **marked as a restatement** the way rules 1, 2 and 8 are (`inherent-stability.md:155-159`
  requires the marking, and requires changing the principle alongside it).
- **`docs/design-philosophy/design-principles.md:411-419`** — principle 10's *Decided* paragraph
  credits Delta Lake alone for atomicity; add the one non-Delta table we write and how it gets the
  same property.
- No roadmap page completes here, so there is no "Implementation details" section to delete and no
  status banner to move. `#508` is not referenced from any `docs/` page.

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
uv run pytest tests/test_s3_data_paths.py tests/test_assets.py tests/test_sentry.py packages/nged_data/tests/test_storage.py -q
```

`mkdocs build --strict` matters here because six prose files change, several adding links — read the
rendered HTML for the edited sections, since the nested-list and wrapped-link traps pass both
linters (`mkdocs-authoring` skill). No `--run-network` run is needed: nothing touches the Dynamical
catalog or NWP conversion conventions.

## Risks and open questions

**Q1 — Is `h3_grid_weights` in scope?** It is the repo's only other in-place bare-parquet write, so
it has the identical fault (a corrupt weights file breaks the NWP spatial join), and once the helper
exists the fix is one line in the same file the plan already edits.
*Recommendation: yes, include it* — leaving a known landmine next to the one we are defusing costs
more than the line.

**Q2 — Append rule 12 to `inherent-stability.md`?** "Write a derived artifact atomically, and treat
an unusable one as absent" is the durable lesson of this bug; its first half restates principle 10
(and would be marked as such), its second half is new.
*Recommendation: yes, append it* — the second clause is the part rules 1–11 do not imply, and it is
the one that would have prevented this issue.

**Q3 — `report_asset_degradation(name, detail: BaseException | str)`, or two functions?** The union
exists only because the rebuild path has no live exception at the asset boundary.
*Recommendation: keep the union* — one function, one tag, one docstring explaining both shapes, and
the alternative is two near-identical six-line functions.

**Q4 — Quarantine file naming.** Timestamped (`…unusable-20260810T204500Z`) keeps every bad file; a
fixed suffix keeps only the latest and cannot accumulate.
*Recommendation: timestamped* — a recurrence is near-impossible once the rebuild has written a valid
file, the file is ~50 KB, and the evidence is the whole point.

**Q5 — Is the roster-repair pass worth its complexity, or should a rebuild just write the thin
snapshot?** The thin roster is not benignly lossy. `_load_engineering_inputs` derives the NWP filter
from the roster (`cells = metadata_df["h3_res_5"].unique()`, `cv_assets.py:328`, feeding the
`h3_index.is_in(cells)` filter at 336-342) while the population comes from the promoted model's
`trained_ids` (`production_assets.py:238`), and `live_forecasts` raises outright on an empty result
(`production_assets.py:285-290`). So a thin roster costs, at best, forecasts for the missing series
and, at worst, the whole slot. `forecast_metrics` (`cv_assets.py:997-1000`) and
`cv_power_forecasts` read the roster too, where thinning would silently shrink a training or
scoring population — the opposite of the fail-fast posture R&D is supposed to have.
*Recommendation: keep the repair pass* — it is ~5 lines over existing functions, it is guarded so it
cannot make things worse, and it turns "degrade and wait ~5 h" into an actual repair.

**Q6 — Follow-up issue for `_load_engineering_inputs`?** Its unguarded
`pl.read_parquet(settings.metadata_path)` means an unusable roster also fails `live_forecasts`
outright — off the ladder entirely — for as long as the corruption lasts. This change shortens that
window to at most one hourly ingest, but does not close it; closing it means splitting the
production caller from the fail-fast CV caller.
*Recommendation: file a separate issue* rather than widening this one. Say the word and I will.

**Q7 — Should the roster be a Delta table instead?** That would give atomic commits, time travel in
place of the quarantine copy, and make this whole class of fault impossible, at the cost of
migrating five read sites (`cv_assets.py` ×2, `checks.py`, both dashboards, one notebook) and of
hourly small-file commits needing occasional `optimize`/`vacuum`.
*Recommendation: not in this issue* — the parquet fix is small, testable and complete for the fault
described. If the roster's *history* turns out to be wanted (metadata changes are currently
overwritten and lost), that is the argument for Delta, and it deserves its own issue.

**Residual risk — the S3 atomicity assumption.** The remote branch writes with no temp object on the
strength of `PUT`/multipart-completion atomicity. If that were wrong, an interrupted remote write
could corrupt the roster — but the recovery half of this change handles that case anyway, which is
the reason to ship both halves rather than either alone.

**Small risk — sibling files in the roster's directory/prefix.** `.<pid>.tmp` and `.unusable-*` files
now appear next to `metadata.parquet`. Nothing globs that directory: every reader (`cv_assets.py`,
`checks.py`, both dashboards, the notebook) opens the exact `settings.metadata_path`. Checked.

## What the adversarial review changed

A fresh sub-agent reviewed the first draft with no knowledge of the reasoning behind it. Findings
verified against the code and accepted:

1. **The `pl.concat` `ShapeError` (now raiser 2 above).** The draft asserted that "after a
   successful `validate` … the `pl.concat` cannot raise a `ShapeError`". False: four fields are
   `allow_missing=True`, so validate passes on a narrower frame and the concat then raises — and a
   recovery keyed only on the read never reaches it. This was the review's headline finding and it
   added `_align_to_contract`, two tests (4 and 5), and the `hash_rows` order-sensitivity
   consequence.
2. **The freshness-check consequence was backwards**, inherited from the issue body and destined for
   the operator runbook. `stale` is computed from coverage and deliberately *not* restricted to the
   roster (`checks.py:167-173`), so a dropped id keeps being flagged stale; what a rebuild really
   does is stop never-seen ids being reported at all. Corrected in the plan and in the
   `operations.md` text it specifies.
3. **A thin roster can fail `live_forecasts` outright**, not merely thin it
   (`production_assets.py:285-290` raises on 0 rows, and the NWP cell filter comes from the roster).
   The draft stated only the benign version. This is what promoted the roster-repair pass from "not
   considered" to part of the plan (Q5).
4. **Rule 12 restates principle 10** ("every write is atomic and idempotent",
   `design-principles.md:387`); the draft claimed it restated no principle, which
   `inherent-stability.md:155-159` requires to be marked and paired.
5. **The fixed `.tmp` suffix was unsafe for two writers.** Nothing serialises this asset (no
   `pool=`; `default_limit: 1` governs pooled ops only), so a shared temp name lets one writer
   publish another's half-written file. Now per-pid.
6. **Four more docstrings/pages the change falsifies** — `_sentry.py:98-99`'s "three explicit
   senders", `production-deployment.md:174-179`'s "both of them", `checks.py:311-313`'s own
   half-written-`metadata.parquet` example, and `_uri.py:7-11`'s "the two things it does" — plus
   `delta_store/README.md`'s charter sentence, since a bare-parquet codec now lives in `contracts`.
7. **The R&D readers of the roster** (`cv_assets.py:997-1000`, `cv_power_forecasts`) were missing
   from the impact list; a silently-thinned training or scoring population is exactly what R&D is
   supposed to fail fast on. Folded into Q5.
8. **Line-reference drift**: `validate(existing_metadata)` is at 408, not 407, and the read+validate
   pair spans 405–408; `assets.py` was wrongly listed among the roster's *read* sites in Q7.
9. **`os.replace` changes the inode**, so a symlinked or specially-permissioned roster path is
   replaced by a plain default-mode file. Accepted as a docstring note.

Considered and **partly rejected**:

- The review read the asset-level `except BaseException` as newly swallowing "the strict-contract
  boundary on *incoming* metadata". Rejected as stated: `_extract_time_series_metadata`'s
  `validate` raises at `read_nged_json.py:58`, inside `download_and_parse_files`, which is outside
  the guard — so an NGED contract break still fails the run. The narrower point (a bug in our own
  merge code is now degraded rather than raised) is real and is now stated explicitly in the
  design-philosophy check, along with why narrowing the guard is the worse trade.
