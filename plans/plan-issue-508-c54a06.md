# Plan — #508: an unreadable `metadata.parquet` must not wedge the hourly ingest

Issue: <https://github.com/openclimatefix/nged-substation-forecast/issues/508>
Branch: `claude/plan-issue-508-c54a06`

## Verdict: worth implementing, roughly as described

Both halves of the issue's premise reproduce on `main`, and the issue is right that this needs a
decision rather than a `try`. Verified in this worktree:

- A junk `metadata.parquet` makes `pl.read_parquet` raise
  `polars.exceptions.ComputeError: parquet: File out of specification: The file must end with PAR1`
  — exactly the message the issue quotes, and it comes out of `upsert_metadata`
  (`packages/nged_data/src/nged_data/storage.py:405`), inside the asset op.
- The corruption mechanism is real and specific to a **local** path: a subprocess writing a large
  frame with `write_parquet` straight to the live path, killed with `SIGKILL` once it had started
  emitting bytes, left a 256 KB file that no longer parses. The same experiment writing to
  `<path>.tmp` and then `os.replace`-ing left the original file **intact and readable**, with the
  part-written temp file beside it.

Two things the issue does not say, both found while verifying it, and both of which the fix should
cover because they wedge the ingest in exactly the same way:

1. **`TimeSeriesMetadata.validate(existing_metadata)` (`storage.py:407`) is a second, independent
   raiser on the same code path.** An existing roster that no longer satisfies the contract raises
   `patito.exceptions.DataFrameValidationError` — verified with a roster missing `substation_type`
   ("Missing column"), and Patito rejects superfluous columns too. So the *next time we add or
   rename a `TimeSeriesMetadata` field*, every deployment's roster wedges the hourly ingest with a
   different exception and the same symptom. "The existing roster is unusable" is one condition with
   two causes, and it wants one recovery path.
2. **An unreadable roster also breaks `live_forecasts`**, via the unguarded
   `pl.read_parquet(settings.metadata_path)` in `_load_engineering_inputs`
   (`src/nged_substation_forecast/defs/cv_assets.py:322`). That is *off* the degradation ladder — no
   forecast at all, not even rung 4 — but fixing it properly means separating the production caller
   from the fail-fast CV caller, which is a different change. **Out of scope; see open question Q6.**

This is an [H1](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/engineering-hypotheses/#h1-a-service-that-mostly-runs-itself)
bug in the most direct sense. Today's recovery is "someone deletes the file by hand", which is a
**T1.1** intervention that is not an upstream format change, and two of the five scripted **T1.4**
game-day failures — "disk full" and "daemon killed" — are precisely the events that produce a
half-written roster. `ENOSPC` mid-write is the likelier cause than a kill, incidentally: the roster
is small (50 KB on the dev box), so the kill window is microseconds, whereas a full disk truncates
the write deterministically.

### Departures from the issue body

- **Do options 1 and 3 together, not one of the three.** Option 1 (atomic write) fixes the cause but
  cannot un-wedge a file that is already corrupt, and does nothing about the contract-drift raiser
  above; options 2/3 fix the symptom but leave the cause in place. Prevention plus automatic,
  noisy recovery is the pair that makes the failure survivable *and* rare.
- **Option 3 over option 2** for the recovery half: copy the unreadable file aside before
  rebuilding, so the bad bytes are available for diagnosis instead of being destroyed by the very
  write that recovers from them.
- **Atomic write for local paths only, and deliberately not for remote URIs.** The issue notes that
  "rename is not atomic on S3" and treats that as a cost of option 1. It is instead a reason not to
  do it there: an S3 `PUT` publishes the whole object or nothing, and a multipart upload only
  materialises on `CompleteMultipartUpload`, so no half-written object is ever visible to a reader.
  A temp key plus copy-and-delete would *add* a failure mode. (Our roster is ~50 KB, far below the
  5 MB multipart threshold, so it is a single `PUT`.)
- **The asset keeps its current order** (roster upsert, then the power write). The issue asks that
  "the power time series write should not be blocked by a roster failure", and a guard around the
  upsert call delivers that without reordering; see [What changes](#what-changes-file-by-file) for
  the one consequence that ordering has either way.

## What changes, file by file

### `packages/contracts/src/contracts/_uri.py`

The module already owns the local-or-remote-aware helpers that wrap every Delta/parquet write
(`object_exists`, `if_local_path_then_make_parent_dir`), so both new helpers go here. Adds imports
of `os`, `shutil` and `polars as pl`.

- **`write_parquet_atomically(df: pl.DataFrame, uri: str, storage_options: ObjectStoreOptions | None = None) -> None`**
  — for a local `uri`, write to `<uri>.tmp` and then `os.replace` onto `uri` (same directory, so the
  same filesystem, so the rename is atomic); for a remote URI, write straight to it. Fixes
  `compression="zstd"` inside, making this the single place the physical write policy for our bare
  parquet files lives — the counterpart to `delta_store`'s writer properties for Delta. (`zstd` is
  also Polars' default, so `h3_grid_weights` sees no change in what it writes.) The docstring must
  carry three pieces of reasoning: why remote needs no temp object, why there is deliberately no
  `fsync` (the failure modes in scope are `SIGKILL`, an OOM kill, `ENOSPC` and a process crash, all
  of which the page cache survives — not power loss, and a derived 50 KB roster does not justify
  the durability barrier), and that the fixed `.tmp` suffix bounds the debris to one stale file that
  the next write overwrites.
- **`copy_object(src_uri: str, dst_uri: str, storage_options: ObjectStoreOptions | None = None) -> None`**
  — `shutil.copyfile` for a local path, `obstore.copy(store, from_, to, overwrite=True)` for a
  remote one. Copy rather than move so one code shape serves both, and because the caller overwrites
  the source immediately afterwards anyway.

### `packages/nged_data/src/nged_data/storage.py`

- **`UpsertMetadataStats`** gains three optional keys (it is already `total=False`), all published
  straight into the asset's Dagster metadata: `metadata_roster_rebuilt_reason: str`,
  `metadata_unreadable_roster_copied_to: str`, and `metadata_upsert_failed: str` (set by the asset,
  not by this function).
- **New `_ExistingRoster` `NamedTuple`** — `frame: pt.DataFrame[TimeSeriesMetadata] | None`,
  `unreadable_reason: str | None`, `copied_to: str | None`. Matches the module's existing
  `DownloadAndParseResult` shape.
- **New `_read_existing_roster(metadata_path, storage_options) -> _ExistingRoster`** — reads and
  validates, returning the frame on success. On `Exception` it logs the traceback with
  `log.exception`, calls `_copy_unreadable_roster_aside`, and returns a frame of `None` with the
  reason (`repr(exc)`) filled in. Read *and* validate live behind this one call because "is the
  existing roster usable?" is one question: a `ComputeError` from corrupt bytes and a
  `DataFrameValidationError` from contract drift both mean no, and both want the same recovery.
  Catching `Exception` and **not** `BaseException` is deliberate and differs from the `checks.py`
  guards: a pyo3 panic is not evidence about *the file*, and overwriting the roster on that evidence
  is worse than skipping the update, so a panic falls through to the asset-level guard below, which
  degrades the upsert without rebuilding.
- **New `_copy_unreadable_roster_aside(metadata_path, storage_options) -> str | None`** — copies to
  `<metadata_path>.unreadable-<YYYYmmddTHHMMSSZ>` via `copy_object` and returns that path. Wrapped in
  its own `try`/`except Exception` that logs and returns `None`: quarantine is a diagnostic
  convenience and must never block the recovery it precedes.
- **`upsert_metadata`** — the bare `pl.read_parquet` + `validate` pair at lines 405–407 becomes a
  `_read_existing_roster` call. When the frame is `None`, skip the diff and the merge entirely and
  write `new_metadata` as the whole roster, returning
  `metadata_n_new_TimeSeriesIDs=new_metadata.height`, `metadata_n_updated_TimeSeriesIDs=0`, the
  reason and the quarantine path. Both write sites (the create branch at 394 and the merge write at
  438) become `write_parquet_atomically`, and the local `COMPRESSION: Final[str] = "zstd"` constant
  moves into that helper. The merge path is unchanged: after a successful `validate` the existing
  roster has exactly the contract's columns, so the `pl.concat` cannot raise a `ShapeError`.

  The rebuild's one real cost, which belongs in the docstring: `new_metadata` covers only the
  `time_series_id`s whose JSON files were *new this run*, not every id NGED publishes (the issue
  body claims otherwise), because `download_and_parse_files` is fed `list_of_new_json_files`. So a
  rebuild can thin the roster to a handful of ids. It self-heals within one NGED publication cycle
  (~5 h) as the remaining series publish and get merged back in; the cost meanwhile is that
  `power_data_is_fresh` reports the absent ids as "never reported" rather than "stale", and that a
  `live_forecasts` slot firing inside that window forecasts only the series still in the roster.
  That is strictly better than an ingest that never runs again, and it is why the quarantined copy
  matters: it is what a human restores from if the thinning is not acceptable in the moment.

### `src/nged_substation_forecast/defs/assets.py`

- **`power_time_series_and_metadata`** — the `upsert_metadata` call at line 125 goes under a
  `try`/`except BaseException` that re-raises `KeyboardInterrupt | SystemExit |
  DagsterExecutionInterruptedError` (the `checks.py` idiom, and the same reasoning: a pyo3
  `PanicException` from polars/delta-rs/obstore does not derive from `Exception`, and each extension
  compiles its own class). The handler logs the traceback, calls
  `report_asset_degradation("power_time_series_and_metadata", exc)`, and substitutes
  `UpsertMetadataStats(metadata_upsert_failed=repr(exc))` so the power write below still runs. Needs
  a `DagsterExecutionInterruptedError` import.
- Where the returned stats carry `metadata_roster_rebuilt_reason`, log at ERROR and call
  `report_asset_degradation` with the reason string — a rebuild is a state a human should look at,
  and without an explicit send it would reach nobody: this job has no cron monitor, and the run now
  succeeds so `sentry_capture_failure` does not fire (the #480/PR #511 argument, one level up).
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
  `report_check_degradation`. The union is what lets one function serve both call sites: the
  asset-level guard has a live exception, the rebuild path has only a reason (the exception was
  handled a layer down, inside `nged_data`, which must not depend on Sentry or Dagster). The shared
  `new_scope`/`try` body is extracted into a module-private helper so there are not two copies of
  it; `report_check_degradation`'s public signature and behaviour are unchanged. See Q3.

## Design-philosophy check

This code path is **production** — the hourly `power_time_series_and_metadata_job`, which carries
`sentry_capture_failure` — so it degrades rather than raises.

- **Rule 1** (never raise because an input is absent or stale): the roster is our own derived
  artifact rather than an outside input, but the shape is the rule's: the ingest now keeps running
  and records the degradation instead of stopping.
- **Rule 2** (liberal about missing, strict about malformed): the malformed roster is still rejected
  at the Patito boundary — it is never merged, never trusted, and copied aside as evidence. What
  changes is only that rejecting it no longer rejects the power data with it. That is the exact
  boundary the issue draws.
- **Rule 3** (treat detectably-wrong input as missing): this is the clean instantiation. An
  unreadable or contract-violating roster is detectably wrong, so it is treated as *absent*, which
  routes it into the same rebuild branch a first-ever run takes.
- **Rule 7** (a warning path may never fail the thing it warns about): unchanged and respected. No
  asset check is added or edited; `power_data_is_fresh` keeps `WARN`/`blocking=False` and its
  existing catch-all. The new reporting helper cannot raise, and the quarantine step is itself
  guarded so the recovery path has no raiser in it either.
- **Rules 6, 11**: untouched — no new check, no new cross-job run-status dependency.
- **[H1](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/engineering-hypotheses/#h1-a-service-that-mostly-runs-itself)**:
  removes a hand-intervention class from **T1.1** and covers two of the **T1.4** game-day failures
  (disk full, daemon killed) with automatic recovery.
- No design principle is traded away. Nothing moves into the serving path (rule 8): the recovery
  lives in the ingest, which is where the write is.

## Tests

Every assertion below fails on `main` today.

`packages/nged_data/tests/test_storage.py`:

1. `test_upsert_metadata_rebuilds_from_an_unreadable_existing_roster` — junk bytes at
   `metadata_path`, then `upsert_metadata(new_metadata, path)`. Asserts it returns rather than
   raising, the file now reads back as exactly `new_metadata`, the stats carry
   `metadata_roster_rebuilt_reason`, and no `.tmp` debris is left in the directory. On `main`:
   `ComputeError`.
2. `test_upsert_metadata_copies_the_unreadable_roster_aside` — same setup; asserts the path named in
   `metadata_unreadable_roster_copied_to` exists and its bytes are the original junk. On `main`:
   `ComputeError`.
3. `test_upsert_metadata_rebuilds_when_the_existing_roster_fails_its_contract` — the existing file is
   a *valid* parquet that is missing `substation_type`. Asserts a rebuild, not a
   `DataFrameValidationError`. On `main`: `DataFrameValidationError` (verified).
4. `test_upsert_metadata_leaves_the_existing_roster_intact_when_the_write_fails` — the atomicity
   test, made deterministic instead of racing a `SIGKILL`: monkeypatch `pl.DataFrame.write_parquet`
   with a fake that writes junk to whatever path it is handed and then raises `RuntimeError`.
   Asserts the `RuntimeError` propagates (the asset-level guard, not this function, is what absorbs
   it) *and* that `pl.read_parquet(metadata_path)` still returns the old roster. On `main` the fake
   junks the live path, so the read raises.
5. `test_upsert_metadata_rebuilds_even_if_the_quarantine_copy_fails` — monkeypatch
   `nged_data.storage.copy_object` to raise; asserts the rebuild still happens and the stats carry
   the reason with no quarantine path. On `main`: `ComputeError` before any of this is reached.

`tests/test_s3_data_paths.py` (moto, `integration`-marked; the per-test reset fixture already
handles moto's process-global backend):

6. `test_upsert_metadata_rebuilds_from_an_unreadable_object_on_s3` — `put` junk at the roster key,
   upsert, assert the rebuild and that the quarantine key exists. This is the only test that
   exercises `copy_object`'s remote branch and the remote (no-temp-object) write path. On `main`:
   `ComputeError`.

`tests/test_assets.py` (the existing `_FakeS3Store` + `env` harness):

7. `test_power_time_series_and_metadata_writes_power_when_the_roster_upsert_fails` — monkeypatch
   `assets.upsert_metadata` to raise `RuntimeError`; assert `result.success`, that the
   `power_time_series` Delta table has the fixture rows, and that `metadata_upsert_failed` is in the
   materialisation metadata. This is the issue's headline property. On `main`: the run fails.
8. `test_power_time_series_and_metadata_reports_a_roster_rebuild` — corrupt the roster before
   materialising; assert success, power rows written, the roster rebuilt to the fixtures' ids, and
   `metadata_roster_rebuilt_reason` present in the metadata. On `main`: the run fails.

`tests/test_sentry.py`:

9. Mirror the two existing `report_check_degradation` tests for `report_asset_degradation` — the
   built event carries `{"degraded_asset": ...}` and no tag leaks into the current or isolation
   scope; and a `capture_*` that raises is swallowed and logged, because it is called from inside
   the asset's own `except` handler. Add the message-shaped call as a third case. On `main`: the
   function does not exist.

`tests/test_checks.py`:

10. Not a new test — the docstring of
    `test_power_data_is_fresh_degrades_on_a_corrupt_metadata_parquet` (line ~336) states that
    "`upsert_metadata` reads the same file first and fails the asset outright", which this change
    makes false. Rewrite it to describe what now happens (the ingest rebuilds the roster, so the
    corrupt state is transient, and this test pins the check's own half of the guard).

## Docs to update

Written to describe how the code works now, per CLAUDE.md's "write about the present".

- **`docs/architecture/production-deployment.md`** — the passage at ~86 that offers "a half-written
  `metadata.parquet`" as the motivating example for the check's catch-all now describes a state the
  ingest prevents and self-heals; reword to an object-store error or an unreadable roster, and add a
  short subsection near the freshness-check one for the roster's own policy: written atomically,
  rebuilt from the current snapshot if unreadable or off-contract, bad file copied aside, and the
  power write no longer coupled to any of it.
- **`docs/live_service/operations.md`** — in "Degraded input data", the same "left half-written by a
  killed process" phrase at ~171 needs the same correction, plus a new operator paragraph: what
  `metadata_roster_rebuilt_reason` / `metadata_upsert_failed` in the asset's metadata mean, the
  `degraded_asset:power_time_series_and_metadata` Sentry filter, where the
  `metadata.parquet.unreadable-*` file is and that it is what you restore from, and that a
  rebuilt roster can briefly show ids as "never reported" in the freshness check. Both are
  next-business-day, not emergencies. It should also say to append to the intervention log, as the
  rest of that section does.
- **`packages/nged_data/README.md`** — the one-line `upsert_metadata` entry gains the rebuild
  behaviour.
- **`docs/design-philosophy/inherent-stability.md`** — one sentence in "Missing versus wrong" giving
  the roster as the second worked example of rule 3 (an unreadable derived artifact is detectably
  wrong, so it is treated as absent and rebuilt), and, subject to Q2, a new **rule 12**: write a
  derived artifact atomically, and treat an unreadable one as absent. Appended, so no existing rule
  is renumbered. It is a genuinely new rule rather than a restatement of a `design-principles.md`
  principle, so no matching principle needs changing.
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
uv run pytest tests/test_s3_data_paths.py packages/nged_data/tests/test_storage.py tests/test_assets.py -q
```

`mkdocs build --strict` matters here because four prose files change, two of them adding links —
read the rendered HTML for the edited sections, since the nested-list and wrapped-link traps pass
both linters (`mkdocs-authoring` skill). No `--run-network` run is needed: nothing touches the
Dynamical catalog or NWP conversion conventions.

## Risks and open questions

**Q1 — Is `h3_grid_weights` in scope?** It is the repo's only other in-place bare-parquet write, so
it has the identical fault (a corrupt weights file breaks the NWP spatial join), and once the helper
exists the fix is one line in the same file the plan already edits.
*Recommendation: yes, include it* — leaving a known landmine next to the one we are defusing costs
more than the line.

**Q2 — Append rule 12 to `inherent-stability.md`?** "Write a derived artifact atomically, and treat
an unreadable one as absent" is portable, is not implied by rules 1–11, and is the durable lesson of
this bug.
*Recommendation: yes, append it* — it is the kind of rule that only gets written after it bites.

**Q3 — `report_asset_degradation(name, detail: BaseException | str)`, or two functions?** The union
exists only because the rebuild path has no live exception at the asset boundary.
*Recommendation: keep the union* — one function, one tag, one docstring explaining both shapes, and
the alternative is two near-identical six-line functions.

**Q4 — Quarantine file naming.** Timestamped (`…unreadable-20260810T204500Z`) keeps every bad file;
a fixed suffix keeps only the latest and cannot accumulate.
*Recommendation: timestamped* — a recurrence is near-impossible once the rebuild has written a valid
file, the file is ~50 KB, and the evidence is the whole point.

**Q5 — Accept the roster thinning after a rebuild?** A rebuild writes only the ids that published
this run, self-healing over ~5 h, with a possible thin `live_forecasts` slot in between.
*Recommendation: accept it, and document it* — the alternative (re-download every JSON file to
re-derive the full roster) is far more expensive, and the quarantined copy is the escape hatch.

**Q6 — Follow-up issue for `_load_engineering_inputs`?** Its unguarded
`pl.read_parquet(settings.metadata_path)` means an unreadable roster also fails `live_forecasts`
outright — off the ladder entirely — for as long as the corruption lasts. This change shortens that
window to at most one hourly ingest, but does not close it; closing it means splitting the
production caller from the fail-fast CV caller.
*Recommendation: file a separate issue* rather than widening this one. Say the word and I will.

**Q7 — Should the roster be a Delta table instead?** That would give atomic commits, time travel in
place of the quarantine copy, and make this whole class of fault impossible, at the cost of
migrating ~6 read sites (`assets.py`, `cv_assets.py` ×2, `checks.py`, both dashboards, one
notebook) and of hourly small-file commits needing occasional `optimize`/`vacuum`.
*Recommendation: not in this issue* — the parquet fix is small, testable and complete for the fault
described. If the roster's *history* turns out to be wanted (metadata changes are currently
overwritten and lost), that is the argument for Delta, and it deserves its own issue.

**Residual risk — the S3 atomicity assumption.** The remote branch writes with no temp object on the
strength of `PUT`/multipart-completion atomicity. If that were wrong, an interrupted remote write
could corrupt the roster — but the recovery half of this change handles that case anyway, which is
the reason to ship both halves rather than either alone.

**Small risk — sibling files in the roster's directory/prefix.** `.tmp` and `.unreadable-*` files
now appear next to `metadata.parquet`. Nothing globs that directory: every reader
(`assets.py`, `cv_assets.py`, `checks.py`, both dashboards, the notebook) opens the exact
`settings.metadata_path`. Checked.
