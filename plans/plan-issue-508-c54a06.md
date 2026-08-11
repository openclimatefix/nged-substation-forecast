# Plan — #508: stop an unusable metadata roster wedging the hourly power ingest

Issue: <https://github.com/openclimatefix/nged-substation-forecast/issues/508>
Branch: `claude/plan-issue-508-c54a06`

## Verdict: worth implementing, as about forty lines in the existing parquet path

The bug is real and reproduces. The fix is a guard around the roster read, a rebuild when the stored
roster is unusable, one Polars keyword, and a Sentry sender so the swallowed failure still reaches a
human. No format change, no migration, no new modules.

**Jack's decision, after two earlier drafts converted the roster to a Delta table:** close #508 with
the small fix, and judge the Delta conversion separately on its own merits rather than on this bug.
The in-run roster repair pass is dropped for now. The reasoning that produced that decision is at the
bottom, under "History of this plan", because it is the part most likely to be re-litigated.

## What reproduces on `main`, and what does not

Verified in this worktree. There are three raisers on the path, and **which of them can fire depends
on where the data root points**:

| # | Raiser | Reachable in production? |
|---|---|---|
| 1 | Torn write from a kill or `ENOSPC` (`storage.py:438`) | **No** |
| 2 | `TimeSeriesMetadata.validate(existing_metadata)` (`storage.py:408`) raises `DataFrameValidationError` when the stored roster no longer satisfies the contract | Yes |
| 3 | `pl.concat([new_metadata, existing_metadata])` (`storage.py:431`) raises `ShapeError` on a width or column-order mismatch | Yes |

**Raiser 1 cannot happen in production, and this is the fact that sizes the whole change.** Production
sets `DATA_PATH_INTERNAL=s3://nged-forecast-internal/data` (`docs/live_service/aws.md:510`), so the
roster is an S3 object. A killed writer abandons its multipart upload, the old object stays intact,
and `ENOSPC` does not exist. Reproduced against a moto S3 server: overwriting a live roster key with a
large frame and killing the writer mid-flight (`returncode -9`) left the previous object readable and
unchanged. Locally the torn write is real — a `SIGKILL`ed `write_parquet` left a 256 KB file that no
longer parses, and a 0-byte file, the shape a full disk produces, gives the same
`ComputeError: parquet: File out of specification: The file must end with PAR1` the issue quotes.

So the state the issue reports — a roster file that will not parse — is a **local**-root fault, and
raisers 2 and 3 are *our own code* on every root. None of the three needs a storage-format change.

**Raiser 3 is live rather than hypothetical.** Four `TimeSeriesMetadata` fields are
`allow_missing=True` (`information`, `area_wkt`, `area_center_lat`, `area_center_lon` —
`power_schemas.py:247-283`), so a narrower frame validates cleanly and then `pl.concat` raises
`ShapeError: unable to append to a DataFrame of width 10 with a DataFrame of width 11`; with the same
columns reordered it raises `unable to vstack, column names don't match`. It is reachable because
`_extract_time_series_metadata` derives its columns from each JSON file's own keys
(`read_nged_json.py:44-47`), which is exactly why `download_and_parse_files` already unions them with
`how="diagonal"` (`storage.py:198`).

**The same asymmetry makes the diff wrong.** `hash_rows` is column-order sensitive, so a stored roster
whose columns are in a different order reports every row as changed and gets rewritten every run.

### An unusable roster also breaks `live_forecasts` — out of scope

`_load_engineering_inputs` reads the roster unguarded (`cv_assets.py:324`), so an unusable roster takes
the forecast off the degradation ladder entirely, not even rung 4. Filed separately as
[#528](https://github.com/openclimatefix/nged-substation-forecast/issues/528); the real fix there is a
design question about whether the live path should depend on the roster at all.

This is an [H1](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/engineering-hypotheses/#h1-a-service-that-mostly-runs-itself)
bug. Today's recovery is "someone deletes the file by hand", a **T1.1** intervention that is not an
upstream format change, and two of the five scripted **T1.4** game-day failures — "disk full" and
"daemon killed" — are the events that produce it.

## What changes, file by file

### `packages/nged_data/src/nged_data/storage.py`

**New `_read_existing_roster(metadata_path, storage_options) -> tuple[pt.DataFrame[TimeSeriesMetadata] | None, str | None]`.**
Wraps today's `object_exists` check, `pl.read_parquet` and `TimeSeriesMetadata.validate`
(`storage.py:388`, `:405-408`), returning the roster and `None`, or `None` and a one-line reason. An
absent file returns `(None, None)` — no file is not a fault. Anything else returns `(None, repr(exc))`.

Catching `Exception` and **not** `BaseException` is deliberate here, and differs from the `checks.py`
guards: a pyo3 panic from polars or obstore is not evidence about the *file*, and overwriting the
roster on that evidence is worse than skipping the update, so a panic falls through to the asset-level
guard, which degrades without rewriting anything.

**`upsert_metadata`'s new shape.** Validate and sort the snapshot as today (`storage.py:386`), then:

- `existing, reason = _read_existing_roster(...)`. When `existing is None`, take the branch that
  already exists at `storage.py:388-402` — create the parent directory if local, write the snapshot,
  return. When `reason` is not `None`, log at ERROR and add `metadata_roster_rebuilt_reason` to the
  stats. That is the whole rebuild: the create branch and the rebuild branch are the same code, which
  is rule 3 ("treat detectably-wrong input as missing") falling out of the structure rather than being
  bolted on.
- Otherwise compute the diff and merge as today, with **one concat serving both** (below).

**`pl.concat(..., how="diagonal")` at `storage.py:431`, and the diff taken from that same frame.** The
keyword alone disposes of raiser 3, and it is the keyword this module already uses 230 lines earlier.
Verified: vertical concat raises `ShapeError`, while diagonal plus `unique(subset="time_series_id",
keep="first")` succeeds, validates, and yields `[(1, 'note'), (2, None), (3, None), (4, None)]` for a
stored roster carrying `information` and a snapshot that omits the column — a field NGED stops sending
is **cleared**, which is today's semantics and the behaviour D1 chose.

The diff has to come from the concatenated frame rather than from the two inputs, or the column-order
bug survives:

```python
combined = pl.concat([new_metadata, existing_metadata], how="diagonal")
new_rows, stored_rows = combined.head(new_metadata.height), combined.tail(existing_metadata.height)
metadata_diff = new_rows.filter(~new_rows.hash_rows().is_in(stored_rows.hash_rows().implode()))
```

Both halves now carry the union schema in one order, so `hash_rows` compares like with like whatever
the stored file's column order is. The early return when `metadata_diff.is_empty()`
(`storage.py:417-422`) stays exactly as it is — that gate is what already keeps an unchanged roster
from being rewritten, and every test of it keeps passing.

**`UpsertMetadataStats`** gains two optional keys: `metadata_roster_rebuilt_reason: str` and
`metadata_upsert_failed: str` (the latter set by the asset). The three existing keys keep their
meanings. Note that on a rebuild, `metadata_n_new_TimeSeriesIDs` reports the whole snapshot, because
the create branch already does that (`storage.py:399-402`) and it is *true* there. What makes it
honest on a rebuild is `metadata_roster_rebuilt_reason` sitting beside it, telling the operator the
count means "rewritten from scratch" rather than "newly appeared".

**The docstring** stops promising it "assumes it is called by one thread at a time" as the whole story
and says what it now guarantees: a stored roster it cannot read or cannot validate is treated as
absent and rewritten, and the caller is told via the stats.

### `src/nged_substation_forecast/defs/assets.py`

- **`power_time_series_and_metadata`** — the `upsert_metadata` call (`assets.py:125`) goes under a
  `try`/`except BaseException` that re-raises `KeyboardInterrupt | SystemExit |
  DagsterExecutionInterruptedError`, the `checks.py:347-361` idiom and the same reasoning: a pyo3
  `PanicException` does not derive from `Exception`, and each compiled extension defines its own class.
  The handler logs the traceback, calls
  `report_asset_degradation("power_time_series_and_metadata", exc)`, and substitutes
  `UpsertMetadataStats(metadata_upsert_failed=repr(exc))` so the power write below still runs. Needs a
  `DagsterExecutionInterruptedError` import.
- **Where the stats carry `metadata_roster_rebuilt_reason`**, log at ERROR and call
  `report_asset_degradation` with the reason. The rebuild is a silent-by-default event otherwise, and
  it is the operator's cue that the roster may now be thin.
- **The consequence worth a comment**, because it is a genuine trade: the power Delta table is what
  `select_new_rows` uses to decide which JSON files are new (`storage.py:343`), so once the power rows
  land, a *failed* roster update is not retried. That run's metadata change waits until NGED
  republishes those series, about five hours. Losing one refresh of derived, re-delivered data is much
  cheaper than blocking the power stream.
- **A rebuild can leave the roster thin, and nothing in this change repairs it.** The snapshot covers
  only ids whose files were new this run, so a rebuild drops every quiet id — and because
  `select_new_rows` never re-reads a file already represented in the power table, those rows do not
  come back on their own. The runbook gets the manual re-derivation step. This is the cost of dropping
  the repair pass, and it is stated rather than glossed.

### `src/nged_substation_forecast/_sentry.py`

**New `report_asset_degradation(asset_name: str, detail: BaseException | str) -> None`** — tags
`degraded_asset` on a forked scope, sending `capture_exception` for an exception or
`capture_message(…, level="error")` for a string, never raising, exactly like
`report_check_degradation` (`_sentry.py:133`). The shared `new_scope`/`try` body moves into a
module-private helper so there are not two copies; `report_check_degradation` is otherwise untouched.

**This is not optional garnish, and it is the piece that survives every version of this plan.**
`init_sentry` configures `LoggingIntegration(event_level=None)` (`_sentry.py:93-101`), so an ERROR log
is *not* an event, and a step that no longer fails never fires `sentry_capture_failure`. Swallow the
exception without this and #508's fix converts a loud failure into total silence — the exact
fail-open-becomes-invisible trap rule 7 exists to prevent.

`init_sentry`'s docstring enumerates "the three explicit senders" (`_sentry.py:98`); it becomes four.

The union signature exists because the two call sites differ: the asset guard holds a live exception,
while the rebuild path holds only a reason string, the exception having been handled inside
`nged_data`, which must not depend on Sentry or Dagster.

### What this plan deliberately does not touch

Stated so a reviewer can see the scope line rather than infer it: no `settings.py` change, no
`CLAUDE.md` architecture edit, no new `delta_store` module, no `scan_delta` classmethods, no migration,
no dashboard edits, no test-helper churn, and `h3_grid_weights` is not touched at all — it is written
by a different asset and read by `ecmwf_ens`, and `upsert_metadata` never sees it.

## Design-philosophy check

This path is **production** — the hourly `power_time_series_and_metadata_job`, which carries
`sentry_capture_failure` (`defs/schedules.py:19`) — so it degrades rather than raises.

- **Rule 1** (never raise because an input is absent or stale): the ingest keeps running and records
  the degradation.
- **Rule 2** (liberal about missing, strict about malformed): an unusable stored roster is still
  rejected at the Patito boundary, never merged, never trusted. What changes is only that rejecting it
  no longer rejects the power data with it, which is the boundary the issue draws.
- **Rule 3** (treat detectably-wrong input as missing): the clean instantiation, and here it is
  structural — the rebuild branch *is* the create branch.
- **Rule 7** (a warning path may never fail the thing it warns about): respected. No asset check is
  added or edited; `power_data_is_fresh` keeps `WARN`/`blocking=False` and its catch-all. The new
  Sentry sender cannot raise.
- **Principle 10** ("every write is atomic and idempotent", `design-principles.md:387`) is the one
  place this plan trades something away, and it is worth naming. The roster stays a bare parquet file,
  so on a **local** root a killed writer can still tear it. What the change buys instead is that the
  torn file now *self-heals*: it is detected, reported and rewritten on the next run rather than
  wedging the job. On production's S3 root the atomicity property was never being violated. A Delta
  conversion would deliver atomicity properly, and is filed separately.
- **What the asset-level guard trades away, stated rather than glossed:** it wraps the whole
  `upsert_metadata` call, whose first statement validates the *snapshot* (`storage.py:386`), so a
  contract violation — which rule 1 says should raise, being our own bug — is degraded to a Sentry
  event. The strict boundary on incoming data is untouched: `_extract_time_series_metadata` validates
  every file and raises at `read_nged_json.py:58`, inside `download_and_parse_files`, which is
  **outside** this guard, so a genuine NGED contract break still fails the run there. What the guard
  absorbs is a bug in our own upsert code, and absorbing that is the price of the property the issue
  asks for.
- **Rules 6, 11**: untouched. **H1**: removes a hand-intervention class from **T1.1** and covers two
  **T1.4** game-day failures with automatic recovery.

## Tests

`packages/nged_data/tests/test_storage.py`:

1. `test_upsert_metadata_rebuilds_when_the_stored_roster_will_not_parse` — write junk bytes at
   `metadata_path`, then upsert. Assert the roster is rewritten, the stats carry
   `metadata_roster_rebuilt_reason`, and nothing raises. **This is the issue's literal reported state.**
   On `main`: `ComputeError` out of `upsert_metadata` (verified).
2. `test_upsert_metadata_rebuilds_when_the_stored_roster_fails_its_contract` — a stored roster missing
   `substation_type`. On `main`: `DataFrameValidationError` (verified).
3. `test_upsert_metadata_merges_a_snapshot_missing_the_optional_columns` — stored roster carries
   `information`, snapshot omits the column. Assert the merge succeeds, the merged frame validates, and
   the field is **cleared** for ids the snapshot carries while surviving for ids it does not — the
   semantics verified above. On `main`: `ShapeError` (verified).
4. `test_upsert_metadata_ignores_the_stored_column_order` — same data, columns reordered on disk.
   Assert a no-op: `metadata_n_updated_TimeSeriesIDs == 0` and the file is not rewritten. On `main`:
   `hash_rows` is order-sensitive, so it reports a spurious update or raises on the vstack (verified).
5. `test_upsert_metadata_does_not_rebuild_on_a_panic` — monkeypatch the read to raise a
   `BaseException` subclass; assert it propagates and the stored file is untouched. This pins the
   `Exception`-not-`BaseException` choice, which is otherwise invisible and easy to "tidy" later.

`tests/test_assets.py`:

6. `test_power_time_series_and_metadata_writes_power_when_the_roster_upsert_fails` — monkeypatch
   `assets.upsert_metadata` to raise `RuntimeError`; assert `result.success`, that the
   `power_time_series` Delta table holds the fixture rows, and that `metadata_upsert_failed` appears in
   the materialisation metadata. **The issue's headline property.** On `main`: the run fails.

`tests/test_sentry.py`:

7. Mirror the two existing `report_check_degradation` tests (`:176`, `:220`) for
   `report_asset_degradation` — the event carries `{"degraded_asset": …}`, no tag leaks into the
   current or isolation scope, and a raising `capture_*` is swallowed and logged. Add the
   message-shaped call as a third case.

**Two existing tests must keep passing untouched**, and they are the regression guard on the diff:
`test_upsert_metadata_returns_diff` (`test_storage.py:145-232`) and
`test_metadata_parquet_round_trip_over_s3` (`test_s3_data_paths.py:237`). Confirmed: the first passes
against a prototype of this design with its assertions unchanged.

## Docs to update

Written in the present tense, describing how the code works now — and note `main` added prose rules
since this plan was drafted (see below), which these edits must satisfy.

- **`docs/live_service/operations.md:182`** — the "`metadata.parquet` left half-written" phrase, plus a
  new operator paragraph: what `metadata_roster_rebuilt_reason` and `metadata_upsert_failed` mean, the
  `degraded_asset:power_time_series_and_metadata` Sentry filter, **the manual re-derivation step for a
  thin roster after a rebuild**, and the instruction to append to the intervention log as the rest of
  that section does.
- **`docs/architecture/production-deployment.md:91`** — the half-written-roster example, which now
  degrades rather than wedging the run, plus the Sentry section at `:179`, whose "one production fault
  the hook cannot see is a standalone `@asset_check` that caught its own …" is falsified by an *asset*
  now catching its own exception and by the new `degraded_asset` tag.
- **`src/nged_substation_forecast/defs/checks.py:340-341`** and the docstring of
  `test_power_data_is_fresh_degrades_on_a_corrupt_metadata_parquet` (`tests/test_checks.py:395`), which
  both state that `upsert_metadata` "reads the same file first and fails the asset outright" — no
  longer true, and the second is the more misleading of the two because it reads as a specification.
- **`docs/design-philosophy/inherent-stability.md`** — a sentence in "Missing versus wrong" (`:243`)
  giving the roster as a second worked example of rule 3, and a new **rule 12**: *a derived artifact we
  cannot read is absent, not fatal — rebuild it and keep going.* Appended after rule 11 (`:199`), so
  nothing is renumbered. It restates no design principle, so it needs no pairing marker
  (`inherent-stability.md:153-159`).
- **`packages/nged_data/README.md:10`** — `upsert_metadata`'s one-liner gains the rebuild.

No roadmap page completes here, so there is no "Implementation details" section to delete and no status
banner to move. `#508` is referenced from no `docs/` page.

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
uv run pytest packages/nged_data/tests/test_storage.py tests/test_assets.py tests/test_sentry.py tests/test_checks.py -q
```

The S3 round-trip test is `integration`-marked, so run it explicitly:

```bash
uv run pytest tests/test_s3_data_paths.py -q -m integration
```

No `--run-network` run is needed: nothing touches the Dynamical catalog or NWP conversion conventions.
No marimo notebook changes, so `tests/test_marimo_notebooks.py` is unaffected.

## House rules `main` added after this plan was first written

`main` moved thirty commits while this plan sat on the branch. Two of the rules it landed bind the
implementation:

- **A docs link in code is spelled as its rendered URL, never a repo path**
  (`docs/architecture/code-style.md`). Every docstring and `#` comment this change adds must follow it.
- **"One home per argument"** — rationale worth a paragraph lives on one docs page and the docstring
  links to it. Relevant to `_read_existing_roster`'s `Exception`-not-`BaseException` choice: one
  sentence in the docstring, and if it needs more, it goes on the inherent-stability page beside
  rule 12.
- **The prose rules in `CLAUDE.md`** (concrete and skim-readable; concise by cutting whole sentences
  rather than clipping words; full sentences with subjects; present tense only) apply to all five
  documentation edits and to the PR body, which must not be hard-wrapped.

## Decisions, and remaining risks

**D1 — A field the snapshot no longer carries is cleared, not retained.** `how="diagonal"` plus
`unique(keep="first")` delivers this for free, so it needs no mechanism of its own — but it does need
test 3, because it is a semantic that would be easy to reverse by accident.

**D2 — The roster repair pass is dropped.** Its premise holds: a thin roster can cost a whole forecast
slot, because `_load_engineering_inputs` derives the NWP cell filter from the roster (`cv_assets.py:330`,
feeding `h3_index.is_in(cells)` at `:338-344`) while the population comes from the promoted model's
`trained_ids` (`production_assets.py:238`), and `live_forecasts` raises on an empty result
(`production_assets.py:285-290`). But that is #528's hazard, the repair only narrows the window, and it
costs a second `upsert_metadata` call site, stats merging, and three tests for a path that should fire
once in the project's life. The manual re-derivation step goes in the runbook, which the failed-repair
case would have required anyway. **Revisit after #528 decides whether the live path should depend on
the roster at all.**

**D3 — The roster stays bare parquet; the Delta conversion is filed separately.** It is a real want,
for reasons this bug does not supply: version history on the roster helps reproduce past CV runs, and
principle 10 would then hold project-wide with no exception. Judged on those merits, in its own issue.

**D4 — `report_asset_degradation` keeps the `BaseException | str` union.** One function, one tag, one
docstring covering both shapes, versus two near-identical six-line functions.

**Remaining risk, named because it is the one thing this plan makes worse than a Delta conversion
would:** on a local root, a torn roster write is still possible. It now self-heals on the next run
instead of wedging the job, which is a strict improvement on today, but it is not the same as being
impossible.

## History of this plan

Three drafts and three adversarial reviews. Recorded because the discarded direction is the one most
likely to be proposed again.

### Draft 1 — harden the parquet write

Implemented the issue's own options: a `write_parquet_atomically` helper (local temp file plus
`os.replace`), a `copy_object` helper for a quarantine copy, and an `_align_to_contract` step to stop
the `concat` raising. Its review found that the draft wrongly claimed `pl.concat` "cannot raise" after
a successful validate; that the freshness-check consequence was backwards, inherited from the issue
body (`stale` is computed from coverage and deliberately *not* restricted to the roster,
`checks.py:196-204`, so a thin roster instead stops never-seen ids being reported at `:205-208` and
shrinks `n_series_total` at `:221-226`); that a thin roster can fail `live_forecasts` outright; that a
fixed `.tmp` suffix is unsafe for two writers, nothing serialising this asset; and that `os.replace`
would replace a symlinked path with a plain file.

### Draft 2 — convert the roster and the weights to Delta, around `MERGE`

Jack asked whether converting every bare-parquet file to Delta was not the better fix, removing the
bespoke atomic-write machinery. Its review confirmed sixteen defects, of which the blocking one was
that `TimeSeriesMetadata` has four `pl.Enum` columns and `write_delta` panics on them
(`cannot downcast Utf8View dictionary value to byte array`) — while `MERGE` tolerates an Enum source,
so a merge-only test suite would pass while the create path aborted in production. It also found that
`MERGE` destroys row order (`[3, 7, 1, 2, 4, 5, 6, 8, 9, 10]` measured); that a predicate naming a
column the target lacks raises, so a routine contract addition would trigger a rebuild; and that the
documented migration one-liner fails, parquet preserving the Enum dtype that Delta cannot store.

### Draft 3 — this one

The third review was asked to test Jack's own reaction that the plan had become far more complicated
than the issue warranted. It found the two facts that collapse the design, both of which I verified:

1. **Raiser 1 is unreachable in production**, because the data root is S3. Draft 2 contained the
   sentence "specific to a local path" and never drew the conclusion. Since raiser 1 is the only raiser
   Delta removes by itself, the entire storage conversion was motivated by a fault the production
   deployment cannot experience.
2. **`how="diagonal"` disposes of raiser 3 in one keyword**, and delivers D1's semantics for free.
   `_align_to_contract`, the merge predicate and two of draft 2's five documented traps existed only to
   serve `MERGE`.

It also showed `MERGE` buys nothing here: the no-op-commits-nothing property comes from the
`metadata_diff.is_empty()` gate that **already exists** on `main` (`storage.py:417-422`), not from
`MERGE`; and draft 2 kept the full read anyway, twice — once as a validating gate, once for
`metadata_updated_TimeSeriesIDs`, which merge metrics cannot supply — so it was read-modify-write
*plus* a merge, against an asset nothing runs concurrently. Worst of all, Delta **regresses** recovery
for the state #508 actually reports: junk bytes at the path self-heal under parquet
(`rebuilt, reason=ComputeError(...)`) and cannot under Delta (`DeltaError: Path does not exist`).

Findings from that review folded in above: the `_delta_log`-corruption analysis, the `metadata.py`
"one function" contradiction and the fact-5 gate-ordering hole all became moot with `MERGE`. Two
`settings.py` docstrings it found missing from draft 2's inventory (`:291`, `:311`) are moot too, since
no path changes. Its point that `metadata_n_new_TimeSeriesIDs` reports the whole snapshot on a rebuild
is kept, resolved by pairing the count with the rebuild reason rather than by renaming keys, since the
create branch already behaves that way and is asserted by two existing tests.

The pieces of draft 2 that survive on their own merits are the asset-level guard, the two new stats
keys, `report_asset_degradation`, rule 12, and the `Exception`-not-`BaseException` reasoning.
