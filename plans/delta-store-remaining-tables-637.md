# Move the four remaining Delta writes into `delta_store` (#637)

## Status and how to resume

**No code has been written.** This plan (below) has been through both adversarial plan reviews
required by its "complex" sizing, plus a further adversarial review after `main` was merged into
this branch to catch staleness from concurrent work (see "Adversarial review outcomes" at the
end); all three rounds are resolved. It is now waiting on **human review of the plan itself** —
that review has not happened yet.

- **Branch:** `delta-store-remaining-tables-637`, pushed to `origin`.
- **Worktree:** `.claude/worktrees/delta-store-remaining-tables-637` off the main repo checkout at
  `/home/jack/dev/nged-substation-forecast`. If that worktree no longer exists when this resumes,
  recreate it with `git worktree add .claude/worktrees/delta-store-remaining-tables-637
  delta-store-remaining-tables-637` from the main repo checkout (the branch already exists on
  `origin`, so no `-b` and no fresh plan needed).
- **This plan file** lives at `plans/delta-store-remaining-tables-637.md` inside that worktree, and
  is also attached to the PR for #637 that carries this status note (see the PR body for a link
  back to this branch, or `gh pr list --search 637 --state open`).
- **Next step once a human approves this plan:** resume under the `implement-issue` skill at its
  step 2 (worktree/plan already exist — skip step 1), in the worktree above: implement the "What
  changes, file by file" section below, run the verification commands, open a PR carrying labels
  and `JackKelly` as assignee, then run the two diff reviews (`implement-issue`'s own correctness
  and mutation-testing passes) that "complex" sizing calls for.
- **Nothing here is time-sensitive** — this is a pure code-organisation move with no external
  dependency, so it is safe to leave paused indefinitely.
- **Epic context:** #637 is wave 1 of the v0.2.1 epic (#642). Two other issues were running
  concurrently in the same wave and are fenced off in "Territory check" below: #638
  (`trained_cv_model`/`cv_power_forecasts` in `cv_assets.py`), which has since **merged to
  `main`** and added no `write_deltalake` call in `cv_assets.py` (confirmed by grep during the
  post-merge review below), and #639 (the `_BaseSummary`/`_FileListingSummary`/
  `_PowerTimeSeriesSummary` classes in `assets.py`), still open as of the post-merge review and
  still non-overlapping. #656 and #657 are held back until this PR merges, since they build on the
  functions this plan moves — check the epic for their current state before resuming, in case
  either has since landed and shifted a line number or a function signature this plan cites.

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
byte-for-byte what it produces after — with one small accepted exception in
`write_power_time_series`'s URI handling, noted where that module is described below. No new
`WriterProperties`, sort order or significand rounding is added for these four tables in this
issue — see "Departures from the issue" below.

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
    *,
    storage_options: ObjectStoreOptions | None = None,
) -> None: ...
```

Replaces `assets.py:202`'s `new_power_ts_deduped.write_delta(delta_path, mode="append",
storage_options=typeddict_to_dict(storage_options), delta_write_options={"partition_by":
"time_series_id"})`. Ports the polars `.write_delta()` call to `deltalake.write_deltalake` (arrow
input, `mode="append"`, `partition_by=["time_series_id"]`) to match the `nwp.py`/
`power_forecasts.py` pattern the issue asks for — functionally identical to the polars method for
the code path this table exercises, confirmed by reading the installed `polars` (1.44.2) source:
`pl.DataFrame.write_delta`'s non-merge branch is exactly `write_deltalake(table_or_uri=target,
data=self, mode=mode, storage_options=storage_options, **delta_write_options)`, so today's
`delta_write_options={"partition_by": "time_series_id"}` is already just
`write_deltalake(partition_by="time_series_id")` (a bare string, not a list — `write_deltalake`
accepts both; the new module passes `["time_series_id"]` to match the list form the other three
modules use, which is cosmetic). Two things `write_delta` does before that call are not carried
over, both accepted rather than replicated: `_check_for_unsupported_types(self.dtypes)` is a no-op
for `PowerTimeSeries` (`Int32`/`Datetime`/`Float32`, no `Enum` column, so `.to_arrow()` needs no
help); and `_resolve_delta_lake_uri(str(target), strict=False)` expands a scheme-less path via
`Path(target).expanduser().resolve()` before handing it to delta-rs — `write_nwp` and
`write_power_forecasts` already skip this step, so `write_power_time_series` moving to the same
un-resolved behaviour is consistent with the rest of the package, but note it as a real (if small)
change: a `~`-prefixed `Settings.power_time_series_data_path` would today be expanded by polars,
and after this change would be passed to delta-rs literally, creating a directory named `~`. No
current `Settings` value uses `~`, so this is not expected to bite, but it is not the same as "no
behaviour change" the earlier phrasing claimed.

### New: `packages/delta_store/src/delta_store/eligible_time_series.py`

```python
def write_eligible_time_series(
    eligible: pt.DataFrame[EligibleTimeSeries],
    table_uri: str | Path,
    fold_id: str,
    *,
    storage_options: ObjectStoreOptions | None = None,
) -> None: ...
```

Replaces `cv_assets.py:153`. `fold_id` is an explicit parameter rather than read off the frame
(`eligible.item(0, "fold_id")`) for the same reason `write_power_forecasts` takes
`replace_partition` explicitly: a fold with zero eligible series must still clear its Delta
partition, and an empty frame has no row to read a fold id from. `storage_options` is keyword-only
(a bare `*` after `fold_id`, matching `write_power_forecasts`'s `*` after `table_uri`); the call
site passes every argument by keyword, per this repo's keyword-argument rule
(`docs/architecture/code-style.md`, "Calling functions") — `fold_id` and `table_uri` are both bare
strings, exactly the case that rule calls out as needing the keyword to stay readable and to keep
a future accidental argument-swap from compiling silently.

### New: `packages/delta_store/src/delta_store/effective_capacity.py`

```python
def write_effective_capacity(
    capacity: pt.DataFrame[EffectiveCapacity],
    table_uri: str | Path,
    *,
    storage_options: ObjectStoreOptions | None = None,
) -> None: ...
```

Replaces `cv_assets.py:204`. Whole-table overwrite, no partition/predicate — matches the current
call exactly (this table has no `partition_by` today; not adding one here is in scope for the
same "pure move" reason as above).

### New: `packages/delta_store/src/delta_store/forecast_metrics.py`

```python
def write_forecast_metrics(
    metrics: pt.DataFrame[Metrics],
    table_uri: str | Path,
    experiment_name: str,
    fold_id: str,
    *,
    storage_options: ObjectStoreOptions | None = None,
) -> None: ...
```

Moves the whole body of `cv_assets.py`'s `_write_metrics_to_delta` (line 713-746) verbatim,
including its docstring's Enum→String rationale and the `pl.Enum` cast loop. `cv_assets.py` loses
this private helper; its single call site (line 915) becomes
`_write_metrics_to_delta(metrics_path, enriched, exp_name, fold_id, storage_options)` →
`write_forecast_metrics(enriched, table_uri=metrics_path, experiment_name=exp_name,
fold_id=fold_id, storage_options=storage_options)` — `metrics`/`table_uri` stay positional
(different types, `pt.DataFrame[Metrics]` vs. `str | Path`, so a swap fails loudly), but
`experiment_name` and `fold_id` are both bare `str` and must be passed by keyword: a positional
swap between them would type-check, compile, and silently write the wrong overwrite predicate,
which is exactly the failure mode the keyword-argument rule exists to close off and which the
original plan's positional call site did not. `write_forecast_metrics` keeps `experiment_name` and
`fold_id` as two flat `str` parameters rather than `write_power_forecasts`'s single
`replace_partition: tuple[str, str]` — the two overwrites differ in whether the caller can *omit*
the partition-scoping replace (a multi-chunk `power_forecasts` write only replaces on its first
chunk); `forecast_metrics` has no such chunking and always scopes the overwrite, so there is
nothing optional to express as `| None`, and the flat parameters are the simpler match for that
call shape.

### `packages/delta_store/src/delta_store/__init__.py`

Add the four new `write_*` imports and `__all__` entries; update the module docstring's "One
module per table (currently `power_forecasts` and `nwp`)" line to name all six, and reword "the
two tables land on different choices" — that sentence is only true of `power_forecasts`/`nwp`
(the pair with writer-properties tuning), not of the other four, which carry no tuning at all. Say
explicitly that only two of the six modules carry measured writer-properties tuning, and name
which two.

### `packages/delta_store/README.md`

Add a "Contents" bullet for each of the four new modules — `nwp` already has one (added on `main`
since this plan was first written), so it needs no new bullet. Also reword the intro's "Every
lever below is measured against real data… for the comparison across both tables" — "both" reads
oddly once four lever-free modules join `power_forecasts` and `nwp` in Contents; make it explicit
that the measured-levers claim covers only those two tables, not all six.

### `docs/api/delta_store/index.md`

Add a `::: delta_store.<module>` mkdocstrings directive for each of the four new modules, in the
same style as the existing `::: delta_store.power_forecasts` / `::: delta_store.nwp` lines —
without this, the new modules' docstrings render nowhere in the published API docs even though the
package's own `__init__.py` claims to own all six tables.

### `src/nged_substation_forecast/defs/assets.py`

`power_time_series_and_metadata` (lines 198-207): replace the `.write_delta(...)` call with
`write_power_time_series(new_power_ts_deduped, delta_path, storage_options=storage_options)`,
keeping the surrounding `if not new_power_ts_deduped.is_empty(): if_local_path_then_make_parent_dir(...)`
guard exactly as-is (unchanged control flow, so the asset's retry/degradation boundaries — the
S3-read retry guard is the `try/except` at lines 124-164, which ends well before this write — are
untouched). Add `from delta_store.power_time_series import write_power_time_series` to the import
block, alongside the existing `from delta_store.nwp import write_nwp`.

### `src/nged_substation_forecast/defs/cv_assets.py`

- `eligible_time_series` (lines 113-160): replace the `write_deltalake(...)` call with
  `write_eligible_time_series(eligible_df, settings.eligible_time_series_data_path, fold_id=fold_id,
  storage_options=storage_options)`.
- `effective_capacity` (lines 176-209): replace the `write_deltalake(...)` call with
  `write_effective_capacity(capacity_df, settings.effective_capacity_data_path,
  storage_options=storage_options)`.
- `_write_metrics_to_delta` (lines 713-746): delete the function; its one call site (line 915)
  calls `write_forecast_metrics` from `delta_store` instead (see the exact keyword-argument call
  above).
- Import block (line 39): add the three new `delta_store` imports; drop `from deltalake import
  write_deltalake` — confirmed by `grep -n write_deltalake cv_assets.py` against the post-`main`-merge
  file: the only three hits are the three call sites this plan removes, and #638 (merged to `main`
  since this plan was first written, touching `trained_cv_model`/`cv_power_forecasts` in the same
  file) added no `write_deltalake` call, so the import is safe to remove outright rather than
  flagged for a re-check at implementation time.

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

- `packages/delta_store/README.md` — "Contents" section (four new bullets; `nwp` already has one)
  and the intro's "both tables" wording, as described above.
- `packages/delta_store/src/delta_store/__init__.py` module docstring — table count and the
  "different choices" sentence, as described above.
- `docs/api/delta_store/index.md` — add a `::: delta_store.<module>` mkdocstrings directive per
  new module, as described above. This is new since the plan's first draft: the page enumerates
  modules explicitly, so without this edit the four new modules' docstrings would render nowhere
  in the published API docs.
- No `docs/roadmap/` page references this issue directly (checked: the only `delta_store` hit
  outside `docs/architecture/` is `docs/roadmap/xgboost-improvements.md:110`, which cites
  `delta_store.nwp.write_nwp` specifically and is unaffected); no roadmap status banner to update.
- `docs/architecture/overview.md:20` and `docs/architecture/performance.md` describe `delta_store`
  owning "each table's physical layout" and detail the `nwp`/`power_forecasts` formats
  specifically — neither makes a "only two tables" claim that this plan would falsify, so no edit
  needed there (checked by grep across `docs/architecture/`).

## Verification commands

Standard green-before-push set: `uv run ruff check .`, `uv run ruff format .`, `uv run
--all-packages ty check`, `uv run pytest`. Add `uv run pymarkdown scan -r docs README.md
CLAUDE.md packages/*/README.md` since `packages/delta_store/README.md` is touched. Add `uv run
mkdocs build --strict` since `docs/api/delta_store/index.md` is touched (a broken mkdocstrings
directive fails that build, not the markdown linter). No network tests apply (no NWP conversion or
S3 code touched).

## Risks and open questions

1. **Should `power_time_series`, `eligible_time_series`, `effective_capacity` and
   `forecast_metrics` get their own writer-properties/rounding tuning, now or as a follow-up?**
   This plan's recommendation: follow-up issue, after implementation, once there's real production
   volume to measure against — `power_forecasts` and `nwp`'s tuning was each backed by a measured
   before/after on real data (6.33 GB → 0.73 GB; a specific row-group pruning benchmark), and
   guessing at settings without that measurement risks repeating the `nwp` module's own cautionary
   tale (`BYTE_STREAM_SPLIT` helped one table and hurt the other). Flag for human confirmation.
2. **The `write_deltalake` import removal in `cv_assets.py`**: resolved. #638 merged to `main`
   before this plan's post-merge review and touches only `trained_cv_model`/`cv_power_forecasts`
   in the same file, adding no `write_deltalake` call; `grep -n write_deltalake cv_assets.py`
   against the current branch confirms exactly the three call sites this plan removes and nothing
   else. Still worth a final `grep` at implementation time as a cheap check, but this is no longer
   an open question.
3. **`write_power_time_series`'s move from polars' `.write_delta()` to `deltalake.write_deltalake`
   is a mechanical API swap for the write itself, with one small accepted behaviour change in URI
   handling** — verified by reading the installed `polars` (1.44.2) package's `DataFrame.write_delta`
   source directly (see the module section above): its non-merge branch calls `write_deltalake`
   with the exact same `mode`, `storage_options` and `delta_write_options` this plan ports across.
   `write_delta` additionally resolves a scheme-less path via `Path(...).expanduser().resolve()`
   before that call; `write_power_time_series` does not, matching `write_nwp`/`write_power_forecasts`
   already not doing so. No current `Settings` value depends on `~`-expansion, so this is accepted
   as in scope for the "match the sibling modules' pattern" goal, not left as an open question.
   Still worth the regression test (`test_append_partitions_by_time_series_id` above) as a durable
   guard, not as a substitute for this verification.

## Adversarial review outcomes

Both plan reviews ran (this issue is sized complex). Neither found anything to change; nothing was
rejected because nothing was proposed.

- **Simplicity review**: verified every empirical claim above independently (the `fold_id`
  emptiness scenario, the `.write_delta()` → `write_deltalake()` equivalence, the exact call-site
  line numbers) and considered collapsing the four modules into fewer files or a shared generic
  writer; concluded the four-module, no-new-tuning design is already the simplest one that
  satisfies the issue, not merely the cautious one.
- **Correctness/testability review**: independently re-derived every file/line claim, ran the
  empty-frame partition-clear case live against `deltalake.write_deltalake` to confirm the
  proposed test would actually catch a regression, checked `inherent-stability.md`'s raise-vs-
  degrade rule against the production write path, and confirmed the `write_forecast_metrics`
  argument-order swap is unambiguous and fails loudly (`AttributeError`) if implemented backwards.
  No factual, logical, or testability defects found.
- **Post-`main`-merge review** (run after `main` was merged into this branch, before human review
  of the plan): re-verified every factual claim against current code, since #638 and #639 were
  both in flight in the same files when the plan was first written and #638 has since merged.
  Found and fixed: two staleness issues (`docs/api/delta_store/index.md` enumerates `delta_store`
  modules explicitly and was missing from "Docs to update"; several cited line numbers had shifted
  and are now corrected), one code-style gap (the four new `write_*` signatures and their call
  sites did not follow this repo's keyword-argument rule — fixed by making `storage_options` and
  friends keyword-only, matching `write_power_forecasts`'s existing `*` placement), one overstated
  claim (`write_power_time_series`'s move off `.write_delta()` drops that method's URI
  `expanduser().resolve()` step — a real, if inconsequential, behaviour change, now stated as
  such rather than claimed as byte-identical), and one now-stale plan claim (the `nwp` README
  bullet the plan proposed to add already exists on `main`). No defect found in the plan's core
  design, the "pure move" claim for `mode`/`predicate`/`partition_by`/`storage_options`, or the
  `fold_id`-as-explicit-parameter rationale.
