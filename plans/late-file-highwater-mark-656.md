# Fix `select_new_rows`' high-water mark to allow late/corrected files (#656)

**Problem.** `select_new_rows` (`packages/nged_data/src/nged_data/storage.py:368-435`) decides
what is new by comparing a candidate row's time against the per-`time_series_id` maximum already
on disk. A row (or a whole file) whose time falls at or before that maximum is dropped forever,
even when it fills a genuine gap earlier in that series' history. NGED's telemetry can arrive out
of order — a file landing late, or a meter catching up after a stall — so a gap created that way
can never be backfilled today.

**Solution.** Replace the row-level check with a true "is this `(time_series_id, time)` pair
already on disk" anti-join, so a genuinely-missing reading is ingested regardless of arrival
order. Loosen the file-listing-level pre-filter with a bounded lookback margin so a late file is
still downloaded, while keeping that filter's job — bounding how much of the bucket gets
re-downloaded every hour — intact. Treat a same-timestamp *correction* (revised `power` value for
a time already stored) as out of scope: it needs an upsert into an append-only Delta table, which
is a materially bigger change than this issue's title suggests, and there is no evidence yet that
NGED does this. Recommend a follow-up issue for it.

## Verdict, size and departures from the issue body

**Worth implementing, roughly as described**, with one scope departure: the issue frames "late" and
"corrected" files as one bug with one fix; this plan splits them. The late-file case has a
concrete, checkable mechanism (a gap in `(time_series_id, time)` coverage) and a fix with no new
persisted state. The corrected-republication case is a different problem — Polars/`deltalake`
never overwrites an existing row, and `write_power_time_series`'s docstring says "Appends only" —
and would need a MERGE-based upsert, a decision on which of two conflicting values wins, and a
test for a state Delta doesn't naturally reach today. Bundling both into one Complex change is
exactly the "is the plan generalising for a problem not yet confirmed to exist" risk the
`plan-issue` simplicity review looks for. This plan fixes the late-file case now and lists the
correction case as a "Risks and open questions" item with a recommendation to split it into its
own issue.

**Size: Complex**, per the issue's own sizing — this changes what `select_new_rows` treats as new
into the `power_time_series` Delta table, and it is on the production ingest path. Full plan
review (both passes) and both diff reviews in `implement-issue`.

## What changes, file by file

### `packages/nged_data/src/nged_data/storage.py`

**`select_new_rows`, the `PowerTimeSeries` branch (currently storage.py:406-409, 425-433).**
Replace the join against `TimeSeriesCoverage.last_time` with a genuine existence check: scan the
Delta table's `time_series_id`/`time` columns, restricted to the `time_series_id`s present in the
candidate frame, and anti-join on `["time_series_id", "time"]` instead of filtering on
`time > last_time`. Add a small helper, `_existing_power_time_series_keys(delta_path,
storage_options, time_series_ids) -> pl.LazyFrame`, next to `time_series_coverage`, returning that
restricted `(time_series_id, time)` frame. Unlike `time_series_coverage`, this helper needs no
empty-table branch: its only caller, `select_new_rows`, already returns early via
`delta_table_exists` (storage.py:398-400) before either branch runs, so by the time the helper is
called the table is guaranteed to exist. Filtering by `time_series_ids` before the join keeps the
build side of the anti-join proportional to the handful of series that reported this hour, not the
whole table — see the cost note below. This restriction matters more than it looks: the table is
partitioned by `time_series_id` (`delta_store/power_time_series.py`'s `partition_by`), so an
unrestricted anti-join would force a full 43.8M-row materialisation every hour, while the
restricted version is a partition-pruned scan proportional only to the reporting series' own
history.

Both mechanisms this plan fixes are real, not theoretical: `list_timeseries_json_files`'s own
docstring (storage.py:64-82) says a file's S3 key encodes the time window its *data* covers, not
when it was uploaded, so a file uploaded today can freely carry an old window — exactly the late
arrival and the mid-history gap this plan targets.

`select_new_rows`'s `_ProcessedFileListing` branch keeps using `TimeSeriesCoverage.last_time` (see
next item), so `time_series_coverage` stays as it is and both call sites keep working.

**Move the `time_series_coverage(delta_path, storage_options)` call (currently storage.py:403,
unconditional before the branch split) inside the `elif "end_time" in time_series.columns` branch,
so it runs only for `_ProcessedFileListing`.** Call the new `_existing_power_time_series_keys`
helper only inside the `if "time" in time_series.columns` branch. Leaving the existing
`time_series_coverage()` call where it is and simply adding the new helper call alongside it would
make every `PowerTimeSeries`-branch call pay for both the full-table `time_series_coverage` scan
and the new restricted anti-join scan — silently doubling cost every hour and defeating the
partition-pruning this plan's cost argument rests on. No test can catch this by output alone
(the result is correct, just slower), so this restructuring is a required step, not an
implementation detail to improvise.

**`select_new_rows`, the `_ProcessedFileListing` branch (currently storage.py:410-413,
425-433).** We don't have per-row times for an undownloaded file — only its `start_time`/`end_time`
window from the S3 key — so an anti-join isn't available at this stage; keep the `last_time`
comparison but loosen it with a fixed lookback margin, so a file isn't excluded just for landing a
short while after the current watermark:

```python
_LATE_FILE_LOOKBACK: Final[timedelta] = timedelta(days=3)
```

Filter becomes `pl.col("last_time").is_null() | (pl.col("end_time") > pl.col("last_time") -
_LATE_FILE_LOOKBACK)`. `download_and_parse_files` then downloads a few extra recently-superseded
files per hour, and the row-level anti-join above drops whatever they don't actually add — the
existing `unique(subset=["time_series_id", "time"], keep="last")` dedupe in
`download_and_parse_files` (storage.py:262) doesn't need to change. `3` days is a judgement call
against "NGED's JSON files land at irregular intervals ... several-hours-apart" (the asset
docstring) — generous enough for an ordinary late arrival, small enough that it doesn't reopen the
whole bucket every hour. Flagged as an open question below for the human reviewer to confirm or
adjust; it does not need to be exactly right, because the row-level fix is what makes correctness
no longer depend on this margin.

**Docstrings.** `select_new_rows`'s docstring currently says the comparison is "made on a
time_series_id by time_series_id basis" against "the most recent time... already stored" for both
input kinds — rewrite the `PowerTimeSeries` half to describe the anti-join instead, and note the
`_ProcessedFileListing` half's lookback margin and what it does and doesn't guarantee (a file more
than `_LATE_FILE_LOOKBACK` late is still silently dropped — this is now a bounded, named gap
rather than an unbounded one). `time_series_coverage`'s docstring's paragraph on
`select_new_rows`'s two call sites should note that only the file-listing call still uses
`last_time`; the row-level call now scans `time_series_id`/`time` directly.

### `packages/nged_data/src/nged_data/storage.py` cost note (for the docstring, not a design
change)

The new anti-join scan is filtered to the reporting hour's `time_series_id`s (typically a handful,
even at V2's ~2,500-series scale, since NGED's files land a few at a time), so its cost is
proportional to those series' own history, not `time_series_coverage`'s whole-table scan. Document
this in `_existing_power_time_series_keys`'s docstring the same way `time_series_coverage`
documents its own cost, so the next person to touch this understands why it doesn't need the same
streaming-engine treatment applied to a full-table scan.

### No changes to `write_power_time_series`, `delta_store`, or any Patito contract

Confirms the scope departure above: nothing about *how* rows are written changes, only which rows
`select_new_rows` decides are new. `power_time_series` stays append-only.

## Design-philosophy check

This is on the production ingest path (`power_time_series_and_metadata`, tagged
`PRODUCTION_LAYER_TAGS`), so `docs/design-philosophy/inherent-stability.md` applies: nothing here
introduces a new way to raise on absent or malformed input. The anti-join and the lookback filter
are both ordinary Polars operations over data already validated by `PowerTimeSeries`/
`_ProcessedFileListing`; a missing Delta table is still handled by `select_new_rows`'s existing
early return (storage.py:398-400), unchanged by this plan. No asset check is added or changed. This
change is squarely **H-series adjacent** (fewer silently-dropped genuine readings improves the
completeness of what `eligible_time_series` and the CV pipeline ultimately see) but does not
itself correspond to a numbered hypothesis in `engineering-hypotheses.md`.

No principle in `docs/design-philosophy/design-principles.md` is traded away: no new column,
table, or config field is added; the one new constant (`_LATE_FILE_LOOKBACK`) is a private,
module-level tuning knob, not a public contract change.

## Tests

All in `packages/nged_data/tests/test_storage.py`, alongside the existing `select_new_rows` tests.

- **`test_select_new_rows_power_time_series_fills_a_gap_before_the_watermark`**: write a Delta
  table with `time_series_id=1` at `12:00` and `13:00` (a gap at `12:30`), then call
  `select_new_rows` with rows at `12:00` (already on disk), `12:30` (the gap), and `13:30` (past
  the watermark). Assert the result keeps exactly `12:30` and `13:30`, and drops `12:00`. **This
  fails on `main` today**: the current `last_time`-based filter treats `12:30` as `<= last_time`
  (`13:00`) and drops it, alongside the genuinely-old `12:00` row — the existing
  `test_select_new_rows_power_time_series` only ever exercises rows after the watermark, so it
  can't tell the two apart.
- **`test_select_new_rows_file_listing_keeps_a_file_within_the_lookback_margin`**: write a Delta
  table with `time_series_id=1`'s `last_time` at, say, `2026-01-04T00:00`. Pass a file listing row
  for `time_series_id=1` whose `end_time` is one day *before* that `last_time` (within the 3-day
  margin) and one whose `end_time` is five days before it (outside the margin). Assert the first is
  kept and the second is dropped. **Fails on `main` today**: the current filter drops both, since
  both are `<= last_time`.
- Keep `test_select_new_rows_power_time_series` unchanged as regression coverage for the ordinary
  in-order case — its rows all postdate the watermark either way, so the anti-join and the old
  `last_time` filter agree on it.
- **`test_select_new_rows_file_listing` needs updating, not left as-is.** Its `old.json` fixture
  row has `end_time` exactly equal to `time_series_id=1`'s on-disk `last_time`
  (`2026-01-01T12:00`), asserted excluded under today's strict `end_time > last_time`. Under the
  loosened `end_time > last_time - _LATE_FILE_LOOKBACK` filter, `old.json` now falls inside the
  3-day margin and is correctly *included* — a value sitting exactly at the watermark is genuinely
  within the lookback window under the new semantics. Update the test's expected `result.height`
  to `3` and its expected path set to add `"old.json"`, so it keeps exercising the ordinary
  in-order case without asserting behaviour the plan is deliberately changing.
- Extend `test_select_new_rows_power_time_series` (or add a sibling) to cover **more than one
  reporting `time_series_id` in the same call**, confirming the `time_series_ids`-restricted
  anti-join scan doesn't accidentally scope to only the first series' keys or leak another
  series' rows into the result.

## Docs to update

- `select_new_rows` and `time_series_coverage` docstrings, as described above — the "Write about
  the present, not the past" rule means these should describe the anti-join and the lookback
  margin as how the code works now, not narrate that a high-water mark used to be there.
- **`packages/nged_data/README.md`'s public-surface entry for `select_new_rows`** (currently:
  "filters `time_series` down to rows newer than what the `power_time_series` Delta table ...
  already holds, per `time_series_id`") describes the old per-series high-water-mark mechanism for
  both input kinds. Rewrite it to say the `PowerTimeSeries` input is filtered by existence — an
  anti-join on `(time_series_id, time)` — while the file-listing input keeps the `last_time`
  comparison, now with the lookback margin.
- `docs/live_service/operations.md`'s one mention of `select_new_rows` ("won't offer those files
  again") stays accurate — the anti-join still backs that guarantee — so it needs no change.
- No other `docs/` page names `select_new_rows`' mechanism directly (checked
  `docs/architecture/` and `docs/design-philosophy/`), so no further cross-page update is needed. This
  issue doesn't complete a roadmap item, so no ship-time triage section applies.

## Verification commands

```bash
uv run ruff check packages/nged_data
uv run ruff format --check packages/nged_data
uv run ty check
uv run pytest packages/nged_data
uv run pytest  # full suite, since select_new_rows is called from src/nged_substation_forecast/defs/assets.py
```

No network-gated or markdown-lint commands apply — this change touches no NWP conversion code and
no rendered docs page.

## Risks and open questions

- **Is `_LATE_FILE_LOOKBACK = timedelta(days=3)` the right margin?** It only bounds re-download
  cost; it cannot cause an incorrect row to be *stored*, because the row-level anti-join is the
  actual correctness gate. A file arriving later than the margin is still silently dropped at the
  listing stage — same failure mode as today, just narrower. *Recommendation*: ship with 3 days,
  since nothing today can log how late a real late file actually was; revisit once
  `_FileListingSummary`'s Dagster-UI counts show whether files repeatedly land older than that.
- **Should a same-timestamp correction (revised `power` for a `time` already on disk) be in
  scope?** This plan says no — see "Verdict" above. *Recommendation*: open a follow-up issue that
  first confirms NGED actually republishes revised values (rather than only ever adding new
  timestamps), before designing a Delta MERGE-based upsert and deciding which value wins a
  conflict.
- **Should the file-listing filter instead track ingested file paths (with S3 `e_tag`) rather than
  a time lookback margin?** That would remove the "still silently dropped past the margin" gap
  entirely, and `obstore`'s `ObjectMeta` already carries `e_tag`/`last_modified` for free from the
  listing call already made. It needs a new persisted manifest (a table or parquet file of
  ingested `(path, e_tag)` pairs), which is real new state for a benefit that only matters if NGED
  files actually land later than a few days, or get overwritten in place at the same key.
  *Recommendation*: don't build this speculatively; the bounded lookback plus the row-level
  anti-join fixes the case the issue actually reports, and the manifest is easy to add later
  without touching this change's row-level logic if evidence shows it's needed.
- **For the corrected-republication follow-up, when it's planned**: prefer `DeltaTable.merge()`
  (`WHEN NOT MATCHED THEN INSERT`, keyed on `(time_series_id, time)`) over extending
  `_existing_power_time_series_keys` with update semantics — it pushes the existence/overwrite
  decision into the storage engine instead of a bespoke Python anti-join, and upgrading a
  read-only-membership merge to also handle `WHEN MATCHED THEN UPDATE` is a small step from there.
  No `DeltaTable.merge()` call exists anywhere in the repo today, so this is a new pattern to
  introduce carefully, not a drop-in swap for this issue's fix.

## Simplicity review (first adversarial pass)

A fresh sub-agent reviewed this plan for a simpler approach before the correctness pass below. One
finding was applied; the rest confirmed the plan's existing scope calls:

- **Applied**: dropped the empty-table branch originally planned for
  `_existing_power_time_series_keys` — unreachable, since `select_new_rows` already guards
  `delta_table_exists` before either branch runs. Reflected above.
- **Applied**: added the "both mechanisms this plan fixes are real" paragraph above, citing
  `list_timeseries_json_files`'s docstring, so the plan states explicitly why a file can be
  uploaded today covering an old time window.
- **Rejected — restricting the anti-join to reporting `time_series_id`s is premature
  optimisation**: no. The table is partitioned by `time_series_id`, so the restriction is what
  keeps the new anti-join partition-pruned rather than a full-table materialisation; doing it
  unrestricted would be strictly worse than today's `time_series_coverage` scan, not simpler.
- **Rejected — reuse `TimeSeriesCoverage` instead of a new helper**: no. `TimeSeriesCoverage` only
  carries `first_time`/`last_time` per series, which cannot answer "does this exact
  `(time_series_id, time)` exist" — the new helper is genuinely new capability, not reinvention.
- **Rejected — drop the file-listing lookback margin for a pure wall-clock cutoff**: no. A new
  `time_series_id` still needs the existing `last_time.is_null()` branch to download its full
  backlog regardless of age, which still requires the coverage join — the wall-clock alternative
  relocates the arbitrary constant without removing the join it would need to remove to be
  simpler.
- **Rejected — building the `(path, e_tag)` manifest now instead of the lookback margin**: no,
  per the plan's own "Risks and open questions" item above — speculative state for an unconfirmed
  failure mode.

## Correctness and testability review (second adversarial pass)

A second, independent fresh sub-agent checked this revised plan for correctness and whether its
tests would actually catch a wrong implementation. All three findings were applied:

- **Applied**: `test_select_new_rows_file_listing`'s `old.json` fixture row sits exactly at
  `time_series_id=1`'s on-disk `last_time`, so under the loosened lookback filter it moves from
  excluded to included — the plan's earlier claim that this test needs no change was wrong.
  Updated the "Tests" section to say so and to specify the corrected expected assertions, instead
  of listing it as unchanged regression coverage.
- **Applied**: `packages/nged_data/README.md`'s `select_new_rows` public-surface entry documents
  the old per-series high-water-mark mechanism for both input kinds; the plan's docs sweep had
  checked `docs/architecture/` and `docs/design-philosophy/` but missed the package READMEs.
  Added it to "Docs to update", along with confirming `docs/live_service/operations.md`'s one
  mention needs no change.
- **Applied**: the plan named the control-flow split (anti-join for `PowerTimeSeries`,
  `time_series_coverage` for `_ProcessedFileListing`) but never said to actually move the existing
  unconditional `time_series_coverage()` call inside its own branch. Left as-is, an implementer
  could add the new helper call alongside the existing one and silently pay for both full-table
  and restricted scans every hour on the `PowerTimeSeries` path — defeating the plan's own cost
  argument, with no test able to catch it (the output would still be correct, only slower). Added
  an explicit instruction to the "What changes" section.

Everything else the reviewer attacked checked out unchanged: the claimed current-behaviour line
numbers all matched `main`, both new tests were confirmed to fail before the change and pass
after, the anti-join has no null/duplicate-key or cross-model-join hazard (`PowerTimeSeries`
already enforces `(time_series_id, time)` uniqueness, and the helper returns a plain
`pl.LazyFrame`), the retry-idempotency guarantee in `assets.py` still holds because it never
depended on which mechanism backs the dedupe, the lookback margin compares against on-disk
`last_time` rather than wall-clock time so the new test needs no time-freezing, and the
same-timestamp-correction scope cut interacts safely with `download_and_parse_files`' existing
`unique(..., keep="last")` dedupe.
