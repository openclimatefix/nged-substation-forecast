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
restricted `(time_series_id, time)` frame (or an empty one of the right schema if the table
doesn't exist, mirroring `time_series_coverage`'s empty-table branch). Filtering by
`time_series_ids` before the join keeps the build side of the anti-join proportional to the
handful of series that reported this hour, not the whole table — see the cost note below.

`select_new_rows`'s `_ProcessedFileListing` branch keeps using `TimeSeriesCoverage.last_time` (see
next item), so `time_series_coverage` stays as it is and both call sites keep working.

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
`_ProcessedFileListing`; a missing Delta table is handled the same way `time_series_coverage`
already handles it (empty frame, not an exception). No asset check is added or changed. This
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
- Keep `test_select_new_rows_power_time_series` and `test_select_new_rows_file_listing` unchanged
  as regression coverage for the ordinary in-order case.
- Extend `test_select_new_rows_power_time_series` (or add a sibling) to cover **more than one
  reporting `time_series_id` in the same call**, confirming the `time_series_ids`-restricted
  anti-join scan doesn't accidentally scope to only the first series' keys or leak another
  series' rows into the result.

## Docs to update

- `select_new_rows` and `time_series_coverage` docstrings, as described above — the "Write about
  the present, not the past" rule means these should describe the anti-join and the lookback
  margin as how the code works now, not narrate that a high-water mark used to be there.
- No `docs/` page names `select_new_rows`' mechanism directly (checked
  `docs/architecture/` and `docs/design-philosophy/`), so no cross-page update is needed. This
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
