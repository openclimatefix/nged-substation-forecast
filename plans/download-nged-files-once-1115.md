# Plan: download each NGED file once, by `LastModified` (#1115)

## Problem

**The ingest re-downloads about 3 days of NGED files every hour.** `select_new_rows` offers every
file whose `end_time` is later than its series' newest stored reading minus
`_LATE_FILE_LOOKBACK` (3 days). Nothing records that a file was downloaded, so each hour the ingest
fetches about 12 to 13 files per reporting series: roughly 394 requests per hour for the 33 series
now ingested, and up to about 32,500 for the roughly 2,500 series Flexpectation v2 will ingest. The
row-level dedupe makes the repeats harmless, so the cost is requests and run time.

## Solution

**Filter the bucket listing on each object's `LastModified` against one stored watermark, kept as a
custom property on the Delta commit that appends the power rows.** The watermark is the largest
`LastModified` among the files already processed. Each hour the ingest downloads files whose
`LastModified` is later than `watermark - margin`, then appends their power rows, stamping the new
watermark on that same commit. The append happens whenever any file was selected, even when every
selected file was data-less, so the commit is empty of rows but still advances the watermark. This
deletes the size filter, the 3-day lookback, and `add_newest_file_of_each_series`.

## Verdict, size and departures

**Verdict: worth implementing, with one of the issue's two homes chosen and four departures.**

- Departure 1: the plan uses the Delta commit property, as the issue's first option, and answers
  the issue's worry that "an hour that appends no rows writes no commit". delta-rs 1.6.6 writes a
  commit for an append of zero rows and keeps its `custom_metadata` (verified: an empty append with
  `custom_metadata={"wm": "2"}` after one with `"1"` gives history `['2', '1']`). The asset therefore
  appends an empty frame when the selected files carried no readings.
- Departure 2: the watermark advances after every run that selected files, whether or not the
  `TimeSeriesMetadata` upsert succeeded. A persistent upsert fault (the stored table is corrupt or
  off-contract) would otherwise freeze the watermark, and each hour would re-download every file
  since the freeze, a number that grows by about 10,000 files a day at Flexpectation v2 scale. The
  cost is bounded: a silent series publishes about 4 data-less files a day, so the next file
  restores its `Information` note within about 6 hours. `docs/live_service/operations.md` already
  accepts that a failed upsert loses that run's metadata change.
- Departure 3: files are downloaded and grouped in `last_modified` order, not `end_time` order. A
  late file (old `end_time`, new `last_modified`) can be the only file of its series in a run.
  Metadata comes from the last file in download order, and `upsert_metadata` replaces the series
  wholesale, so `end_time` order would regress the series' metadata to the late file's fields.
  Today the 3-day window and `add_newest_file_of_each_series` hide this. With `last_modified`
  order, the most recently written file wins both within a run and across runs. Where two files
  cover overlapping windows, the more recently written file's readings now win the
  `unique(..., keep="last")` dedupe, which is the meaning "newer file wins" always intended.
- Departure 4: the equivalence test compares the new ingest with a "download every file, every
  run" reference, not with a copy of the old ingest. See Tests.

**Size: complex.** One line per trigger:

- What gets stored: yes. The watermark is new stored state, and empty commits now appear in
  the `power_time_series` Delta table's history.
- Production serving path: no. The ingest runs upstream of `live_forecasts`, and no code on the
  serving path changes.
- A degradation rule: yes. Where the watermark may advance when the metadata upsert fails, and what
  an absent or unreadable watermark means, are degradation decisions.
- More than one defensible design: yes. The commit property, a state file, and a shorter lookback
  each satisfy the issue.
- Code whose callers could not be named without searching: no. `remove_small_files_from_listing`,
  `add_newest_file_of_each_series`, `download_and_parse_files` and `select_new_rows`'s file-listing
  branch each have one production caller, the asset.

**Reviews bought: all four** (two plan reviews, then the two diff reviews in `implement-issue`).

## What changes, file by file

### `packages/delta_store/src/delta_store/power_time_series.py`

- `write_power_time_series` gains a keyword argument `custom_metadata: dict[str, str] | None` and
  passes it to `write_deltalake` as `CommitProperties(custom_metadata=...)`, the mechanism
  `cleaned_power_time_series.py` already uses. It accepts a frame with zero rows.
- A new `read_ingest_watermark(table_uri, storage_options)` modelled on `read_cleaning_provenance`:
  it walks a small history window to the newest commit carrying the watermark key and returns a
  UTC datetime, or `None` for a missing table, an unreadable log, or no commit carrying the key. It
  never raises. A new module constant names the commit key.

### `packages/nged_data/src/nged_data/storage.py`

- `_RawFileListItem` and `_ProcessedFileListing` gain `last_modified`, a UTC datetime read from
  `object_meta["last_modified"]` in `list_timeseries_json_files`.
- New `select_files_modified_since(file_listing, watermark)` returns files with
  `last_modified > watermark - _LAST_MODIFIED_MARGIN`, or every file when `watermark` is `None`.
  `_LAST_MODIFIED_MARGIN` (1 hour; see open questions) replaces `_LATE_FILE_LOOKBACK`. The margin
  covers an upload still in flight during a listing, which can carry an earlier `LastModified` than
  a file that finished uploading first.
- `download_and_parse_files` sorts and groups by `last_modified`, and no longer raises `NoNewData`
  when every file was data-less: it returns the metadata and an empty, validated `PowerTimeSeries`
  frame. `NoNewData` stays for an empty listing only. The change is required, because a silent
  series' newest file is data-less, is now downloaded once, and its metadata must reach the upsert
  in that run.
- Deleted: `remove_small_files_from_listing`, `add_newest_file_of_each_series`,
  `_LATE_FILE_LOOKBACK`, and `select_new_rows`'s `_ProcessedFileListing` overload and branch. The
  `PowerTimeSeries` branch stays, because it is the row dedupe that makes a re-download safe. The
  module docstring and the `time_series_coverage` docstring (which says the whole-table scan runs
  twice an hour) are updated.

### `src/nged_substation_forecast/defs/assets.py` (`power_time_series_and_metadata`)

- Read the watermark with `read_ingest_watermark`. List, select with `select_files_modified_since`,
  download, upsert the metadata (still swallowed on failure), dedupe the rows with `select_new_rows`.
- When any file was selected, call `write_power_time_series` with the deduped rows (possibly none)
  and `custom_metadata` set to the largest `last_modified` among the selected files. An empty
  listing selection still raises `NoNewData` and commits nothing, so idle hours add no commits.
- Remove the size filter, the listing-level `select_new_rows`, and `add_newest_file_of_each_series`.
  The `nged_s3_paths` table collapses its "Files above the size threshold", "Files with new data"
  and "Files downloaded" rows to "Files modified since the watermark".
- The docstring says the watermark exists, where it lives, and that a run with no watermark (a new
  table) downloads every file in the bucket.

### Docs

- `packages/nged_data/README.md`: remove the entries for the deleted functions and describe the
  new ones.
- `docs/live_service/operations.md`, "Reading a failed metadata table upsert": replace "`select_new_rows`
  will not offer those files again" with the watermark having moved past those files, and say that a
  silent series' note is restored by its next file.
- Search `docs/` for `select_new_rows`, `_LATE_FILE_LOOKBACK`, `3 days`, and `re-lists NGED's bucket`
  and rewrite each hit to describe the present behaviour.

## Design-philosophy check

- **Production code, so degrade.** A missing, unreadable, or keyless history reads as `None`, which
  means "download everything", and the row dedupe makes that safe. `read_ingest_watermark` never
  raises. If the Delta log is unreadable the append fails anyway, through the existing path.
- **The watermark comes from S3's `LastModified`, never the wall clock**, so clock skew on the
  control-plane VM cannot skip files.
- **Atomicity.** The watermark moves in the same commit as the readings, so no crash can leave it
  ahead of the rows. A crash before the commit re-downloads files that the dedupe absorbs.
- No asset check is added or changed. `power_data_is_fresh` still reads `time_series_coverage`.
- Hypotheses: this serves H1 (the pipeline keeps running through an outage) by lowering the request
  load, and trades away no principle in `design-principles.md`.

## Tests

Each test below states the assertion that fails on `main` today.

1. **Equivalence replay** (`tests/test_assets.py`). The fake store `_FakeS3Store` gains a
   `last_modified` per file, a `put(path, data, last_modified)` for adding files between runs, and
   a count of `get` calls. Run the real asset over one replay twice, each time from empty storage:
   once normally, once with `read_ingest_watermark` monkeypatched to return `None` every run, which
   downloads the whole fake bucket each run. After each run, assert exact frame equality, ignoring
   row order, of the two `power_time_series` tables and the two metadata parquet files. The replay:
   initial files for 3 series; new files; a late file (old `end_time`, new `last_modified`) that is
   the only file of its series in its run; a series that stops reporting and publishes a data-less
   file with a new `Information` note; a rewritten key (same path, new `last_modified`); a run with
   no new files. The only expected difference is the `get` count, asserted to be smaller, and zero
   on the run with no new files. On `main` the watermark reader does not exist, so the test fails.
2. **Empty commit advances the watermark.** A run whose only new file is data-less appends zero rows
   and the table's newest commit carries the larger watermark. The mutation to catch is skipping
   the write when there are no rows.
3. **Margin comparison.** `select_files_modified_since` keeps a file one second after
   `watermark - margin`, drops one exactly at it, and returns every file for `None`.
4. **Watermark survives a metadata fault.** Monkeypatch `upsert_metadata` to raise. The power rows
   land, the run succeeds, and the watermark advances.
5. **`read_ingest_watermark`** returns `None` for a missing table, a table whose commits carry no
   key, and an unreadable log, and returns the newest value after several commits including an
   unrelated commit.
6. **`download_and_parse_files`** returns metadata and an empty power frame when every file is
   data-less, and orders by `last_modified`: given two files for one series, the one written later
   supplies the metadata whatever its `end_time`.
7. **`write_power_time_series`** with `custom_metadata` and zero rows adds a commit whose history
   carries the key.

The deleted functions' tests are deleted with them.

## Verification commands

```bash
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run pytest
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run pydoclint packages src   # CI step the skill's set omits
uv run mkdocs build --strict    # read the rendered operations page
```

## Risks and open questions

- **Margin length.** Recommend 1 hour. S3 sets `LastModified` when the upload starts, and NGED's
  files are small, so a file should become visible within seconds. A longer margin re-downloads
  every file written inside it on every run. At 2,500 series each publishing about 4 files a day, a
  1-hour margin re-fetches about 400 files an hour, against about 32,500 today. Does the maintainer
  want a longer margin?
- **Forcing a full re-ingest.** With a state file this would be "delete the file". With a commit
  property it means deleting the Delta table or adding an override. The issue does not ask for the
  capability. Is that acceptable?
- **First run.** A table with no watermark commit downloads the whole bucket in one run, all held in
  memory. This is how an empty table behaves today too, and on the existing deployment the first run
  after this change does it once. At Flexpectation v2 scale a fresh deployment could run out of
  memory. Out of scope here; worth a follow-up issue that chunks the first run.
- **A rewritten key with corrected values** is downloaded again, but `select_new_rows` drops rows
  that already exist, so the corrected values are never applied. This is today's behaviour. Whether
  NGED ever rewrites a key is not known.
- **Listing cost.** The ingest still lists the whole `timeseries` prefix every hour. At 2,500 series
  the listing, not the downloads, becomes the dominant request count. Possible follow-up issue.
- **Test departs from the issue's wording.** The issue says to compare with the old ingest. The
  plan compares with "download everything", because the old ingest has known defects (it skips files
  more than 3 days late, and its metadata goes stale) and a copy of it in the tests would only be
  there to be deleted. Please confirm.

## Review record

**Simplicity review (Opus).** Accepted: the commit property in place of a state file (verified
empty appends commit and keep their property; verified `write_power_time_series` is the only
writer of the table); dropping the hold-back on a metadata fault; ordering by `last_modified`; the
"download everything" test reference. Rejected: nothing. Kept as the reviewer judged: the margin,
and `select_files_modified_since` as a separate function.

**Correctness review.** (Filled in after the review.)
