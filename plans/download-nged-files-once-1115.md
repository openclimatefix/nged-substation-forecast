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
watermark on that same commit. The append also happens when the selected files carried no readings
but raised the watermark, so a commit of zero rows advances the watermark. A run that adds no rows
and no later watermark commits nothing. This deletes the size filter, the 3-day lookback, and
`add_newest_file_of_each_series`.

## Verdict, size and departures

**Verdict: worth implementing, with one of the issue's two homes chosen and six departures.**

- Departure 1: the plan uses the Delta commit property, the issue's first option, and answers the
  issue's worry that "an hour that appends no rows writes no commit". delta-rs 1.6.6 writes a
  commit for an append of zero rows and keeps its `custom_metadata` (verified: an empty append with
  `custom_metadata={"wm": "2"}` after one with `"1"` gives history `['2', '1']`).
- Departure 2: the watermark advances after every run that raised it, whether or not the
  `TimeSeriesMetadata` upsert succeeded. A persistent upsert fault (the stored table is corrupt or
  off-contract) would otherwise freeze the watermark, and each hour would re-download every file
  since the freeze, a number that grows by about 10,000 files a day at Flexpectation v2 scale. The
  cost is bounded: a silent series publishes about 4 data-less files a day, so its next file
  restores its `Information` note within about 6 hours. `docs/live_service/operations.md` already
  accepts that a failed upsert loses that run's metadata change.
- Departure 3: a series' metadata comes only from its newest file by `end_time` in the whole
  listing, and only on a run that selected that file. `upsert_metadata` replaces a series
  wholesale, and a late or back-filled file has an old `end_time` but a new `last_modified`, so
  taking metadata from the last file downloaded would overwrite the series' current `Information`
  note and fields with those of an old window. A silent series' newest file is its latest data-less
  file, so its note still updates. Files are downloaded in `(last_modified, end_time, path)` order,
  which gives files with equal `LastModified` (the bulk backfill files were written within a few
  days) a fixed order. Where two files cover overlapping windows, the more recently written file's
  readings win the `unique(..., keep="last")` dedupe.
- Departure 4: the equivalence test compares the new ingest with a "download every file, every
  run" reference, not with a copy of the old ingest. See Tests.
- Departure 5: each file is downloaded once only up to a band. The rule
  `last_modified > watermark - margin` always re-selects files whose `last_modified` lies within
  one margin below the watermark, because nothing records which files were downloaded. Every hour
  until the next batch arrives, the ingest re-fetches that band. The band is small if files arrive
  spread over time. It is a whole batch if NGED writes every series' file for one window together
  (all series share one `<start_ms>_<end_ms>` directory, and live files land about 3.0 hours after
  the window ends). The saving is then a factor of about 12 to 13, not "once". The plan commits
  nothing for a band-only run and measures the band before it fixes the margin.
- Departure 6: one malformed file no longer stalls the whole ingest. Today a file that fails
  `pl.read_json` or Patito validation raises, the run fails, and the stall ends when the file ages
  out of the 3-day window. With a watermark the bad file would be selected forever and the backlog
  behind it would grow. `download_and_parse_files` instead skips a file that fails to parse,
  returns its path in the result, and the asset reports each skipped path to Sentry through
  `report_asset_degradation`, tagged with the path. The watermark then advances past the file. The
  cost is that a skipped file's readings are not retried, and an operator re-ingests them by hand
  from the Sentry event. This follows "liberal about missing inputs, strict about malformed ones":
  the file is rejected at the boundary and the run is not.

**Size: complex.** One line per trigger:

- What gets stored: yes. The watermark is new stored state, and empty commits now appear in
  the `power_time_series` Delta table's history.
- Production serving path: no code on the serving path changes. The ingest runs upstream of
  `live_forecasts`. Empty commits bump the raw table's Delta version, which `clean_nged_power_data`
  uses to decide whether to rebuild the cleaned table that `live_forecasts` reads. That is an
  indirect effect, covered in Risks.
- A degradation rule: yes. Where the watermark may advance when the metadata upsert fails or a file
  is malformed, and what a missing watermark means, are degradation decisions.
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
  UTC datetime. It returns `None` only for a missing table or no commit carrying the key, which
  mean "first run". A read error raises, inside the asset's S3 retry guard, so a transient fault
  reading our own store cannot trigger a whole-bucket download. It opens the table with
  `without_files=True` so the history read does not load every add action. A new module constant
  names the commit key.
- `packages/delta_store/README.md` (line 40) gains the new keyword argument and the reader.

### `packages/nged_data/src/nged_data/storage.py`

- `_RawFileListItem` and `_ProcessedFileListing` gain `last_modified`, a UTC datetime read from
  `object_meta["last_modified"]` in `list_timeseries_json_files` (obstore returns a UTC datetime).
- New `select_files_modified_since(file_listing, watermark)` returns files with
  `last_modified > watermark - _LAST_MODIFIED_MARGIN`, or every file when `watermark` is `None`.
  `_LAST_MODIFIED_MARGIN` replaces `_LATE_FILE_LOOKBACK`. The margin has to exceed the time one
  listing takes, because a listing of the whole `timeseries` prefix takes minutes, and a file
  created after the listing passed its key can be missed while a later file sets the watermark. The
  value is fixed at implementation time from a measured listing duration and a measured spread of
  `LastModified` values.
- New `newest_file_of_each_series(file_listing)` returns the path of each series' newest file by
  `(end_time, last_modified, path)`, taken from the whole listing before any selection.
  `download_and_parse_files` takes it as an argument and extracts metadata only from those files,
  whatever else it downloads. It sorts by `(last_modified, end_time, path)`. It no longer raises
  `NoNewData` when every file was data-less: it returns the metadata and an empty, validated
  `PowerTimeSeries` frame. It skips a file that fails to parse and returns the skipped paths in
  `DownloadAndParseResult`. `NoNewData` stays for an empty listing only. The data-less change is
  required, because a silent series' newest file is data-less, is now downloaded once, and its
  metadata must reach the upsert in that run.
- Deleted: `remove_small_files_from_listing`, `add_newest_file_of_each_series`,
  `_LATE_FILE_LOOKBACK`, and `select_new_rows`'s `_ProcessedFileListing` overload and branch. The
  `PowerTimeSeries` branch stays, because it is the row dedupe that makes a re-download safe.
- Docstrings that describe the removed behaviour are rewritten: the module docstring, the
  `time_series_coverage` docstring (the whole-table scan no longer runs twice an hour),
  `UpsertMetadataStats.metadata_upsert_failed` (the stale metadata is no longer retried next hour),
  and the `upsert_metadata` docstring (it says metadata comes only from files `select_new_rows`
  judged new, and gives rebuild advice).

### `src/nged_substation_forecast/defs/assets.py` (`power_time_series_and_metadata`)

- Read the watermark with `read_ingest_watermark`, inside the retry guard. List, select with
  `select_files_modified_since`, download, upsert the metadata (still swallowed on failure), dedupe
  the rows with `select_new_rows`.
- Report each skipped path with `report_asset_degradation`.
- Commit with `write_power_time_series` and `custom_metadata` set to the largest `last_modified`
  among the selected files, when the deduped rows are not empty or that value is later than the
  stored watermark. Otherwise commit nothing. An empty selection still raises `NoNewData`.
- Remove the size filter, the listing-level `select_new_rows`, and `add_newest_file_of_each_series`.
  The `nged_s3_paths` table collapses its "Files above the size threshold", "Files with new data"
  and "Files downloaded" rows to "Files modified since the watermark".
- The docstring says the watermark exists, where it lives, and that a table with no watermark
  commit downloads every file in the bucket.

### Docs and comments the change invalidates

- `packages/nged_data/README.md`: remove the entries for the deleted functions, describe the new
  ones.
- `docs/live_service/operations.md`: lines 234-235 ("The ingest downloads the newest file of every
  series, even a small file…"), lines 263-266 (a failed upsert: the watermark has moved past those
  files, and a silent series' note is restored by its next file), and the "6 to 12 hours" wording.
  Add the one-off seeding step (see Risks) and what a skipped malformed file looks like in Sentry.
- The "commits only when NGED delivers new rows, about every 6 hours" wording, which empty commits
  make untrue: `checks.py` (`_CLEANING_LAG_WARN_COMMITS` and the `cleaned_power_keeps_up_with_raw`
  description string), `cleaning_assets.py` (lines 117-118), `schedules.py` (lines 23-24),
  `docs/live_service/operations.md` (lines 263-266), and `docs/roadmap/data-cleaning.md` (lines
  40-41).
- Grep `docs/` and `src/` for `select_new_rows`, `_LATE_FILE_LOOKBACK`, `3 days`, `newest file`, and
  `re-lists NGED's bucket` and rewrite each remaining hit to describe the present behaviour.
- Test docstrings and fixtures: `test_storage.py:307`, `tests/test_assets.py:388` (its premise is
  the size filter), and every fixture that builds `_ProcessedFileListing` or `_RawFileListItem`
  without `last_modified` (`tests/test_assets.py:1463-1480`, `:1523`; `test_storage.py:29-48`,
  `:670-672`). `_FakeS3Store.list` must return `last_modified` or every existing asset test fails.

## Design-philosophy check

- **Production code, so degrade.** A missing watermark means "download everything", and the row
  dedupe makes that safe. A malformed NGED file is rejected, reported with its path, and the run
  goes on. A read error on our own Delta log raises inside the retry guard: that is our own store
  misbehaving, not the outside world, and an unreadable log would fail the append anyway.
- **The watermark comes from S3's `LastModified`, never the wall clock**, so clock skew on the
  control-plane VM cannot skip files.
- **Atomicity.** The watermark moves in the same commit as the readings, so no crash can leave it
  ahead of the rows. A crash or a `RetryRequested` retry before the commit re-downloads files that
  the dedupe absorbs. Overlapping runs can only move the watermark backwards, which costs
  re-downloads and loses nothing.
- No asset check is added or changed beyond the wording above. `power_data_is_fresh` still reads
  `time_series_coverage`, which empty commits do not affect.
- Hypotheses: this serves H1 (the pipeline keeps running through an outage) by lowering the request
  load, and trades away no principle in `design-principles.md`.

## Tests

Each test states the assertion that fails on `main` today.

1. **Equivalence replay** (`tests/test_assets.py`). `_FakeS3Store` gains a `last_modified` per file,
   a `put(path, data, last_modified)` for adding files between runs, and a count of `get` calls.
   Run the real asset over one replay twice, each time from empty storage: once normally, once with
   `read_ingest_watermark` monkeypatched to return `None` every run, which downloads the whole fake
   bucket each run. After each run, assert exact frame equality, ignoring row order, of the two
   `power_time_series` tables and the two metadata parquet files. The replay: initial files for 3
   series; new files; a back-fill of old windows (files with `end_time` two months old and a new
   `last_modified`, for series that also have a newer live file), which must add only the missing
   readings and leave the metadata unchanged; a late file whose `end_time` is more than 3 days before its series' newest
   reading and whose `last_modified` is new (the old ingest's known loss); a file that appears
   after a run with a `last_modified` earlier than the watermark but inside the margin; a series
   that stops reporting and publishes a data-less file with a new `Information` note; a rewritten
   key (same path, new `last_modified`); a run with no new files. The only expected difference is
   the `get` count. On the run with no new files, the count equals the number of files inside the
   margin band, computed from the fake store, and is smaller than the reference's. On `main` the
   watermark reader does not exist, so the test fails.
2. **A run with nothing new commits nothing.** Two runs with no new files leave the table's version
   unchanged after the first. The mutation to catch is committing on every run.
3. **Empty commit advances the watermark.** A run whose only new file is data-less appends zero rows
   and the table's newest commit carries the larger watermark. The mutation to catch is skipping the
   write when there are no rows.
4. **Margin comparison.** `select_files_modified_since` keeps a file one second after
   `watermark - margin`, drops one exactly at it, and returns every file for `None`.
5. **Watermark survives a metadata fault.** Monkeypatch `upsert_metadata` to raise. The power rows
   land, the run succeeds, and the watermark advances.
6. **A malformed file is skipped, reported and passed.** One file with invalid JSON among valid
   ones: the valid rows land, `report_asset_degradation` receives the bad file's path, and the
   watermark advances past it. The mutation to catch is letting the exception escape.
7. **`read_ingest_watermark`** returns `None` for a missing table and for a table whose commits
   carry no key, returns the newest value after several commits including an unrelated commit, and
   raises for a table whose log holds junk (written into `_delta_log/00000000000000000000.json`).
8. **`download_and_parse_files`** returns metadata and an empty power frame when every file is
   data-less. Built from a listing with explicit `last_modified` values, it extracts metadata only
   from the file `newest_file_of_each_series` names: given a newer-`end_time` file and a
   later-written older-`end_time` file of one series, the metadata comes from the newer-`end_time`
   file. It orders two files with equal `last_modified` by `end_time` then `path`. A
   `newest_file_of_each_series` test pins the `(end_time, last_modified, path)` tiebreak.
9. **`write_power_time_series`** with `custom_metadata` and zero rows adds a commit whose history
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

- **Margin and band, measured before fixing.** Before choosing the margin, the implementer lists the
  real bucket, times the listing, and tabulates the spread of `LastModified` per window directory.
  The margin must exceed the listing time. The spread says how big the re-fetched band is. Does the
  maintainer accept a band of repeat downloads, up to a whole batch, if NGED writes batches?
- **Empty commits and the cleaning asset.** `clean_nged_power_data` rebuilds the whole cleaned
  table whenever the raw table's Delta version changes. An empty commit from a data-less file
  therefore triggers a full rebuild, where today a rebuild follows only a delivery of rows. At 33
  series the silent series add about 4 such commits a day each. At Flexpectation v2 scale, hourly
  commits are likely anyway, because some series deliver nearly every hour. Recommend accepting this
  and updating the wording listed above. The alternative is a cleaning change that ignores commits
  adding no rows, which is a separate issue.
- **Cleaning-lag check.** `cleaned_power_keeps_up_with_raw` counts raw commits, and 2 commits no
  longer means 6 to 12 hours. The wording changes; whether to retune the threshold is a separate
  question.
- **First run on the existing deployment.** A table with no watermark commit downloads the whole
  bucket in one run: about 22,000 sequential GETs for 33 series over about 165 days, which could
  outlast the next hourly tick. Two overlapping runs could append duplicate rows. Recommend a
  one-off seeding step before the first deployed run: append a zero-row commit carrying a watermark
  of 3 days before the deploy, documented in `operations.md`, which mimics today's window. Does
  the maintainer want the seeding as a documented command or a small script? The implementer also
  checks whether the hourly job has a run-concurrency limit.
- **A fresh Flexpectation v2 table** would download the whole bucket in one run, all held in
  memory. Out of scope; worth a follow-up issue that chunks the first run.
- **Forcing a full re-ingest** now means deleting the Delta table or seeding an old watermark. The
  issue does not ask for the capability.
- **A rewritten key with corrected values** is downloaded again, but `select_new_rows` drops rows
  that already exist, so the corrected values are never applied. This is today's behaviour.
- **Listing cost.** The ingest still lists the whole `timeseries` prefix every hour. At 2,500 series
  the listing, not the downloads, becomes the dominant request count. Possible follow-up issue.
- **Test departs from the issue's wording.** The issue says to compare with the old ingest. The
  plan compares with "download everything", because the old ingest has known defects and a copy of
  it in the tests would only be there to be deleted. Please confirm.

## Review record

**Simplicity review (Opus).** Accepted: the commit property in place of a state file (verified
empty appends commit and keep their property, and that `write_power_time_series` is the only writer
of the table); dropping the hold-back on a metadata fault; ordering by `last_modified`; the
"download everything" test reference. Rejected: nothing. Kept, as the reviewer judged: the margin,
and `select_files_modified_since` as a separate function.

**Maintainer question after the reviews.** Asked whether a back-fill of old files is handled,
the answer was yes for the rows (a back-fill file has a new `last_modified`), but not for the
metadata, which the old-window file would have overwritten. Departure 3 and the back-fill scenario
in the replay test now cover it.

**Correctness review (Opus).** Accepted: the margin band re-fetch, with no commit on a band-only run
and a measured margin (defect 1); the cleaning-asset rebuild and the cleaning-lag check wording
(defects 2 and 3, as open questions with a recommendation); the malformed-file stall (defect 4,
Departure 6); `read_ingest_watermark` raising on read errors (defect 5); the tiebreak (defect 6);
the first-run seeding (defect 7); the missed docs, docstrings and fixtures (defect 8); the extra
replay scenarios. Accepted with a correction: the review says Departure 3's reason was wrong when a
late file is the only file of its series in a run, which is true, so the reason is restated for runs
with several files of one series. Rejected: nothing.
