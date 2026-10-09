# Plan: download each NGED file once, with a ledger of downloaded files (#1115)

## Problem

**The ingest re-downloads about 3 days of NGED files every hour.** `select_new_rows` offers every
file whose `end_time` is later than its series' newest stored reading minus
`_LATE_FILE_LOOKBACK` (3 days). Nothing records that a file was downloaded, so each hour the ingest
fetches about 12 to 13 files per reporting series: roughly 394 requests per hour for the 33 series
now ingested, and up to about 32,500 for the roughly 2,500 series Flexpectation v2 will ingest. The
row-level dedupe makes the repeats harmless, so the cost is requests and run time.

## Solution

**Record every downloaded file in a ledger, and download only listed files that the ledger lacks.**
The ledger is a parquet file of `(path, last_modified)`, one row per downloaded NGED file, stored
beside `metadata.parquet`, together with the Delta table id of the `power_time_series` table the
files were loaded into. Each hour the ingest lists the bucket, which returns each object's
`LastModified`, and anti-joins the listing with the ledger. A file whose path is new, or whose path
is known but whose `LastModified` changed because NGED rewrote it, is downloaded. After the power
rows are appended, the ingest writes the ledger again with the downloaded files added. A ledger
whose table id differs from the current table's id, or whose table does not exist, counts as empty.
This deletes the size filter, the 3-day lookback, and `add_newest_file_of_each_series`, and it adds
no margin, no watermark, and no change to the `power_time_series` Delta table.

**Assumption.** NGED have said they plan to change how they deliver files before Flexpectation
v2, without details, and the change may be delayed. The plan assumes the files keep today's layout.

### Walk through

**Each hour, the ingest compares the bucket's listing with a record of what it has already
downloaded, downloads only the difference, and updates the record last.** The sub-sections below
follow one run, then its branches, then its failures.

#### The happy path

1. **The run reads the ledger.** `read_downloaded_files` returns the ledger as a frame of
   `(path, last_modified)`, or an empty frame when the ledger's table id does not match the
   `power_time_series` table's current id. The id check exists because the ledger describes one
   specific table: if an operator rebuilds the table, the old ledger would claim that files are
   loaded which the new table lacks. The read comes before the listing is filtered because the
   filter is an anti-join with the ledger, and it sits outside the S3 retry guard because a
   fault in our own ledger must not be reported as a fault in NGED's bucket.
2. **The run lists the bucket.** `list_timeseries_json_files` now also returns each object's
   `last_modified`. The ledger needs `last_modified` because a rewritten key keeps its path, and
   only `LastModified` shows that the content changed.
3. **The run selects files.** `select_files_not_yet_downloaded` keeps the listed files whose
   `(path, last_modified)` is not in the ledger. No time margin is needed: a file that was not yet
   visible in an earlier listing was never recorded, so a later listing selects the file.
4. **The run notes each series' newest file.** The asset computes the newest file of each series by
   `end_time` from the whole listing, before the selection. The asset needs the whole listing
   because a series' newest file may have been downloaded in an earlier hour and be absent from
   the selection.
5. **The run downloads and parses the selected files.** `download_and_parse_files` sorts the
   selection by `end_time` and works through it in chunks. For each chunk it fetches all the files
   concurrently, at most 32 at a time, through `asyncio.run`, and then parses them in order. The
   function sorts for itself because an anti-join does not promise to keep the listing's order, and
   both in-batch dedupes keep the last row, so the more recent window must come last. The
   concurrency exists because a first run fetches tens of thousands of files, and `gather` returns
   results in input order so that the concurrency cannot reorder the dedupes. The chunks exist so
   that the raw JSON of tens of thousands of files is never held in memory at once.
6. **The run keeps only the metadata of newest files.** The asset filters the returned metadata to
   the series whose newest file (step 4) is in the selection. The filter exists because
   `upsert_metadata` replaces a series wholesale, and a late or back-filled file would otherwise
   overwrite the series' current `Information` note with an old one.
7. **The run upserts the `TimeSeriesMetadata` table.** This happens before the power write, as it
   does today, and a failure is swallowed so that a fault in a derived table cannot stop the power
   readings landing.
8. **The run drops readings already on disk.** `select_new_rows` anti-joins the parsed rows with the
   stored `(time_series_id, time)` keys, because a rewritten key and a back-filled file can
   re-deliver readings the table already has.
9. **The run appends the new rows.** `write_power_time_series` is unchanged.
10. **The run writes the ledger last.** `write_downloaded_files` writes the old ledger plus the
    files this run downloaded, stamped with the table's current id, after the append. The ledger
    comes second so that it can never be ahead of the readings: a crash between the two steps
    leaves some files unrecorded, and the next run downloads them again and the row dedupe absorbs
    them.

#### Branches off the happy path

- **First run, with no ledger file.** `read_downloaded_files` returns an empty frame, so every
  listed file is selected. This is the same as today's first run on an empty table. On the existing
  deployment the operator seeds the ledger first (see Risks).
- **The operator rebuilds the `power_time_series` table.** The new table has a new id, so the old
  ledger counts as empty and the full history is downloaded again, as the rebuild intends. Restoring
  the table to an older Delta version keeps the id, so the operator must delete the ledger too.
- **An hour with nothing new.** Every listed file is in the ledger, so the selection is empty.
  `download_and_parse_files` raises `NoNewData`, as today, the asset catches it and writes nothing,
  and the run makes no `get` request.
- **An hour in which every new file is data-less.** A series that has stopped reporting publishes
  files with a fault note and no readings. `download_and_parse_files` returns the metadata and an
  empty power frame instead of raising, the metadata upsert runs, no power rows are appended, and
  the ledger records the files. Without the ledger entry the same files would be downloaded every
  hour. The `power_time_series` table gets no commit, so the only change `clean_nged_power_data`
  can notice is the new metadata, as today.
- **A late file.** A file with an old `end_time` and a new path or `last_modified` is selected, and
  its readings land. Its metadata is dropped in step 6 unless it is the series' newest file.
- **A back-fill of old windows.** The same as a late file, in bulk. The run may download thousands
  of files, and the readings the table lacks are appended.
- **A file that becomes visible late.** An earlier listing did not contain the file, so the ledger
  does not either, and the next listing selects it. Nothing is lost.
- **A run in which no series' newest file was selected.** A back-fill-only run is one. The filter in
  step 6 leaves no metadata, so the upsert is skipped.
- **A first run that appends no rows.** The `power_time_series` table does not exist yet, so there
  is no table id to stamp. The ledger is not written, and the next run downloads the same data-less
  files again. The state ends once a run appends rows.

#### Error handling

- **NGED's bucket fails to list or download.** The existing retry guard raises `RetryRequested`.
  Nothing was appended or recorded, so the retry repeats the same selection.
- **The ledger cannot be read.** A missing file reads as an empty ledger, but a corrupt or
  unreadable file raises, with a message naming the ledger's path. The read is outside the retry
  guard, so the run fails at once and Sentry names the ledger, not NGED's bucket. A fault in our
  own ledger must not read as an empty ledger, because that would download the whole bucket.
  `read_downloaded_files` uses `object_exists`, so only a `FileNotFoundError` counts as missing and
  a transient S3 error raises.
- **The metadata upsert fails.** The failure is swallowed and reported as today, and the ledger is
  still written. If the ledger were held back, a persistent upsert fault would make every hour
  download a growing backlog. A silent series' note is restored by its next file, about 6 hours
  later.
- **A file is malformed or breaks the `TimeSeriesMetadata` contract.** The run raises before
  anything is appended or recorded. The same file is selected every later hour, so the ingest
  stalls until someone fixes it. The failure is loud because a contract violation is our own bug
  (for example the `licence_area` enum holding only `EMids` when a new licence area appears), and
  skipping the file would lose its readings for good. After the fix, the whole backlog ingests.
- **The process dies between the append and the ledger write.** The ledger is behind the rows, and
  the next run downloads the unrecorded files again. The row dedupe drops the repeated rows.
- **The ledger write fails.** The failure is swallowed and reported through
  `report_asset_degradation`, as the metadata upsert's failure is. The rows have landed, and
  raising would stop `clean_nged_power_data` in the same job and leave the cleaned table stale for
  `live_forecasts`. The next run downloads the unrecorded files again, which the row dedupe
  absorbs, and Sentry names the ledger.

## Verdict, size and departures

**Verdict: worth implementing, with a ledger in place of the issue's watermark and five
departures.**

- Departure 1: the plan records downloaded files in a ledger, not a watermark (the issue's two
  homes both store one). A watermark filters on a time, so it needs a margin for files that appear
  in a listing late, and the margin re-selects a band of files every hour. It also needs empty
  commits to advance when a batch is all data-less, and those commits make the cleaning asset
  rebuild. The ledger has no margin, no band, no empty commits, and no lost-file hazard from late
  visibility: a file that is absent from a listing is simply not recorded. Its cost is state that
  grows with the bucket instead of one timestamp, and a link to the table it describes (the table
  id). A synthetic ledger of 3.65 million rows (2,500 series, 4 files a day, one year) read,
  anti-joined with a same-size listing, de-duplicated and written in about 0.4 seconds in total,
  holds about 395 MB in memory, and peaks near 2.7 GB with the listing, the ledger and the merged
  ledger all held. The peak is no problem at 33 series.
- Departure 2: the ledger is written even when the `TimeSeriesMetadata` upsert failed. A persistent
  upsert fault (the stored table is corrupt or off-contract) would otherwise freeze the ledger,
  and each hour would re-download every file since the freeze, a number that grows by about 10,000
  files a day at Flexpectation v2 scale. The cost is bounded: a silent series publishes about 4
  data-less files a day, so its next file restores its `Information` note within about 6 hours.
  `docs/live_service/operations.md` already accepts that a failed upsert loses that run's metadata
  change.
- Departure 3: a series' metadata comes only from its newest file by `end_time` in the whole
  listing, and only on a run that selected that file. `upsert_metadata` replaces a series
  wholesale, and a late or back-filled file has an old `end_time`, so taking metadata from the last
  file downloaded would overwrite the series' current `Information` note and fields with those of
  an old window. A silent series' newest file is its latest data-less file, so its note still
  updates. The rule lives in the asset, so `download_and_parse_files` keeps its signature.
- Departure 4: the equivalence test compares the new ingest with a "download every file, every
  run" reference, not with a copy of the old ingest. See Tests.
- Departure 5: the plan folds in the concurrent download that the issue does not mention. With the
  ledger, the cost that remains is the first run, which downloads every file once. Fetching
  concurrently turns a first run of 20 to 45 minutes into about 5, so an overlapping hourly run
  becomes unlikely and a seeded cutover becomes optional. At today's 33 series, the saving in
  steady state is small.

**Size: complex.** One line per trigger:

- What gets stored: yes. The ledger is a new stored file.
- Production serving path: no. The ingest runs upstream of `live_forecasts`, and the plan adds no
  commit to the `power_time_series` table, so `clean_nged_power_data` rebuilds exactly when it does
  today.
- A degradation rule: yes. That the ledger is written when the metadata upsert fails, that a
  failed ledger write is swallowed, that a missing ledger means "download everything", and that a
  malformed file stalls the ingest, are degradation decisions.
- More than one defensible design: yes. A ledger, a watermark in the Delta commit, a watermark
  in a state file, and a shorter lookback each satisfy the issue.
- Code whose callers could not be named without searching: no. `remove_small_files_from_listing`,
  `add_newest_file_of_each_series`, `download_and_parse_files` and `select_new_rows`'s file-listing
  branch each have one production caller, the asset.

**Reviews bought: all four** (two plan reviews, then the two diff reviews in `implement-issue`).
The plan has had two simplicity reviews and three correctness reviews so far.

## What changes, file by file

### `packages/nged_data/src/nged_data/storage.py`

- `_RawFileListItem` and `_ProcessedFileListing` gain `last_modified`, a UTC datetime read from
  `object_meta["last_modified"]` in `list_timeseries_json_files` (obstore returns a UTC datetime
  with microseconds, and the value recorded in the ledger must come from the listing, not from a
  `get` response, whose header has only seconds).
- A new private Patito model `_DownloadedFiles` holds `path` and `last_modified`, beside
  `_ProcessedFileListing`. The ledger file also carries the power table's Delta id.
- New `read_downloaded_files(ledger_path, power_table_path, storage_options)` returns the ledger as
  `pt.DataFrame[_DownloadedFiles]`. It returns an empty, typed frame when the ledger file is
  missing (tested with `object_exists`), when the power table does not exist, or when the ledger's
  table id differs from `DeltaTable(...).metadata().id`, the id the cleaning asset already reads.
  An unreadable or invalid file raises, naming the path.
- New `write_downloaded_files(ledger_path, power_table_path, downloaded, storage_options)` writes
  the old ledger plus `downloaded`, de-duplicated on `path` keeping the later `last_modified`,
  stamped with the table's current id. It does nothing when the table does not exist. Its I/O
  mirrors `upsert_metadata`'s handling of local paths and object-store URIs. A local write goes
  through a temporary file and a rename, so a crash cannot leave a torn ledger.
- New `select_files_not_yet_downloaded(file_listing, ledger)` anti-joins the listing with the
  ledger on `(path, last_modified)`.
- `download_and_parse_files` stays synchronous and keeps its signature. It sorts its input by
  `end_time`, then works through the sorted input in chunks of `_DOWNLOAD_CHUNK_FILES` files (a
  `Final` constant, about 500). For each chunk it calls `asyncio.run` on a coroutine that fetches
  every file of the chunk with `store.get_async` and `bytes_async`, at most
  `_MAX_REQUESTS_IN_FLIGHT` (about 32) at a time through an `asyncio.Semaphore`, and joins them
  with `asyncio.gather`, which returns the bytes in input order. It then parses the chunk's files
  sequentially in that order. Chunks bound the memory held in raw JSON. The semaphore bounds the
  load on NGED's bucket. `gather` returning in input order keeps the `end_time` order that both
  `keep="last"` dedupes need. The first failing request raises out of `gather`, and
  `asyncio.run` cancels the unfinished requests when it exits, so a failure fails the run as it
  does today. A fault cannot arise from an event loop already running, because Dagster runs a
  synchronous asset in a thread with no loop. The function no longer raises `NoNewData` when every
  file was data-less: it returns the metadata and an empty, validated `PowerTimeSeries` frame.
  `NoNewData` stays for an empty selection only. The change is required, because a silent series'
  newest file is data-less, is now downloaded once, and its metadata must reach the upsert in that
  run. The `get_async` TODO is deleted.
- Deleted: `remove_small_files_from_listing`, `add_newest_file_of_each_series`,
  `_LATE_FILE_LOOKBACK`, and `select_new_rows`'s `_ProcessedFileListing` overload and branch. The
  `PowerTimeSeries` branch stays, because it is the row dedupe that makes a re-download safe.
- Docstrings that describe the removed behaviour are rewritten: the module docstring, the
  `time_series_coverage` docstring (the whole-table scan no longer runs twice an hour), the
  `TimeSeriesCoverage` docstring (lines 318-321, which says `select_new_rows` reads `last_time`),
  `UpsertMetadataStats.metadata_upsert_failed` (the stale metadata is no longer retried next
  hour), and the `upsert_metadata` docstring (it says metadata comes only from files
  `select_new_rows` judged new, and gives rebuild advice that must be restated).

### `packages/contracts/src/contracts/settings.py`

- New `downloaded_files_path`, defaulting to `uri_join(nged_data_path, "downloaded_files.parquet")`,
  beside `metadata_path`. The name does not end in `_data_path`, so the settings tests that
  enumerate path fields (`packages/contracts/tests/test_settings.py:33-46` and the guard at
  `:79-80`) gain it explicitly.

### `src/nged_substation_forecast/defs/assets.py` (`power_time_series_and_metadata`)

- Read the ledger before the retry guard. Inside the guard, list, select with
  `select_files_not_yet_downloaded`, compute each series' newest file from the whole listing, and
  download. After the guard, filter the metadata to the series whose newest file was selected,
  upsert the metadata when any remains (still swallowed on failure), dedupe the rows with
  `select_new_rows`, append when any rows remain, then write the ledger (swallowed and reported on
  failure). An empty selection still raises `NoNewData` and writes nothing.
- Remove the size filter, the listing-level `select_new_rows`, and `add_newest_file_of_each_series`.
  The `nged_s3_paths` table collapses its "Files above the size threshold", "Files with new data"
  and "Files downloaded" rows to "Files not yet downloaded". The ledger write comes after the
  `_PowerTimeSeriesSummary` table only if `tests/test_assets.py:557-560` still holds; otherwise
  that comment is rewritten.
- The comments and docstring that call the metadata table "data NGED re-delivers every run"
  (`assets.py:202-207`) are rewritten: a lost metadata change now lasts until the series' next
  file. The asset docstring says the ledger exists, where it lives, and that a missing ledger
  downloads every file in the bucket.

### Docs and comments the change invalidates

- `packages/nged_data/README.md`: remove the entries for the deleted functions, describe the new
  ones.
- `docs/live_service/operations.md`: lines 234-235 ("The ingest downloads the newest file of every
  series, even a small file…") and lines 276-283 (a failed upsert: the ledger has recorded those
  files, and a silent series' note is restored by its next file). Add the cutover procedure, how
  to force a re-download of a file (remove its row from the ledger), that a restore of the
  `power_time_series` table to an older version needs the ledger deleted, and how to rebuild the
  metadata table (delete `metadata.parquet` and the ledger, then run the ingest, which downloads
  the whole bucket). `docs/live_service/intervention-log.md:208-211` records a past rebuild and
  needs no edit.
- Grep `docs/` and `src/` for `select_new_rows`, `_LATE_FILE_LOOKBACK`, `3 days`, `newest file`,
  `re-lists NGED's bucket`, `re-delivers`, and `re-delivered`, and rewrite each remaining hit to
  describe the present behaviour.
- Test docstrings, stubs and fixtures: `test_storage.py:307`, `tests/test_assets.py:237` ("derived
  data NGED re-delivers"), `:388` (its premise is the size filter), and every fixture that builds
  `_ProcessedFileListing` or `_RawFileListItem` without `last_modified` (`tests/test_assets.py:1463-1480`,
  `:1523`; `test_storage.py:29-48`, `:379`, `:531`, `:626-628`, `:670-672`). `_FakeS3Store.list`
  must return `last_modified` or every existing asset test fails. The stubs of
  `download_and_parse_files` at `tests/test_assets.py:368, 430, 474, 520` keep their signature.
- The `get_async` TODO in `download_and_parse_files` is deleted. The operations page's first-run
  estimate becomes a few minutes. `packages/nged_data/README.md` mentions the concurrent
  download.
- No change is needed to `checks.py`, `cleaning_assets.py`, `schedules.py`, or the
  `delta_store` package, because the plan adds no commit to the `power_time_series` table.

## Design-philosophy check

- **Production code, so degrade, except where the fault is ours.** A missing ledger means "download
  everything", and the row dedupe makes that safe. A failed ledger write or metadata upsert is
  swallowed and reported, because each is a derived-state fault and the readings must still land.
  Two faults stop the ingest: a corrupt or unreadable ledger, because reading it as empty would
  download the whole bucket, and a malformed NGED file or contract violation, because "contract
  violation" is our own bug in `CLAUDE.md`'s terms and skipping the file would lose its readings
  silently and for good. Both stalls lose no data and end when the ledger, the contract, or the
  parser is fixed.
- **Idempotence replaces atomicity.** The ledger is written after the append, so it is never ahead
  of the rows, and a crash re-downloads files that the dedupe absorbs. The inherent-stability
  principle of idempotent writes is kept.
- **The telemetry names the fault.** A ledger fault names the ledger's path, outside the guard whose
  warning says NGED's bucket failed.
- No asset check is added or changed. `power_data_is_fresh` still reads `time_series_coverage`.
- Hypotheses: this serves H1 (the pipeline keeps running through an outage) by lowering the request
  load, and trades away no principle in `design-principles.md`.

## Tests

Each test states the assertion that fails on `main` today. Most tests fail on `main` only because
the new function does not exist, so the named mutation is the real test of each.

1. **Equivalence replay** (`tests/test_assets.py`). `_FakeS3Store` gains a `get_async` (with a
   `bytes_async` on its result), a `last_modified` per file, a `put(path, data, last_modified)` for
   adding files between runs, and a count of `get_async` calls.
   Run the real asset over one replay twice, each time from empty storage: once normally, once with
   `read_downloaded_files` monkeypatched to return an empty ledger every run, which downloads the
   whole fake bucket each run. After each run, assert exact frame equality, ignoring row order, of
   the two `power_time_series` tables and the two metadata parquet files. The replay: initial files
   for 3 series; new files; a back-fill of old windows (files with `end_time` two months old, a new
   path, and an `Information` note that differs from the live file's) for series that also have a
   newer live file, which must add only the missing readings and leave the metadata unchanged; a
   late file whose `end_time` is more than 3 days before its series' newest reading; a file that
   appears in the bucket only after a later file was already downloaded; a series that stops
   reporting and publishes a data-less file with a new `Information` note; a rewritten key (same
   path, new `last_modified`) that adds one reading and changes the newest file's note; a run with
   no new files. The two arms share the parsing code, so equality alone cannot catch a parsing bug.
   The test therefore also asserts directly: the stopped series' new `Information` note is stored,
   the late file's rows landed, the rewritten key's added reading landed, and the back-fill left
   the metadata unchanged. The ledger arm's `get_async` count is asserted exactly on every run, and is
   zero on the run with no new files. The reference arm's count is every file, every run.
2. **A run with nothing new does nothing.** A second run over an unchanged bucket makes zero `get_async`
   calls, appends nothing, and leaves the ledger file unchanged. The mutation to catch is selecting
   files within a window instead of by the ledger.
3. **A data-less hour is recorded once.** A run whose only new file is data-less upserts the
   metadata, appends no rows, and records the file, so the next run makes zero `get_async` calls. The
   mutation to catch is not recording a file that yielded no rows.
4. **`select_files_not_yet_downloaded`** excludes a file whose `(path, last_modified)` is in the
   ledger, includes one whose path is in the ledger with a different `last_modified`, includes a
   new path, and returns every file for an empty ledger. **`download_and_parse_files`** given a
   shuffled selection processes files in `end_time` order: of two files of one series that overlap
   in time and differ in a reading, the later window's reading survives.
5. **Ledger survives a metadata fault.** Monkeypatch `upsert_metadata` to raise. The power rows
   land, the run succeeds, and the ledger records the files.
6. **A malformed file fails the run and records nothing.** One file with invalid JSON among valid
   ones: the run raises (as `RetryRequested`, then failure), nothing is appended, and the ledger is
   unchanged. The mutation to catch is a `try`/`except` that skips the file.
7. **Ledger write failures lose nothing.** Monkeypatch `write_downloaded_files` to raise on one run.
   The run succeeds, the rows are on disk, `report_asset_degradation` is called, and the next run
   downloads the same files again and appends no duplicate rows. The mutation to catch is writing
   the ledger before the append, which fails the "rows are on disk" assertion once the patched
   write raises first.
8. **The ledger's storage.** `read_downloaded_files` returns an empty typed frame for a missing
   file, and raises, naming the path, for a corrupt one and for a transient error on a moto S3
   bucket. A write followed by a read returns the same rows, on a local path and on moto S3
   (reset per test, per `docs/architecture/testing.md`), and `last_modified` compares equal after
   the round trip. `write_downloaded_files` de-duplicates on `path` keeping the later
   `last_modified`.
9. **A rebuilt table re-downloads its history.** Run the asset, delete the `power_time_series`
   table, run again: the full history is downloaded and appended. The mutation to catch is a
   ledger with no table id. A restore case: deleting only the ledger also re-downloads everything.
10. **`download_and_parse_files`** returns metadata and an empty power frame when every file is
    data-less.
11. **Settings.** `downloaded_files_path` is covered by the enumerating settings tests.
12. **The download order does not depend on completion order.** A fake store whose `get_async`
    finishes the later-window files first: of two overlapping files of one series that differ in a
    reading, the later window's reading survives, and the series' metadata comes from the later
    file. The mutation to catch is parsing in completion order.
13. **The download is concurrent and capped.** With 100 files, a fake store that records its peak
    number of requests in flight reports a peak above 1 and no higher than the cap. The mutation to
    catch is a sequential loop (peak 1) and a missing semaphore (peak 100).
14. **A failing request fails the run and records nothing.** One `get_async` raises among many: the
    function raises, no rows are appended, and the ledger is unchanged. The tests are synchronous,
    because `asyncio.run` raises inside a running event loop.

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

- **Cutover on the existing deployment.** With no ledger file, the first run would download the
  whole bucket: about 22,000 GETs for 33 series over about 165 days. With the concurrent download
  the run takes about 3 to 6 minutes: about 1 minute of GETs at 32 in flight and 100 ms a GET
  (unmeasured), plus 1.5 to 4 minutes of sequential parsing. The size filter has meant that no
  data-less file has been parsed except each series' newest, so the first run also parses every
  historical data-less file, and one odd file would stall the ingest under the loud-failure rule.
  A second run that started before the first finished would find no ledger and append duplicate
  rows, so the operator pauses the hourly schedule for the first run. An optional seeded cutover
  avoids the historical data-less files: before the first deployed run, the operator writes a
  ledger from the current listing, containing every file whose `LastModified` is more than 3 days
  old, using `write_downloaded_files` in a documented snippet. Those files' rows are already in
  the table, except for files the old 3-day rule had skipped, which are lost already. The first run
  then downloads about 3 days of files. Recommend the unseeded first run with the schedule paused,
  because a stall on an odd file is better found now than at v2 ingestion. Does the maintainer
  prefer the seeded cutover?
- **No run-concurrency limit exists on the ingest.** The hazard of two overlapping runs appending
  the same rows exists today and the plan neither adds nor removes it. A `pool` of 1 for the asset
  would remove it. Recommend a separate issue. The concurrent download shortens the first run to a
  few minutes, so an overlap with the next hourly tick becomes unlikely, not impossible.
- **A fresh table at Flexpectation v2 scale is parked.** About 2,500 series over 165 days is about
  1.65 million files. That needs chunked commits with a ledger write per chunk, and parsing in
  parallel, because sequential parsing at about 5 ms a file would take about 2 hours. NGED has said
  they plan to change how they deliver files before v2, without details, and the change may be
  delayed. The plan assumes the files keep today's layout and does not build this machinery. If
  NGED's change lands, the machinery may be moot.
- **A stalled ingest on a contract violation.** At Flexpectation v2 scale, files from licence areas
  other than `EMids` will fail the `TimeSeriesMetadata` enum, and the ingest will stall on them
  until the contract is widened. Widening a contract needs the maintainer's agreement first, so
  the stall is also a prompt to do that work before v2 ingestion begins. Does the maintainer
  prefer that the ingest skips only files that are not valid JSON, with a circuit breaker and a
  Sentry event naming the file? Recommend no: `report_asset_degradation` takes no path tag and
  needs a fingerprint, and skipping gives up the guarantee that no readings are lost.
- **Ledger size and rewrite cost.** The ledger grows by one row per downloaded file and is written
  whole each run that downloads a file. At 2,500 series for one year that is 3.65 million rows,
  about 16 MB on disk, with a peak near 2.7 GB of memory while the listing, the ledger and the
  merged ledger are all held. The implementer may key the ledger on integers (`time_series_id`,
  `start_time`, `end_time`, `last_modified`) instead of path strings to cut memory. Neither is a
  problem at 33 series.
- **The ledger's home.** The `_DownloadedFiles` model is private to `storage.py`, like
  `_ProcessedFileListing`, so no contract in `packages/contracts/` changes. It is nevertheless
  persisted. Does the maintainer want it in `contracts` instead? Recommend `storage.py`.
- **Stored metadata no longer repairs itself every hour.** Since PR #1112, every run re-reads each
  series' newest file, so deleting `metadata.parquet` rebuilds it within an hour. After this
  change, a rebuild needs the ledger deleted too, which downloads the whole bucket, and a series
  first seen while upserts are failing has no metadata row until its next file. The cleaning
  joins metadata with a left join, so the impact is limited. The operations page gets the
  procedure.
- **A rewritten key with corrected values** is downloaded again, because its `LastModified`
  changes, but `select_new_rows` drops rows that already exist, so the corrected values are never
  applied. This is today's behaviour.
- **Listing cost.** The ingest still lists the whole `timeseries` prefix every hour. At 2,500 series
  the listing, not the downloads, becomes the dominant request count. Possible follow-up issue.
  Listing only recent windows would miss a back-fill of old windows.
- **Test departs from the issue's wording.** The issue says to compare with the old ingest. The
  plan compares with "download everything", because the old ingest has known defects and a copy of
  it in the tests would only be there to be deleted. Please confirm.
- **The plan departs from the issue's proposal.** The issue proposes a watermark on `LastModified`.
  The plan proposes a ledger instead. Please confirm.

## Review record

**Simplicity review 1 (Opus).** Proposed the commit property in place of a state file, dropping
the hold-back on a metadata fault, and the "download everything" test reference. The plan kept the
second and third. The first is superseded by simplicity review 2.

**Correctness review 1 (Opus).** Found that a margin re-selects a band of files every hour, that
empty commits trigger cleaning rebuilds, that a malformed file stalls a watermark for good, that
`read_ingest_watermark` should raise on read errors, and that the first run needs a procedure. The
ledger removes the first two. Its malformed-file finding led to a skip-the-file design, which
correctness review 2 overturned. The loud stall and the first-run procedure remain.

**Maintainer question after review 1.** Asked whether a back-fill of old files is handled, the
answer was yes for the rows, but not for the metadata, which the old-window file would have
overwritten. Departure 3 and the back-fill scenario in the replay test cover it.

**Correctness review 2 (Opus).** Found that skipping malformed files silently drops a whole
licence area's readings (the plan now fails loudly), that overlapping runs append duplicate rows
(now a separate issue, because the hazard predates this change), that `without_files=True` is
deprecated in deltalake 1.6.6 (gone with the Delta watermark), and that a run selecting no
series' newest file crashes `pl.concat` (the filter in step 6 now leaves an empty frame and the
upsert is skipped).

**Simplicity review 2 (Opus, wide net).** Accepted: replace the watermark with a ledger, because
every complication in the watermark design (margin, band, empty commits, cleaning rebuilds, the
`max` rule, the history-window reader, the lost-file hazard) comes from keeping state in the
`power_time_series` table's Delta log; filter the metadata in the asset and keep
`download_and_parse_files`'s signature; move the `NGED_INGEST` pool to its own issue; replace the
seeding commit with a documented cutover. Rejected: nothing. Noted and left out of scope: merging
the ingest and cleaning assets, listing only recent windows, S3 event notifications, and a
manifest object.

**Correctness review 3 (Opus, of the ledger design).** Accepted: the ledger was not tied to the
table it describes, so a rebuilt or restored table would silently lose history, so the ledger now
carries the table's Delta id (defect 1); `download_and_parse_files` must sort by `end_time` itself,
because an anti-join does not keep order (defect 2); a failed ledger write is swallowed and
reported, because raising stops the cleaning asset (defect 3, which also corrects the plan's
earlier reason for raising); the replay test needs back-fill files with a differing note, a
rewritten key that changes something observable, and exact `get` counts (defect 4); the ledger
read moves outside the retry guard so that a ledger fault is not reported as an NGED outage
(defect 5); missed comments, docstrings and settings tests are listed (finding 6); the cutover
risks, including parsing every historical data-less file and the 22,000-request restart, are in
Risks, and the recommendation is now a seeded cutover (finding 7); the peak-memory figure is
corrected (finding 8); the metadata-rebuild procedure goes in the operations page (finding 9); the
wording errors are fixed (finding 10). Rejected: nothing. The reviewer checked and cleared the
`LastModified` round trip through parquet, transient S3 errors being mistaken for a missing file,
the claim that no commit is added to the power table, and the ledger ordering in each crash case.
