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
beside `metadata.parquet`. Each hour the ingest lists the bucket, which returns each object's
`LastModified`, and anti-joins the listing with the ledger. A file whose path is new, or whose path
is known but whose `LastModified` changed because NGED rewrote it, is downloaded. After the power
rows are appended, the ingest writes the ledger again with the downloaded files added. This deletes
the size filter, the 3-day lookback, and `add_newest_file_of_each_series`, and it adds no margin, no
watermark, and no change to the `power_time_series` Delta table.

### Walk through

**Each hour, the ingest compares the bucket's listing with a record of what it has already
downloaded, downloads only the difference, and updates the record last.** The sub-sections below
follow one run, then its branches, then its failures.

#### The happy path

1. **The run reads the ledger.** `read_downloaded_files` returns the ledger as a frame of
   `(path, last_modified)`. This comes before the listing is filtered because the filter is an
   anti-join with the ledger.
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
5. **The run downloads and parses the selected files.** `download_and_parse_files` reads files in
   `end_time` order, as today, and returns the metadata, the power rows, and a count of dropped
   rows. The `end_time` order makes the more recent window's readings win the dedupe inside the
   batch.
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
    files this run downloaded, after the append. The ledger comes second so that it can never be
    ahead of the readings: a crash between the two steps leaves some files unrecorded, and the next
    run downloads them again and the row dedupe absorbs them.

#### Branches off the happy path

- **First run, with no ledger file.** `read_downloaded_files` returns an empty frame, so every
  listed file is selected. This is the same as today's first run on an empty table. On the existing
  deployment it would download the whole bucket (see Risks, and the first-run procedure there).
- **An hour with nothing new.** Every listed file is in the ledger, so the selection is empty. The
  asset raises `NoNewData`, as today, writes nothing, and makes no `get` request.
- **An hour in which every new file is data-less.** A series that has stopped reporting publishes
  files with a fault note and no readings. `download_and_parse_files` returns the metadata and an
  empty power frame instead of raising, the metadata upsert runs, no power rows are appended, and
  the ledger records the files. Without the ledger entry the same files would be downloaded every
  hour. The `power_time_series` table gets no commit, so `clean_nged_power_data` does not rebuild.
- **A late file.** A file with an old `end_time` and a new path or `last_modified` is selected, and
  its readings land. Its metadata is dropped in step 6 unless it is the series' newest file.
- **A back-fill of old windows.** The same as a late file, in bulk. The run may download thousands
  of files, and the readings the table lacks are appended.
- **A file that becomes visible late.** An earlier listing did not contain the file, so the ledger
  does not either, and the next listing selects it. Nothing is lost.
- **A run in which no series' newest file was selected.** A back-fill-only run is one. The filter in
  step 6 leaves no metadata, so the upsert is skipped.

#### Error handling

- **NGED's bucket fails to list or download.** The existing retry guard raises `RetryRequested`.
  Nothing was appended or recorded, so the retry repeats the same selection.
- **The ledger cannot be read.** `read_downloaded_files` raises, inside the same retry guard. A
  missing file reads as an empty ledger, but a corrupt or unreadable file must not, because
  that would download the whole bucket because of a fault in our own store.
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
- **The ledger write fails.** The write raises and the run fails. The next run downloads the same
  files again, which the row dedupe absorbs. The failure is not swallowed, because a ledger that
  never advances would make the backlog grow every hour.

## Verdict, size and departures

**Verdict: worth implementing, with a ledger in place of the issue's watermark and four
departures.**

- Departure 1: the plan records downloaded files in a ledger, not a watermark (the issue's two
  homes both store one). A watermark filters on a time, so it needs a margin for files that appear
  in a listing late, and the margin re-selects a band of files every hour. It also needs empty
  commits to advance when a batch is all data-less, and those commits make the cleaning asset
  rebuild. The ledger has no margin, no band, no empty commits, and no lost-file hazard: a file
  that is absent from a listing is simply not recorded. Its cost is state that grows with the
  bucket instead of one timestamp. A synthetic ledger of 3.65 million rows (2,500 series, 4 files
  a day, one year) read and anti-joined with a same-size listing in 0.15 seconds, and holds about
  350 MB in memory, the same as the listing frame the ingest already builds each hour.
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
  updates. The rule lives in the asset, so `download_and_parse_files` keeps its signature and its
  `end_time` order.
- Departure 4: the equivalence test compares the new ingest with a "download every file, every
  run" reference, not with a copy of the old ingest. See Tests.

**Size: complex.** One line per trigger:

- What gets stored: yes. The ledger is a new stored file.
- Production serving path: no. The ingest runs upstream of `live_forecasts`, and the plan adds no
  commit to the `power_time_series` table, so `clean_nged_power_data` rebuilds exactly when it does
  today.
- A degradation rule: yes. That the ledger is written when the metadata upsert fails, that a
  missing ledger means "download everything", and that a malformed file stalls the ingest, are
  degradation decisions.
- More than one defensible design: yes. A ledger, a watermark in the Delta commit, a watermark
  in a state file, and a shorter lookback each satisfy the issue.
- Code whose callers could not be named without searching: no. `remove_small_files_from_listing`,
  `add_newest_file_of_each_series`, `download_and_parse_files` and `select_new_rows`'s file-listing
  branch each have one production caller, the asset.

**Reviews bought: all four** (two plan reviews, then the two diff reviews in `implement-issue`).
The plan has had two simplicity reviews and two correctness reviews so far.

## What changes, file by file

### `packages/nged_data/src/nged_data/storage.py`

- `_RawFileListItem` and `_ProcessedFileListing` gain `last_modified`, a UTC datetime read from
  `object_meta["last_modified"]` in `list_timeseries_json_files` (obstore returns a UTC datetime).
- A new private Patito model `_DownloadedFiles` holds `path` and `last_modified`, beside
  `_ProcessedFileListing`.
- New `read_downloaded_files(path, storage_options)` returns the ledger as
  `pt.DataFrame[_DownloadedFiles]`. A missing file returns an empty, typed frame. An unreadable or
  invalid file raises.
- New `write_downloaded_files(path, downloaded, storage_options)` writes the old ledger plus
  `downloaded`, de-duplicated on `path` keeping the later `last_modified`. Its I/O mirrors
  `upsert_metadata`'s handling of local paths and object-store URIs. A local write goes through a
  temporary file and a rename, so a crash cannot leave a torn ledger.
- New `select_files_not_yet_downloaded(file_listing, ledger)` anti-joins the listing with the
  ledger on `(path, last_modified)`.
- `download_and_parse_files` no longer raises `NoNewData` when every file was data-less: it returns
  the metadata and an empty, validated `PowerTimeSeries` frame. `NoNewData` stays for an empty
  selection only. The change is required, because a silent series' newest file is data-less, is now
  downloaded once, and its metadata must reach the upsert in that run.
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
  beside `metadata_path`.

### `src/nged_substation_forecast/defs/assets.py` (`power_time_series_and_metadata`)

- Read the ledger inside the retry guard. List, select with `select_files_not_yet_downloaded`,
  compute each series' newest file from the whole listing, download, filter the metadata to the
  series whose newest file was selected, upsert the metadata when any remains (still swallowed on
  failure), dedupe the rows with `select_new_rows`, append when any rows remain, then write the
  ledger. An empty selection still raises `NoNewData` and writes nothing.
- Remove the size filter, the listing-level `select_new_rows`, and `add_newest_file_of_each_series`.
  The `nged_s3_paths` table collapses its "Files above the size threshold", "Files with new data"
  and "Files downloaded" rows to "Files not yet downloaded".
- The docstring says the ledger exists, where it lives, and that a missing ledger downloads every
  file in the bucket.

### Docs and comments the change invalidates

- `packages/nged_data/README.md`: remove the entries for the deleted functions, describe the new
  ones.
- `docs/live_service/operations.md`: lines 234-235 ("The ingest downloads the newest file of every
  series, even a small file…") and lines 276-283 (a failed upsert: the ledger has recorded those
  files, and a silent series' note is restored by its next file). Add the first-run procedure and
  how to force a re-download of a file (remove its row from the ledger).
- Grep `docs/` and `src/` for `select_new_rows`, `_LATE_FILE_LOOKBACK`, `3 days`, `newest file`, and
  `re-lists NGED's bucket` and rewrite each remaining hit to describe the present behaviour.
- Test docstrings, stubs and fixtures: `test_storage.py:307`, `tests/test_assets.py:388` (its
  premise is the size filter), and every fixture that builds `_ProcessedFileListing` or
  `_RawFileListItem` without `last_modified` (`tests/test_assets.py:1463-1480`, `:1523`;
  `test_storage.py:29-48`, `:379`, `:531`, `:626-628`, `:670-672`). `_FakeS3Store.list` must return
  `last_modified` or every existing asset test fails. The stubs of `download_and_parse_files` at
  `tests/test_assets.py:368, 430, 474, 520` keep their signature.
- No change is needed to `checks.py`, `cleaning_assets.py`, `schedules.py`, or the
  `delta_store` package, because the plan adds no commit to the `power_time_series` table.

## Design-philosophy check

- **Production code, so degrade, except where the fault is ours.** A missing ledger means "download
  everything", and the row dedupe makes that safe. A read error or corruption in our own ledger
  raises inside the retry guard, because that is our own store misbehaving. A malformed NGED file
  or a contract violation raises, because a contract violation is our own bug in `CLAUDE.md`'s
  terms, and skipping the file would lose its readings silently and for good. This is the one
  place the plan chooses a stall over degrading; the stall loses no data and ends when the
  contract or the parser is fixed.
- **Idempotence replaces atomicity.** The ledger is written after the append, so it is never ahead
  of the rows, and a crash re-downloads files that the dedupe absorbs. The inherent-stability
  principle of idempotent writes is kept.
- No asset check is added or changed. `power_data_is_fresh` still reads `time_series_coverage`.
- Hypotheses: this serves H1 (the pipeline keeps running through an outage) by lowering the request
  load, and trades away no principle in `design-principles.md`.

## Tests

Each test states the assertion that fails on `main` today. Most tests fail on `main` only because
the new function does not exist, so the named mutation is the real test of each.

1. **Equivalence replay** (`tests/test_assets.py`). `_FakeS3Store` gains a `last_modified` per file,
   a `put(path, data, last_modified)` for adding files between runs, and a count of `get` calls.
   Run the real asset over one replay twice, each time from empty storage: once normally, once with
   `read_downloaded_files` monkeypatched to return an empty ledger every run, which downloads the
   whole fake bucket each run. After each run, assert exact frame equality, ignoring row order, of
   the two `power_time_series` tables and the two metadata parquet files. The replay: initial files
   for 3 series; new files; a back-fill of old windows (files with `end_time` two months old and a
   new path, for series that also have a newer live file), which must add only the missing readings
   and leave the metadata unchanged; a late file whose `end_time` is more than 3 days before its
   series' newest reading (the old ingest's known loss); a file that appears in the bucket only
   after a later file was already downloaded; a series that stops reporting and publishes a
   data-less file with a new `Information` note; a rewritten key (same path, new `last_modified`);
   a run with no new files. The two arms share the parsing code, so equality alone cannot catch a
   parsing bug. The test therefore also asserts directly: the stopped series' new `Information`
   note is stored, the late file's rows landed, and the back-fill left the metadata unchanged. The
   only expected difference between the arms is the `get` count, which is smaller for the ledger
   arm and zero on the run with no new files.
2. **A run with nothing new does nothing.** A second run over an unchanged bucket makes zero `get`
   calls, appends nothing, and leaves the ledger file unchanged. The mutation to catch is selecting
   files within a window instead of by the ledger.
3. **A data-less hour is recorded once.** A run whose only new file is data-less upserts the
   metadata, appends no rows, and records the file, so the next run makes zero `get` calls. The
   mutation to catch is not recording a file that yielded no rows.
4. **`select_files_not_yet_downloaded`** excludes a file whose `(path, last_modified)` is in the
   ledger, includes one whose path is in the ledger with a different `last_modified`, includes a
   new path, and returns every file for an empty ledger.
5. **Ledger survives a metadata fault.** Monkeypatch `upsert_metadata` to raise. The power rows
   land, the run succeeds, and the ledger records the files.
6. **A malformed file fails the run and records nothing.** One file with invalid JSON among valid
   ones: the run raises (as `RetryRequested`, then failure), nothing is appended, and the ledger is
   unchanged. The mutation to catch is a `try`/`except` that skips the file. A second case: a
   failing `get` also raises and leaves the ledger unchanged.
7. **A crash between the append and the ledger write loses nothing.** Monkeypatch
   `write_downloaded_files` to raise on one run. The rows are on disk. The next run downloads the
   same files again and appends no duplicate rows.
8. **The ledger's storage.** `read_downloaded_files` returns an empty typed frame for a missing
   file and raises for a corrupt one. A write followed by a read returns the same rows, on a local
   path and on a moto S3 bucket (reset per test, per `docs/architecture/testing.md`).
   `write_downloaded_files` de-duplicates on `path` keeping the later `last_modified`.
9. **`download_and_parse_files`** returns metadata and an empty power frame when every file is
   data-less.

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

- **First run on the existing deployment.** With no ledger file, the first run downloads the whole
  bucket: about 22,000 sequential GETs for 33 series over about 165 days. One GET has not been
  measured here; at 50 to 100 ms the run takes 20 to 40 minutes and could outlast the next hourly
  tick. A second run that started before the first finished would also find no ledger, download
  everything, and append duplicate rows, which would stop `clean_nged_power_data` on its
  uniqueness check. Recommend a documented procedure: pause the hourly schedule, run the ingest
  once by hand, check that the ledger exists, then resume. No seeding code is needed. Does the
  maintainer prefer a seeding script that writes a ledger from the current listing, which avoids
  the long run?
- **No run-concurrency limit exists on the ingest.** The hazard of two overlapping runs appending
  the same rows exists today and the plan neither adds nor removes it. A `pool` of 1 for the asset
  would remove it. Recommend a separate issue, together with chunking or parallelising the first
  run of a fresh Flexpectation v2 table (the `get_async` TODO in `download_and_parse_files`).
- **A stalled ingest on a contract violation.** At Flexpectation v2 scale, files from licence areas
  other than `EMids` will fail the `TimeSeriesMetadata` enum, and the ingest will stall on them
  until the contract is widened. Widening a contract needs the maintainer's agreement first, so
  the stall is also a prompt to do that work before v2 ingestion begins. Does the maintainer
  prefer that the ingest skips only files that are not valid JSON, with a circuit breaker and a
  Sentry event naming the file? Recommend no: `report_asset_degradation` takes no path tag and
  needs a fingerprint, and skipping gives up the guarantee that no readings are lost.
- **Ledger size and rewrite cost.** The ledger grows by one row per downloaded file and is written
  whole each run that downloads a file. At 2,500 series for one year that is 3.65 million rows. The
  implementer may key the ledger on integers (`time_series_id`, `start_time`, `end_time`,
  `last_modified`) instead of path strings to cut memory to about a third. Neither is a problem at
  33 series.
- **The ledger's home.** The `_DownloadedFiles` model is private to `storage.py`, like
  `_ProcessedFileListing`, so no contract in `packages/contracts/` changes. It is nevertheless
  persisted. Does the maintainer want it in `contracts` instead? Recommend `storage.py`.
- **Stored metadata no longer repairs itself every hour.** Since PR #1112, every run re-reads each
  series' newest file. After this change, a series that publishes no further files keeps whatever
  stored metadata it has. The cleaning joins metadata with a left join, so the impact is limited.
  Recovery after a metadata-table loss is a run with no ledger, which re-reads every file.
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
`download_and_parse_files`'s signature and `end_time` order; move the `NGED_INGEST` pool to its own
issue; replace the seeding step with a documented first-run procedure; the plan's saving under the
watermark design was only about the same as the one-line interim option when NGED writes batches.
Rejected: nothing. Noted and left out of scope: merging the ingest and cleaning assets, listing
only recent windows, S3 event notifications, and a manifest object.
