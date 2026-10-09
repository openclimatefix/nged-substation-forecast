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

### Walk through

**Each hour, the ingest asks the bucket what changed since the last commit, downloads only that, and
records how far it got in the same commit that stores the readings.** The sub-sections below follow
one run, then its branches, then its failures.

#### The happy path

1. **The run takes the ingest's concurrency slot.** `power_time_series_and_metadata` is declared
   with a new `pool="NGED_INGEST"` limited to 1 run. Two overlapping runs would both compare their
   rows with the same snapshot of the table and both append them, and the duplicate rows would make
   `clean_nged_power_data` fail its `(time_series_id, time)` uniqueness check every hour after.
2. **The run reads the watermark.** `delta_store.power_time_series.read_ingest_watermark` returns
   the watermark stamped on the newest commit of the `power_time_series` table. This comes before
   the listing because the listing is filtered by the watermark.
3. **The run lists the bucket.** `list_timeseries_json_files` now also returns each object's
   `last_modified`. The filter needs `LastModified` because a file's `end_time` says which window
   the file covers, not when NGED wrote it, so `end_time` cannot see a late or back-filled file.
4. **The run notes each series' newest file.** `newest_file_of_each_series` runs on the whole
   listing, before any selection. The run needs the whole listing because a series' newest file
   may have been downloaded in an earlier hour and be absent from the selection.
5. **The run selects files.** `select_files_modified_since` keeps files with
   `last_modified > watermark - margin`. The margin exists because a file can appear in a listing
   some time after its `LastModified`: a file created after the listing passed its key can be
   missed while a later file sets the watermark, and the margin lets the next run pick the missed
   file up. The margin also means files within one margin below the watermark are selected again
   every hour (the band) until newer files move the watermark.
6. **The run downloads and parses the selected files.** `download_and_parse_files` takes the
   selection and the newest-file paths as arguments. The function reads files in
   `(last_modified, end_time, path)` order and takes a series' metadata only from that series'
   newest file. Metadata comes from the newest file only because `upsert_metadata` replaces a
   series wholesale, and a late or back-filled file would otherwise overwrite the current
   `Information` note with an old one.
7. **The run upserts the `TimeSeriesMetadata` table.** This happens before the power write, as it
   does today, and a failure is swallowed so that a fault in a derived table cannot stop the power
   readings landing.
8. **The run drops readings already on disk.** `select_new_rows` anti-joins the parsed rows with the
   stored keys, because the band and a back-filled file both re-deliver readings the table has.
9. **The run commits.** `write_power_time_series(..., custom_metadata=...)` appends the new rows
   and stamps `max(stored watermark, largest last_modified selected)` on the same Delta commit.
   One commit holds both so a crash cannot leave the watermark ahead of the rows. The `max` is
   there because NGED could delete the file that set the watermark, and the watermark must never
   move backwards.

#### Branches off the happy path

- **First run on an empty table.** `read_ingest_watermark` returns `None`, so
  `select_files_modified_since` selects every file. The first commit creates the table and carries
  the watermark.
- **First run on the existing deployment.** The table exists but no commit carries a watermark, so
  the run would download the whole bucket in one run. A one-off seeding commit before the first
  deployed run avoids this: a zero-row append stamped with the date 3 days before the deploy. The
  seed copies today's 3-day window.
- **An hour with new files.** Steps 1 to 9 as above.
- **An hour with nothing new.** The band files are selected and downloaded, their rows are all on
  disk, and the largest `last_modified` equals the stored watermark. Nothing commits, so the table
  gains no version, and `clean_nged_power_data` sees an unchanged table and skips its rebuild.
- **An hour in which every new file is data-less.** A series that has stopped reporting publishes
  files with a fault note and no readings. The run commits zero rows with a later watermark.
  Without that commit the watermark would stay put, and the same files would be downloaded every
  hour. The commit does bump the table's version, which makes the next `clean_nged_power_data`
  rebuild the cleaned table (see Risks).
- **A late file.** A file with an old `end_time` and a new `last_modified` is selected, and its
  readings land. Its metadata is ignored unless it is the series' newest file.
- **A back-fill of old windows.** The same as a late file, in bulk. The run may download thousands
  of files, and the readings the table lacks are appended.
- **A run in which no series' newest file was selected.** A back-fill-only run is one. No metadata
  is extracted, so the metadata upsert is skipped.

#### Error handling

- **NGED's bucket fails to list or download.** The existing retry guard raises `RetryRequested`.
  Nothing was committed, so the watermark has not moved and the retry repeats the same selection.
- **The Delta log cannot be read.** `read_ingest_watermark` raises, inside the same retry guard.
  A read error must not read as "no watermark", because that would download the whole bucket
  because of a fault in our own store.
- **The metadata upsert fails.** The failure is swallowed and reported as today, and the watermark
  still advances. If the watermark did not advance, a persistent upsert fault would make every hour
  download a growing backlog. A silent series' note is restored by its next file, about 6 hours
  later.
- **A file is malformed or breaks the `TimeSeriesMetadata` contract.** The run raises and commits
  nothing. The watermark stays put, so every later hour fails on the same file until someone fixes
  it. The failure is loud because a contract violation is our own bug (for example the
  `licence_area` enum holding only `EMids` when a new licence area appears), and skipping the file
  would lose its readings for good. After the fix, the whole backlog ingests.
- **The process dies before the commit.** Nothing was committed, so the next run downloads the
  same files and the row dedupe absorbs any repeat.
- **Two runs overlap.** The pool prevents this. If two runs overlapped anyway, the `max` rule means
  the later commit's watermark cannot move backwards past an earlier commit's.

## Verdict, size and departures

**Verdict: worth implementing, with one of the issue's two homes chosen and five departures.**

- Departure 1: the plan uses the Delta commit property, the issue's first option, and answers the
  issue's worry that "an hour that appends no rows writes no commit". delta-rs 1.6.6 writes a
  commit for an append of zero rows, including the first write to a table, and keeps its
  `custom_metadata` (verified: an empty append with `custom_metadata={"wm": "2"}` after one with
  `"1"` gives history `['2', '1']`).
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

**Size: complex.** One line per trigger:

- What gets stored: yes. The watermark is new stored state, and empty commits now appear in
  the `power_time_series` Delta table's history.
- Production serving path: no code on the serving path changes. The ingest runs upstream of
  `live_forecasts`. Empty commits bump the raw table's Delta version, which `clean_nged_power_data`
  uses to decide whether to rebuild the cleaned table that `live_forecasts` reads. That is an
  indirect effect, covered in Risks.
- A degradation rule: yes. Where the watermark may advance when the metadata upsert fails, what a
  missing watermark means, and what a malformed file does are degradation decisions.
- More than one defensible design: yes. The commit property, a state file, and a shorter lookback
  each satisfy the issue.
- Code whose callers could not be named without searching: no. `remove_small_files_from_listing`,
  `add_newest_file_of_each_series`, `download_and_parse_files` and `select_new_rows`'s file-listing
  branch each have one production caller, the asset.

**Reviews bought: all four** (two plan reviews, then the two diff reviews in `implement-issue`).
The plan has had one simplicity review and two correctness reviews so far.

## What changes, file by file

### `packages/delta_store/src/delta_store/power_time_series.py`

- `write_power_time_series` gains a keyword argument `custom_metadata: dict[str, str] | None` and
  passes it to `write_deltalake` as `CommitProperties(custom_metadata=...)`, the mechanism
  `cleaned_power_time_series.py` already uses. It accepts a frame with zero rows.
- A new `read_ingest_watermark(table_uri, storage_options)` modelled on `read_cleaning_provenance`:
  it walks a small history window to the newest commit carrying the watermark key and returns a
  UTC datetime. It returns `None` only for a missing table or no commit carrying the key within the
  window, which mean "first run". A read error raises, inside the asset's S3 retry guard, so a
  transient fault reading our own store cannot trigger a whole-bucket download. A new module
  constant names the commit key.
- `packages/delta_store/README.md` (line 40) gains the new keyword argument and the reader.

### `packages/nged_data/src/nged_data/storage.py`

- `_RawFileListItem` and `_ProcessedFileListing` gain `last_modified`, a UTC datetime read from
  `object_meta["last_modified"]` in `list_timeseries_json_files` (obstore returns a UTC datetime).
- New `select_files_modified_since(file_listing, watermark)` returns files with
  `last_modified > watermark - _LAST_MODIFIED_MARGIN`, or every file when `watermark` is `None`.
  `_LAST_MODIFIED_MARGIN` replaces `_LATE_FILE_LOOKBACK`. The margin has to exceed the longest gap
  between a file's `LastModified` and the moment the file first appears in a listing, which
  includes the time one listing takes. The value is fixed at implementation time from
  measurements (see Risks).
- New `newest_file_of_each_series(file_listing)` returns the path of each series' newest file by
  `(end_time, last_modified, path)`, taken from the whole listing before any selection.
  `download_and_parse_files` takes it as an argument and extracts metadata only from those files,
  whatever else it downloads. It sorts by `(last_modified, end_time, path)`.
- `download_and_parse_files` no longer raises `NoNewData` when every file was data-less: it returns
  an empty, validated `PowerTimeSeries` frame, and an empty `TimeSeriesMetadata` frame when no
  selected file was a series' newest. `NoNewData` stays for an empty selection only. The
  data-less change is required, because a silent series' newest file is data-less, is now
  downloaded once, and its metadata must reach the upsert in that run. The `pl.concat` of the
  metadata frames must tolerate the empty case.
- Deleted: `remove_small_files_from_listing`, `add_newest_file_of_each_series`,
  `_LATE_FILE_LOOKBACK`, and `select_new_rows`'s `_ProcessedFileListing` overload and branch. The
  `PowerTimeSeries` branch stays, because it is the row dedupe that makes a re-download safe.
- Docstrings that describe the removed behaviour are rewritten: the module docstring, the
  `time_series_coverage` docstring (the whole-table scan no longer runs twice an hour), the
  `TimeSeriesCoverage` docstring (lines 318-321, which says `select_new_rows` reads `last_time`),
  `UpsertMetadataStats.metadata_upsert_failed` (the stale metadata is no longer retried next
  hour), and the `upsert_metadata` docstring (it says metadata comes only from files
  `select_new_rows` judged new, and gives rebuild advice that must be restated).

### `src/nged_substation_forecast/defs/assets.py` (`power_time_series_and_metadata`)

- Declare `pool="NGED_INGEST"`, with a comment modelled on the `ECMWF` pool's, and document the
  limit of 1 where the `ECMWF` pool's limit is documented (`dagster.yaml` or
  `dagster instance concurrency set NGED_INGEST 1`).
- Read the watermark with `read_ingest_watermark`, inside the retry guard. List, select with
  `select_files_modified_since`, download, upsert the metadata when any was extracted (still
  swallowed on failure), dedupe the rows with `select_new_rows`.
- Commit with `write_power_time_series` and `custom_metadata` set to
  `max(stored watermark, largest last_modified among the selected files)`, when the deduped rows
  are not empty or that value is later than the stored watermark. Otherwise commit nothing. An
  empty selection still raises `NoNewData`.
- Remove the size filter, the listing-level `select_new_rows`, and `add_newest_file_of_each_series`.
  The `nged_s3_paths` table collapses its "Files above the size threshold", "Files with new data"
  and "Files downloaded" rows to "Files modified since the watermark".
- The docstring says the watermark exists, where it lives, and that a table with no watermark
  commit downloads every file in the bucket.

### Docs and comments the change invalidates

- `packages/nged_data/README.md`: remove the entries for the deleted functions, describe the new
  ones.
- `docs/live_service/operations.md`: lines 234-235 ("The ingest downloads the newest file of every
  series, even a small file…"), lines 263-266 (the cleaning-lag wording), and lines 276-283 (a
  failed upsert: the watermark has moved past those files, and a silent series' note is restored by
  its next file). Add the one-off seeding step and the `NGED_INGEST` pool.
- The "commits only when NGED delivers new rows, about every 6 hours" wording, which empty commits
  make untrue: `checks.py` (`_CLEANING_LAG_WARN_COMMITS` and the `cleaned_power_keeps_up_with_raw`
  description string), `cleaning_assets.py` (lines 117-118), `schedules.py` (lines 23-24), and
  `docs/live_service/operations.md` (lines 263-266). `docs/roadmap/data-cleaning.md` does not
  carry that wording and needs no edit.
- Grep `docs/` and `src/` for `select_new_rows`, `_LATE_FILE_LOOKBACK`, `3 days`, `newest file`, and
  `re-lists NGED's bucket` and rewrite each remaining hit to describe the present behaviour.
- Test docstrings, stubs and fixtures: `test_storage.py:307`, `tests/test_assets.py:388` (its
  premise is the size filter), the stubs of `download_and_parse_files(store, paths_df)` at
  `tests/test_assets.py:368, 430, 474, 520` (the signature gains the newest-files argument), and
  every fixture that builds `_ProcessedFileListing` or `_RawFileListItem` without `last_modified`
  (`tests/test_assets.py:1463-1480`, `:1523`; `test_storage.py:29-48`, `:379`, `:531`, `:626-628`,
  `:670-672`). `_FakeS3Store.list` must return `last_modified` or every existing asset test
  fails.

## Design-philosophy check

- **Production code, so degrade, except where the fault is ours.** A missing watermark means
  "download everything", and the row dedupe makes that safe. A read error on our own Delta log
  raises inside the retry guard, because that is our own store misbehaving. A malformed NGED file
  or a contract violation raises, because a contract violation is our own bug in `CLAUDE.md`'s
  terms, and skipping the file would lose its readings silently and for good. This is the one
  place the plan chooses a stall over degrading; the stall loses no data and ends when the
  contract or the parser is fixed.
- **The watermark comes from S3's `LastModified`, never the wall clock**, so clock skew on the
  control-plane VM cannot skip files.
- **Atomicity.** The watermark moves in the same commit as the readings, so no crash can leave it
  ahead of the rows. A crash or a `RetryRequested` retry before the commit re-downloads files that
  the dedupe absorbs. The `NGED_INGEST` pool keeps two runs from appending the same rows.
- No asset check is added or changed beyond the wording above. `power_data_is_fresh` still reads
  `time_series_coverage`, which empty commits do not affect.
- Hypotheses: this serves H1 (the pipeline keeps running through an outage) by lowering the request
  load, and trades away no principle in `design-principles.md`.

## Tests

Each test states the assertion that fails on `main` today. Most tests fail on `main` only because
the new function does not exist, so the named mutation is the real test of each.

1. **Equivalence replay** (`tests/test_assets.py`). `_FakeS3Store` gains a `last_modified` per file,
   a `put(path, data, last_modified)` for adding files between runs, and a count of `get` calls.
   Run the real asset over one replay twice, each time from empty storage: once normally, once with
   `read_ingest_watermark` monkeypatched to return `None` every run, which downloads the whole fake
   bucket each run. After each run, assert exact frame equality, ignoring row order, of the two
   `power_time_series` tables and the two metadata parquet files. The replay spaces its
   `last_modified` values wider than the margin, so that some runs select no series' newest
   file. The replay: initial files for 3 series; new files; a back-fill of old windows (files with
   `end_time` two months old and a new `last_modified`, for series that also have a newer live
   file), which must add only the missing readings and leave the metadata unchanged; a late file
   whose `end_time` is more than 3 days before its series' newest reading and whose `last_modified`
   is new (the old ingest's known loss); a file that appears after a run with a `last_modified`
   earlier than the watermark but inside the margin; a series that stops reporting and publishes a
   data-less file with a new `Information` note; a rewritten key (same path, new `last_modified`);
   a run with no new files. The two arms share the parsing code, so equality alone cannot catch a
   parsing bug. The test therefore also asserts directly: the stopped series' new `Information`
   note is stored, the late file's rows landed, and the back-fill left the metadata unchanged. The
   only expected difference between the arms is the `get` count. On the run with no new files, the
   count equals the number of files inside the margin band, computed from the fake store, and is
   smaller than the reference's.
2. **A run with nothing new commits nothing.** Two runs with no new files leave the table's version
   unchanged after the first. The mutation to catch is committing on every run.
3. **Empty commit advances the watermark.** A run whose only new file is data-less appends zero rows
   and the table's newest commit carries the larger watermark. The mutation to catch is skipping
   the write when there are no rows.
4. **Margin comparison.** `select_files_modified_since` keeps a file one second after
   `watermark - margin`, drops one exactly at it, and returns every file for `None`.
5. **Watermark survives a metadata fault.** Monkeypatch `upsert_metadata` to raise. The power rows
   land, the run succeeds, and the watermark advances.
6. **A malformed file fails the run and holds the watermark.** One file with invalid JSON among
   valid ones: the run raises (as `RetryRequested`, then failure), no commit is written, and the
   stored watermark is unchanged. The mutation to catch is a `try`/`except` that skips the file.
   A second case: a failing `get` also raises and leaves the watermark unchanged.
7. **`read_ingest_watermark`** returns `None` for a missing table and for a table whose commits
   carry no key, returns the newest value after several commits including an unrelated commit, and
   raises for a table whose log holds junk (written into `_delta_log/00000000000000000000.json`).
8. **`download_and_parse_files`** returns metadata and an empty power frame when every file is
   data-less, and returns empty metadata when no listed file is a series' newest. Built from a
   listing with explicit `last_modified` values, it extracts metadata only from the file
   `newest_file_of_each_series` names: given a newer-`end_time` file and a later-written
   older-`end_time` file of one series, the metadata comes from the newer-`end_time` file. It
   orders two files with equal `last_modified` by `end_time` then `path`. A
   `newest_file_of_each_series` test pins the `(end_time, last_modified, path)` tiebreak.
9. **`write_power_time_series`** with `custom_metadata` and zero rows adds a commit whose history
   carries the key, including as the first write to a table.
10. **The pool is declared.** The asset's definition carries `pool="NGED_INGEST"`.

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

- **Margin and band, measured before fixing.** The margin has to exceed the longest gap between a
  file's `LastModified` and the moment the file first shows up in a listing, which is not the
  same as the listing time or the spread of `LastModified` within a window. The implementer
  measures it by listing the real bucket repeatedly and recording when each new object first
  appears. The same listings show how big the re-fetched band is. A file that appears later than
  the margin is lost silently and permanently; the old 3-day window was not designed as a backstop
  for that and the plan builds none. Does the maintainer want a daily backstop run that selects
  from `watermark - 3 days`? Recommend no, unless the measurement shows large visibility lags.
  Does the maintainer accept a band of repeat downloads, up to a whole batch, if NGED writes
  batches?
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
- **A stalled ingest on a contract violation.** At Flexpectation v2 scale, files from licence areas
  other than `EMids` will fail the `TimeSeriesMetadata` enum, and the ingest will stall on them
  until the contract is widened. Widening a contract needs the maintainer's agreement first, so
  the stall is also a prompt to do that work before v2 ingestion begins. Does the maintainer
  prefer instead that the ingest skips only files that are not valid JSON, with a circuit breaker
  and a Sentry event naming the file? Recommend no: it costs a tagged-report mechanism that
  `report_asset_degradation` lacks today (it takes no path tag and needs a fingerprint), and it
  gives up the guarantee that no readings are lost.
- **First run on the existing deployment.** A table with no watermark commit downloads the whole
  bucket in one run: about 22,000 sequential GETs for 33 series over about 165 days, which could
  outlast the next hourly tick. The pool queues the next run behind it, so overlap cannot happen.
  Recommend the one-off seeding commit described in the walk through. Does the maintainer want
  the seeding as a documented command or a small script?
- **A fresh Flexpectation v2 table** would download the whole bucket in one run, all held in
  memory. Out of scope; worth a follow-up issue that chunks the first run.
- **Stored metadata no longer repairs itself every hour.** Since PR #1112, every run re-reads each
  series' newest file. After this change, a series that publishes no further files keeps whatever
  stored metadata it has. The cleaning joins metadata with a left join, so the impact is limited.
  Recovery after a metadata-table loss is a run with no watermark, which re-reads every file.
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
of the table); dropping the hold-back on a metadata fault; the "download everything" test
reference. Rejected: nothing. Kept, as the reviewer judged: the margin, and
`select_files_modified_since` as a separate function.

**Correctness review 1 (Opus).** Accepted: the margin band re-fetch, with no commit on a band-only
run and a measured margin; the cleaning-asset rebuild and the cleaning-lag check wording (as open
questions with a recommendation); `read_ingest_watermark` raising on read errors; the tiebreak; the
first-run seeding; the missed docs, docstrings and fixtures; the extra replay scenarios. Its
malformed-file stall finding led to a skip-the-file design, which review 2 overturned. Corrected:
its reason for Departure 3 was wrong when a late file is the only file of its series in a run, so
the reason is restated for the back-fill case.

**Maintainer question after review 1.** Asked whether a back-fill of old files is handled, the
answer was yes for the rows (a back-fill file has a new `last_modified`), but not for the metadata,
which the old-window file would have overwritten. Departure 3 and the back-fill scenario in the
replay test now cover it.

**Correctness review 2 (Opus).** Accepted: skipping malformed files drops a whole licence area's
readings silently (the `licence_area` enum holds only `EMids`), so the plan now fails loudly and
holds the watermark, and the skip, its Sentry mechanism and its tests are removed; overlapping runs
append duplicate rows and freeze the cleaned table, so the plan adds the `NGED_INGEST` pool
(limit 1) instead of leaving it to the implementer; `without_files=True` is deprecated and ignored
in deltalake 1.6.6 and `pyproject.toml` turns warnings into errors, so it is dropped; a run that
selects no series' newest file crashes `pl.concat`, so the empty-metadata case is specified and
tested; the margin must cover the visibility lag, not the listing time; the committed watermark
is `max(stored, selected)`; the test needs absolute assertions because both arms share parsing
code; the missed docs, stubs and fixtures are listed. Corrected: `docs/roadmap/data-cleaning.md`
does not carry the "every 6 hours" wording, so it needs no edit. Rejected: the circuit-breaker
variant of skipping, because failing loudly is simpler and loses nothing.
