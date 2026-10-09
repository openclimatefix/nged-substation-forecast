# Plan: download each NGED file once, by `LastModified` (#1115)

## Problem

**The ingest re-downloads about 3 days of NGED files every hour.** `select_new_rows` offers every
file whose `end_time` is later than its series' newest stored reading minus
`_LATE_FILE_LOOKBACK` (3 days). Nothing records that a file was downloaded, so each hour the ingest
fetches about 12 to 13 files per reporting series: roughly 394 requests per hour for the 33 series
now ingested, and up to about 32,500 for the roughly 2,500 series Flexpectation v2 will ingest. The
row-level dedupe makes the repeats harmless, so the cost is requests and run time.

## Solution

**Filter the bucket listing on each object's `LastModified` against one stored watermark.** The
listing already returns `LastModified`. The ingest downloads files whose `LastModified` is later
than `watermark - margin`, appends the power rows, upserts the metadata, and only then stores the
largest `LastModified` it processed as the new watermark. The watermark lives in a small state file
beside the metadata parquet, never in the Delta table or in `TimeSeriesMetadata`. This deletes the
size filter, the 3-day lookback, and `add_newest_file_of_each_series`.

## Verdict, size and departures

**Verdict: worth implementing, with one mechanism chosen and two departures from the issue body.**

- Departure 1: the plan picks the state file and rejects the Delta commit property. A commit exists
  only in an hour that appends rows. An hour whose only new files are data-less (a silent series
  publishing its fault note) appends nothing, so the watermark would not advance and the ingest
  would re-download those files every hour until the next append.
- Departure 2: the watermark advances only after both the power append and the metadata upsert
  succeeded. The issue does not say this. Once files are downloaded once, a swallowed metadata
  upsert fault would lose a silent series' `Information` note for good, because
  `add_newest_file_of_each_series` no longer re-supplies it every hour. Holding the watermark back
  on a metadata fault costs only repeated downloads while the fault lasts, and the power rows still
  land.

**Size: complex.** One line per trigger:

- What gets stored: yes. The watermark is new stored state.
- Production serving path: no. The ingest runs upstream of `live_forecasts`, and no code on the
  serving path changes.
- A degradation rule: yes. The swallowed metadata fault and the power write's ordering decide where
  the watermark may advance, and an unreadable watermark must degrade rather than raise.
- More than one defensible design: yes. The state file, the commit property, and the interim
  shorter lookback each satisfy the issue.
- Code whose callers could not be named without searching: no. `select_new_rows`'s file-listing
  branch, `remove_small_files_from_listing`, `add_newest_file_of_each_series` and
  `download_and_parse_files` each have one production caller, the asset.

**Reviews bought: all four** (two plan reviews, then the two diff reviews in `implement-issue`).

## What changes, file by file

### `packages/nged_data/src/nged_data/storage.py`

- `_RawFileListItem` and `_ProcessedFileListing` gain `last_modified`, a UTC datetime read from
  `object_meta["last_modified"]` in `list_timeseries_json_files`.
- New `select_files_modified_since(file_listing, watermark, margin)` returns files with
  `last_modified > watermark - margin`, or every file when `watermark` is `None`. A new
  module constant `_LAST_MODIFIED_MARGIN` (recommended 1 hour; see open questions) replaces
  `_LATE_FILE_LOOKBACK`. The margin covers a file that becomes visible with a `LastModified`
  slightly earlier than the newest file already processed.
- New `IngestWatermark` helpers `read_ingest_watermark(path, storage_options)` and
  `write_ingest_watermark(path, watermark, storage_options)`. The I/O mirrors `upsert_metadata`'s
  handling of local paths and object-store URIs. `read_ingest_watermark` returns `None` for an
  absent file and, for an unreadable or malformed file, also returns `None` after logging, so the
  asset can report the degradation. It never raises.
- `download_and_parse_files` stops raising `NoNewData` when every file was data-less. It returns
  the metadata and an empty, validated `PowerTimeSeries` frame. `NoNewData` stays for an empty
  listing only. The change is required: a silent series' newest file is data-less, and it is now
  downloaded once, so its metadata must reach the upsert in that same run.
- Deleted: `remove_small_files_from_listing`, `add_newest_file_of_each_series`,
  `_LATE_FILE_LOOKBACK`, and `select_new_rows`'s `_ProcessedFileListing` overload and branch. The
  `PowerTimeSeries` branch of `select_new_rows` stays, because it is the row dedupe that makes a
  re-download or a crash safe. The module docstring and the `time_series_coverage` docstring (which
  says the whole-table scan runs twice an hour) are updated to match.

### `src/nged_substation_forecast/defs/assets.py` (`power_time_series_and_metadata`)

- Read the watermark. If the file is unreadable, call `report_asset_degradation` and carry on as
  though it were absent.
- List, select with `select_files_modified_since`, download, upsert the metadata, append the new
  rows. Remove the size filter, `select_new_rows` on the listing, and
  `add_newest_file_of_each_series`.
- After both the append and the metadata upsert succeed, write the new watermark: the largest
  `last_modified` among the files selected this run. A failed upsert leaves the old watermark and
  adds `watermark_held_back: True` to the run's output metadata. A watermark write failure is
  swallowed and reported like the upsert failure, because the cost is only repeated downloads.
- The `NoNewData` branch (empty selection) writes nothing and leaves the watermark unchanged.
- The "Files with new data" and "Files downloaded" rows in the `nged_s3_paths` table collapse to
  "Files modified since the watermark".
- The asset docstring says the watermark exists, where it lives, and that the first run, or a run
  after the state file is deleted, downloads every file in the bucket.

### `packages/contracts/src/contracts/settings.py`

- New `ingest_watermark_path`, defaulting to `uri_join(nged_data_path, "ingest_watermark.json")`,
  beside `metadata_path`.

### Docs

- `packages/nged_data/README.md`: remove the entries for the deleted functions and describe the
  watermark helpers.
- `docs/live_service/operations.md`: the "Reading a failed metadata table upsert" paragraph says
  the next run will not offer the files again. It now says the watermark is held back, so the next
  run re-downloads the files and retries the upsert. Add how to force a full re-ingest (delete
  `ingest_watermark.json`) and what a stuck `watermark_held_back` means.
- Search `docs/` for `select_new_rows`, `_LATE_FILE_LOOKBACK`, `3 days` and `re-lists NGED's
  bucket` and rewrite each hit to describe the present behaviour.

## Design-philosophy check

- **Production code, so degrade.** An absent watermark means "download everything", which the row
  dedupe makes safe. An unreadable watermark is treated as absent and reported to Sentry tagged
  `degraded_asset:power_time_series_and_metadata` and naming the path. Neither case raises.
- **The watermark is derived from S3's `LastModified`, never from the wall clock**, so clock skew on
  the control-plane VM cannot skip files.
- **Fault ordering.** The watermark moves last, so a crash at any earlier step re-downloads files
  that the dedupe absorbs. No step can advance the watermark past a file whose readings or metadata
  did not land.
- No asset check is added or changed. The existing `power_data_is_fresh` check still reads
  `time_series_coverage`, which is unchanged.
- Hypotheses: this serves H1 (the pipeline keeps running through an outage) by lowering the request
  load, and trades away nothing in `design-principles.md`.

## Tests

The issue asks for a comparison of the old and new ingest. The old ingest is deleted from production
code, so the test keeps a short **oracle**: a test-local copy of the old selection rule (size filter,
3-day lookback against `time_series_coverage`, newest file of each series). The oracle and the new
ingest run from empty storage against one frozen fake bucket, in `tests/test_assets.py` beside the
existing `_FakeS3Store`, which gains a `last_modified` per file, a `put(path, data, last_modified)`
method for adding files between runs, and a count of `get` calls.

1. **Equivalence replay.** Run the oracle ingest and the new ingest through a sequence of runs.
   After each run, assert exact frame equality, ignoring row order, of the two `power_time_series`
   Delta tables and of the two metadata parquet files, except for the named expected differences.
   The sequence: initial files for 3 series; new files appear; a late file (`end_time` older than
   the series' newest reading, `last_modified` new); a series that stops reporting and publishes a
   data-less file with a new `Information` note; a run with no new files. Expected differences,
   each asserted by name: the silent series' `information` is current under the new ingest, and the
   new ingest's `get` count is smaller. On `main` today the new ingest does not exist, so the
   equivalence test fails to import.
2. **Each file is fetched once.** After a full run, a second run with no new files makes zero `get`
   calls. On `main` it makes one per file inside the 3-day window.
3. **Late file is caught.** A file with an old `end_time` and a new `last_modified` is downloaded
   and its rows land. A file with an old `last_modified` and a late `end_time` is not downloaded.
4. **Margin boundary.** A file whose `last_modified` is exactly `watermark - margin` is not
   downloaded, and one a second later is. This pins the comparison operator.
5. **Watermark is held back on a metadata fault.** Monkeypatch `upsert_metadata` to raise. The power
   rows land, the run succeeds, the watermark file is unchanged, and the next run (fault removed)
   re-downloads the files and the metadata lands. The mutation to catch is advancing the watermark
   unconditionally.
6. **A data-less-only hour keeps metadata current.** When the only new file is data-less, the
   metadata upsert still runs and the watermark advances. The mutation to catch is the old
   `NoNewData` early return.
7. **Unreadable watermark degrades.** A corrupt state file is treated as absent, every file is
   downloaded, `report_asset_degradation` is called, and the run does not raise.
8. **Rewritten key.** A file re-uploaded under the same key with a new `last_modified` is
   downloaded again and adds no duplicate rows.
9. **Unit tests** for `read_ingest_watermark` / `write_ingest_watermark` round-trip on a local path
   and on a moto S3 bucket (reset per test, per `docs/architecture/testing.md`), and for
   `download_and_parse_files` returning an empty power frame plus metadata when every file is
   data-less. The deleted functions' tests are deleted with them.

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
  JSON files are small, so a file should become visible within seconds. A longer margin re-downloads
  every file written inside it on every run. At 2,500 series each publishing about 4 files a day, a
  1-hour margin re-fetches about 400 files an hour, against about 32,500 today. Does the
  maintainer want a longer margin for safety?
- **First run on the existing deployment.** There is no state file, so the first run downloads every
  file in the bucket (about 33 series since April), once. The row dedupe absorbs it. Recommend
  accepting that cost over seeding the watermark by hand. Alternative: seed `ingest_watermark.json`
  with a date before deploying.
- **A rewritten key with corrected values** is downloaded again, but `select_new_rows` drops rows
  that already exist, so corrected values are never applied. This is the behaviour today and is out
  of scope here. Whether NGED ever rewrites a key is not known.
- **Listing cost.** The ingest still lists the whole `timeseries` prefix every hour. At 2,500
  series that listing, not the downloads, becomes the dominant request count. A `start_after` listing
  is possible only if key order tracks time, and it does not (keys start with the window times, but
  late files break the order). Flagged as a possible follow-up issue, not planned here.
- **Interim option** (shorten the lookback to 12 to 24 hours) is rejected: it is superseded by the
  watermark and risks permanently skipping a file that arrives later than the margin.

## Review record

(Filled in after each review.)
