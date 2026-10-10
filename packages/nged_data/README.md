# NGED JSON Data

This package reads NGED's telemetry JSON files from S3 and parses them into the `PowerTimeSeries`
and `TimeSeriesMetadata` schemas (see `contracts`). The metadata table and the downloaded-files
list are the only files this package owns and writes; the parsed power observations are handed back
to the caller, which appends them to the `power_time_series` Delta table (see [Usage](#usage)
below). `nged_data.storage`'s module docstring, below on this page, says which functions read that
Delta table and which write the metadata table.

## Public surface

Only `upsert_metadata` is re-exported from the package root (`from nged_data import
upsert_metadata`); the other functions live in `nged_data.storage` and `nged_data.cleaning` (`from
nged_data.storage import list_timeseries_json_files`, etc.).

- `nged_data.storage.list_timeseries_json_files(store)` — lists the timeseries JSON files on NGED's
  S3 bucket, parsing `time_series_id`, `start_time`, and `end_time` out of each file's path.
- `nged_data.storage.read_downloaded_files(...)`, `write_downloaded_files(...)` — read and replace
  the downloaded-files list, which records the bucket listing (path and `LastModified`) that the
  ingest last processed in full.
- `nged_data.storage.select_files_not_yet_downloaded(file_listing, downloaded_files)` — keeps the
  listed files that the downloaded-files list lacks, so each file is downloaded once.
- `nged_data.storage.download_and_parse_files(store, paths_df)` — downloads the listed files
  concurrently, in chunks, and parses them in `end_time` order, returning a `DownloadAndParseResult`
  of `metadata` (`TimeSeriesMetadata`), `power_time_series` (`PowerTimeSeries`), and
  `n_implausible_power_rows_dropped`. When every file's `data` field was null, the power frame is
  empty.
- `nged_data.storage.select_new_rows(time_series, delta_path, storage_options=None)` — filters
  `PowerTimeSeries` rows down to those missing from the `power_time_series` Delta table at
  `delta_path`, by an anti-join on `(time_series_id, time)`, so a reading is kept regardless of
  arrival order.
- `nged_data.storage.time_series_coverage(delta_path, storage_options=None)` — the earliest and
  latest observation `time` on disk for each `time_series_id` in the `power_time_series` Delta
  table.
- `nged_data.storage.coverage_from_power(power)` — the same earliest and latest `time` per
  `time_series_id`, from any lazy `PowerTimeSeries` frame.
- `nged_data.storage.scan_cleaned_power(delta_path, storage_options=None)` — scans the
  `cleaned_power_time_series` Delta table, keeping only the rows no cleaning rule flagged. Every
  reader of observed power except the ingest and its freshness check uses `scan_cleaned_power`.
- `nged_data.cleaning.flag_nged_power(power, metadata)` — where cleaning rules go. `flag_nged_power`
  returns every raw power row plus a `drop_reason` column, and the `clean_nged_power_data` Dagster
  asset writes the result to the cleaned table. The function's docstring says how to add a rule.
- `nged_data.upsert_metadata(new_metadata, metadata_path, storage_options=None)` — merges a
  `TimeSeriesMetadata` snapshot into the stored metadata Parquet file, keeping the newest values per
  `time_series_id` and rewriting the file only if the incoming metadata differs from what is stored.

`nged_data.read_nged_json` parses one downloaded JSON file into the two schemas. All three of its
functions are private, and the two that parse a whole file are called only by
`download_and_parse_files`, so the module appears on the API page below carrying just its
`ExtractedPowerTimeSeries` result type.

## Data quality

`download_and_parse_files` drops rows whose `time` is malformed — outside the plausible datetime
range, null, or not aligned to the top or bottom of the hour — via
`PowerTimeSeries.drop_implausible_rows`, and reports how many as `n_implausible_power_rows_dropped`.
Degrading rather than raising on a malformed reading follows [inherent
stability](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/):
a malformed `time` originates upstream of our pipeline, at the meter or in the telemetry export, not
in our own code. Ingestion therefore keeps the rest of the batch rather than aborting it. No other
cleaning happens during ingestion. Cleaning rules run afterwards, over the whole stored table, in
`nged_data.cleaning`.

## Usage

This package is used by the `power_time_series_and_metadata` Dagster asset in
`src/nged_substation_forecast/defs/assets.py`.
