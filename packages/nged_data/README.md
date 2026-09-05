# NGED JSON Data

This package reads NGED's telemetry JSON files from S3 and parses them into the `PowerTimeSeries`
and `TimeSeriesMetadata` schemas (see `contracts`). The metadata roster is the only thing this
package owns and writes; the parsed power observations are handed back to the caller, which
appends them to the `power_time_series` Delta table (see [Usage](#usage) below).
`nged_data.storage`'s module docstring, below on this page, says which functions read that Delta
table and which write the roster.

## Public surface

Only `upsert_metadata` is re-exported from the package root (`from nged_data import
upsert_metadata`); the other five live in `nged_data.storage` (`from nged_data.storage import
list_timeseries_json_files`, etc.).

- `nged_data.storage.list_timeseries_json_files(store)` — lists the timeseries JSON files on
  NGED's S3 bucket, parsing `time_series_id`, `start_time`, and `end_time` out of each file's path.
- `nged_data.storage.remove_small_files_from_listing(file_listing, size_threshold_bytes=520)` —
  drops files too small to carry any readings, so `download_and_parse_files` never fetches and
  parses one only to discard the result.
- `nged_data.storage.download_and_parse_files(store, paths_df)` — downloads and parses each
  listed file, returning a `DownloadAndParseResult` of `metadata` (`TimeSeriesMetadata`),
  `power_time_series` (`PowerTimeSeries`), and `n_implausible_power_rows_dropped`. Raises
  `NoNewData` if the listing was empty, or if every listed file's `data` field was null.
- `nged_data.storage.select_new_rows(time_series, delta_path, storage_options=None)` — filters
  `time_series` down to rows newer than what the `power_time_series` Delta table at `delta_path`
  already holds, per `time_series_id`. Accepts either `PowerTimeSeries` rows, compared on `time`,
  or the file listing `list_timeseries_json_files` returns, compared on `end_time`.
- `nged_data.storage.time_series_coverage(delta_path, storage_options=None)` — the earliest and
  latest observation `time` on disk for each `time_series_id` in the `power_time_series` Delta
  table.
- `nged_data.upsert_metadata(new_metadata, metadata_path, storage_options=None)` — merges a
  `TimeSeriesMetadata` snapshot into the stored metadata Parquet file, keeping the newest values
  per `time_series_id` and rewriting the file only if the incoming metadata differs from what is
  stored.

`nged_data.read_nged_json` parses one downloaded JSON file into the two schemas. Every function in
it is private and called only by `download_and_parse_files`, so the module appears on the API page
below carrying just its `ExtractedPowerTimeSeries` result type.

## Data quality

`download_and_parse_files` drops rows whose `time` is malformed — outside the plausible
datetime range, null, or not aligned to the top or bottom of the hour — via
`PowerTimeSeries.drop_implausible_rows`, and reports how many as
`n_implausible_power_rows_dropped`. Degrading rather than raising on a malformed reading follows
[inherent
stability](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/):
a malformed `time` is NGED's feed misbehaving, not a bug in our own pipeline, so ingestion keeps
the rest of the batch rather than aborting it. No other cleaning happens during ingestion.

## Usage

This package is used by the `power_time_series_and_metadata` Dagster asset in
`src/nged_substation_forecast/defs/assets.py`.
