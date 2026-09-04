# NGED JSON Data

This package reads NGED's telemetry JSON files from S3 and parses them into the `PowerTimeSeries`
and `TimeSeriesMetadata` schemas (see `contracts`). It writes only the metadata itself, upserting
it into a Parquet roster; the parsed `PowerTimeSeries` rows are returned to the caller, which
appends them to the `power_time_series` Delta table (see [Usage](#usage) below). The package does
read that Delta table, though: `select_new_rows` and `time_series_coverage` both scan it to work
out which rows are already stored.

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
  `NoNewData` if none of the listed files yielded a single power-observation row.
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
