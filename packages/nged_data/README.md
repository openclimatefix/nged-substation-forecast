# NGED JSON Data

This package handles the ingestion and processing of NGED's new JSON data format.

## Key Components

- `load_nged_json`: Loads and parses NGED JSON files.
- `clean_power_time_series`: Cleans power data, including filtering out "stuck" sensors based on daily variance and insane values.
- `append_to_delta`: Appends cleaned data to a Delta table, ensuring no duplicates.
- `upsert_metadata`: Upserts metadata to a Parquet file.

## Usage

This package is used by the `nged_json_live_asset` and `nged_json_archive_asset` in the `nged_substation_forecast` package.
