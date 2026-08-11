# NGED JSON Data

This package handles the ingestion and processing of NGED's new JSON data format.

## Key Components

- `load_nged_json`: Loads and parses NGED JSON files.
- `clean_power_time_series`: Cleans power data, including filtering out "stuck" sensors based on daily variance and insane values.
- `append_to_delta`: Appends cleaned data to a Delta table, ensuring no duplicates.
- `upsert_metadata`: Upserts metadata to a Parquet file.

## Data Quality

NGED provides data every 6 hours via JSON files on S3. The data has several known quality issues that are handled during ingestion:

- **Stuck values**: Detected by computing the standard deviation over a 24-hour rolling window; periods where std is below a threshold are removed.
- **Outliers / invalid values**: Isolated zeros in non-zero time series and values beyond a threshold number of standard deviations from the mean are removed.
- **Early ramp-up period**: The first ~two months of each time series are typically low quality (meter calibration period) and are discarded.

## Usage

This package is used by the `power_time_series_and_metadata` Dagster asset in `src/nged_substation_forecast/defs/assets.py`.
