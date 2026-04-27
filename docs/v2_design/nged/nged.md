# V2 Blueprint: NGED Data Pipeline

This document outlines the V2 architecture for the NGED power data pipeline. It serves as the strict specification for rebuilding the ingestion and cleaning layers.

Instead of a monolithic document, the actual implementation details and pseudocode are broken out into separate Python files in the `docs/v2_design/nged/` directory. This allows us to focus on the "why" here, and the "how" in the code.

## Core Architectural Principles

1. **Pure Functional Assets:** Assets must **never** perform disk I/O (`pl.read_parquet`, `df.write_delta`, etc.). Assets receive Polars DataFrames as arguments and return Polars DataFrames.
2. **Universal Delta Lake Storage:** All data (including NWP weather data, which will be cast to `int16`) is stored in Delta Lake.
3. **Smart I/O Management:** A single `DeltaPolarsIOManager` handles all reads and writes. It uses asset metadata to determine the write strategy (`incremental_append` vs `overwrite`).
4. **Strict Data Contracts:** All asset inputs and outputs are typed using Patito models converted to Dagster types.
5. **Materialized Intermediates:** Cleaned data is explicitly materialized to disk to enable reproducibility, debugging, and efficient ML experimentation.

---

## 1. Storage & IO Management
**See pseudocode:** [`docs/v2_design/nged/io_managers.py`](nged/io_managers.py)

We use a single `DeltaPolarsIOManager` to handle all database interactions. It uses a clean switch statement to route to specific write strategies based on asset metadata:
*   **`incremental_append`:** Uses a "High Water Mark" pattern. It finds the max timestamp per entity and appends only strictly new rows. This is blazing fast and safe against asynchronous reporting delays.
*   **`overwrite`:** Performs a partition overwrite (or full overwrite if unpartitioned). Used for derived assets where we want the database to exactly reflect the function output (dropping orphaned rows).

It also supports `delta_table_name` overrides (allowing multiple assets to write to one table) and `delta_read_lookback_days` (for rolling features).

## 2. Data Ingestion (Archive vs. Live)
**See pseudocode:** [`docs/v2_design/nged/ingestion_assets.py`](nged/ingestion_assets.py)

We face a partitioning mismatch: Archive data is partitioned by entity (substation), while live data is partitioned by time (6-hourly).
To solve this, we use **two separate ingestion assets** that write to the **same physical Delta table** (`raw_power_time_series`).
*   `archive_raw_power_time_series`: Unpartitioned. Runs once on initialization.
*   `live_raw_power_time_series`: 6-hourly partitions. Runs continuously.

Both use the `incremental_append` strategy, so they safely upsert data without duplicating.

## 3. Data Cleaning
**See pseudocode:** [`docs/v2_design/nged/cleaning_assets.py`](nged/cleaning_assets.py)

Cleaning is handled by a single, daily-partitioned asset (`cleaned_power_time_series`).
*   **Backfilling:** To clean the 4-year archive, we simply trigger a Dagster backfill on this asset. Dagster will run the daily partitions concurrently.
*   **Lookback:** Because cleaning requires a 24-hour rolling standard deviation, the asset metadata requests a 1-day lookback. The IOManager automatically fetches the previous day's data, the asset computes the rolling features, filters out the lookback period, and overwrites the current partition.

## 4. ML Consumption
**See pseudocode:** [`docs/v2_design/nged/ml_assets.py`](nged/ml_assets.py)

Because of this architecture, the ML assets become incredibly simple. They are pure functions that take materialized Polars DataFrames as inputs. The ML engineer doesn't need to know about Delta Lake, Parquet, or Dagster partitions.
