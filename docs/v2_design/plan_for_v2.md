# V2 Plan: NGED Substation Forecast

## High-Level Architecture

The project is structured as a modular monorepo using `uv` workspaces. The core technologies are:

*   **Orchestration:** Dagster (Software-Defined Assets). No asset shall contain disk I/O operations (pl.read_parquet, df.write_delta, etc.). All data loading and saving must be handled exclusively by Dagster I/O Managers. Assets must take DataFrames as inputs and return DataFrames as outputs.
*   **Data Processing:** Polars (chosen for speed and `join_asof` to prevent data leakage).
*   **Data Contracts:** Patito (used to validate Polars DataFrames against strict schemas).
    - Use `pt.DataFrame.cast()` to cast non-conforming columns to their correct type.
    - Use Patito's `examples` constructor to create testing data.
    - Use Patito's `derive` to define how to derive wind direction, wind speed, and wind chill, and
      then use Patito `drop` to drop the u- and v-components of wind.
*   **Storage:** Delta Lake
*   **Configuration:** Hydra (for managing complex ML model configurations).
*   **Experiment Tracking:** MLflow.

The codebase is divided into several packages under `packages/` (e.g., `contracts`, `ml_core`, `nged_data`, `xgboost_forecaster`) and a main Dagster application under `src/nged_substation_forecast/`.

## Ingesting Power Data from NGED

The ingestion pipeline is orchestrated by Dagster assets in `src/nged_substation_forecast/defs/nged_assets.py`.

*   **Raw Data:** NGED provides power time-series data in JSON format.
*   **Processing:** The `process_nged_json_file` function loads the JSON, extracts metadata, and validates the time-series data against the `PowerTimeSeries` Patito schema.
*   **Storage:** Validated data is appended to a Delta table (`raw_power_time_series`).
*   **Assets:** There are assets for:
    - historical backfills (`nged_json_archive_asset`),
    - live updates (`nged_json_live_asset`) - NGED will provide new JSON files every 6 hours. The
      JSON files will each contain the last 2 weeks of data. So we must be sure we only append new
      data to the Delta table, 
    - and a temporary SharePoint drop `nged_sharepoint_json_asset` (this asset will be removed after testing)

### Manual TODO
- [ ] Manually tweak `process_nged_json_file`.


## Data Cleaning

Data cleaning is handled by `cleaned_power_time_series` in `src/nged_substation_forecast/defs/data_cleaning_assets.py`.

*   **Logic:** It reads from `raw_power_time_series` and applies two main checks:
    1.  **Stuck Sensors:** Rolling standard deviation < 0.01 MW over a 24-hour window.
    2.  **Insane Power:** Values outside the physically plausible range (-20.0 MW to 100.0 MW).
*   **Null Preservation:** Crucially, bad values are replaced with `null` rather than being dropped or imputed. This preserves the strict 30-minute temporal grid required for downstream feature engineering (lags, rolling means).
*   **Partitioning:** The asset uses a 1-day lookback when scanning the Delta table to ensure rolling window calculations have sufficient history at partition boundaries.

## NWP Data

* Maybe have two NWP contracts:
    - NwpFloat32 (this is what we first put the NWP data into when it first arrives from Dynamical.
      And also where we put the Nwp data after loading and rescaling back to physical.) Uses Patito
      `derive` to calculate wind speed and direction and wind chill from u- and v-components.
    - NwpInt12 (what's saved on disk). Has two methods: `from_float32` and `to_float32`.

## Model Training & Evaluation

The ML pipeline is defined in `src/nged_substation_forecast/defs/xgb_assets.py` and relies heavily on `packages/ml_core/src/ml_core/utils.py`.

*   **Configuration:** Dagster passes configuration overrides to Hydra via `XGBoostConfig`. Hydra resolves the full configuration tree.
*   **Data Preparation:** `_prepare_xgboost_inputs` lazily loads the cleaned power data and metadata, filtering for healthy substations.
*   **Training (`train_xgboost`):** Uses `train_and_log_model` from `ml_core`. It trains an `XGBoostForecaster` on the control member of the NWP data.
*   **Evaluation (`evaluate_xgboost`):** Uses `evaluate_and_save_model` from `ml_core`. It generates forecasts, calculates metrics (MAE, RMSE, nMAE) per lead time, and logs them to MLflow.

## The "Universal Model Interface"

The system attempts to decouple the Dagster pipeline from specific ML implementations using a Universal Model Interface (ADR-009, ADR-016).

*   **`train_and_log_model`:** Handles temporal slicing of the training data (including lookback periods) and MLflow logging. It calls `trainer.fit()`.
*   **`evaluate_and_save_model`:** Handles temporal slicing for the test set, calls `forecaster.predict()`, calculates metrics, and logs artifacts to MLflow.

## Identified Issues & "Prototype Cruft"

While the prototype successfully proves the end-to-end flow, several areas need refactoring in V2:

1.  **Leaky Abstractions in `ml_core/utils.py`:**
    *   The `evaluate_and_save_model` function is doing too much. It handles temporal slicing, inference, metric calculation, and MLflow logging.
    *   There are explicit `TODO` comments highlighting confusing logic around `nwp_init_time` and `forecast_time` (e.g., blindly using `datetime.now()`).
    *   Metric calculation is hardcoded inside the evaluation utility rather than being a separate, testable Dagster asset.
2.  **Hardcoded Paths & Temporary Assets:** `nged_sharepoint_json_asset` contains hardcoded paths and is marked for deletion.
3.  **Type Hinting & Validation:** The `_slice_temporal_data` function uses `Any` and handles `dict`, `pl.LazyFrame`, and `pl.DataFrame`. This should be strictly typed.
4.  **Data Contract Enforcement:** While ADR-021 introduces `create_dagster_type_from_patito_model`, its usage and enforcement across all asset boundaries need to be reviewed for consistency.
5.  **Separation of Concerns:** The ML core utilities are tightly coupled with Dagster context objects (`dg.AssetExecutionContext`), making them hard to test in isolation outside of Dagster.

## Conclusion

The V1 codebase successfully establishes the core patterns: Polars for speed, Delta Lake for storage, Patito for contracts, and Dagster for orchestration. The V2 rewrite should focus on cleaning up the ML interfaces, strictly enforcing type boundaries, decoupling metric calculation, and removing prototype-era hardcoded logic.
