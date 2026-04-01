---
review_iteration: 7
reviewer: "review"
total_flaws: 3
critical_flaws: 0
---

# Code Review

## FLAW-001: Missing Field Documentation in Dagster Configs
* **File & Line Number:** `src/nged_substation_forecast/defs/xgb_assets.py`, lines 16-23; `src/nged_substation_forecast/defs/weather_assets.py`, lines 52-55
* **The Issue:** The `XGBoostConfig` and `ProcessedNWPConfig` classes have docstrings, but their individual fields (e.g., `train_start`, `substation_ids`) lack descriptions.
* **Concrete Failure Mode:** Users interacting with these assets in the Dagster UI will not see helpful tooltips or descriptions for the configuration parameters, making the system harder to use for non-developers.
* **Required Fix:** Add `dg.Field` or Pydantic `Field` with descriptions to each configuration parameter.

## FLAW-002: Potential OOM Risk in `processed_nwp_data` Asset
* **File & Line Number:** `packages/xgboost_forecaster/src/xgboost_forecaster/data.py`, line 170
* **The Issue:** The `process_nwp_data` function eagerly collects the entire NWP LazyFrame into memory for upsampling and interpolation.
* **Concrete Failure Mode:** As the historical NWP dataset grows (e.g., scanning `*.parquet` in `all_nwp_data`), this asset will eventually exceed available RAM and crash the Dagster process. While there is a warning for > 5M rows, it doesn't prevent the OOM.
* **Required Fix:** Partition the `processed_nwp_data` asset (e.g., by `init_time` or day) to match the `ecmwf_ens_forecast` partitions, ensuring only a manageable chunk of data is processed at once.

## FLAW-003: Silent Dropping of Nulls in Non-Critical Columns
* **File & Line Number:** `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`, lines 419, 515
* **The Issue:** The `train` and `predict` methods drop nulls only for "critical" columns (target and temperature).
* **Concrete Failure Mode:** If other features (e.g., wind speed, cyclical features) have high null rates due to data quality issues, the model will still train/predict using XGBoost's default null handling, but the user will have no visibility into how much data was effectively "corrupted" or ignored.
* **Required Fix:** Add logging to report the percentage of rows containing nulls across all features before and after dropping critical nulls, or use a more comprehensive validation step.

# General Observations
- **Code Readability:** The lazy data preparation logic in `XGBoostForecaster` is excellent and follows Polars best practices.
- **Maintainability:** The layered plotting logic in `plotting_assets.py` is clean, modular, and easy to extend.
- **Style Guidelines:** The code strictly adheres to Python 3.14+ standards, uses expressive type hints, and follows Google-style docstrings.
- **Redundancy:** Verified that `tests/manual_xgboost_integration.py` has been successfully removed.
- **Naming Consistency:** Asset and configuration names are consistent across the Dagster definitions.
