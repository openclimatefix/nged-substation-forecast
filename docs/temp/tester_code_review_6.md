---
review_iteration: 6
reviewer: "tester"
total_flaws: 3
critical_flaws: 2
test_status: "tests_passed"
---

# Testability & QA Review

## Verification of Loop 5 Fixes
The specific flaws identified in Loop 5 (multi-NWP contract, backtesting leakage, and empty data checks) have been successfully addressed:
- **Multi-NWP support:** `XGBoostForecaster._prepare_and_join_nwps` now correctly handles multiple NWP sources with column prefixing and `join_asof`.
- **Backtesting Leakage:** NWP data is now strictly filtered by `available_time <= valid_time` during training and inference, preventing lookahead bias.
- **Empty Data Checks:** Robust checks for empty DataFrames have been added to `evaluate_and_save_model` and `forecast_vs_actual_plot`.

## Phase 2: Fresh Audit Findings

### FLAW-001: Memory Spike in `forecast_vs_actual_plot`
* **File & Line Number:** `src/nged_substation_forecast/defs/plotting_assets.py`, line 40
* **The Issue:** The `forecast_vs_actual_plot` asset collects the entire `combined_actuals` LazyFrame into memory before performing any substation-specific filtering.
* **Concrete Failure Mode:** In a production environment with ~3000 substations, calling `.collect()` on the downsampled actuals will likely cause an Out-Of-Memory (OOM) error, even if the `predictions` DataFrame only contains a few substations.
* **Required Fix:** Filter `combined_actuals` to only include the `substation_number`s present in the `predictions` DataFrame *before* calling `downsample_power_flows(...).collect()`.

### FLAW-002: OOM Risk in `XGBoostForecaster` Training and Inference
* **File & Line Number:** `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`, line 322
* **The Issue:** `_prepare_data_for_model` eagerly collects the entire joined dataset (flows, NWPs, and metadata) into a `pl.DataFrame`.
* **Concrete Failure Mode:** For large-scale training (e.g., all substations over several years), this eager collection will exhaust available memory. While XGBoost eventually requires the data in memory, collecting it before final feature selection and null-dropping is premature and increases the peak memory footprint.
* **Required Fix:** Keep the data preparation pipeline lazy. `_prepare_features` should be refactored to accept a `LazyFrame` and only `collect()` the final feature matrix `X` immediately before passing it to the XGBoost `DMatrix` or `fit` method.

### FLAW-003: Potential `ValueError` in `XGBoostConfig` Date Parsing
* **File & Line Number:** `src/nged_substation_forecast/defs/xgb_assets.py`, lines 37-43
* **The Issue:** `date.fromisoformat()` is used to parse configuration strings without any error handling or pre-validation.
* **Concrete Failure Mode:** If a user provides an invalid date string (e.g., "2026-13-01") via the Dagster UI or a configuration file, the asset will fail with a `ValueError` during execution rather than failing gracefully during configuration validation.
* **Required Fix:** Add a validator to the `XGBoostConfig` class (using Pydantic's `@validator` or similar) to ensure that date strings are valid ISO formats before the asset starts running.
