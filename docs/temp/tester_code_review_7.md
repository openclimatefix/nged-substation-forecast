---
review_iteration: 7
reviewer: "tester"
total_flaws: 2
critical_flaws: 1
test_status: "tests_failed"
---

# Testability & QA Review

## FLAW-001: Type Mismatch in Plotting Logic (Critical)
* **File & Line Number:** `src/nged_substation_forecast/defs/plotting_assets.py`, line 90
* **The Issue:** `pl.concat([preds_df, actuals_df], how="diagonal")` fails with a `SchemaError` if the `ensemble_member` column types do not match exactly. In `preds_df`, `ensemble_member` is typically `UInt8` (from the model output), but in `actuals_df`, it is created using `pl.lit(0, dtype=pl.UInt8)`. If for any reason the input `predictions` has `ensemble_member` as `Int64` (e.g., during manual testing or if loaded from a format that doesn't preserve `UInt8`), the plot asset crashes.
* **Concrete Failure Mode:** `polars.exceptions.SchemaError: type UInt8 is incompatible with expected type Int64` when attempting to generate the forecast vs actual plot.
* **Required Fix:** Use `how="diagonal_relaxed"` in `pl.concat` to allow automatic type coercion, or explicitly cast `ensemble_member` to `UInt8` in both DataFrames before concatenation.

## FLAW-002: Integration Test Date Calculation Bug
* **File & Line Number:** `tests/test_xgboost_dagster_integration.py`, lines 24-38
* **The Issue:** The dynamic date calculation logic is fragile when local data has a small temporal range. If `max_dt - min_dt` is less than 15 days, the calculated `train_start` (which is `min_dt`) will be chronologically *after* `train_end` (which is `max_dt - 15 days`).
* **Concrete Failure Mode:** The `train_xgboost` asset receives an empty training set and raises `ValueError: No training data remaining after dropping nulls in critical columns.`, causing the integration test to fail on environments with limited local data.
* **Required Fix:** Add a check to ensure `train_start <= train_end`. If not, fallback to a default range or adjust the split (e.g., use the first 20% for training and the last 80% for testing if the range is very small, or just use a fixed small window).

# Audit Summary

### Plotting Robustness
- **Empty Data**: Correctly handled via early returns.
- **No Overlap**: Correctly handled via early returns.
- **Missing Columns**: The asset will crash if `nwp_init_time` or `ensemble_member` are missing from the `predictions` input. While these are required by the schema, the asset could be more defensive.

### XGBoost Pipeline
- **Lazy Evaluation**: The refactoring to a lazy Polars pipeline is successful and maintains performance while reducing memory pressure.
- **Lookahead Bias**: Verified that `_add_lags` correctly calculates dynamic lags based on lead time, preventing any lookahead bias.
- **Feature Engineering**: Cyclical temporal features and weather features are correctly integrated into the lazy pipeline.

### Configuration Validation
- **Pydantic Validation**: `XGBoostConfig` successfully validates ISO date strings and provides clear error messages for invalid formats.
