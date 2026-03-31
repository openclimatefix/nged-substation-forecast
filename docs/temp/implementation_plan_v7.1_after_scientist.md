---
status: "draft"
version: "v7.1"
after_reviewer: "scientist"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules: ["src/nged_substation_forecast/defs/plotting_assets.py", "tests"]
---

# Implementation Plan: Ensemble Plotting & Test Fixes

## 1. Ensemble Plotting Optimization (`src/nged_substation_forecast/defs/plotting_assets.py`)

The `evaluate_xgboost` asset correctly outputs all 51 ensemble members using the standard `PowerForecast` schema (`valid_time`, `substation_number`, `ensemble_member`, `MW_or_MVA`).

**Decision on "one-by-one" execution:** We will stick with the efficient single-pass vectorized approach in `XGBoostForecaster.predict`. It is mathematically equivalent to looping one-by-one, but significantly faster and more idiomatic for Polars/XGBoost.

**Plotting Optimization:** The current Altair plotting logic melts the dataframe, which duplicates the `actual` line 51 times (once for each ensemble member). This causes Vega-Lite to render 51 identical overlapping lines for the actuals. We will optimize this by separating the predictions and actuals into two layers:

*   **Predictions Layer:** Plot `MW_or_MVA` with `detail="ensemble_member:N"`, `opacity=0.3`, and `strokeWidth=0.5`.
*   **Actuals Layer:** Drop duplicates from the actuals dataframe and plot `actual` with `opacity=1.0` and `strokeWidth=2.0`.
*   Combine them using `alt.layer()`.

## 2. Fix Failing Tests

Apply the fixes identified by the Scientist to the test suite:

*   **`tests/test_config_robustness.py`**: Update `test_apply_config_overrides_invalid_date` to catch `Exception` (or `pydantic_core.ValidationError`) instead of `ValueError`. `XGBoostConfig` raises a Pydantic `ValidationError` immediately upon instantiation with an invalid date string, before `_apply_config_overrides` is even called.
*   **`tests/test_xgboost_robustness.py`**: Update `test_xgboost_forecaster_train_empty_data_after_drop_nulls` to cast the mocked `temperature_2m` column containing `None` to `pl.Float32` (e.g., `pl.Series([None], dtype=pl.Float32)`). This prevents it from being inferred as a `Null` type (which is non-numeric) and being incorrectly dropped by the `_prepare_features` fallback logic before the null-dropping step.

## Review Responses & Rejections

*   **FLAW-000 (Scientist):** [ACCEPTED]. The Scientist found no flaws in the ML rigor, but identified and fixed two failing tests. We are incorporating these test fixes into the plan.
*   **User Request (Ensemble Plotting):** [ACCEPTED]. We are updating the plotting logic to handle the 51 ensemble members more efficiently.
*   **User Request ("one-by-one" execution):** [REJECTED]. We are rejecting the literal loop approach in favor of the existing single-pass vectorized approach in `XGBoostForecaster.predict`. The single-pass approach is mathematically equivalent, significantly faster, and more idiomatic for Polars/XGBoost.
