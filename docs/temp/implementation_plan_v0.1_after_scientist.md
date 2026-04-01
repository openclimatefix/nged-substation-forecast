---
status: "draft"
version: "v0.1"
after_reviewer: "scientist"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules: ["src/nged_substation_forecast/defs/plotting_assets.py", "tests/test_xgboost_dagster_integration.py"]
---

# Implementation Plan: Forecast vs Actual Plot Enhancements

## 1. Objective
Enhance the `forecast_vs_actual_plot` asset to display independent y-axes, include substation names in subplot titles, and specifically visualize a 14-day forecast horizon starting exactly 14 days before the latest available actuals. Update the integration test to verify these changes.

## 2. Target Files
- `src/nged_substation_forecast/defs/plotting_assets.py`
- `tests/test_xgboost_dagster_integration.py`

## 3. Implementation Details

### 3.1. `src/nged_substation_forecast/defs/plotting_assets.py`

**A. Add Substation Metadata Input**
- Update the `@dg.asset` decorator for `forecast_vs_actual_plot` to include `"substation_metadata": dg.AssetIn("substation_metadata")`.
- Add `substation_metadata: pl.DataFrame` to the function signature.

**B. Specific 14-Day Forecast Selection**
- Calculate `max_actual_time = actuals_30m.get_column("timestamp").max()`.
- Calculate `target_init_time = max_actual_time - timedelta(days=14)`.
- Extract unique `nwp_init_time` values from the `predictions` DataFrame. *(Note: The `predictions` DataFrame already contains the `nwp_init_time` column as guaranteed by the `PowerForecast` Patito schema and the `XGBoostForecaster.predict` method).*
- Select the `chosen_init_time` as the maximum `nwp_init_time` that is `<= target_init_time`. If none exist, fallback to the minimum available `nwp_init_time`. This ensures the 14-day horizon from the chosen init time has corresponding actuals.
- Filter `predictions` to only include rows where `nwp_init_time == chosen_init_time`.

**C. Filter to 14-Day Horizon**
- After joining `latest_predictions` with `actuals_30m` to create `eval_df`, filter `eval_df` to the 14-day window:
  ```python
  horizon_end = chosen_init_time + timedelta(days=14)
  plot_df = eval_df.filter(
      (pl.col("valid_time") >= chosen_init_time) &
      (pl.col("valid_time") <= horizon_end)
  )
  ```
- *(Note: The `predictions` DataFrame is already descaled to physical units (MW/MVA) by the `XGBoostForecaster.predict` method, which multiplies the raw predictions by `peak_capacity`. Therefore, no additional descaling step is required here before plotting against actuals).*

**D. Substation Names in Titles**
- Join `plot_df` with `substation_metadata` on `substation_number` to get `substation_name_in_location_table`.
- Create a new column `substation_name_with_id` formatted as `"{substation_name_in_location_table} ({substation_number})"`.
- Update the `select` statements for `preds_df` and `actuals_df` to use `substation_name_with_id` instead of `substation_number`.

**E. Independent Y-Axes in Altair**
- Update the Altair chart configuration to facet by `substation_name_with_id` instead of `substation_number`.
- Add `.resolve_scale(y='independent')` to the faceted chart to allow each subplot to scale its y-axis independently.
- Update the chart title to reflect the chosen initialization time: `f"Actuals vs Predictions (Init: {chosen_init_time})"`.

**F. Update Metadata Return**
- Update the returned `dg.MaterializeResult` metadata to include `chosen_init_time` instead of `latest_nwp`.

### 3.2. `tests/test_xgboost_dagster_integration.py`

**A. Verify Substation Names in Plot**
- The integration test already executes the full job, which includes the `substation_metadata` asset.
- Add an assertion after the plot generation to read the generated HTML file and verify that a known substation name is present in the output.
- For example, substation `110375` is named "Woodland Way". Assert that `"Woodland Way"` is in the HTML content to confirm the join and title formatting worked correctly.

## 4. Code Comments & Rationale
- **Why `target_init_time` logic?** Add a comment explaining that we select the latest `nwp_init_time` that is `<= max_actual_time - 14 days` to guarantee that the entire 14-day forecast horizon has corresponding actuals for comparison.
- **Why `resolve_scale(y='independent')`?** Add a comment explaining that different substations have vastly different power capacities (e.g., 10 MW vs 100 MW), so independent y-axes are necessary to visualize the forecast accuracy for smaller substations.
- **Why join metadata after filtering?** Add a comment explaining that joining `substation_metadata` after filtering `eval_df` to the 14-day window minimizes the size of the dataframe being joined, improving performance.

## Review Responses & Rejections

* **FLAW-001 (Scientist):** REJECTED. The scientist raised a concern that predictions might be scaled and need to be inverse-transformed before plotting. However, the `XGBoostForecaster.predict` method already descales the predictions by multiplying them by `peak_capacity` before returning them. The `predictions` DataFrame is therefore already in physical units (MW/MVA) and can be plotted directly against actuals.
* **FLAW-002 (Scientist):** REJECTED. The scientist raised a concern that the `predictions` dataframe might not contain the `nwp_init_time` column. However, the `PowerForecast` Patito schema explicitly requires the `nwp_init_time` column, and the `XGBoostForecaster.predict` method correctly populates it. Therefore, filtering on `nwp_init_time` is safe and will not crash.
