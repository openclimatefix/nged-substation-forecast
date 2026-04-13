---
status: "draft"
version: "v1.0"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules: ["packages/xgboost_forecaster", "packages/nged_data", "tests", "conf"]
---
# Implementation Plan: Rename `substation_number` to `time_series_id`

## Objective
Rename `substation_number` to `time_series_id` throughout the codebase, while strictly preserving `substation_number` within the `TimeSeriesMetadata` schema and its related mock data.

## Constraints
**CRITICAL CONSTRAINT:** Do NOT remove or rename `substation_number` within the `TimeSeriesMetadata` schema or any mock dataframes that are validated against it, as it remains a required field in the contract.

## Files to Update

### 1. Configuration
*   **`conf/model/xgboost.yaml`**:
    *   In `features.feature_names`, change `"substation_number"` to `"time_series_id"`.

### 2. Model Code
*   **`packages/xgboost_forecaster/src/xgboost_forecaster/model.py`**:
    *   In `_prepare_data_for_model`, remove `"substation_number"` from the `select` list when creating `metadata_lf`. (It currently selects `["time_series_id", "h3_res_5", "substation_number"]`).
    *   *Rationale:* `time_series_id` is already present and cast to categorical later in the pipeline. `substation_number` is no longer needed as a feature.

### 3. Data Processing
*   **`packages/nged_data/src/nged_data/clean.py`**:
    *   Update the comment `# Note: The original cleaning.py used "substation_number" for rolling_std.` to `# Note: The original cleaning.py used "time_series_id" for rolling_std.`

### 4. Tests
*   **`tests/test_xgboost_forecaster.py`**:
    *   Fix the broken test `test_prepare_training_data_prevents_row_explosion` (around line 214). The `metadata` dictionary currently has duplicate `"substation_number"` keys and is missing `"time_series_id"`.
    *   *Change from:*
        ```python
        metadata = pl.DataFrame(
            {
                "substation_number": [1],
                "substation_number": [1],
                "h3_res_5": [1],
            }
        ).with_columns(pl.col("time_series_id").cast(pl.Int32))
        ```
    *   *Change to:*
        ```python
        metadata = pl.DataFrame(
            {
                "time_series_id": [1],
                "substation_number": [1],
                "h3_res_5": [1],
            }
        ).with_columns(pl.col("time_series_id").cast(pl.Int32))
        ```
*   **`tests/conftest.py`**:
    *   In the `mock_ckan_primary_substation_locations` fixture, rename `"substation_number"` to `"time_series_id"`. (This is dead code since CKAN ingestion was removed, but renaming it ensures no stray references remain).

## Files to STRICTLY PRESERVE (Do NOT Rename `substation_number`)
The following files contain `substation_number` as part of the `TimeSeriesMetadata` contract or its mock data. **Do not modify these instances:**
*   `packages/contracts/src/contracts/data_schemas.py` (The schema definition)
*   `packages/contracts/tests/test_data_schemas.py` (Schema tests)
*   `packages/nged_data/src/nged_data/io.py` (Casting `substation_number` to `pl.Int32` for the schema)
*   `packages/dashboard/main.py` (Displaying `substation_number` in the dashboard table)
*   All mock `TimeSeriesMetadata` DataFrames in tests (e.g., `tests/test_xgboost_robustness.py`, `tests/test_xgboost_adversarial.py`, `tests/test_xgboost_forecaster.py` (except the bug fix above), `tests/test_plotting_robustness.py`, `tests/test_xgboost_mlflow.py`, `tests/test_xgboost_dagster_mocked.py`, `packages/xgboost_forecaster/tests/test_universal_model.py`, `packages/xgboost_forecaster/tests/test_xgboost_model.py`, `packages/ml_core/tests/test_ml_core_model.py`).

## Strategy
1.  **Manual Edits:** Use the `Edit` tool to manually update the specific files listed in the "Files to Update" section. This is safer than a global search-and-replace because the vast majority of remaining `substation_number` instances are within `TimeSeriesMetadata` mock data and must be preserved.
2.  **Verification:** Run a global search (`rg "substation_number"`) to verify that all remaining instances are strictly related to `TimeSeriesMetadata`.
3.  **Testing:** Run the full test suite (`pytest`) with a timeout of 600000ms to ensure no regressions and that the broken test is fixed.

## Verification Steps
1.  Run `rg "substation_number"` and manually inspect the output. Ensure it only appears in the "Files to STRICTLY PRESERVE" list.
2.  Run `pytest packages/contracts/tests/test_data_schemas.py` to ensure the schema constraint is intact.
3.  Run `pytest tests/test_xgboost_forecaster.py -k "test_prepare_training_data_prevents_row_explosion"` to ensure the bug is fixed.
4.  Run `pytest` (with `-s` and timeout `600000`) to ensure all tests pass.
