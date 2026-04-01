---
status: "draft"
version: "v1.0"
after_reviewer: "conductor"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules: ["src/nged_substation_forecast/defs", "tests"]
---

# Implementation Plan: Dagster Integration Test for XGBoost

This plan details the transition of the manual XGBoost integration test into a formal Dagster integration test.

## 1. `src/nged_substation_forecast/defs/xgb_assets.py`

**Intent:** Parameterize the XGBoost assets to allow overriding the training/testing dates and filtering by specific substations, which is necessary for a fast integration test. We also integrate the `healthy_substations` asset to ensure we only train on clean data.

**Changes:**
- Add `XGBoostConfig(dg.Config)` with optional fields: `train_start`, `train_end`, `test_start`, `test_end` (all `str | None`), and `substation_ids` (`list[int] | None`).
- Update `train_xgboost` and `evaluate_xgboost`:
  - Add `config: XGBoostConfig` and `healthy_substations: list[int]` to the function signatures.
  - Override the Hydra `data_split` configuration with the values from `XGBoostConfig` if they are provided.
  - Filter `substation_power_flows` and `substation_metadata` to only include substations present in `healthy_substations`.
  - Further filter by `config.substation_ids` if provided.
  - Change the type hint of `substation_power_flows` from `pl.LazyFrame` to `pl.DataFrame` (since `combined_actuals` returns a DataFrame) and call `.lazy()` before filtering.

## 2. `src/nged_substation_forecast/defs/plotting_assets.py`

**Intent:** Formalize the Altair plotting logic from the manual test into a Dagster asset that visualizes the forecast vs. actuals.

**Changes:**
- Remove `partitions_def=model_partitions` from `forecast_vs_actual_plot` as it will now directly consume the unpartitioned `evaluate_xgboost` output.
- Update inputs to `evaluate_xgboost: pl.DataFrame` and `combined_actuals: pl.DataFrame`.
- Implement the Altair plotting logic:
  - Downsample `combined_actuals` to 30-minute intervals.
  - Filter the predictions (`evaluate_xgboost`) to only include the latest `nwp_init_time` (single NWP run filtering).
  - Join predictions with actuals.
  - Filter the plot data to the last 14 days.
  - Generate the Altair chart and save it to `tests/xgboost_integration_plot.html`.
  - Return a `dg.MaterializeResult` with the plot path in the metadata.

## 3. `src/nged_substation_forecast/defs/xgb_jobs.py`

**Intent:** Create a dedicated Dagster job that selects all the necessary assets for the integration test, making it easy to execute the entire pipeline in one go.

**Changes:**
- Define `xgboost_integration_job = dg.define_asset_job(...)`.
- Include the following assets in the selection:
  - `substation_metadata`
  - `live_primary_flows`
  - `combined_actuals`
  - `healthy_substations`
  - `all_nwp_data`
  - `processed_nwp_data`
  - `train_xgboost`
  - `evaluate_xgboost`
  - `forecast_vs_actual_plot`

## 4. `tests/test_xgboost_dagster_integration.py`

**Intent:** Create a formal `pytest` test that executes the Dagster job with a specific configuration, ensuring the pipeline runs end-to-end and produces the expected outputs.

**Changes:**
- Create `test_xgboost_integration_job()`.
- Dynamically calculate `train_start`, `train_end`, `test_start`, and `test_end` by reading the local Delta table (skipping the test if local data is missing, as this is a "true" integration test relying on real data).
- Define a `run_config` dictionary:
  - Limit `live_primary_flows` to the 5 specified substations (`[110375, 110644, 110772, 110803, 110804]`).
  - Pass the dynamic dates and the 5 substation IDs to `train_xgboost` and `evaluate_xgboost`.
- Execute the job using `xgboost_integration_job.execute_in_process(...)`, providing the latest date as the `partition_key` for `live_primary_flows`.
- Assert that the job execution is successful.
- Assert that the `tests/xgboost_integration_plot.html` file was created.
