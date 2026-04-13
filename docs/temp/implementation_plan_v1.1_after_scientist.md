---
status: "draft"
version: "v1.1"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules: ["packages/dynamical_data", "packages/xgboost_forecaster", "tests"]
---

# Implementation Plan: Addressing Scientist Review Flaws

This plan addresses the flaws identified in `docs/temp/scientist_code_review_1.md`, adhering to the constraints provided.

## 1. FLAW-001: Lingering CSV References in Non-Core Packages

**Objective:** Replace CSV usage with JSON across the repository, except for the scaling parameters generation script.

**Steps:**
1.  **`packages/dynamical_data/scaling/compute_scaling_params.py`**:
    *   Keep the existing `pl.DataFrame(scaling_params).write_csv("ecmwf_scaling_params.csv")` to satisfy the constraint.
    *   Add a new line immediately after it to also write the parameters to JSON: `pl.DataFrame(scaling_params).write_json("ecmwf_scaling_params.json")`.
2.  **`packages/dynamical_data/assets/`**:
    *   Convert the existing `ecmwf_scaling_params.csv` to `ecmwf_scaling_params.json`. (The Builder can do this by running the updated `compute_scaling_params.py` script or manually converting the file).
3.  **`packages/dynamical_data/src/dynamical_data/scaling.py`**:
    *   Update `load_scaling_params` to accept `json_path` instead of `csv_path`.
    *   Change the implementation to use `pl.read_json` instead of `pl.read_csv`.
4.  **`packages/xgboost_forecaster/src/xgboost_forecaster/scaling.py`**:
    *   Update `scaling_params_path` to point to `ecmwf_scaling_params.json`.
    *   Update `load_scaling_params` to use `pl.read_json` instead of `pl.read_csv`.
5.  **`packages/dynamical_data/src/dynamical_data/processing.py`**:
    *   Update `scaling_params_path` to point to `ecmwf_scaling_params.json`.
6.  **`tests/conftest.py`**:
    *   Rename the `mock_ckan_csv_resources` fixture to `mock_ckan_json_resources`.
    *   Update the mocked data within the fixture to use `.json` extensions and `resource_type: "json"`.
7.  **`tests/test_xgboost_forecaster.py`**:
    *   Update the test setup to create a `scaling_params.json` file instead of `scaling_params.csv`, and use `write_json`.

## 2. FLAW-002: Complexity in Unit Tests

**Objective:** Simplify unit tests by reducing mocking, focusing on individual functions/assets, and using simpler test data.

**Steps:**
1.  **`tests/test_xgboost_robustness.py`**:
    *   **Refactor Data Generation:** Create a single, centralized helper function or pytest fixture that generates a minimal, valid base DataFrame for `Nwp`, `PowerTimeSeries`, and `TimeSeriesMetadata`.
    *   **Simplify Tests:** Instead of redefining massive 15+ column DataFrames in every test (`test_xgboost_forecaster_train_with_nans`, `test_xgboost_forecaster_train_with_infs`, etc.), call the helper function to get the base data and use Polars' `.with_columns()` to inject the specific NaN/Inf values required for the test.
2.  **`tests/test_xgboost_dagster_integration.py`**:
    *   **Reduce Dagster Overhead:** The current test spins up an ephemeral Dagster instance and executes the entire `xgboost_integration_job`. Refactor this to test the underlying Python functions directly (e.g., the logic inside `train_xgboost` and `evaluate_xgboost`), or use Dagster's `build_op_context()` to test the ops in isolation without the full job execution overhead.
    *   **Simplify Data:** Reduce the time window and the number of `time_series_ids` used in the test to the absolute minimum required to verify integration, speeding up the test execution.

## 3. General Code Quality & Comments

*   **Rigorous Commenting:** When updating the test files or scaling logic, ensure that comments explain the *why* (e.g., "Using JSON for scaling parameters to align with the core ingestion pipeline's data format").
*   **No FLAW IDs in Code:** Do not reference `FLAW-001` or `FLAW-002` in any source code or test comments.

---

## Review Responses & Rejections

*   **FLAW-001 (Scientist):** [PARTIALLY REJECTED]. The reviewer requested a repository-wide replacement of `.csv` with JSON/Parquet. However, per architectural constraints, we must *keep* the CSV usage for scaling parameters in `packages/dynamical_data/scaling/compute_scaling_params.py`. We will add JSON output to this script and update the rest of the repository to consume the JSON, but the CSV generation will remain.
*   **FLAW-002 (Scientist):** [ACCEPTED]. The unit tests in `test_xgboost_robustness.py` and `test_xgboost_dagster_integration.py` will be simplified by centralizing test data generation and reducing Dagster job execution overhead.
