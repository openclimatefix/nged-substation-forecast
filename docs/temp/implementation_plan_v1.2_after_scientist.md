---
status: "draft"
version: "v1.2"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules: ["tests"]
---

# Implementation Plan: Addressing Scientist Review Flaws

This plan addresses the flaws identified in `docs/temp/scientist_code_review_1.md`, adhering to the constraints provided and subsequent user feedback.

## 1. FLAW-001: Lingering CSV References in Non-Core Packages

**Objective:** Keep scaling parameters as CSV only. Remove CKAN-related tests entirely.

**Steps:**
1.  **CKAN Tests Removal**:
    *   Completely remove any tests related to CKAN.
    *   Remove the `mock_ckan_csv_resources` fixture from `tests/conftest.py` (or any other location).
    *   Remove any test files or functions that depend on CKAN mocking.

*(Note: All other suggestions from FLAW-001 regarding converting scaling parameters to JSON have been rejected. Scaling parameters will remain as CSV only, and no JSON output will be added.)*

## 2. FLAW-002: Complexity in Unit Tests

**Objective:** Simplify unit tests by reducing mocking, focusing on individual functions/assets, and refactoring large integration tests.

**Steps:**
1.  **`tests/test_xgboost_robustness.py`**:
    *   **Refactor Data Generation:** Create a single, centralized helper function or pytest fixture that generates a minimal, valid base DataFrame for `Nwp`, `PowerTimeSeries`, and `TimeSeriesMetadata`.
    *   **Simplify Tests:** Instead of redefining massive 15+ column DataFrames in every test (`test_xgboost_forecaster_train_with_nans`, `test_xgboost_forecaster_train_with_infs`, etc.), call the helper function to get the base data and use Polars' `.with_columns()` to inject the specific NaN/Inf values required for the test.
2.  **`tests/test_xgboost_dagster_integration.py`**:
    *   **Refactor Large Functions:** Do NOT simplify the test by reducing Dagster overhead or data size. Instead, refactor the huge single test function into a set of smaller, clearly defined helper functions. Maintain the full Dagster execution context and data size. But just make the file easier to read by extracting helper functions (that can live in `tests/test_xgboost_dagster_integration.py`).

## 3. General Code Quality & Comments

*   **Rigorous Commenting:** When updating the test files, ensure that comments explain the *why* (e.g., "Refactored into smaller functions to improve readability and isolate specific integration steps").
*   **No FLAW IDs in Code:** Do not reference `FLAW-001` or `FLAW-002` in any source code or test comments.

---

## Review Responses & Rejections

*   **FLAW-001 (Scientist):** [REJECTED]. The reviewer requested a repository-wide replacement of `.csv` with JSON/Parquet. This is rejected. Scaling parameters will remain as CSV only, and no JSON output will be added. Furthermore, any CKAN-related tests (which previously used CSV mocks) are to be completely removed.
*   **FLAW-002 (Scientist):** [PARTIALLY ACCEPTED]. The unit tests in `test_xgboost_robustness.py` will be simplified by centralizing test data generation. However, the suggestion to reduce Dagster overhead and data size in `test_xgboost_dagster_integration.py` is REJECTED. Instead, the large integration test function will be refactored into smaller, clearly defined functions without reducing the scope or overhead of the test.
