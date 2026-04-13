---
reviewer: "scientist"
total_flaws: 2
critical_flaws: 0
test_status: "needs_simplification"
---

# ML Rigor Review

## FLAW-001: Lingering CSV References in Non-Core Packages
* **File & Line Number:** `packages/dynamical_data/scaling/compute_scaling_params.py`
* **The Theoretical Issue:** While the core ingestion pipeline has been successfully migrated to JSON, there are lingering references to CSV in auxiliary packages (e.g., `compute_scaling_params.py`).
* **Concrete Failure Mode:** Inconsistent data handling across the repository, potentially leading to confusion for new developers and maintenance overhead.
* **Required Architectural Fix:** The Builder should perform a repository-wide search for `.csv` and replace all remaining ingestion/export logic with JSON or Parquet, ensuring consistency with the core pipeline.

## FLAW-002: Complexity in Unit Tests
* **File & Line Number:** `tests/test_xgboost_dagster_integration.py`, `tests/test_xgboost_robustness.py`
* **The Theoretical Issue:** The unit tests are overly complex, often mocking entire Dagster environments or using adversarial data generation that is difficult to maintain.
* **Concrete Failure Mode:** High maintenance burden for tests, making it harder to refactor the core logic without breaking tests.
* **Required Architectural Fix:** The Builder should simplify unit tests by focusing on testing individual functions/assets with minimal mocking, and using simpler, more representative test data.

# General Observations
- **JSON Ingestion:** The core ingestion pipeline in `src/nged_substation_forecast/defs/nged_assets.py` is successfully using JSON.
- **"MW"/"MVA" Selection:** No explicit selection logic found; the pipeline correctly processes the `PowerTimeSeries` schema.
- **Generalization:** The codebase has been successfully generalized from `substation_number` to `time_series_id`.
- **Dead Code:** Some unused imports and potentially dead code in `packages/` should be cleaned up.
