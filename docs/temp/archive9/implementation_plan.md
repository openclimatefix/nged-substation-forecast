---
status: "draft"
version: "v1.0"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules: ["packages/contracts/src/contracts", "tests"]
---
# Implementation Plan: Enforce Power Value Constraints in PowerTimeSeries

## Objective
Fix FLAW-005 by updating the `PowerTimeSeries` schema in `packages/contracts/src/contracts/data_schemas.py` to enforce power value constraints. The power values must be constrained between -1000 and 1000.

## 1. Identify the Failing Test
The test `test_substation_power_flows_validation_extreme_values` in `tests/test_xgboost_robustness.py` currently fails because it expects a `DataFrameValidationError` when a power value of `2000.0` is provided, but the schema currently lacks the constraints to trigger this error.

## 2. Modify the Code
Update the `PowerTimeSeries` class in `packages/contracts/src/contracts/data_schemas.py`.

**Target File:** `packages/contracts/src/contracts/data_schemas.py`

**Changes:**
Modify the `power` field in the `PowerTimeSeries` class to include `ge=-1000` and `le=1000` constraints.

```python
class PowerTimeSeries(pt.Model):
    time_series_id: int = pt.Field(dtype=pl.Int32)
    # period_end_time represents the end of the 30-minute settlement period.
    period_end_time: datetime = pt.Field(
        dtype=UTC_DATETIME_DTYPE,
        description="The end time of the 30-minute period. Note that all the JSON time series data is already 30-minutely.",
    )
    # Add ge=-1000 and le=1000 to enforce power value constraints
    power: float | None = pt.Field(dtype=pl.Float32, ge=-1000, le=1000)
```

**Rationale:**
Adding `ge=-1000` and `le=1000` to the `pt.Field` definition ensures that any dataframe validated against the `PowerTimeSeries` schema will raise a `DataFrameValidationError` if it contains power values outside this physically realistic range. This prevents erroneous extreme values from propagating through the forecasting pipeline.

## 3. Verify the Fix
Run the following commands to verify the fix:

1. Run the specific failing test to ensure it now passes:
   ```bash
   pytest tests/test_xgboost_robustness.py::test_substation_power_flows_validation_extreme_values
   ```
2. Run all tests in the contracts package to ensure no regressions:
   ```bash
   pytest packages/contracts/tests/
   ```
3. Run the full test suite to ensure system-wide stability:
   ```bash
   pytest
   ```

## Code Commenting Mandate
- Ensure that any new code or modifications include comments explaining *why* the constraints are set to `[-1000, 1000]` (e.g., "Constrain power values to a physically realistic range of [-1000, 1000] MW/MVA to prevent extreme outliers from affecting downstream models").
- Do not reference FLAW-005 or any FLAW IDs in the source code comments.
