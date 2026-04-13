---
status: "draft"
version: "v3.3"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules: ["src/nged_substation_forecast/defs"]
---

# Implementation Plan: Fix DagsterInvalidDefinitionError

## 1. Investigation of `DagsterInvalidDefinitionError`

The `DagsterInvalidDefinitionError` reported in `FLAW-001` and `FLAW-002` occurs during test collection in `tests/test_data_cleaning_robustness.py` and `tests/test_xgboost_dagster_integration.py`.

The root cause is the return type annotation on the `cleaned_power_time_series` asset in `src/nged_substation_forecast/defs/data_cleaning_assets.py`:

```python
@dg.asset(...)
def cleaned_power_time_series(
    context: dg.AssetExecutionContext,
    settings: ResourceParam[Settings],
) -> pt.DataFrame[PowerTimeSeries]:
```

Dagster attempts to infer the `dagster_type` from the return type annotation. However, `pt.DataFrame[PowerTimeSeries]` is a generic type alias that Dagster's type resolution system does not natively understand, leading to the error:
`dagster._core.errors.DagsterInvalidDefinitionError: Problem using type 'patito.polars.DataFrame[contracts.data_schemas.PowerTimeSeries]' from return type annotation, correct the issue or explicitly set the dagster_type via Out().`

Because the test files import the Dagster definitions (either directly or via `nged_substation_forecast.definitions`), the error is triggered during test collection, preventing the test suite from running.

## 2. Steps to Fix the Dagster Definitions

To resolve this issue, we need to remove the problematic return type annotation from the `cleaned_power_time_series` asset. This is consistent with other assets in the codebase (e.g., in `nged_assets.py` and `xgb_assets.py`) which either omit the return type annotation or use standard types like `pl.DataFrame`.

**Step 1: Update `src/nged_substation_forecast/defs/data_cleaning_assets.py`**
- Locate the `cleaned_power_time_series` function definition.
- Remove the `-> pt.DataFrame[PowerTimeSeries]:` return type annotation.
- Ensure the function signature ends with `):` instead.

*Note: The actual return value inside the function remains a `pt.DataFrame[PowerTimeSeries]`, preserving the Patito schema validation at runtime. We are only removing the type hint from the function signature to satisfy Dagster's type inference.*

**Step 2: Verify Test Collection**
- Run `pytest tests/test_data_cleaning_robustness.py tests/test_xgboost_dagster_integration.py` to ensure the tests are collected and executed successfully without the `DagsterInvalidDefinitionError`.

## Review Responses & Rejections

* **FLAW-001 (Tester):** ACCEPTED. The `DagsterInvalidDefinitionError` in `test_data_cleaning_robustness.py` is caused by the `pt.DataFrame[PowerTimeSeries]` return type annotation on the `cleaned_power_time_series` asset. The fix is to remove this annotation.
* **FLAW-002 (Tester):** ACCEPTED. The same error in `test_xgboost_dagster_integration.py` is caused by the same root issue, as it imports the global `defs` which includes the problematic asset. The fix for FLAW-001 will resolve this as well.
