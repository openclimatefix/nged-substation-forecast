---
status: "draft"
version: "v1.0"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules: ["packages/ml_core/src/ml_core", "packages/ml_core/tests"]
---
# Implementation Plan: Fix LocalForecasters Interface

## Objective
Fix the failing test suite by ensuring `LocalForecasters` and `MockForecaster` correctly implement the `BaseForecaster` interface, specifically by implementing the `fit` method instead of `train`.

## Context
The `BaseForecaster` abstract base class defines a `fit` method for training models. However, `LocalForecasters` (in `experimental.py`) and `MockForecaster` (in `test_ml_core_model.py`) incorrectly implement a `train` method instead. This causes a `TypeError` when attempting to instantiate `LocalForecasters` because it fails to implement the abstract method `fit`.

## Changes Required

### 1. `packages/ml_core/src/ml_core/experimental.py`
- **Rename Method:** Rename the `train` method in the `LocalForecasters` class to `fit`.
- **Update Type Hints:** Update the type hints of the `fit` method to exactly match the `BaseForecaster.fit` signature.
  - Change `power_time_series: pl.LazyFrame` to `power_time_series: pt.LazyFrame[PowerTimeSeries]`.
  - Change `nwps: Mapping[NwpModel, pl.LazyFrame] | None = None` to `nwps: Mapping[NwpModel, pt.LazyFrame[Nwp]] | None = None`.
- **Update Return Type:** Keep the return type as `"LocalForecasters"` (or `Any` as per the base class, but returning `self` is standard for scikit-learn style estimators).
- **Code Comments:** Ensure the docstring for `fit` clearly explains *why* this method exists (to satisfy the `BaseForecaster` interface and delegate training to individual local models).
- **Internal Calls:** Ensure that inside `LocalForecasters.fit`, the call to the underlying model is `model.fit(...)` (which it already is, but verify it remains correct).

### 2. `packages/ml_core/tests/test_ml_core_model.py`
- **Rename Method:** Rename the `train` method in the `MockForecaster` class to `fit`.
- **Update Type Hints:** Update the type hints of the `fit` method to exactly match the `BaseForecaster.fit` signature.
  - Change `power_time_series: pl.LazyFrame` to `power_time_series: pt.LazyFrame[PowerTimeSeries]`.
  - Change `nwps: Mapping[NwpModel, pl.LazyFrame] | None = None` to `nwps: Mapping[NwpModel, pt.LazyFrame[Nwp]] | None = None`.
- **Test Calls:** Ensure that the test function `test_local_forecasters` calls `local_forecasters.fit(...)` (which it already does).

## Review Responses & Rejections
None yet.
