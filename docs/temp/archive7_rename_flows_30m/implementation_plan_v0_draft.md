---
status: "draft"
version: "v0"
after_reviewer: "architect"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules: ["packages/ml_core", "packages/xgboost_forecaster", "tests"]
---

# Implementation Plan: Rename `flows_30m` to `power_time_series`

## 1. Objective
Rename the variable, parameter, and dictionary key `flows_30m` to `power_time_series` throughout the entire codebase. This change improves domain language consistency, as the data represents a generic power time series rather than being strictly tied to a 30-minute resolution in its naming.

## 2. Strategy for Coordinated Refactoring
To ensure consistency and minimize the risk of breaking changes, the refactoring should be performed in a specific order:
1.  **Core Interfaces (`ml_core`)**: Update the base protocols, abstract classes, and utility functions first. This establishes the new contract.
2.  **Implementations (`xgboost_forecaster`)**: Update the concrete model implementations and feature engineering functions that depend on the core interfaces.
3.  **Tests**: Update all unit and integration tests, including mock data generation and test assertions.

**Important Note on Code Comments:**
When updating these files, ensure that any new or modified comments focus on the *why* (intent and rationale) rather than the *how*. For example, if a docstring is updated, ensure it clearly explains that `power_time_series` represents the historical power measurements used for autoregressive features or target calculation.

## 3. Files to Update

Based on codebase exploration, the following files contain references to `flows_30m` and must be updated:

**Core Interfaces (`packages/ml_core`)**
*   `packages/ml_core/src/ml_core/model.py`
*   `packages/ml_core/src/ml_core/experimental.py`
*   `packages/ml_core/src/ml_core/utils.py`
*   `packages/ml_core/tests/test_ml_core_model.py`

**Implementations (`packages/xgboost_forecaster`)**
*   `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`
*   `packages/xgboost_forecaster/src/xgboost_forecaster/features.py`
*   `packages/xgboost_forecaster/tests/test_universal_model.py`
*   `packages/xgboost_forecaster/tests/test_xgboost_features.py`
*   `packages/xgboost_forecaster/tests/test_xgboost_model.py`

**Integration Tests (`tests/`)**
*   `tests/test_xgboost_adversarial.py`
*   `tests/test_xgboost_robustness.py`
*   `tests/test_xgboost_forecaster.py`

## 4. Specific Update Instructions

### A. Function Signatures and Docstrings
Update all method signatures in `model.py`, `experimental.py`, and `features.py` to replace `flows_30m` with `power_time_series`.
*   *Example Before:* `def train(self, df: pl.LazyFrame, flows_30m: pl.LazyFrame, ...) -> None:`
*   *Example After:* `def train(self, df: pl.LazyFrame, power_time_series: pl.LazyFrame, ...) -> None:`
*   Update the corresponding docstrings to match the new parameter name.

### B. Variable Names and Type Casting
Update local variable assignments and type casting calls.
*   *Example Before:* `flows_30m = cast(pt.LazyFrame[PowerTimeSeries], flows_30m)`
*   *Example After:* `power_time_series = cast(pt.LazyFrame[PowerTimeSeries], power_time_series)`

### C. Dictionary Keys
In `packages/ml_core/src/ml_core/utils.py`, update the dictionary keys used for slicing data.
*   *Example Before:* `sliced_data["flows_30m"] = flows_30m`
*   *Example After:* `sliced_data["power_time_series"] = power_time_series`

### D. Logging Statements
Update any logging or error messages that reference the variable name.
*   *Example Before:* `log.info(f"Input flows_30m columns: {flows_30m.collect_schema().names()}")`
*   *Example After:* `log.info(f"Input power_time_series columns: {power_time_series.collect_schema().names()}")`

### E. Test Mocks and Fixtures
In all test files, update the instantiation of mock data and the keyword arguments passed to the models/functions.
*   *Example Before:* `model.predict(df, flows_30m=flows_30m)`
*   *Example After:* `model.predict(df, power_time_series=power_time_series)`

## 5. Verification Step
After all replacements are made, the Builder must run the following verification steps:
1.  **Type Checking:** Run `uv run mypy .` or `uv run pyright` to ensure no interface mismatches occurred during the rename.
2.  **Unit Tests:** Run `uv run pytest` across the entire repository to ensure all tests pass.
3.  **Grep Check:** Run `rg "\bflows_30m\b"` to verify that zero instances of the old variable name remain in the codebase.

## 6. Risk Minimization Strategy
*   **Strict Word Boundaries:** When performing search-and-replace, use strict word boundaries (e.g., `\bflows_30m\b` in regex) to avoid accidentally modifying partial matches (though unlikely for this specific string).
*   **No Silent Failures:** Ensure that dictionary key lookups (like in `utils.py`) are updated symmetrically (both the write and the read) to prevent `KeyError` or silent fallback behaviors.
*   **Atomic Commits:** The Conductor should commit these changes as a single, atomic commit to ensure the codebase is never in a broken, intermediate state where interfaces mismatch.

## Review Responses & Rejections
*(To be filled after review)*
