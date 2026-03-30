---
review_iteration: 1
reviewer: "tester"
total_flaws: 4
critical_flaws: 0
test_status: "tests_passed"
---

# Testability & QA Review

The current branch introduces a significant architectural shift towards a unified global model and strictly typed configurations. The overall test coverage is good, with a mix of unit, integration, and adversarial tests. The robustness of the `XGBoostForecaster` is notable, particularly the explicit NaN/Inf checks and the handling of data leakage through lead-time filtering and dynamic lags.

However, several flaws related to data consistency and edge-case handling were identified.

## FLAW-001: Potential for mixing MW and MVA in `downsample_power_flows`
* **File & Line Number:** `packages/ml_core/src/ml_core/data.py`, lines 38-43
* **The Issue:** The `downsample_power_flows` function performs a per-row (per 30-minute window) fallback from `MW` to `MVA`. If a substation has `MW` data for some windows and only `MVA` data for others within the same training or inference set, the resulting `MW_or_MVA` column will contain a mixture of both.
* **Concrete Failure Mode:** If `MW` and `MVA` have different physical characteristics or scaling (e.g., due to power factor), mixing them in a single target/feature column without normalization can introduce noise and bias into the ML model, leading to degraded forecast accuracy.
* **Required Fix:** Implement a more robust selection logic that chooses either `MW` or `MVA` for the entire substation's timeseries based on completeness, rather than switching per-row. This should be consistent with `SubstationFlows.choose_power_column`.

## FLAW-002: Simplistic power column selection in `SubstationFlows`
* **File & Line Number:** `packages/contracts/src/contracts/data_schemas.py`, lines 90-92
* **The Issue:** `SubstationFlows.choose_power_column` returns `"MW"` only if *all* values in the column are non-null.
* **Concrete Failure Mode:** If a substation's `MW` data is 99% complete but has a single null value, and its `MVA` data is only 10% complete but happens to be non-null for the first few rows, the logic might choose the much less complete `MVA` column (or fail later if `MVA` also has nulls).
* **Required Fix:** Update the selection logic to choose the column with the highest percentage of non-null values, or prioritize `MW` if it meets a certain completeness threshold (e.g., >90%).

## FLAW-003: Potential for missing weather lags in `add_weather_features`
* **File & Line Number:** `packages/xgboost_forecaster/src/xgboost_forecaster/features.py`, lines 79-80
* **The Issue:** If `history` is not explicitly provided to `add_weather_features`, it uses the `weather` LazyFrame itself to calculate 7-day and 14-day lags.
* **Concrete Failure Mode:** During inference, if the `nwps` dictionary passed to `predict` only contains the latest forecast (e.g., the next 10 days), the 7-day and 14-day lags will be entirely null for the first week of the forecast. While `evaluate_and_save_model` currently adds a 14-day lookback, other entry points to `predict` might not, leading to silent failures where the model receives null features.
* **Required Fix:** Add a check in `add_weather_features` to ensure that the provided data covers the required lag range, or raise a warning/error if lags cannot be calculated. Ensure all callers of `predict` are aware of the lookback requirement.

## FLAW-004: Lack of Property-Based Testing for Data Contracts
* **File & Line Number:** `packages/contracts/tests/test_data_schemas.py`
* **The Issue:** The current tests for Patito data contracts use hardcoded examples and do not leverage property-based testing.
* **Concrete Failure Mode:** Subtle edge cases in validation logic (e.g., interaction between `MW`/`MVA` presence and `ingested_at` nullability) might be missed by manual test cases.
* **Required Fix:** Integrate a property-based testing library (e.g., `hypothesis`) to generate a wide range of valid and invalid DataFrames to rigorously test the `validate` methods of all Patito models.

# Coverage & Quality Summary
- **Unit Tests:** Good coverage for feature engineering and utility functions.
- **Integration Tests:** The `tests/test_xgboost_forecaster.py` provides a solid end-to-end test of the training and prediction pipeline.
- **Adversarial Tests:** New tests in `tests/test_xgboost_robustness.py` (added during this review) confirm that the model correctly handles NaNs, Infs, and extreme values by raising appropriate errors.
- **Hydra Configuration:** Well-tested through Pydantic validation and integration tests.
