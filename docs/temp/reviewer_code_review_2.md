---
review_iteration: 2
reviewer: "review"
total_flaws: 6
critical_flaws: 2
---

# Code Review

## FLAW-001: Crash when NWPs are missing (AR-only models)
* **File & Line Number:** `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`, lines 170 and 303
* **The Issue:** The code attempts to calculate `lead_time_hours_temp` by subtracting `init_time` from `valid_time`. However, `init_time` is only present if NWP data has been joined.
* **Concrete Failure Mode:** If `nwps` is `None` or empty (e.g., for a purely autoregressive model), the join with NWPs is skipped, and the subsequent `with_columns` call will fail with a `ColumnNotFoundError` for `init_time`.
* **Required Fix:** Wrap the lag selection logic in a check for `init_time` presence, or implement the proposed `_add_lags` method with a fallback strategy for missing `init_time`.

## FLAW-002: Lead Time Evaluation still broken (FLAW-005 from Iteration 1)
* **File & Line Number:** `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`, lines 227-234
* **The Issue:** `XGBoostForecaster.predict` still collapses lead times by taking the `last()` forecast for each `valid_time`, and it does not implement the `collapse_lead_times` parameter from the `BaseForecaster` interface.
* **Concrete Failure Mode:** It is impossible to evaluate model performance across different lead times because only the most recent forecast for any given timestamp is preserved.
* **Required Fix:** Implement the `collapse_lead_times` parameter. If `False`, return all available forecasts from all NWP initialization times.

## FLAW-003: Duplicated Lag Logic
* **File & Line Number:** `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`, lines 118-128 and 282-292
* **The Issue:** The logic for generating 7-day and 14-day lags, as well as the dynamic seasonal lag selection, is duplicated across `train` and `predict`.
* **Concrete Failure Mode:** Increased maintenance burden and risk of logic divergence between training and inference.
* **Required Fix:** Extract the lag generation logic into a private method (e.g., `_add_lags`) as originally proposed in the implementation plan.

## FLAW-004: Restrictive Type Hints in Feature Engineering
* **File & Line Number:** `packages/ml_core/src/ml_core/features.py`, line 7
* **The Issue:** `add_cyclical_temporal_features` is typed to accept `pl.DataFrame`, even though its implementation is fully compatible with `pl.LazyFrame`.
* **Concrete Failure Mode:** Static type checkers (like `ty check`) will flag errors when this function is used in a lazy pipeline, despite it being functionally correct.
* **Required Fix:** Update the type hint to `pl.DataFrame | pl.LazyFrame` or use a `TypeVar`.

## FLAW-005: Dead Code in `xgboost_forecaster`
* **File & Line Number:** `packages/xgboost_forecaster/src/xgboost_forecaster/features.py`
* **The Issue:** This file contains old feature engineering logic that is no longer used by `XGBoostForecaster` (which now uses `ml_core.features`).
* **Concrete Failure Mode:** Confusion for future maintainers and unnecessary bloat in the codebase.
* **Required Fix:** Delete `packages/xgboost_forecaster/src/xgboost_forecaster/features.py`.

## FLAW-006: Non-deterministic `power_fcst_init_time`
* **File & Line Number:** `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`, line 332
* **The Issue:** `XGBoostForecaster.predict` uses `datetime.now(timezone.utc)` to populate `power_fcst_init_time`.
* **Concrete Failure Mode:** Forecast outputs are non-deterministic. In backtesting or re-runs, the `power_fcst_init_year_month` partition key will change based on when the code is executed, making it difficult to maintain a stable evaluation table.
* **Required Fix:** Allow passing a `forecast_init_time` or derive it from the `inference_params`.
