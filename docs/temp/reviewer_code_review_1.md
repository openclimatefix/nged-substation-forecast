---
review_iteration: 1
reviewer: "review"
total_flaws: 5
critical_flaws: 0
---

# Code Review

## FLAW-001: Redundant Feature Engineering Calls
* **File & Line Number:** `docs/temp/implementation_plan.md`, line 44
* **The Issue:** The plan proposes calling both `add_weather_features` and `add_physical_features`, but the former already calls the latter.
* **Concrete Failure Mode:** Redundant computation and potential column name collisions.
* **Required Fix:** Update the plan to call `add_weather_features` only, or ensure they are independent.

## FLAW-002: Lazy Pipeline Break in Feature Engineering
* **File & Line Number:** `packages/xgboost_forecaster/src/xgboost_forecaster/features.py`, lines 43-147
* **The Issue:** Existing feature engineering functions are typed for `pl.DataFrame` and may not fully support `pl.LazyFrame`.
* **Concrete Failure Mode:** Performance degradation due to premature collection or runtime errors when integrated into the proposed lazy pipeline.
* **Required Fix:** Refactor all functions in `features.py` to accept and return `pl.LazyFrame`.

## FLAW-003: Missing `latest_available_weekly_lag` in `SubstationFeatures` Contract
* **File & Line Number:** `packages/contracts/src/contracts/data_schemas.py`, lines 281-326
* **The Issue:** The `SubstationFeatures` contract does not include the `latest_available_weekly_lag` column used by the model.
* **Concrete Failure Mode:** `SubstationFeatures.validate(df)` will fail during training and inference.
* **Required Fix:** Update the `SubstationFeatures` Patito model to include `latest_available_weekly_lag`.

## FLAW-004: Incomplete `_add_lags` Logic for Missing `init_time`
* **File & Line Number:** `docs/temp/implementation_plan.md`, line 67
* **The Issue:** The proposed `_add_lags` method does not specify how to handle cases where `init_time` is missing (Point 6).
* **Concrete Failure Mode:** Runtime error when calculating `lead_time` for autoregressive-only models.
* **Required Fix:** Ensure `_add_lags` handles the absence of `init_time` by defaulting to a static lag strategy.

## FLAW-005: `evaluate_and_save_model` Lead Time Evaluation
* **File & Line Number:** `docs/temp/implementation_plan.md`, lines 17-20
* **The Issue:** Evaluation by lead time requires `predict` to return multiple forecasts per `valid_time`, but `predict` currently collapses them.
* **Concrete Failure Mode:** Evaluation remains biased towards short lead times as only the latest forecast is seen.
* **Required Fix:** Add a `collapse_lead_times` toggle to `predict` or refactor evaluation to iterate over initialization times.
