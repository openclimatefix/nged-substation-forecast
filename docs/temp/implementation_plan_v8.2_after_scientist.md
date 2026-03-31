---
status: "draft"
version: "v8.2"
after_reviewer: "scientist"
task_type: "standard"
requires_ml_review: true
requires_data_engineer: false
target_modules: ["packages/xgboost_forecaster/src/xgboost_forecaster", "conf/model"]
---

# Implementation Plan: Fix Ensemble Variance in XGBoost

## Objective
Address the lack of ensemble variance in the XGBoost forecaster by fixing a meteorological bug in weather lags, increasing model capacity, and ensuring critical weather features are present. We are intentionally keeping `substation_number` as a feature for now, deferring its removal.

## 1. Fix Weather Lags
**File:** `packages/xgboost_forecaster/src/xgboost_forecaster/features.py`
**Function:** `_add_lag_asof` inside `add_weather_features`

*   **Intent:** Ensemble members in NWP models are drawn randomly and independently for each `init_time`. Matching historical lags by `ensemble_member` introduces noise for perturbed members. We must use the Control Member (Member 0) as the consistent historical baseline for *all* ensemble members.
*   **Implementation:**
    *   Modify `_add_lag_asof` to use the **Control Member (Member 0)** for all weather lags.
    *   Check if `NwpColumns.ENSEMBLE_MEMBER` is in `schema_names`.
    *   If it is, filter `source_df` to only include the control member: `source_df = source_df.filter(pl.col(NwpColumns.ENSEMBLE_MEMBER) == 0)`.
    *   Remove `NwpColumns.ENSEMBLE_MEMBER` from the `by_cols` list. This will cause the `join_asof` to broadcast the control member's lag to all ensemble members in the target `df`.
    *   *Note:* The user requested this change for *all* weather lags (including `6h_ago`), so apply it universally within `_add_lag_asof`.

## 2. Increase Model Capacity
**File:** `conf/model/xgboost.yaml`

*   **Intent:** The model is currently underfitting, using its limited capacity to learn the `substation_number` baseline and ignoring weather features. We need to increase capacity so it can learn both the baseline and the subtle weather sensitivities that drive ensemble variance.
*   **Implementation:**
    *   Update `n_estimators` to 500.
    *   Update `learning_rate` to 0.05.

## 3. Add Solar/Wind to Critical Columns
**File:** `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`
**Functions:** `train` and `predict`

*   **Intent:** Ensure that the model does not attempt to train or predict on rows where critical weather features (solar radiation and wind speed) are missing, as these are essential for capturing weather-driven variance.
*   **Implementation:**
    *   In both `train` and `predict`, locate the `critical_cols` list definition.
    *   Update the `if nwps:` block to append `f"{NwpColumns.SW_RADIATION}_uint8_scaled"` and `f"{NwpColumns.WIND_SPEED_10M}_uint8_scaled"` to `critical_cols`, in addition to the existing temperature column.

## 4. Verification
*   Run the Dagster integration test: `uv run pytest tests/test_xgboost_dagster_integration.py`.
*   Inspect the generated plot `tests/xgboost_dagster_integration_plot.html` to visually verify that the ensemble members now show visible variance around the mean forecast.

## Review Responses & Rejections

* **FLAW-001 (Scientist):** [REJECTED IN PART]. The Scientist recommended removing `substation_number` from the features to prevent feature dominance. However, per user request, we are **KEEPING** `substation_number` as a feature for now. We are ACCEPTING the recommendation to increase model capacity (`n_estimators` to 500, `learning_rate` to 0.05) to help the model learn weather effects alongside the substation baseline.
* **FLAW-002 (Scientist):** [ACCEPTED]. We are modifying `_add_lag_asof` to use the Control Member (Member 0) for weather lags to fix the meteorological inconsistency. Per user instruction, this is applied to *all* weather lags.
