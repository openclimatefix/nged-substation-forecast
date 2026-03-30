---
review_iteration: 2
reviewer: "scientist"
total_flaws: 3
critical_flaws: 3
---

# ML Rigor Review

## FLAW-001: Substation Number Treated as Continuous Feature (Unresolved from Loop 1)
* **File & Line Number:** `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`, lines 156-161 and 58-63
* **The Theoretical Issue:** `substation_number` is cast to `pl.Int32` and included in the feature matrix. XGBoost treats `Int32` columns as continuous numerical features, assuming that substation 10 is mathematically "greater than" substation 5.
* **Concrete Failure Mode:** The model will make nonsensical splits on the `substation_number` (e.g., `substation_number < 42.5`), grouping unrelated substations together and failing to learn substation-specific behaviors accurately.
* **Required Architectural Fix:** Cast `substation_number` to `pl.Categorical` *after* schema validation (right before passing to XGBoost), or explicitly exclude it from the feature matrix and rely on static metadata features.

## FLAW-002: Loss of Substation-Specific Static Features in Global Model (Unresolved from Loop 1)
* **File & Line Number:** `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`, line 112
* **The Theoretical Issue:** The global model discards all substation metadata except `substation_number` and `h3_res_5` during the join. Furthermore, `h3_res_5` is explicitly excluded from the feature matrix in `_prepare_features`.
* **Concrete Failure Mode:** The global model lacks any spatial or categorical context (e.g., `substation_type`, `latitude`, `longitude`) to differentiate between substations, leading to severe underfitting compared to local models.
* **Required Architectural Fix:** Include relevant static features from `substation_metadata` (like `substation_type`, `latitude`, `longitude`) in the feature matrix when training the global model. Update `SubstationFeatures` schema to allow these columns.

## FLAW-003: Evaluation Metrics Include Training Data (Data Leakage in Evaluation)
* **File & Line Number:** `packages/ml_core/src/ml_core/utils.py`, lines 120-130
* **The Theoretical Issue:** The `evaluate_and_save_model` function correctly adds a `lookback` period to `slice_start` to ensure autoregressive features can be calculated for the beginning of the test set. However, it fails to filter the resulting predictions back to the actual `test_start` before calculating evaluation metrics.
* **Concrete Failure Mode:** The evaluation metrics (MAE, RMSE, MAPE) will be calculated over the period `[test_start - 14 days, test_end]`. Since the 14 days prior to `test_start` are typically part of the training set, the evaluation metrics will be artificially inflated by including data the model has already seen.
* **Required Architectural Fix:** Filter `eval_df` to only include rows where `valid_time >= test_start` before calculating the metrics.
