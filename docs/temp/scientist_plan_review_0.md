---
review_iteration: 0
reviewer: "scientist"
total_flaws: 2
critical_flaws: 1
---

# ML Rigor Review

## FLAW-001: Missing Target Descaling for Physical Interpretability
* **File & Line Number:** `src/nged_substation_forecast/defs/plotting_assets.py` (Section 3.1)
* **The Theoretical Issue:** The plan fails to account for target normalization. If the ML model was trained on normalized targets (e.g., scaled by substation capacity or standard deviation), the raw predictions will be in a dimensionless, scaled space, whereas the actuals are in physical units (MW).
* **Concrete Failure Mode:** The plot will compare normalized predictions (e.g., values between -1 and 1) against raw actuals (e.g., 10 to 100 MW). This will result in a visually broken plot where the forecast appears as a flat line near zero, rendering it physically meaningless and impossible to evaluate.
* **Required Architectural Fix:** The Architect must explicitly add a step to inverse-transform (descale) the predictions back to their original physical units (MW) *before* plotting. This requires loading the scaler or capacity metadata used during training and applying the inverse transformation to the `predictions` dataframe.

## FLAW-002: Unverified Assumption of `nwp_init_time` in Predictions
* **File & Line Number:** `src/nged_substation_forecast/defs/plotting_assets.py` (Section 3.1.B)
* **The Theoretical Issue:** The plan assumes that the `predictions` dataframe contains an `nwp_init_time` column to filter on. Often, prediction outputs only contain the target `valid_time` and the predicted value, losing the initialization context.
* **Concrete Failure Mode:** If the upstream prediction asset does not explicitly include `nwp_init_time` in its output schema, the operation `predictions.filter(pl.col("nwp_init_time") == chosen_init_time)` will crash with a `ColumnNotFoundError`.
* **Required Architectural Fix:** The Architect must verify that the upstream prediction asset (e.g., `xgboost_predictions`) includes `nwp_init_time` in its output. If it does not, the plan must be updated to either modify the upstream asset to pass this column through, or join the predictions with the feature dataset to recover the `nwp_init_time` before filtering.
