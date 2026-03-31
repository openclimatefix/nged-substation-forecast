---
review_iteration: 8
reviewer: "scientist"
total_flaws: 2
critical_flaws: 1
---

# ML Rigor Review: Ensemble Variance Deep Dive

I have conducted a rigorous audit of the XGBoost pipeline to investigate why the 51 ensemble members produce suspiciously similar power forecasts. The NWP data itself *does* contain variance across ensemble members, but this variance is being drowned out by a combination of feature dominance, model underfitting, and a subtle meteorological bug in the feature engineering.

## FLAW-001: Feature Dominance & Severe Underfitting Drowning Out Weather Variance
* **File & Line Number:** `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`, lines 65-75 (in `_prepare_features`) and `conf/model/xgboost.yaml`
* **The Theoretical Issue:** The model includes `substation_number` as a categorical feature. Because the model is severely capacity-constrained (`n_estimators=100`, `max_depth=6`, `learning_rate=0.01`), it greedily uses almost all its tree splits to learn the mean normalized load for each substation (resulting in the 0.77 feature importance). It runs out of capacity before it can learn the subtle weather sensitivities.
* **Concrete Failure Mode:** Since `substation_number` and `latest_available_weekly_lag` are identical across all 51 ensemble members, and they dominate the model's predictions, the output forecasts for all 51 members are nearly identical. The weather features (which contain the actual ensemble variance) are effectively ignored, destroying the probabilistic value of the ensemble.
* **Required Architectural Fix:**
  1. The Architect must add `"substation_number"` to the `exclude_cols` set in `_prepare_features`. The model should rely on `latest_available_weekly_lag` to establish the substation baseline, forcing it to use weather features for the residuals. This makes it a true generalizable global model.
  2. Increase model capacity in `conf/model/xgboost.yaml` (e.g., `n_estimators: 1000`, `learning_rate: 0.05`) so it has enough trees to fit the weather effects after establishing the baseline.

## FLAW-002: Meteorological Inconsistency in Weather Lags
* **File & Line Number:** `packages/xgboost_forecaster/src/xgboost_forecaster/features.py`, lines 120-130 (in `_add_lag_asof`)
* **The Theoretical Issue:** The function joins historical weather lags (`lag_7d`, `lag_14d`) by matching `ensemble_member`. However, in NWP models like ECMWF, ensemble perturbations are drawn randomly and independently for each `init_time`. Ensemble member 5 from today has no physical relationship to ensemble member 5 from 7 days ago.
* **Concrete Failure Mode:** For `ensemble_member > 0`, the `lag_7d` and `lag_14d` features contain random noise rather than a consistent weather baseline. Since the model is trained only on the control member (`ensemble_member == 0`, which *is* consistent across time), it expects a meaningful lag but receives noise during prediction, degrading the forecast quality for the perturbed members.
* **Required Architectural Fix:** The Architect must modify `_add_lag_asof` so that for cross-run lags (`lag_7d`, `lag_14d`), it always fetches the historical lag from the control member (`ensemble_member == 0`). For intra-run lags (`6h_ago`), matching by `ensemble_member` is correct and should be preserved.
