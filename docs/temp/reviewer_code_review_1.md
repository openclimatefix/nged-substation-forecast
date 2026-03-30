---
review_iteration: 1
reviewer: "review"
total_flaws: 8
critical_flaws: 1
---

# Code Review

## FLAW-001: Duplicated Scaling Logic
* **File & Line Number:** `packages/ml_core/src/ml_core/scaling.py`, lines 8-27 and `packages/xgboost_forecaster/src/xgboost_forecaster/scaling.py`, lines 24-43
* **The Issue:** The `uint8_to_physical_unit` function is duplicated across two packages. The version in `ml_core` is more robust as it uses Patito type hints.
* **Concrete Failure Mode:** Maintenance burden and potential for logic divergence. If the scaling logic needs to change, it must be updated in two places.
* **Required Fix:** Remove the implementation in `xgboost_forecaster` and import it from `ml_core`.

## FLAW-002: Hardcoded and Insufficient Lookback in Evaluation
* **File & Line Number:** `packages/ml_core/src/ml_core/utils.py`, lines 107-113
* **The Issue:** `evaluate_and_save_model` uses a hardcoded 14-day lookback for power flows and NWPs. However, `XGBoostForecaster` calculates lags dynamically, and for large lead times, it may require more than 14 days of history (e.g., `ceil(15/7)*7 = 21` days).
* **Concrete Failure Mode:** Models requiring longer lags will receive null values for those features during evaluation, leading to `ValueError` or degraded performance.
* **Required Fix:** Make the lookback period configurable in `TrainingConfig` or allow the forecaster to specify its required lookback.

## FLAW-003: Inefficient CSV Loading in Feature Engineering
* **File & Line Number:** `packages/xgboost_forecaster/src/xgboost_forecaster/features.py`, line 25 and `packages/xgboost_forecaster/src/xgboost_forecaster/scaling.py`, lines 9-21
* **The Issue:** `load_scaling_params` reads a CSV file from disk every time it is called. It is called inside `add_physical_features`, which is called twice for every NWP processed in `add_weather_features`.
* **Concrete Failure Mode:** Significant I/O overhead during training and inference, especially when using multiple NWPs or processing many substations.
* **Required Fix:** Cache the scaling parameters (e.g., using `functools.lru_cache`) or load them once at the start of the training/inference run and pass them down.

## FLAW-004: Tight Coupling of Configuration to XGBoost
* **File & Line Number:** `packages/contracts/src/contracts/hydra_schemas.py`, lines 40-54
* **The Issue:** `ModelConfig` explicitly includes `XGBoostHyperparameters`. This violates the goal of a "Universal ML Interface" (ADR-002/ADR-009) as it forces all models to use XGBoost-specific configuration structures.
* **Concrete Failure Mode:** Adding a new model type (e.g., a GNN) requires modifying the core `contracts` package and potentially breaking existing XGBoost configurations.
* **Required Fix:** Use a more generic hyperparameter structure (e.g., a `dict[str, Any]` or a Pydantic Union of model-specific hyperparameter classes).

## FLAW-005: Redundant/Confusing `LocalForecasters` Implementation
* **File & Line Number:** `packages/ml_core/src/ml_core/model.py`, lines 86-202
* **The Issue:** `LocalForecasters` is implemented and emphasized in ADRs, but the project has shifted to a "unified global model" (as per the summary and `xgb_assets.py`). `LocalForecasters` is currently unused in production assets and contains an unimplemented `log_model` method.
* **Concrete Failure Mode:** Increased cognitive load for developers and potential for "dead code" to accumulate.
* **Required Fix:** Either use `LocalForecasters` if per-substation models are still desired, or move it to a "research" or "legacy" area if the global model is the definitive direction. At minimum, implement `log_model` or clearly mark the class as experimental.

## FLAW-006: Missing Dependency Declarations in `pyproject.toml`
* **File & Line Number:** `packages/ml_core/pyproject.toml` and `packages/xgboost_forecaster/pyproject.toml`
* **The Issue:** `ml_core` imports from `contracts`, and `xgboost_forecaster` imports from `ml_core`, but these internal dependencies are not listed in their respective `pyproject.toml` files.
* **Concrete Failure Mode:** `uv sync` or `pip install` might fail in isolated environments or CI/CD pipelines where the full workspace is not present.
* **Required Fix:** Add `contracts` to `ml_core` dependencies and `ml_core` to `xgboost_forecaster` dependencies.

## FLAW-007: Expensive `collect_schema()` Calls in Loops
* **File & Line Number:** `packages/xgboost_forecaster/src/xgboost_forecaster/features.py`, lines 29, 42, 43, 70, 74, 76, 93, 95, 110, 112
* **The Issue:** `collect_schema()` is called repeatedly on LazyFrames within feature engineering functions. While faster than `collect()`, it still involves overhead that accumulates when processing multiple NWPs or in iterative loops.
* **Concrete Failure Mode:** Unnecessary CPU overhead and slightly slower execution times for the feature engineering pipeline.
* **Required Fix:** Call `collect_schema()` once at the beginning of the function and use the resulting schema object for subsequent checks.

## FLAW-008: CRITICAL: Potential Data Leakage in `add_weather_features`
* **File & Line Number:** `packages/xgboost_forecaster/src/xgboost_forecaster/features.py`, lines 80-83
* **The Issue:** When `history` is `None`, it uses `weather.sort("init_time").group_by(group_cols).last()`. If `weather` contains future `init_time`s (which it might during training if not carefully sliced), this could pick the "latest" forecast which might not have been available at the time of the target power flow.
* **Concrete Failure Mode:** Over-optimistic backtesting results due to lookahead bias.
* **Required Fix:** Ensure that `weather` is strictly filtered by `available_time` (init_time + 3h) relative to the `valid_time` of the target being predicted, even when constructing the "history" for weather lags.
