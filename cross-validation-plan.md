# Phase 2: Dynamic Time Frames & Cross-Validation

## Motivation
Currently, the XGBoost training asset (`xgb_models` in `src/nged_substation_forecast/defs/xgb_assets.py`) relies on hardcoded training dates (`2026-02-01` to `2026-02-28`). 

To mature this pipeline for production and rigorous ML research, we need to make these time frames dynamic. This will unlock two critical capabilities:
1. **Production Retraining**: The ability to automatically train on "all historical data up to 1 week ago" without changing code.
2. **Time Series Cross-Validation**: The ability to rigorously backtest the model using an "expanding window" approach (e.g., train on Jan, test on Feb; train on Jan+Feb, test on Mar; etc.), which prevents data leakage and provides a realistic estimate of production performance.

## Implementation Steps

### Step 1: Refactor the Training Loop
**Target:** `src/nged_substation_forecast/defs/xgb_assets.py`
There is currently a `TODO` inside the `xgb_models` asset to extract the `try/except` block that trains individual substation models. 
*   **Action:** Extract this logic into a standalone, pure Python function (e.g., `train_local_xgboost_model(substation_number, df, settings)`).
*   **Why:** This makes the code unit-testable and allows the training logic to be easily reused by both the standard Dagster asset and the new Cross-Validation Job.

### Step 2: Introduce Dagster Configuration
**Target:** `src/nged_substation_forecast/defs/xgb_assets.py`
*   **Action:** Create a new class `XGBoostTrainingConfig(dg.Config)`.
*   **Fields:**
    *   `train_start_date: str | None = None`
    *   `train_end_date: str | None = None`
    *   `test_end_date: str | None = None` (Used to define the evaluation set for early stopping/metrics).
*   **Action:** Update the `xgb_models` asset signature to accept this config: `def xgb_models(context, config: XGBoostTrainingConfig, settings: ...)`

### Step 3: Implement Default "Production" Date Logic
**Target:** `src/nged_substation_forecast/defs/xgb_assets.py`
*   **Action:** Inside the `xgb_models` asset, implement logic to handle `None` values in the config.
*   **Logic:** If `train_end_date` is not provided, default it to `today - 7 days` (to ensure we only train on fully settled/verified data). If `train_start_date` is not provided, default it to the earliest available data (or a sensible default like `train_end_date - 1 year`).
*   **Action:** Pass these resolved dates into the `prepare_training_data` function.

### Step 4: Build the Cross-Validation Orchestrator
**Target:** `src/nged_substation_forecast/defs/xgb_assets.py` (or a new `xgb_jobs.py` file)
*   **Action:** Create a Dagster `@op` called `generate_expanding_windows` that takes a start date, end date, and fold size (e.g., 1 month), and yields a list of `XGBoostTrainingConfig` objects representing the expanding windows.
*   **Action:** Create a Dagster `@op` called `train_cv_fold` that takes a `XGBoostTrainingConfig`, calls the refactored `train_local_xgboost_model` function, and logs the results to MLflow. 
    *   *Crucial MLflow detail:* Wrap this in a nested MLflow run (`with mlflow.start_run(nested=True):`) so all folds for a single CV experiment are grouped together in the MLflow UI.
*   **Action:** Create a Dagster `@job` called `xgboost_cross_validation_job` that uses Dagster's `DynamicOut` to map the generated windows to the training op in parallel.

### Step 5: Clean Up & Test
*   **Action:** Ensure all new functions have Google-style docstrings and strict type hints.
*   **Action:** Run `uv run ruff check . --fix`, `uv run ruff format .`, and `uv run --all-packages ty check`.
*   **Action:** Write unit tests for the new date-resolution logic and the expanding window generator. Run `uv run --all-packages pytest`.
