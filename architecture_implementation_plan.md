# Architecture Implementation Plan: The "Tiny Wrapper" Pipeline

This document outlines the detailed plan for refactoring the MLOps architecture to use explicit, "tiny wrapper" Dagster assets that delegate to shared MLOps utilities. This design achieves a "Write-Once" downstream pipeline and eliminates boilerplate without relying on complex metaprogramming or Dagster factories.

## 🎯 Architectural Philosophy

1.  **Explicit Dagster Assets:** We write explicit `@asset def train_xgboost(...)` functions. These assets are "tiny wrappers" that simply declare dependencies and pass them as `**kwargs` to a shared utility.
2.  **Shared MLOps Utilities:** We extract universal boilerplate (temporal slicing, MLflow logging, Delta Lake saving, and dynamic partition triggering) into reusable functions in `ml_core`.
3.  **Pure ML Logic:** The actual model training and inference logic lives in pure Python classes (`Trainer` and `Forecaster`) that are completely ignorant of Dagster and MLflow. They just accept and return Polars DataFrames.
4.  **Native MLflow:** We use native MLflow flavors (e.g., `mlflow.xgboost.log_model`) instead of the generic `pyfunc` wrapper.

---

## 🛠️ Implementation Plan

### Phase 1: Create Shared MLOps Utilities (`ml_core/utils.py`)
We will create the universal functions that handle the boilerplate for all models.
1.  **Create `ml_core/utils.py`**.
2.  **Implement `train_and_log_model`**:
    -   Signature: `def train_and_log_model(context, model_name: str, trainer, config: dict, flavor: str, **kwargs)`
    -   Logic:
        -   Extract `train_start` and `train_end` from `config["data_split"]`.
        -   Temporally slice all `LazyFrames` in `**kwargs` (using `timestamp` or `valid_time` depending on the key name).
        -   Call `model = trainer.train(config, **sliced_kwargs)`.
        -   Start an MLflow run, log parameters, and save the model using the specified `flavor` (e.g., `mlflow.xgboost.log_model`).
        -   Add the `mlflow_run_id` to the Dagster context metadata.
        -   Return the trained model.
3.  **Implement `evaluate_and_save_model`**:
    -   Signature: `def evaluate_and_save_model(context, model_name: str, forecaster, config: dict, **kwargs)`
    -   Logic:
        -   Extract `test_start` and `test_end` from `config["data_split"]`.
        -   Temporally slice and `.collect()` all `LazyFrames` in `**kwargs`.
        -   Call `predictions_df = forecaster.predict(**sliced_kwargs)`.
        -   Add metadata columns (`power_fcst_model_name`, `power_fcst_init_time`, `power_fcst_init_year_month`, `nwp_init_time`).
        -   *(Placeholder)* Save `predictions_df` to Delta Lake.
        -   Call `context.instance.add_dynamic_partitions("model_partitions", [model_name])`.
        -   Return `predictions_df`.

### Phase 2: Refactor XGBoost Logic (`packages/xgboost_forecaster`)
We will strip Dagster and MLflow out of the XGBoost classes, making them pure ML logic.
1.  **Create `trainer.py`**:
    -   Implement `class XGBoostTrainer`.
    -   Implement `def train(self, config: dict, weather: pl.LazyFrame, scada: pl.LazyFrame)`.
    -   Move the joining, feature selection, and `XGBRegressor.fit()` logic here. Return the trained model.
2.  **Refactor `model.py`**:
    -   Ensure `XGBoostForecaster` inherits from `BaseForecaster`.
    -   Ensure `predict(self, weather_ecmwf_ens_0_25: pt.DataFrame[ProcessedNwp], **kwargs)` contains only the feature selection and `model.predict()` logic.

### Phase 3: Implement "Tiny Wrapper" Assets (`src/nged_substation_forecast/defs/xgb_assets.py`)
We will rewrite the Dagster assets to be extremely thin.
1.  **Refactor `train_xgboost`**:
    -   Signature: `@asset(ins={"weather": dg.AssetIn("ecmwf_ens_forecast"), "scada": dg.AssetIn("combined_actuals")}) def train_xgboost(context, weather: pl.LazyFrame, scada: pl.LazyFrame)`
    -   Logic: Load config, instantiate `XGBoostTrainer`, and `return train_and_log_model(...)`.
2.  **Refactor `evaluate_xgboost`**:
    -   Signature: `@asset(ins={"model": dg.AssetIn("train_xgboost"), "weather": dg.AssetIn("ecmwf_ens_forecast")}) def evaluate_xgboost(context, model, weather: pl.LazyFrame)`
    -   Logic: Load config, instantiate `XGBoostForecaster(model)`, and `return evaluate_and_save_model(...)`.

### Phase 4: Cleanup & Verification
1.  **Update `ml_core/__init__.py`**: Export the new utility functions.
2.  **Update Documentation**: Ensure READMEs reflect the new "Tiny Wrapper" and shared utility architecture.
3.  **Run Checks**: Execute `uv run ruff check . --fix`, `uv run ruff format .`, `uv run --all-packages ty check`, and `uv run --all-packages pytest` to ensure the new architecture is sound.
4.  **Verify Dagster**: Run `uv run dg dev` to ensure the DAG loads correctly and the dynamic partitions are recognized.
