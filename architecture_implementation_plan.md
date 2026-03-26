# Architecture Implementation Plan: The Pythonic "Zero Magic" Pipeline

This document outlines the detailed, step-by-step plan for refactoring the MLOps architecture from a highly abstracted, factory-driven approach to a clean, explicit, Pythonic architecture. This new design prioritizes readability, explicit dependencies, and native framework integrations while maintaining strict data contracts and a "Write-Once" downstream analysis pipeline.

## 🎯 Architectural Philosophy

1.  **Zero Dagster Magic:** No asset factories. We write explicit `@asset def train_xgboost(...)` and `@asset def evaluate_xgboost(...)` functions. This makes the DAG instantly understandable and grep-able.
2.  **Native MLflow:** We abandon the generic `mlflow.pyfunc` wrapper and its rigid dictionary-based signature. We use native MLflow flavors (e.g., `mlflow.xgboost.log_model`) for robust serialization.
3.  **Pythonic Data Contracts:** We delete Pydantic data requirement wrappers. We rely on standard Python function signatures and Patito type hints (e.g., `def predict(self, weather: pt.DataFrame[ProcessedNwp])`) for perfect IDE autocomplete and runtime validation.
4.  **Write-Once Downstream:** The `metrics` and `plot` assets remain completely model-agnostic. They are single, partitioned assets that wake up automatically when an evaluation asset finishes and adds a dynamic partition key.

---

## 🛠️ Implementation Plan

### Phase 1: Simplify `ml_core` (The Foundation)
We will strip away the complex Pydantic and MLflow pyfunc abstractions.
1.  **Delete `ml_core/trainer.py`**: We no longer need `BaseDataRequirements`, `DataRequirementsMixin`, or `BaseTrainer`. The Dagster asset itself will handle dependency injection.
2.  **Refactor `ml_core/model.py`**:
    -   Remove `BaseInferenceModel` and all MLflow/Pydantic dependencies.
    -   Create a radically simple `BaseForecaster(ABC)` class with a single abstract method: `def predict(self, **kwargs) -> pt.DataFrame[PowerForecast]`.
3.  **Update `ml_core/__init__.py`**: Ensure it only exports `BaseForecaster` and `FeatureAsset`.

### Phase 2: Refactor XGBoost Implementation (`packages/xgboost_forecaster`)
We will update the XGBoost code to use native MLflow and the new, simple interface.
1.  **Delete `trainer.py`**: The training logic will move directly into the Dagster asset (or a shared utility function if we want to reuse the exact XGBoost training loop later).
2.  **Refactor `model.py`**:
    -   Rename `XGBoostInferenceModel` to `XGBoostForecaster` (inheriting from `BaseForecaster`).
    -   Update the `predict` signature to explicitly require the data it needs: `def predict(self, weather_ecmwf_ens_0_25: pt.DataFrame[ProcessedNwp], **kwargs) -> pt.DataFrame[PowerForecast]`.
    -   Implement the prediction logic directly on the Patito DataFrame.

### Phase 3: Explicit Dagster Assets (`src/nged_substation_forecast/defs/`)
We will replace the "magic" factory with explicit, easy-to-read assets.
1.  **Delete `model_assets.py`**.
2.  **Create `xgb_assets.py`**:
    -   Implement `@asset def train_xgboost(context, weather_ecmwf_ens_0_25, substation_power_flows)`:
        -   Load Hydra config.
        -   Filter data temporally.
        -   Train the model.
        -   Log using `mlflow.xgboost.log_model`.
    -   Implement `@asset def evaluate_xgboost(context, train_xgboost, weather_ecmwf_ens_0_25)`:
        -   Load model using `mlflow.xgboost.load_model`.
        -   Wrap in `XGBoostForecaster`.
        -   Call `.predict(weather_ecmwf_ens_0_25=...)`.
        -   Save results to Delta Lake (`data/evaluation_results.delta`).
        -   **Crucial Step:** Call `context.instance.add_dynamic_partitions("model_partitions", ["xgboost_baseline"])`.

### Phase 4: The "Write-Once" Downstream Pipeline (Metrics & Plotting)
We will configure the downstream assets to run automatically when a new partition is added.
1.  **Define Dynamic Partitions**: In a central definitions file (e.g., `definitions.py` or a new `partitions.py`), define `model_partitions = dg.DynamicPartitionsDefinition(name="model_partitions")`.
2.  **Refactor `metrics_assets.py`**:
    -   Remove the `create_metrics_asset` factory.
    -   Create a single `@asset(partitions_def=model_partitions)` named `metrics`.
    -   Add `auto_materialize_policy=dg.AutoMaterializePolicy.eager()` so it runs automatically when `evaluate_xgboost` adds a partition key.
    -   Update the logic to read the specific model's forecasts from the Delta Lake table using `context.partition_key`.
3.  **Refactor `plotting_assets.py`**:
    -   Apply the exact same partition and auto-materialize logic as the `metrics` asset.

### Phase 5: Cleanup & Verification
1.  **Remove Unused Code**: Delete `test_mixin.py` and any other remnants of the factory architecture.
2.  **Update Documentation**: Revise `ml_core/README.md` and the root `README.md` to reflect the new "Explicit Assets + Shared Utilities" philosophy.
3.  **Run Checks**: Execute `uv run ruff check . --fix`, `uv run ruff format .`, `uv run --all-packages ty check`, and `uv run --all-packages pytest` to ensure the new architecture is sound.
