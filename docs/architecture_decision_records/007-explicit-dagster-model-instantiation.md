---
status: "Accepted"
date: "2026-03-27"
author: "Software Architect & Machine Learning Engineer (gemini-3.1-pro-preview)"
tags: ["orchestration", "mlops", "dagster", "hydra", "configuration"]
---

# ADR-003: Explicit Dagster Model Instantiation over Dynamic Hydra Instantiation

## 1. Context & Problem Statement
We needed to clarify the roles of `power_fcst_model_name` and `trainer_class` in the Hydra configuration schemas (`ModelConfig`). Specifically, we needed to determine if they were redundant, if they should be Enums, and how they should be used to manage different machine learning experiments (e.g., various XGBoost configurations) versus different model architectures (e.g., XGBoost vs. PyTorch GNNs). The `trainer_class` field was present in the configuration but unused in the codebase, as the `XGBoostForecaster` was explicitly instantiated in the Dagster assets.

## 2. Options Considered

### Option A: Use Enums for `power_fcst_model_name` and `trainer_class`
* **Description:** Restrict `power_fcst_model_name` and `trainer_class` to predefined Enums in the Pydantic schemas.
* **Why it was rejected:** Using an Enum for `power_fcst_model_name` is an anti-pattern for experiment tracking. Data scientists frequently run dozens of experiments using the same algorithm (e.g., `"xgboost_baseline"`, `"xgboost_with_solar_features"`). Requiring a code change to add a new Enum value for every experiment would severely hinder the experimentation workflow. It must remain a free-text string to be purely configuration-driven. Similarly, if `trainer_class` were used, an Enum would require hardcoding a registry of all possible models, reducing flexibility.

### Option B: The "Factory" Approach (Dynamic Instantiation via Hydra)
* **Description:** Retain the `trainer_class` field in the configuration (e.g., `"xgboost_forecaster.model.XGBoostForecaster"`) and use `hydra.utils.instantiate` within a generic Dagster asset (`model_assets.py`) to dynamically import and build the model class.
* **Why it was rejected:** While this approach is DRY (Don't Repeat Yourself) and allows adding new model architectures purely via YAML configuration, it obscures the pipeline in the Dagster UI. Explicitly defining assets for different model types (e.g., `train_xgboost`, `train_pytorch`) provides better observability and makes it easier to track the execution and dependencies of specific model pipelines visually.

### Option C: The "Explicit Dagster" Approach (Remove `trainer_class`)
* **Description:** Remove the unused `trainer_class` from the configuration schemas and YAML files. Continue to explicitly instantiate the specific model classes (e.g., `XGBoostForecaster()`) within their respective Dagster assets (e.g., `train_xgboost`). Retain `power_fcst_model_name` as a free-text string for experiment tracking.

## 3. Decision
We chose **Option C: The "Explicit Dagster" Approach**, because it provides the optimal balance of pipeline observability in the Dagster UI and flexibility for experiment tracking. We removed the dead `trainer_class` configuration and added clear documentation to `power_fcst_model_name` explaining its purpose as a unique identifier for model configurations and MLflow runs.

## 4. Consequences
* **Positive:** The Dagster UI remains explicit, showing distinct assets for different model architectures (e.g., `train_xgboost`), which improves observability and debugging.
* **Positive:** Data scientists can freely name their experiments using `power_fcst_model_name` in the YAML configuration without modifying Python code.
* **Negative/Trade-offs:** Adding a completely new model architecture (e.g., a PyTorch GNN) requires writing new Dagster assets (e.g., `train_pytorch`, `evaluate_pytorch`) rather than just adding a YAML file, resulting in slightly more boilerplate code.
