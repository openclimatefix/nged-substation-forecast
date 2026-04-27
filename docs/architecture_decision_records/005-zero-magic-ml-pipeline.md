---
status: "Accepted"
date: "2026-03-27"
author: "Jack & opencode (gemini-3.1-pro-preview)"
tags: ["mlops", "architecture", "dagster", "mlflow", "pythonic"]
---

# ADR-005: The "Zero Magic" Pythonic ML Pipeline

## 1. Context & Problem Statement

We needed a way to orchestrate the training and evaluation of multiple ML architectures (e.g., XGBoost, GNNs) using Dagster and MLflow.

Initially, we considered a highly abstracted "Write-Once" pipeline using Dagster Asset Factories, Pydantic data wrappers, and MLflow's generic `pyfunc` interface. This would allow a single `train` and `evaluate` asset definition to handle any model dynamically based on configuration.

However, this "magical" approach introduced significant cognitive load, disconnected the Dagster UI lineage (making it hard to trace dependencies), and forced us to use undocumented MLflow hacks (`_skip_type_hint_validation`) to maintain type safety. It also violated the "grep-ability" of the codebase, as assets like `train_xgboost` were generated at runtime and didn't exist in the source code.

## 2. Options Considered

### Option A: The "Dagster Factory" Approach
* **Description:** A single factory function reads a model's Python-defined data requirements and dynamically generates Dagster `@asset` definitions. Models are wrapped in `mlflow.pyfunc.PythonModel` and Pydantic classes to enforce schemas.
* **Why it was rejected:** Too much "magic". It is difficult for new developers to understand, requires undocumented MLflow workarounds, and serializes heavy Python classes instead of native model artifacts.

### Option B: The "God Signature" Partitioned Asset
* **Description:** A single `@asset(partitions_def=model_partitions) def train_model(...)` that lists *every possible dataset* any model might ever need in its signature.
* **Why it was rejected:** Breaks the "Write-Once" philosophy. Adding a new dataset for a specific model requires modifying the core orchestration signature, and Dagster would inefficiently load unused data into memory for models that don't need it.

### Option C: The "Zero Magic" Pythonic Architecture (Tiny Wrappers + Shared Utilities)
* **Description:** Developers write explicit, tiny Dagster assets for each model (e.g., `train_xgboost`). These assets declare their specific dependencies and immediately delegate to shared MLOps utility functions. The ML logic lives in pure Python classes with standard type hints.

## 3. Decision

We chose **Option C: The "Zero Magic" Pythonic Architecture**.

Specifically, we implemented the following design patterns:
1. **Explicit Dagster Assets**: We write explicit, "tiny wrapper" assets for each ML architecture. This makes the DAG instantly understandable and grep-able.
2. **Shared MLOps Utilities**: To prevent boilerplate, universal tasks (temporal slicing, MLflow logging, Delta Lake saving) are extracted into shared functions (`train_and_log_model`, `evaluate_and_save_model`) in a dedicated `ml_core` package.
3. **Native MLflow Flavors**: We abandoned the generic `mlflow.pyfunc` wrapper in favor of native flavors (e.g., `mlflow.xgboost.log_model`), allowing us to serialize only the raw model object.
4. **Pure Python ML Interfaces**: We merged the `Trainer` and `Forecaster` into a single `BaseForecaster` class per architecture. We rely on standard Python function signatures and Patito type hints (e.g., `def predict(self, nwp: pt.DataFrame[ProcessedNwp])`) for perfect IDE autocomplete and runtime validation, rather than complex Pydantic wrappers.
5. **Automated Downstream Pipeline**: Model-agnostic downstream assets (`metrics`, `plot`) are implemented as single, partitioned assets. They use `AutoMaterializePolicy.eager()` to run automatically when an `evaluate_*` asset finishes and adds a dynamic partition key (`model_partitions`).

## 4. Consequences

* **Positive**: The codebase is highly readable, grep-able, and approachable for developers new to Dagster and MLflow.
* **Positive**: Perfect IDE type hinting and native framework serialization without hacks.
* **Positive**: The downstream analysis pipeline remains "Write-Once" and fully automated via Dagster's daemon.
* **Positive**: Feature engineering logic can be easily shared between training and inference within a single `BaseForecaster` class.
* **Negative/Trade-offs**: The Dagster UI will have more asset boxes (one train/evaluate pair per ML architecture) rather than a single partitioned box, but they are logically grouped and explicitly show their unique data dependencies.
