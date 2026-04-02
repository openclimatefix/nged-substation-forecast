---
status: "Accepted"
date: "2026-03-27"
author: "Jack & opencode (gemini-3.1-pro-preview)"
tags: ["architecture", "dependencies", "contracts", "mlops"]
---

# ADR-006: Dependency Isolation for Data Contracts

## 1. Context & Problem Statement

We needed a way to define the shape and semantics of our data sources (e.g., `SubstationPowerFlows`, `ProcessedNwp`, `PowerForecast`) so that any component in the system could validate incoming data and understand its structure.

Initially, these schemas might have lived alongside the machine learning code that consumes them. However, this creates a tight coupling between data definition and heavy ML dependencies (like MLflow, PyTorch, XGBoost). If a simple data ingestion script or a frontend dashboard needed to know what a `SubstationPowerFlows` dataframe looked like, it would be forced to install the entire ML stack.

## 2. Options Considered

### Option A: Monolithic Package
* **Description:** All data schemas, ML interfaces, and orchestration logic live in a single Python package.
* **Why it was rejected:** Leads to "dependency bloat". Lightweight scripts and APIs would require installing heavy ML libraries, slowing down deployments and increasing the risk of dependency conflicts.

### Option B: Strict Package Separation (`contracts` vs `ml_core`)
* **Description:** Create a lightweight `contracts` package for data schemas and a separate `ml_core` package for ML machinery.

## 3. Decision

We chose **Option B: Strict Package Separation**.

We maintain a strict separation between `contracts` and `ml_core` within our `uv` workspace:
1. **`packages/contracts`**: A lightweight package defining data schemas (using Patito/Polars) and project settings (using Pydantic). This package has minimal dependencies to ensure it can be used by any component.
2. **`packages/ml_core`**: A package containing the unified ML model interface (`BaseForecaster`) and shared utilities (`train_and_log_model`). It depends on `mlflow-skinny` and other ML-related libraries.

## 4. Consequences

* **Positive**: Any component in the system (e.g., a simple data ingestion script, a frontend dashboard, or a lightweight API) can import the data schemas from `contracts` without bringing in the entire ML stack.
* **Positive**: Clear architectural boundaries between data engineering (defining the *shape* of the data) and machine learning (defining the *machinery* for ML).
* **Negative/Trade-offs**: Requires managing multiple packages within the `uv` workspace, slightly increasing the complexity of the repository structure and requiring developers to understand which package to import from.
