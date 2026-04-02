---
status: "Accepted"
date: "2026-04-02"
author: "Architect (gemini-3.1-pro-preview)"
tags: ["data-engineering", "mlops", "polars", "architecture"]
---

# ADR-014: Centralized Data Preparation for ML Models

## 1. Context & Problem Statement
During the development of the XGBoost forecasting pipeline, we noticed that data preparation logic (such as handling missing substations, enforcing data types, and filtering targets) was duplicated or scattered across different modules (`features.py`, `model.py`, `data.py`). This led to inconsistencies, potential bugs (e.g., missing substations causing crashes during inference), and made the code harder to maintain and test. We needed a unified approach to prepare data before it enters the ML model training and inference phases.

## 2. Options Considered

### Option A: Decentralized Preparation (Status Quo)
* **Description:** Each component (feature engineering, model training, inference) handles its own data preparation and validation.
* **Why it was rejected:** This violates the DRY (Don't Repeat Yourself) principle. It led to subtle bugs where inference data lacked certain substations present in training data, causing shape mismatches. It also made it difficult to enforce strict data contracts uniformly.

### Option B: Centralized Preparation in `ml_core`
* **Description:** Move all data preparation logic into a shared utility within the `ml_core` package, making it available to all model implementations.
* **Why it was rejected:** While better than Option A, different model architectures (e.g., XGBoost vs. GNNs) might require slightly different data preparation steps (e.g., handling missing values differently, or requiring specific tensor formats). A single centralized function in `ml_core` might become a bottleneck or overly complex as it tries to accommodate all model types.

### Option C: Centralized Preparation within Model Packages
* **Description:** Create a dedicated `prepare_data` function within each model package (e.g., `xgboost_forecaster.data.prepare_data`). This function acts as a gatekeeper, ensuring all data entering the model pipeline conforms to the required schema, handles missing entities, and enforces consistent types.

## 3. Decision
We chose **Option C: Centralized Preparation within Model Packages**. This approach provides a clear boundary for data entering the model pipeline. It ensures consistency within a specific model architecture while allowing flexibility across different architectures. The `prepare_data` function acts as a single point of truth for data cleaning and formatting before feature engineering and modeling.

## 4. Consequences
* **Positive:** Reduces code duplication, improves robustness by catching data issues early, and simplifies testing. It ensures that both training and inference pipelines use identically prepared data.
* **Negative/Trade-offs:** Requires developers to remember to call `prepare_data` at the start of their model pipelines. It adds a slight overhead to the data processing step, but this is negligible compared to the benefits of data consistency.
