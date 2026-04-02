---
status: "Superseded by ADR-016"
date: "2026-03-27"
author: "Jack & opencode (gemini-3.1-pro-preview)"
tags: ["mlops", "architecture", "interfaces", "polars"]
---

# ADR-002: Universal ML Interface with LocalForecasters Routing

## 1. Context & Problem Statement
We need to support various machine learning model architectures (e.g., XGBoost, PyTorch Geometric GNNs, simple baselines) for forecasting net demand at individual NGED substations. A monolithic approach where orchestration code knows the details of model training and inference creates tight coupling. We needed an abstraction layer that allows the orchestration system (Dagster) to interact uniformly with any model type, while still supporting the reality that we often train distinct "local" models for each individual substation.

## 2. Options Considered

### Option A: Direct Model Instantiation in Dagster Assets
* **Description:** Dagster assets directly instantiate `XGBoost` or `PyTorch` models and manage loops over substations.
* **Why it was rejected:** Highly couples orchestration logic with ML implementation details. Adding a new model type would require rewriting Dagster assets.

### Option B: A Single Giant Forecaster Class
* **Description:** One class that handles loops over substations and contains the `if/else` logic for all different model types.
* **Why it was rejected:** Violates the Open/Closed Principle. The class would grow infinitely and become unmaintainable as new model types are added.

### Option C: Abstract `BaseForecaster` with `LocalForecasters` Wrapper
* **Description:**
  1. Define a `BaseForecaster` abstract base class defining `fit()` and `predict()` methods operating on Polars DataFrames.
  2. Implement concrete model classes (e.g., `XGBoostForecaster`) inheriting from `BaseForecaster`.
  3. Create a `LocalForecasters` class (also adhering to a similar interface) that acts as a router/manager. It takes a configuration, instantiates the correct concrete model *per substation*, and handles the orchestration of training/predicting across multiple locations.

## 3. Decision
We chose **Option C: Abstract `BaseForecaster` with `LocalForecasters` Wrapper**. This enforces a strict separation of concerns. Dagster only interacts with `LocalForecasters`, which in turn manages collections of `BaseForecaster` implementations.

## 4. Consequences
* **Positive:** Adding a new model type (like a GNN) only requires creating a new class that implements `BaseForecaster`, without touching orchestration or routing logic.
* **Positive:** Clean, testable ML code isolated from infrastructure concerns.
* **Negative/Trade-offs:** Introduces an extra layer of abstraction (`LocalForecasters`) which slightly increases the initial cognitive load for new developers tracing the execution path.
