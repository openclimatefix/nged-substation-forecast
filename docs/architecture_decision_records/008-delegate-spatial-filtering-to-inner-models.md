---
status: "Accepted"
date: "2026-03-27"
author: "Jack Kelly & gemini-3.1-pro-preview"
tags: ["mlops", "polars", "architecture", "spatial-modeling"]
---

# ADR-008: Delegate Spatial Filtering to Inner Models

## 1. Context & Problem Statement
We needed to decide where the spatial filtering logic for Numerical Weather Prediction (NWP) data should live when generating forecasts for individual substations. Previously, the `LocalForecasters` router class was eagerly filtering the NWP data to the exact `h3_res_5` hexagon of each substation before passing it to the inner `BaseForecaster`'s `predict` method. This created an asymmetry with the `train` method (which passed unfiltered `LazyFrame`s), forced eager evaluation, and actively prevented future models (like Graph Neural Networks or Convolutional models) from accessing wider spatial contexts (e.g., neighboring hexagons to track incoming weather fronts).

## 2. Options Considered

### Option A: Keep spatial filtering in `LocalForecasters` (Status Quo)
* **Description:** `LocalForecasters` filters NWP data to the exact H3 index of the substation before passing it to the inner model.
* **Why it was rejected:** This hardcodes the assumption that a model only cares about a single spatial point. It actively prevents the implementation of advanced spatial models (GNNs, CNNs) that require neighborhood context. It also forces redundant filtering, as inner models like `XGBoostForecaster` already perform their own spatial joins. Finally, it creates a dangerous asymmetry between the `train` and `predict` signatures.

### Option B: Parameterize spatial filtering in `LocalForecasters`
* **Description:** Add a parameter to `LocalForecasters` (e.g., `spatial_context_size` or `h3_k_ring`) to let the router know how much spatial data to filter for each model.
* **Why it was rejected:** This violates the separation of concerns. The router (`LocalForecasters`) should not need to know the internal feature engineering requirements of the models it manages. It unnecessarily complicates the orchestration layer and still forces eager evaluation before the model receives the data.

### Option C: Delegate spatial filtering to `BaseForecaster` and use `LazyFrame`s
* **Description:** `LocalForecasters` passes the full, unfiltered NWP `LazyFrame` to the inner models via `**kwargs`. The inner models are responsible for joining and filtering the data based on the provided substation metadata and their own spatial requirements.

## 3. Decision
We chose **Option C: Delegate spatial filtering to `BaseForecaster` and use `LazyFrame`s**, because it provides the optimal balance of encapsulation, performance, and future-proofing. By removing `nwps` from the explicit signature of `LocalForecasters.predict`, we make the router completely oblivious to weather data, allowing it to handle any global or spatial datasets via `**kwargs`. This leverages Polars' lazy evaluation engine to optimize the repeated query execution across thousands of local models.

## 4. Consequences
* **Positive:** The architecture is now future-proofed for advanced spatial models (like GNNs). Performance is improved by deferring execution until the inner model calls `.collect()`. The `train` and `predict` interfaces are now perfectly aligned, reducing the risk of training/inference skew.
* **Negative/Trade-offs:** Inner models now have slightly more responsibility (they must handle the spatial join themselves), which adds a small amount of boilerplate to new `BaseForecaster` implementations.
