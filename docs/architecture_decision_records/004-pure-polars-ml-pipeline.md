---
status: "Accepted"
date: "2026-03-27"
author: "Jack & opencode (gemini-3.1-pro-preview)"
tags: ["polars", "performance", "xgboost", "pandas"]
---

# ADR-004: Pure Polars ML Pipeline for Zero-Copy Memory

## 1. Context & Problem Statement
Historically, many Python ML pipelines default to using Pandas DataFrames. However, our data layer and upstream processing heavily utilize Polars for its speed and multithreading capabilities. Passing data into ML models (specifically XGBoost) often involved implicit or explicit conversions to Pandas or NumPy, causing memory bloat, unnecessary data copying, and performance bottlenecks, particularly when dealing with high-resolution substation data across many locations.

## 2. Options Considered

### Option A: Polars for ETL, Pandas for ML
* **Description:** Keep Polars for data loading and feature engineering, but call `.to_pandas()` right before passing data to `BaseForecaster` or `xgboost.train()`.
* **Why it was rejected:** The `.to_pandas()` conversion is computationally expensive and memory-intensive, negating many of the performance benefits gained from using Polars in the ETL phase.

### Option B: Conversion to NumPy Arrays
* **Description:** Convert Polars DataFrames to NumPy arrays (`.to_numpy()`) for ML training.
* **Why it was rejected:** While better than Pandas, it strips away column names and schema information, making feature importance tracking and debugging much more difficult.

### Option C: Pure Polars pipeline via Arrow integration
* **Description:** Strictly enforce that the `BaseForecaster` interface (`fit` and `predict` methods) accepts and returns *only* `polars.DataFrame`. Leverage the fact that modern ML libraries like XGBoost (and LightGBM) natively support Arrow/Polars data structures, allowing for zero-copy memory transfers.

## 3. Decision
We chose **Option C: Pure Polars pipeline via Arrow integration**. We are eliminating Pandas dependencies in the ML training and inference code. All data passed through the `BaseForecaster` and `LocalForecasters` boundaries must be `polars.DataFrame` objects.

## 4. Consequences
* **Positive:** Significant reduction in memory usage and execution time due to zero-copy memory transfers between Polars and XGBoost via Apache Arrow.
* **Positive:** Consistent data structures (Polars) from the data loading step all the way through to inference output.
* **Negative/Trade-offs:** Developers must ensure they do not accidentally introduce Pandas operations. Certain niche ML libraries might not support Polars natively, requiring future workarounds if those specific libraries are adopted, though the major ones (XGBoost, PyTorch) are well-supported.
