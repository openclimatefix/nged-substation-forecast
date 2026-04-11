# Implementation Plan: Avoid Massive Join in Post-Processing

## Problem
The current `evaluate_and_save_model` function performs an inner join between the full `results_df` (containing all 51 ensemble members) and the `actuals_lf` (ground truth). This replicates the actual power value for every ensemble member, creating a massive DataFrame that causes OOM errors.

## Proposed Solution
Instead of joining everything upfront, we will:

1.  **Separate Metrics Calculation from Plotting Data:**
    *   **Metrics:** Calculate metrics (MAE, RMSE, nMAE) by aggregating the ensemble members *before* joining with actuals, or by joining only the necessary aggregated predictions.
    *   **Plotting:** Do not join the full ensemble predictions with actuals. Instead, save the predictions and actuals as separate Parquet files (or separate partitions in the same file).

2.  **Refactor `evaluate_and_save_model`:**
    *   **Step 1: Save Predictions and Actuals Separately:**
        *   Save `results_lf` to `predictions.parquet`.
        *   Save `actuals_lf` to `actuals.parquet`.
    *   **Step 2: Calculate Metrics Efficiently:**
        *   If metrics are needed, calculate them by aggregating `results_lf` first (e.g., mean prediction per lead time) and then joining with `actuals_lf`. This avoids the 51x replication.
    *   **Step 3: Update Plotting Logic:**
        *   The plotting op should read `predictions.parquet` and `actuals.parquet` separately and perform the join only for the specific substation/time range it is plotting.

## Benefits
*   **Memory Efficiency:** Eliminates the massive join that replicates actuals 51 times.
*   **Scalability:** Allows handling much larger ensemble sizes and longer forecast horizons.
*   **Flexibility:** Decouples metric calculation from plotting, making both more efficient.
