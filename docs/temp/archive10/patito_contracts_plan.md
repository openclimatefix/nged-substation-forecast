---
status: "draft"
version: "v1.0"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: true
target_modules: ["src", "packages"]
---
# Plan: Map Remaining Functions to Patito Data Contracts

## Overview
This plan categorizes the remaining functions that use plain `pl.DataFrame` or `pl.LazyFrame` and proposes mapping them to existing Patito data contracts or creating new ones. This ensures type safety and explicit data schemas across the codebase.

## Rationale
Explicit data contracts improve code readability, maintainability, and robustness. By replacing generic Polars types with Patito contracts, we document the expected schema at function boundaries.

**Important Note on Code Comments:**
When implementing these changes, ensure that any added or modified comments focus on the *why* (intent and rationale) rather than the *how*. Comments should "connect the dots" across the codebase, explaining how these contracts enforce data integrity between different modules.

## Categorization and Mapping

### 1. Feature Engineering & Model Inputs/Outputs (`xgboost_forecaster`, `ml_core`)
These functions handle the core ML pipeline, joining NWP data, power flows, and metadata.

*   **`predict` and `fit` (in `ml_core/model.py`, `ml_core/experimental.py`, `xgboost_forecaster/model.py`)**
    *   **Inputs:** `power_time_series: pt.LazyFrame[PowerTimeSeries]`, `nwps: Mapping[NwpModel, pt.LazyFrame[Nwp]] | None`
    *   **Output (predict):** `pt.LazyFrame[PowerForecast]`
*   **`xgboost_predictions` and `xgboost_model` (`src/nged_substation_forecast/defs/xgb_assets.py`)**
    *   **Inputs:** `nwp: pt.LazyFrame[Nwp]`, `power_time_series: pt.LazyFrame[PowerTimeSeries]`
    *   **Output (xgboost_predictions):** `pt.DataFrame[PowerForecast]`
*   **`train_xgboost_model_job` (`src/nged_substation_forecast/defs/xgb_jobs.py`)**
    *   **Inputs:** `nwp: pt.LazyFrame[Nwp]`, `power_time_series: pt.LazyFrame[PowerTimeSeries]`
*   **Intermediate Feature Functions (`xgboost_forecaster/features.py`, `xgboost_forecaster/data.py`, `xgboost_forecaster/model.py`)**
    *   *Recommendation:* Map inputs to existing contracts (`ProcessedNwp`, `PowerTimeSeries`, `TimeSeriesMetadata`, `XGBoostInputFeatures`) where applicable. For outputs that are intermediate and don't match a specific contract (e.g., adding a single column before the final join), keep `pl.LazyFrame` but add clear docstrings explaining the intermediate state.

### 2. Data Ingestion & Cleaning
*   **`ingest_data` (`src/nged_substation_forecast/ingestion/helpers.py`)**
    *   **Input:** `ingested_data_df: pt.DataFrame[PowerTimeSeries] | None`
    *   **Output:** `pt.DataFrame[PowerTimeSeries]`
*   **`cleaned_actuals` and `cleaned_power_time_series` (`src/nged_substation_forecast/defs/data_cleaning_assets.py`, `cleaned_power_time_series_assets.py`)**
    *   **Output:** `pt.DataFrame[PowerTimeSeries]`
*   **`clean_substation_flows` (`src/nged_substation_forecast/cleaning.py`)**
    *   *Recommendation:* Keep as `pl.DataFrame`. This function operates on raw data (with `MW` and `MVA` columns) before it is validated into the `PowerTimeSeries` contract (which only has a `power` column).

### 3. Metrics & Plotting
*   **`xgboost_metrics` (`src/nged_substation_forecast/defs/metrics_assets.py`)**
    *   **Inputs:** `cleaned_actuals: pt.DataFrame[PowerTimeSeries]`, `predictions: pt.DataFrame[PowerForecast]`
    *   **Output:** `pt.DataFrame[Metrics]` (Requires new contract, see below).
*   **`forecast_plots` (`src/nged_substation_forecast/defs/plotting_assets.py`)**
    *   **Inputs:** `predictions: pt.DataFrame[PowerForecast]`, `time_series_metadata: pt.DataFrame[TimeSeriesMetadata]`

### 4. Geospatial & Weather
*   **`gb_h3_grid_weights` (`packages/geo/src/geo/assets.py`)**
    *   **Output:** `pt.DataFrame[H3GridWeights]`
*   **`all_nwp_data` (`src/nged_substation_forecast/defs/weather_assets.py`)**
    *   **Input:** `gb_h3_grid_weights: pt.DataFrame[H3GridWeights]`
    *   **Output:** `pt.LazyFrame[Nwp]`
*   **`create_h3_grid` (`packages/geo/src/geo/h3.py`)**
    *   *Recommendation:* Keep input `df` as `pl.DataFrame` (raw boundary data). Output is already `pt.DataFrame[H3GridWeights]`.

### 5. Dynamical Data
*   **`download_and_scale_ecmwf` (`packages/dynamical_data/src/dynamical_data/processing.py`)**
    *   *Recommendation:* Keep return type as `pl.DataFrame`. The output contains `UInt8` columns (scaled weather variables), which do not match the `Float32` types defined in the `Nwp` schema.

## New Contracts Needed

### `Metrics`
A new contract is needed for the output of the metrics computation.

**Location:** `packages/contracts/src/contracts/data_schemas.py`

**Schema:**
```python
class Metrics(pt.Model):
    """Evaluation metrics for power forecasts."""
    time_series_id: int = pt.Field(dtype=pl.Int32)
    power_fcst_model_name: str = pt.Field(dtype=pl.Categorical)
    lead_time_hours: float = pt.Field(dtype=pl.Float32)
    mae: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    rmse: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    nmae: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
```

## Implementation Rules
1.  **Imports:** Ensure `import patito as pt` and the specific contract from `contracts.data_schemas` are imported in each modified file.
2.  **No Source Code Modification:** The Builder agent will execute this plan. The Architect must not modify the source code directly.
3.  **No FLAW IDs:** Do not include FLAW-XXX IDs in code comments.
