---
status: "draft"
version: "v1.0"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules: ["packages", "src", "tests", "exploration_scripts"]
---
# Plan: Replace Polars Type Hints with Patito Data Contracts

## Overview
This plan outlines the replacement of generic `pl.DataFrame` and `pl.LazyFrame` type hints with specific `patito` data contracts across the codebase. This will improve type safety, self-documenting code, and ensure data schemas are explicitly defined at function boundaries.

## Rationale
Using generic `pl.DataFrame` obscures the expected schema of the data passing through the system. By replacing these with `pt.DataFrame[Contract]` and `pt.LazyFrame[Contract]`, we make the expected columns and data types explicit.

**Important Note on Code Comments:**
When implementing these changes, ensure that any added or modified comments focus on the *why* (intent and rationale) rather than the *how*. Comments should "connect the dots" across the codebase, explaining how these contracts enforce data integrity between different modules (e.g., how `ProcessedNwp` ensures safe handoffs between the ingestion pipeline and the ML feature engineering).

## Function Mapping

### 1. `packages/dynamical_data/src/dynamical_data/scaling.py`
*   `load_scaling_params`
    *   **Current:** `-> pl.DataFrame`
    *   **New:** `-> pt.DataFrame[ScalingParams]`
*   `scale_to_uint8`
    *   **Current:** `(df: pl.DataFrame, scaling_params: pl.DataFrame) -> pl.DataFrame`
    *   **New:** `(df: pt.DataFrame[Nwp], scaling_params: pt.DataFrame[ScalingParams]) -> pl.DataFrame`
    *   *Note:* The return type remains `pl.DataFrame` because the output contains `UInt8` columns, which do not have a corresponding Patito schema (the schemas define the physical `Float32` types).
*   `recover_physical_units`
    *   **Current:** `(df: pl.DataFrame, scaling_params: pl.DataFrame) -> pl.DataFrame`
    *   **New:** `(df: pl.DataFrame, scaling_params: pt.DataFrame[ScalingParams]) -> pt.DataFrame[Nwp]`
    *   *Note:* The input remains `pl.DataFrame` (as it contains `UInt8` columns), but the output is restored to the physical `Nwp` schema.

### 2. `packages/nged_data/src/nged_data/clean.py`
*   `sort_data`
    *   **Current:** `(df: pl.DataFrame) -> pl.DataFrame`
    *   **New:** `(df: pt.DataFrame[PowerTimeSeries]) -> pt.DataFrame[PowerTimeSeries]`
*   `calculate_rolling_variance`
    *   **Current:** `(df: pl.DataFrame) -> pl.DataFrame`
    *   **New:** `(df: pt.DataFrame[PowerTimeSeries]) -> pl.DataFrame`
    *   *Note:* The return type remains `pl.DataFrame` because this function appends a `rolling_variance` column, which is not part of the `PowerTimeSeries` schema.
*   `validate_data`
    *   **Current:** `(df: pl.DataFrame) -> pt.DataFrame[PowerTimeSeries]`
    *   **New:** `(df: pl.DataFrame) -> pt.DataFrame[PowerTimeSeries]` (No change needed, but explicitly confirming it's correct as it takes a raw DataFrame and validates it into the contract).

### 3. `packages/nged_data/src/nged_data/metadata.py`
*   `get_df_hash`
    *   **Current:** `(df: pl.DataFrame) -> int`
    *   **New:** `(df: pt.DataFrame[TimeSeriesMetadata]) -> int`
*   `upsert_metadata`
    *   **Current:** `(new_metadata: pl.DataFrame, metadata_path: Path) -> None`
    *   **New:** `(new_metadata: pt.DataFrame[TimeSeriesMetadata], metadata_path: Path) -> None`

### 4. `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`
*   `_prepare_features`
    *   **Current:** `(self, df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame`
    *   **New:** `(self, df: pt.DataFrame[XGBoostInputFeatures] | pt.LazyFrame[XGBoostInputFeatures]) -> pl.DataFrame | pl.LazyFrame`
    *   *Note:* The return type remains generic because the function selects a dynamic subset of columns (the feature matrix), which no longer matches the full `XGBoostInputFeatures` schema.

### 5. `packages/xgboost_forecaster/src/xgboost_forecaster/features.py`
*   `add_time_features`
    *   **Current:** `(df: pl.LazyFrame) -> pl.LazyFrame`
    *   **New:** `(df: pt.LazyFrame[ProcessedNwp]) -> pl.LazyFrame`
    *   *Note:* The return type remains `pl.LazyFrame` because the function appends `nwp_init_hour`, which is not part of the `ProcessedNwp` schema (it belongs to the final `XGBoostInputFeatures` schema, which is constructed later).

### 6. `exploration_scripts/test_ensure_h3*.py`
*   `ensure_h3_res_5` (in `test_ensure_h3_v3.py`, `test_ensure_h3.py`, `test_ensure_h3_v2.py`)
    *   **Current:** `(df: pl.DataFrame) -> pl.DataFrame`
    *   **New:** `(df: pt.DataFrame[TimeSeriesMetadata]) -> pt.DataFrame[TimeSeriesMetadata]`

### 7. `tests/test_longitude_sorting.py` & `tests/test_nwp_ingestion_robustness.py`
*   `h3_grid`
    *   **Current:** Implicit or `pl.DataFrame`
    *   **New:** `-> pt.DataFrame[H3GridWeights]`

### 8. `src/nged_substation_forecast/defs/weather_assets.py`
*   `all_nwp_data`
    *   **Current:** `-> pl.LazyFrame`
    *   **New:** `-> pt.LazyFrame[Nwp]`

### 9. `src/nged_substation_forecast/utils.py`
*   `scan_delta_table`
    *   **Current:** `(delta_path: str) -> pl.LazyFrame`
    *   **New:** `(delta_path: str) -> pt.LazyFrame[PowerTimeSeries]`
    *   *Note:* While this is a generic utility, in the context of this codebase it is exclusively used to scan raw power flows (`cleaned_power_time_series_assets.py` and `data_cleaning_assets.py`). If a generic approach is preferred, it could be refactored to accept a schema type variable, but typing it to `PowerTimeSeries` reflects its current usage.

## Implementation Rules
1.  **Imports:** Ensure `import patito as pt` and the specific contract from `contracts.data_schemas` are imported in each modified file.
2.  **No Source Code Modification:** The Builder agent will execute this plan. The Architect must not modify the source code directly.
3.  **No FLAW IDs:** Do not include FLAW-XXX IDs in code comments.
