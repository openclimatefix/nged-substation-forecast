---
status: "draft"
version: "v1.1"
after_reviewer: "scientist"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules: ["tests"]
---

# Implementation Plan: Dagster Integration Memory Fix Evaluation

This plan evaluates the memory optimization fixes implemented by the Scientist and determines whether further changes are needed for the integration test and other assets.

## 1. Evaluation of `combined_actuals` and `healthy_substations` Filtering

**Finding:** `combined_actuals` and `healthy_substations` do **not** need configuration-based filtering.
**Rationale:**
- The `combined_actuals` asset returns a `pl.LazyFrame` backed directly by a Delta Lake scan (`pl.scan_delta`). It does not eagerly collect data into memory. When downstream assets (like `train_xgboost` and `evaluate_xgboost`) apply filters (e.g., `pl.col("substation_number").is_in(sub_ids)`), Polars' query optimizer automatically pushes these filters down to the Delta Lake storage layer. This means only the required substations are ever read from disk, making it highly memory-efficient by default.
- The `healthy_substations` asset performs aggregations on the `combined_actuals` LazyFrame (`group_by`, `agg`, `unique`) before calling `.collect()`. Polars' query engine optimizes this execution plan to only read the necessary columns (`substation_number`, `timestamp`, `MW_or_MVA`) and performs the aggregation efficiently without loading the entire raw dataset into memory.

## 2. Evaluation of IO Manager for Integration Test

**Finding:** We should **keep `dg.mem_io_manager`** for the integration test and **reject** switching to `dg.fs_io_manager`.
**Rationale:**
1. **Patito Pickling Errors:** `dg.fs_io_manager` uses Python's `pickle` module. The `substation_metadata` asset returns a `Patito` DataFrame (`pt.DataFrame[SubstationMetadata]`). Patito dynamically generates Pydantic-backed subclasses (e.g., `SubstationMetadataDataFrame`) that are not defined in the module scope, causing `pickle` to fail with a `PicklingError`.
2. **Inefficient Serialization:** The `processed_nwp_data` asset collects data into an in-memory `pl.DataFrame` for interpolation and returns it via `.lazy()`. This means the returned `LazyFrame` is backed by in-memory data. Pickling this object with `dg.fs_io_manager` would require duplicating the data in memory to create a massive `bytes` object, which is slow and temporarily spikes memory usage—defeating the purpose of avoiding OOM.
3. **Sufficient Fixes:** The Scientist's fix (filtering NWP data *before* processing via `ProcessedNWPConfig`) already reduces the memory footprint of the integration test (which runs on 5 substations) to a few megabytes. OOM is no longer a risk for the test's intended scope.

## 3. Long-Term Scaling Recommendation (Out of Scope for this PR)

If `processed_nwp_data` needs to scale to all ~3000 substations in production, neither `mem_io_manager` nor `fs_io_manager` will suffice. The asset itself should be refactored to write its interpolated output to a partitioned Parquet dataset on disk and return a `pl.scan_parquet(...)` LazyFrame, or a custom `PolarsParquetIOManager` should be implemented to handle Polars serialization natively without `pickle`.

## Review Responses & Rejections

* **FLAW-OOM-RISK (Conductor/Scientist):** REJECTED switching to `dg.fs_io_manager`. Technical justification: `dg.fs_io_manager` uses `pickle`, which fails on Patito DataFrames and is highly inefficient for in-memory Polars DataFrames. The Scientist's upstream filtering fix is sufficient for the integration test's memory footprint. `combined_actuals` also does not need filtering as it leverages Polars' lazy evaluation and predicate pushdown to Delta Lake.
