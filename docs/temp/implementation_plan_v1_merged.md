---
status: "reviewed"
version: "v1_merged"
after_reviewer: "all"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules: ["src/nged_substation_forecast", "packages/xgboost_forecaster", "packages/contracts", "packages/ml_core", "tests"]
---

# Implementation Plan: Fix Pytest Warnings and Exceptions

This plan addresses multiple warnings and exceptions that occur during `uv run pytest`.

## Conflicts

**Polars `join_asof` Sorting:**
There was a direct conflict between the Simplicity Reviewer and the Scientist regarding the sorting strategy for Polars `join_asof`.
- The Simplicity Reviewer argued that sorting by the `by` columns (`VALID_TIME`, `H3_INDEX`) is over-engineered and that sorting only by the `on` column (`available_time`) is sufficient.
- The Scientist countered that Polars strictly requires the data to be sorted by the `by` columns first, and then the `on` column. Failing to do so results in silent data leakage and incorrect matches.
**Resolution:** The Scientist's recommendation is upheld. Correctness and data integrity in ML pipelines supersede minor performance optimizations or code brevity. The explicit multi-column sort will be implemented.

## Review Responses & Rejections

* **FLAW-001 (Simplicity):** REJECTED. The Simplicity reviewer suggested sorting only by `available_time` to simplify the code. However, as detailed by the Scientist, Polars `join_asof` requires the dataframe to be sorted by the `by` columns first, then the `on` column. Sorting only by `available_time` while grouping by `VALID_TIME` and `H3_INDEX` will lead to incorrect joins and silent data leakage. We will proceed with the explicit multi-column sort.
* **FLAW-002 (Simplicity):** ACCEPTED. The Simplicity reviewer correctly pointed out that explicitly calling `mlflow.set_tracking_uri()` in every utility function is redundant. We will instead rely on the `MLFLOW_TRACKING_URI` environment variable, which MLflow automatically reads. We will still add the setting to the `Settings` contract for validation and documentation purposes, but remove the explicit setter calls.
* **Tester Requirements:** ACCEPTED. We will add the requested tests to verify the Polars join correctness and the SQLite teardown behavior.

## 1. Dagster Deprecation: AutoMaterializePolicy
**File:** `src/nged_substation_forecast/defs/data_cleaning_assets.py`
**Action:**
- Replace `auto_materialize_policy=dg.AutoMaterializePolicy.eager()` with `automation_condition=dg.AutomationCondition.eager()`.

## 2. Polars UserWarning: Sortedness of columns cannot be checked
**File:** `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`
**Action:**
- In `_prepare_and_join_nwps`, update the `join_asof` call to explicitly sort by the group keys (`valid_time`, `h3_index`) and the join key (`available_time`), and pass `check_sortedness=False` to suppress the false-positive warning from Polars.
- Change:
  ```python
  combined_nwps = (
      combined_nwps.sort("available_time")
      .join_asof(
          other_nwp.sort("available_time"),
          on="available_time",
          by=[NwpColumns.VALID_TIME, NwpColumns.H3_INDEX],
      )
  ```
  to:
  ```python
  combined_nwps = (
      combined_nwps.sort([NwpColumns.VALID_TIME, NwpColumns.H3_INDEX, "available_time"])
      .join_asof(
          other_nwp.sort([NwpColumns.VALID_TIME, NwpColumns.H3_INDEX, "available_time"]),
          on="available_time",
          by=[NwpColumns.VALID_TIME, NwpColumns.H3_INDEX],
          check_sortedness=False,
      )
  ```

## 3. Numpy UserWarning: no explicit representation of timezones
**File:** `tests/test_nwp_ingestion_robustness.py`
**Action:**
- In `test_temporal_deduplication_last_update_wins`, replace the `np.datetime64` comparison with a `.timestamp()` comparison to avoid timezone warnings while maintaining precision.
- Change:
  ```python
  assert np.datetime64(init_times.item(), "us") == np.datetime64(dt2, "us")
  ```
  to:
  ```python
  assert init_times.item().timestamp() == dt2.timestamp()
  ```

## 4. Dagster UserWarning: UnresolvedAssetJobDefinition
**File:** `tests/test_xgboost_dagster_integration.py`
**Action:**
- Change `job = defs.get_job_def("xgboost_integration_job")` to `job = defs.resolve_job_def("xgboost_integration_job")` to correctly retrieve the job definition without triggering the deprecation warning.

## 5. MLflow FutureWarning: Filesystem tracking backend is deprecated
**File:** `packages/contracts/src/contracts/settings.py`
**Action:**
- Add `mlflow_tracking_uri: str = Field(default="sqlite:///mlflow.db", description="MLflow tracking URI.")` to the `Settings` class.
- *Note: Based on FLAW-002, we will NOT add explicit `mlflow.set_tracking_uri()` calls to `utils.py`. We will rely on the environment variable `MLFLOW_TRACKING_URI` being set (which can be populated from this Settings class by the application entrypoint).*

## 6. SQLite Exception: Cannot operate on a closed database
**File:** `tests/test_xgboost_dagster_integration.py`
**Action:**
- Wrap the `job.execute_in_process(...)` call with `with dg.DagsterInstance.ephemeral() as instance:` and pass `instance=instance` to ensure proper cleanup of the SQLite database and SQLAlchemy connection pool.
- Change:
  ```python
  result = job.execute_in_process(
      run_config=run_config,
      partition_key=test_end.isoformat(),
      resources={
          **resources,
          "io_manager": dg.mem_io_manager,
      },
  )
  ```
  to:
  ```python
  with dg.DagsterInstance.ephemeral() as instance:
      result = job.execute_in_process(
          run_config=run_config,
          partition_key=test_end.isoformat(),
          resources={
              **resources,
              "io_manager": dg.mem_io_manager,
          },
          instance=instance,
      )
  ```

## 7. Required Test Additions (QA Requirements)
**Action:**
- **Polars Join Test:** Add a test in `tests/` (or `packages/xgboost_forecaster/tests/`) that explicitly checks that the result of `_prepare_and_join_nwps` handles unsorted input correctly and maintains join semantics.
- **Integration Teardown Test:** Ensure `test_xgboost_dagster_integration.py` is robust against the "Closed Database" exception during teardown, verifying the ephemeral instance context manager resolves the issue.
