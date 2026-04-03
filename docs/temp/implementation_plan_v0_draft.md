---
status: "draft"
version: "v0"
after_reviewer: "none"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules: ["src/nged_substation_forecast", "packages/xgboost_forecaster", "packages/contracts", "packages/ml_core", "tests"]
---

# Implementation Plan: Fix Pytest Warnings and Exceptions

This plan addresses multiple warnings and exceptions that occur during `uv run pytest`.

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
**File 1:** `packages/contracts/src/contracts/settings.py`
**Action:**
- Add `mlflow_tracking_uri: str = Field(default="sqlite:///mlflow.db", description="MLflow tracking URI.")` to the `Settings` class.

**File 2:** `packages/ml_core/src/ml_core/utils.py`
**Action:**
- Import `Settings` from `contracts.settings`.
- In `train_and_log_model` and `evaluate_and_save_model`, add `mlflow.set_tracking_uri(Settings().mlflow_tracking_uri)` immediately before `mlflow.set_experiment(model_name)`.

## 6. SQLite Exception: Cannot operate on a closed database
**File:** `tests/test_xgboost_dagster_integration.py`
**Action:**
- The SQLite error during teardown is caused by Dagster's ephemeral instance not being properly disposed of when `execute_in_process` is called without an explicit context manager.
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
