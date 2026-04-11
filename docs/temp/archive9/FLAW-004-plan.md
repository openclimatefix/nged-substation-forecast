---
status: "draft"
version: "v1.0"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules: ["src/nged_substation_forecast/defs/xgb_assets.py", "tests/test_xgboost_dagster_mocked.py"]
---

# Implementation Plan: Fix FLAW-004 (Dagster Integration Test Configuration Error)

## 1. Context and Rationale
The `xgboost_integration_job` fails to resolve during the integration test (`tests/test_xgboost_dagster_integration.py`) with the following error:
`dagster._core.errors.DagsterInvalidDefinitionError: Input asset "["time_series_metadata"]" is not produced by any of the provided asset ops and is not one of the provided sources.`

This occurs because the `train_xgboost` and `evaluate_xgboost` assets in `src/nged_substation_forecast/defs/xgb_assets.py` have an optional argument `time_series_metadata: pl.DataFrame | None = None`. In Dagster, any argument to an asset function that is not explicitly configured (e.g., via `dg.Config`, `dg.ResourceParam`, or mapped in `ins`) is treated as an upstream input asset. Since `time_series_metadata` is not defined as an asset and is not included in the job's asset selection, Dagster throws an error when trying to resolve the job definition.

The `time_series_metadata` argument was likely added to facilitate passing dummy data during unit testing (`tests/test_xgboost_dagster_mocked.py`). However, this violates Dagster's asset definition rules. The correct approach is to remove the argument from the asset signatures and instead write the dummy metadata to the expected file path during testing, allowing the assets to load it normally via `pl.read_parquet`.

## 2. Required Changes

### 2.1. Update `src/nged_substation_forecast/defs/xgb_assets.py`
Remove the `time_series_metadata` argument from the asset signatures and unconditionally load the metadata from the parquet file.

**Modifications:**
1. In `train_xgboost`:
   - Remove `time_series_metadata: pl.DataFrame | None = None` from the function signature.
   - Replace the conditional loading block:
     ```python
     # Load time series metadata
     if time_series_metadata is None:
         time_series_metadata = pl.read_parquet(
             settings.nged_data_path / "parquet" / "time_series_metadata.parquet"
         )
     ```
     with unconditional loading:
     ```python
     # Load time series metadata
     time_series_metadata = pl.read_parquet(
         settings.nged_data_path / "parquet" / "time_series_metadata.parquet"
     )
     ```

2. In `evaluate_xgboost`:
   - Remove `time_series_metadata: pl.DataFrame | None = None` from the function signature.
   - Replace the conditional loading block with unconditional loading, exactly as done for `train_xgboost`.

### 2.2. Update `tests/test_xgboost_dagster_mocked.py`
Update the mocked test to write the dummy metadata to the filesystem instead of passing it as an argument.

**Modifications:**
1. In `test_xgboost_dagster_assets_materialize_with_dummy_data`:
   - After creating the `metadata` DataFrame, write it to the `tmp_path` (which is used as `settings.nged_data_path`):
     ```python
     metadata_dir = tmp_path / "parquet"
     metadata_dir.mkdir(parents=True, exist_ok=True)
     metadata.write_parquet(metadata_dir / "time_series_metadata.parquet")
     ```
   - Remove the `time_series_metadata=metadata` argument from the calls to `train_xgboost` and `evaluate_xgboost`.

## 3. Verification Steps
1. Run the mocked tests to ensure they still pass with the new filesystem-based metadata loading:
   ```bash
   uv run pytest tests/test_xgboost_dagster_mocked.py -v
   ```
2. Run the integration test to verify that the Dagster job definition resolves correctly and the test executes successfully:
   ```bash
   uv run pytest tests/test_xgboost_dagster_integration.py -v -m manual
   ```

## 4. Code Comments Mandate
- Ensure that no `FLAW-XXX` IDs are referenced in any code comments.
- The changes are straightforward signature updates and do not require extensive new architectural comments, but ensure the existing docstrings remain accurate.
