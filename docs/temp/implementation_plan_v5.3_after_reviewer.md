---
status: "draft"
version: "v5.3"
after_reviewer: "reviewer"
task_type: "standard"
requires_ml_review: true
requires_data_engineer: true
target_modules: ["packages/xgboost_forecaster", "packages/dynamical_data", "packages/contracts"]
---

# Implementation Plan: Address Reviewer Loop 5 Flaws

## Summary of Changes
This plan addresses the 5 flaws identified by the Reviewer in Loop 5. The fixes improve performance, robustness, and configuration management across the data loading and processing pipelines.

## Step-by-Step Plan

### 1. Fix FLAW-001: Inefficient Delta Lake Read
**Target File:** `packages/xgboost_forecaster/src/xgboost_forecaster/data.py`
**Action:** Replace the eager `pl.read_delta` with a lazy `pl.scan_delta` to optimize memory usage and speed when extracting unique substation numbers.
**Code Snippet:**
```python
    # Only return substations we have local power data for in Delta Lake
    substations_with_telemetry = (
        pl.scan_delta(str(config.base_power_path))
        .select("substation_number")
        .unique()
        .collect()
        .to_series()
        .to_list()
    )
```

### 2. Fix FLAW-002: Brittle Filename Parsing
**Target File:** `packages/xgboost_forecaster/src/xgboost_forecaster/data.py`
**Action:** Use a robust `try...except` block within a helper function or loop to parse dates from filenames, ignoring files that do not match the expected format.
**Code Snippet:** Replace the `relevant_files` list comprehension with:
```python
    # We expect NWP files to follow the `YYYY-MM-DDTHHZ.parquet` naming contract.
    files = sorted(config.base_weather_path.glob("*.parquet"))
    relevant_files = []
    for f in files:
        try:
            file_date = datetime.strptime(f.stem[:10], "%Y-%m-%d").date()
            if start_date <= file_date <= end_date:
                relevant_files.append(f)
        except ValueError:
            # Skip files that don't match the expected date format
            continue

    if not relevant_files:
```

### 3. Fix FLAW-003: Hardcoded S3 Paths
**Target File 1:** `packages/contracts/src/contracts/settings.py`
**Action:** Add ECMWF S3 configuration to the `Settings` class.
**Code Snippet:** Add to `Settings` class:
```python
    # ECMWF Data Settings
    ecmwf_s3_bucket: str = Field(
        default="dynamical-ecmwf-ifs-ens",
        description="S3 bucket for ECMWF Icechunk store.",
    )
    ecmwf_s3_prefix: str = Field(
        default="ecmwf-ifs-ens-forecast-15-day-0-25-degree/v0.1.0.icechunk/",
        description="S3 prefix for ECMWF Icechunk store.",
    )
```

**Target File 2:** `packages/dynamical_data/src/dynamical_data/processing.py`
**Action:** Import `_SETTINGS` and use the new settings in `download_ecmwf`.
**Code Snippet:**
Add import:
```python
from contracts.settings import Settings
_SETTINGS = Settings()
```
Update `download_ecmwf`:
```python
    if ds is None:
        # Connect to the production icechunk store
        storage = icechunk.s3_storage(
            bucket=_SETTINGS.ecmwf_s3_bucket,
            prefix=_SETTINGS.ecmwf_s3_prefix,
            region=DEFAULT_AWS_REGION,
            anonymous=True,
        )
```

### 4. Fix FLAW-004: Redundant Variable Lists
**Target File:** `packages/dynamical_data/src/dynamical_data/processing.py`
**Action:** Define a single `REQUIRED_NWP_VARS` constant at the module level and use it in both `validate_dataset_schema` and `download_ecmwf`.
**Code Snippet:**
Add at module level (after imports):
```python
REQUIRED_NWP_VARS = {
    "temperature_2m",
    "dew_point_temperature_2m",
    "wind_u_10m",
    "wind_v_10m",
    "wind_u_100m",
    "wind_v_100m",
    "pressure_surface",
    "pressure_reduced_to_mean_sea_level",
    "geopotential_height_500hpa",
    "downward_long_wave_radiation_flux_surface",
    "downward_short_wave_radiation_flux_surface",
    "precipitation_surface",
    "categorical_precipitation_type_surface",
}
```
In `validate_dataset_schema`:
```python
    # Check for a minimal set of required data variables
    # These are the variables required by the Nwp schema in contracts.data_schemas
    missing_vars = REQUIRED_NWP_VARS - set(ds.data_vars)
```
In `download_ecmwf`:
```python
    # We subset the dataset to only the required variables defined in the Nwp schema
    # to save network bandwidth and memory during the download process.
    # We also include the raw wind components (10u, 10v, 100u, 100v) which are
    # needed for calculating wind speed and direction later.
    # Cast to xr.Dataset to satisfy the type checker, as indexing with a list
    # can sometimes be misidentified as returning a DataArray.
    ds = cast(xr.Dataset, ds[list(REQUIRED_NWP_VARS)])
```

### 5. Fix FLAW-005: Silent Data Loss
**Target File:** `packages/xgboost_forecaster/src/xgboost_forecaster/data.py`
**Action:** Calculate the percentage of dropped single-row groups and include it in the warning message.
**Code Snippet:** Update the single-row group filtering logic:
```python
    # FLAW-005: Ensure each group has at least two points for interpolation.
    # Groups with only 1 row cannot be interpolated and would violate the
    # 30-minute temporal resolution contract.
    group_counts = df.group_by(["init_time", "h3_index", "ensemble_member"]).len()
    total_groups = group_counts.height
    single_row_groups = group_counts.filter(pl.col("len") == 1)

    if single_row_groups.height > 0:
        dropped_pct = (single_row_groups.height / total_groups) * 100 if total_groups > 0 else 0
        log.warning(
            f"Dropping {single_row_groups.height} groups ({dropped_pct:.2f}%) with only 1 row as they cannot be interpolated."
        )
        df = df.filter(pl.len().over(["init_time", "h3_index", "ensemble_member"]) > 1)
```

## Review Responses & Rejections

* **FLAW-001 (Inefficient Delta Lake Read):** ACCEPTED. Using `pl.scan_delta` avoids loading the entire dataset into memory, preventing OOM errors.
* **FLAW-002 (Brittle Filename Parsing):** ACCEPTED. A `try...except` block makes the parsing robust to unexpected files in the directory.
* **FLAW-003 (Hardcoded S3 Paths):** ACCEPTED. Moving S3 configuration to `Settings` improves maintainability and environment separation.
* **FLAW-004 (Redundant Variable Lists):** ACCEPTED. A single source of truth (`REQUIRED_NWP_VARS`) prevents drift and bugs.
* **FLAW-005 (Silent Data Loss):** ACCEPTED. Adding the percentage to the warning provides better visibility. *Architect Note:* We reject the suggestion to use a constant fill strategy for single-row groups. The 30-minute temporal resolution contract requires interpolation to capture intra-hour dynamics. A single point provides no gradient information, making constant fill physically unrealistic for weather variables. Dropping with a clear warning is the safest approach to avoid introducing artificial step functions into the training data.
