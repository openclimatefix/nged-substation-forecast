"""Dagster assets for data cleaning.

This module defines the `cleaned_actuals` asset, which takes raw live primary flows and
applies data quality cleaning logic. The cleaning process identifies "stuck" sensors
(rolling std below threshold) and "insane" values (outside physical bounds) and replaces
them with null.

Key Design Decisions:
---------------------

1. **Null Preservation for Temporal Grid**: We replace problematic values with null
   instead of removing rows or imputing. This preserves the strict 30-minute temporal
   grid which is critical for accurate lag and rolling feature generation downstream.

2. **Patito Validation**: The output is validated against the `SubstationPowerFlows` schema
   which allows null values. This enforces data contracts at the asset boundary.

3. **No Imputation**: We explicitly do NOT impute missing values. Models downstream
   (like XGBoost) should handle null targets by dropping rows after feature engineering,
   not before.

4. **Partition Mapping with Lookback**: To ensure rolling window calculations have
   sufficient history at partition boundaries, we use a 1-day lookback when scanning
   the Delta table. This prevents NaN issues at the start of daily partitions.

Complex Partitioning Note:
--------------------------
The implementation uses TimeWindowPartitionMapping with a 2-day lookback window. This
ensures that when computing rolling std for the first timestamp in a partition, we
have 48 periods (24 hours) of historical data available, even at the partition boundary.
"""

import dagster as dg
import polars as pl
import patito as pt
from typing import cast
from datetime import datetime, timedelta, timezone
from dagster import (
    ResourceParam,
    DailyPartitionsDefinition,
)

from contracts.data_schemas import SubstationPowerFlows
from contracts.settings import Settings
from nged_data import clean_substation_flows


def _get_delta_path(settings: Settings, table_name: str) -> str:
    """Get the Delta table path for a given table name.

    Args:
        settings: Global settings object.
        table_name: Name of the Delta table (e.g., "live_primary_flows").

    Returns:
        Absolute path as a string.
    """
    return str(settings.nged_data_path / "delta" / table_name)


def get_cleaned_actuals_lazy(
    settings: Settings, context: dg.AssetExecutionContext | None = None
) -> pt.LazyFrame[SubstationPowerFlows]:
    """Retrieves the cleaned actuals from the Delta table.

    This function serves as the single source of truth for accessing cleaned actuals
    data across the pipeline. It reads directly from the `cleaned_actuals` Delta table.

    Args:
        settings: Global settings object.
        context: Optional Dagster context for logging.

    Returns:
        A Polars LazyFrame of the cleaned actuals.
    """
    delta_path = _get_delta_path(settings, "cleaned_actuals")

    # Attempt to scan the Delta table and verify it exists by fetching schema.
    # We fetch the schema to force an immediate failure if the table is missing,
    # as pl.scan_delta() is lazy and might not fail until collection.
    lf = pl.scan_delta(delta_path)
    schema = lf.collect_schema()

    # Ensure timestamp has UTC timezone. Delta tables sometimes lose timezone info.
    timestamp_dtype = schema["timestamp"]
    if isinstance(timestamp_dtype, pl.Datetime) and timestamp_dtype.time_zone is None:
        lf = lf.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))
    elif isinstance(timestamp_dtype, pl.Datetime) and timestamp_dtype.time_zone != "UTC":
        lf = lf.with_columns(pl.col("timestamp").dt.convert_time_zone("UTC"))

    if context:
        context.log.info(f"Reading cleaned actuals from {delta_path}")
    return cast(pt.LazyFrame[SubstationPowerFlows], lf)


@dg.asset(
    partitions_def=DailyPartitionsDefinition(start_date="2026-03-10", end_offset=1),
    auto_materialize_policy=dg.AutoMaterializePolicy.eager(),
    deps=["live_primary_flows"],
)
def cleaned_actuals(
    context: dg.AssetExecutionContext,
    settings: ResourceParam[Settings],
) -> dg.MaterializeResult:
    """Clean raw live primary flows and apply data quality checks.

    This asset manually scans the live primary flows Delta table for the current partition
    plus a 1-day lookback window. It applies data quality cleaning logic (stuck
    sensor detection, insane value detection).
    The output is validated against the SubstationPowerFlows schema which allows null values,
    then saved to a Delta table named "cleaned_actuals".

    Cleaning Logic:
    ---------------
    - **Stuck sensors**: Rolling std dev < 0.01 MW over 48-period (24-hour) window.
    - **Insane values**: MW < -20.0 or MW > 100.0 (physically implausible for primary substations).
    - Both checks are performed per-substation to prevent data leakage.
    - Bad values are replaced with null (NOT removed) to preserve temporal grid.

    Notes:
        - Rolling operations are strictly backward-looking to prevent data leakage.
        - A 1-day lookback is used to ensure rolling windows are fully populated at the
          start of the partition.
        - Null values are preserved from the input; no rows are removed and no
          imputation is performed.
        - The output is validated against SubstationPowerFlows schema which allows
          null values for MW, MVA, and MVAr columns.
        - The result is ALWAYS saved to Delta table to ensure persistence.

    Args:
        context: Dagster asset execution context.
        settings: Global settings containing data quality thresholds.

    Returns:
        MaterializeResult containing metadata about the cleaned data.
    """
    partition_key = context.partition_key
    context.log.info(f"Cleaning partition: {partition_key}")

    partition_start = datetime.fromisoformat(partition_key).replace(tzinfo=timezone.utc)
    partition_end = partition_start + timedelta(days=1)
    lookback_start = partition_start - timedelta(days=1)

    delta_path = _get_delta_path(settings, "live_primary_flows")

    # Manually load from Delta table and filter to the required date range (including 1 day lookback)
    # This fix is needed because the InMemoryIOManager in tests doesn't have the "yesterday"
    # partition data, and live_primary_flows is a side-effect only asset.
    live_primary_flows = pl.scan_delta(delta_path).filter(
        pl.col("timestamp").is_between(lookback_start, partition_end, closed="left")
    )

    # Materialize the LazyFrame once
    df_joined_materialized = cast(pl.DataFrame, live_primary_flows.collect())

    # If the DataFrame is empty, return empty MaterializeResult
    if df_joined_materialized.is_empty():
        context.log.warning("Input DataFrame is empty, returning empty MaterializeResult.")
        return dg.MaterializeResult(metadata={"num_rows": 0})

    # Ensure timestamp has UTC timezone before cleaning/validation.
    # Delta tables sometimes lose timezone info or Polars scans them as naive.
    timestamp_dtype = df_joined_materialized.schema["timestamp"]
    if isinstance(timestamp_dtype, pl.Datetime) and timestamp_dtype.time_zone is None:
        df_joined_materialized = df_joined_materialized.with_columns(
            pl.col("timestamp").dt.replace_time_zone("UTC")
        )
    elif isinstance(timestamp_dtype, pl.Datetime) and timestamp_dtype.time_zone != "UTC":
        df_joined_materialized = df_joined_materialized.with_columns(
            pl.col("timestamp").dt.convert_time_zone("UTC")
        )

    context.log.info(f"Materialized data shape before cleaning: {df_joined_materialized.shape}")

    # Clean the data using the shared cleaning module
    df_cleaned = clean_substation_flows(df_joined_materialized, settings)

    context.log.info(f"Cleaned data shape after cleaning: {df_cleaned.shape}")

    # Validate the output against Patito schema
    validated_df = SubstationPowerFlows.validate(df_cleaned)

    context.log.info(f"Validated data shape: {validated_df.shape}")

    # Filter validated_df to current partition's time window.
    # The TimeWindowPartitionMapping includes historical data for lookback (previous partitions),
    # but we must only append data for the current partition to avoid data duplication in the
    # Delta table. Example: For partition "2026-03-10", we only include rows with timestamp
    # in [2026-03-10 00:00:00 UTC, 2026-03-11 00:00:00 UTC).

    # Apply filter only for the current partition's time range
    validated_df = validated_df.filter(
        pl.col("timestamp").is_between(partition_start, partition_end, closed="left")
    )

    context.log.info(
        f"Filtered cleaned actuals to current partition. "
        f"Partition range: [{partition_start}, {partition_end}). "
        f"Data shape: {validated_df.shape}"
    )

    # Save to Delta table using overwrite with replace_where to ensure idempotency
    # within the partition's time range.
    min_time = cast(datetime, validated_df.get_column("timestamp").min())
    max_time = cast(datetime, validated_df.get_column("timestamp").max())

    delta_path = _get_delta_path(settings, "cleaned_actuals")

    if validated_df.is_empty():
        context.log.info("No data to write to Delta table for this partition.")
    else:
        validated_df.write_delta(
            delta_path,
            mode="overwrite",
            delta_write_options={
                "partition_by": ["substation_number"],
                "predicate": f"timestamp >= '{min_time.isoformat()}' AND timestamp <= '{max_time.isoformat()}'",
            },
        )

    context.log.info(f"Saved cleaned actuals to Delta table at {delta_path}")

    return dg.MaterializeResult(metadata={"num_rows": len(validated_df)})
