"""Dagster assets for data cleaning.

This module defines the `cleaned_actuals` asset, which takes raw live primary flows and
applies data quality cleaning logic. The cleaning process identifies "stuck" sensors
(rolling std below threshold) and "insane" power (outside physical bounds) and replaces
them with null.

Key Design Decisions:
---------------------

1. **Null Preservation for Temporal Grid**: We replace problematic power with null
   instead of removing rows or imputing. This preserves the strict 30-minute temporal
   grid which is critical for accurate lag and rolling feature generation downstream.

2. **Patito Validation**: The output is validated against the `PowerTimeSeries` schema
   which allows null values. This enforces data contracts at the asset boundary.

3. **No Imputation**: We explicitly do NOT impute missing values. Models downstream
   (like XGBoost) should handle null targets by dropping rows after feature engineering,
   not before.

4. **Partition Mapping with Lookback**: To ensure rolling window calculations have
   sufficient history at partition boundaries, we use a 1-day lookback when scanning
   the Delta table. This prevents NaN issues at the start of daily partitions.

Complex Partitioning Note:
--------------------------
The implementation uses a manual 1-day lookback window when scanning the Delta table. This ensures that when computing rolling std for the first timestamp in a partition, we have 48 periods (24 hours) of historical data available, even at the partition boundary.
"""

from typing import cast

import dagster as dg
import patito as pt
import polars as pl
from contracts.data_schemas import PowerTimeSeries
from contracts.settings import Settings
from dagster import ResourceParam
from nged_data.clean import clean_power_time_series

from ..utils import get_partition_window, scan_delta_table
from .partitions import DAILY_PARTITIONS, SIX_HOURLY_PARTITIONS


def _get_delta_path(settings: Settings, table_name: str) -> str:
    """Get the Delta table path for a given table name.

    Args:
        settings: Global settings object.
        table_name: Name of the Delta table (e.g., "raw_power_time_series").

    Returns:
        Absolute path as a string.
    """
    return str(settings.nged_data_path / "delta" / table_name)


def get_cleaned_actuals_lazy(
    settings: Settings, context: dg.AssetExecutionContext | None = None
) -> pt.LazyFrame[PowerTimeSeries]:
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

    # Use the new scan_delta_table helper which handles UTC timezone boilerplate.
    lf = scan_delta_table(delta_path)

    if context:
        context.log.info(f"Reading cleaned actuals from {delta_path}")
    return lf


@dg.asset(
    partitions_def=DAILY_PARTITIONS,
    # We use automation_condition instead of auto_materialize_policy because
    # AutoMaterializePolicy is deprecated in newer versions of Dagster.
    # AutomationCondition provides a more flexible and modern way to define
    # when assets should be automatically materialized.
    automation_condition=dg.AutomationCondition.eager(),
    deps=["raw_power_time_series"],
)
def cleaned_actuals(
    context: dg.AssetExecutionContext,
    settings: ResourceParam[Settings],
) -> pl.DataFrame:
    """Clean raw power time series and apply data quality checks.

    This asset manually scans the raw power time series Delta table for the current partition
    plus a 1-day lookback window. It applies data quality cleaning logic (stuck
    sensor detection, insane power detection).
    The output is validated against the PowerTimeSeries schema which allows null values,
    then saved to a Delta table named "cleaned_actuals".

    Cleaning Logic:
    ---------------
    - **Stuck sensors**: Rolling std dev < 0.01 MW over 48-period (24-hour) window.
    - **Insane power**: MW < -20.0 or MW > 100.0 (physically implausible for primary substations).
    - Both checks are performed per-substation to prevent data leakage.
    - Bad power are replaced with null (NOT removed) to preserve temporal grid.

    Notes:
        - Rolling operations are strictly backward-looking to prevent data leakage.
        - A 1-day lookback is used to ensure rolling windows are fully populated at the
          start of the partition.
        - Null values are preserved from the input; no rows are removed and no
          imputation is performed.
        - The output is validated against PowerTimeSeries schema which allows
          null values for MW, MVA, and MVAr columns.
        - The result is ALWAYS saved to Delta table to ensure persistence.

    Args:
        context: Dagster asset execution context.
        settings: Global settings containing data quality thresholds.

    Returns:
        MaterializeResult containing metadata about the cleaned data.
    """
    # Use the shared helper to get the partition window with a 1-day lookback.
    partition_start, partition_end, lookback_start = get_partition_window(
        context.partition_key, lookback_days=1
    )
    context.log.info(f"Cleaning partition: {context.partition_key}")

    delta_path = _get_delta_path(settings, "raw_power_time_series")

    # Manually load from Delta table and filter to the required date range (including 1 day lookback)
    # This fix is needed because the InMemoryIOManager in tests doesn't have the "yesterday"
    # partition data, and raw_power_time_series is a side-effect only asset.
    # We use the new scan_delta_table helper which handles UTC timezone boilerplate.
    raw_flows = scan_delta_table(delta_path).filter(
        pl.col("period_end_time").is_between(lookback_start, partition_end, closed="left")
    )

    # Materialize the LazyFrame once
    df_joined_materialized = cast(pl.DataFrame, raw_flows.collect())

    context.log.info(f"Materialized data shape before cleaning: {df_joined_materialized.shape}")

    # Clean the data using the shared cleaning module
    df_cleaned = clean_power_time_series(
        df_joined_materialized,
        stuck_std_threshold=settings.data_quality.stuck_std_threshold,
        min_mw_threshold=settings.data_quality.min_mw_threshold,
        max_mw_threshold=settings.data_quality.max_mw_threshold,
    )

    context.log.info(f"Cleaned data shape after cleaning: {df_cleaned.shape}")

    # Validate the output against Patito schema
    validated_df = PowerTimeSeries.validate(df_cleaned, allow_superfluous_columns=True)

    context.log.info(f"Validated data shape: {validated_df.shape}")

    # Filter validated_df to current partition's time window.
    # The TimeWindowPartitionMapping includes historical data for lookback (previous partitions),
    # but we must only append data for the current partition to avoid data duplication in the
    # Delta table. Example: For partition "2026-03-10", we only include rows with period_end_time
    # in [2026-03-10 00:00:00 UTC, 2026-03-11 00:00:00 UTC).

    # Apply filter only for the current partition's time range
    validated_df = validated_df.filter(
        pl.col("period_end_time").is_between(partition_start, partition_end, closed="left")
    )

    context.log.info(
        f"Filtered cleaned actuals to current partition. "
        f"Partition range: [{partition_start}, {partition_end}). "
        f"Data shape: {validated_df.shape}"
    )

    # Save to Delta table using overwrite with replace_where to ensure idempotency
    # within the partition's time range. We use the partition's temporal boundaries
    # for the predicate instead of the data's min/max time. This ensures that if
    # the source data is removed, the partition in the Delta table is correctly
    # cleared out (by writing an empty DataFrame), maintaining idempotency.
    delta_path = _get_delta_path(settings, "cleaned_actuals")

    validated_df.write_delta(
        delta_path,
        mode="overwrite",
        delta_write_options={
            "partition_by": ["time_series_id"],
            "predicate": f"period_end_time >= '{partition_start.isoformat()}' AND period_end_time < '{partition_end.isoformat()}'",
        },
    )

    context.log.info(f"Saved cleaned actuals to Delta table at {delta_path}")
    return validated_df


@dg.asset(
    partitions_def=SIX_HOURLY_PARTITIONS,
    automation_condition=dg.AutomationCondition.eager(),
    deps=["raw_power_time_series"],
)
def cleaned_power_time_series(
    context: dg.AssetExecutionContext,
    settings: ResourceParam[Settings],
) -> pl.DataFrame:
    """Clean raw power time series and apply data quality checks.

    This asset manually scans the raw power time series Delta table for the current partition
    plus a 1-day lookback window. It applies data quality cleaning logic (stuck
    sensor detection, insane power detection).
    The output is validated against the PowerTimeSeries schema which allows null values,
    then saved to a Delta table named "cleaned_power_time_series".

    Cleaning Logic:
    ---------------
    - **Stuck sensors**: Rolling std dev < 0.01 MW over 48-period (24-hour) window.
    - **Insane power**: MW < -20.0 or MW > 100.0 (physically implausible for primary substations).
    - Both checks are performed per-substation to prevent data leakage.
    - Bad power are replaced with null (NOT removed) to preserve temporal grid.

    Args:
        context: Dagster asset execution context.
        settings: Global settings containing data quality thresholds.

    Returns:
        MaterializeResult containing metadata about the cleaned data.
    """
    # Use the shared helper to get the partition window with a 1-day lookback.
    partition_start, partition_end, lookback_start = get_partition_window(
        context.partition_key, lookback_days=1
    )
    context.log.info(f"Cleaning partition: {context.partition_key}")

    delta_path = _get_delta_path(settings, "raw_power_time_series")

    # Manually load from Delta table and filter to the required date range (including 1 day lookback)
    raw_flows = scan_delta_table(delta_path).filter(
        pl.col("period_end_time").is_between(lookback_start, partition_end, closed="left")
    )

    # Materialize the LazyFrame once
    df_joined_materialized = cast(pl.DataFrame, raw_flows.collect())

    context.log.info(f"Materialized data shape before cleaning: {df_joined_materialized.shape}")

    # Clean the data using the shared cleaning module
    df_cleaned = clean_power_time_series(
        df_joined_materialized,
        stuck_std_threshold=settings.data_quality.stuck_std_threshold,
        min_mw_threshold=settings.data_quality.min_mw_threshold,
        max_mw_threshold=settings.data_quality.max_mw_threshold,
    )

    context.log.info(f"Cleaned data shape after cleaning: {df_cleaned.shape}")

    # Validate the output against Patito schema
    validated_df = PowerTimeSeries.validate(df_cleaned, allow_superfluous_columns=True)

    context.log.info(f"Validated data shape: {validated_df.shape}")

    # Filter validated_df to current partition's time window.
    validated_df = validated_df.filter(
        pl.col("period_end_time").is_between(partition_start, partition_end, closed="left")
    )

    context.log.info(
        f"Filtered cleaned power time series to current partition. "
        f"Partition range: [{partition_start}, {partition_end}). "
        f"Data shape: {validated_df.shape}"
    )

    # Save to Delta table using overwrite with replace_where to ensure idempotency
    delta_path = _get_delta_path(settings, "cleaned_power_time_series")

    validated_df.write_delta(
        delta_path,
        mode="overwrite",
        delta_write_options={
            "partition_by": ["time_series_id"],
            "predicate": f"period_end_time >= '{partition_start.isoformat()}' AND period_end_time < '{partition_end.isoformat()}'",
        },
    )

    context.log.info(f"Saved cleaned power time series to Delta table at {delta_path}")
    return validated_df
