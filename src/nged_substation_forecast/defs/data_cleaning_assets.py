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
The implementation uses a manual 1-day lookback window when scanning the Delta table. This ensures that when computing rolling std for the first timestamp in a partition, we have 48 periods (24 hours) of historical data available, even at the partition boundary.
"""

import dagster as dg
import polars as pl
import patito as pt
from typing import cast
from dagster import ResourceParam

from contracts.data_schemas import SubstationPowerFlows
from contracts.settings import Settings
from nged_data import clean_substation_flows, get_partition_window, scan_delta_table
from .partitions import DAILY_PARTITIONS


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

    # Use the new scan_delta_table helper which handles UTC timezone boilerplate.
    lf = scan_delta_table(delta_path)

    if context:
        context.log.info(f"Reading cleaned actuals from {delta_path}")
    return cast(pt.LazyFrame[SubstationPowerFlows], lf)


@dg.asset(
    partitions_def=DAILY_PARTITIONS,
    auto_materialize_policy=dg.AutoMaterializePolicy.eager(),
    deps=["live_primary_flows"],
)
def cleaned_actuals(
    context: dg.AssetExecutionContext,
    settings: ResourceParam[Settings],
) -> pl.DataFrame:
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
    # Use the shared helper to get the partition window with a 1-day lookback.
    partition_start, partition_end, lookback_start = get_partition_window(
        context.partition_key, lookback_days=1
    )
    context.log.info(f"Cleaning partition: {context.partition_key}")

    delta_path = _get_delta_path(settings, "live_primary_flows")

    # Manually load from Delta table and filter to the required date range (including 1 day lookback)
    # This fix is needed because the InMemoryIOManager in tests doesn't have the "yesterday"
    # partition data, and live_primary_flows is a side-effect only asset.
    # We use the new scan_delta_table helper which handles UTC timezone boilerplate.
    live_primary_flows = scan_delta_table(delta_path).filter(
        pl.col("timestamp").is_between(lookback_start, partition_end, closed="left")
    )

    # Materialize the LazyFrame once
    df_joined_materialized = cast(pl.DataFrame, live_primary_flows.collect())

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
    # within the partition's time range. We use the partition's temporal boundaries
    # for the predicate instead of the data's min/max time. This ensures that if
    # the source data is removed, the partition in the Delta table is correctly
    # cleared out (by writing an empty DataFrame), maintaining idempotency.
    delta_path = _get_delta_path(settings, "cleaned_actuals")

    validated_df.write_delta(
        delta_path,
        mode="overwrite",
        delta_write_options={
            "partition_by": ["substation_number"],
            "predicate": f"timestamp >= '{partition_start.isoformat()}' AND timestamp < '{partition_end.isoformat()}'",
        },
    )

    context.log.info(f"Saved cleaned actuals to Delta table at {delta_path}")
    return validated_df
