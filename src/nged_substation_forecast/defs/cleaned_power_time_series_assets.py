import dagster as dg
import polars as pl
from typing import cast
from dagster import ResourceParam

from contracts.data_schemas import PowerTimeSeries
from contracts.settings import Settings
from .partitions import SIX_HOURLY_PARTITIONS
from ..utils import scan_delta_table, get_partition_window
from ..cleaning import clean_substation_flows


def _get_delta_path(settings: Settings, table_name: str) -> str:
    """Get the Delta table path for a given table name."""
    return str(settings.nged_data_path / "delta" / table_name)


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
    df_cleaned = clean_substation_flows(df_joined_materialized, settings)

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
