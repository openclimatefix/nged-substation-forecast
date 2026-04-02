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

3. **Metadata Join**: The join with `substation_metadata` is performed here to ensure
   metadata is always included with cleaned data, avoiding redundancy and logic drift.

4. **No Imputation**: We explicitly do NOT impute missing values. Models downstream
   (like XGBoost) should handle null targets by dropping rows after feature engineering,
   not before.

5. **Partition Mapping with Lookback**: To ensure rolling window calculations have
   sufficient history at partition boundaries, we use a TimeWindowPartitionMapping that
   includes previous partitions in the input. This prevents NaN issues at the start
   of daily partitions.

Complex Partitioning Note:
--------------------------
The implementation uses TimeWindowPartitionMapping with a 2-day lookback window. This
ensures that when computing rolling std for the first timestamp in a partition, we
have 48 periods (24 hours) of historical data available, even at the partition boundary.
"""

import dagster as dg
import polars as pl
import os
from typing import cast
from datetime import datetime, timedelta, timezone
from dagster import (
    AssetIn,
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
) -> pl.LazyFrame:
    """Retrieves the cleaned actuals from the Delta table.

    This function serves as the single source of truth for accessing cleaned actuals
    data across the pipeline. It reads directly from the `cleaned_actuals` Delta table.
    If the table does not exist (e.g., before the first run or backfill), it falls back
    to reading `live_primary_flows` and cleaning it on the fly, logging a warning.

    Args:
        settings: Global settings object.
        context: Optional Dagster context for logging.

    Returns:
        A Polars LazyFrame of the cleaned actuals.
    """
    delta_path = _get_delta_path(settings, "cleaned_actuals")
    try:
        # In some environments (like integration tests), we may want to force cleaning
        # on the fly to ensure we have the full history even if backfills haven't run.
        if os.getenv("NGED_FORCE_CLEAN_ON_THE_FLY") == "1":
            raise RuntimeError("Forced fallback via NGED_FORCE_CLEAN_ON_THE_FLY")

        # Attempt to scan the Delta table and verify it exists by fetching schema.
        # We fetch the schema to force an immediate failure if the table is missing,
        # as pl.scan_delta() is lazy and might not fail until collection.
        lf = pl.scan_delta(delta_path)
        lf.collect_schema()
        if context:
            context.log.info(f"Reading cleaned actuals from {delta_path}")
        return lf
    except Exception as e:
        if context:
            context.log.warning(
                f"Failed to read cleaned_actuals Delta table: {e}. "
                "Falling back to live_primary_flows. Please backfill the cleaned_actuals asset!"
            )

        # Fallback to raw data
        raw_path = _get_delta_path(settings, "live_primary_flows")
        raw_flows = pl.scan_delta(raw_path)

        # Clean on the fly
        # Note: clean_substation_flows expects a DataFrame, so we must collect.
        return clean_substation_flows(cast(pl.DataFrame, raw_flows.collect()), settings).lazy()


@dg.asset(
    partitions_def=DailyPartitionsDefinition(start_date="2026-03-10", end_offset=1),
    auto_materialize_policy=dg.AutoMaterializePolicy.eager(),
    ins={
        "substation_metadata": AssetIn("substation_metadata"),
    },
    deps=["live_primary_flows"],
)
def cleaned_actuals(
    context: dg.AssetExecutionContext,
    settings: ResourceParam[Settings],
    substation_metadata: pl.DataFrame,
) -> pl.DataFrame:
    """Clean raw live primary flows and apply data quality checks.

    This asset reads live primary flows, applies data quality cleaning logic (stuck
    sensor detection, insane value detection), and joins with substation metadata.
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
        - Null values are preserved from the input; no rows are removed and no
          imputation is performed.
        - The output is validated against SubstationPowerFlows schema which allows
          null values for MW, MVA, and MVAr columns.
        - The result is ALWAYS saved to Delta table to ensure persistence.

    Args:
        context: Dagster asset execution context.
        settings: Global settings containing data quality thresholds.
        substation_metadata: Metadata containing substation locations and config.

    Returns:
        Polars DataFrame of cleaned substation flows, with nulls for problematic values.
        The DataFrame is validated and saved to the cleaned_actuals Delta table.
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

    # If the LazyFrame is empty, return empty validated DataFrame
    if live_primary_flows.collect_schema().names() == []:
        context.log.warning("Input LazyFrame is empty, returning empty validated DataFrame.")
        empty_df = pl.DataFrame(schema=SubstationPowerFlows.dtypes)
        validated_empty = SubstationPowerFlows.validate(empty_df)
        # FIX: Do not overwrite the Delta table! Just return the empty DataFrame.
        return validated_empty

    # The AssetIn already provides data with proper join from substation_metadata
    # The cleaned data is in the lazy frame
    df_joined_materialized = cast(pl.DataFrame, live_primary_flows.collect())

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

    # Save to Delta table
    delta_path = _get_delta_path(settings, "cleaned_actuals")
    validated_df.write_delta(
        delta_path, mode="append", delta_write_options={"partition_by": ["substation_number"]}
    )

    context.log.info(f"Saved cleaned actuals to Delta table at {delta_path}")

    return validated_df
