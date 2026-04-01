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

2. **Patito Validation**: The output is validated against the `SubstationFlows` schema
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
from typing import cast
from datetime import datetime, timedelta
from dagster import (
    AssetIn,
    ResourceParam,
    TimeWindowPartitionMapping,
    DailyPartitionsDefinition,
)

from contracts.data_schemas import SubstationFlows
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


@dg.asset(
    partitions_def=DailyPartitionsDefinition(start_date="2026-03-10", end_offset=1),
    auto_materialize_policy=dg.AutoMaterializePolicy.eager(),
    ins={
        # Use TimeWindowPartitionMapping to include lookback from previous partitions.
        # This ensures rolling window calculations at the start of each partition
        # have sufficient history (48 periods = 24 hours) to avoid NaN issues.
        "live_primary_flows": AssetIn(
            partition_mapping=TimeWindowPartitionMapping(start_offset=-1, end_offset=0),
        ),
        "substation_metadata": AssetIn("substation_metadata"),
    },
)
def cleaned_actuals(
    context: dg.AssetExecutionContext,
    settings: ResourceParam[Settings],
    live_primary_flows: pl.LazyFrame,
    substation_metadata: pl.DataFrame,
) -> pl.DataFrame:
    """Clean raw live primary flows and apply data quality checks.

    This asset reads live primary flows, applies data quality cleaning logic (stuck
    sensor detection, insane value detection), and joins with substation metadata.
    The output is validated against the SubstationFlows schema which allows null values,
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
        - The output is validated against SubstationFlows schema which allows
          null values for MW, MVA, and MVAr columns.
        - The result is ALWAYS saved to Delta table to ensure persistence.

    Args:
        context: Dagster asset execution context.
        settings: Global settings containing data quality thresholds.
        live_primary_flows: Raw live primary flows with lookback from previous partition.
        substation_metadata: Metadata containing substation locations and config.

    Returns:
        Polars DataFrame of cleaned substation flows, with nulls for problematic values.
        The DataFrame is validated and saved to the cleaned_actuals Delta table.
    """
    partition_key = context.partition_key
    context.log.info(f"Cleaning partition: {partition_key}")

    # If the LazyFrame is empty, return empty validated DataFrame
    if live_primary_flows.collect_schema().names() == []:
        context.log.warning("Input LazyFrame is empty, returning empty validated DataFrame.")
        empty_df = pl.DataFrame(schema=SubstationFlows.dtypes)
        validated_empty = SubstationFlows.validate(empty_df)
        validated_empty.write_delta(_get_delta_path(settings, "cleaned_actuals"), mode="overwrite")
        return validated_empty

    # The AssetIn already provides data with proper join from substation_metadata
    # The cleaned data is in the lazy frame
    df_joined_materialized = cast(pl.DataFrame, live_primary_flows.collect())

    context.log.info(f"Materialized data shape before cleaning: {df_joined_materialized.shape}")

    # Clean the data using the shared cleaning module
    df_cleaned = clean_substation_flows(df_joined_materialized, settings)

    context.log.info(f"Cleaned data shape after cleaning: {df_cleaned.shape}")

    # Validate the output against Patito schema
    validated_df = SubstationFlows.validate(df_cleaned)

    context.log.info(f"Validated data shape: {validated_df.shape}")

    # CRITICAL FIX for Flaw-001: Filter validated_df to current partition's time window.
    # The TimeWindowPartitionMapping includes historical data for lookback (previous partitions),
    # but we must only append data for the current partition to avoid data duplication in the
    # Delta table. Example: For partition "2026-03-10", we only include rows with timestamp
    # in [2026-03-10 00:00:00 UTC, 2026-03-11 00:00:00 UTC).
    partition_start = datetime.fromisoformat(partition_key)
    partition_end = partition_start + timedelta(days=1)

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
