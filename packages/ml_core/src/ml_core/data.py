"""Shared data processing logic for ML models."""

from typing import cast

import patito as pt
import polars as pl
from contracts.data_schemas import (
    POWER_MVA,
    POWER_MW,
    POWER_MW_OR_MVA,
    SimplifiedSubstationPowerFlows,
    SubstationPowerFlows,
    SubstationTargetMap,
)


# (Delete previous placeholder)


def calculate_target_map(
    flows: pt.LazyFrame[SubstationPowerFlows] | pt.DataFrame[SubstationPowerFlows],
) -> pt.DataFrame[SubstationTargetMap]:
    """Calculate the target map (power_col and peak_capacity) for each substation.

    This function analyzes historical power flows to determine whether MW or MVA
    is the more reliable target variable (based on data availability) and
    calculates the peak capacity for normalization.

    POTENTIAL DATA LEAKAGE: The 90-day dead sensor rule uses the entire history,
    introducing temporal leakage, but the user has consciously accepted this to
    simplify the global decision.

    Args:
        flows: Historical power flow data (LazyFrame or DataFrame).

    Returns:
        A Patito DataFrame containing the target map for each substation.
    """
    flows_lazy = flows.lazy() if isinstance(flows, pl.DataFrame) else flows

    # Calculate valid counts, last seen timestamps, and peak capacity
    stats = flows_lazy.group_by("substation_number").agg(
        mw_valid_count=pl.col(POWER_MW).is_not_null().sum(),
        mva_valid_count=pl.col(POWER_MVA).is_not_null().sum(),
        mw_last_seen=pl.col("timestamp").filter(pl.col(POWER_MW).is_not_null()).max(),
        mva_last_seen=pl.col("timestamp").filter(pl.col(POWER_MVA).is_not_null()).max(),
        max_timestamp=pl.col("timestamp").max(),
        peak_capacity_MW_or_MVA=pl.max_horizontal(
            pl.col(POWER_MW).cast(pl.Float32).abs().max(),
            pl.col(POWER_MVA).cast(pl.Float32).abs().max(),
        ).fill_null(1.0),
    )

    # Determine initial choice based on volume (Priority: MW > MVA if equal counts)
    stats = stats.with_columns(
        preferred_power_col=pl.when(pl.col("mw_valid_count") >= pl.col("mva_valid_count"))
        .then(pl.lit(POWER_MW))
        .otherwise(pl.lit(POWER_MVA)),
    )

    # Apply "Dead Sensor" Exception (90 day rule)
    # If the preferred column's last seen is > 90 days before the total max_timestamp,
    # we switch to the alternative.
    dead_sensor_threshold = pl.duration(days=90)

    # Use with_columns with expressions to avoid overload issues.
    # We compute the expressions first to keep the with_columns call clean.
    pref_col_expr = (
        pl.when(
            (pl.col("preferred_power_col") == POWER_MW)
            .and_((pl.col("max_timestamp") - pl.col("mw_last_seen")) > dead_sensor_threshold)
            .and_(pl.col("mva_valid_count") > 0)
        )
        .then(pl.lit(POWER_MVA))
        .when(
            (pl.col("preferred_power_col") == POWER_MVA)
            .and_((pl.col("max_timestamp") - pl.col("mva_last_seen")) > dead_sensor_threshold)
            .and_(pl.col("mw_valid_count") > 0)
        )
        .then(pl.lit(POWER_MW))
        .otherwise(pl.col("preferred_power_col"))
    )

    peak_cap_expr = (
        pl.when(pl.col("peak_capacity_MW_or_MVA") == 0.0)
        .then(pl.lit(1.0))
        .otherwise(pl.col("peak_capacity_MW_or_MVA"))
    )

    # To avoid type overload issues, we compute the target map entirely lazily
    # and only collect at the very end.
    target_map_lazy = stats.with_columns(
        preferred_power_col=pref_col_expr,
        peak_capacity_MW_or_MVA=peak_cap_expr,
    ).select(["substation_number", "preferred_power_col", "peak_capacity_MW_or_MVA"])

    target_map_df = target_map_lazy.collect()

    # Explicitly cast to DataFrame to ensure the subsequent .with_columns works on a materialized DF
    target_map_df = cast(pl.DataFrame, target_map_df)

    target_map_df = target_map_df.with_columns(
        [
            pl.col("substation_number").cast(pl.Int32),
            pl.col("peak_capacity_MW_or_MVA").cast(pl.Float32),
        ]
    )

    return SubstationTargetMap.validate(target_map_df)


def downsample_power_flows(
    flows: pt.DataFrame[SubstationPowerFlows] | pt.LazyFrame[SubstationPowerFlows],
    target_map: pt.DataFrame[SubstationTargetMap] | pt.LazyFrame[SubstationTargetMap],
) -> pt.LazyFrame[SimplifiedSubstationPowerFlows]:
    """Downsample power flows to 30m using period-ending semantics.

    We assume that NWP data represents the average (or accumulated) value for the
    period *ending* at `valid_time`. For example, a weather forecast for 10:00
    describes the weather from 09:00 to 10:00.

    To align our targets with these features, we downsample power flows using
    `closed="right", label="right"`. This ensures that power readings from
    09:30 to 10:00 are aggregated and labeled as `10:00`.

    Args:
        flows: Historical power flow data.
        target_map: Map of substation_number to target_col (MW or MVA).

    Returns:
        Downsampled power flows.
    """
    flows_lazy = flows.lazy() if isinstance(flows, pl.DataFrame) else flows
    target_map_lazy = target_map.lazy() if isinstance(target_map, pl.DataFrame) else target_map

    # Select the correct column before downsampling to avoid expensive operations on both MW and MVA
    flows_lazy = (
        flows_lazy.join(
            target_map_lazy.select(["substation_number", "preferred_power_col"]),
            on="substation_number",
            how="left",
        )
        .with_columns(
            pl.when(pl.col("preferred_power_col") == POWER_MVA)
            .then(pl.col(POWER_MVA))
            .otherwise(pl.col(POWER_MW))
            .alias(POWER_MW_OR_MVA)
        )
        .select(["timestamp", "substation_number", POWER_MW_OR_MVA])
    )

    # Downsample the single MW_or_MVA column
    return cast(
        pt.LazyFrame[SimplifiedSubstationPowerFlows],
        (
            flows_lazy.sort("timestamp")
            .group_by_dynamic(
                "timestamp",
                every="30m",
                group_by="substation_number",
                closed="right",
                label="right",
            )
            .agg(pl.col(POWER_MW_OR_MVA).mean())
        ),
    )
