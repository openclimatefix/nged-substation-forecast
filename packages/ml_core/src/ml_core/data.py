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
        valid_count_mw=pl.col(POWER_MW).is_not_null().sum(),
        valid_count_mva=pl.col(POWER_MVA).is_not_null().sum(),
        last_seen_mw=pl.col("timestamp").filter(pl.col(POWER_MW).is_not_null()).max(),
        last_seen_mva=pl.col("timestamp").filter(pl.col(POWER_MVA).is_not_null()).max(),
        max_timestamp=pl.col("timestamp").max(),
        peak_capacity_MW_or_MVA=pl.max_horizontal(
            pl.col(POWER_MW).cast(pl.Float32).abs().max(),
            pl.col(POWER_MVA).cast(pl.Float32).abs().max(),
        )
        .fill_null(1.0)
        .clip(lower_bound=1.0),
    )

    # Determine initial choice based on volume (Priority: MW > MVA if equal counts)
    stats = stats.with_columns(
        preferred_power_col=pl.when(pl.col("valid_count_mw") >= pl.col("valid_count_mva"))
        .then(pl.lit(POWER_MW))
        .otherwise(pl.lit(POWER_MVA)),
    )

    # Apply "Dead Sensor" Exception (90 day rule)
    # If the preferred column's last seen is > 90 days before the total max_timestamp,
    # we switch to the alternative.
    dead_sensor_threshold = pl.duration(days=90)

    # Check if each sensor is dead (no data in the last 90 days)
    is_mw_dead = (pl.col("max_timestamp") - pl.col("last_seen_mw")) > dead_sensor_threshold
    is_mva_dead = (pl.col("max_timestamp") - pl.col("last_seen_mva")) > dead_sensor_threshold

    # Determine if we need to switch from the initially preferred column
    switch_to_mva = (
        (pl.col("preferred_power_col") == POWER_MW) & is_mw_dead & (pl.col("valid_count_mva") > 0)
    )
    switch_to_mw = (
        (pl.col("preferred_power_col") == POWER_MVA) & is_mva_dead & (pl.col("valid_count_mw") > 0)
    )

    pref_col_expr = (
        pl.when(switch_to_mva)
        .then(pl.lit(POWER_MVA))
        .when(switch_to_mw)
        .then(pl.lit(POWER_MW))
        .otherwise(pl.col("preferred_power_col"))
    )

    # To avoid type overload issues, we compute the target map entirely lazily
    # and only collect at the very end.
    target_map_lazy = stats.with_columns(
        preferred_power_col=pref_col_expr,
        peak_capacity_MW_or_MVA=pl.col("peak_capacity_MW_or_MVA"),
    ).select(["substation_number", "preferred_power_col", "peak_capacity_MW_or_MVA"])

    target_map_df = cast(pl.DataFrame, target_map_lazy.collect()).cast(
        {
            "substation_number": pl.Int32,
            "peak_capacity_MW_or_MVA": pl.Float32,
        }
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
