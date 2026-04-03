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


def calculate_preferred_power_col(cleaned_actuals: pl.LazyFrame) -> pl.LazyFrame:
    """
    Determine the preferred power column (MW vs MVA) for each substation based on
    data availability and sensor health.

    User decided to ignore temporal leakage in favor of a global decision.
    The 3-month rule ensures we don't train on a dead sensor that was once prolific.

    Logic:
    1. Count valid (non-null) timesteps for both MW and MVA.
    2. Track the last time each sensor was seen (max timestamp of non-null).
    3. Compare valid counts:
       - If only one is valid, pick that one.
       - If counts are equal, prefer MW.
       - Otherwise, pick the one with more valid timesteps.
    4. Dead Sensor Exception: If the chosen sensor stopped reporting > 90 days
       before the end of the series, switch to the other sensor.

    Args:
        cleaned_actuals: LazyFrame containing cleaned power flow data.
            Must contain "substation_number", "timestamp", and the power columns.

    Returns:
        A LazyFrame containing "substation_number" and "preferred_power_col".
    """
    # Calculate valid counts and last seen timestamps per substation
    # We assume 'timestamp' exists and is a date/datetime type.
    stats = cleaned_actuals.group_by("substation_number").agg(
        mw_valid_count=pl.col(POWER_MW).is_not_null().sum(),
        mva_valid_count=pl.col(POWER_MVA).is_not_null().sum(),
        mw_last_seen=pl.col("timestamp").filter(pl.col(POWER_MW).is_not_null()).max(),
        mva_last_seen=pl.col("timestamp").filter(pl.col(POWER_MVA).is_not_null()).max(),
        max_timestamp=pl.col("timestamp").max(),
    )

    # Determine initial choice based on volume
    # Priority: MW > MVA if equal counts
    initial_choice = stats.with_columns(
        preferred_power_col=pl.when(pl.col("mw_valid_count") >= pl.col("mva_valid_count"))
        .then(pl.lit(POWER_MW))
        .otherwise(pl.lit(POWER_MVA)),
    )

    # Apply "Dead Sensor" Exception (90 day rule)
    # If the preferred column's last seen is > 90 days before the total max_timestamp,
    # we switch to the alternative.
    # We use pl.duration to handle the 90-day offset.
    dead_sensor_threshold = pl.duration(days=90)

    return initial_choice.with_columns(
        preferred_power_col=pl.when(
            (pl.col("preferred_power_col") == POWER_MW)
            # If MW is preferred, check if it is dead
            .and_((pl.col("max_timestamp") - pl.col("mw_last_seen")) > dead_sensor_threshold)
            # And ensure MVA actually exists as a fallback
            .and_(pl.col("mva_valid_count") > 0)
        )
        .then(pl.lit(POWER_MVA))
        .when(
            (pl.col("preferred_power_col") == POWER_MVA)
            # If MVA is preferred, check if it is dead
            .and_((pl.col("max_timestamp") - pl.col("mva_last_seen")) > dead_sensor_threshold)
            # And ensure MW actually exists as a fallback
            .and_(pl.col("mw_valid_count") > 0)
        )
        .then(pl.lit(POWER_MW))
        .otherwise(pl.col("preferred_power_col")),
    ).select(["substation_number", "preferred_power_col"])


def calculate_target_map(
    flows: pt.LazyFrame[SubstationPowerFlows] | pt.DataFrame[SubstationPowerFlows],
) -> pt.DataFrame[SubstationTargetMap]:
    """Calculate the target map (power_col and peak_capacity) for each substation.

    This function analyzes historical power flows to determine whether MW or MVA
    is the more reliable target variable (based on data availability) and
    calculates the peak capacity for normalization.

    Args:
        flows: Historical power flow data (LazyFrame or DataFrame).

    Returns:
        A Patito DataFrame containing the target map for each substation.
    """
    target_map_df = cast(
        pl.DataFrame,
        flows.lazy()
        .group_by("substation_number")
        .agg(
            mw_count=pl.col(POWER_MW).is_not_null().sum(),
            mva_count=pl.col(POWER_MVA).is_not_null().sum(),
            peak_capacity_MW_or_MVA=pl.max_horizontal(
                pl.col(POWER_MW).abs().max(), pl.col(POWER_MVA).abs().max()
            ).fill_null(1.0),
        )
        .with_columns(
            pl.when(pl.col("mw_count") >= pl.col("mva_count"))
            .then(pl.lit(POWER_MW))
            .otherwise(pl.lit(POWER_MVA))
            .alias("power_col"),
            pl.when(pl.col("peak_capacity_MW_or_MVA") == 0.0)
            .then(pl.lit(1.0))
            .otherwise(pl.col("peak_capacity_MW_or_MVA"))
            .alias("peak_capacity_MW_or_MVA"),
        )
        .select(["substation_number", "power_col", "peak_capacity_MW_or_MVA"])
        .collect(),
    )

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
            target_map_lazy.select(["substation_number", "power_col"]),
            on="substation_number",
            how="left",
        )
        .with_columns(
            pl.when(pl.col("power_col") == POWER_MVA)
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
