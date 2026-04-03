"""Shared data processing logic for ML models."""

from typing import cast

import patito as pt
import polars as pl
from contracts.data_schemas import SubstationPowerFlows, SubstationTargetMap


def calculate_target_map(flows: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
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
            mw_count=pl.col("MW").is_not_null().sum(),
            mva_count=pl.col("MVA").is_not_null().sum(),
            peak_capacity_MW_or_MVA=pl.max_horizontal(
                pl.col("MW").abs().max(), pl.col("MVA").abs().max()
            ).fill_null(1.0),
        )
        .with_columns(
            pl.when(pl.col("mw_count") >= pl.col("mva_count"))
            .then(pl.lit("MW"))
            .otherwise(pl.lit("MVA"))
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
    flows: pt.DataFrame[SubstationPowerFlows] | pl.LazyFrame,
    target_map: pt.DataFrame[SubstationTargetMap] | pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    """Downsample power flows to 30m using period-ending semantics.

    We assume that NWP data represents the average (or accumulated) value for the
    period *ending* at `valid_time`. For example, a weather forecast for 10:00
    describes the weather from 09:00 to 10:00.

    To align our targets with these features, we downsample power flows using
    `closed="right", label="right"`. This ensures that power readings from
    09:30 to 10:00 are aggregated and labeled as `10:00`.

    Args:
        flows: Historical power flow data.
        target_map: Optional map of substation_number to target_col (MW or MVA).

    Returns:
        Downsampled power flows.
    """
    flows_lazy = flows.lazy() if isinstance(flows, pl.DataFrame) else flows

    if target_map is not None:
        target_map_lazy = target_map.lazy() if isinstance(target_map, pl.DataFrame) else target_map

        # Select the correct column before downsampling to avoid expensive operations on both MW and MVA
        flows_lazy = (
            flows_lazy.join(
                target_map_lazy.select(["substation_number", "power_col"]),
                on="substation_number",
                how="left",
            )
            .with_columns(
                pl.when(pl.col("power_col") == "MVA")
                .then(pl.col("MVA"))
                .otherwise(pl.col("MW"))
                .alias("MW_or_MVA")
            )
            .select(["timestamp", "substation_number", "MW_or_MVA"])
        )
    else:
        # Use to_simplified_substation_power_flows to pick the column
        if isinstance(flows, pl.LazyFrame):
            flows_df = pt.DataFrame[SubstationPowerFlows](flows.collect())
        else:
            flows_df = pt.DataFrame[SubstationPowerFlows](flows)

        simplified_df = SubstationPowerFlows.to_simplified_substation_power_flows(flows_df)
        flows_lazy = simplified_df.lazy()

    # Downsample the single MW_or_MVA column
    return (
        flows_lazy.sort("timestamp")
        .group_by_dynamic(
            "timestamp",
            every="30m",
            group_by="substation_number",
            closed="right",
            label="right",
        )
        .agg(pl.col("MW_or_MVA").mean())
    )
