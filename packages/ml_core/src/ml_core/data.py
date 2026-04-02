"""Shared data processing logic for ML models."""

from typing import cast

import polars as pl

from contracts.data_schemas import SubstationTargetMap


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
    flows: pl.LazyFrame | pl.DataFrame, target_map: pl.LazyFrame | pl.DataFrame | None = None
) -> pl.LazyFrame:
    """Downsample power flows to 30m using period-ending semantics.

    We assume that NWP data represents the average (or accumulated) value for the
    period *ending* at `valid_time`. For example, a weather forecast for 10:00
    describes the weather from 09:00 to 10:00 (or 09:30 to 10:00).

    To align our targets with these features, we downsample power flows using
    `closed="right", label="right"`. This ensures that power readings from
    09:30 to 10:00 are aggregated and labeled as `10:00`.

    Args:
        flows: Historical power flow data.
        target_map: Optional map of substation_number to target_col (MW or MVA).

    Returns:
        Downsampled power flows.
    """
    downsampled = (
        flows.lazy()
        .sort("timestamp")
        .group_by_dynamic(
            "timestamp",
            every="30m",
            group_by="substation_number",
            closed="right",
            label="right",
        )
        .agg(
            [
                pl.col("MW").mean(),
                pl.col("MVA").mean(),
            ]
        )
    )

    if target_map is not None:
        # Only join the power_col from target_map to avoid bringing in extra columns like peak_capacity
        downsampled = downsampled.join(
            target_map.lazy().select(["substation_number", "power_col"]),
            on="substation_number",
            how="left",
        )

        return (
            downsampled.with_columns(
                mw_count=pl.when(pl.col("power_col").is_null())
                .then(pl.col("MW").is_not_null().sum().over("substation_number"))
                .otherwise(0),
                mva_count=pl.when(pl.col("power_col").is_null())
                .then(pl.col("MVA").is_not_null().sum().over("substation_number"))
                .otherwise(0),
            )
            .with_columns(
                pl.when(pl.col("power_col") == "MW")
                .then(pl.col("MW"))
                .when(pl.col("power_col") == "MVA")
                .then(pl.col("MVA"))
                .when(pl.col("mw_count") >= pl.col("mva_count"))
                .then(pl.col("MW"))
                .otherwise(pl.col("MVA"))
                .alias("MW_or_MVA")
            )
            .drop(["power_col", "mw_count", "mva_count"])
        )
    else:
        return (
            downsampled.with_columns(
                mw_count=pl.col("MW").is_not_null().sum().over("substation_number"),
                mva_count=pl.col("MVA").is_not_null().sum().over("substation_number"),
            )
            .with_columns(
                pl.when(pl.col("mw_count") >= pl.col("mva_count"))
                .then(pl.col("MW"))
                .otherwise(pl.col("MVA"))
                .alias("MW_or_MVA")
            )
            .drop(["mw_count", "mva_count"])
        )
