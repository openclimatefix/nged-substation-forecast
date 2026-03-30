"""Shared data processing logic for ML models."""

import polars as pl


def downsample_power_flows(
    flows: pl.LazyFrame, target_map: pl.LazyFrame | None = None
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
        flows.sort("timestamp")
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
        downsampled = downsampled.join(target_map, on="substation_number", how="left")

        return (
            downsampled.with_columns(
                mw_count=pl.when(pl.col("target_col").is_null())
                .then(pl.col("MW").is_not_null().sum().over("substation_number"))
                .otherwise(0),
                mva_count=pl.when(pl.col("target_col").is_null())
                .then(pl.col("MVA").is_not_null().sum().over("substation_number"))
                .otherwise(0),
            )
            .with_columns(
                pl.when(pl.col("target_col") == "MW")
                .then(pl.col("MW"))
                .when(pl.col("target_col") == "MVA")
                .then(pl.col("MVA"))
                .when(pl.col("mw_count") >= pl.col("mva_count"))
                .then(pl.col("MW"))
                .otherwise(pl.col("MVA"))
                .alias("MW_or_MVA")
            )
            .drop(["target_col", "mw_count", "mva_count"])
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
