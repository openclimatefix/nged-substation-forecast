"""Shared data processing logic for ML models."""

import polars as pl


def downsample_power_flows(flows: pl.LazyFrame) -> pl.LazyFrame:
    """Downsample power flows to 30m using period-ending semantics.

    We assume that NWP data represents the average (or accumulated) value for the
    period *ending* at `valid_time`. For example, a weather forecast for 10:00
    describes the weather from 09:00 to 10:00 (or 09:30 to 10:00).

    To align our targets with these features, we downsample power flows using
    `closed="right", label="right"`. This ensures that power readings from
    09:30 to 10:00 are aggregated and labeled as `10:00`.

    Args:
        flows: Historical power flow data.

    Returns:
        Downsampled power flows.
    """
    return (
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
        .with_columns(
            pl.when(pl.col("MW").is_not_null())
            .then(pl.col("MW"))
            .otherwise(pl.col("MVA"))
            .alias("MW_or_MVA")
        )
    )
