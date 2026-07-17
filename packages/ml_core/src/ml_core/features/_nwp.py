"""NWP processing and power/NWP join helpers.

Handles temporal upsampling (3-hourly → 30-min), processing into a form the feature pipeline
can consume, and the two NWP join modes (bulk training vs. single-run inference).
"""

from datetime import datetime, timedelta
from typing import Final

import polars as pl
from contracts.common import UTC_DATETIME_DTYPE
from contracts.weather_schemas import Nwp

NWP_PUBLICATION_DELAY_HOURS: Final[int] = 6
"""Default delay between an NWP run's ``init_time`` and when it becomes publicly available.

Used to derive ``power_fcst_init_time`` from ``nwp_init_time`` in bulk mode, and to derive
``nwp_init_time`` from ``power_fcst_init_time`` when it is not supplied in single-run mode (see
``_join_nwp_bulk_mode`` / ``_join_nwp_single_run``). Also the default for
``select_nwp_init_time``'s replay-mode cutoff (``ml_core._production_helpers``), which
reconstructs what was actually available at a historical ``power_fcst_init_time``.
"""


def _join_nwp_bulk_mode(
    power_with_metadata: pl.LazyFrame,
    processed_nwp: pl.LazyFrame | None,
    nwp_publication_delay_hours: int,
) -> pl.LazyFrame:
    """NWP-centric join for bulk training / multi-run backtesting.

    Produces one row per (time_series_id, nwp_init_time, valid_time, ensemble_member) with
    power_fcst_init_time derived per-row as nwp_init_time + nwp_publication_delay_hours.
    Each NWP run's first nwp_publication_delay_hours of valid times therefore precede the
    derived power_fcst_init_time; those hindcast rows are kept here so that window features
    (e.g. weather rolling means) see the same predecessor rows as single-run mode, and are
    dropped by ``_engineer_features`` after feature computation.
    """
    if processed_nwp is None:
        result = power_with_metadata.with_columns(
            power_fcst_init_time=pl.col("valid_time"),
            nwp_init_time=pl.lit(None, dtype=UTC_DATETIME_DTYPE),
        )
    else:
        nwp_with_init = processed_nwp.with_columns(
            power_fcst_init_time=pl.col("nwp_init_time")
            + pl.duration(hours=nwp_publication_delay_hours)
        )
        result = nwp_with_init.join(
            power_with_metadata, on=["time_series_id", "valid_time"], how="left"
        )
    return result


def _join_nwp_single_run(
    power_with_metadata: pl.LazyFrame,
    processed_nwp: pl.LazyFrame | None,
    power_fcst_init_time: datetime,
    nwp_init_time: datetime | None,
    nwp_publication_delay_hours: int,
) -> pl.LazyFrame:
    """Power-centric join for single-run production inference or backfilling.

    Stamps a constant power_fcst_init_time across all rows and joins exclusively the one NWP
    run identified by nwp_init_time. If nwp_init_time is None, it is derived as
    power_fcst_init_time - nwp_publication_delay_hours.
    """
    nwp_init_time_val = (
        nwp_init_time
        if nwp_init_time is not None
        else power_fcst_init_time - timedelta(hours=nwp_publication_delay_hours)
    )
    power_with_init = power_with_metadata.with_columns(
        power_fcst_init_time=pl.lit(power_fcst_init_time),
        nwp_init_time=pl.lit(nwp_init_time_val),
    )
    if processed_nwp is None:
        result = power_with_init
    else:
        result = power_with_init.join(
            processed_nwp, on=["time_series_id", "valid_time", "nwp_init_time"], how="left"
        )
    return result


def _upsample_nwp_to_half_hourly(nwp_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Upsample NWP data to 30-minute resolution.

    Assumes 'init_time' has already been renamed to 'nwp_init_time'. Continuous weather
    variables are linearly interpolated within each group; categorical variables are
    forward-filled within each group. All other columns (e.g. nwp_init_time,
    ensemble_member) are used as group-by keys.

    The implementation stays fully lazy: a 30-min time grid is generated per group via
    datetime_ranges + explode, then the original NWP values are left-joined back in, and
    interpolate/forward_fill are applied with over() to stay within group boundaries.

    Leading-null propagation: Polars' interpolate() fills interior nulls but leaves leading
    nulls (before the first non-null value in a group) as null. Some ECMWF ENS variables
    (precipitation, radiation fluxes) are null at lead time 0 by convention. After upsampling
    from native 3-hourly steps, all interpolated 30-min rows before the first non-null step
    remain null — typically a 3-hour window per NWP run. Callers and downstream models should
    treat these as genuinely missing values, not as a data quality issue.
    """
    schema_names = nwp_lf.collect_schema().names()
    all_weather_vars = Nwp.all_weather_var_names()
    group_cols = [
        col for col in schema_names if col != "valid_time" and col not in all_weather_vars
    ]
    continuous_cols = [col for col in schema_names if col in Nwp.continuous_var_names()]
    categorical_cols = [col for col in schema_names if col in Nwp.categorical_var_names]

    # Build 30-min time grid per group: aggregate min/max valid_time per group,
    # expand each row into a list of half-hourly datetimes, then explode to rows.
    time_grid = (
        nwp_lf.group_by(group_cols)
        .agg(
            pl.col("valid_time").min().alias("_start"),
            pl.col("valid_time").max().alias("_end"),
        )
        .with_columns(
            pl.datetime_ranges(
                start=pl.col("_start"),
                end=pl.col("_end"),
                interval="30m",
            ).alias("valid_time")
        )
        .drop("_start", "_end")
        .explode("valid_time")
    )

    # Left-join original NWP onto the grid; new 30-min rows come in as nulls.
    upsampled = time_grid.join(nwp_lf, on=[*group_cols, "valid_time"], how="left").sort(
        [*group_cols, "valid_time"]
    )

    # Fill nulls within each group, never crossing group boundaries.
    # order_by="valid_time" ensures correct temporal ordering within each group window.
    if continuous_cols:
        upsampled = upsampled.with_columns(
            pl.col(col).interpolate().over(group_cols, order_by="valid_time")
            for col in continuous_cols
        )
    if categorical_cols:
        upsampled = upsampled.with_columns(
            pl.col(col).forward_fill().over(group_cols, order_by="valid_time")
            for col in categorical_cols
        )

    return upsampled


def _build_historical_weather(processed_nwp: pl.LazyFrame) -> pl.LazyFrame:
    """Builds the historical weather frame used by weather lag features.

    Uses the control member (ensemble_member == 0) and selects the freshest forecast run
    (shortest lead time) for each (time_series_id, valid_time).
    """
    return (
        processed_nwp.filter(pl.col("ensemble_member") == 0)
        .drop("ensemble_member")
        .group_by(["time_series_id", "valid_time"])
        .agg(pl.all().sort_by("nwp_lead_time_hours").first())
        .sort(["time_series_id", "valid_time"])
    )
