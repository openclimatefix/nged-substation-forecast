"""NWP processing and power/NWP join helpers.

This module handles three jobs. The first is temporal upsampling from the numerical weather
prediction (NWP) model's native step width onto the half-hourly grid the power observations sit
on. The second is processing the NWP into a form the feature pipeline can consume. The third is
the two NWP join modes: bulk training, and single-run inference.

Also the home of ``NWP_PUBLICATION_DELAY_HOURS``, the one constant that ties the two modes
together. Bulk mode derives each row's ``power_fcst_init_time`` from the delay. Single-run mode
derives ``nwp_init_time`` from the delay when the caller names no run. The constant is
re-exported from ``ml_core.features``, and its value is argued for on the constant itself.
"""

from datetime import datetime, timedelta
from typing import Final

import polars as pl
from contracts.common import UTC_DATETIME_DTYPE
from contracts.weather_schemas import Nwp

NWP_PUBLICATION_DELAY_HOURS: Final[int] = 9
"""Hours after an NWP run's ``init_time`` before we treat that run as usable.

The delay models when a run reaches *our* disk, not when Dynamical publish that run. Dynamical
publish each 00Z run between 08:05 and 08:20 UTC, and ``ecmwf_ens_schedule`` downloads it at 08:30
UTC. A 00Z run is therefore ours from roughly 08:30, which is 8.5 hours after that run's
``init_time``. Nine is the nearest whole hour at or after 8.5.

The feature pipeline uses the delay to derive ``power_fcst_init_time`` from ``nwp_init_time`` in
bulk mode, and to derive ``nwp_init_time`` when a single-run caller omits ``nwp_init_time``.
``select_nwp_init_time`` uses the delay to reconstruct availability for ``"replay"`` backfills.
``select_analysis_proxy`` needs no delay: in single-run mode ``_engineer_features`` caps the
proxy at the NWP run that call already selected.

Of ``select_nwp_init_time``'s two modes, only ``"replay"`` needs the delay. A live forecast run
joins whatever NWP runs are genuinely on disk, so reality already constrains the NWP table to the
runs that were genuinely published. A replay of a past init time would otherwise join runs that only
landed afterwards — lookahead bias rather than mere inaccuracy. The asymmetry in full:
<https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/#resolve-nwp-availability-asymmetrically-live-vs-replay>

Two bounds constrain the value, given one 00Z run a day and forecast slots at 00/06/12/18 UTC. The
06:00 slot must *not* see that morning's run, which has not landed yet, so the value must exceed 6
hours. The 12:00 slot *must* see that morning's run, so the value must not exceed 12 hours. Both
bounds move if ``ecmwf_ens_schedule``'s start time changes.
"""


def _join_nwp_bulk_mode(
    power_lf: pl.LazyFrame,
    processed_nwp: pl.LazyFrame | None,
    nwp_publication_delay_hours: int,
) -> pl.LazyFrame:
    """NWP-centric join for bulk training / multi-run backtesting.

    Produces one row per (time_series_id, nwp_init_time, valid_time, ensemble_member) with
    power_fcst_init_time derived per-row as nwp_init_time + nwp_publication_delay_hours. Each NWP
    run's first nwp_publication_delay_hours of valid times therefore precede the derived
    power_fcst_init_time. Those hindcast rows are kept here so that window features (e.g. weather
    rolling means) see the same predecessor rows as single-run mode. ``_engineer_features`` drops
    the hindcast rows after feature computation.

    ``power_lf`` carries no metadata: ``_engineer_features`` joins the metadata onto the result,
    because a valid_time with no power observation would otherwise lose its metadata here.
    """
    if processed_nwp is None:
        result = power_lf.with_columns(
            power_fcst_init_time=pl.col("valid_time"),
            nwp_init_time=pl.lit(None, dtype=UTC_DATETIME_DTYPE),
        )
    else:
        nwp_with_init = processed_nwp.with_columns(
            power_fcst_init_time=pl.col("nwp_init_time")
            + pl.duration(hours=nwp_publication_delay_hours)
        )
        result = nwp_with_init.join(power_lf, on=["time_series_id", "valid_time"], how="left")
    return result


def _resolve_nwp_init_time(
    nwp_init_time: datetime | None,
    power_fcst_init_time: datetime,
    nwp_publication_delay_hours: int,
) -> datetime:
    """The single-run NWP run identity: the caller's own value, or the derived fallback.

    Two callers share this function. ``_join_nwp_single_run`` does the join itself, and
    ``tabular_feature_engineer._check_or_warn_on_missing_control_member`` names the run in a
    degradation warning. Sharing keeps the two callers from drifting on what "the selected run"
    means.
    """
    return (
        nwp_init_time
        if nwp_init_time is not None
        else power_fcst_init_time - timedelta(hours=nwp_publication_delay_hours)
    )


def _join_nwp_single_run(
    power_lf: pl.LazyFrame,
    processed_nwp: pl.LazyFrame | None,
    power_fcst_init_time: datetime,
    nwp_init_time: datetime | None,
    nwp_publication_delay_hours: int,
) -> pl.LazyFrame:
    """Power-centric join for single-run production inference or backfilling.

    Stamps a constant power_fcst_init_time across all rows and joins exclusively the one NWP run
    identified by nwp_init_time. If nwp_init_time is None, it is derived as
    power_fcst_init_time - nwp_publication_delay_hours.

    ``power_lf`` carries no metadata — see ``_join_nwp_bulk_mode``.
    """
    nwp_init_time_val = _resolve_nwp_init_time(
        nwp_init_time=nwp_init_time,
        power_fcst_init_time=power_fcst_init_time,
        nwp_publication_delay_hours=nwp_publication_delay_hours,
    )
    power_with_init = power_lf.with_columns(
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

    Assumes 'init_time' has already been renamed to 'nwp_init_time'. Continuous weather variables
    are linearly interpolated within each group; categorical variables are forward-filled within
    each group. All other columns (e.g. nwp_init_time, ensemble_member) are used as group-by
    keys.

    The implementation stays fully lazy. A 30-min time grid is generated per group via
    datetime_ranges + explode. The original NWP values are then left-joined back in. Finally,
    interpolate/forward_fill are applied with over() to stay within group boundaries.

    End-null propagation: Polars' interpolate() fills interior nulls. Leading nulls and trailing
    nulls are both left as null. A leading null sits before the group's first non-null value, and
    a trailing null sits after the group's last non-null value. The weather model this pipeline
    consumes is the European Centre for Medium-Range Weather Forecasts ensemble (ECMWF ENS). Some
    ECMWF ENS variables — precipitation and the radiation fluxes — are accumulated over the step,
    so they have nothing to accumulate at lead time 0 and are null there by convention. Every
    interpolated 30-min row before a group's first non-null native step therefore remains null —
    typically a 3-hour window per NWP run. ECMWF ENS runs at a 3-hour native step width out to
    144 hours, then coarsens to a 6-hour step width for the rest of its 360-hour horizon. The
    trailing case is rarer but real: a wholly-null slice at the last native step of the horizon
    is not bridged either, so that slice too reaches the caller as null. Callers and downstream
    models should treat every one of these nulls as a genuinely missing value rather than as a
    corrupted download to be repaired or imputed.
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
        # empty_as_null=False matches the Polars 2.0 default and silences the deprecation warning.
        # The two settings, empty_as_null=True and empty_as_null=False, differ only when the range
        # comes out empty: True yields a null, False yields an empty list. The setting has no effect
        # on output today. _start/_end are the min/max valid_time of a non-empty group, so start <=
        # end and datetime_ranges returns at least the single-point list [_start]. The empty-list
        # branch the two settings disagree on is therefore unreachable. valid_time is a non-nullable
        # datetime, so _start/_end are never null either. Even a null range would explode to a
        # single null row identically under both settings.
        .explode("valid_time", empty_as_null=False)
    )

    # Left-join original NWP onto the grid; new 30-min rows come in as nulls.
    upsampled = time_grid.join(nwp_lf, on=[*group_cols, "valid_time"], how="left")

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
