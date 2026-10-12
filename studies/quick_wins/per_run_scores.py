"""Per-row CRPS and absolute error of an ensemble forecast, scored as the leaderboard scores it.

`ml_core.metrics.compute_metrics` returns metrics already averaged per series and horizon slice,
so a month-resampled interval cannot be drawn from its output. This module runs the same
per-forecast-run collapse that `compute_metrics` runs, and returns one row per
`(time_series_id, power_fcst_init_time, valid_time)` instead. It borrows the fair-CRPS and
horizon-slice expressions from `ml_core.metrics` so the two cannot drift apart, and the tests check
that its means equal `compute_metrics` on the same frames. Importing those two private
expressions is deliberate: a copy would drift from the leaderboard's definition.
"""

from typing import Final

import polars as pl
from ml_core.metrics import _fair_crps_expr, _horizon_slice_expr

FORECAST_COLUMNS: Final[tuple[str, ...]] = (
    "time_series_id",
    "power_fcst_init_time",
    "valid_time",
    "ensemble_member",
    "power_fcst",
)
"""The columns of a forecast frame that `per_run_scores` reads."""

SCORE_KEYS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time")
"""The key of one row of `per_run_scores`: one forecast run for one valid time of one series."""


LEAD_BAND_ORDER: Final[tuple[str, ...]] = ("all", "days 0 to 2", "days 3 to 10")
"""The lead-time bands the study reports, `all` first.

`days 0 to 2` holds leads under 72 hours and `days 3 to 10` holds leads from 72 up to 264 hours.
Leads from 264 hours up belong to `all` only.
"""


def lead_band_means(*, scores: pl.DataFrame) -> pl.DataFrame:
    """Mean CRPS and MAE per series within each band of `LEAD_BAND_ORDER`.

    Args:
        scores: The output of `per_run_scores`, with the columns of `normalised_by_capacity`.

    Returns:
        One row per `(time_series_id, lead_band)` with `crps`, `mae`, and `nmae`.
    """
    bands = {
        "all": pl.lit(value=True),
        "days 0 to 2": pl.col("lead_hours") < 72,
        "days 3 to 10": (pl.col("lead_hours") >= 72) & (pl.col("lead_hours") < 264),
    }
    return pl.concat(
        [
            leaderboard_means(scores=scores.filter(condition)).with_columns(lead_band=pl.lit(band))
            for band, condition in bands.items()
        ]
    )


def per_run_scores(*, forecasts: pl.DataFrame, actuals: pl.DataFrame) -> pl.DataFrame:
    """Score every forecast run for every valid time that has an observation.

    Each group of ensemble members that forecast one `(series, initialisation time, valid time)` is
    collapsed to the ensemble mean and the fair CRPS (Ferro 2014), exactly as `compute_metrics`
    does. Runs are never pooled across initialisation times.

    Args:
        forecasts: One row per ensemble member, with the columns in `FORECAST_COLUMNS`.
        actuals: Observed power with columns `time_series_id`, `time`, and `power`. Duplicate
            `(time_series_id, time)` rows are dropped, because a duplicated observation would
            double every member through the join.

    Returns:
        One row per scored forecast run: `time_series_id`, `power_fcst_init_time`, `valid_time`,
        `horizon_slice`, `lead_hours`, `ensemble_mean`, `actual`, `crps`, and `abs_error`, the
        absolute error of the ensemble mean. Rows whose valid time has no observation are absent.
    """
    observed = (
        actuals.select("time_series_id", "time", "power")
        .unique(subset=["time_series_id", "time"], keep="any")
        .rename({"time": "valid_time", "power": "power_actual"})
    )
    return (
        forecasts.select(*FORECAST_COLUMNS)
        .join(observed, on=["time_series_id", "valid_time"], how="inner")
        .group_by(*SCORE_KEYS)
        .agg(
            ensemble_mean=pl.col("power_fcst").mean(),
            actual=pl.col("power_actual").first(),
            crps=_fair_crps_expr(),
        )
        .with_columns(
            horizon_slice=_horizon_slice_expr(),
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_seconds()
            / 3600,
            abs_error=(pl.col("ensemble_mean") - pl.col("actual")).abs(),
        )
        .sort(*SCORE_KEYS)
    )


def normalised_by_capacity(*, scores: pl.DataFrame, capacity: pl.DataFrame) -> pl.DataFrame:
    """Divide each row's CRPS and absolute error by its own series' effective capacity.

    Args:
        scores: The output of `per_run_scores`.
        capacity: Columns `time_series_id` and `effective_capacity_mw`, one row per series.

    Returns:
        `scores` with `crps_norm` and `abs_error_norm` added (fractions of capacity).

    Raises:
        ValueError: If any scored series has no capacity row.
    """
    if capacity["time_series_id"].n_unique() != capacity.height:
        msg = "capacity must hold exactly one row per series"
        raise ValueError(msg)
    joined = scores.join(
        capacity.select("time_series_id", "effective_capacity_mw"),
        on="time_series_id",
        how="left",
    )
    missing = joined.filter(pl.col("effective_capacity_mw").is_null())["time_series_id"].unique()
    if len(missing):
        msg = f"no effective capacity for series {sorted(missing.to_list())}"
        raise ValueError(msg)
    return joined.with_columns(
        crps_norm=pl.col("crps") / pl.col("effective_capacity_mw"),
        abs_error_norm=pl.col("abs_error") / pl.col("effective_capacity_mw"),
    )


def leaderboard_means(*, scores: pl.DataFrame) -> pl.DataFrame:
    """Mean CRPS and MAE per series, as the leaderboard computes before it averages over series.

    Args:
        scores: The output of `per_run_scores`, optionally with the columns of
            `normalised_by_capacity`.

    Returns:
        One row per series with `crps` and `mae` (in the units of power) and, when the input
        carries capacity, `nmae`.
    """
    aggregations = [pl.col("crps").mean(), pl.col("abs_error").mean().alias("mae")]
    if "abs_error_norm" in scores.columns:
        aggregations.append(pl.col("abs_error_norm").mean().alias("nmae"))
    return scores.group_by("time_series_id").agg(*aggregations).sort("time_series_id")
