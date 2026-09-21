"""Lag and lookahead-bias nullification helpers.

Contains the three functions that implement power lags, weather lags (dual-strategy join), and
post-hoc nullification of leaky lag values.

Three instants recur below and are easy to confuse. ``power_fcst_init_time`` is the moment we
issue the power forecast. ``nwp_init_time`` is the moment the weather model ran, which is earlier
by the publication delay. ``valid_time`` is the future half-hour being forecast. Each lag helper
also computes a ``target_time``, the instant a lag looks up: ``valid_time - lag_hours``.
"""

from collections.abc import Sequence

import polars as pl

from ml_core.features._parsed_features import LagFeature, RollingFeature


def _apply_power_lag(
    engineered_features_lf: pl.LazyFrame,
    observed_power_lf: pl.LazyFrame,
    lag_feature: LagFeature,
) -> pl.LazyFrame:
    """Applies a power lag using a time-aware lazy join on ``valid_time - lag_hours``.

    Args:
        engineered_features_lf: The in-progress feature frame this helper attaches one lag
            column to (each row a forecast instance keyed by
            ``time_series_id, valid_time, nwp_init_time, ensemble_member``).
        observed_power_lf: The lookup frame the lagged power is read from, keyed on ``valid_time``.
            The lookup frame is the dense observed-power series, one row per ``(time_series_id,
            valid_time)``. The frame carries no ``ensemble_member``, because power observations do
            not vary by ensemble member. Each forecast-instance row therefore joins to the single
            observed value for its lagged time.
        lag_feature: The lag to apply (its ``hours`` and output column name).

    Returns:
        ``engineered_features_lf`` with one new column added, named ``lag_feature.string_repr``. The
        column holds the observed power at ``valid_time - lag_hours`` for that row's
        ``time_series_id``. The column is null where no observation exists at that lagged time.
        Still lazy; the left join preserves every row of ``engineered_features_lf``, though not
        necessarily their order.
    """
    lf_with_target_time = engineered_features_lf.with_columns(
        target_time=pl.col("valid_time") - pl.duration(hours=lag_feature.hours)
    )

    right_lf = observed_power_lf.select(
        pl.col("time_series_id"),
        pl.col("valid_time"),
        pl.col("power").alias(lag_feature.string_repr),
    )

    return lf_with_target_time.join(
        right_lf,
        left_on=["time_series_id", "target_time"],
        right_on=["time_series_id", "valid_time"],
        how="left",
    ).drop("target_time")


def _apply_weather_lag(
    engineered_features_lf: pl.LazyFrame,
    nwp_lf: pl.LazyFrame,
    lag_feature: LagFeature,
    historical_weather_lf: pl.LazyFrame,
) -> pl.LazyFrame:
    """Applies a weather lag using a dual-strategy time-aware join.

    - If target_time >= power_fcst_init_time, the target time sits in the NWP forecast window,
      boundary included. The lag then uses the exact same NWP run (nwp_init_time) and ensemble
      member as the weather used for valid_time.
    - If target_time < power_fcst_init_time, the target time is already in the past. The lag then
      uses the freshest NWP run for that target time, reading its control member (ensemble_member
      0, the one unperturbed member). That freshest control member stands in for the weather that
      actually happened, which is why it is called the analysis proxy.

    Args:
        engineered_features_lf: The in-progress feature frame this helper attaches one lag
            column to (each row a forecast instance keyed by
            ``time_series_id, valid_time, nwp_init_time, ensemble_member``).
        nwp_lf: The processed NWP frame the same-run lagged weather is read from, keyed on
            ``(time_series_id, nwp_init_time, ensemble_member, valid_time)``.
        lag_feature: The lag to apply (its ``hours`` and output column name).
        historical_weather_lf: The freshest-run weather frame used for past target times.

    Returns:
        ``engineered_features_lf`` with one new column added, named ``lag_feature.string_repr``. For
        a target time in the forecast window (``target_time >= power_fcst_init_time``), the column
        holds the same-run NWP value. For a target time in the past, the column holds the
        freshest-run value. The column is null where neither join finds a match. Still lazy; the two
        left joins preserve every row of ``engineered_features_lf``, though not necessarily their
        order.
    """
    base_col = lag_feature.base_col

    lf_with_target_time = engineered_features_lf.with_columns(
        target_time=pl.col("valid_time") - pl.duration(hours=lag_feature.hours)
    )

    # Join 1: Same-Run Join (for target_time >= power_fcst_init_time)
    right_same_lf = nwp_lf.select(
        "time_series_id",
        "nwp_init_time",
        "ensemble_member",
        pl.col("valid_time").alias("target_time"),
        pl.col(base_col).alias(f"{lag_feature.string_repr}_same_run"),
    )

    lf_joined = lf_with_target_time.join(
        right_same_lf,
        on=["time_series_id", "nwp_init_time", "ensemble_member", "target_time"],
        how="left",
    )

    # Join 2: Freshest-Run Join (for target_time < power_fcst_init_time)
    right_freshest_lf = historical_weather_lf.select(
        "time_series_id",
        pl.col("valid_time").alias("target_time"),
        pl.col(base_col).alias(f"{lag_feature.string_repr}_freshest_run"),
    )
    lf_joined = lf_joined.join(right_freshest_lf, on=["time_series_id", "target_time"], how="left")

    return lf_joined.with_columns(
        # >= rather than >: at target_time == power_fcst_init_time the row's own run is already
        # available. Bulk mode — the mode that vectorises over every NWP run in the input — derives
        # power_fcst_init_time = nwp_init_time + delay, which is exactly the instant that run
        # becomes usable. No fresher run can be available at that instant either, so the boundary
        # belongs to the same-run branch rather than the freshest-run branch. The boundary choice
        # changes output for every non-control ensemble member at that boundary. The freshest-run
        # branch is `select_analysis_proxy`, which keeps only `ensemble_member == 0`. A `>` boundary
        # would therefore read member 5's own row where `>=` reads member 0's row. The difference is
        # inconsequential today only because `conf/model/xgboost.yaml` selects no weather-lag
        # features.
        pl.when(pl.col("target_time") >= pl.col("power_fcst_init_time"))
        .then(pl.col(f"{lag_feature.string_repr}_same_run"))
        .otherwise(pl.col(f"{lag_feature.string_repr}_freshest_run"))
        .alias(lag_feature.string_repr)
    ).drop(
        [
            f"{lag_feature.string_repr}_same_run",
            f"{lag_feature.string_repr}_freshest_run",
            "target_time",
        ]
    )


def _nullify_leaky_lags(
    lf: pl.LazyFrame, leaky_features: Sequence[LagFeature | RollingFeature]
) -> pl.LazyFrame:
    """Nullifies lagged features that would cause lookahead bias.

    During training, we must ensure that the model cannot access actual data that would not be
    available at inference time. If a requested lag is shorter than or equal to the forecast lead
    time, the feature is effectively a "future" value and must be nullified.

    The lead time is `valid_time - power_fcst_init_time`, and this function computes it as the
    column `power_lead_time_hours`. The lead time is measured from power_fcst_init_time, never
    from nwp_init_time. nwp_init_time is earlier than power_fcst_init_time by the publication
    delay, so measuring from it would make every lead time look longer and would nullify lags
    that are in fact safe.
    """
    lf = lf.with_columns(
        power_lead_time_hours=(
            (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_seconds() / 3600
        ).cast(pl.Float32)
    )

    for feature in leaky_features:
        # >= rather than > is intentionally conservative. When lead_time == lag_hours, the lagged
        # observation falls at exactly power_fcst_init_time. That observation may not yet be
        # published, because half-hourly readings arrive with a small comms delay.
        lf = lf.with_columns(
            pl.when(pl.col("power_lead_time_hours") >= feature.hours)
            .then(pl.lit(None))
            .otherwise(pl.col(feature.string_repr))
            .alias(feature.string_repr)
        )

    return lf.drop("power_lead_time_hours")
