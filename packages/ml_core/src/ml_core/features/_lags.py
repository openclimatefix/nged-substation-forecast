"""Lag and lookahead-bias nullification helpers.

Contains the three functions that implement power lags, weather lags (dual-strategy join),
and post-hoc nullification of leaky lag values.
"""

from typing import Sequence

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
        observed_power_lf: The lookup frame the lagged power is read from, keyed on
            ``valid_time``. This is the dense observed-power series (one row per
            ``(time_series_id, valid_time)``); it carries no ``ensemble_member`` because power
            observations don't vary by ensemble member, so each forecast-instance row joins to
            the single observed value for its lagged time.
        lag_feature: The lag to apply (its ``hours`` and output column name).
    """
    lf_with_target_time = engineered_features_lf.with_columns(
        target_time=pl.col("valid_time") - pl.duration(hours=lag_feature.hours)
    )

    has_ensemble = "ensemble_member" in observed_power_lf.collect_schema().names()
    id_keys = ["time_series_id", "ensemble_member"] if has_ensemble else ["time_series_id"]

    right_lf = observed_power_lf.select(
        *id_keys,
        pl.col("valid_time"),
        pl.col("power").alias(lag_feature.string_repr),
    )

    return lf_with_target_time.join(
        right_lf,
        left_on=[*id_keys, "target_time"],
        right_on=[*id_keys, "valid_time"],
        how="left",
    ).drop("target_time")


def _apply_weather_lag(
    engineered_features_lf: pl.LazyFrame,
    nwp_lf: pl.LazyFrame,
    lag_feature: LagFeature,
    historical_weather_lf: pl.LazyFrame,
) -> pl.LazyFrame:
    """Applies a weather lag using a dual-strategy time-aware join.

    - If target_time > power_fcst_init_time (in the NWP forecast window), uses the exact same
      NWP run (nwp_init_time) and ensemble member as the weather used for valid_time.
    - If target_time <= power_fcst_init_time (in the past), uses the freshest NWP run
      (control member, ensemble 0) for that target time.

    Args:
        engineered_features_lf: The in-progress feature frame this helper attaches one lag
            column to (each row a forecast instance keyed by
            ``time_series_id, valid_time, nwp_init_time, ensemble_member``).
        nwp_lf: The processed NWP frame the same-run lagged weather is read from, keyed on
            ``(time_series_id, nwp_init_time, ensemble_member, valid_time)``.
        lag_feature: The lag to apply (its ``hours`` and output column name).
        historical_weather_lf: The freshest-run weather frame used for past target times.
    """
    base_col = lag_feature.base_col

    lf_with_target_time = engineered_features_lf.with_columns(
        target_time=pl.col("valid_time") - pl.duration(hours=lag_feature.hours)
    )

    # Join 1: Same-Run Join (for target_time > power_fcst_init_time)
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

    # Join 2: Freshest-Run Join (for target_time <= power_fcst_init_time)
    right_freshest_lf = historical_weather_lf.select(
        "time_series_id",
        pl.col("valid_time").alias("target_time"),
        pl.col(base_col).alias(f"{lag_feature.string_repr}_freshest_run"),
    )
    lf_joined = lf_joined.join(right_freshest_lf, on=["time_series_id", "target_time"], how="left")

    return lf_joined.with_columns(
        pl.when(pl.col("target_time") > pl.col("power_fcst_init_time"))
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

    During training, we must ensure that the model cannot access actual data that
    would not be available at inference time. If a requested lag is shorter than
    or equal to the forecast lead time, the feature is effectively a "future"
    value and must be nullified.

    To prevent over-nullification (FLAW-001), we calculate power_lead_time_hours
    relative to power_fcst_init_time, not nwp_init_time.
    """
    lf = lf.with_columns(
        power_lead_time_hours=(
            (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_seconds() / 3600
        ).cast(pl.Float32)
    )

    for feature in leaky_features:
        # >= rather than > is intentionally conservative: when lead_time == lag_hours the
        # lagged observation falls at exactly power_fcst_init_time, which may not yet be
        # published (half-hourly readings arrive with a small comms delay).
        lf = lf.with_columns(
            pl.when(pl.col("power_lead_time_hours") >= feature.hours)
            .then(pl.lit(None))
            .otherwise(pl.col(feature.string_repr))
            .alias(feature.string_repr)
        )

    return lf.drop("power_lead_time_hours")
