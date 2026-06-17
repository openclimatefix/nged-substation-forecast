"""Metric computation for cross-validation results."""

import patito as pt
import polars as pl
from contracts.ml_schemas import (
    HORIZON_SLICES,
    METRIC_NAMES,
    METRIC_PARAMS,
    Metrics,
)
from contracts.power_schemas import PowerForecast, PowerTimeSeries


def compute_metrics(
    cv_forecasts: pt.DataFrame[PowerForecast],
    actuals: pt.LazyFrame[PowerTimeSeries],
) -> pt.DataFrame[Metrics]:
    """Compute evaluation metrics from CV predictions and observed power.

    For each ``(time_series_id, fold_id, power_fcst_model_name)`` group:

    1. Joins predictions to observed ``power`` on ``(time_series_id, valid_time)``.
    2. Averages across ``ensemble_member`` to form a deterministic ensemble mean.
    3. Computes MAE, NMAE, RMSE, and MBE.
    4. Returns one row per ``(time_series_id, fold_id, power_fcst_model_name,
       horizon_slice, metric_name, metric_param)`` in the tall ``Metrics`` format.

    NMAE is normalised by the mean absolute observed power within the same group,
    making it comparable across substations of different sizes.

    Currently only the ``"all"`` horizon slice and ``"all"`` metric_param are computed.
    Horizon-sliced metrics and parametric metrics (Pinball Loss, PICP) can be
    added here later without changing the schema.

    Args:
        cv_forecasts: CV predictions from :func:`ml_core.cross_validate.cross_validate`.
            Must not contain ``fold_id="live"`` rows — pass only CV fold predictions.
        actuals: Observed half-hourly power (lazy — only the joined subset is collected).

    Returns:
        A validated tall ``Metrics`` DataFrame.

    Raises:
        ValueError: If no rows survive the inner join (forecasts cover a period
            with no observed data).
    """
    # Join forecasts to actuals; rename power → power_actual to avoid shadowing.
    # Strip the Patito model subclass from actuals so that Polars' cross-subclass
    # type check (assert_same_type) doesn't reject a join between two differently-typed
    # pt.LazyFrame objects.
    actuals_plain = pl.LazyFrame._from_pyldf(actuals._ldf)
    joined = cv_forecasts.lazy().join(
        actuals_plain.select(["time_series_id", "time", "power"]).rename({"power": "power_actual"}),
        left_on=["time_series_id", "valid_time"],
        right_on=["time_series_id", "time"],
        how="inner",
    )

    # Average across ensemble members to produce a deterministic ensemble mean.
    ensemble_mean = joined.group_by(
        ["time_series_id", "fold_id", "power_fcst_model_name", "valid_time"]
    ).agg(
        power_fcst=pl.col("power_fcst").mean(),
        power_actual=pl.col("power_actual").first(),
    )

    # Compute error column once, reuse in all aggregations.
    with_error = ensemble_mean.with_columns(error=pl.col("power_fcst") - pl.col("power_actual"))

    # Wide metrics: one row per (time_series_id, fold_id, power_fcst_model_name).
    metrics_wide = with_error.group_by(["time_series_id", "fold_id", "power_fcst_model_name"]).agg(
        mae=pl.col("error").abs().mean(),
        nmae=pl.col("error").abs().mean() / pl.col("power_actual").abs().mean(),
        rmse=(pl.col("error").pow(2).mean()).sqrt(),
        mbe=pl.col("error").mean(),
    )

    # Pivot to tall format.
    metrics_tall = (
        metrics_wide.collect()
        .unpivot(
            on=["mae", "nmae", "rmse", "mbe"],
            index=["time_series_id", "fold_id", "power_fcst_model_name"],
            variable_name="metric_name",
            value_name="metric_value",
        )
        .cast(
            {
                "metric_name": pl.Enum(METRIC_NAMES),
                "metric_value": pl.Float32,
            }
        )
        .with_columns(
            horizon_slice=pl.lit("all").cast(pl.Enum(HORIZON_SLICES)),
            metric_param=pl.lit("all").cast(pl.Enum(METRIC_PARAMS)),
            power_fcst_model_name=pl.col("power_fcst_model_name").cast(pl.Categorical),
            fold_id=pl.col("fold_id").cast(pl.Categorical),
        )
    )

    if metrics_tall.is_empty():
        raise ValueError(
            "No rows in the joined forecast/actuals data. "
            "Check that cv_forecasts and actuals overlap in time."
        )

    return Metrics.validate(metrics_tall, allow_superfluous_columns=True)
