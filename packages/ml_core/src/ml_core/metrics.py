"""Metric computation for cross-validation results."""

import re
from datetime import datetime

import patito as pt
import polars as pl
from contracts.common import UTC_DATETIME_DTYPE
from contracts.ml_schemas import (
    EVALUATION_SCOPES,
    HORIZON_SLICES,
    METRIC_NAMES,
    METRIC_PARAMS,
    TIME_SERIES_TYPE_SLICES,
    EvalScopeType,
    Metrics,
)
from contracts.power_schemas import (
    EffectiveCapacity,
    PowerForecast,
    PowerTimeSeries,
    TimeSeriesMetadata,
)


def compute_metrics(
    cv_forecasts: pt.DataFrame[PowerForecast],
    actuals: pt.LazyFrame[PowerTimeSeries],
    metadata: pt.DataFrame[TimeSeriesMetadata],
    capacity: pt.DataFrame[EffectiveCapacity],
) -> pt.DataFrame[Metrics]:
    """Compute evaluation metrics from CV predictions and observed power.

    For each ``(time_series_id, fold_id, power_fcst_model_name)`` group:

    1. Joins predictions to observed ``power`` on ``(time_series_id, valid_time)``.
    2. Averages across ``ensemble_member`` to form a deterministic ensemble mean.
    3. Computes MAE, NMAE, RMSE, and MBE.
    4. Joins ``time_series_type`` from ``metadata`` onto each row.
    5. Returns one row per ``(time_series_id, fold_id, power_fcst_model_name,
       horizon_slice, metric_name, metric_param)`` in the tall ``Metrics`` format.

    NMAE is normalised by the pre-computed full-history ``effective_capacity_mw`` (joined per
    ``time_series_id`` from ``capacity``) — a capacity-like denominator. This ``time_series_id``-only
    join is correct while ``capacity`` is one scalar row per series (the MVP). When the v0.6 / v0.7
    differentiable-physics upgrade makes capacity time-varying (one row per ``(time_series_id,
    time)``), this join must become a temporal as-of join on ``(time_series_id, valid_time)``. See
    the "Normalising NMAE by ``effective_capacity``" section of
    ``docs/roadmap/metrics-and-leaderboard.md``.

    Currently only the ``"all"`` horizon slice and ``"all"`` metric_param are computed.
    Horizon-sliced metrics and parametric metrics (Pinball Loss, PICP) can be
    added here later without changing the schema.

    Args:
        cv_forecasts: CV predictions to evaluate.
            Currently only CV fold rows are handled; ``fold_id="live"`` support will
            be added with the ``production_monitoring`` scope in Phase 8.
        actuals: Observed half-hourly power (lazy — only the joined subset is collected).
        metadata: Substation metadata used to join ``time_series_type`` onto each metric row.
            Series absent from ``metadata`` receive a null ``time_series_type``.
        capacity: Pre-computed per-series effective capacity; ``effective_capacity_mw`` is the
            NMAE denominator. Must cover every scored ``time_series_id``.

    Returns:
        A validated tall ``Metrics`` DataFrame with ``time_series_type`` populated.

    Raises:
        ValueError: If no rows survive the inner join (forecasts cover a period with no observed
            data), or if any scored series has no row in ``capacity``.
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
        rmse=(pl.col("error").pow(2).mean()).sqrt(),
        mbe=pl.col("error").mean(),
    )

    # Join the pre-computed full-history effective capacity — the NMAE denominator. Strip the
    # Patito model so Polars' cross-subclass join type check doesn't reject the join.
    capacity_denom = pl.LazyFrame._from_pyldf(capacity.lazy()._ldf).select(
        ["time_series_id", "effective_capacity_mw"]
    )
    wide = metrics_wide.join(capacity_denom, on="time_series_id", how="left").collect()

    # Every scored series must have a capacity row; fail loudly rather than silently emitting a
    # null NMAE (which Metrics.metric_value, a non-nullable Float32, would reject anyway).
    missing = wide.filter(pl.col("effective_capacity_mw").is_null())["time_series_id"]
    if missing.len() > 0:
        raise ValueError(
            f"No effective_capacity row for time_series_id(s) {sorted(set(missing.to_list()))}; "
            "materialise the effective_capacity asset for these series before scoring."
        )
    wide = wide.with_columns(nmae=pl.col("mae") / pl.col("effective_capacity_mw"))

    # Pivot to tall format. The leftover effective_capacity_mw column is dropped by the unpivot.
    metrics_tall = (
        wide.unpivot(
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

    # Join time_series_type from metadata (left — keeps all metric rows; unmatched → null).
    type_map = metadata.select(["time_series_id", "time_series_type"])
    metrics_tall = metrics_tall.join(type_map, on="time_series_id", how="left").with_columns(
        time_series_type=pl.col("time_series_type").cast(pl.Enum(TIME_SERIES_TYPE_SLICES))
    )

    return Metrics.validate(metrics_tall, allow_superfluous_columns=True)


def _type_slug(type_str: str) -> str:
    """Convert a ``time_series_type`` label to an MLflow metric key slug.

    Lowercases, replaces any run of non-alphanumeric characters with a single underscore,
    and strips leading/trailing underscores.  Examples:
    ``"Disaggregated Demand"`` → ``"disaggregated_demand"``,
    ``"Other (Demand)"`` → ``"other_demand"``, ``"PV"`` → ``"pv"``.
    """
    return re.sub(r"[^a-z0-9]+", "_", type_str.lower()).strip("_")


def build_mlflow_aggregate_metrics(
    metrics_df: pl.DataFrame,
) -> dict[str, float]:
    """Return a flat ``{metric_key: value}`` dict for ``mlflow.log_metrics``.

    Computes mean ``metric_value`` across all series (``"all"`` aggregate) and per
    ``time_series_type``, restricted to ``horizon_slice="all"`` and ``metric_param="all"``
    (the scalar metrics). Key format: ``"{metric_name}__all"`` for the overall aggregate
    and ``"{metric_name}__{type_slug}"`` for per-type aggregates.

    Args:
        metrics_df: Per-series ``Metrics`` rows with ``time_series_type`` populated.

    Returns:
        Flat dict of MLflow metric key → mean float value.
    """
    base = metrics_df.filter((pl.col("horizon_slice") == "all") & (pl.col("metric_param") == "all"))

    result: dict[str, float] = {}

    # Per-type aggregates (only for non-null time_series_type values).
    if "time_series_type" in base.columns:
        per_type = (
            base.filter(pl.col("time_series_type").is_not_null())
            .group_by(["metric_name", "time_series_type"])
            .agg(mean_value=pl.col("metric_value").mean())
        )
        for row in per_type.iter_rows(named=True):
            key = f"{row['metric_name']}__{_type_slug(str(row['time_series_type']))}"
            result[key] = float(row["mean_value"])

    # Overall "all" aggregate (includes all series regardless of type).
    overall = base.group_by("metric_name").agg(mean_value=pl.col("metric_value").mean())
    for row in overall.iter_rows(named=True):
        result[f"{row['metric_name']}__all"] = float(row["mean_value"])

    return result


def enrich_metrics_rows(
    per_series_metrics: pt.DataFrame[Metrics],
    experiment_name: str,
    evaluation_scope: EvalScopeType,
    window_start: datetime,
    window_end: datetime,
    window_label: str,
    computed_at: datetime,
    mlflow_run_id: str | None,
) -> pt.DataFrame[Metrics]:
    """Add scope and evaluation-window provenance columns to a per-series Metrics frame.

    Called by the ``metrics`` Dagster asset after ``compute_metrics()`` returns, once the
    window bounds and MLflow run ID are known. Kept here so the enrichment logic is
    unit-testable without Dagster.

    Args:
        per_series_metrics: Frame produced by ``compute_metrics()``.
        experiment_name: Experiment that produced these forecasts.
        evaluation_scope: ``"leaderboard"`` or ``"ad_hoc"``.
        window_start: Inclusive start of the evaluated ``valid_time`` window.
        window_end: Inclusive end of the evaluated ``valid_time`` window.
        window_label: Human-readable label (``fold_id`` for leaderboard; ``"ad_hoc"``).
        computed_at: UTC timestamp when this metric batch was computed.
        mlflow_run_id: MLflow fold run ID; ``None`` for ``ad_hoc``.

    Returns:
        A validated ``Metrics`` DataFrame with all columns fully populated.
    """
    return Metrics.validate(
        per_series_metrics.with_columns(
            experiment_name=pl.lit(experiment_name).cast(pl.Categorical),
            evaluation_scope=pl.lit(evaluation_scope).cast(pl.Enum(EVALUATION_SCOPES)),
            window_start=pl.lit(window_start).cast(UTC_DATETIME_DTYPE),
            window_end=pl.lit(window_end).cast(UTC_DATETIME_DTYPE),
            window_label=pl.lit(window_label),
            computed_at=pl.lit(computed_at).cast(UTC_DATETIME_DTYPE),
            mlflow_run_id=pl.lit(mlflow_run_id),
        )
    )
