"""Metric computation for cross-validation results."""

import re
from datetime import datetime
from typing import Final

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


class NoOverlappingActualsError(ValueError):
    """Raised by ``compute_metrics`` when no forecast row joins to any observed actual.

    A distinct subclass so callers that score a fold in per-series batches can treat "this
    batch's series have no overlapping actuals" as skippable — mirroring how such series
    silently vanish from the inner join when the whole fold is scored in one call — while
    every other ``ValueError`` (negative lead times, missing capacity) still propagates.
    """


_INTRADAY_MAX_HOURS: Final[int] = 6
"""Exclusive upper bound (hours) of the ``"intraday"`` horizon slice: lead times in [0 h, 6 h)."""

_DAY_AHEAD_MAX_HOURS: Final[int] = 36
"""Exclusive upper bound (hours) of the ``"day_ahead"`` horizon slice: lead times in [6 h, 36 h)."""

_SHORT_MEDIUM_RANGE_MAX_HOURS: Final[int] = 168
"""Exclusive upper bound (hours) of the ``"short_medium_range"`` horizon slice.

Lead times in [36 h, 168 h) — day 2 to day 7. Lead times of 168 h and beyond fall in
``"extended_range"``.
"""


def _horizon_slice_expr() -> pl.Expr:
    """Map each row's lead time onto the ``HORIZON_SLICES`` bands.

    Lead time is ``valid_time − power_fcst_init_time``. Bands are left-closed:
    ``"intraday"`` [0 h, 6 h), ``"day_ahead"`` [6 h, 36 h), ``"short_medium_range"``
    [36 h, 168 h), ``"extended_range"`` [168 h, ∞) — matching the band definitions
    documented on ``contracts.ml_schemas.HORIZON_SLICES``.
    """
    lead_time = pl.col("valid_time") - pl.col("power_fcst_init_time")
    return (
        pl.when(lead_time < pl.duration(hours=_INTRADAY_MAX_HOURS))
        .then(pl.lit("intraday"))
        .when(lead_time < pl.duration(hours=_DAY_AHEAD_MAX_HOURS))
        .then(pl.lit("day_ahead"))
        .when(lead_time < pl.duration(hours=_SHORT_MEDIUM_RANGE_MAX_HOURS))
        .then(pl.lit("short_medium_range"))
        .otherwise(pl.lit("extended_range"))
        .cast(pl.Enum(HORIZON_SLICES))
    )


def _wide_error_metrics(with_error: pl.LazyFrame, group_keys: list[str]) -> pl.LazyFrame:
    """Aggregate the ``error`` column into wide MAE/RMSE/MBE rows, one per ``group_keys`` group."""
    return with_error.group_by(group_keys).agg(
        mae=pl.col("error").abs().mean(),
        rmse=(pl.col("error").pow(2).mean()).sqrt(),
        mbe=pl.col("error").mean(),
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
    2. Assigns each row a ``horizon_slice`` from its lead time
       (``valid_time − power_fcst_init_time``) — see ``_horizon_slice_expr`` for the bands.
    3. Averages across ``ensemble_member`` *within each forecast run* (per
       ``power_fcst_init_time``) to form a deterministic ensemble mean. Each run covering a
       ``valid_time`` is scored independently, exactly as a production consumer would
       experience it — runs at different lead times are never pooled.
    4. Computes MAE, NMAE, RMSE, and MBE per ``horizon_slice``, plus the ``"all"`` aggregate
       over every lead time.
    5. Joins ``time_series_type`` from ``metadata`` onto each row.
    6. Returns one row per ``(time_series_id, fold_id, power_fcst_model_name,
       horizon_slice, metric_name, metric_param)`` in the tall ``Metrics`` format.

    NMAE is normalised by the pre-computed full-history ``effective_capacity_mw`` (joined per
    ``time_series_id`` from ``capacity``) — a capacity-like denominator, computed over the full
    history so it stays stable across folds. This ``time_series_id``-only join is correct while
    ``capacity`` is one scalar row per series (the v0.1 shape). When the v0.7
    upgrade makes capacity time-varying (one row per ``(time_series_id,
    time)``), this join must become a temporal as-of join on ``(time_series_id, valid_time)``.

    Currently only the ``"all"`` metric_param is computed. Parametric metrics
    (Pinball Loss, PICP) and ensemble metrics (CRPS, spread-skill ratio) can be
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
        NoOverlappingActualsError: If no rows survive the inner join (forecasts cover a
            period with no observed data).
        ValueError: If any forecast row has a negative lead time (``valid_time`` before
            ``power_fcst_init_time`` — an undeliverable hindcast row; see issue #346), or if
            any scored series has no row in ``capacity``.
    """
    # A negative lead time means hindcast rows — valid times already in the past at
    # power_fcst_init_time, which a live forecast could never deliver. Scoring them would
    # silently flatter the model (they'd land in "intraday" via the left-closed bands), so
    # fail loudly. The CV inference pass currently emits such rows for valid times inside
    # the NWP publication-delay window; issue #346 tracks removing them at the source.
    n_negative = cv_forecasts.filter(pl.col("valid_time") < pl.col("power_fcst_init_time")).height
    if n_negative > 0:
        raise ValueError(
            f"{n_negative} forecast row(s) have valid_time before power_fcst_init_time "
            "(negative lead time). These are undeliverable hindcast rows and must not be "
            "scored — regenerate the forecasts without them (see issue #346)."
        )

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

    # Average across ensemble members to produce a deterministic ensemble mean per forecast
    # run: power_fcst_init_time is a group key so that runs covering the same valid_time at
    # different lead times are scored independently, never pooled into a lagged-ensemble blend.
    # horizon_slice is constant within a (power_fcst_init_time, valid_time) group.
    ensemble_mean = (
        joined.with_columns(horizon_slice=_horizon_slice_expr())
        .group_by(
            [
                "time_series_id",
                "fold_id",
                "power_fcst_model_name",
                "power_fcst_init_time",
                "valid_time",
            ]
        )
        .agg(
            power_fcst=pl.col("power_fcst").mean(),
            power_actual=pl.col("power_actual").first(),
            horizon_slice=pl.col("horizon_slice").first(),
        )
    )

    # Compute error column once, reuse in all aggregations.
    with_error = ensemble_mean.with_columns(error=pl.col("power_fcst") - pl.col("power_actual"))

    # Wide metrics: one row per (time_series_id, fold_id, power_fcst_model_name, horizon_slice),
    # where horizon_slice covers the four lead-time bands plus the "all" aggregate over every
    # lead time.
    base_keys = ["time_series_id", "fold_id", "power_fcst_model_name"]
    wide_columns = [*base_keys, "horizon_slice", "mae", "rmse", "mbe"]
    per_slice = _wide_error_metrics(with_error, [*base_keys, "horizon_slice"])
    all_slice = _wide_error_metrics(with_error, base_keys).with_columns(
        horizon_slice=pl.lit("all").cast(pl.Enum(HORIZON_SLICES))
    )
    metrics_wide = pl.concat([per_slice.select(wide_columns), all_slice.select(wide_columns)])

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
            index=[*base_keys, "horizon_slice"],
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
            metric_param=pl.lit("all").cast(pl.Enum(METRIC_PARAMS)),
        )
    )

    if metrics_tall.is_empty():
        raise NoOverlappingActualsError(
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

    Computes mean ``metric_value`` across all series (``"all"`` aggregate), per
    ``time_series_type``, and per ``horizon_slice``, restricted to ``metric_param="all"``
    (the scalar metrics). Key formats:

    - ``"{metric_name}__all"`` — overall aggregate (``horizon_slice="all"``).
    - ``"{metric_name}__{type_slug}"`` — per-type aggregates (``horizon_slice="all"``).
    - ``"{metric_name}__all__{horizon_slice}"`` — overall aggregate per lead-time band
      (e.g. ``"nmae__all__day_ahead"``). Per-type sliced aggregates are deliberately not
      logged — that detail stays queryable in the ``forecast_metrics`` Delta table.

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

    # Overall aggregate per lead-time band (includes all series regardless of type).
    per_slice = (
        metrics_df.filter((pl.col("horizon_slice") != "all") & (pl.col("metric_param") == "all"))
        .group_by(["metric_name", "horizon_slice"])
        .agg(mean_value=pl.col("metric_value").mean())
    )
    for row in per_slice.iter_rows(named=True):
        result[f"{row['metric_name']}__all__{row['horizon_slice']}"] = float(row["mean_value"])

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
            experiment_name=pl.lit(experiment_name, dtype=pl.String),
            evaluation_scope=pl.lit(evaluation_scope).cast(pl.Enum(EVALUATION_SCOPES)),
            window_start=pl.lit(window_start).cast(UTC_DATETIME_DTYPE),
            window_end=pl.lit(window_end).cast(UTC_DATETIME_DTYPE),
            window_label=pl.lit(window_label, dtype=pl.String),
            computed_at=pl.lit(computed_at).cast(UTC_DATETIME_DTYPE),
            mlflow_run_id=pl.lit(mlflow_run_id, dtype=pl.String),
        )
    )
