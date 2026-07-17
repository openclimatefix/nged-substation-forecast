"""Metric computation for cross-validation results."""

import re
from datetime import datetime
from typing import Final

import patito as pt
import polars as pl
from contracts.common import DELIVERY_QUANTILES, UTC_DATETIME_DTYPE, quantile_label
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


_BAND_LOWER_QUANTILES: Final[tuple[float, ...]] = tuple(q for q in DELIVERY_QUANTILES if q < 0.5)
"""Lower quantile of each symmetric prediction-interval band scored by PICP/interval width.

Each lower quantile ``q`` pairs with ``1 − q`` to form a band (e.g. 0.1 → the p10–p90 band),
matching ``contracts.ml_schemas.BAND_METRIC_PARAMS``.
"""

_MLFLOW_LOGGED_PARAMETRIC: Final[frozenset[tuple[str, str]]] = frozenset(
    {
        ("pinball_loss", "p10"),
        ("pinball_loss", "p50"),
        ("pinball_loss", "p90"),
        ("picp", "p10_p90"),
        ("interval_width", "p10_p90"),
    }
)
"""The parametric ``(metric_name, metric_param)`` pairs logged to MLflow.

Metrics with ``metric_param="all"`` are always logged; parametric metrics are restricted to
this headline subset to keep the MLflow leaderboard legible (~130 keys instead of ~340). The
full 13-quantile / 6-band detail is always queryable in the ``forecast_metrics`` Delta table.
"""


def _band_label(lower_quantile: float) -> str:
    """Return the band ``metric_param`` label for a lower quantile, e.g. ``0.1`` → ``"p10_p90"``."""
    return f"{quantile_label(lower_quantile)}_{quantile_label(1 - lower_quantile)}"


def _quantile_column(quantile: float) -> str:
    """Name of the per-timestamp empirical-quantile column for ``quantile``, e.g. ``"q_p10"``."""
    return f"q_{quantile_label(quantile)}"


def _fair_crps_expr() -> pl.Expr:
    """Per-timestamp fair CRPS over the ensemble members of one forecast-run group.

    Evaluated inside the per-run collapse ``group_by``, where each group holds the ``m``
    members forecasting one ``(time_series_id, power_fcst_init_time, valid_time)``. The fair
    (finite-ensemble-unbiased, Ferro 2014) form is::

        CRPS = mean_i |x_i − y|  −  Σ_{i<j} |x_i − x_j| / (m(m−1))

    with the pairwise term defined as 0 when ``m = 1`` (so a single-member "ensemble" scores
    its absolute error, and group means reduce to MAE). The pairwise sum uses the sorted-member
    identity ``Σ_{i<j}(x_(j) − x_(i)) = Σ_k (2k − m − 1)·x_(k)`` — O(m log m) instead of a
    member self-join — computed in Float64 because the identity's large cancelling terms (up
    to ±m·|power|) lose percent-level accuracy in Float32 when members are near-identical.

    See
    <https://openclimatefix.github.io/nged-substation-forecast/techniques/evaluation-metrics/>
    for the full rationale.
    """
    m = pl.len()
    members = pl.col("power_fcst").cast(pl.Float64)
    mean_member_ae = (members - pl.col("power_actual").cast(pl.Float64)).abs().mean()
    rank = pl.int_range(1, m + 1)
    pairwise_sum = (members.sort() * (2 * rank - m - 1)).sum()
    pairwise_mean = pl.when(m > 1).then(pairwise_sum / (m * (m - 1))).otherwise(0.0)
    return mean_member_ae - pairwise_mean


def _corrected_variance_expr() -> pl.Expr:
    """Per-timestamp Fortin-corrected ensemble variance, ``((m+1)/m)·Var(members)``.

    Evaluated inside the per-run collapse ``group_by``. For a calibrated ensemble the RMSE of
    the ensemble mean equals ``sqrt((m+1)/m)`` times the RMS ensemble spread (Fortin et
    al. 2014), so folding the factor in here makes the spread-skill ratio's calibrated target
    exactly 1.0 at any ensemble size. Uses the sample variance (``ddof=1``), guarded to 0 for
    single-member groups where ``.var()`` would return null (``metric_value`` is
    non-nullable, and zero spread is the honest description of a deterministic forecast).
    """
    m = pl.len()
    variance = pl.col("power_fcst").cast(pl.Float64).var()
    return pl.when(m > 1).then(variance * (m + 1) / m).otherwise(0.0)


def _wide_metric_columns() -> list[str]:
    """Ordered names of the wide metric columns produced by ``_wide_metrics``.

    Parametric metrics encode their ``metric_param`` after a ``":"`` separator (e.g.
    ``"pinball_loss:p10"``); ``compute_metrics`` splits the encoding apart again after the
    unpivot to tall format. ``"nmae"`` is absent — it is derived from ``"mae"`` after the
    capacity join.
    """
    columns = ["mae", "rmse", "mbe", "crps", "spread_skill_ratio", "mean_pinball_loss"]
    columns += [f"pinball_loss:{quantile_label(q)}" for q in DELIVERY_QUANTILES]
    for lower in _BAND_LOWER_QUANTILES:
        columns += [f"picp:{_band_label(lower)}", f"interval_width:{_band_label(lower)}"]
    return columns


def _wide_metrics(per_run: pl.LazyFrame, group_keys: list[str]) -> pl.LazyFrame:
    """Aggregate per-timestamp values into one wide row of metrics per ``group_keys`` group.

    ``per_run`` is the per-forecast-run frame built by ``compute_metrics``: one row per
    ``(time_series_id, power_fcst_init_time, valid_time)`` carrying the ensemble-mean
    ``error``, per-timestamp ``crps`` and ``corrected_var``, and the empirical
    ``DELIVERY_QUANTILES`` columns. Emits the columns listed by ``_wide_metric_columns``.
    """
    actual = pl.col("power_actual")
    aggs: dict[str, pl.Expr] = {
        "mae": pl.col("error").abs().mean(),
        "rmse": (pl.col("error").pow(2).mean()).sqrt(),
        "mbe": pl.col("error").mean(),
        "crps": pl.col("crps").mean(),
        # RMS spread, not mean-of-std: Jensen's inequality drags mean-of-std well below the
        # RMS form whenever spread varies across timestamps, which would fake underdispersion
        # for a calibrated ensemble. The (m+1)/m factor is already folded into corrected_var.
        "_rms_spread": pl.col("corrected_var").mean().sqrt(),
    }
    for q in DELIVERY_QUANTILES:
        diff = actual - pl.col(_quantile_column(q))
        aggs[f"pinball_loss:{quantile_label(q)}"] = (
            pl.when(diff >= 0).then(diff * q).otherwise(diff * (q - 1)).mean()
        )
    for lower in _BAND_LOWER_QUANTILES:
        low_col = pl.col(_quantile_column(lower))
        high_col = pl.col(_quantile_column(1 - lower))
        aggs[f"picp:{_band_label(lower)}"] = actual.is_between(low_col, high_col).mean()
        aggs[f"interval_width:{_band_label(lower)}"] = (high_col - low_col).mean()
    return (
        per_run.group_by(group_keys)
        .agg(**aggs)
        .with_columns(
            spread_skill_ratio=pl.col("_rms_spread") / pl.col("rmse"),
            mean_pinball_loss=pl.mean_horizontal(
                [f"pinball_loss:{quantile_label(q)}" for q in DELIVERY_QUANTILES]
            ),
        )
        .drop("_rms_spread")
    )


def compute_metrics(
    cv_forecasts: pt.DataFrame[PowerForecast],
    actuals: pt.LazyFrame[PowerTimeSeries],
    metadata: pt.DataFrame[TimeSeriesMetadata],
    capacity: pt.DataFrame[EffectiveCapacity],
) -> pt.DataFrame[Metrics]:
    """Compute evaluation metrics from CV predictions and observed power.

    Full metric definitions, equations, and design rationale:
    <https://openclimatefix.github.io/nged-substation-forecast/techniques/evaluation-metrics/>

    For each ``(time_series_id, fold_id, power_fcst_model_name)`` group:

    1. Joins predictions to observed ``power`` on ``(time_series_id, valid_time)``.
    2. Assigns each row a ``horizon_slice`` from its lead time
       (``valid_time − power_fcst_init_time``) — see ``_horizon_slice_expr`` for the bands.
    3. Collapses the ensemble members *within each forecast run* (per
       ``power_fcst_init_time``) into per-timestamp quantities: the deterministic ensemble
       mean, the fair CRPS, the Fortin-corrected ensemble variance, and the empirical
       ``DELIVERY_QUANTILES``. Each run covering a ``valid_time`` is scored independently,
       exactly as a production consumer would experience it — runs at different lead times
       are never pooled.
    4. Aggregates per ``horizon_slice``, plus the ``"all"`` aggregate over every lead time:
       MAE, NMAE, RMSE, and MBE on the ensemble mean; CRPS and the spread-skill ratio from
       the member-aware quantities; pinball loss at each delivery quantile (plus their
       mean); and PICP and interval width for each symmetric quantile band.
    5. Joins ``time_series_type`` from ``metadata`` onto each row.
    6. Returns one row per ``(time_series_id, fold_id, power_fcst_model_name,
       horizon_slice, metric_name, metric_param)`` in the tall ``Metrics`` format.

    NMAE is normalised by the pre-computed full-history ``effective_capacity_mw`` (joined per
    ``time_series_id`` from ``capacity``) — a capacity-like denominator, computed over the full
    history so it stays stable across folds. This ``time_series_id``-only join is correct while
    ``capacity`` is one scalar row per series (the v0.1 shape). When the v0.7
    upgrade makes capacity time-varying (one row per ``(time_series_id,
    time)``), this join must become a temporal as-of join on ``(time_series_id, valid_time)``.

    Single-member "ensembles" (e.g. a deterministic baseline forecaster) are scored
    unconditionally: their fair CRPS equals their MAE, their spread-skill ratio is 0, and
    their quantile bands are degenerate (all quantiles coincide, so PICP ≈ 0 and interval
    width = 0). Those are honest descriptions of a deterministic forecast, not errors.

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

    # Collapse the ensemble members of each forecast run into per-timestamp quantities: the
    # deterministic ensemble mean plus the member-aware values (fair CRPS, Fortin-corrected
    # variance, empirical delivery quantiles). power_fcst_init_time is a group key so that
    # runs covering the same valid_time at different lead times are scored independently,
    # never pooled into a lagged-ensemble blend. horizon_slice is constant within a
    # (power_fcst_init_time, valid_time) group.
    quantile_aggs = {
        _quantile_column(q): pl.col("power_fcst").cast(pl.Float64).quantile(q, "linear")
        for q in DELIVERY_QUANTILES
    }
    per_run = (
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
            crps=_fair_crps_expr(),
            corrected_var=_corrected_variance_expr(),
            **quantile_aggs,
        )
    )

    # Compute the ensemble-mean error column once, reuse in all aggregations.
    with_error = per_run.with_columns(error=pl.col("power_fcst") - pl.col("power_actual"))

    # Wide metrics: one row per (time_series_id, fold_id, power_fcst_model_name, horizon_slice),
    # where horizon_slice covers the four lead-time bands plus the "all" aggregate over every
    # lead time.
    base_keys = ["time_series_id", "fold_id", "power_fcst_model_name"]
    wide_columns = [*base_keys, "horizon_slice", *_wide_metric_columns()]
    per_slice = _wide_metrics(with_error, [*base_keys, "horizon_slice"])
    all_slice = _wide_metrics(with_error, base_keys).with_columns(
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

    # Pivot to tall format, then split the "name:param" encoding of the parametric wide
    # columns into (metric_name, metric_param); scalar metrics have no separator, so their
    # split field_1 is null and fills to "all". The leftover effective_capacity_mw column is
    # dropped by the unpivot.
    name_parts = pl.col("metric_name").str.split_exact(":", 1)
    metrics_tall = (
        wide.unpivot(
            on=[*_wide_metric_columns(), "nmae"],
            index=[*base_keys, "horizon_slice"],
            variable_name="metric_name",
            value_name="metric_value",
        )
        .with_columns(
            metric_name=name_parts.struct.field("field_0"),
            metric_param=name_parts.struct.field("field_1").fill_null("all"),
        )
        .cast(
            {
                "metric_name": pl.Enum(METRIC_NAMES),
                "metric_param": pl.Enum(METRIC_PARAMS),
                "metric_value": pl.Float32,
            }
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


def _mlflow_logged_expr() -> pl.Expr:
    """Predicate selecting the metric rows logged to MLflow.

    Every ``metric_param="all"`` metric is logged; parametric metrics are restricted to the
    ``_MLFLOW_LOGGED_PARAMETRIC`` headline subset.
    """
    logged = pl.col("metric_param") == "all"
    for name, param in sorted(_MLFLOW_LOGGED_PARAMETRIC):
        logged = logged | ((pl.col("metric_name") == name) & (pl.col("metric_param") == param))
    return logged


def _metric_key_token_expr() -> pl.Expr:
    """MLflow key token for a metric row: ``{metric_name}`` or ``{metric_name}_{metric_param}``.

    ``metric_param="all"`` metrics keep the bare name (so the pre-probabilistic key formats
    are unchanged); parametric metrics append the param, e.g. ``"pinball_loss_p10"``.
    """
    name = pl.col("metric_name").cast(pl.String)
    return (
        pl.when(pl.col("metric_param") == "all")
        .then(name)
        .otherwise(name + "_" + pl.col("metric_param").cast(pl.String))
    )


def build_mlflow_aggregate_metrics(
    metrics_df: pl.DataFrame,
) -> dict[str, float]:
    """Return a flat ``{metric_key: value}`` dict for ``mlflow.log_metrics``.

    Computes mean ``metric_value`` across all series (``"all"`` aggregate), per
    ``time_series_type``, and per ``horizon_slice``. The key token is ``{metric_name}`` for
    ``metric_param="all"`` metrics and ``{metric_name}_{metric_param}`` for parametric ones
    (e.g. ``pinball_loss_p10``, ``picp_p10_p90``); parametric metrics are restricted to the
    ``_MLFLOW_LOGGED_PARAMETRIC`` headline subset. Key formats:

    - ``"{token}__all"`` — overall aggregate (``horizon_slice="all"``).
    - ``"{token}__{type_slug}"`` — per-type aggregates (``horizon_slice="all"``).
    - ``"{token}__all__{horizon_slice}"`` — overall aggregate per lead-time band
      (e.g. ``"nmae__all__day_ahead"``). Per-type sliced aggregates are deliberately not
      logged — that detail stays queryable in the ``forecast_metrics`` Delta table, as does
      the full 13-quantile / 6-band parametric detail.

    Args:
        metrics_df: Per-series ``Metrics`` rows with ``time_series_type`` populated.

    Returns:
        Flat dict of MLflow metric key → mean float value.
    """
    logged = metrics_df.filter(_mlflow_logged_expr()).with_columns(
        metric_key=_metric_key_token_expr()
    )
    base = logged.filter(pl.col("horizon_slice") == "all")

    result: dict[str, float] = {}

    # Per-type aggregates (only for non-null time_series_type values).
    if "time_series_type" in base.columns:
        per_type = (
            base.filter(pl.col("time_series_type").is_not_null())
            .group_by(["metric_key", "time_series_type"])
            .agg(mean_value=pl.col("metric_value").mean())
        )
        for row in per_type.iter_rows(named=True):
            key = f"{row['metric_key']}__{_type_slug(str(row['time_series_type']))}"
            result[key] = float(row["mean_value"])

    # Overall "all" aggregate (includes all series regardless of type).
    overall = base.group_by("metric_key").agg(mean_value=pl.col("metric_value").mean())
    for row in overall.iter_rows(named=True):
        result[f"{row['metric_key']}__all"] = float(row["mean_value"])

    # Overall aggregate per lead-time band (includes all series regardless of type).
    per_slice = (
        logged.filter(pl.col("horizon_slice") != "all")
        .group_by(["metric_key", "horizon_slice"])
        .agg(mean_value=pl.col("metric_value").mean())
    )
    for row in per_slice.iter_rows(named=True):
        result[f"{row['metric_key']}__all__{row['horizon_slice']}"] = float(row["mean_value"])

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
