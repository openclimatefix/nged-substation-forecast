"""Scoring forecasts against the power that was actually observed.

A forecast here is a *quantile forecast*: for one time series, one forecast run, and one future
half-hour, the forecaster emits one ``PowerForecast`` row holding the forecast at the 13
``DELIVERY_QUANTILES``, and those quantile columns are what the probabilistic scores below measure.
The deterministic scores read the median column, ``power_fcst_p50``. A cross-validation (CV) fold
is one train-and-validate split of the history, identified by a ``fold_id``. MLflow is the CV run
tracker the scores are logged to, and its leaderboard is the ranked comparison of one run's scores
against another's.

Four scoring functions, in the order the cross-validation Dagster assets call them.
``compute_effective_capacity`` derives the per-series denominator that normalised mean absolute
error (NMAE) divides by. ``compute_metrics`` joins predictions to observed power and returns the
tall ``Metrics`` frame. ``enrich_metrics_rows`` stamps the evaluation window and scope onto that
frame once the calling asset knows the window and the scope. The enriched frame is what the asset
writes to Delta. ``build_mlflow_aggregate_metrics`` takes the un-enriched frame that
``compute_metrics`` returned. That function reduces the frame to the flat key/value dictionary
the MLflow leaderboard displays. The ``require_*`` functions beside the four scoring functions
refuse a group the scorer must not score: rows outside the fold's window, and a study whose row
keys differ from the reference CV run's.

Every function here is pure: no Dagster, no MLflow, and no IO. Each function is therefore
unit-testable on an in-memory frame. The asset that calls the function owns every read and write.

The equations, and the argument for choosing each metric over the alternatives, are on
<https://openclimatefix.github.io/nged-substation-forecast/techniques/evaluation-metrics/>. The
docstrings below say what this implementation does with those equations. Those docstrings also
record the corner where a degenerate input needed a defined answer rather than a NaN: a
deterministic forecast, whose 13 quantile columns are equal.
"""

import hashlib
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
    POWER_FCST_QUANTILE_COLUMNS,
    EffectiveCapacity,
    PowerForecast,
    PowerTimeSeries,
    TimeSeriesMetadata,
)


class NoOverlappingActualsError(ValueError):
    """Raised by ``compute_metrics`` when no forecast row joins to any observed actual.

    A distinct subclass, so that a caller scoring a fold in per-series batches can treat "this
    batch's series have no overlapping actuals" as skippable. Skipping mirrors what happens when
    the whole fold is scored in one call: those series silently vanish from the inner join. Every
    other ``ValueError`` — a negative lead time, a missing capacity row — still propagates.
    """


ROW_KEY_COLUMNS: Final[tuple[str, ...]] = PowerForecast.PRIMARY_KEY
"""The columns that identify one forecast row: the primary key of `PowerForecast`.

Two CV runs whose rows carry the same keys forecast the same series, from the same
initialisation times, for the same valid times. ``require_same_row_keys`` compares these columns
and ``row_key_fingerprint`` hashes them.
"""


class RowsOutsideWindowError(ValueError):
    """Raised when a forecast row's ``valid_time`` lies outside the fold's evaluation window."""


class RowKeyMismatchError(ValueError):
    """Raised when a study's forecast row keys differ from the reference CV run's."""


def require_valid_times_within_window(
    *,
    valid_time_min: datetime,
    valid_time_max: datetime,
    window_start: datetime,
    window_end: datetime,
    group_label: str,
) -> None:
    """Refuse forecast rows whose ``valid_time`` falls outside the fold's evaluation window.

    ``compute_metrics`` joins forecasts to actuals on ``(time_series_id, valid_time)`` and never
    filters to the window, so a forecast file carrying extra rows would be scored and then labelled
    as the fold's window. In leaderboard scope this check keeps rows past ``val_end`` out of the
    score.

    Args:
        valid_time_min: Earliest ``valid_time`` in the group.
        valid_time_max: Latest ``valid_time`` in the group.
        window_start: Inclusive start of the evaluation window.
        window_end: Inclusive end of the evaluation window.
        group_label: ``"{experiment_name}, {fold_id}"``, for the error message.

    Raises:
        RowsOutsideWindowError: If any row is before ``window_start`` or after ``window_end``.
    """
    if valid_time_min < window_start or valid_time_max > window_end:
        raise RowsOutsideWindowError(
            f"Group ({group_label}) has valid_time from {valid_time_min.isoformat()} to "
            f"{valid_time_max.isoformat()}, outside the evaluation window "
            f"{window_start.isoformat()} to {window_end.isoformat()}."
        )


def _distinct_series_ids(forecasts: pl.LazyFrame) -> list[int]:
    """Return the sorted ``time_series_id`` values present in ``forecasts``."""
    return sorted(
        forecasts.select("time_series_id").unique().collect(engine="streaming")["time_series_id"]
    )


def require_same_row_keys(
    *,
    study: pl.LazyFrame,
    reference: pl.LazyFrame,
    group_label: str,
    reference_label: str,
    series_batch_size: int,
) -> None:
    """Refuse a study whose forecast row keys differ from the reference CV run's.

    The keys are ``ROW_KEY_COLUMNS``. A study that omits a row the reference forecasts, or adds a
    row the reference does not, cannot raise its score by abstaining on hard rows, by adding easy
    series, or by forecasting only at short lead times. The reference's own forecasts are trusted:
    they come from a reviewed CV run.

    The series sets are compared first. The keys are then compared in batches of
    ``series_batch_size`` series, with a distinct-key anti-join in each direction, so peak memory
    is one batch rather than a whole fold (a leaderboard fold is about 364 million rows).

    Args:
        study: Lazy scan of the study's forecast rows, carrying at least ``ROW_KEY_COLUMNS``.
        reference: Lazy scan of the reference CV run's rows for the same fold and the same
            ``valid_time`` filter, carrying at least ``ROW_KEY_COLUMNS``.
        group_label: ``"{experiment_name}, {fold_id}"`` of the study, for the error message.
        reference_label: The reference CV run's name, for the error message.
        series_batch_size: How many ``time_series_id`` values to compare at once.

    Raises:
        RowKeyMismatchError: If the reference has no rows, the two series sets differ, or any
            batch has a key present on one side only.
    """
    reference_ids = _distinct_series_ids(reference)
    if not reference_ids:
        raise RowKeyMismatchError(
            f"Reference CV run {reference_label!r} has no forecast rows for group "
            f"({group_label}). Materialise the reference CV run for this fold first."
        )
    study_ids = _distinct_series_ids(study)
    if study_ids != reference_ids:
        missing = sorted(set(reference_ids) - set(study_ids))
        extra = sorted(set(study_ids) - set(reference_ids))
        raise RowKeyMismatchError(
            f"Group ({group_label}) forecasts different series from reference "
            f"{reference_label!r}: missing {missing}, extra {extra}."
        )
    keys = list(ROW_KEY_COLUMNS)
    for start in range(0, len(reference_ids), series_batch_size):
        batch_ids = reference_ids[start : start + series_batch_size]
        study_keys = (
            study.filter(pl.col("time_series_id").is_in(batch_ids))
            .select(keys)
            .unique()
            .collect(engine="streaming")
        )
        reference_keys = (
            reference.filter(pl.col("time_series_id").is_in(batch_ids))
            .select(keys)
            .unique()
            .collect(engine="streaming")
        )
        n_missing = reference_keys.join(study_keys, on=keys, how="anti").height
        n_extra = study_keys.join(reference_keys, on=keys, how="anti").height
        if n_missing or n_extra:
            raise RowKeyMismatchError(
                f"Group ({group_label}) differs from reference {reference_label!r} for series "
                f"{batch_ids}: {n_missing} reference row keys are missing and {n_extra} study row "
                f"keys are extra, on the columns {keys}."
            )


def row_key_fingerprint(*, forecasts: pl.LazyFrame, series_batch_size: int) -> str:
    """Return a hash of the distinct ``ROW_KEY_COLUMNS`` keys in ``forecasts``.

    Two CV runs have the same fingerprint exactly when they forecast the same series, from the
    same initialisation times, for the same valid times, in whatever row order. The ``metrics``
    asset stamps the fingerprint on every leaderboard fold run, so a reader can see whether two
    scores rest on the same forecast problem, and whether a study was scored against a reference
    that has since changed.

    The keys are hashed in batches of ``series_batch_size`` series, so peak memory is one batch.

    Args:
        forecasts: Lazy scan of one group's forecast rows, carrying ``ROW_KEY_COLUMNS``.
        series_batch_size: How many ``time_series_id`` values to hash at once. The fingerprint
            does not depend on it.

    Returns:
        A hexadecimal SHA-256 digest. The digest of no rows is the digest of the empty string.
    """
    keys = list(ROW_KEY_COLUMNS)
    series_ids = _distinct_series_ids(forecasts)
    digest = hashlib.sha256()
    for start in range(0, len(series_ids), series_batch_size):
        batch_ids = series_ids[start : start + series_batch_size]
        batch = (
            forecasts.filter(pl.col("time_series_id").is_in(batch_ids))
            .select(keys)
            .unique()
            .sort(keys)
            .collect(engine="streaming")
        )
        digest.update(batch.cast(pl.Int64).to_numpy().tobytes())
    return digest.hexdigest()


def compute_effective_capacity(
    power_lf: pt.LazyFrame[PowerTimeSeries],
) -> pt.DataFrame[EffectiveCapacity]:
    """Compute version 0.1 of the effective capacity, per time series.

    Version 0.1 of the capacity definition is the full-history P99 of ``|power|``.

    One row per ``time_series_id``: ``effective_capacity_mw`` is the 99th percentile of
    ``abs(power)`` over all non-null observations. A series is dropped when its P99 is null or
    non-positive (e.g. all-null or all-zero power), since ``EffectiveCapacity`` requires
    ``effective_capacity_mw > 0``.

    ``time`` is set to that series' **latest timestep carrying a non-null power** (``time.max()``
    over the same filtered rows the percentile is taken over). The v0.1 capacity is a single
    scalar per series, so ``time`` is really an "as of" marker rather than a timestep the value
    varies over. The marker stamps the estimate as current to the end of the observed history.
    Version 0.7 of the capacity definition makes capacity genuinely time-varying (one row per
    ``(time_series_id, time)``), and only then does ``time`` carry per-row meaning. Version 0.1
    stays one scalar row per series, rather than the value repeated at every half-hour.
    Densifying a constant adds rows without information. The metrics join is by
    ``time_series_id`` alone until capacity varies.

    Kept as a pure helper (no Dagster, no IO) so the P99 logic is unit-testable in isolation.
    """
    capacity = (
        power_lf.filter(pl.col("power").is_not_null())
        .group_by("time_series_id")
        .agg(
            effective_capacity_mw=pl.col("power").abs().quantile(0.99),
            time=pl.col("time").max(),
        )
        .filter(pl.col("effective_capacity_mw") > 0)
        .sort("time_series_id")
        .collect()
        .cast({"effective_capacity_mw": pl.Float32})
    )
    return EffectiveCapacity.validate(capacity)


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

    Lead time is ``valid_time − power_fcst_init_time``. Bands are left-closed: ``"intraday"`` [0
    h, 6 h), ``"day_ahead"`` [6 h, 36 h), ``"short_medium_range"`` [36 h, 168 h),
    ``"extended_range"`` [168 h, ∞). Those four bands match the definitions documented on
    ``contracts.ml_schemas.HORIZON_SLICES``.
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
"""Lower quantile of each symmetric prediction-interval band scored by prediction-interval
coverage probability (PICP) and by interval width.

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

Metrics with ``metric_param="all"`` are always logged. Parametric metrics are restricted to this
headline subset, to keep the MLflow leaderboard legible. How many MLflow metric keys that
restriction leaves depends on how many distinct ``time_series_type`` values the scored population
spans, since each value adds a per-type key family. The 7 types present in the V1 trial area give
144 keys, against 384 keys if every parametric metric were logged. The pinball loss at all 13
delivery quantiles, and the PICP and interval width of all 6 bands, stay queryable in the
``forecast_metrics`` Delta table whichever pairs the headline subset above names.
"""


def _band_label(lower_quantile: float) -> str:
    """Return the quantile-band label for a lower quantile, e.g. ``0.1`` → ``"p10_p90"``.

    A quantile band is a symmetric pair of delivery quantiles, such as p10 and p90, whose
    interval the forecast is scored for coverage and width. The label returned here is what the
    tall ``Metrics`` frame stores in its ``metric_param`` column, which says which quantile or
    band a metric row belongs to. The lead-time bands are a separate axis, carried by the
    ``horizon_slice`` column rather than by ``metric_param``.
    """
    return f"{quantile_label(lower_quantile)}_{quantile_label(1 - lower_quantile)}"


def _quantile_column(quantile: float) -> str:
    """Name of the stored ``PowerForecast`` column for ``quantile``, e.g. ``"power_fcst_p10"``."""
    return POWER_FCST_QUANTILE_COLUMNS[DELIVERY_QUANTILES.index(quantile)]


def _wide_metric_columns() -> list[str]:
    """Ordered names of the wide metric columns produced by ``_wide_metrics``.

    A *parametric* metric takes a parameter — a delivery quantile, or a quantile band — and so
    appears once per value of that parameter. Every other metric is *scalar* and appears once.
    Parametric metrics encode their ``metric_param`` after a ``":"`` separator, as in
    ``"pinball_loss:p10"``. ``compute_metrics`` splits the encoding apart again after the unpivot
    to tall format. ``"nmae"`` is absent — it is derived from ``"mae"`` after the capacity join.
    """
    columns = ["mae", "rmse", "mbe", "crps", "mean_pinball_loss"]
    columns += [f"pinball_loss:{quantile_label(q)}" for q in DELIVERY_QUANTILES]
    for lower in _BAND_LOWER_QUANTILES:
        columns += [f"picp:{_band_label(lower)}", f"interval_width:{_band_label(lower)}"]
    return columns


def _wide_metrics(scored: pl.LazyFrame, group_keys: list[str]) -> pl.LazyFrame:
    """Aggregate per-row values into one wide row of metrics per ``group_keys`` group.

    ``scored`` is the frame built by ``compute_metrics``: one ``PowerForecast`` row per
    ``(time_series_id, fold_id, power_fcst_model_name, power_fcst_init_time, valid_time)``, with
    ``power_actual`` and the median forecast's ``error`` joined on. Emits the columns listed by
    ``_wide_metric_columns``.

    The continuous ranked probability score (CRPS) is estimated as twice the unweighted mean
    pinball loss over the 13 ``DELIVERY_QUANTILES``. Twice the integral of the pinball loss over
    every level is the CRPS exactly, and the mean over the 13 levels is a discrete approximation of
    that integral. The 13 levels are tail-heavy, so this estimate weights the tails more than an
    estimate over evenly spaced levels would.
    """
    actual = pl.col("power_actual").cast(pl.Float64)
    aggs: dict[str, pl.Expr] = {
        "mae": pl.col("error").abs().mean(),
        "rmse": (pl.col("error").pow(2).mean()).sqrt(),
        "mbe": pl.col("error").mean(),
    }
    for q in DELIVERY_QUANTILES:
        diff = actual - pl.col(_quantile_column(q)).cast(pl.Float64)
        aggs[f"pinball_loss:{quantile_label(q)}"] = (
            pl.when(diff >= 0).then(diff * q).otherwise(diff * (q - 1)).mean()
        )
    for lower in _BAND_LOWER_QUANTILES:
        low_col = pl.col(_quantile_column(lower))
        high_col = pl.col(_quantile_column(1 - lower))
        aggs[f"picp:{_band_label(lower)}"] = actual.is_between(low_col, high_col).mean()
        aggs[f"interval_width:{_band_label(lower)}"] = (high_col - low_col).mean()
    mean_pinball_loss = pl.mean_horizontal(
        [f"pinball_loss:{quantile_label(q)}" for q in DELIVERY_QUANTILES]
    )
    return (
        scored.group_by(group_keys)
        .agg(**aggs)
        .with_columns(mean_pinball_loss=mean_pinball_loss, crps=2 * mean_pinball_loss)
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
    3. Scores each row on its own. Each forecast run covering a ``valid_time`` is scored
       independently, exactly as a production consumer would experience that run. Runs at
       different lead times are never pooled.
    4. Aggregates per ``horizon_slice``, plus the ``"all"`` aggregate over every lead time. The
       median forecast, ``power_fcst_p50``, gives MAE, NMAE, RMSE, and mean bias error (MBE). Each
       of the 13 stored quantile columns gives a pinball loss; the pinball losses get a mean, and
       twice that mean is the CRPS. Each symmetric pair of quantile columns gives a
       prediction-interval coverage probability (PICP) and an interval width.
    5. Joins ``time_series_type`` from ``metadata`` onto each row.
    6. Returns one row per ``(time_series_id, fold_id, power_fcst_model_name, horizon_slice,
       metric_name, metric_param)`` in the tall ``Metrics`` format.

    NMAE is normalised by the pre-computed ``effective_capacity_mw``, joined per ``time_series_id``
    from ``capacity``. That denominator is capacity-like, and is computed over the full history so
    that the denominator stays stable across folds. This ``time_series_id``-only join is correct
    while ``capacity`` is one scalar row per series, the shape at version 0.1 of the capacity
    definition. Version 0.7 makes capacity time-varying, at one row per ``(time_series_id, time)``.
    This join must then become a temporal as-of join on ``(time_series_id, valid_time)``.

    A deterministic forecast — one whose 13 quantile columns are equal — is scored
    unconditionally. Its CRPS equals its MAE. Its quantile bands are degenerate, so PICP ≈ 0 and
    interval width = 0.

    Args:
        cv_forecasts: CV predictions to evaluate. Currently only CV fold rows are handled. Support
            for ``fold_id="live"`` arrives with the ``production_monitoring`` scope, in roadmap
            phase 8 — see [production
            monitoring](https://openclimatefix.github.io/nged-substation-forecast/roadmap/live-service/#production-monitoring).
        actuals: Observed half-hourly power (lazy — only the joined subset is collected).
            Deduplicated on ``(time_series_id, time)`` before the join — a duplicated actual
            would otherwise double the forecast rows and their weight in every metric.
        metadata: Substation metadata used to join ``time_series_type`` onto each metric row.
            Must cover every scored ``time_series_id``.
        capacity: Pre-computed per-series effective capacity; ``effective_capacity_mw`` is the
            NMAE denominator. Must cover every scored ``time_series_id``.

    Returns:
        A validated tall ``Metrics`` DataFrame with ``time_series_type`` populated.

    Raises:
        NoOverlappingActualsError: If no rows survive the inner join (forecasts cover a
            period with no observed data).
        ValueError: If any forecast row has a negative lead time, meaning ``valid_time`` before
            ``power_fcst_init_time``. Such a row is an undeliverable hindcast row; see issue #346.
            Also raised if any scored series has no row in ``capacity`` or in ``metadata``. Also
            raised if any computed metric value is non-finite (NaN or inf). ``Metrics.validate``
            would otherwise accept a NaN, because NaN is not null, and that NaN would poison the
            MLflow aggregate means.
    """
    # A negative lead time means hindcast rows — valid times already in the past at
    # power_fcst_init_time, which a live forecast could never deliver. Scoring those rows would
    # silently flatter the model, because they would land in "intraday" via the left-closed bands.
    # Fail loudly instead. A numerical weather prediction (NWP) run reaches our disk some hours
    # after the time it was initialised at, and that gap is the publication delay. The CV inference
    # pass currently emits hindcast rows for valid times inside that NWP publication-delay window;
    # issue #346 tracks removing them at the source.
    n_negative = cv_forecasts.filter(pl.col("valid_time") < pl.col("power_fcst_init_time")).height
    if n_negative > 0:
        raise ValueError(
            f"{n_negative} forecast row(s) have valid_time before power_fcst_init_time "
            "(negative lead time). These are undeliverable hindcast rows and must not be "
            "scored — regenerate the forecasts without them (see issue #346)."
        )

    # Join forecasts to actuals; rename power → power_actual to avoid shadowing. Patito wraps a
    # Polars frame in a schema class, and that class is a Python subclass carried in the frame's own
    # type. Strip the Patito model subclass from actuals so that Polars' cross-subclass type check
    # (assert_same_type) doesn't reject a join between two differently-typed pt.LazyFrame objects.
    # Dedupe actuals on the join key. A duplicated (time_series_id, time) row would double every
    # forecast row it joins to, and so that row's weight in every metric. Nothing enforces this
    # uniqueness upstream, so guard it here.
    actuals_plain = pl.LazyFrame._from_pyldf(actuals._ldf)
    joined = cv_forecasts.lazy().join(
        actuals_plain.select(["time_series_id", "time", "power"])
        .unique(subset=["time_series_id", "time"], keep="any")
        .rename({"power": "power_actual"}),
        left_on=["time_series_id", "valid_time"],
        right_on=["time_series_id", "time"],
        how="inner",
    )

    # One row per (time_series_id, power_fcst_init_time, valid_time) already: each run covering a
    # valid_time is scored on its own, never pooled into a lagged-ensemble blend. The deterministic
    # metrics read the median forecast.
    with_error = joined.with_columns(
        horizon_slice=_horizon_slice_expr(),
        error=pl.col("power_fcst_p50").cast(pl.Float64) - pl.col("power_actual").cast(pl.Float64),
    )

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

    # Fail loudly rather than silently emitting a null NMAE. Metrics.metric_value is a non-nullable
    # Float32, so it would reject a null NMAE anyway.
    missing = wide.filter(pl.col("effective_capacity_mw").is_null())["time_series_id"]
    if missing.len() > 0:
        raise ValueError(
            f"No effective_capacity row for time_series_id(s) {sorted(set(missing.to_list()))}; "
            "materialise the effective_capacity asset for these series before scoring."
        )
    wide = wide.with_columns(nmae=pl.col("mae") / pl.col("effective_capacity_mw"))

    # Unpivot to tall format, then split the "name:param" encoding of the parametric wide columns
    # into (metric_name, metric_param). Splitting a string column in Polars yields positionally
    # named struct fields, field_0 and field_1. Scalar metrics have no separator, so their field_1
    # is null and fills to "all". The leftover effective_capacity_mw column is dropped by the
    # unpivot.
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

    # NaN/inf are not null, so Metrics.validate would accept them — and a single non-finite
    # value poisons every downstream MLflow mean. Fail loudly instead, naming the offenders.
    non_finite = metrics_tall.filter(~pl.col("metric_value").is_finite())
    if not non_finite.is_empty():
        offenders = non_finite.select(
            ["time_series_id", "horizon_slice", "metric_name", "metric_param", "metric_value"]
        )
        raise ValueError(
            f"{non_finite.height} metric row(s) have a non-finite metric_value, which would "
            f"silently poison the MLflow aggregate means. First offenders:\n{offenders.head(10)}"
        )

    # Join time_series_type from metadata. Left, so a series with no metadata row surfaces as a
    # null to be named below rather than vanishing from the leaderboard.
    type_map = metadata.select(["time_series_id", "time_series_type"])
    metrics_tall = metrics_tall.join(type_map, on="time_series_id", how="left").with_columns(
        time_series_type=pl.col("time_series_type").cast(pl.Enum(TIME_SERIES_TYPE_SLICES))
    )

    # Every scored series must have a metadata row. A null here would drop the series out of every
    # per-type MLflow aggregate while still counting towards the overall mean. Two CV runs
    # scored over the same population would then not be comparable.
    missing_type = metrics_tall.filter(pl.col("time_series_type").is_null())["time_series_id"]
    if missing_type.len() > 0:
        raise ValueError(
            f"No metadata row for time_series_id(s) {sorted(set(missing_type.to_list()))}; "
            "materialise the time-series metadata for these series before scoring."
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

    A ``metric_param="all"`` metric keeps the bare name as its key token. A parametric metric
    appends its param, as in ``"pinball_loss_p10"``.
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
    ``metric_param="all"`` metrics. For parametric metrics the key token is
    ``{metric_name}_{metric_param}``, as in ``pinball_loss_p10`` and ``picp_p10_p90``. Parametric
    metrics are restricted to the ``_MLFLOW_LOGGED_PARAMETRIC`` headline subset. Key formats:

    - ``"{token}__all"`` — overall aggregate (``horizon_slice="all"``).
    - ``"{token}__{type_slug}"`` — per-type aggregates (``horizon_slice="all"``).
    - ``"{token}__all__{horizon_slice}"`` — overall aggregate per lead-time band (e.g.
      ``"nmae__all__day_ahead"``). Per-type sliced aggregates are deliberately not logged. The
      per-type mean for each lead-time band stays queryable in the ``forecast_metrics`` Delta table,
      as do the pinball loss at all 13 delivery quantiles and the PICP and interval width of all 6
      bands.

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

    # Per-type aggregates. The time_series_type column is `allow_missing` on `Metrics`, so guard
    # that column's presence. Never guard its nullability: `compute_metrics` raises rather than emit
    # a null type.
    if "time_series_type" in base.columns:
        per_type = base.group_by(["metric_key", "time_series_type"]).agg(
            mean_value=pl.col("metric_value").mean()
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
        experiment_name: CV run that produced these forecasts.
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
