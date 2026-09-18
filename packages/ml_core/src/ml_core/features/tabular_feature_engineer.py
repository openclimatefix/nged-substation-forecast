"""Tabular feature-engineering implementation.

``TabularFeatureEngineer`` is the default ``FeatureEngineer``: it maps each gridded-NWP H3 cell to
the nearest time series (``_attach_nearest_nwp_cell``), then runs the declarative tabular pipeline
(``_engineer_features``). Numerical weather prediction (NWP) output is stored one row per H3 cell,
H3 being a hexagonal grid over the globe at numbered resolutions, and each time series records the
resolution-5 cell it sits in.

Architecture/Flow:
    ``_engineer_features`` is the main orchestrator. The function takes raw string requests,
    compiles them into structured instructions using ``ParsedFeatures.from_strings``, joins the
    necessary base data (power, weather, metadata), then executes the instructions.

Lazy Evaluation:
    The pipeline stays lazy from input to output: ``_engineer_features`` returns a
        ``pt.LazyFrame[AllFeatures]`` without collecting the result. Two eager operations are the
        exceptions, and neither reads the full table. ``collect_schema()`` inspects the query plan's
        column names and dtypes without executing it. The NWP control-member check calls
        ``.collect()`` on a frame already cut to ``.limit(1)``, so the eager read is bounded to at
        most one row rather than the whole table. That bound is what makes collecting here
        acceptable. In bulk mode (training/backtesting), a missing control member raises
        immediately. In single-run mode (production inference, replay), a missing control member is
        absent input, not a contract violation. The absence is logged. Past-target-time weather lags
        are then left to degrade to null through the ordinary join-miss path.

Nullify Leaky Lags Rationale:
    ``_nullify_leaky_lags`` (in ``_lags.py``) is called at the end of the pipeline to enforce
    physical forecasting constraints: you cannot use a 24-hour lag when forecasting 48 hours ahead.
    ``_nullify_leaky_lags`` nullifies any lag shorter than or equal to the forecast lead time.
"""

import logging
import math
from datetime import datetime

import patito as pt
import polars as pl
from contracts.ml_schemas import AllFeatures
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from contracts.weather_schemas import Nwp
from weather_utils import NWP_ANALYSIS_MEMBER, select_analysis_proxy

from ml_core.features._lags import _apply_power_lag, _apply_weather_lag, _nullify_leaky_lags
from ml_core.features._nwp import (
    NWP_PUBLICATION_DELAY_HOURS,
    _join_nwp_bulk_mode,
    _join_nwp_single_run,
    _resolve_nwp_init_time,
    _upsample_nwp_to_half_hourly,
)
from ml_core.features._parsed_features import STATIC_FEATURE_REGISTRY, LagFeature, ParsedFeatures
from ml_core.features.feature_engineer import DEFAULT_LOCAL_TIMEZONE, FeatureEngineer

logger = logging.getLogger(__name__)


def _attach_nearest_nwp_cell(
    nwp: pt.LazyFrame[Nwp],
    time_series_metadata: pt.DataFrame[TimeSeriesMetadata],
) -> pl.LazyFrame:
    """Map each gridded-NWP H3 cell to the time series that sits in it (nearest-cell join).

    NWP is stored per H3 cell at resolution 5; each time series carries its own resolution-5 cell
    in ``h3_res_5``. Joining ``h3_index == h3_res_5`` gives every time series the weather of its
    containing cell. The inner join drops NWP cells with no time series, and replicates a cell
    shared by several series across each of those series. Both behaviours are correct. The result
    is keyed by ``time_series_id`` (not ``h3_index``), which is what ``_engineer_features``
    expects.
    """
    # Patito wraps a Polars frame in a schema class, and that class is a Python subclass carried in
    # the frame's own type. Strip the Patito subclasses so Polars' cross-subclass join type check
    # doesn't reject the join (see the `polars-patito-gotchas` skill). Zero-copy: same underlying
    # Rust LazyFrames.
    nwp_plain = pl.LazyFrame._from_pyldf(nwp._ldf)
    cell_to_ts = pl.LazyFrame._from_pyldf(time_series_metadata.lazy()._ldf).select(
        "time_series_id", "h3_res_5"
    )
    return nwp_plain.join(cell_to_ts, left_on="h3_index", right_on="h3_res_5", how="inner").drop(
        "h3_index"
    )


class TabularFeatureEngineer(FeatureEngineer):
    """Nearest res-5 NWP cell per time series, then the declarative tabular pipeline."""

    def engineer(
        self,
        *,
        selected_features: set[str],
        power_time_series: pt.LazyFrame[PowerTimeSeries],
        time_series_metadata: pt.DataFrame[TimeSeriesMetadata],
        nwp: pt.LazyFrame[Nwp],
        power_fcst_init_time: datetime | None = None,
        nwp_init_time: datetime | None = None,
        nwp_publication_delay_hours: int = NWP_PUBLICATION_DELAY_HOURS,
        local_timezone: str = DEFAULT_LOCAL_TIMEZONE,
    ) -> pt.LazyFrame[AllFeatures]:
        """Map each NWP cell to its nearest time series, then run the tabular feature pipeline.

        See `FeatureEngineer.engineer` for the argument and operating-mode contract, and
        ``_engineer_features`` in this module for the pipeline itself — including
        ``local_timezone``, the IANA zone the local-time features are computed in.
        """
        nwp_per_time_series = _attach_nearest_nwp_cell(nwp, time_series_metadata)
        return _engineer_features(
            selected_features,
            power_time_series,
            time_series_metadata,
            nwp=nwp_per_time_series,
            power_fcst_init_time=power_fcst_init_time,
            nwp_init_time=nwp_init_time,
            nwp_publication_delay_hours=nwp_publication_delay_hours,
            local_timezone=local_timezone,
        )


def _check_or_warn_on_missing_control_member(
    nwp_lf: pl.LazyFrame | None,
    *,
    weather_lags: list[LagFeature],
    power_fcst_init_time: datetime | None,
    nwp_init_time: datetime | None,
    nwp_publication_delay_hours: int,
) -> None:
    """When no control member exists: fail fast in bulk mode, degrade and log in single-run mode."""
    # Probes the raw frame, not the upsampled frame: `SLICE` cannot push through the upsample's
    # window functions, so probing post-upsample would run the whole upsample before answering.
    control_member_missing = (
        nwp_lf is not None
        and bool(weather_lags)
        and nwp_lf.filter(pl.col("ensemble_member") == NWP_ANALYSIS_MEMBER)
        .limit(1)
        .collect()
        .is_empty()
    )
    if not control_member_missing:
        return
    if power_fcst_init_time is None:
        # Bulk mode: fail fast (inherent-stability.md rule 9).
        raise ValueError(
            "Weather lag features require the NWP control member (ensemble_member == 0) to "
            "build historical weather during bulk training or backtesting, but no such rows "
            "were found in the NWP data."
        )
    # Single-run mode: degrade rather than raise (inherent-stability.md rule 1). Every
    # past-target-time weather lag comes back null through the ordinary join-miss path in
    # `_engineer_features`.
    resolved_nwp_init_time = _resolve_nwp_init_time(
        nwp_init_time=nwp_init_time,
        power_fcst_init_time=power_fcst_init_time,
        nwp_publication_delay_hours=nwp_publication_delay_hours,
    )
    logger.warning(
        "NWP run %s has no control member (ensemble_member == 0). Every weather lag feature will "
        "be null for this slot over the first lag_hours of its horizon, where the lag points "
        "back before power_fcst_init_time. The same-run join answers the rest of the horizon.",
        resolved_nwp_init_time,
    )


def _engineer_features(
    selected_features: set[str],
    power_time_series: pt.LazyFrame[PowerTimeSeries],
    time_series_metadata: pt.DataFrame[TimeSeriesMetadata],
    nwp: pl.LazyFrame | None = None,
    power_fcst_init_time: datetime | None = None,
    nwp_init_time: datetime | None = None,
    nwp_publication_delay_hours: int = NWP_PUBLICATION_DELAY_HOURS,
    local_timezone: str = DEFAULT_LOCAL_TIMEZONE,
) -> pt.LazyFrame[AllFeatures]:
    """Engineer features.

    Args:
        selected_features: Set of features to engineer.
        power_time_series: Input power time series.
        time_series_metadata: Metadata for the time series.
        nwp: NWP weather forecast data in physical units, already mapped to
            **per-time-series** rows. The frame is joined on
            `time_series_id`, so the frame must carry a `time_series_id` column rather than the raw
            `h3_index` spatial key. Callers attach `time_series_id` first (e.g. via
            ``_attach_nearest_nwp_cell``, which is lazy and does not trigger a collect).
            Deliberately a plain ``pl.LazyFrame``. A frame in this codebase normally carries a
            Patito schema class — a *contract* — that declares and validates its columns, and the
            ``pt.LazyFrame[X]`` type records which one. This frame is ``Nwp`` minus ``h3_index``
            plus ``time_series_id``, so no existing contract fits, and the frame falls back to an
            unvalidated Polars frame.
        power_fcst_init_time: Controls the operating mode of the function.

            **None — bulk training and multi-run backtesting (recommended for most callers):** The
            function is NWP-centric. It produces one row per (time_series_id, nwp_init_time,
            valid_time, ensemble_member), and derives power_fcst_init_time = nwp_init_time +
            nwp_publication_delay_hours per-row. Hindcast rows are dropped. A hindcast row is a row
            whose valid_time falls at or before the derived power_fcst_init_time, which is each NWP
            run's first nwp_publication_delay_hours of valid times. Both training and backtesting
            therefore see only rows the live service could deliver (valid_time >
            power_fcst_init_time, strictly). Leaky power lags are nullified relative to each row's
            power_fcst_init_time, so the resulting dataset is safe for training.

            **Single datetime — real-time production inference or backfilling:** A constant
            power_fcst_init_time is stamped onto every row, and the NWP join matches exclusively the
            one NWP run identified by nwp_init_time (see below). This branch is only appropriate
            when power_time_series contains observations for a single forecast run. Passing a
            multi-run dataset (e.g., a full year of data) will silently produce null weather
            features for every row except the rows belonging to the matching NWP run.

        nwp_init_time: The NWP run to join in single-run mode (when power_fcst_init_time
            is not None).

            **Live production:** pass the init_time of the NWP run you actually downloaded. Passing
            the downloaded run's init_time is the preferred path, because the run identity is then
            exact regardless of publication delays.

            **Backfilling / None:** if omitted, nwp_init_time is derived as
            power_fcst_init_time - nwp_publication_delay_hours (useful when replaying
            historical runs and the exact run is not known).

            Must be None when power_fcst_init_time is None (bulk mode).

        nwp_publication_delay_hours: Hours after an NWP run's ``nwp_init_time`` before that run is
            usable. Usable means arrival on our own disk, not upstream publication. See
            ``ml_core.features._nwp.NWP_PUBLICATION_DELAY_HOURS`` for what drives the default and
            its derivation. ``_engineer_features`` uses the delay to derive ``power_fcst_init_time``
            from ``nwp_init_time`` in bulk mode, and to derive ``nwp_init_time`` when a single-run
            caller omits ``nwp_init_time``. Single-run mode consumes the delay nowhere else.
        local_timezone: IANA zone the local-time features (time of day, time of year, day of week,
            offset from UTC) are computed in. Defaults to ``DEFAULT_LOCAL_TIMEZONE``.

    Returns:
        A ``pt.LazyFrame[AllFeatures]``, still lazy. Bulk mode returns one row per
        ``time_series_id``, ``nwp_init_time``, ``valid_time``, and ``ensemble_member``, minus each
        NWP run's hindcast rows. With no NWP requested, bulk mode instead returns one row per
        ``time_series_id`` and ``valid_time``, with ``nwp_init_time`` null and
        ``power_fcst_init_time`` equal to ``valid_time``. Single-run mode returns one row per
        ``time_series_id``, ``valid_time``, and (when NWP was requested) ``ensemble_member``, all
        sharing the one stamped ``power_fcst_init_time`` and ``nwp_init_time``. Every row carries
        one column per name in ``selected_features``, alongside the identifying columns above. Every
        row also carries the observed ``power`` for that ``valid_time``, where an observation is
        available.
    """
    if nwp_init_time is not None and power_fcst_init_time is None:
        raise ValueError(
            "nwp_init_time can only be provided in single-run mode (when power_fcst_init_time "
            "is not None). In bulk mode, nwp_init_time is derived per-row from the NWP data."
        )
    power_lf = pl.LazyFrame._from_pyldf(power_time_series._ldf).rename({"time": "valid_time"})
    metadata_lf = pl.LazyFrame._from_pyldf(time_series_metadata.lazy()._ldf)
    # The metadata is the registry of what we forecast, so a series with power observations but no
    # metadata row is dropped here rather than carried to the model. Bulk mode drops a series with
    # no metadata row regardless, because `_attach_nearest_nwp_cell` inner-joins on `h3_res_5`.
    # Single-run mode is power-centric, so single-run mode would otherwise keep the series with
    # every weather feature null and predict on it. That forecast would be garbage, and would read
    # as healthy because the series is present. Dropping the series instead makes
    # `live_forecasts_are_healthy` report that series, via `missing_time_series_ids`. Unconditional,
    # so the output row set never depends on which features were requested.
    power_lf = power_lf.join(metadata_lf.select("time_series_id"), on="time_series_id", how="semi")
    nwp_lf: pl.LazyFrame | None = nwp

    parsed_features = ParsedFeatures.from_strings(selected_features)
    if nwp_lf is None and parsed_features.requires_weather_data():
        raise ValueError("Weather features were requested but no NWP data was provided.")

    if nwp_lf is not None:
        _renamed = nwp_lf.rename({"init_time": "nwp_init_time"})
        _upsampled = _upsample_nwp_to_half_hourly(_renamed)
        processed_nwp: pl.LazyFrame | None = _upsampled.with_columns(
            nwp_lead_time_hours=(
                (pl.col("valid_time") - pl.col("nwp_init_time")).dt.total_seconds() / 3600
            ).cast(pl.Float32)
        )
    else:
        processed_nwp = None
    weather_lags = [lag for lag in parsed_features.lags if lag.base_col != "power"]
    _check_or_warn_on_missing_control_member(
        nwp_lf,
        weather_lags=weather_lags,
        power_fcst_init_time=power_fcst_init_time,
        nwp_init_time=nwp_init_time,
        nwp_publication_delay_hours=nwp_publication_delay_hours,
    )
    if processed_nwp is None or not weather_lags:
        historical_weather = None
    elif power_fcst_init_time is not None:
        # Single-run mode: cap the freshest-run (analysis-proxy) selection at the NWP run this call
        # already selected, so a later run cannot answer a past target time. That run is by
        # construction the freshest run that was available. `select_nwp_init_time` returns the
        # newest qualifying run in either of its two availability modes: `"live"`, where the cutoff
        # is power_fcst_init_time itself because the table only holds runs that were genuinely
        # published, and `"replay"`, where the cutoff is power_fcst_init_time minus
        # nwp_publication_delay_hours. The runs at or before the selected run are therefore exactly
        # the runs that were available, so no modelled publication delay is needed to work out which
        # runs those are. A slot is one scheduled forecast the live service issues. A `"live"` slot
        # therefore keeps the run it is forecasting with even when that run is fresher than
        # nwp_publication_delay_hours. Keeping that run matches bulk mode's effective ceiling — the
        # invariant spelled out in the branch below.
        # The filter is input validation for a reusable package, not a guard on a state production
        # can reach. The single-run caller in this repo today, `live_forecasts`, passes a frame
        # holding just the selected run. The filter therefore drops nothing. The filter guards the
        # multi-run frame this function's docstring lets a single-run caller pass.
        historical_weather = select_analysis_proxy(
            processed_nwp.filter(
                pl.col("nwp_init_time")
                <= _resolve_nwp_init_time(
                    nwp_init_time=nwp_init_time,
                    power_fcst_init_time=power_fcst_init_time,
                    nwp_publication_delay_hours=nwp_publication_delay_hours,
                )
            ),
            group_key="time_series_id",
            init_time_col="nwp_init_time",
        )
    else:
        # Bulk mode: no ceiling, deliberately. historical_weather is built once, globally, here. The
        # build happens before every row gets its own derived power_fcst_init_time, so there is no
        # single scalar cutoff to give the selection. None is needed either. For a hindcast target
        # (valid_time <= that row's own power_fcst_init_time), the only causally-eligible runs are
        # the row's own run or an earlier run. An NWP run's earliest valid_time is its own
        # init_time, so a later run can never hold a valid_time that early. That alone rules out a
        # later run holding an earlier target. The same argument does not rule out a run initialised
        # between the row's own run and its derived power_fcst_init_time (= nwp_init_time +
        # nwp_publication_delay_hours). No run can fall inside that gap as long as runs are spaced
        # further apart than the delay. We ingest exactly one ECMWF ENS run per day,
        # and NWP_PUBLICATION_DELAY_HOURS is 9, so no run falls strictly between a row's own run and
        # its power_fcst_init_time. That absence of an intervening run is the invariant to re-check
        # if the freshest-run selection ever needs to become row-aware, or if a second daily run or
        # a delay past 24 hours is introduced.
        historical_weather = select_analysis_proxy(
            processed_nwp, group_key="time_series_id", init_time_col="nwp_init_time"
        )

    if power_fcst_init_time is None:
        raw_data = _join_nwp_bulk_mode(
            power_lf=power_lf,
            processed_nwp=processed_nwp,
            nwp_publication_delay_hours=nwp_publication_delay_hours,
        )
    else:
        raw_data = _join_nwp_single_run(
            power_lf=power_lf,
            processed_nwp=processed_nwp,
            power_fcst_init_time=power_fcst_init_time,
            nwp_init_time=nwp_init_time,
            nwp_publication_delay_hours=nwp_publication_delay_hours,
        )

    # Metadata joins *after* the NWP join, not before. Bulk mode left-joins power onto NWP. A row
    # whose valid_time has no power observation would therefore lose time_series_type, even though
    # that row's time_series_id is known. The join is conditional because Polars keeps a left join
    # whose right-hand columns are all projected away. An unconditional join here would therefore
    # run for every caller that never asked for the column.
    if "time_series_type" in selected_features:
        raw_data = raw_data.join(metadata_lf, on="time_series_id", how="left")

    engineered_lf = _apply_post_join_features(
        raw_data,
        parsed_features,
        observed_power_lf=power_lf,
        processed_nwp=processed_nwp,
        historical_weather=historical_weather,
        local_timezone=local_timezone,
    )
    if power_fcst_init_time is None and nwp_lf is not None:
        # Bulk mode with NWP: drop hindcast rows (each NWP run's first nwp_publication_delay_hours
        # Filtering *after* feature computation keeps window features (e.g. weather rolling means)
        # identical to single-run mode. Single-run mode likewise computes on the full frame and lets
        # the caller filter before predicting. The strict `>` mirrors what the live service
        # delivers. The no-NWP bulk branch is exempt: it sets power_fcst_init_time = valid_time
        # (lead 0 by construction), so this filter would drop every row. That exemption makes the
        # no-NWP branch training-only: predict output built from that branch is all-lead-0 and
        # always fails PowerForecast.validate. A power-only forecaster (e.g. a persistence baseline)
        # must therefore synthesise genuine power_fcst_init_times for inference rather than predict
        # through this branch.
        engineered_lf = engineered_lf.filter(pl.col("valid_time") > pl.col("power_fcst_init_time"))
    final_lf = _select_output_columns(engineered_lf, selected_features)
    return pt.LazyFrame.from_existing(final_lf).set_model(AllFeatures)


def _select_output_columns(
    engineered_lf: pl.LazyFrame, selected_features: set[str]
) -> pl.LazyFrame:
    """Assert all requested features were produced, then select and order the output columns."""
    schema_names = set(engineered_lf.collect_schema().names())

    missing_cols = selected_features - schema_names
    if missing_cols:
        raise ValueError(f"Feature engineering failed to create or find: {missing_cols}")

    base_cols = [
        "valid_time",
        "time_series_id",
        "power",
        "power_fcst_init_time",
        "nwp_init_time",
    ]
    if "nwp_lead_time_hours" in schema_names:
        base_cols.append("nwp_lead_time_hours")
    if "ensemble_member" in schema_names:
        base_cols.append("ensemble_member")

    # dict.fromkeys preserves insertion order while deduplicating
    cols_to_select = list(dict.fromkeys(base_cols + list(selected_features)))
    return engineered_lf.select(cols_to_select)


def _apply_post_join_features(
    raw_data: pl.LazyFrame,
    parsed_features: ParsedFeatures,
    observed_power_lf: pl.LazyFrame,
    processed_nwp: pl.LazyFrame | None = None,
    historical_weather: pl.LazyFrame | None = None,
    local_timezone: str = DEFAULT_LOCAL_TIMEZONE,
) -> pl.LazyFrame:
    """Applies requested features dynamically based on parsed feature configurations.

    Args:
        raw_data: The power frame already joined to NWP, one row per forecast instance, that
            feature columns are accumulated onto.
        parsed_features: The requested features, parsed into typed objects.
        observed_power_lf: The dense observed-power series (one row per ``(time_series_id,
            valid_time)``), used as the lookup source for power lags. Power lags are sourced from
            this mode-independent frame rather than from a self-join of the NWP-gridded
            ``raw_data``. Sourcing that way keeps power lags identical across bulk and single-run
            modes. Sourcing that way also avoids fan-out when overlapping NWP runs replicate a
            ``valid_time``.
        processed_nwp: The processed NWP frame, required for weather lag features.
        historical_weather: The freshest-run weather frame, required for weather lag features.
        local_timezone: IANA zone the local-time features are computed in, when requested.

    Returns:
        ``raw_data`` with the requested feature columns added: local time features when
        ``parsed_features.time_features`` is set, ``STATIC_FEATURE_REGISTRY`` columns for each
        requested static feature, one column per requested lag (power lags via ``_apply_power_lag``,
        weather lags via ``_apply_weather_lag``), and one column per requested rolling mean. Any
        leaky lag or rolling-mean column is then nullified by ``_nullify_leaky_lags``. Still lazy;
        the row count of ``raw_data`` is unchanged, though the row order may not be.
    """
    engineered_lf = raw_data

    if parsed_features.time_features:
        engineered_lf = _apply_local_time_features(engineered_lf, local_timezone=local_timezone)

    if parsed_features.static_features:
        exprs = [STATIC_FEATURE_REGISTRY[f] for f in parsed_features.static_features]
        engineered_lf = engineered_lf.with_columns(exprs)

    for lag_feat in parsed_features.lags:
        if lag_feat.base_col == "power":
            engineered_lf = _apply_power_lag(engineered_lf, observed_power_lf, lag_feat)
        else:
            # Both processed_nwp and historical_weather are non-None whenever a weather lag is
            # requested, by construction: `_engineer_features` raises for a weather lag with no NWP,
            # and builds `historical_weather` from the same condition.
            assert processed_nwp is not None
            assert historical_weather is not None
            engineered_lf = _apply_weather_lag(
                engineered_lf, processed_nwp, lag_feat, historical_weather
            )

    for rolling_feat in parsed_features.rolling_means:
        engineered_lf = _apply_rolling_mean_feature(
            engineered_lf, rolling_feat.base_col, rolling_feat.hours
        )

    leaky_features = parsed_features.get_leaky_features()
    if leaky_features:
        engineered_lf = _nullify_leaky_lags(engineered_lf, leaky_features)

    return engineered_lf


def _apply_rolling_mean_feature(lf: pl.LazyFrame, base_col: str, window_hours: int) -> pl.LazyFrame:
    """Applies a rolling mean feature, grouped by (time_series_id, nwp_init_time, ensemble_member).

    Grouping by nwp_init_time prevents the rolling window from mixing values across different NWP
    runs, which would contaminate the feature with data from other forecast initialisations.

    Cross-mode invariant: the rolling aggregation MUST be null-skipping over the value column
    (mean/min/max/std/median/sum) and MUST NOT be row-count-dependent (e.g. ``.len()``). The two
    modes present different numbers of rows to a group. Single-run mode stamps a constant
    nwp_init_time. Rows the NWP join missed carry a null ensemble_member, so those rows form a
    group of their own. A null-skipping aggregation therefore matches across modes. A
    row-count-dependent aggregation would not match, and would silently skew the feature between
    training and serving. The cross-mode invariant is locked by test_cross_mode_equivalence.py.

    ``rolling_mean_by`` inside ``over(..., order_by=)`` rather than ``lf.rolling().agg()``: the
    window form sorts only within each group and emits one value per input row. The window form
    therefore needs neither a join back onto the frame nor a global sort to satisfy a sortedness
    precondition.
    """
    return lf.with_columns(
        **{
            f"{base_col}_rolling_mean_{window_hours}h": pl.col(base_col)
            .rolling_mean_by("valid_time", window_size=f"{window_hours}h", closed="right")
            .over(["time_series_id", "nwp_init_time", "ensemble_member"], order_by="valid_time")
        }
    )


def _local_utc_offset_minutes(local_time: pl.Expr) -> pl.Expr:
    """Returns the local offset from UTC in minutes.

    Minutes represent every offset in scope exactly, so sub-hour zones stay distinct rather than
    collapsing into one another. India's +5:30 is ``330`` and Nepal's +5:45 is ``345``. Australia's
    +9:30 (``570``) does not collide with +9:00 (``540``). ``Int16`` is ample — the real extremes
    are ``Etc/GMT+12`` (−720) and ``Pacific/Kiritimati`` (+840).

    Whole minutes depend on the era bound on ``PowerTimeSeries.time``: ``PowerTimeSeries.validate``
    rejects any ``time`` before ``MIN_PLAUSIBLE_DATETIME``, which is 2000-01-01 (issue #466). A
    handful of IANA offsets are not whole minutes, all of them mean solar time from before a zone
    standardised. ``Europe/London`` stood at UTC−0:01:15 until 1847, and Liberia stood at
    UTC−0:44:30, which was legal time there until 1972. ``_local_utc_offset_minutes`` truncates
    those offsets. A timestamp from that era is malformed input, which belongs at the contract
    boundary rather than being defended against here.

    Args:
        local_time: An expression yielding a time-zone-aware datetime in the local time zone.

    Returns:
        An ``Int16`` expression holding the offset from UTC in minutes.
    """
    offset = local_time.dt.base_utc_offset() + local_time.dt.dst_offset()
    return offset.dt.total_minutes().cast(pl.Int16)


def _apply_local_time_features(
    lf: pl.LazyFrame, local_timezone: str = DEFAULT_LOCAL_TIMEZONE
) -> pl.LazyFrame:
    """Applies the local-time features to the LazyFrame.

    The features are the time of day, the time of year, the day of week, and the offset from UTC,
    each derived from ``valid_time`` converted into ``local_timezone``.

    Why local time? Energy consumption patterns are driven by human behaviour, which follows local
    time (including daylight saving time), not UTC. A 9 AM peak in winter (UTC) is different from a
    9 AM peak in summer (UTC+1).

    Args:
        lf: The LazyFrame containing a 'valid_time' column in UTC.
        local_timezone: IANA zone to convert ``valid_time`` into before deriving the local-time
            features. Defaults to ``DEFAULT_LOCAL_TIMEZONE``.

    Returns:
        A LazyFrame with new local time features.
    """
    lf = lf.with_columns(
        local_time=pl.col("valid_time")
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone(local_timezone)
    )

    lf = lf.with_columns(local_utc_offset_minutes=_local_utc_offset_minutes(pl.col("local_time")))

    local_hour_float = pl.col("local_time").dt.hour() + pl.col("local_time").dt.minute() / 60.0
    local_year_fraction = pl.col("local_time").dt.ordinal_day() / 366.0
    local_weekday = pl.col("local_time").dt.weekday()

    weekday_map = {
        1: "Monday",
        2: "Tuesday",
        3: "Wednesday",
        4: "Thursday",
        5: "Friday",
        6: "Saturday",
        7: "Sunday",
    }

    lf = lf.with_columns(
        local_time_of_day_sin=(local_hour_float / 24.0 * 2 * math.pi).sin().cast(pl.Float32),
        local_time_of_day_cos=(local_hour_float / 24.0 * 2 * math.pi).cos().cast(pl.Float32),
        local_time_of_year_sin=(local_year_fraction * 2 * math.pi).sin().cast(pl.Float32),
        local_time_of_year_cos=(local_year_fraction * 2 * math.pi).cos().cast(pl.Float32),
        local_day_of_week_sin=(local_weekday / 7.0 * 2 * math.pi).sin().cast(pl.Float32),
        local_day_of_week_cos=(local_weekday / 7.0 * 2 * math.pi).cos().cast(pl.Float32),
        local_day_of_week=local_weekday.replace_strict(weekday_map, return_dtype=pl.String).cast(
            pl.Enum(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        ),
    )

    return lf.drop("local_time")
