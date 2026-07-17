"""Tabular feature-engineering implementation.

``TabularFeatureEngineer`` is the default ``FeatureEngineer``: it maps each gridded-NWP H3 cell
to the nearest time series (``_attach_nearest_nwp_cell``), then runs the declarative tabular
pipeline (``_engineer_features``).

Architecture/Flow:
    ``_engineer_features`` is the main orchestrator. It takes raw string requests, compiles them
    into structured instructions using ``ParsedFeatures.from_strings``, joins the necessary base
    data (power, weather, metadata), then executes the instructions.

Lazy Evaluation:
    The entire pipeline is lazy. ``_engineer_features`` returns a ``pt.LazyFrame[AllFeatures]``
    and never calls ``.collect()``. The only eager operations are ``collect_schema()`` calls,
    which inspect the query plan without executing it.

Nullify Leaky Lags Rationale:
    ``_nullify_leaky_lags`` (in ``_lags.py``) is called at the end of the pipeline to enforce
    physical forecasting constraints: you cannot use a 24-hour lag when forecasting 48 hours
    ahead. It nullifies any lag shorter than or equal to the forecast lead time.
"""

import math
from datetime import datetime

import patito as pt
import polars as pl
from contracts.ml_schemas import AllFeatures
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from contracts.weather_schemas import Nwp

from ml_core.features._lags import _apply_power_lag, _apply_weather_lag, _nullify_leaky_lags
from ml_core.features._nwp import (
    NWP_PUBLICATION_DELAY_HOURS,
    _build_historical_weather,
    _join_nwp_bulk_mode,
    _join_nwp_single_run,
    _upsample_nwp_to_half_hourly,
)
from ml_core.features._parsed_features import STATIC_FEATURE_REGISTRY, ParsedFeatures
from ml_core.features.feature_engineer import FeatureEngineer


def _attach_nearest_nwp_cell(
    nwp: pt.LazyFrame[Nwp],
    time_series_metadata: pt.DataFrame[TimeSeriesMetadata],
) -> pl.LazyFrame:
    """Map each gridded-NWP H3 cell to the time series that sits in it (nearest-cell join).

    NWP is stored per H3 cell at resolution 5; each time series carries its own resolution-5 cell
    in ``h3_res_5``. Joining ``h3_index == h3_res_5`` gives every time series the weather of its
    containing cell. The inner join drops NWP cells with no time series and replicates a cell
    shared by several series across each of them — both correct. The result is keyed by
    ``time_series_id`` (not ``h3_index``), which is what ``_engineer_features`` expects.
    """
    # Strip the Patito subclasses so Polars' cross-subclass join type check doesn't reject the
    # join (see CLAUDE.md). Zero-copy: same underlying Rust LazyFrames.
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
    ) -> pt.LazyFrame[AllFeatures]:
        nwp_per_time_series = _attach_nearest_nwp_cell(nwp, time_series_metadata)
        return _engineer_features(
            selected_features,
            power_time_series,
            time_series_metadata,
            nwp=nwp_per_time_series,
            power_fcst_init_time=power_fcst_init_time,
            nwp_init_time=nwp_init_time,
            nwp_publication_delay_hours=nwp_publication_delay_hours,
        )


def _engineer_features(
    selected_features: set[str],
    power_time_series: pt.LazyFrame[PowerTimeSeries],
    time_series_metadata: pt.DataFrame[TimeSeriesMetadata],
    nwp: pl.LazyFrame | None = None,
    power_fcst_init_time: datetime | None = None,
    nwp_init_time: datetime | None = None,
    nwp_publication_delay_hours: int = NWP_PUBLICATION_DELAY_HOURS,
) -> pt.LazyFrame[AllFeatures]:
    """Engineer features.

    Args:
        selected_features: Set of features to engineer.
        power_time_series: Input power time series.
        time_series_metadata: Metadata for the time series.
        nwp: NWP weather forecast data in physical units, already mapped to
            **per-time-series** rows — it is joined on `time_series_id`, so it must carry a
            `time_series_id` column rather than the raw `h3_index` spatial key. Callers attach
            `time_series_id` first (e.g. via ``_attach_nearest_nwp_cell``, which is lazy and
            does not trigger a collect). Deliberately a plain ``pl.LazyFrame``: the frame is
            ``Nwp`` minus ``h3_index`` plus ``time_series_id``, so no existing contract fits.
        power_fcst_init_time: Controls the operating mode of the function.

            **None — bulk training and multi-run backtesting (recommended for most callers):**
            The function is NWP-centric. It produces one row per
            (time_series_id, nwp_init_time, valid_time, ensemble_member), and derives
            power_fcst_init_time = nwp_init_time + nwp_publication_delay_hours per-row.
            Hindcast rows (valid_time at or before the derived power_fcst_init_time — each
            NWP run's first nwp_publication_delay_hours of valid times) are dropped, so both
            training and backtesting see only rows the live service could deliver
            (valid_time > power_fcst_init_time, strictly). Leaky power lags are nullified
            relative to each row's power_fcst_init_time, so the resulting dataset is safe
            for training.

            **Single datetime — real-time production inference or backfilling:**
            A constant power_fcst_init_time is stamped onto every row, and the NWP join
            matches exclusively the one NWP run identified by nwp_init_time (see below).
            This branch is only appropriate when power_time_series contains observations
            for a single forecast run. Passing a multi-run dataset (e.g., a full year of
            data) will silently produce null weather features for every row except those
            belonging to the matching NWP run.

        nwp_init_time: The NWP run to join in single-run mode (when power_fcst_init_time
            is not None).

            **Live production:** pass the init_time of the NWP run you actually downloaded.
            This is the preferred path because it is exact regardless of publication delays.

            **Backfilling / None:** if omitted, nwp_init_time is derived as
            power_fcst_init_time - nwp_publication_delay_hours (useful when replaying
            historical runs and the exact run is not known).

            Must be None when power_fcst_init_time is None (bulk mode).

        nwp_publication_delay_hours: The delay in hours between the initialization time
            of the NWP forecast and when it becomes publicly available. Used only when
            nwp_init_time is None and power_fcst_init_time is not None.
    """
    if nwp_init_time is not None and power_fcst_init_time is None:
        raise ValueError(
            "nwp_init_time can only be provided in single-run mode (when power_fcst_init_time "
            "is not None). In bulk mode, nwp_init_time is derived per-row from the NWP data."
        )
    power_lf = pl.LazyFrame._from_pyldf(power_time_series._ldf).rename({"time": "valid_time"})
    metadata_lf = pl.LazyFrame._from_pyldf(time_series_metadata.lazy()._ldf)
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
    if processed_nwp is not None and weather_lags:
        if processed_nwp.filter(pl.col("ensemble_member") == 0).limit(1).collect().is_empty():
            raise ValueError(
                "Weather lag features require the NWP control member (ensemble_member == 0) "
                "to build historical weather, but no such rows were found in the NWP data."
            )
    historical_weather = (
        _build_historical_weather(processed_nwp)
        if processed_nwp is not None and weather_lags
        else None
    )

    power_with_metadata = power_lf.join(metadata_lf, on="time_series_id", how="left")

    if power_fcst_init_time is None:
        raw_data = _join_nwp_bulk_mode(
            power_with_metadata, processed_nwp, nwp_publication_delay_hours
        )
    else:
        raw_data = _join_nwp_single_run(
            power_with_metadata,
            processed_nwp,
            power_fcst_init_time,
            nwp_init_time,
            nwp_publication_delay_hours,
        )

    engineered_lf = _apply_post_join_features(
        raw_data,
        parsed_features,
        observed_power_lf=power_lf,
        processed_nwp=processed_nwp,
        historical_weather=historical_weather,
    )
    if power_fcst_init_time is None and nwp_lf is not None:
        # Bulk mode with NWP: drop hindcast rows (each NWP run's first
        # nwp_publication_delay_hours of valid times, which precede the derived
        # power_fcst_init_time). Filtering *after* feature computation keeps window features
        # (e.g. weather rolling means) identical to single-run mode, which likewise computes
        # on the full frame and lets the caller filter before predicting. The strict `>`
        # mirrors what the live service delivers. The no-NWP bulk branch is exempt: it sets
        # power_fcst_init_time = valid_time (lead 0 by construction), so this filter would
        # drop every row. That exemption makes the no-NWP branch training-only: predict
        # output built from it is all-lead-0 and always fails PowerForecast.validate, so a
        # power-only forecaster (e.g. a persistence baseline) must synthesise genuine
        # power_fcst_init_times for inference rather than predict through this branch.
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
        "time_series_type",
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
) -> pl.LazyFrame:
    """Applies requested features dynamically based on parsed feature configurations.

    Args:
        raw_data: The power-with-metadata frame already joined to NWP, one row per forecast
            instance, that feature columns are accumulated onto.
        parsed_features: The requested features, parsed into typed objects.
        observed_power_lf: The dense observed-power series (one row per
            ``(time_series_id, valid_time)``), used as the lookup source for power lags. Sourcing
            from this mode-independent frame — rather than self-joining the NWP-gridded
            ``raw_data`` — keeps power lags identical across bulk and single-run modes and avoids
            fan-out when overlapping NWP runs replicate a ``valid_time``.
        processed_nwp: The processed NWP frame, required for weather lag features.
        historical_weather: The freshest-run weather frame, required for weather lag features.
    """
    engineered_lf = raw_data

    if parsed_features.time_features:
        engineered_lf = _apply_local_time_features(engineered_lf)

    if parsed_features.static_features:
        exprs = [STATIC_FEATURE_REGISTRY[f] for f in parsed_features.static_features]
        engineered_lf = engineered_lf.with_columns(exprs)

    for lag_feat in parsed_features.lags:
        if lag_feat.base_col == "power":
            engineered_lf = _apply_power_lag(engineered_lf, observed_power_lf, lag_feat)
        else:
            if processed_nwp is None:
                raise ValueError(
                    "processed_nwp cannot be None when applying a weather lag feature."
                )
            assert historical_weather is not None
            engineered_lf = _apply_weather_lag(
                engineered_lf, processed_nwp, lag_feat, historical_weather
            )

    for rolling_feat in parsed_features.rolling_means:
        engineered_lf = _apply_rolling_mean_feature(
            engineered_lf, rolling_feat.base_col, rolling_feat.hours
        )

    leaky_features = parsed_features.get_leaky_features()
    if leaky_features and "power_fcst_init_time" in engineered_lf.collect_schema().names():
        engineered_lf = _nullify_leaky_lags(engineered_lf, leaky_features)

    return engineered_lf


def _apply_rolling_mean_feature(lf: pl.LazyFrame, base_col: str, window_hours: int) -> pl.LazyFrame:
    """Applies a rolling mean feature, grouped by (time_series_id, nwp_init_time, ensemble_member).

    Grouping by nwp_init_time prevents the rolling window from mixing values across different
    NWP runs, which would contaminate the feature with data from other forecast initializations.

    Cross-mode invariant: the rolling aggregation MUST be null-skipping over the value column
    (mean/min/max/std/median/sum) and MUST NOT be row-count-dependent (e.g. ``.len()``). Single-run
    mode stamps a constant nwp_init_time, so each group is padded with out-of-window rows whose
    weather is null; a null-skipping aggregation ignores them (so values match bulk mode), but a
    row count would not — silently skewing the feature between training and serving. This is locked
    by test_cross_mode_equivalence.py.
    """
    rolled = lf.rolling(
        index_column="valid_time",
        period=f"{window_hours}h",
        group_by=["time_series_id", "nwp_init_time", "ensemble_member"],
    ).agg(pl.col(base_col).mean().alias(f"{base_col}_rolling_mean_{window_hours}h"))

    join_keys = ["time_series_id", "nwp_init_time", "ensemble_member", "valid_time"]
    return lf.join(rolled, on=join_keys, how="left")


def _apply_local_time_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Applies local time features (time of day, day of week, time of year) to the LazyFrame.

    Why local time? Energy consumption patterns are driven by human behavior, which follows
    local time (including Daylight Saving Time), not UTC. A 9 AM peak in winter (UTC) is
    different from a 9 AM peak in summer (UTC+1).

    Args:
        lf: The LazyFrame containing a 'valid_time' column in UTC.

    Returns:
        A LazyFrame with new local time features.
    """
    lf = lf.with_columns(
        local_time=pl.col("valid_time")
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone("Europe/London")
    )

    lf = lf.with_columns(
        local_utc_offset=(
            pl.col("local_time").dt.base_utc_offset() + pl.col("local_time").dt.dst_offset()
        ).dt.total_seconds()
        / 3600
    )

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
