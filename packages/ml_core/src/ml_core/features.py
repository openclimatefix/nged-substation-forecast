"""Declarative feature engineering pipeline for time-series forecasting.

Architecture/Flow:
    `engineer_features` is the main orchestrator of this module. It takes raw string requests,
    compiles them into structured instructions using `ParsedFeatures.from_strings`, joins the necessary
    base data (power, weather, metadata), and then executes the instructions via
    `_apply_post_join_features`.

Lazy Evaluation:
    The entire pipeline is lazy. `engineer_features` returns a `pt.LazyFrame[AllFeatures]` and
    never calls `.collect()`. Callers should defer `.collect()` as late as possible — ideally only
    at the model boundary (e.g., inside `BaseForecaster.train` or `BaseForecaster.predict`), so
    that Polars can optimise the full query plan end-to-end. The only eager operations in this
    module are `collect_schema()` calls, which inspect the query plan without executing it.

Nullify Leaky Lags Rationale:
    `_nullify_leaky_lags` is called at the end of the pipeline to enforce physical forecasting
    constraints. In a real-world scenario, you cannot use a 24-hour lag if you are forecasting
    48 hours ahead (the 24-hour lag hasn't happened yet at the time the forecast is issued).
    This function prevents lookahead bias by nullifying any lag that is shorter than or equal
    to the forecast `lead_time_hours`.
"""

import math
import re
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta  # noqa: F401
from typing import Annotated, ClassVar, Final, Literal, Self, Sequence, cast, get_args

import patito as pt
import polars as pl
from contracts.common import UTC_DATETIME_DTYPE
from contracts.ml_schemas import AllFeatures, SafeInputBaseColumn, TimeFeature
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from contracts.weather_schemas import NwpInMemory, WeatherFeature
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator  # noqa: F401

# Static features registry
# These are basic features that don't require parameterization.
StaticFeature = Literal["windchill"]

STATIC_FEATURE_REGISTRY: Final[dict[StaticFeature, pl.Expr]] = {
    "windchill": (
        13.12
        + 0.6215 * pl.col("temperature_2m")
        - 11.37 * ((pl.col("wind_speed_10m") * 3.6) ** 0.16)
        + 0.3965 * pl.col("temperature_2m") * ((pl.col("wind_speed_10m") * 3.6) ** 0.16)
    ).alias("windchill"),
}


# Prevents physically impossible time shifts and lookahead bias by enforcing strict bounds on lag
# hours and rolling window hours.
Hours = Annotated[int, Field(gt=0, le=365 * 24 * 2)]


class BaseLookbackFeature(BaseModel):
    """Base class for lookback features like lags and rolling means.

    Its main job is to parse strings like 'power_lag_24h'.
    """

    model_config = ConfigDict(frozen=True)

    SUFFIX: ClassVar[str]

    base_col: WeatherFeature
    hours: Hours

    @property
    def string_repr(self) -> str:
        return f"{self.base_col}_{self.SUFFIX}_{self.hours}h"

    @classmethod
    def from_str(cls, value: str) -> Self:
        """Parse and validate a feature name string into an instance."""
        pattern = re.compile(rf"^(.*)_{cls.SUFFIX}_(\d+)h$")
        match = pattern.match(value)
        if not match:
            raise ValueError(f"Invalid {cls.SUFFIX} feature name format: {value}")
        base_col, hours_str = match.groups()
        return cls(base_col=base_col, hours=int(hours_str))  # ty: ignore[invalid-argument-type]

    @abstractmethod
    def is_leaky(self) -> bool:
        """Returns True if this feature leaks information that wouldn't be available at inference
        time into the ML model's inputs, and hence must be nullified."""
        pass

    def is_weather_feature(self) -> bool:
        return self.base_col in get_args(WeatherFeature)


class LagFeature(BaseLookbackFeature):
    """Represents a parsed lag feature."""

    SUFFIX: ClassVar[str] = "lag"
    base_col: WeatherFeature | Literal["power"]

    def is_leaky(self) -> bool:
        """Power lags are always leaky: observed power may not exist at forecast-issue time
        if the lagged observation post-dates power_fcst_init_time. Per-row nullification is
        handled downstream by _nullify_leaky_lags."""
        return self.base_col == "power"


class RollingFeature(BaseLookbackFeature):
    """Represents a parsed rolling mean feature.

    Note that computing the rolling mean of 'power' is currently forbidden to prevent lookahead
    bias."""

    # TODO: Generalise to support more weather summary stats over the rolling window, i.e.
    # rolling_{mean,min,max,std,median,sum} (add an `agg` field here + dispatch in
    # _apply_rolling_mean_feature). All of these are null-skipping, so they preserve the
    # cross-mode invariant documented on that function; a row-count-based agg (.len()) would not.
    #
    # TODO: (separate concern) Implement "Latest Available Rolling Mean anchored to T_init" to
    # allow non-leaky rolling *power* features (e.g. mean of the most recent 24h of observed power,
    # broadcast to every forecast horizon). Power rolling stays forbidden until then.

    SUFFIX: ClassVar[str] = "rolling_mean"

    def is_leaky(self) -> bool:
        """Weather rolling means are never leaky: NWP forecasts are available for future valid_times,
        so a weather rolling mean (e.g., mean temperature over the 6h window ending at valid_time)
        is always known at inference time."""
        return False


@dataclass
class ParsedFeatures:
    """Compiled configuration object for feature engineering.

    This class acts as a compiled configuration object. It translates raw string requests
    (e.g., `"power_lag_24h"`) into structured, typed instructions so downstream execution
    functions don't have to parse strings.

    Attributes:
        lags: List of `LagFeature` definitions. Dictates which base columns to
            shift and by how much, enabling safe, time-aware joins for historical data.
        rolling_means: List of `RollingFeature` definitions. Defines moving
            average computations, ensuring they are grouped correctly by time series and
            ensemble member.
        static_features: List of static features. Identifies simple row-wise transformations (like
            windchill) that require no time-shifting or complex aggregations.
        time_features: List of time-based features. Triggers timezone conversions. Energy
            consumption is driven by human behavior, which follows local time (including DST),
            not UTC.
        weather_features: List of raw weather features. Identifies raw weather variables
            requested directly as input features.
        base_features: List of safe input base columns. Identifies base columns
            requested directly as input features.
    """

    lags: list[LagFeature]
    rolling_means: list[RollingFeature]
    static_features: list[StaticFeature]
    time_features: list[TimeFeature]
    weather_features: list[WeatherFeature]
    base_features: list[SafeInputBaseColumn]

    @classmethod
    def from_strings(cls, selected_features: set[str]) -> Self:
        """Parse a list of selected features into a ParsedFeatures object.

        Rationale:
            Parsing upfront allows us to fail fast on invalid requests and cleanly separates the parsing
            logic from the execution logic. It specifically identifies lags on the target variable
            (`power`) and flags them in the `get_leaky_features` method, ensuring the execution phase knows
            exactly which features require lags to be nullified.

            Furthermore, this parser enforces strict architectural guardrails to prevent target leakage
            and index column misuse. For example, requesting the raw target variable 'power' as an input
            feature is forbidden because it would allow downstream models to learn a trivial identity function,
            rendering them useless at inference time when the actual power is unknown. Similarly, 'valid_time'
            is an index column and should not be used directly as a feature; instead, local time features
            should be used to capture behavioral patterns.

        Args:
            selected_features: A set of raw feature name strings requested for engineering. Valid
                values include all TIME_FEATURES, and all StaticFeatures, and feature names like
                'power_lag_24h' and 'temperature_2m_rolling_mean_6h'.

        Returns:
            A ParsedFeatures configuration object containing structured instructions.
        """
        lags: list[LagFeature] = []
        rolling_means: list[RollingFeature] = []
        static_features: list[StaticFeature] = []
        time_features: list[TimeFeature] = []
        weather_features: list[WeatherFeature] = []
        base_features: list[SafeInputBaseColumn] = []

        for feature_name in selected_features:
            if LagFeature.SUFFIX in feature_name and RollingFeature.SUFFIX in feature_name:
                raise ValueError(f"Feature stacking is not supported: {feature_name}")

            elif LagFeature.SUFFIX in feature_name:
                lags.append(LagFeature.from_str(feature_name))

            elif RollingFeature.SUFFIX in feature_name:
                rolling_means.append(RollingFeature.from_str(feature_name))

            elif feature_name in STATIC_FEATURE_REGISTRY:
                static_features.append(cast(StaticFeature, feature_name))

            elif feature_name in get_args(TimeFeature):
                time_features.append(cast(TimeFeature, feature_name))

            elif feature_name == "power":
                # Target leakage prevention guardrail:
                raise ValueError(
                    "The target variable 'power' cannot be requested as an input feature "
                    "in 'selected_features' to prevent target leakage. Use lagged power features "
                    "(e.g., 'power_lag_24h') instead."
                )

            elif feature_name == "valid_time":
                # Index column guardrail:
                raise ValueError(
                    "The index column 'valid_time' cannot be requested as an input feature. "
                    "Use local time features (e.g., 'local_time_of_day_sin') instead."
                )

            elif feature_name in get_args(WeatherFeature):
                weather_features.append(cast(WeatherFeature, feature_name))

            elif feature_name in get_args(SafeInputBaseColumn):
                base_features.append(cast(SafeInputBaseColumn, feature_name))

            else:
                raise ValueError(f"Unrecognised feature name: {feature_name}")

        return cls(
            lags=lags,
            rolling_means=rolling_means,
            static_features=static_features,
            time_features=time_features,
            weather_features=weather_features,
            base_features=base_features,
        )

    def _get_all_lookback_features(self) -> list[LagFeature | RollingFeature]:
        return self.lags + self.rolling_means

    def get_leaky_features(self) -> list[LagFeature | RollingFeature]:
        """List features (like lagged power) that could cause lookahead bias, allowing the pipeline
        to selectively nullify them based on the forecast lead time."""
        return [feature for feature in self._get_all_lookback_features() if feature.is_leaky()]

    def requires_weather_data(self) -> bool:
        """Determine if the requested features require weather (NWP) data.

        This checks:
        1. If any lookback features (lags or rolling means) are based on weather variables.
        2. If any static features (like windchill) require weather variables.
        3. If any raw weather features are requested directly.
        """
        lookback_features_require_weather = any(
            feature.is_weather_feature() for feature in self._get_all_lookback_features()
        )
        static_features_require_weather = any(
            feature in ["windchill"] for feature in self.static_features
        )
        return (
            lookback_features_require_weather
            or static_features_require_weather
            or len(self.weather_features) > 0
        )


def engineer_features(
    selected_features: set[str],
    power_time_series: pt.LazyFrame[PowerTimeSeries],
    time_series_metadata: pt.DataFrame[TimeSeriesMetadata],
    nwp: pt.LazyFrame[NwpInMemory] | None = None,
    power_fcst_init_time: datetime | None = None,
    nwp_init_time: datetime | None = None,
    nwp_publication_delay_hours: int = 6,
) -> pt.LazyFrame[AllFeatures]:
    """Engineer features.

    Args:
        selected_features: Set of features to engineer.
        power_time_series: Input power time series.
        time_series_metadata: Metadata for the time series.
        nwp: NWP weather forecast data in physical units (NwpInMemory). Callers loading
            from Delta Lake should convert with NwpOnDisk.to_nwp_in_memory() first, which
            is lazy and does not trigger a collect.
        power_fcst_init_time: Controls the operating mode of the function.

            **None — bulk training and multi-run backtesting (recommended for most callers):**
            The function is NWP-centric. It produces one row per
            (time_series_id, nwp_init_time, valid_time, ensemble_member), and derives
            power_fcst_init_time = nwp_init_time + nwp_publication_delay_hours per-row.
            Leaky power lags are nullified relative to each row's power_fcst_init_time,
            so the resulting dataset is safe for training. For backtesting, pass the full
            historical dataset here and filter the output to valid_time >= power_fcst_init_time
            before evaluating.

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
    nwp_lf: pt.LazyFrame[NwpInMemory] | None = nwp

    parsed_features = ParsedFeatures.from_strings(selected_features)
    if nwp_lf is None and parsed_features.requires_weather_data():
        raise ValueError("Weather features were requested but no NWP data was provided.")

    processed_nwp = _process_nwp(nwp_lf) if nwp_lf is not None else None
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
    final_lf = _select_output_columns(engineered_lf, selected_features)
    return pt.LazyFrame.from_existing(final_lf).set_model(AllFeatures)


def _join_nwp_bulk_mode(
    power_with_metadata: pl.LazyFrame,
    processed_nwp: pl.LazyFrame | None,
    nwp_publication_delay_hours: int,
) -> pl.LazyFrame:
    """NWP-centric join for bulk training / multi-run backtesting.

    Produces one row per (time_series_id, nwp_init_time, valid_time, ensemble_member) with
    power_fcst_init_time derived per-row as nwp_init_time + nwp_publication_delay_hours.
    """
    if processed_nwp is None:
        result = power_with_metadata.with_columns(
            power_fcst_init_time=pl.col("valid_time"),
            nwp_init_time=pl.lit(None, dtype=UTC_DATETIME_DTYPE),
        )
    else:
        nwp_with_init = processed_nwp.with_columns(
            power_fcst_init_time=pl.col("nwp_init_time")
            + pl.duration(hours=nwp_publication_delay_hours)
        )
        result = nwp_with_init.join(
            power_with_metadata, on=["time_series_id", "valid_time"], how="left"
        )
    return result


def _join_nwp_single_run(
    power_with_metadata: pl.LazyFrame,
    processed_nwp: pl.LazyFrame | None,
    power_fcst_init_time: datetime,
    nwp_init_time: datetime | None,
    nwp_publication_delay_hours: int,
) -> pl.LazyFrame:
    """Power-centric join for single-run production inference or backfilling.

    Stamps a constant power_fcst_init_time across all rows and joins exclusively the one NWP
    run identified by nwp_init_time. If nwp_init_time is None, it is derived as
    power_fcst_init_time - nwp_publication_delay_hours.
    """
    nwp_init_time_val = (
        nwp_init_time
        if nwp_init_time is not None
        else power_fcst_init_time - timedelta(hours=nwp_publication_delay_hours)
    )
    power_with_init = power_with_metadata.with_columns(
        power_fcst_init_time=pl.lit(power_fcst_init_time),
        nwp_init_time=pl.lit(nwp_init_time_val),
    )
    if processed_nwp is None:
        result = power_with_init
    else:
        result = power_with_init.join(
            processed_nwp, on=["time_series_id", "valid_time", "nwp_init_time"], how="left"
        )
    return result


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


def _upsample_nwp_to_half_hourly(nwp_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Upsample NWP data to 30-minute resolution.

    Assumes 'init_time' has already been renamed to 'nwp_init_time'. Continuous weather
    variables are linearly interpolated within each group; categorical variables are
    forward-filled within each group. All other columns (e.g. nwp_init_time,
    ensemble_member) are used as group-by keys.

    The implementation stays fully lazy: a 30-min time grid is generated per group via
    datetime_ranges + explode, then the original NWP values are left-joined back in, and
    interpolate/forward_fill are applied with over() to stay within group boundaries.

    Leading-null propagation: Polars' interpolate() fills interior nulls but leaves leading
    nulls (before the first non-null value in a group) as null. Some ECMWF ENS variables
    (precipitation, radiation fluxes) are null at lead time 0 by convention. After upsampling
    from native 3-hourly steps, all interpolated 30-min rows before the first non-null step
    remain null — typically a 3-hour window per NWP run. Callers and downstream models should
    treat these as genuinely missing values, not as a data quality issue.
    """
    schema_names = nwp_lf.collect_schema().names()
    all_weather_vars = NwpInMemory.all_weather_var_names()
    group_cols = [
        col for col in schema_names if col != "valid_time" and col not in all_weather_vars
    ]
    continuous_cols = [col for col in schema_names if col in NwpInMemory.continuous_var_names()]
    categorical_cols = [col for col in schema_names if col in NwpInMemory.categorical_var_names]

    # Build 30-min time grid per group: aggregate min/max valid_time per group,
    # expand each row into a list of half-hourly datetimes, then explode to rows.
    time_grid = (
        nwp_lf.group_by(group_cols)
        .agg(
            pl.col("valid_time").min().alias("_start"),
            pl.col("valid_time").max().alias("_end"),
        )
        .with_columns(
            pl.datetime_ranges(
                start=pl.col("_start"),
                end=pl.col("_end"),
                interval="30m",
            ).alias("valid_time")
        )
        .drop("_start", "_end")
        .explode("valid_time")
    )

    # Left-join original NWP onto the grid; new 30-min rows come in as nulls.
    upsampled = time_grid.join(nwp_lf, on=[*group_cols, "valid_time"], how="left").sort(
        [*group_cols, "valid_time"]
    )

    # Fill nulls within each group, never crossing group boundaries.
    # order_by="valid_time" ensures correct temporal ordering within each group window.
    if continuous_cols:
        upsampled = upsampled.with_columns(
            pl.col(col).interpolate().over(group_cols, order_by="valid_time")
            for col in continuous_cols
        )
    if categorical_cols:
        upsampled = upsampled.with_columns(
            pl.col(col).forward_fill().over(group_cols, order_by="valid_time")
            for col in categorical_cols
        )

    return upsampled


def _process_nwp(nwp_lf: pt.LazyFrame[NwpInMemory]) -> pl.LazyFrame:
    """Rename 'init_time', upsample to 30-minute resolution, and calculate NWP lead time.

    Expects data already in physical units (NwpInMemory). Callers are responsible for
    converting NwpOnDisk → NwpInMemory before passing data here; that conversion is lazy
    and can be done with NwpOnDisk.to_nwp_in_memory().
    """
    renamed = nwp_lf.rename({"init_time": "nwp_init_time"})
    upsampled = _upsample_nwp_to_half_hourly(renamed)
    return upsampled.with_columns(
        nwp_lead_time_hours=(
            (pl.col("valid_time") - pl.col("nwp_init_time")).dt.total_seconds() / 3600
        ).cast(pl.Float32)
    )


def _build_historical_weather(processed_nwp: pl.LazyFrame) -> pl.LazyFrame:
    """Builds the historical weather frame used by weather lag features.

    Uses the control member (ensemble_member == 0) and selects the freshest forecast run
    (shortest lead time) for each (time_series_id, valid_time).
    """
    return (
        processed_nwp.filter(pl.col("ensemble_member") == 0)
        .drop("ensemble_member")
        .group_by(["time_series_id", "valid_time"])
        .agg(pl.all().sort_by("nwp_lead_time_hours").first())
        .sort(["time_series_id", "valid_time"])
    )


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


def _apply_rolling_mean_feature(
    lf: pl.LazyFrame, base_col: WeatherFeature, window_hours: int
) -> pl.LazyFrame:
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
