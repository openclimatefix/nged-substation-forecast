"""Declarative feature engineering pipeline for time-series forecasting.

Architecture/Flow:
    `engineer_features` is the main orchestrator of this module. It takes raw string requests,
    compiles them into structured instructions using `ParsedFeatures.from_selected_features`, joins the necessary
    base data (power, weather, metadata), and then executes the instructions via
    `_apply_post_join_features`.

Nullify Leaky Lags Rationale:
    `nullify_leaky_lags` is called at the end of the pipeline to enforce physical forecasting
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
from contracts.ml_schemas import AllFeatures
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from contracts.weather_schemas import NwpOnDisk
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator  # noqa: F401

# Valid weather features (from NwpInMemory contract)
WeatherFeature = Literal[
    "temperature_2m",
    "dew_point_temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_speed_100m",
    "wind_direction_100m",
    "pressure_surface",
    "pressure_reduced_to_mean_sea_level",
    "geopotential_height_500hpa",
    "downward_long_wave_radiation_flux_surface",
    "downward_short_wave_radiation_flux_surface",
    "precipitation_surface",
    "categorical_precipitation_type_surface",
]


# Valid time features (from AllFeatures contract)
TimeFeature = Literal[
    "local_time_of_day_sin",
    "local_time_of_day_cos",
    "local_time_of_year_sin",
    "local_time_of_year_cos",
    "local_day_of_week_sin",
    "local_day_of_week_cos",
    "local_day_of_week",
    "local_utc_offset",
]


# Base columns that are safe to use as direct input features for the ML model.
# These do not cause target leakage or lookahead bias.
SafeInputBaseColumn = Literal[
    "time_series_id",
    "time_series_type",
    "nwp_lead_time_hours",
    "ensemble_member",
    "power_fcst_init_time",
    "nwp_init_time",
]


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
    string_repr: str  # String representation, e.g. 'power_lag_24h'

    @classmethod
    def from_str(cls, value: str) -> Self:
        """Parse and validate a feature name string into an instance."""
        pattern = re.compile(rf"^(.*)_{cls.SUFFIX}_(\d+)h$")
        match = pattern.match(value)
        if not match:
            raise ValueError(f"Invalid {cls.SUFFIX} feature name format: {value}")
        base_col, hours_str = match.groups()
        return cls(base_col=base_col, hours=hours_str, string_repr=value)  # ty: ignore[invalid-argument-type]

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
        """A power lag is only known at inference time if the lagged time is in the past
        relative to the init time."""
        return self.base_col == "power"


class RollingFeature(BaseLookbackFeature):
    """Represents a parsed rolling mean feature.

    Note that computing the rolling mean of 'power' is currently forbidden to prevent lookahead
    bias."""

    # TODO: Implement "Latest Available Rolling Mean anchored to T_init" to allow non-leaky
    # rolling power features. Maybe also give the model other stats about recent observed
    # power over some time window, like min, max, std, mean.

    SUFFIX: ClassVar[str] = "rolling_mean"

    def is_leaky(self) -> bool:
        """Weather forecasts (NWP) are available for the future, so a weather lag (e.g., temperature
        6 hours ago relative to the forecast target time) is always known at inference time."""
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
            selected_features: A set of raw feature name strings requested for engineering. Valid Includes all
            TIME_FEATURES, and all StaticFeatures, and feature names like 'power_lag_24h' and
            'temperature_2m_rolling_mean_6h'.

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
            [feature.is_weather_feature() for feature in self._get_all_lookback_features()]
        )
        static_features_require_weather = any(
            [feature in ["windchill"] for feature in self.static_features]
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
    nwp: pt.LazyFrame[NwpOnDisk] | None = None,
    power_fcst_init_time: datetime | None = None,
    nwp_publication_delay_hours: int = 6,
) -> pt.LazyFrame[AllFeatures]:
    """Engineer features.

    Args:
        selected_features: Set of features to engineer.
        power_time_series: Input power time series.
        time_series_metadata: Metadata for the time series.
        nwp: NWP weather forecast data.
        power_fcst_init_time: Controls the operating mode of the function.

            **None — bulk training and multi-run backtesting (recommended for most callers):**
            The function is NWP-centric. It produces one row per
            (time_series_id, nwp_init_time, valid_time, ensemble_member), and derives
            power_fcst_init_time = nwp_init_time + nwp_publication_delay_hours per-row.
            Leaky power lags are nullified relative to each row's power_fcst_init_time,
            so the resulting dataset is safe for training. For backtesting, pass the full
            historical dataset here and filter the output to valid_time >= power_fcst_init_time
            before evaluating.

            **Single datetime — real-time production inference only:**
            A constant power_fcst_init_time is stamped onto every row, and the NWP join
            matches exclusively the one NWP run whose nwp_init_time equals
            power_fcst_init_time - nwp_publication_delay_hours. This branch is only
            appropriate when power_time_series contains observations for a single forecast
            run. Passing a multi-run dataset (e.g., a full year of data) will silently
            produce null weather features for every row except those belonging to the
            matching NWP run.

        nwp_publication_delay_hours: The delay in hours between the initialization time
            of the NWP forecast and when it becomes publicly available for use in power
            forecasting.
    """
    power_lf = pl.LazyFrame._from_pyldf(power_time_series._ldf).rename({"time": "valid_time"})
    metadata_lf = pl.LazyFrame._from_pyldf(time_series_metadata.lazy()._ldf)
    nwp_lf = pl.LazyFrame._from_pyldf(nwp._ldf) if nwp is not None else None

    parsed_features = ParsedFeatures.from_strings(selected_features)
    if nwp_lf is None and parsed_features.requires_weather_data():
        raise ValueError("Weather features were requested but no NWP data was provided.")

    processed_nwp, historical_weather = _process_nwp_data(nwp_lf, parsed_features)
    power_with_metadata = power_lf.join(metadata_lf, on="time_series_id", how="left")

    if power_fcst_init_time is None:
        raw_data = _join_nwp_bulk_mode(
            power_with_metadata, processed_nwp, nwp_publication_delay_hours
        )
    else:
        raw_data = _join_nwp_single_run(
            power_with_metadata, processed_nwp, power_fcst_init_time, nwp_publication_delay_hours
        )

    engineered_lf = _apply_post_join_features(
        raw_data,
        parsed_features,
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
            nwp_lead_time_hours=pl.lit(None, dtype=pl.Float32),
            ensemble_member=pl.lit(None, dtype=pl.UInt8),
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
    nwp_publication_delay_hours: int,
) -> pl.LazyFrame:
    """Power-centric join for single-run production inference.

    Stamps a constant power_fcst_init_time across all rows and joins exclusively the one NWP
    run whose nwp_init_time equals power_fcst_init_time - nwp_publication_delay_hours.
    """
    nwp_init_time_val = power_fcst_init_time - timedelta(hours=nwp_publication_delay_hours)
    power_with_init = power_with_metadata.with_columns(
        power_fcst_init_time=pl.lit(power_fcst_init_time),
        nwp_init_time=pl.lit(nwp_init_time_val),
    )
    if processed_nwp is None:
        result = power_with_init.with_columns(
            nwp_lead_time_hours=pl.lit(None, dtype=pl.Float32),
            ensemble_member=pl.lit(None, dtype=pl.UInt8),
        )
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


def _process_nwp_data(
    nwp_lf: pl.LazyFrame | None,
    parsed_features: ParsedFeatures,
) -> tuple[pl.LazyFrame | None, pl.LazyFrame | None]:
    """Process NWP data and prepare historical weather for lag features.

    Unlike the previous implementation, we do NOT pre-aggregate to the "best" forecast run.
    Instead, we keep all available forecast runs (init_times) to allow the caller to perform
    an explicit temporal alignment join. This naturally prevents row multiplication and lookahead bias.
    """
    if nwp_lf is None:
        return None, None

    processed_nwp = calculate_nwp_lead_time(nwp_lf)

    weather_lag_requested = any(lag_feat.base_col != "power" for lag_feat in parsed_features.lags)

    historical_weather = None
    if weather_lag_requested:
        # For weather lags, we use the control member (ensemble_member == 0)
        # and select the "freshest" forecast run (shortest lead time) for each valid_time.
        historical_weather = (
            processed_nwp.filter(pl.col("ensemble_member") == 0)
            .drop("ensemble_member")
            .group_by(["time_series_id", "valid_time"])
            .agg(pl.all().sort_by("nwp_lead_time_hours").first())
            .sort(["time_series_id", "valid_time"])
        )

    return processed_nwp, historical_weather


def _apply_post_join_features(
    raw_data: pl.LazyFrame,
    parsed_features: ParsedFeatures,
    processed_nwp: pl.LazyFrame | None = None,
    historical_weather: pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    """Applies requested features dynamically based on parsed feature configurations."""
    engineered_lf = raw_data

    # Static and Local Time
    if parsed_features.time_features:
        engineered_lf = apply_local_time_features(engineered_lf)

    if parsed_features.static_features:
        exprs = [STATIC_FEATURE_REGISTRY[f] for f in parsed_features.static_features]
        engineered_lf = engineered_lf.with_columns(exprs)

    # Lags
    for lag_feat in parsed_features.lags:
        if lag_feat.base_col == "power":
            engineered_lf = apply_lag_feature(
                engineered_lf,
                source_lf=engineered_lf,
                lag_feature=lag_feat,
            )
        else:
            if processed_nwp is None:
                raise ValueError(
                    "processed_nwp cannot be None when applying a weather lag feature."
                )
            # Weather lag: use the dual-strategy
            engineered_lf = apply_lag_feature(
                engineered_lf,
                source_lf=processed_nwp,
                lag_feature=lag_feat,
                historical_weather_lf=historical_weather,
            )

    # Rolling Means
    for rolling_feat in parsed_features.rolling_means:
        if rolling_feat.base_col in engineered_lf.collect_schema().names():
            engineered_lf = apply_rolling_mean_feature(
                engineered_lf, rolling_feat.base_col, rolling_feat.hours
            )

    # Nullify leaky lags
    leaky_features = parsed_features.get_leaky_features()
    if leaky_features and "power_fcst_init_time" in engineered_lf.collect_schema().names():
        engineered_lf = nullify_leaky_lags(engineered_lf, leaky_features)

    return engineered_lf


def apply_lag_feature(
    target_lf: pl.LazyFrame,
    source_lf: pl.LazyFrame,
    lag_feature: LagFeature,
    historical_weather_lf: pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    """Applies a lag feature using a time-aware lazy self-join.

    For power lags, we use an exact join on valid_time - lag_hours.
    For weather lags, we implement a dual-strategy:
      - If target_time > power_fcst_init_time (future relative to power forecast run time),
        we use the exact same NWP run (nwp_init_time) and ensemble member as the weather used for valid_time.
      - If target_time <= power_fcst_init_time (past relative to power forecast run time),
        we use the "freshest" NWP run (control member, ensemble 0) for that target time.
    """
    lag_hours = lag_feature.hours
    base_col = lag_feature.base_col

    if base_col == "power":
        # Power lag: exact join on valid_time - lag_hours
        lf_with_target_time = target_lf.with_columns(
            target_time=pl.col("valid_time") - pl.duration(hours=lag_hours)
        )

        join_keys = ["time_series_id"]
        if (
            "ensemble_member" in target_lf.collect_schema().names()
            and "ensemble_member" in source_lf.collect_schema().names()
        ):
            join_keys.append("ensemble_member")

        right_lf = source_lf.select(
            *join_keys,
            pl.col("valid_time"),
            pl.col(base_col).alias(lag_feature.string_repr),
        )

        return lf_with_target_time.join(
            right_lf,
            left_on=join_keys + ["target_time"],
            right_on=join_keys + ["valid_time"],
            how="left",
        ).drop("target_time")

    else:
        # Weather lag: dual-strategy
        lf_with_target_time = target_lf.with_columns(
            target_time=pl.col("valid_time") - pl.duration(hours=lag_hours)
        )

        # Join 1: Same-Run Join (for target_time > power_fcst_init_time)
        join_keys_same = ["time_series_id", "nwp_init_time"]
        if (
            "ensemble_member" in target_lf.collect_schema().names()
            and "ensemble_member" in source_lf.collect_schema().names()
        ):
            join_keys_same.append("ensemble_member")

        right_same_lf = source_lf.select(
            *join_keys_same,
            pl.col("valid_time").alias("target_time"),
            pl.col(base_col).alias(f"{lag_feature.string_repr}_same_run"),
        )

        lf_joined = lf_with_target_time.join(
            right_same_lf,
            on=join_keys_same + ["target_time"],
            how="left",
        )

        # Join 2: Freshest-Run Join (for target_time <= power_fcst_init_time)
        if historical_weather_lf is not None:
            right_freshest_lf = historical_weather_lf.select(
                "time_series_id",
                pl.col("valid_time").alias("target_time"),
                pl.col(base_col).alias(f"{lag_feature.string_repr}_freshest_run"),
            )

            lf_joined = lf_joined.join(
                right_freshest_lf,
                on=["time_series_id", "target_time"],
                how="left",
            )
        else:
            lf_joined = lf_joined.with_columns(
                **{f"{lag_feature.string_repr}_freshest_run": pl.lit(None, dtype=pl.Float32)}
            )

        # Combine using the dual-strategy
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


def calculate_nwp_lead_time(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Calculates the NWP lead time in hours if 'init_time' is present in the schema.

    Also renames 'init_time' to 'nwp_init_time' to distinguish it from power forecast init time.
    """
    if "init_time" in lf.collect_schema().names():
        return lf.with_columns(
            nwp_lead_time_hours=(
                (pl.col("valid_time") - pl.col("init_time")).dt.total_seconds() / 3600
            ).cast(pl.Float32)
        ).rename({"init_time": "nwp_init_time"})
    return lf


def nullify_leaky_lags(
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
    # Calculate power_lead_time_hours dynamically
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

    # Drop power_lead_time_hours to keep the schema clean
    return lf.drop("power_lead_time_hours")


def apply_rolling_mean_feature(
    lf: pl.LazyFrame, base_col: WeatherFeature, window_hours: int
) -> pl.LazyFrame:
    """
    Applies a rolling mean feature using time-aware rolling aggregations.

    Why dynamically check for ensemble_member? If we group by time_series_id alone
    when ensemble members are present, the rolling window will mix data across different
    ensemble members, causing data contamination. Grouping by ensemble_member (if present)
    prevents this.

    Args:
        lf: The LazyFrame to apply the rolling mean to.
        base_col: The column to calculate the rolling mean for.
        window_hours: The window size in hours.

    Returns:
        A LazyFrame with the new rolling mean column.
    """
    schema_names = lf.collect_schema().names()
    group_by_cols = ["time_series_id"]
    if "ensemble_member" in schema_names:
        group_by_cols.append("ensemble_member")

    rolled = lf.rolling(
        index_column="valid_time", period=f"{window_hours}h", group_by=group_by_cols
    ).agg(pl.col(base_col).mean().alias(f"{base_col}_rolling_mean_{window_hours}h"))

    # Join the result back to the original lf
    join_keys = group_by_cols + ["valid_time"]
    return lf.join(rolled, on=join_keys, how="left")


def apply_local_time_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Applies local time features (time of day, day of week, time of year) to the LazyFrame.

    Why local time? Energy consumption patterns are driven by human behavior, which follows
    local time (including Daylight Saving Time), not UTC. A 9 AM peak in winter (UTC) is
    different from a 9 AM peak in summer (UTC+1).

    Args:
        lf: The LazyFrame containing a 'valid_time' column in UTC.

    Returns:
        A LazyFrame with new local time features.
    """
    # Convert valid_time to Europe/London timezone
    # We first ensure it's treated as UTC, then convert to the target timezone
    lf = lf.with_columns(
        local_time=pl.col("valid_time")
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone("Europe/London")
    )

    # Calculate local UTC offset in hours
    # base_utc_offset + dst_offset gives the total offset from UTC
    lf = lf.with_columns(
        local_utc_offset=(
            pl.col("local_time").dt.base_utc_offset() + pl.col("local_time").dt.dst_offset()
        ).dt.total_seconds()
        / 3600
    )

    # Calculate local time of day (0-24)
    local_hour_float = pl.col("local_time").dt.hour() + pl.col("local_time").dt.minute() / 60.0

    # Calculate local time of year (0-1)
    # Using 366 to safely handle leap years without complex logic
    local_year_fraction = pl.col("local_time").dt.ordinal_day() / 366.0

    # Calculate local day of week (1-7)
    local_weekday = pl.col("local_time").dt.weekday()

    # TODO: This map could just be a list, surely?
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

    # Drop the temporary local_time column
    return lf.drop("local_time")
