"""Feature name parsing and typed feature descriptors.

Translates raw string requests (e.g. ``"power_lag_24h"``) into structured, typed objects so the
rest of the pipeline never parses strings. ``ParsedFeatures.from_strings`` is the entry point; it
also enforces architectural guardrails (no raw target, no index columns as features).
"""

import re
from abc import abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from typing import Annotated, ClassVar, Final, Literal, Self, cast, get_args

import polars as pl
from contracts.ml_schemas import SafeInputBaseColumn, TimeFeature
from contracts.weather_schemas import WeatherFeature
from pydantic import BaseModel, ConfigDict, Field

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

    The class's main job is to parse strings like 'power_lag_24h'.
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
        """Whether this feature must be nullified to avoid lookahead bias.

        True if the feature leaks information into the ML model's inputs that would not be
        available at inference time.
        """

    def is_weather_feature(self) -> bool:
        return self.base_col in get_args(WeatherFeature)


class LagFeature(BaseLookbackFeature):
    """Represents a parsed lag feature."""

    SUFFIX: ClassVar[str] = "lag"
    base_col: WeatherFeature | Literal["power"]

    def is_leaky(self) -> bool:
        """Power lags are always leaky.

        Observed power may not exist at forecast-issue time if the lagged observation post-dates
        power_fcst_init_time. Per-row nullification is handled downstream by _nullify_leaky_lags.
        """
        return self.base_col == "power"


class RollingFeature(BaseLookbackFeature):
    """Represents a parsed rolling mean feature.

    Computing the rolling mean of 'power' is currently forbidden, because a rolling window over
    observed power would reach past the forecast-issue time and leak the target.
    """

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
        """Weather rolling means are never leaky.

        NWP forecasts are available for future valid_times, so a weather rolling mean (e.g. the
        mean temperature over the 6h window ending at valid_time) is always known at inference
        time.
        """
        return False


@dataclass
class ParsedFeatures:
    """Compiled configuration object for feature engineering.

    ``ParsedFeatures`` translates raw string requests, such as `"power_lag_24h"`, into structured,
    typed instructions, so that no downstream execution function has to parse a string.

    Attributes:
        lags: List of `LagFeature` definitions. Dictates which base columns to
            shift and by how much, enabling safe, time-aware joins for historical data.
        rolling_means: List of `RollingFeature` definitions. Defines moving
            average computations, ensuring they are grouped correctly by time series and
            ensemble member.
        static_features: List of static features. Identifies simple row-wise transformations (like
            windchill) that require no time-shifting or complex aggregations.
        time_features: List of time-based features. Triggers timezone conversions. Energy
            consumption is driven by human behaviour, which follows local time (including daylight
            saving time), not UTC.
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
        """Parse a set of selected feature names into a ``ParsedFeatures`` object.

        Rationale:
            Parsing upfront allows us to fail fast on invalid requests and cleanly separates the
            parsing logic from the execution logic. Parsing also separates out the lags on the
            target variable (`power`), which `get_leaky_features` later selects. The execution phase
            therefore knows exactly which features require lags to be nullified.

            Furthermore, this parser enforces strict architectural guardrails to prevent target
            leakage and index column misuse. For example, requesting the raw target variable 'power'
            as an input feature is forbidden, because it would let a downstream model learn a
            trivial identity function. That identity function is useless at inference time, when the
            actual power is unknown. Similarly, 'valid_time' is an index column and should not be
            used directly as a feature. The local time features capture the behavioural patterns a
            caller reaching for 'valid_time' is after.

        Args:
            selected_features: A set of raw feature name strings requested for engineering. Six
                kinds of name are accepted: every member of `contracts.ml_schemas.TimeFeature`;
                every key of `STATIC_FEATURE_REGISTRY`, which today holds `windchill` alone; every
                member of `contracts.weather_schemas.WeatherFeature`; every member of
                `contracts.ml_schemas.SafeInputBaseColumn`; a lag name such as `power_lag_24h`;
                and a rolling-mean name such as `temperature_2m_rolling_mean_6h`. Every other
                string raises `ValueError`, including the two guarded names above.

        Returns:
            A `ParsedFeatures` configuration object containing structured instructions, with one
            list per accepted kind of name.
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

            if LagFeature.SUFFIX in feature_name:
                lags.append(LagFeature.from_str(feature_name))

            elif RollingFeature.SUFFIX in feature_name:
                rolling_means.append(RollingFeature.from_str(feature_name))

            elif feature_name in STATIC_FEATURE_REGISTRY:
                static_features.append(feature_name)

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
        """List the features that could cause lookahead bias, such as lagged power.

        The pipeline uses this list to nullify those features selectively, based on the forecast
        lead time.
        """
        return [feature for feature in self._get_all_lookback_features() if feature.is_leaky()]

    def max_power_lag(self) -> timedelta:
        """The longest power lag these features request, or zero when none of them is a power lag.

        Sizes ``load_engineering_inputs``'s ``power_lookback``: a caller needs power history
        reaching back at least the returned duration before its window to keep every requested
        power lag non-null near the window's start.
        """
        return timedelta(
            hours=max((lag.hours for lag in self.lags if lag.base_col == "power"), default=0)
        )

    def requires_weather_data(self) -> bool:
        """Determine if the requested features require weather (NWP) data.

        Any one of three conditions makes weather data necessary:

        1. If any lookback features (lags or rolling means) are based on weather variables.
        2. If any static features (like windchill) require weather variables.
        3. If any raw weather features are requested directly.
        """
        lookback_features_require_weather = any(
            feature.is_weather_feature() for feature in self._get_all_lookback_features()
        )
        static_features_require_weather = any(
            feature == "windchill" for feature in self.static_features
        )
        return (
            lookback_features_require_weather
            or static_features_require_weather
            or len(self.weather_features) > 0
        )
