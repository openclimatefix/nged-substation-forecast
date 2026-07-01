"""Feature name parsing and typed feature descriptors.

Translates raw string requests (e.g. ``"power_lag_24h"``) into structured, typed objects so the
rest of the pipeline never parses strings. ``ParsedFeatures.from_strings`` is the entry point;
it also enforces architectural guardrails (no raw target, no index columns as features).
"""

import re
from abc import abstractmethod
from dataclasses import dataclass
from typing import Annotated, ClassVar, Final, Literal, Self, cast, get_args

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from contracts.ml_schemas import SafeInputBaseColumn, TimeFeature
from contracts.weather_schemas import WeatherFeature

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
