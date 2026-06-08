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
from dataclasses import dataclass
from typing import Final, Literal, NamedTuple, Self, cast, get_args

import patito as pt
import polars as pl
from contracts.ml_schemas import AllFeatures
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from contracts.weather_schemas import NwpOnDisk

# Valid base columns that can be lagged or rolled (from AllFeatures and NwpInMemory contracts)
BaseColumn = Literal[
    "power",
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

StaticFeature = Literal["windchill"]

TIME_FEATURES: Final[set[TimeFeature]] = set(get_args(TimeFeature))

# Static features registry
# These are basic features that don't require parameterization.
STATIC_FEATURE_REGISTRY: dict[StaticFeature, pl.Expr] = {
    "windchill": (
        13.12
        + 0.6215 * pl.col("temperature_2m")
        - 11.37 * ((pl.col("wind_speed_10m") * 3.6) ** 0.16)
        + 0.3965 * pl.col("temperature_2m") * ((pl.col("wind_speed_10m") * 3.6) ** 0.16)
    ).alias("windchill"),
}


class LagFeature(NamedTuple):
    base_col: BaseColumn
    lag_hours: int


class RollingFeature(NamedTuple):
    base_col: BaseColumn
    window_hours: int


@dataclass
class ParsedFeatures:
    """Compiled configuration object for feature engineering.

    This class acts as a compiled configuration object. It translates raw string requests
    (e.g., `"power_lag_24h"`) into structured, typed instructions so downstream execution
    functions don't have to parse strings.

    Attributes:
        lags: Maps feature names to `LagFeature` definitions. Dictates which base columns to
            shift and by how much, enabling safe, time-aware joins for historical data.
        rolling_means: Maps feature names to `RollingFeature` definitions. Defines moving
            average computations, ensuring they are grouped correctly by time series and
            ensemble member.
        static: List of static features. Identifies simple row-wise transformations (like
            windchill) that require no time-shifting or complex aggregations.
        local_time: List of time-based features. Triggers timezone conversions. Energy
            consumption is driven by human behavior, which follows local time (including DST),
            not UTC.
        leaky_lags: Maps target-derived lag features to their lag hours. Explicitly tracks
            features (like lagged power) that could cause lookahead bias, allowing the pipeline
            to selectively nullify them based on the forecast lead time.
    """

    lags: dict[str, LagFeature]
    rolling_means: dict[str, RollingFeature]
    static: list[StaticFeature]
    local_time: list[TimeFeature]
    leaky_lags: dict[str, int]

    @classmethod
    def from_selected_features(cls, selected_features: set[str]) -> Self:
        """Parse a list of selected features into a ParsedFeatures object.

        Rationale:
            Parsing upfront allows us to fail fast on invalid requests and cleanly separates the parsing
            logic from the execution logic. It specifically identifies lags on the target variable
            (`power`) and flags them in the `leaky_lags` dictionary, ensuring the execution phase knows
            exactly which features require lags to be nullified.

        Args:
            selected_features: A set of raw feature name strings requested for engineering. Valid Includes all
            TIME_FEATURES, and all StaticFeatures, and feature names like 'power_lag_24h'.

        Returns:
            A ParsedFeatures configuration object containing structured instructions.
        """
        lags: dict[str, LagFeature] = {}
        rolling_means: dict[str, RollingFeature] = {}
        static: list[StaticFeature] = []
        local_time: list[TimeFeature] = []
        leaky_lags: dict[str, int] = {}

        for feature_name in selected_features:
            if "lag" in feature_name and "rolling_mean" in feature_name:
                raise ValueError(f"Feature stacking is not supported: {feature_name}")

            # Extract lagged features like 'power_lag_24h`
            lag_match = re.match(r"^(.*)_lag_(\d+)h$", feature_name)
            if lag_match:
                base_col, lag_hours_str = lag_match.groups()
                lag_hours = int(lag_hours_str)
                lags[feature_name] = LagFeature(
                    base_col=cast(BaseColumn, base_col), lag_hours=lag_hours
                )
                if base_col == "power":
                    leaky_lags[feature_name] = lag_hours
                continue

            rolling_match = re.match(r"^(.*)_rolling_mean_(\d+)h$", feature_name)
            if rolling_match:
                base_col, window_hours_str = rolling_match.groups()
                window_hours = int(window_hours_str)
                rolling_means[feature_name] = RollingFeature(
                    base_col=cast(BaseColumn, base_col), window_hours=window_hours
                )
                if base_col == "power":
                    leaky_lags[feature_name] = 0
                continue

            if feature_name in STATIC_FEATURE_REGISTRY:
                static.append(cast(StaticFeature, feature_name))
            elif feature_name in TIME_FEATURES:
                local_time.append(cast(TimeFeature, feature_name))
            else:
                raise ValueError(f"Unrecognised feature name: {feature_name}")

        return cls(
            lags=lags,
            rolling_means=rolling_means,
            static=static,
            local_time=local_time,
            leaky_lags=leaky_lags,
        )


def engineer_features(
    selected_features: set[str],
    power_time_series: pt.LazyFrame[PowerTimeSeries],
    time_series_metadata: pt.DataFrame[TimeSeriesMetadata],
    nwp: pt.LazyFrame[NwpOnDisk] | None = None,
) -> pt.LazyFrame[AllFeatures]:
    """Engineer features."""
    # Convert Patito to Polars immediately for cleaner downstream code
    power_lf = pl.LazyFrame._from_pyldf(power_time_series._ldf).rename({"time": "valid_time"})
    metadata_lf = pl.LazyFrame._from_pyldf(time_series_metadata.lazy()._ldf)
    nwp_lf = pl.LazyFrame._from_pyldf(nwp._ldf) if nwp is not None else None

    parsed_features = ParsedFeatures.from_selected_features(selected_features)

    # Detect if weather lags are requested
    weather_lag_requested = False
    if nwp_lf is not None:
        nwp_cols = nwp_lf.collect_schema().names()
        for lag_feat in parsed_features.lags.values():
            if lag_feat.base_col in nwp_cols and lag_feat.base_col != "power":
                weather_lag_requested = True
                break

    # Process NWP for current forecast
    processed_nwp: pl.LazyFrame | None = None
    if nwp_lf is not None:
        processed_nwp = calculate_lead_time(nwp_lf)

    # Compute historical_weather for lags
    historical_weather: pl.LazyFrame | None = None
    if processed_nwp is not None and weather_lag_requested:
        historical_weather = (
            processed_nwp.filter(pl.col("ensemble_member") == 0)
            .drop("ensemble_member")
            .group_by(["time_series_id", "valid_time"])
            .agg(pl.all().sort_by("lead_time_hours").first())
            .sort(["time_series_id", "valid_time"])
        )

    # Join Data
    raw_data = power_lf.join(metadata_lf, on="time_series_id", how="left")
    if processed_nwp is not None:
        raw_data = processed_nwp.join(raw_data, on=["time_series_id", "valid_time"], how="left")
        # Ensure unique rows if multiple init_times exist in the provided nwp
        raw_data = raw_data.unique(["time_series_id", "valid_time", "ensemble_member"])

    # Apply Features
    engineered_lf = _apply_post_join_features(raw_data, parsed_features, historical_weather)

    # Schema Assertion and Selection
    available_columns = engineered_lf.collect_schema().names()
    missing_cols = set(selected_features) - set(available_columns)
    if missing_cols:
        if "lead_time_hours" in missing_cols and "lead_time_hours" not in selected_features:
            missing_cols.remove("lead_time_hours")
        if missing_cols:
            raise ValueError(f"Feature engineering failed to create or find: {missing_cols}")

    base_cols = ["valid_time", "time_series_id", "time_series_type", "power"]
    if "lead_time_hours" in engineered_lf.collect_schema().names():
        base_cols.append("lead_time_hours")
    if "ensemble_member" in engineered_lf.collect_schema().names():
        base_cols.append("ensemble_member")

    cols_to_select = list(set(base_cols + list(selected_features)))
    final_lf = engineered_lf.select(cols_to_select)

    return pt.LazyFrame.from_existing(final_lf).set_model(AllFeatures)


def _apply_post_join_features(
    raw_data: pl.LazyFrame,
    parsed_features: ParsedFeatures,
    historical_weather: pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    """Applies requested features dynamically based on parsed feature configurations."""
    engineered_lf = raw_data

    # Static and Local Time
    if parsed_features.local_time:
        engineered_lf = apply_local_time_features(engineered_lf)

    if parsed_features.static:
        exprs = [STATIC_FEATURE_REGISTRY[f] for f in parsed_features.static]
        engineered_lf = engineered_lf.with_columns(exprs)

    # Lags
    for feature_name, lag_feat in parsed_features.lags.items():
        source_lf = engineered_lf
        if historical_weather is not None and lag_feat.base_col != "power":
            if lag_feat.base_col in historical_weather.collect_schema().names():
                source_lf = historical_weather

        if lag_feat.base_col in source_lf.collect_schema().names():
            engineered_lf = apply_lag_feature(
                engineered_lf, source_lf, lag_feat.base_col, lag_feat.lag_hours
            )

    # Rolling Means
    for feature_name, rolling_feat in parsed_features.rolling_means.items():
        if rolling_feat.base_col in engineered_lf.collect_schema().names():
            engineered_lf = apply_rolling_mean_feature(
                engineered_lf, rolling_feat.base_col, rolling_feat.window_hours
            )

    # Nullify leaky lags
    if parsed_features.leaky_lags and "lead_time_hours" in engineered_lf.collect_schema().names():
        engineered_lf = nullify_leaky_lags(engineered_lf, parsed_features.leaky_lags)

    return engineered_lf


def apply_lag_feature(
    target_lf: pl.LazyFrame, source_lf: pl.LazyFrame, base_col: BaseColumn, lag_hours: int
) -> pl.LazyFrame:
    """
    Applies a lag feature using a time-aware lazy self-join.

    Why a lazy join instead of shift()? Using shift() assumes the data is perfectly
    contiguous without missing rows. If a row is missing, shift() will pull data from
    the wrong time period, silently corrupting the feature. A time-aware join guarantees
    mathematical safety: if the lagged time doesn't exist, it correctly returns null.

    Args:
        target_lf: The LazyFrame to apply the lag to.
        source_lf: The LazyFrame containing the historical data to pull from.
        base_col: The column to lag (e.g., 'power').
        lag_hours: The number of hours to lag.

    Returns:
        A LazyFrame with the new lagged column.
    """
    # Create a dataframe with the target time we want to join against
    lf_with_target_time = target_lf.with_columns(
        target_time=pl.col("valid_time") - pl.duration(hours=lag_hours)
    )

    target_schema = target_lf.collect_schema().names()
    source_schema = source_lf.collect_schema().names()
    join_keys = ["time_series_id"]
    if "ensemble_member" in target_schema and "ensemble_member" in source_schema:
        join_keys.append("ensemble_member")

    # The right side of the join only needs the keys and the base column
    # We dynamically check if base_col exists in the schema to avoid errors
    if base_col not in source_lf.collect_schema().names():
        raise ValueError(f"Base column '{base_col}' not found in source dataframe for lag feature.")

    right_lf = source_lf.select(
        *join_keys,
        pl.col("valid_time"),
        pl.col(base_col).alias(f"{base_col}_lag_{lag_hours}h"),
    )

    # Join target_lf with right_lf
    result = lf_with_target_time.join(
        right_lf,
        left_on=join_keys + ["target_time"],
        right_on=join_keys + ["valid_time"],
        how="left",
    ).drop("target_time")

    return result


def calculate_lead_time(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Calculates the lead time in hours if 'init_time' is present in the schema."""
    if "init_time" in lf.collect_schema().names():
        return lf.with_columns(
            lead_time_hours=(
                (pl.col("valid_time") - pl.col("init_time")).dt.total_seconds() / 3600
            ).cast(pl.Float32)
        )
    return lf


def nullify_leaky_lags(lf: pl.LazyFrame, lag_cols: dict[str, int]) -> pl.LazyFrame:
    """
    Nullifies lagged features that would cause lookahead bias.

    During training, we must ensure that the model cannot access actual data that
    would not be available at inference time. If a requested lag is shorter than
    or equal to the forecast lead time, the feature is effectively a "future"
    value and must be nullified.

    Args:
        lf: The LazyFrame containing the features and 'lead_time_hours'.
        lag_cols: A dictionary mapping column names to their lag in hours.

    Returns:
        A LazyFrame with leaky lag columns set to null.
    """
    for col_name, lag_hours in lag_cols.items():
        lf = lf.with_columns(
            pl.when(pl.col("lead_time_hours") >= lag_hours)
            .then(pl.lit(None))
            .otherwise(pl.col(col_name))
            .alias(col_name)
        )
    return lf


def apply_rolling_mean_feature(
    lf: pl.LazyFrame, base_col: BaseColumn, window_hours: int
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


def apply_latest_weekly_lag_feature(
    lf: pl.LazyFrame,
    source_lf: pl.LazyFrame,
    base_col: BaseColumn,
    new_col: str,
    join_cols: list[Literal["time_series_id", "ensemble_member"]],
) -> pl.LazyFrame:
    """
    Applies a dynamic lag feature based on the lead time.

    Why dynamic lags? If we are forecasting 10 days ahead, a 7-day lag is useless because
    we won't have the actual data 7 days from now. We need to use the *latest available*
    weekly lag (e.g., 14 days ago). This function calculates the appropriate multiple of
    168 hours (1 week) based on the lead time.

    Args:
        lf: The main LazyFrame containing the target times and lead times.
        source_lf: The LazyFrame containing the historical data to pull from.
        base_col: The column in source_lf to lag (e.g., 'power').
        new_col: The name of the resulting lagged column.
        join_cols: Additional columns to join on (e.g., ['time_series_id', 'ensemble_member']).

    Returns:
        A LazyFrame with the new dynamically lagged column.
    """
    schema_names = lf.collect_schema().names()

    if "lead_time_hours" in schema_names:
        # Calculate the required lag in hours.
        # If lead_time is 0-167, lag is 168.
        # If lead_time is 168-335, lag is 336.
        lag_hours = (((pl.col("lead_time_hours") / 168).floor() + 1) * 168).cast(pl.Int64)
    else:
        # Default to 1 week if no lead time is available
        lag_hours = pl.lit(168).cast(pl.Int64)

    # Calculate the target time we need to look up in the source data
    lf_with_target = lf.with_columns(
        target_time=pl.col("valid_time") - pl.duration(hours=lag_hours)
    )

    # Prepare the right side of the join
    right_lf = source_lf.select(
        *join_cols,
        pl.col("valid_time"),
        pl.col(base_col).alias(new_col),
    )

    # Perform the join
    result = lf_with_target.join(
        right_lf,
        left_on=join_cols + ["target_time"],
        right_on=join_cols + ["valid_time"],
        how="left",
    ).drop("target_time")

    return result


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
