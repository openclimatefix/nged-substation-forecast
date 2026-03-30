"""Data schemas for the NGED substation forecast project."""

from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, cast

import patito as pt
import polars as pl
from pydantic import BaseModel

if TYPE_CHECKING:
    import pandas as pd


# Define our standard datetime type for all schemas
UTC_DATETIME_DTYPE = pl.Datetime(time_unit="us", time_zone="UTC")


class MissingCorePowerVariablesError(ValueError):
    """Raised when a substation CSV lacks both MW and MVA data."""

    pass


class SubstationFlows(pt.Model):
    timestamp: datetime = pt.Field(dtype=UTC_DATETIME_DTYPE)

    # The unique identifier for the substation.
    substation_number: int = pt.Field(dtype=pl.Int32)

    # Primary substations usually have flows in the tens of MW.
    # We'll set a loose range for now to catch extreme errors.
    # If we want to reduce storage space we could store kW and kVAr as Int16.

    # Active power:
    MW: float | None = pt.Field(dtype=pl.Float32, allow_missing=True, ge=-1_000, le=1_000)

    # Apparent power:
    MVA: float | None = pt.Field(dtype=pl.Float32, allow_missing=True, ge=-1_000, le=1_000)

    # Reactive power:
    MVAr: float | None = pt.Field(dtype=pl.Float32, allow_missing=True, ge=-1_000, le=1_000)

    # The datetime this data was ingested into our system. When we update our datasets, we examine
    # `ingested_at` to figure out whether we need to get new data from NGED for this substation.
    # `ingested_at` is only missing for data ingested before around mid-March 2026 (prior to this,
    # we didn't record when the data was ingested).
    ingested_at: datetime | None = pt.Field(dtype=UTC_DATETIME_DTYPE, allow_missing=True)

    @classmethod
    def validate(
        cls,
        dataframe: pl.DataFrame | "pd.DataFrame",
        columns: Sequence[str] | None = None,
        allow_missing_columns: bool = False,
        allow_superfluous_columns: bool = False,
        drop_superfluous_columns: bool = False,
    ) -> pt.DataFrame["SubstationFlows"]:
        """Validate the given dataframe, ensuring either MW or MVA is present and has data."""
        if "MW" not in dataframe.columns and "MVA" not in dataframe.columns:
            raise MissingCorePowerVariablesError(
                "SubstationFlows dataframe must contain at least one of 'MW' or 'MVA' columns."
                f" {dataframe.columns=}, {len(dataframe)=}"
            )

        # Ensure at least one of MW or MVA has non-null data
        if isinstance(dataframe, pl.DataFrame):
            mw_null = "MW" not in dataframe.columns or dataframe["MW"].null_count() == len(
                dataframe
            )
            mva_null = "MVA" not in dataframe.columns or dataframe["MVA"].null_count() == len(
                dataframe
            )
            if mw_null and mva_null:
                raise MissingCorePowerVariablesError(
                    "SubstationFlows dataframe must have non-null data in either 'MW' or 'MVA'."
                )

        return cast(
            pt.DataFrame["SubstationFlows"],
            super().validate(
                dataframe=dataframe,
                columns=columns,
                allow_missing_columns=allow_missing_columns,
                allow_superfluous_columns=allow_superfluous_columns,
                drop_superfluous_columns=drop_superfluous_columns,
            ),
        )

    @staticmethod
    def choose_power_column(dataframe: pt.DataFrame["SubstationFlows"]) -> str:
        mw_valid = dataframe["MW"].is_not_null().sum() if "MW" in dataframe.columns else 0
        mva_valid = dataframe["MVA"].is_not_null().sum() if "MVA" in dataframe.columns else 0
        return "MW" if mw_valid >= mva_valid else "MVA"

    @staticmethod
    def to_simplified_substation_flows(
        dataframe: pt.DataFrame["SubstationFlows"],
    ) -> pt.DataFrame[SimplifiedSubstationFlows]:
        power_col = SubstationFlows.choose_power_column(dataframe)
        simplified_df = (
            dataframe.rename({power_col: "MW_or_MVA"})
            .select(["timestamp", "MW_or_MVA"])
            .drop_nulls()
        )
        return cast(pt.DataFrame[SimplifiedSubstationFlows], simplified_df)


class SimplifiedSubstationFlows(pt.Model):
    timestamp: datetime = pt.Field(dtype=UTC_DATETIME_DTYPE)
    MW_or_MVA: float = pt.Field(dtype=pl.Float32, ge=-1_000, le=1_000)


class SubstationTargetMap(pt.Model):
    """Map of substation_number to the target column (MW or MVA) to use."""

    substation_number: int = pt.Field(dtype=pl.Int32, unique=True)
    target_col: str = pt.Field(dtype=pl.String)
    peak_capacity: float = pt.Field(dtype=pl.Float32, gt=0)


class SubstationLocations(pt.Model):
    """The data structure of the raw substation location data from NGED."""

    # NGED has 192,000 substations.
    substation_number: int = pt.Field(dtype=pl.Int32, unique=True, gt=0, lt=1_000_000)

    # The min and max string lengths are actually 3 and 48 chars, respectively.
    # Note that there are two "Park Lane" substations, with different locations and different
    # substation numbers.
    substation_name: str = pt.Field(dtype=pl.String, min_length=2, max_length=64)

    substation_type: str = pt.Field(dtype=pl.Categorical)
    latitude: float | None = pt.Field(dtype=pl.Float32, ge=49, le=61)  # UK latitude range
    longitude: float | None = pt.Field(dtype=pl.Float32, ge=-9, le=2)  # UK longitude range


class SubstationLocationsWithH3(SubstationLocations):
    """Substation locations including their H3 index."""

    h3_res_5: int | None = pt.Field(dtype=pl.UInt64)


class SubstationMetadata(pt.Model):
    """Metadata for a substation, joining location data with live telemetry info."""

    # NGED has 192,000 substations.
    substation_number: int = pt.Field(dtype=pl.Int32, unique=True, gt=0, lt=1_000_000)

    substation_name_in_location_table: str = pt.Field(dtype=pl.String, min_length=2, max_length=64)

    # This will be null if the substation doesn't have live telemetry.
    substation_name_in_live_primaries: str | None = pt.Field(
        dtype=pl.String, min_length=2, max_length=128, allow_missing=True
    )

    # The URL to the live telemetry CSV on NGED's CKAN portal.
    url: str | None = pt.Field(dtype=pl.String, allow_missing=True)

    substation_type: str = pt.Field(dtype=pl.Categorical)
    latitude: float | None = pt.Field(dtype=pl.Float32, ge=49, le=61)  # UK latitude range
    longitude: float | None = pt.Field(dtype=pl.Float32, ge=-9, le=2)  # UK longitude range
    h3_res_5: int | None = pt.Field(dtype=pl.UInt64)

    # When this metadata record was last updated from the upstream NGED datasets.
    last_updated: datetime = pt.Field(dtype=UTC_DATETIME_DTYPE)


class PowerForecast(pt.Model):
    """Forecast data schema for deterministic ensemble forecasts."""

    valid_time: datetime = pt.Field(dtype=UTC_DATETIME_DTYPE)
    substation_number: int = pt.Field(dtype=pl.Int32)
    ensemble_member: int = pt.Field(dtype=pl.UInt8)

    # Identifier for our ML-based power forecasting model.
    power_fcst_model_name: str = pt.Field(dtype=pl.Categorical)

    # The datetime that the power forecast was initialised.
    power_fcst_init_time: datetime = pt.Field(dtype=UTC_DATETIME_DTYPE)

    # The datetime that the underlying weather forecast was initialised.
    nwp_init_time: datetime = pt.Field(dtype=UTC_DATETIME_DTYPE)

    # Year and month of the power forecast initialisation (for partitioning).
    power_fcst_init_year_month: str = pt.Field(dtype=pl.String)

    # Active power (megawatts) or apparent power (mega volt amp).
    MW_or_MVA: float = pt.Field(dtype=pl.Float32)


class ScalingParams(pt.Model):
    """Schema for weather variable scaling parameters.

    Used when scaling between physical units (e.g. degrees C) and their unsigned 8-bit integer
    (uint8) representations (in the range [0, 255])."""

    col_name: str = pt.Field(dtype=pl.String)
    buffered_min: float = pt.Field(dtype=pl.Float32)
    buffered_range: float = pt.Field(dtype=pl.Float32)


class InferenceParams(BaseModel):
    """Parameters for ML model inference."""

    nwp_init_time: datetime
    power_fcst_model_name: str | None = None


class Nwp(pt.Model):
    """Weather data schema for NWP forecasts."""

    init_time: datetime = pt.Field(dtype=UTC_DATETIME_DTYPE)
    valid_time: datetime = pt.Field(dtype=UTC_DATETIME_DTYPE)
    ensemble_member: int = pt.Field(dtype=pl.UInt8)
    h3_index: int = pt.Field(dtype=pl.UInt64)

    # Variables stored as uint8 on disk
    temperature_2m: int = pt.Field(dtype=pl.UInt8)
    dew_point_temperature_2m: int = pt.Field(dtype=pl.UInt8)
    wind_speed_10m: int = pt.Field(dtype=pl.UInt8)
    wind_direction_10m: int = pt.Field(dtype=pl.UInt8)
    wind_speed_100m: int = pt.Field(dtype=pl.UInt8)
    wind_direction_100m: int = pt.Field(dtype=pl.UInt8)
    pressure_surface: int = pt.Field(dtype=pl.UInt8)
    pressure_reduced_to_mean_sea_level: int = pt.Field(dtype=pl.UInt8)
    geopotential_height_500hpa: int = pt.Field(dtype=pl.UInt8)

    # Precipitation and radiation variables are null for the first forecast step (lead time 0)
    # in ECMWF ENS, as they are accumulated over the previous interval.
    downward_long_wave_radiation_flux_surface: int | None = pt.Field(dtype=pl.UInt8)
    downward_short_wave_radiation_flux_surface: int | None = pt.Field(dtype=pl.UInt8)
    precipitation_surface: int | None = pt.Field(dtype=pl.UInt8)

    categorical_precipitation_type_surface: int = pt.Field(dtype=pl.UInt8)

    @classmethod
    def validate(
        cls,
        dataframe: pl.DataFrame | "pd.DataFrame",
        columns: Sequence[str] | None = None,
        allow_missing_columns: bool = False,
        allow_superfluous_columns: bool = False,
        drop_superfluous_columns: bool = False,
    ) -> pt.DataFrame["Nwp"]:
        """Validate the given dataframe, ensuring no nulls from second step onwards."""
        validated_df = super().validate(
            dataframe=dataframe,
            columns=columns,
            allow_missing_columns=allow_missing_columns,
            allow_superfluous_columns=allow_superfluous_columns,
            drop_superfluous_columns=drop_superfluous_columns,
        )

        if not isinstance(validated_df, pl.DataFrame):
            return cast(pt.DataFrame["Nwp"], validated_df)

        # Check for nulls from second forecast step onwards
        # (i.e. where valid_time > init_time)
        cols_to_check = [
            "precipitation_surface",
            "downward_short_wave_radiation_flux_surface",
            "downward_long_wave_radiation_flux_surface",
        ]

        second_step_onwards = validated_df.filter(pl.col("valid_time") > pl.col("init_time"))

        for col in cols_to_check:
            null_count = second_step_onwards.select(pl.col(col).is_null().sum()).item()
            if null_count > 0:
                raise ValueError(
                    f"Column '{col}' contains {null_count} null values from the second forecast "
                    "step onwards. These variables are only allowed to be null for the first "
                    "forecast step (lead time 0)."
                )

        return cast(pt.DataFrame["Nwp"], validated_df)


class NwpColumns:
    """Constants for NWP column names."""

    VALID_TIME = "valid_time"
    INIT_TIME = "init_time"
    LEAD_TIME_HOURS = "lead_time_hours"
    H3_INDEX = "h3_index"
    ENSEMBLE_MEMBER = "ensemble_member"
    TEMPERATURE_2M = "temperature_2m"
    WIND_SPEED_10M = "wind_speed_10m"
    WIND_DIRECTION_10M = "wind_direction_10m"
    SW_RADIATION = "downward_short_wave_radiation_flux_surface"


class ProcessedNwp(pt.Model):
    """Weather data after ensemble selection and interpolation.

    Note: Accumulated variables (e.g., precipitation, radiation) are already
    de-accumulated by Dynamical.org prior to download, and should not be differenced.

    Clever Optimization:
    To save memory, weather variables are scaled to a 0-255 range (uint8) before being saved to disk.
    The scaling formula is: uint8_value = (physical_value - min_bound) / (max_bound - min_bound) * 255.
    When loaded, they are cast to Float32 but retain the 0-255 scale.
    """

    valid_time: datetime = pt.Field(dtype=UTC_DATETIME_DTYPE)
    init_time: datetime = pt.Field(dtype=UTC_DATETIME_DTYPE)
    lead_time_hours: float = pt.Field(dtype=pl.Float32)
    h3_index: int = pt.Field(dtype=pl.UInt64)
    ensemble_member: int | None = pt.Field(dtype=pl.UInt8, allow_missing=True)

    # Weather variables as Float32
    temperature_2m: float = pt.Field(dtype=pl.Float32)
    dew_point_temperature_2m: float = pt.Field(dtype=pl.Float32)
    wind_speed_10m: float = pt.Field(dtype=pl.Float32)
    wind_direction_10m: float = pt.Field(dtype=pl.Float32)
    wind_speed_100m: float = pt.Field(dtype=pl.Float32)
    wind_direction_100m: float = pt.Field(dtype=pl.Float32)
    pressure_surface: float = pt.Field(dtype=pl.Float32)
    pressure_reduced_to_mean_sea_level: float = pt.Field(dtype=pl.Float32)
    geopotential_height_500hpa: float = pt.Field(dtype=pl.Float32)
    downward_long_wave_radiation_flux_surface: float | None = pt.Field(dtype=pl.Float32)
    downward_short_wave_radiation_flux_surface: float | None = pt.Field(dtype=pl.Float32)
    precipitation_surface: float | None = pt.Field(dtype=pl.Float32)
    categorical_precipitation_type_surface: float = pt.Field(dtype=pl.Float32)


class SubstationFeatures(pt.Model):
    """Final joined dataset ready for XGBoost.

    Clever Optimization:
    Weather features are kept in their 0-255 scaled representation (e.g., `temperature_2m_uint8_scaled`)
    to save memory and computation. XGBoost is invariant to monotonic transformations, so this does not
    affect model performance. For SHAP analysis or EDA, use `descale_for_analysis` to convert them back
    to physical units.
    """

    valid_time: datetime = pt.Field(dtype=UTC_DATETIME_DTYPE)
    substation_number: int = pt.Field(dtype=pl.Int32)
    ensemble_member: int | None = pt.Field(dtype=pl.UInt8, allow_missing=True)
    MW_or_MVA: float = pt.Field(dtype=pl.Float32)

    # Power lags
    latest_available_weekly_lag: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)

    # Weather features
    temperature_2m_uint8_scaled: float = pt.Field(dtype=pl.Float32)
    dew_point_temperature_2m_uint8_scaled: float | None = pt.Field(
        dtype=pl.Float32, allow_missing=True
    )
    wind_speed_10m_uint8_scaled: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    wind_direction_10m_uint8_scaled: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    wind_speed_100m_uint8_scaled: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    wind_direction_100m_uint8_scaled: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    pressure_surface_uint8_scaled: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    pressure_reduced_to_mean_sea_level_uint8_scaled: float | None = pt.Field(
        dtype=pl.Float32, allow_missing=True
    )
    geopotential_height_500hpa_uint8_scaled: float | None = pt.Field(
        dtype=pl.Float32, allow_missing=True
    )
    downward_long_wave_radiation_flux_surface_uint8_scaled: float | None = pt.Field(
        dtype=pl.Float32, allow_missing=True
    )
    downward_short_wave_radiation_flux_surface_uint8_scaled: float | None = pt.Field(
        dtype=pl.Float32, allow_missing=True
    )
    precipitation_surface_uint8_scaled: float | None = pt.Field(
        dtype=pl.Float32, allow_missing=True
    )
    categorical_precipitation_type_surface_uint8_scaled: float | None = pt.Field(
        dtype=pl.Float32, allow_missing=True
    )

    # Physical features
    windchill: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)

    # Weather lags/trends
    temperature_2m_lag_7d: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    sw_radiation_lag_7d: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    temperature_2m_lag_14d: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    sw_radiation_lag_14d: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    temperature_2m_6h_ago: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    temp_trend_6h: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)

    # Temporal features
    hour_sin: float = pt.Field(dtype=pl.Float32)
    hour_cos: float = pt.Field(dtype=pl.Float32)
    day_of_year_sin: float = pt.Field(dtype=pl.Float32)
    day_of_year_cos: float = pt.Field(dtype=pl.Float32)
    day_of_week: int = pt.Field(dtype=pl.Int8)
