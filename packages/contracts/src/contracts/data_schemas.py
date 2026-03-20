"""Data schemas for the NGED substation forecast project."""

from collections.abc import Sequence
from datetime import datetime
from typing import cast

import patito as pt
import polars as pl
from pydantic import BaseModel


class MissingCorePowerVariablesError(ValueError):
    """Raised when a substation CSV lacks both MW and MVA data."""

    pass


class SubstationFlows(pt.Model):
    timestamp: datetime = pt.Field(dtype=pl.Datetime(time_unit="us", time_zone="UTC"))

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

    # When this data was ingested into our system. When we update our datasets, we examine
    # `ingested_at` to figure out whether we need to get new data from NGED for this substation.
    # `ingested_at` is only missing for data ingested before around mid-March 2026 (prior to this,
    # we didn't record when the data was ingested).
    ingested_at: datetime | None = pt.Field(dtype=pl.Datetime(time_zone="UTC"), allow_missing=True)

    @classmethod
    def validate(
        cls,
        dataframe: pl.DataFrame,
        columns: Sequence[str] | None = None,
        allow_missing_columns: bool = False,
        allow_superfluous_columns: bool = False,
        drop_superfluous_columns: bool = False,
    ) -> pt.DataFrame["SubstationFlows"]:  # type: ignore[invalid-method-override]
        """Validate the given dataframe, ensuring either MW or MVA is present."""
        if "MW" not in dataframe.columns and "MVA" not in dataframe.columns:
            raise MissingCorePowerVariablesError(
                "SubstationFlows dataframe must contain at least one of 'MW' or 'MVA' columns."
                f" {dataframe.columns=}, {dataframe.height=}"
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
        return "MW" if dataframe["MW"].is_not_null().all() else "MVA"

    @staticmethod
    def to_simplified_substation_flows(
        dataframe: pt.DataFrame["SubstationFlows"],
    ) -> pt.DataFrame[SimplifiedSubstationFlows]:
        power_col = SubstationFlows.choose_power_column(dataframe)
        dataframe = dataframe.rename({power_col: "MW_or_MVA"})  # type: ignore[invalid-assignment]
        dataframe = dataframe.select(["timestamp", "MW_or_MVA"]).drop_nulls()  # type: ignore[invalid-assignment]
        return cast(pt.DataFrame[SimplifiedSubstationFlows], dataframe)


class SimplifiedSubstationFlows(pt.Model):
    timestamp: datetime = pt.Field(dtype=pl.Datetime(time_unit="us", time_zone="UTC"))
    MW_or_MVA: float = pt.Field(dtype=pl.Float32, ge=-1_000, le=1_000)


class SubstationLocations(pt.Model):
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
    last_updated: datetime = pt.Field(dtype=pl.Datetime(time_zone="UTC"))


class PowerForecast(pt.Model):
    """Forecast data schema."""

    nwp_init_time: datetime = pt.Field(dtype=pl.Datetime(time_zone="UTC"))
    substation_number: int = pt.Field(dtype=pl.Int32)
    valid_time: datetime = pt.Field(dtype=pl.Datetime(time_zone="UTC"))
    ensemble_member: int = pt.Field(dtype=pl.UInt8)

    # Identifier for our ML-based power forecasting model.
    # Returned by `Forecaster.model_name_and_version()`.
    power_fcst_model: str = pt.Field(dtype=pl.Categorical)

    # Active power (megawatts) or apparent power (mega volt amp).
    MW_or_MVA: float = pt.Field(dtype=pl.Float32)

    # TODO: Capture probabilistic information.


class InferenceParams(BaseModel):
    """Parameters for ML model inference."""

    nwp_init_time: datetime
    power_fcst_model: str | None = None


class Nwp(pt.Model):
    """Weather data schema for NWP forecasts."""

    init_time: datetime = pt.Field(dtype=pl.Datetime(time_zone="UTC"))
    valid_time: datetime = pt.Field(dtype=pl.Datetime(time_zone="UTC"))
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
        dataframe: pl.DataFrame,
        columns: Sequence[str] | None = None,
        allow_missing_columns: bool = False,
        allow_superfluous_columns: bool = False,
        drop_superfluous_columns: bool = False,
    ) -> pt.DataFrame["Nwp"]:  # type: ignore[invalid-method-override]
        """Validate the given dataframe, ensuring no nulls from second step onwards."""
        validated_df = super().validate(
            dataframe=dataframe,
            columns=columns,
            allow_missing_columns=allow_missing_columns,
            allow_superfluous_columns=allow_superfluous_columns,
            drop_superfluous_columns=drop_superfluous_columns,
        )

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


class ProcessedNwp(pt.Model):
    """Weather data after ensemble selection and interpolation."""

    valid_time: datetime = pt.Field(dtype=pl.Datetime(time_unit="us", time_zone="UTC"))
    h3_index: int = pt.Field(dtype=pl.UInt64)

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
    """Final joined dataset ready for XGBoost."""

    valid_time: datetime = pt.Field(dtype=pl.Datetime(time_unit="us", time_zone="UTC"))
    substation_number: int = pt.Field(dtype=pl.Int32)
    MW_or_MVA: float = pt.Field(dtype=pl.Float32)

    # Power lags
    power_lag_7d: float | None = pt.Field(dtype=pl.Float32)
    power_lag_14d: float | None = pt.Field(dtype=pl.Float32)

    # Weather features
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

    # Physical features
    wind_speed_10m_phys: float = pt.Field(dtype=pl.Float32)
    wind_direction_10m_phys: float = pt.Field(dtype=pl.Float32)
    windchill: float = pt.Field(dtype=pl.Float32)

    # Weather lags/trends
    temperature_2m_lag_7d: float | None = pt.Field(dtype=pl.Float32)
    sw_radiation_lag_7d: float | None = pt.Field(dtype=pl.Float32)
    temperature_2m_lag_14d: float | None = pt.Field(dtype=pl.Float32)
    sw_radiation_lag_14d: float | None = pt.Field(dtype=pl.Float32)
    temperature_2m_6h_ago: float | None = pt.Field(dtype=pl.Float32)
    temp_trend_6h: float | None = pt.Field(dtype=pl.Float32)

    # Temporal features
    hour_sin: float = pt.Field(dtype=pl.Float32)
    hour_cos: float = pt.Field(dtype=pl.Float32)
    day_of_year_sin: float = pt.Field(dtype=pl.Float32)
    day_of_year_cos: float = pt.Field(dtype=pl.Float32)
    day_of_week: int = pt.Field(dtype=pl.Int8)
