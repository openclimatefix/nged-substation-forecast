"""Data schemas for the NGED substation forecast project."""

from collections.abc import Sequence
from datetime import datetime

import patito as pt
import polars as pl
from typing import cast


class SubstationFlows(pt.Model):
    timestamp: datetime = pt.Field(dtype=pl.Datetime(time_zone="UTC"))

    # Primary substations usually have flows in the tens of MW.
    # We'll set a loose range for now to catch extreme errors.
    # If we want to reduce storage space we could store kW and kVAr as Int16.

    # Active power:
    MW: float | None = pt.Field(dtype=pl.Float32, allow_missing=True, ge=-1_000, le=1_000)

    # Apparent power:
    MVA: float | None = pt.Field(dtype=pl.Float32, allow_missing=True, ge=-1_000, le=1_000)

    # Reactive power:
    MVAr: float | None = pt.Field(dtype=pl.Float32, allow_missing=True, ge=-1_000, le=1_000)

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
            raise ValueError(
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


class PowerForecast(pt.Model):
    """Forecast data schema."""

    nwp_init_time: datetime = pt.Field(dtype=pl.Datetime(time_zone="UTC"))
    substation_id: int = pt.Field(dtype=pl.Int32)
    power_mw: float = pt.Field(dtype=pl.Float32)
    valid_time: datetime = pt.Field(dtype=pl.Datetime(time_zone="UTC"))
    power_fcst_model: str = pt.Field(dtype=pl.Categorical)


class Nwp(pt.Model):
    """Weather data schema for NWP forecasts."""

    nwp_source: str = pt.Field(dtype=pl.String)
    init_time: datetime = pt.Field(dtype=pl.Datetime(time_zone="UTC"))
    valid_time: datetime = pt.Field(dtype=pl.Datetime(time_zone="UTC"))
    ensemble_member: int = pt.Field(dtype=pl.UInt16)
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
