from collections.abc import Sequence
from datetime import datetime
from typing import Self

import patito as pt
import polars as pl

from .common import UTC_DATETIME_DTYPE


class _NwpBase(pt.Model):
    """Weather data schema for NWP forecasts, using"""

    init_time: datetime = pt.Field(dtype=UTC_DATETIME_DTYPE)
    valid_time: datetime = pt.Field(dtype=UTC_DATETIME_DTYPE)
    ensemble_member: int = pt.Field(dtype=pl.UInt8)
    h3_index: int = pt.Field(dtype=pl.UInt64)


class NwpInMemory(_NwpBase):
    """Variables stored as Float32 in memory."""

    temperature_2m: float = pt.Field(dtype=pl.Float32)
    dew_point_temperature_2m: float = pt.Field(dtype=pl.Float32)
    wind_speed_10m: float = pt.Field(dtype=pl.Float32)
    wind_direction_10m: float = pt.Field(dtype=pl.Float32)
    wind_speed_100m: float = pt.Field(dtype=pl.Float32)
    wind_direction_100m: float = pt.Field(dtype=pl.Float32)
    pressure_surface: float = pt.Field(dtype=pl.Float32)
    pressure_reduced_to_mean_sea_level: float = pt.Field(dtype=pl.Float32)
    geopotential_height_500hpa: float = pt.Field(dtype=pl.Float32)

    # Precipitation and radiation variables are null for the first forecast step (lead time 0) in
    # ECMWF ENS. Also note that, whilst these variables accumulate over forecast steps in ECMWF's
    # raw forecasts, we get ECMWF ENS from Dynamical.org, and Dynamical.org de-accumulates these
    # values before we receive them. So these are true _rates_.
    downward_long_wave_radiation_flux_surface: float | None = pt.Field(dtype=pl.Float32)
    downward_short_wave_radiation_flux_surface: float | None = pt.Field(dtype=pl.Float32)
    precipitation_surface: float | None = pt.Field(dtype=pl.Float32)

    # Categorical variables
    categorical_precipitation_type_surface: int = pt.Field(dtype=pl.UInt8)

    @classmethod
    def validate(
        cls,
        dataframe: pl.DataFrame,
        columns: Sequence[str] | None = None,
        allow_missing_columns: bool = False,
        allow_superfluous_columns: bool = False,
        drop_superfluous_columns: bool = False,
    ) -> pt.DataFrame[Self]:  # ty:ignore[invalid-method-override]
        """Validate the given dataframe, ensuring no nulls from second step onwards and uniqueness."""
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
            has_nulls = second_step_onwards[col].is_null().any()
            if has_nulls:
                raise ValueError(
                    f"Column '{col}' contains has null values from the second forecast "
                    "step onwards. These variables are only allowed to be null for the first "
                    "forecast step (lead time 0)."
                )

        # Validate uniqueness of (init_time, valid_time, ensemble_member, h3_index)
        if (
            validated_df.select(["init_time", "valid_time", "ensemble_member", "h3_index"])
            .is_duplicated()
            .any()
        ):
            raise ValueError(
                "Duplicate entries found for (init_time, valid_time, ensemble_member, h3_index)."
            )

        return validated_df


class NwpOnDisk(_NwpBase):
    """Variables stored as integers on disk to reduce storage requirements."""

    temperature_2m: int = pt.Field(dtype=pl.Int16)
    dew_point_temperature_2m: int = pt.Field(dtype=pl.Int16)
    wind_speed_10m: int = pt.Field(dtype=pl.Int16)
    wind_direction_10m: int = pt.Field(dtype=pl.Int16)
    wind_speed_100m: int = pt.Field(dtype=pl.Int16)
    wind_direction_100m: int = pt.Field(dtype=pl.Int16)
    pressure_surface: int = pt.Field(dtype=pl.Int16)
    pressure_reduced_to_mean_sea_level: int = pt.Field(dtype=pl.Int16)
    geopotential_height_500hpa: int = pt.Field(dtype=pl.Int16)

    # Precipitation and radiation variables are null for the first forecast step (lead time 0) in
    # ECMWF ENS. Also note that, whilst these variables accumulate over forecast steps in ECMWF's
    # raw forecasts, we get ECMWF ENS from Dynamical.org, and Dynamical.org de-accumulates these
    # values before we receive them. So these are true _rates_.
    downward_long_wave_radiation_flux_surface: int | None = pt.Field(dtype=pl.Int16)
    downward_short_wave_radiation_flux_surface: int | None = pt.Field(dtype=pl.Int16)
    precipitation_surface: int | None = pt.Field(dtype=pl.Int16)

    # Categorical variables
    categorical_precipitation_type_surface: int = pt.Field(dtype=pl.UInt8)

    @classmethod
    def from_nwp_in_memory(
        cls,
        nwp_in_memory: pt.DataFrame[NwpInMemory],
        scaling_params: pt.DataFrame[NwpScalingParams],
    ) -> Self:
        pass


class NwpScalingParams(pt.Model):
    """Schema for weather variable scaling parameters.

    Used when scaling between physical units (e.g. degrees C) and their integer representations."""

    col_name: str = pt.Field(dtype=pl.String)

    # The minimum from the actual values, minus a small buffer.
    buffered_min: float = pt.Field(dtype=pl.Float32)

    # The range from the actual values, plus a small buffer.
    buffered_range: float = pt.Field(dtype=pl.Float32)

    # The maximum from the actual values, plus a small buffer.
    buffered_max: float = pt.Field(dtype=pl.Float32)
