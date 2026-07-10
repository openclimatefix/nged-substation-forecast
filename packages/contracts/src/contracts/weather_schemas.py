from collections.abc import Sequence
from datetime import datetime, timezone
from enum import StrEnum, auto
from pathlib import Path
from typing import ClassVar, Literal, Self

import patito as pt
import polars as pl

from contracts._uri import ObjectStoreOptions
from contracts.settings import Settings
from contracts.typing_utils import typeddict_to_dict

from .common import UTC_DATETIME_DTYPE

SETTINGS = Settings()


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


class NwpModelId(StrEnum):
    ECMWF_ENS_0_25_degree = auto()


NWP_MODEL_ID_DTYPE = pl.Enum([model.name for model in NwpModelId])


class NwpMetaData(pt.Model):
    """Metadata about numerical weather prediction models."""

    nwp_model_id: str = pt.Field(
        dtype=NWP_MODEL_ID_DTYPE,
        description="Primary key for joining with NWP data (e.g. 'ECMWF_ENS_0_25_degree').",
        unique=True,
    )

    provider: str = pt.Field(
        dtype=pl.Enum(["ECMWF"]),
        description="NWP data provider (currently always 'ECMWF').",
    )

    h3_resolution: int = pt.Field(
        dtype=pl.Int8,
        description="H3 spatial resolution used for this NWP model's grid.",
    )

    is_ensemble: bool = pt.Field(description="Whether this NWP model produces ensemble forecasts.")

    @classmethod
    def load(cls, csv_path: str | Path = SETTINGS.nwp_metadata_csv_path) -> pt.DataFrame[Self]:
        """Load NWP metadata from a static CSV file."""
        df = pl.read_csv(csv_path)
        # Patito's .cast() will handle the conversion to the Enum type defined in the model
        return pt.DataFrame(df).set_model(cls).cast().validate()


class Nwp(pt.Model):
    """Weather data schema for NWP forecasts: gridded ECMWF ENS ensemble weather, one row per
    (nwp_model_id, init_time, valid_time, ensemble_member, h3_index).

    Stored on disk as plain Float32, rounded to a significand-bit budget by
    `delta_store.nwp.write_nwp` — see `docs/architecture/overview.md` for the physical format
    and measured numbers.
    """

    nwp_model_id: str = pt.Field(
        dtype=NWP_MODEL_ID_DTYPE,
        description="The primary key for joining with NwpMetaData (e.g. 'ECMWF_ENS_0_25_degree').",
    )

    init_time: datetime = pt.Field(
        dtype=UTC_DATETIME_DTYPE,
        description="When the NWP model run was initialised.",
    )

    valid_time: datetime = pt.Field(
        dtype=UTC_DATETIME_DTYPE,
        description=(
            "The time for which this NWP value is valid. Most variables (temperature, wind, "
            "pressure, geopotential) are instantaneous — they describe conditions at this moment. "
            "Precipitation and radiation (downward_long_wave_radiation_flux_surface, "
            "downward_short_wave_radiation_flux_surface, precipitation_surface) are period-ending "
            "rates: each value represents the average rate over the period that ends at valid_time "
            "(i.e. the preceding forecast step interval). Dynamical.org de-accumulates these from "
            "ECMWF's raw cumulative fields before we receive them."
        ),
    )

    ensemble_member: int = pt.Field(
        dtype=pl.UInt8,
        description="Ensemble member index (0-based).",
    )

    h3_index: int = pt.Field(
        dtype=pl.UInt64,
        description="H3 cell index. The H3 resolution for the nwp_model_id is stored in NwpMetaData.",
    )

    temperature_2m: float = pt.Field(
        dtype=pl.Float32,
        description="Air temperature at 2 m above ground. Unit: degrees C.",
        ge=-100,
        le=100,
    )

    dew_point_temperature_2m: float = pt.Field(
        dtype=pl.Float32,
        description="Dew-point temperature at 2 m. Unit: degrees C.",
        ge=-100,
        le=100,
    )

    wind_speed_10m: float = pt.Field(
        dtype=pl.Float32,
        description="Wind speed at 10 m. Unit: meters per second.",
        ge=0,
        le=200,  # Gemini says the highest non-tornadic surface wind speed recorded was 113 m/s
    )

    wind_direction_10m: float = pt.Field(
        dtype=pl.Float32,
        description="Wind direction at 10 m. The angle where the wind is coming from. Degrees. 0° is North; 90° is East.",
        ge=0,
        le=360,
    )

    wind_speed_100m: float = pt.Field(
        dtype=pl.Float32,
        description="Wind speed at 100 m. Unit: meters per second.",
        ge=0,
        le=200,  # Gemini says the highest non-tornadic surface wind speed recorded was 113 m/s
    )

    wind_direction_100m: float = pt.Field(
        dtype=pl.Float32,
        description="Wind direction at 100 m. The angle where the wind is coming from. Degrees. 0° is North; 90° is East.",
        ge=0,
        le=360,
    )

    pressure_surface: float = pt.Field(
        dtype=pl.Float32,
        description="Surface pressure. Unit: Pa.",
        ge=0,
        le=200_000,  # Max in 2 years of ECMWF ENS = 105_760
    )

    pressure_reduced_to_mean_sea_level: float = pt.Field(
        dtype=pl.Float32,
        description="Mean sea-level pressure. Unit: Pa.",
        ge=0,
        le=200_000,  # Max in 2 years of ECMWF ENS = 105_727
    )

    geopotential_height_500hpa: float = pt.Field(
        dtype=pl.Float32,
        description="Geopotential height of the 500 hPa pressure surface. Unit: m.",
        ge=0,
        le=10_000,  # Max in 2 years of ECMWF ENS = 6_030
    )

    # Precipitation and radiation variables are null for the first forecast step (lead time 0) in
    # ECMWF ENS. Also note that, whilst these variables accumulate over forecast steps in ECMWF's
    # raw forecasts, we get ECMWF ENS from Dynamical.org, and Dynamical.org de-accumulates these
    # values before we receive them. So these are true _rates_.
    downward_long_wave_radiation_flux_surface: float | None = pt.Field(
        dtype=pl.Float32,
        description="Downward long-wave radiation flux at surface. Note that this variable is all-null for lead time 0. Unit: W m-2.",
        ge=0,
        le=1500,  # Max in 2 years of ECMWF ENS = 445
    )

    downward_short_wave_radiation_flux_surface: float | None = pt.Field(
        dtype=pl.Float32,
        description="Downward short-wave (solar) radiation flux at surface. Note that this variable is all-null for lead time 0. Unit: W m-2.",
        ge=0,
        le=1500,  # Max in 2 years of ECMWF ENS = 892
    )

    precipitation_surface: float | None = pt.Field(
        dtype=pl.Float32,
        description="Total precipitation rate at surface, de-accumulated by Dynamical. Note that this variable is all-null for lead time 0. Unit: kg m-2 s-1.",
        ge=0,
        le=0.01,  # Max in 2 years of ECMWF ENS = 0.006
    )

    categorical_precipitation_type_surface: int | None = pt.Field(
        dtype=pl.UInt8,
        description=(
            "This field is always NaN for init_times on and before 2024-11-13. Derived from"
            " ECMWF's `ptype` field, which was added to ECMWF's Open Data dataset when IFS Cycle"
            " 49r1 was released in Nov 2024. See https://codes.ecmwf.int/grib/param-db/260015"
            " 0=No precipitation; 1=Rain; 2=Thunderstorm; 3=Freezing rain; 4=Mixed/ice;"
            " 5=Snow; 6=Wet snow; 7=Mixture of rain and snow; 8=Ice pellets; 9=Graupel;"
            " 10=Hail; 11=Drizzle; 12=Freezing drizzle; 13=Hail (less than 5 mm);"
            " 14=Hail (greater than or equal to 5 mm);"
            " 15-191=Reserved; 192-254=Reserved for local use; 255=Missing"
        ),
    )

    # ClassVars: excluded from Patito/Pydantic model fields so they're not treated as data columns.
    categorical_var_names: ClassVar[frozenset[str]] = frozenset(
        {"categorical_precipitation_type_surface"}
    )

    # Columns that aren't NWP variables:
    _non_var_column_names: ClassVar[frozenset[str]] = frozenset(
        {"nwp_model_id", "init_time", "valid_time", "ensemble_member", "h3_index"}
    )

    @classmethod
    def all_weather_var_names(cls) -> frozenset[str]:
        """All meteorological variable field names (continuous + categorical)."""
        return frozenset(cls.model_fields) - cls._non_var_column_names

    @classmethod
    def continuous_var_names(cls) -> frozenset[str]:
        """Meteorological variable field names suitable for linear interpolation."""
        return cls.all_weather_var_names() - cls.categorical_var_names

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

        cls._check_nulls_from_second_forecast_step_onwards(validated_df)
        cls._check_unique(validated_df)
        cls._check_variables_that_were_introduced_after_start_of_dataset(validated_df)
        return validated_df

    @classmethod
    def _check_nulls_from_second_forecast_step_onwards(cls, dataframe: pt.DataFrame[Self]) -> None:
        cols_to_check = [
            "precipitation_surface",
            "downward_short_wave_radiation_flux_surface",
            "downward_long_wave_radiation_flux_surface",
        ]

        second_step_onwards = dataframe.filter(pl.col("valid_time") > pl.col("init_time"))

        for col in cols_to_check:
            has_nulls = second_step_onwards[col].is_null().any()
            if has_nulls:
                raise ValueError(
                    f"Column '{col}' contains has null values from the second forecast "
                    "step onwards. These variables are only allowed to be null for the first "
                    "forecast step (lead time 0)."
                )

    @classmethod
    def _check_unique(cls, dataframe: pt.DataFrame[Self]) -> None:
        if (
            dataframe.select(["init_time", "valid_time", "ensemble_member", "h3_index"])
            .is_duplicated()
            .any()
        ):
            raise ValueError(
                "Duplicate entries found for (init_time, valid_time, ensemble_member, h3_index)."
            )

    @classmethod
    def _check_variables_that_were_introduced_after_start_of_dataset(
        cls, dataframe: pt.DataFrame[Self]
    ) -> None:
        """Check that `categorical_precipitation_type_surface` is all-null when
        init_time <= 2024-11-13, and is never null afterwards.

        ECMWF only introduced `ptype` into their public data on 2024-11-14.
        """
        threshold_date = datetime(2024, 11, 13, tzinfo=timezone.utc)

        # Partition the dataframe based on the threshold date
        partition_col = "is_before_or_on_threshold"
        partitioned_df = dataframe.with_columns(
            (pl.col("init_time") <= threshold_date).alias(partition_col)
        )
        partitions = partitioned_df.partition_by(partition_col, as_dict=True)

        empty_df = pl.DataFrame(schema=dataframe.schema)
        before_or_on = partitions.get((True,), empty_df)
        after = partitions.get((False,), empty_df)

        # Check before or on threshold
        if not before_or_on["categorical_precipitation_type_surface"].is_null().all():
            raise ValueError(
                "categorical_precipitation_type_surface must be all null for "
                "init_time <= 2024-11-13"
            )

        # Check after threshold
        if after["categorical_precipitation_type_surface"].is_null().any():
            raise ValueError(
                "categorical_precipitation_type_surface must not be null for init_time > 2024-11-13"
            )

    @classmethod
    def scan_delta(
        cls,
        path: str | Path = SETTINGS.nwp_data_path,
        storage_options: ObjectStoreOptions = SETTINGS.storage_options,
    ) -> pt.LazyFrame[Self]:
        """Lazily scan the NWP Delta table, typed and cast to this contract's dtypes.

        The table stores physical-unit `Float32` directly (see `delta_store.nwp.write_nwp`),
        so no rescale step is needed.

        Args:
            path: Path or URI of the ``nwp`` Delta table.
            storage_options: delta-rs object-store options (credentials/endpoint) for a remote
                ``path``; defaults to the ``SETTINGS`` singleton's options (empty for a local
                ``path``). Read-only — never mutated here (shared mutable default).
        """
        return (
            pt.LazyFrame.from_existing(
                pl.scan_delta(path, storage_options=typeddict_to_dict(storage_options))
            )
            .set_model(cls)
            .cast()
        )
