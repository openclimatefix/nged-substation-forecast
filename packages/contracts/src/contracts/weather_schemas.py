from collections.abc import Sequence
from datetime import datetime, timezone
from enum import StrEnum, auto
from pathlib import Path
from typing import ClassVar, Final, Literal, Self, overload

import patito as pt
import polars as pl

from contracts.common import validate_schema
from contracts.settings import PROJECT_ROOT, Settings

from .common import UTC_DATETIME_DTYPE

# TODO: These paths should be moved to `contracts.settings`.
_METADATA_PATH: Final[Path] = PROJECT_ROOT / "metadata"
_NWP_METADATA_CSV_PATH: Final[Path] = _METADATA_PATH / "nwp_metadata.csv"
_SCALING_PARAMS_FOR_ECMWF_ENS_0_25_DEGREE_CSV_PATH: Final[Path] = (
    _METADATA_PATH / "scaling_params_for_ecmwf_ens_0_25_degree.csv"
)


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
        description="The primary key for joining with NWP data.",
        unique=True,
    )
    """Primary key for joining with NWP data (e.g. `'ECMWF_ENS_0_25_degree'`)."""

    provider: str = pt.Field(dtype=pl.Enum(["ECMWF"]))
    """NWP data provider (currently always `'ECMWF'`)."""

    h3_resolution: int = pt.Field(dtype=pl.Int8)
    """H3 spatial resolution used for this NWP model's grid."""

    is_ensemble: bool
    """Whether this NWP model produces ensemble forecasts."""

    @classmethod
    def load(cls, csv_path: Path = _NWP_METADATA_CSV_PATH) -> pt.DataFrame[Self]:
        """Load NWP metadata from a static CSV file."""
        df = pl.read_csv(csv_path)
        # Patito's .cast() will handle the conversion to the Enum type defined in the model
        return pt.DataFrame(df).set_model(cls).cast().validate()


_NULL_FOR_LEAD_TIME_0 = "Note that this variable is all-null for lead time 0. "


class _NwpBase(pt.Model):
    """Weather data schema for NWP forecasts, using"""

    nwp_model_id: str = pt.Field(
        dtype=NWP_MODEL_ID_DTYPE,
        description="The primary key for joining with NwpMetaData.",
    )
    """Primary key for joining with `NwpMetaData` (e.g. `'ECMWF_ENS_0_25_degree'`)."""

    init_time: datetime = pt.Field(dtype=UTC_DATETIME_DTYPE)
    """When the NWP model run was initialised."""

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
    """Target time. Most variables are instantaneous; precipitation/radiation are period-ending rates (Dynamical de-accumulates them)."""

    ensemble_member: int = pt.Field(dtype=pl.UInt8)
    """Ensemble member index (0-based)."""

    h3_index: int = pt.Field(
        dtype=pl.UInt64,
        description="The H3 resolution for the nwp_model_id is stored in NwpMetaData.",
    )
    """H3 cell index; resolution is stored in `NwpMetaData.h3_resolution`."""

    # Categorical variables
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
    """ECMWF `ptype`: 0=None, 1=Rain, 2=Thunderstorm, 5=Snow, 10=Hail, etc. Null before 2024-11-14."""

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


_WIND_SPEED_DTYPE = pt.Field(
    dtype=pl.Float32,
    description="Meters per second",
    ge=0,
    le=200,  # Gemini says the highest non-tornadic surface wind speed recorded was 113 m/s
)
_WIND_DIRECTION_DTYPE = pt.Field(
    dtype=pl.Float32,
    description="The angle where the wind is coming *from*. Degrees. 0° is North; 90° is East",
    ge=0,
    le=360,
)


class NwpInMemory(_NwpBase):
    """Variables stored as Float32 in memory.

    NWP data is first converted to NwpInMemory when ingested from Dynamical.
    """

    temperature_2m: float = pt.Field(dtype=pl.Float32, description="Degrees C", ge=-100, le=100)
    """Air temperature at 2 m above ground (°C)."""

    dew_point_temperature_2m: float = pt.Field(
        dtype=pl.Float32, description="Degrees C", ge=-100, le=100
    )
    """Dew-point temperature at 2 m (°C)."""

    wind_speed_10m: float = _WIND_SPEED_DTYPE
    """Wind speed at 10 m (m/s)."""

    wind_direction_10m: float = _WIND_DIRECTION_DTYPE
    """Wind direction at 10 m — degrees the wind is coming *from* (0° = N, 90° = E)."""

    wind_speed_100m: float = _WIND_SPEED_DTYPE
    """Wind speed at 100 m (m/s)."""

    wind_direction_100m: float = _WIND_DIRECTION_DTYPE
    """Wind direction at 100 m — degrees the wind is coming *from* (0° = N, 90° = E)."""

    pressure_surface: float = pt.Field(
        dtype=pl.Float32,
        description="Pa",
        ge=0,
        le=200_000,  # Max in 2 years of ECMWF ENS = 105_760
    )
    """Surface pressure (Pa)."""

    pressure_reduced_to_mean_sea_level: float = pt.Field(
        dtype=pl.Float32,
        description="Pa",
        ge=0,
        le=200_000,  # Max in 2 years of ECMWF ENS = 105_727
    )
    """Mean sea-level pressure (Pa)."""

    geopotential_height_500hpa: float = pt.Field(
        dtype=pl.Float32,
        description="m",
        ge=0,
        le=10_000,  # Max in 2 years of ECMWF ENS = 6_030
    )
    """Geopotential height of the 500 hPa pressure surface (m)."""

    # Precipitation and radiation variables are null for the first forecast step (lead time 0) in
    # ECMWF ENS. Also note that, whilst these variables accumulate over forecast steps in ECMWF's
    # raw forecasts, we get ECMWF ENS from Dynamical.org, and Dynamical.org de-accumulates these
    # values before we receive them. So these are true _rates_.
    downward_long_wave_radiation_flux_surface: float | None = pt.Field(
        dtype=pl.Float32,
        description=_NULL_FOR_LEAD_TIME_0 + "Unit: W m-2",
        ge=0,
        le=1500,  # Max in 2 years of ECMWF ENS = 445
    )
    """Downward long-wave radiation flux at surface (W m⁻²); null at lead time 0."""

    downward_short_wave_radiation_flux_surface: float | None = pt.Field(
        dtype=pl.Float32,
        description=_NULL_FOR_LEAD_TIME_0 + "Unit: W m-2",
        ge=0,
        le=1500,  # Max in 2 years of ECMWF ENS = 892
    )
    """Downward short-wave (solar) radiation flux at surface (W m⁻²); null at lead time 0."""

    precipitation_surface: float | None = pt.Field(
        dtype=pl.Float32,
        description=_NULL_FOR_LEAD_TIME_0 + "Unit: kg m-2 s-1",
        ge=0,
        le=0.01,  # Max in 2 years of ECMWF ENS = 0.006
    )
    """Total precipitation rate at surface (kg m⁻² s⁻¹, de-accumulated by Dynamical); null at lead time 0."""

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


_NWP_ON_DISK_DTYPE: Final[pl.datatypes.DataTypeClass] = pl.Int16
_NWP_ON_DISK_MAX_INT_VALUE: Final[int] = 2**12 - 1  # Use 12 bits per value


class NwpOnDisk(_NwpBase):
    """Variables stored as integers on disk to reduce storage requirements."""

    temperature_2m: int = pt.Field(dtype=_NWP_ON_DISK_DTYPE)
    dew_point_temperature_2m: int = pt.Field(dtype=_NWP_ON_DISK_DTYPE)
    wind_speed_10m: int = pt.Field(dtype=_NWP_ON_DISK_DTYPE)
    wind_direction_10m: int = pt.Field(dtype=_NWP_ON_DISK_DTYPE)
    wind_speed_100m: int = pt.Field(dtype=_NWP_ON_DISK_DTYPE)
    wind_direction_100m: int = pt.Field(dtype=_NWP_ON_DISK_DTYPE)
    pressure_surface: int = pt.Field(dtype=_NWP_ON_DISK_DTYPE)
    pressure_reduced_to_mean_sea_level: int = pt.Field(dtype=_NWP_ON_DISK_DTYPE)
    geopotential_height_500hpa: int = pt.Field(dtype=_NWP_ON_DISK_DTYPE)

    # Precipitation and radiation variables are null for the first forecast step (lead time 0) in
    # ECMWF ENS. Also note that, whilst these variables accumulate over forecast steps in ECMWF's
    # raw forecasts, we get ECMWF ENS from Dynamical.org, and Dynamical.org de-accumulates these
    # values before we receive them. So these are true _rates_.
    downward_long_wave_radiation_flux_surface: int | None = pt.Field(dtype=_NWP_ON_DISK_DTYPE)
    downward_short_wave_radiation_flux_surface: int | None = pt.Field(dtype=_NWP_ON_DISK_DTYPE)
    precipitation_surface: int | None = pt.Field(dtype=_NWP_ON_DISK_DTYPE)

    @classmethod
    def from_nwp_in_memory(
        cls,
        nwp_in_memory: pt.DataFrame[NwpInMemory],
        scaling_params: pt.DataFrame[NwpScalingParams],
    ) -> pt.DataFrame[Self]:
        """Scale numeric columns to integer representation based on scaling parameters.

        Storing NWPs as integers on disk significantly reduces the storage requirements.

        For example, if we use 12 bits per value, then we will encode each value as an integer
        in the range [0, 4,096) (where 4,096 == 2^12).

        Args:
            nwp_in_memory: DataFrame or LazyFrame with float32 columns.
            scaling_params: DataFrame with col_name, buffered_min, buffered_max, buffered_range.

        Returns:
            DataFrame or LazyFrame with rescaled integer columns.
        """

        exprs = []
        for row in scaling_params.to_dicts():
            col_name = row["col_name"]
            if col_name not in nwp_in_memory.columns:
                raise ValueError(f"Column {col_name} is not in `nwp_in_memory`!")

            buffered_min = row["buffered_min"]
            buffered_max = row["buffered_max"]
            buffered_range = row["buffered_range"]

            clipped_col = pl.col(col_name).clip(lower_bound=buffered_min, upper_bound=buffered_max)

            expr = (
                (((clipped_col - buffered_min) / buffered_range) * _NWP_ON_DISK_MAX_INT_VALUE)
                .round()
                .cast(_NWP_ON_DISK_DTYPE)
            )

            exprs.append(expr)

        nwp_on_disk = nwp_in_memory.with_columns(exprs)
        return cls.validate(nwp_on_disk)

    # overload to indicate that you get a DataFrame back if you feed in a DataFrame
    @overload
    @classmethod
    def to_nwp_in_memory(
        cls,
        nwp_on_disk: pt.DataFrame[Self],
        scaling_params: pt.DataFrame[NwpScalingParams] | None = None,
    ) -> pt.DataFrame[NwpInMemory]: ...

    # overload to indicate that you get a LazyFrame back if you feed in a LazyFrame
    @overload
    @classmethod
    def to_nwp_in_memory(
        cls,
        nwp_on_disk: pt.LazyFrame[Self],
        scaling_params: pt.DataFrame[NwpScalingParams] | None = None,
    ) -> pt.LazyFrame[NwpInMemory]: ...

    @classmethod
    def to_nwp_in_memory(
        cls,
        nwp_on_disk: pt.DataFrame[Self] | pt.LazyFrame[Self],
        scaling_params: pt.DataFrame[NwpScalingParams] | None = None,
    ) -> pt.DataFrame[NwpInMemory] | pt.LazyFrame[NwpInMemory]:
        """Scale integer columns to floating point representations in physical units."""
        scaling_params = NwpScalingParams.load() if scaling_params is None else scaling_params

        exprs = []
        for row in scaling_params.to_dicts():
            col_name = row["col_name"]
            if col_name not in nwp_on_disk.columns:
                raise ValueError(f"Column {col_name} is not in `nwp_on_disk`!")

            buffered_min = row["buffered_min"]
            buffered_range = row["buffered_range"]

            expr = (
                pl.col(col_name).cast(pl.Float32) / _NWP_ON_DISK_MAX_INT_VALUE
            ) * buffered_range + buffered_min

            exprs.append(expr)

        # The `ignore[unresolved-attribute]` is necessary because `ty` doesn't believe that
        # `.set_model` is defined on `pt.LazyFrame`. But `pt.LazyFrame.set_model()` DOES exist!
        nwp_in_memory = nwp_on_disk.with_columns(exprs).set_model(NwpInMemory)  # ty: ignore[unresolved-attribute]

        # Use use `validate_schema` not `validate` because `validate` won't work on LazyFrames.
        # And we want this function to be compatible with LazyFrames.
        validate_schema(NwpInMemory, nwp_in_memory)
        return nwp_in_memory

    @classmethod
    def scan_delta(cls, path: Path = SETTINGS.nwp_data_path) -> pt.LazyFrame[Self]:
        df = pl.scan_delta(path)
        return pt.LazyFrame.from_existing(df).set_model(cls).cast()


class NwpScalingParams(pt.Model):
    """Schema for weather variable scaling parameters.

    Used when scaling between physical units (e.g. degrees C) and their integer representations."""

    col_name: str = pt.Field(
        dtype=pl.Enum(  # Note this this does NOT include the categorical variables because we don't scale categorical variables!
            [
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
            ]
        ),
        unique=True,
    )

    # The minimum from the actual values, minus a small buffer.
    buffered_min: float = pt.Field(dtype=pl.Float32)

    # The range from the actual values, plus a small buffer.
    buffered_range: float = pt.Field(dtype=pl.Float32)

    # The maximum from the actual values, plus a small buffer.
    buffered_max: float = pt.Field(dtype=pl.Float32)

    @classmethod
    def load(
        cls, csv_path: Path = _SCALING_PARAMS_FOR_ECMWF_ENS_0_25_DEGREE_CSV_PATH
    ) -> pt.DataFrame[Self]:
        """Load scaling parameters from a CSV file."""
        return pt.DataFrame(pl.read_csv(csv_path)).set_model(cls).drop().cast().validate()
