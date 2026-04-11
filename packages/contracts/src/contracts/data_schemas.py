"""Data schemas for the NGED substation forecast project."""

from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Literal, cast

import patito as pt
import polars as pl
from pydantic import BaseModel

if TYPE_CHECKING:
    import pandas as pd


# Define our standard datetime type for all schemas
UTC_DATETIME_DTYPE = pl.Datetime(time_unit="us", time_zone="UTC")


class PowerTimeSeries(pt.Model):
    time_series_id: int = pt.Field(dtype=pl.Int32)
    # period_end_time represents the end of the 30-minute settlement period.
    period_end_time: datetime = pt.Field(
        dtype=UTC_DATETIME_DTYPE,
        description="The end time of the 30-minute period. Note that all the JSON time series data is already 30-minutely.",
    )
    # Constrain power values to a physically realistic range of [-1000, 1000] MW/MVA to prevent extreme outliers from affecting downstream models.
    power: float | None = pt.Field(dtype=pl.Float32, ge=-1000, le=1000)

    @classmethod
    def validate(
        cls,
        dataframe: pl.DataFrame | "pd.DataFrame",
        columns: Sequence[str] | None = None,
        allow_missing_columns: bool = False,
        allow_superfluous_columns: bool = False,
        drop_superfluous_columns: bool = False,
    ) -> pt.DataFrame["PowerTimeSeries"]:
        """Validate the given dataframe, ensuring period_end_time is at :00 or :30."""
        validated_df = super().validate(
            dataframe=dataframe,
            columns=columns,
            allow_missing_columns=allow_missing_columns,
            allow_superfluous_columns=allow_superfluous_columns,
            drop_superfluous_columns=drop_superfluous_columns,
        )

        # Validate period_end_time is at :00 or :30
        minutes = validated_df["period_end_time"].dt.minute()
        if not minutes.is_in([0, 30]).all():
            raise ValueError(
                "period_end_time must be at the top or bottom of the hour (minute 00 or 30)."
            )

        return cast(pt.DataFrame["PowerTimeSeries"], validated_df)


class TimeSeriesMetadata(pt.Model):
    """Metadata for a time series, joining location data with live telemetry info."""

    time_series_id: int = pt.Field(
        dtype=pl.Int32,
        unique=True,
        description=(
            "Provided by NGED. This is the primary key for identifying the time series."
            " There's _almost_ a one-to-one mapping between time_series_id and the"
            " asset ID, so you can think of time_series_id as the asset ID"
            " (where an 'asset' is a physical asset like a substation or PV farm)"
        ),
    )
    time_series_name: str = pt.Field(
        dtype=pl.String,
        examples=[
            "ALFORD 33 11kV S STN",
            "BAMBERS FARM WIND GENERATION MABLETHORPE 33kV S ST",
            "Leverton Solar Park",
        ],
    )
    time_series_type: Literal[
        "BESS",  # Battery energy storage system. Present in trial area
        "Biofuel",  # Present in trial area
        "CHP",
        "Data Centre",
        # In the trial area, "Disaggregated Demand" is exclusively associated with "Primary" substations,
        # and all "Primary" substations in the trial area have their TimeSeriesType set to "Disaggregated Demand".
        # "Disaggregated Demand" indicates that NGED have already removed any metered generation connected to that primary.
        "Disaggregated Demand",  # Present in trial area.
        "Energy from Waste",
        "EV Charging",
        "Geothermal",
        "Hydro",
        "Hydrogen Electrolysis",
        "Industrial Demand",
        "Mixed (Demand)",
        "Mixed (Generation)",
        "Other (Demand)",
        "Other (Generation)",  # Present in trial area
        "Other (Storage)",
        "Peaking Plant",
        "PV",  # Present in trial area
        "Rail",
        "Raw Flow",  # Present in trial area
        "Synchronous Condenser",
        "Wind",  # Present in trial area
    ] = pt.Field(dtype=pl.String)
    units: Literal["MVA", "MW"] = pt.Field(dtype=pl.String)
    licence_area: Literal["EMids"] = pt.Field(dtype=pl.String)
    substation_number: int = pt.Field(
        dtype=pl.Int32,
        gt=0,
        lt=1_000_000,
        description="For customer time series, substation_number is the substation to which that customer is connected.",
    )
    substation_type: Literal["BSP", "EHV Customer", "GSP", "HV Customer", "Primary"] = pt.Field(
        dtype=pl.Categorical
    )
    latitude: float = pt.Field(
        dtype=pl.Float32,
        ge=49,
        le=61,  # UK latitude range
        description=(
            "For customer time series, the latitude and longitude give the location of the"
            " _substation_, not the customer's site."
        ),
    )
    longitude: float = pt.Field(
        dtype=pl.Float32,
        ge=-9,
        le=2,  # UK longitude range
        description=(
            "For customer time series, the latitude and longitude give the location of the"
            " _substation_, not the customer's site."
        ),
    )
    information: str | None = pt.Field(
        dtype=pl.String,
        allow_missing=True,
        description="Always None in the trial area",
    )
    area_wkt: str | None = pt.Field(
        dtype=pl.String,
        allow_missing=True,
        # Maps to the nested Area.WKT field in the JSON data.
        description=(
            "For customer sites, the area, where present, refers to the area covered by the generator itself."
            " NGED don’t have polygons for the customer sites, though NGED hope to add that in the future."
        ),
    )
    area_center_lat: float | None = pt.Field(
        dtype=pl.Float32,
        allow_missing=True,
        description=(
            "For customer sites, the area, where present, refers to the area covered by the generator itself."
        ),
    )
    area_center_lon: float | None = pt.Field(
        dtype=pl.Float32,
        allow_missing=True,
        description=(
            "For customer sites, the area, where present, refers to the area covered by the generator itself."
        ),
    )
    h3_res_5: int = pt.Field(
        dtype=pl.UInt64,
        description="H3 spatial index at resolution 5.",
    )


class PowerForecast(pt.Model):
    """Forecast data schema for deterministic ensemble forecasts."""

    valid_time: datetime = pt.Field(dtype=UTC_DATETIME_DTYPE)
    # time_series_id is an int32 for memory efficiency and consistency with substation numbers.
    time_series_id: int = pt.Field(dtype=pl.Int32)
    ensemble_member: int = pt.Field(dtype=pl.UInt8)

    # The datetime that the underlying weather forecast was initialised.
    nwp_init_time: datetime = pt.Field(dtype=UTC_DATETIME_DTYPE)

    # The hour of the day that the weather forecast was initialised (0, 6, 12, 18).
    nwp_init_hour: int = pt.Field(dtype=pl.Int32)

    # The number of hours between the weather forecast initialisation and the valid time.
    lead_time_hours: float = pt.Field(dtype=pl.Float32)

    # Identifier for our ML-based power forecasting model.
    # This is manually specified in `hydra_schemas.ModelConfig.power_fcst_model_name`.
    power_fcst_model_name: str = pt.Field(dtype=pl.Categorical)

    # The datetime that the power forecast was initialised.
    power_fcst_init_time: datetime = pt.Field(dtype=UTC_DATETIME_DTYPE)

    # Year and month of the power forecast initialisation (for partitioning).
    power_fcst_init_year_month: str = pt.Field(dtype=pl.String)

    # The power forecast itself in units of MW (active power) or MVA (apparent power).
    # The unit is defined in the `TimeSeriesMetadata` for this `time_series_id`.
    power_fcst: float = pt.Field(dtype=pl.Float32)


class ScalingParams(pt.Model):
    """Schema for weather variable scaling parameters.

    Used when scaling between physical units (e.g. degrees C) and their unsigned 8-bit integer
    (uint8) representations. uint8 represents integers in the range [0, 255]."""

    col_name: str = pt.Field(dtype=pl.String)

    # The minimum from the actual values, minus a small buffer.
    buffered_min: float = pt.Field(dtype=pl.Float32)

    # The range from the actual values, plus a small buffer.
    buffered_range: float = pt.Field(dtype=pl.Float32)

    # The maximum from the actual values, plus a small buffer.
    buffered_max: float = pt.Field(dtype=pl.Float32)


class InferenceParams(BaseModel):
    """Parameters for ML model inference."""

    # The time that we create our power forecast. This might be called `t0` in some other OCF
    # projects. When running backtests, we cannot use any NWPs available after `forecast_time`
    # (i.e. init_time + delay > forecast_time).
    forecast_time: datetime

    power_fcst_model_name: str | None = None


class Nwp(pt.Model):
    """Weather data schema for NWP forecasts."""

    init_time: datetime = pt.Field(dtype=UTC_DATETIME_DTYPE)
    valid_time: datetime = pt.Field(dtype=UTC_DATETIME_DTYPE)
    ensemble_member: int = pt.Field(dtype=pl.UInt8)
    h3_index: int = pt.Field(dtype=pl.UInt64)

    # Variables stored as Float32 in memory (descaled from uint8 on disk)
    temperature_2m: float = pt.Field(dtype=pl.Float32)
    dew_point_temperature_2m: float = pt.Field(dtype=pl.Float32)
    # WIND VECTOR COMPONENTS:
    # We store raw U and V components as Float32 to allow physically realistic
    # linear interpolation in the forecasting pipeline, avoiding the "phantom high wind"
    # artifacts caused by interpolating speed/direction or circular variables.
    wind_u_10m: float = pt.Field(dtype=pl.Float32)
    wind_v_10m: float = pt.Field(dtype=pl.Float32)
    wind_u_100m: float = pt.Field(dtype=pl.Float32)
    wind_v_100m: float = pt.Field(dtype=pl.Float32)
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
    """Centralized constants for NWP column names.

    Used to prevent typos and ensure consistency across feature engineering and model training.
    """

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
    The scaling formula is:
        uint8_value = round(((physical_value - buffered_min) / buffered_range) * 255).

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
    categorical_precipitation_type_surface: int = pt.Field(dtype=pl.UInt8)


class XGBoostInputFeatures(pt.Model):
    """Final joined dataset ready for XGBoost.

    Weather features are kept in their physical units (e.g., degrees Celsius, m/s)
    to ensure precision during interpolation and feature engineering.
    """

    valid_time: datetime = pt.Field(dtype=UTC_DATETIME_DTYPE)
    # time_series_id is an int32 for memory efficiency and consistency with substation numbers.
    time_series_id: int = pt.Field(dtype=pl.Int32)
    time_series_type: str = pt.Field(dtype=pl.Categorical)
    ensemble_member: int | None = pt.Field(dtype=pl.UInt8, allow_missing=True)
    power: float = pt.Field(dtype=pl.Float32)
    lead_time_hours: float = pt.Field(dtype=pl.Float32)
    lead_time_days: float = pt.Field(dtype=pl.Float32)
    nwp_init_hour: int = pt.Field(dtype=pl.Int32)

    # Power lags
    latest_available_weekly_power_lag: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)

    # Weather features
    temperature_2m: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    dew_point_temperature_2m: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    # PHYSICAL WIND FEATURES:
    # These are calculated from interpolated U/V components in the forecasting
    # pipeline, ensuring physically realistic wind speed and direction.
    wind_speed_10m: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    wind_direction_10m: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    wind_speed_100m: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    wind_direction_100m: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    pressure_surface: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    pressure_reduced_to_mean_sea_level: float | None = pt.Field(
        dtype=pl.Float32, allow_missing=True
    )
    geopotential_height_500hpa: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    downward_long_wave_radiation_flux_surface: float | None = pt.Field(
        dtype=pl.Float32, allow_missing=True
    )
    downward_short_wave_radiation_flux_surface: float | None = pt.Field(
        dtype=pl.Float32, allow_missing=True
    )
    precipitation_surface: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    categorical_precipitation_type_surface: int | None = pt.Field(
        dtype=pl.UInt8, allow_missing=True
    )

    # Physical features
    windchill: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)

    # Weather lags/trends
    temperature_2m_lag_7d: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    downward_short_wave_radiation_flux_surface_lag_7d: float | None = pt.Field(
        dtype=pl.Float32, allow_missing=True
    )
    temperature_2m_lag_14d: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    downward_short_wave_radiation_flux_surface_lag_14d: float | None = pt.Field(
        dtype=pl.Float32, allow_missing=True
    )
    temperature_2m_6h_ago: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    temp_trend_6h: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)

    # Temporal features
    hour_sin: float = pt.Field(dtype=pl.Float32)
    hour_cos: float = pt.Field(dtype=pl.Float32)
    day_of_year_sin: float = pt.Field(dtype=pl.Float32)
    day_of_year_cos: float = pt.Field(dtype=pl.Float32)
    day_of_week: int = pt.Field(dtype=pl.Int8)


class H3GridWeights(pt.Model):
    """Schema for the pre-computed H3 grid weights.

    This contract defines the mapping between H3 hexagons and a regular latitude/longitude grid.
    It is used to ensure type safety when passing spatial mapping data from generic geospatial
    utilities (like `packages/geo`) to dataset-specific ingestion pipelines (like `packages/dynamical_data`).
    """

    h3_index: int = pt.Field(dtype=pl.UInt64)
    nwp_lat: float = pt.Field(dtype=pl.Float64, ge=-90, le=90)
    nwp_lng: float = pt.Field(dtype=pl.Float64, ge=-180, le=180)
    len: int = pt.Field(dtype=pl.UInt32)
    total: int = pt.Field(dtype=pl.UInt32)
    proportion: float = pt.Field(dtype=pl.Float64)


class Metrics(pt.Model):
    """Evaluation metrics for power forecasts."""

    time_series_id: int = pt.Field(dtype=pl.Int32)
    power_fcst_model_name: str = pt.Field(dtype=pl.Categorical)
    lead_time_hours: float = pt.Field(dtype=pl.Float32)
    mae: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    rmse: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    nmae: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
