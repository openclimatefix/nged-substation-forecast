"""Data schemas for the NGED substation forecast project."""

from collections.abc import Sequence
from datetime import datetime
from typing import ClassVar, Final, Self

import patito as pt
import polars as pl

from .common import UTC_DATETIME_DTYPE, _get_time_series_id_dtype


class PowerTimeSeries(pt.Model):
    time_series_id: int = _get_time_series_id_dtype()
    time: datetime = pt.Field(
        dtype=UTC_DATETIME_DTYPE,
        description=(
            "The end time of the 30-minute period."
            " Note that all the NGED JSON time series data is already 30-minutely."
        ),
    )
    power: float = pt.Field(
        dtype=pl.Float32,
        ge=-1000,
        le=1000,
        description=(
            "The average power in MW or MVA over the previous 30-minute period."
            " The unit is defined in the TimeSeriesMetadata for this time_series_id"
        ),
    )

    @classmethod
    def validate(  # ty: ignore[invalid-method-override]
        cls,
        dataframe: pl.DataFrame,
        columns: Sequence[str] | None = None,
        allow_missing_columns: bool = False,
        allow_superfluous_columns: bool = False,
        drop_superfluous_columns: bool = False,
    ) -> pt.DataFrame[Self]:
        """Validate the given dataframe, ensuring time is at :00 or :30 and uniqueness."""
        validated_df = super().validate(
            dataframe=dataframe,
            columns=columns,
            allow_missing_columns=allow_missing_columns,
            allow_superfluous_columns=allow_superfluous_columns,
            drop_superfluous_columns=drop_superfluous_columns,
        )

        # Validate time is at :00 or :30
        minutes = validated_df["time"].dt.minute()
        if not minutes.is_in([0, 30]).all():
            raise ValueError("time must be at the top or bottom of the hour (minute 00 or 30).")

        # Validate uniqueness of (time_series_id, time)
        if validated_df.select(["time_series_id", "time"]).is_duplicated().any():
            raise ValueError("Duplicate entries found for (time_series_id, time).")

        # Validate the time_series_id column is sorted
        if not validated_df["time_series_id"].is_sorted():
            raise ValueError("time_series_id is not sorted!")

        # Validate the time column is sorted (within each time_series_id group)
        if (
            not validated_df.group_by("time_series_id")
            .agg(pl.col("time").diff().min() > 0)["time"]
            .all()
        ):
            raise ValueError("the `time` column is not sorted!")

        return validated_df

    # Define it as a ClassVar so Patito/Pydantic knows it's not a data field
    columns_to_sort_by: ClassVar[tuple[str, str]] = ("time_series_id", "time")


LIST_OF_TIME_SERIES_TYPES: Final[tuple[str, ...]] = (
    "BESS",  # Battery energy storage system. Present in trial area
    "Biofuel",  # Present in trial area
    "CHP",
    "Data Centre",
    # In the trial area, "Disaggregated Demand" is exclusively associated with "Primary" substations,
    # and all "Primary" substations in the trial area have their TimeSeriesType set to "Disaggregated Demand".
    # "Disaggregated Demand" indicates that NGED have already removed metered generation connected to that primary.
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
    "Raw Flow",  # Present in trial area. Used for BSP and GSP substations.
    "Synchronous Condenser",
    "Wind",  # Present in trial area
)


class TimeSeriesMetadata(pt.Model):
    time_series_id: int = _get_time_series_id_dtype(unique=True)
    time_series_name: str = pt.Field(
        dtype=pl.String,
        examples=[
            "ALFORD 33 11kV S STN",
            "BAMBERS FARM WIND GENERATION MABLETHORPE 33kV S ST",
            "Leverton Solar Park",
        ],
    )
    time_series_type: str = pt.Field(dtype=pl.Enum(LIST_OF_TIME_SERIES_TYPES))
    units: str = pt.Field(dtype=pl.Enum(["MW", "MVA"]))
    licence_area: str = pt.Field(dtype=pl.Enum(["EMids"]))
    substation_number: int = pt.Field(
        dtype=pl.Int32,
        gt=0,
        lt=1_000_000,
        description="For customer time series, substation_number is the substation to which that customer is connected.",
    )
    substation_type: str = pt.Field(
        dtype=pl.Enum(["BSP", "EHV Customer", "GSP", "HV Customer", "Primary"])
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
            "In the trial, the only time series to have area_wkt are the Primary substations."
            " NGED don’t have polygons for the customer sites, though NGED hope to add that in the future."
            " If/when NGED publish polygons for customer sites, the area, where present, refers to the"
            " area covered by the generator itself."
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
        description="H3 discrete spatial index at resolution 5.",
    )


class PowerForecast(pt.Model):
    """Forecast data schema for deterministic ensemble forecasts."""

    valid_time: datetime = pt.Field(dtype=UTC_DATETIME_DTYPE)
    time_series_id: int = _get_time_series_id_dtype()
    ensemble_member: int = pt.Field(dtype=pl.Int8)
    ml_flow_experiment_id: int = pt.Field(dtype=pl.Int32)

    nwp_init_time: datetime = pt.Field(
        dtype=UTC_DATETIME_DTYPE,
        description="The datetime that the underlying weather forecast was initialised.",
    )

    power_fcst_model_name: str = pt.Field(
        dtype=pl.Categorical,
        description=(
            "Identifier for our ML-based power forecasting model."
            " This is manually specified in `hydra_schemas.ModelConfig.power_fcst_model_name`."
        ),
    )

    power_fcst_model_version: int = pt.Field(dtype=pl.Int16)

    power_fcst_init_time: datetime = pt.Field(
        dtype=UTC_DATETIME_DTYPE,
        description=(
            "The datetime that the power forecast was initialised."
            " This might be called `t0` in some other OCF projects."
        ),
    )

    power_fcst: float = pt.Field(
        dtype=pl.Float32,
        description=(
            "The power forecast itself in units of MW (active power) or MVA (apparent power)."
            " The unit is defined in the `TimeSeriesMetadata` for this `time_series_id`."
            """ Positive values mean "power sent to NGED's grid","""
            """ and negative values mean "power drawn from NGED's grid"."""
        ),
    )
