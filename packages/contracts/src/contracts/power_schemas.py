"""Data schemas for the NGED substation forecast project."""

from collections.abc import Sequence
from datetime import datetime
from typing import ClassVar, Final, Literal, Self

import patito as pt
import polars as pl

from .common import UTC_DATETIME_DTYPE, _get_time_series_id_dtype


class PowerTimeSeries(pt.Model):
    time_series_id: int = _get_time_series_id_dtype()
    """NGED-provided primary key identifying the substation or asset."""

    time: datetime = pt.Field(
        dtype=UTC_DATETIME_DTYPE,
        description=(
            "The end time of the 30-minute period."
            " Note that all the NGED JSON time series data is already 30-minutely."
        ),
    )
    """End time of the 30-minute observation period (all NGED data is already half-hourly)."""

    power: float = pt.Field(
        dtype=pl.Float32,
        ge=-1000,
        le=1000,
        description=(
            "The average power in MW or MVA over the previous 30-minute period."
            " The unit is defined in the TimeSeriesMetadata for this time_series_id"
        ),
    )
    """Average power (MW or MVA) over the preceding 30-minute period. Unit defined in `TimeSeriesMetadata`."""

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
    """NGED-provided primary key identifying the substation or asset (unique within this table)."""

    time_series_name: str = pt.Field(
        dtype=pl.String,
        examples=[
            "ALFORD 33 11kV S STN",
            "BAMBERS FARM WIND GENERATION MABLETHORPE 33kV S ST",
            "Leverton Solar Park",
        ],
    )
    """Human-readable name for the substation or asset."""

    time_series_type: str = pt.Field(dtype=pl.Enum(LIST_OF_TIME_SERIES_TYPES))
    """Asset category (e.g. `’PV’`, `’Wind’`, `’Disaggregated Demand’`). See `LIST_OF_TIME_SERIES_TYPES`."""

    units: str = pt.Field(dtype=pl.Enum(["MW", "MVA"]))
    """Power unit for this time series: `’MW’` (active power) or `’MVA’` (apparent power)."""

    licence_area: str = pt.Field(dtype=pl.Enum(["EMids"]))
    """NGED licence area (currently always `’EMids’`)."""

    substation_number: int = pt.Field(
        dtype=pl.Int32,
        gt=0,
        lt=1_000_000,
        description="Perhaps surprisingly, each customer meter in the NGED trial area has its own substation_number.",
    )
    """Each customer meter has its own substation_number (not one per physical substation)."""

    substation_type: str = pt.Field(
        dtype=pl.Enum(["BSP", "EHV Customer", "GSP", "HV Customer", "Primary"])
    )
    """Substation voltage level / role: BSP, EHV Customer, GSP, HV Customer, or Primary."""

    latitude: float = pt.Field(
        dtype=pl.Float32,
        ge=49,
        le=61,  # UK latitude range
        description=(
            "For customer time series, the latitude and longitude give the location of the"
            " _substation_, not the customer’s site."
        ),
    )
    """Latitude of the substation (not the customer site) in decimal degrees."""

    longitude: float = pt.Field(
        dtype=pl.Float32,
        ge=-9,
        le=2,  # UK longitude range
        description=(
            "For customer time series, the latitude and longitude give the location of the"
            " _substation_, not the customer’s site."
        ),
    )
    """Longitude of the substation (not the customer site) in decimal degrees."""

    information: str | None = pt.Field(
        dtype=pl.String,
        allow_missing=True,
        description="Always None in the trial area",
    )
    """Free-text NGED notes field; always null in the V1 trial area."""

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
    """WKT polygon for the asset’s area; only Primary substations have this in the trial."""

    area_center_lat: float | None = pt.Field(
        dtype=pl.Float32,
        allow_missing=True,
        description=(
            "For customer sites, the area, where present, refers to the area covered by the generator itself."
        ),
    )
    """Centroid latitude of the area polygon (generator footprint for customer sites)."""

    area_center_lon: float | None = pt.Field(
        dtype=pl.Float32,
        allow_missing=True,
        description=(
            "For customer sites, the area, where present, refers to the area covered by the generator itself."
        ),
    )
    """Centroid longitude of the area polygon (generator footprint for customer sites)."""

    h3_res_5: int = pt.Field(
        dtype=pl.UInt64,
        description="H3 discrete spatial index at resolution 5.",
    )
    """H3 discrete spatial index at resolution 5."""


#: Fold identifier for ``PowerForecast.fold_id``.
#: Each fold validates on one whole year.
#: Each validation year in the CV protocol gets a string label matching that year.
#: ``"live"`` denotes a production forecast (no CV fold).
#: Extend this Literal as new CV epochs are added.
FoldId = Literal["live", "2022", "2023", "2024", "2025", "2026"]


class PowerForecast(pt.Model):
    """Forecast data schema for deterministic ensemble forecasts.

    Internal vs delivered schema (Milestone 1 report Table 1, p.28): the columns
    ``experiment_name``, ``fold_id``, and ``ml_flow_experiment_id`` are INTERNAL-ONLY —
    they exist on this schema and the internal ``power_forecasts`` Delta table to support
    cross-validation and the leaderboard, but they are NOT part of the ``power_forecast``
    table delivered to NGED. The delivery step projects them out (the delivered table is
    essentially the ``fold_id="live"`` rows with these internal columns dropped).
    """

    valid_time: datetime = pt.Field(dtype=UTC_DATETIME_DTYPE)
    """The target time this forecast is valid for."""

    time_series_id: int = _get_time_series_id_dtype()
    """NGED-provided primary key identifying the substation or asset."""

    ensemble_member: int = pt.Field(dtype=pl.Int8)
    """Ensemble member index (0-based)."""

    ml_flow_experiment_id: int | None = pt.Field(dtype=pl.Int32, allow_missing=True)
    """MLflow experiment ID; links to the MLflow experiment that produced this forecast."""

    nwp_init_time: datetime | None = pt.Field(
        dtype=UTC_DATETIME_DTYPE,
        allow_missing=True,
        description=(
            "The datetime that the underlying weather forecast was initialised. "
            "Null for models that do not use NWP (e.g. persistence baselines)."
        ),
    )
    """NWP model init time; null for non-NWP models (e.g. persistence baselines)."""

    power_fcst_model_name: str = pt.Field(
        dtype=pl.Categorical,
        description=(
            "Identifier for our ML-based power forecasting model."
            " Specified in the BaseForecaster subclass."
        ),
    )
    """Model-family identifier set by the `BaseForecaster` subclass (`MODEL_NAME`)."""

    experiment_name: str = pt.Field(
        dtype=pl.Categorical,
        description=(
            "Per-experiment key identifying the experiment that produced this forecast."
            " Distinct from `power_fcst_model_name`, which is the model-family identity"
            " (`MODEL_NAME`); do not overload that with experiment identity."
            " Forecasts are partitioned in Delta by (experiment_name, fold_id)."
            " INTERNAL-ONLY: projected out of the `power_forecast` table delivered to NGED."
        ),
    )
    """Per-experiment key (distinct from `power_fcst_model_name`); internal-only, not delivered to NGED."""

    power_fcst_model_version: int = pt.Field(dtype=pl.Int16)
    """Model version integer, bumped with each breaking change to the model implementation."""

    power_fcst_init_time: datetime = pt.Field(
        dtype=UTC_DATETIME_DTYPE,
        description=(
            "The datetime that the power forecast was initialised."
            " This might be called `t0` in some other OCF projects."
        ),
    )
    """When the power forecast was initialised (`t0` in some OCF projects)."""

    power_fcst: float = pt.Field(
        dtype=pl.Float32,
        description=(
            "The power forecast itself in units of MW (active power) or MVA (apparent power)."
            " The unit is defined in the `TimeSeriesMetadata` for this `time_series_id`."
            """ Positive values mean "power sent to NGED's grid","""
            """ and negative values mean "power drawn from NGED's grid"."""
            # PLANNED: We intend to change `power_fcst` to a normalised value in the range
            # [-1, +1] (which NGED multiplies by a capacity to recover MW/MVA), as described in
            # docs/roadmap/delivery-tables.md and docs/roadmap/forecast-building-blocks.md.
            # For this very early version we forecast raw MW/MVA because we are not yet
            # estimating capacity; we will switch to the scaled value once capacity estimation
            # lands (roadmap v0.6 / v0.7).
        ),
    )
    """Forecast power in MW or MVA (unit per `TimeSeriesMetadata`); positive = export to grid."""

    fold_id: FoldId = pt.Field(
        dtype=pl.Categorical,
        description=(
            "Identifies the source of this forecast row.  "
            "For cross-validation runs, the value is the validation year (e.g. '2022'), "
            "matching the CV fold whose validation period starts on 1 Jan of that year.  "
            "'live' means a production forecast with no associated CV fold.  "
            "All forecasts — CV and live — live in the same Delta table; "
            "filter on this column to select the population you need.  "
            "Extend the FoldId Literal in power_schemas.py as new CV epochs are added."
        ),
    )
    """CV validation year (e.g. `'2022'`) or `'live'` for production; all forecasts share one Delta table."""
