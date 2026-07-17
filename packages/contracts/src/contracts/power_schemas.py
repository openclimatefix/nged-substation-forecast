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
        description="End time of the 30-minute observation period (all NGED data is already half-hourly).",
    )

    power: float = pt.Field(
        dtype=pl.Float32,
        ge=-1000,
        le=1000,
        description="Average power (MW or MVA) over the preceding 30-minute period. Unit defined in TimeSeriesMetadata.",
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
    "BESS",
    "Biofuel",
    "CHP",
    "Data Centre",
    "Disaggregated Demand",
    "Energy from Waste",
    "EV Charging",
    "Geothermal",
    "Hydro",
    "Hydrogen Electrolysis",
    "Industrial Demand",
    "Mixed (Demand)",
    "Mixed (Generation)",
    "Other (Demand)",
    "Other (Generation)",
    "Other (Storage)",
    "Peaking Plant",
    "PV",
    "Rail",
    "Raw Flow",
    "Synchronous Condenser",
    "Wind",
)
"""All time-series type values used in NGED data.

Types present in the V1 trial area: BESS, Biofuel, Disaggregated Demand, Other (Generation), PV,
Raw Flow, Wind.

Notes:

- BESS: Battery energy storage system.
- Disaggregated Demand: In the trial area, exclusively associated with "Primary" substations. All
  "Primary" substations in the trial area have their TimeSeriesType set to "Disaggregated Demand".
  Indicates that NGED have already removed metered generation connected to that primary.
- Raw Flow: Used for BSP and GSP substations.
"""


class TimeSeriesMetadata(pt.Model):
    time_series_id: int = _get_time_series_id_dtype(unique=True)

    time_series_name: str = pt.Field(
        dtype=pl.String,
        description="Human-readable name for the substation or asset.",
        examples=[
            "ALFORD 33 11kV S STN",
            "BAMBERS FARM WIND GENERATION MABLETHORPE 33kV S ST",
            "Leverton Solar Park",
        ],
    )

    time_series_type: str = pt.Field(
        dtype=pl.Enum(LIST_OF_TIME_SERIES_TYPES),
        description="Asset category (e.g. ‘PV’, ‘Wind’, ‘Disaggregated Demand’). See LIST_OF_TIME_SERIES_TYPES.",
    )

    units: str = pt.Field(
        dtype=pl.Enum(["MW", "MVA"]),
        description="Power unit for this time series: ‘MW’ (active power) or ‘MVA’ (apparent power).",
    )

    licence_area: str = pt.Field(
        dtype=pl.Enum(["EMids"]),
        description="NGED licence area (for the trail area, this is always ‘EMids’).",
    )

    substation_number: int = pt.Field(
        dtype=pl.Int32,
        gt=0,
        lt=1_000_000,
        description="Perhaps surprisingly, each customer meter in the NGED trial area has its own substation_number (not one per physical substation).",
    )

    substation_type: str = pt.Field(
        dtype=pl.Enum(["BSP", "EHV Customer", "GSP", "HV Customer", "Primary"]),
        description="Substation voltage level / role: BSP, EHV Customer, GSP, HV Customer, or Primary. HV = high voltage. EHV = extra high voltage.",
    )

    latitude: float = pt.Field(
        dtype=pl.Float32,
        ge=49,
        le=61,  # UK latitude range
        description="Latitude in decimal degrees. For customer time series, gives the location of the substation, not the customer's site.",
    )

    longitude: float = pt.Field(
        dtype=pl.Float32,
        ge=-9,
        le=2,  # UK longitude range
        description="Longitude in decimal degrees. For customer time series, gives the location of the substation, not the customer's site.",
    )

    information: str | None = pt.Field(
        dtype=pl.String,
        allow_missing=True,
        description="Free-text NGED notes field; always null in the V1 trial area.",
    )

    area_wkt: str | None = pt.Field(
        dtype=pl.String,
        allow_missing=True,
        # Maps to the nested Area.WKT field in the JSON data.
        description=(
            "WKT polygon for the asset’s area. In the trial, only Primary substations have this."
            " NGED don’t have polygons for customer sites (though they hope to add that in future)."
            " For customer sites, where present, refers to the area covered by the generator itself."
        ),
    )

    area_center_lat: float | None = pt.Field(
        dtype=pl.Float32,
        allow_missing=True,
        description="Centroid latitude of the area polygon. For customer sites, the area, where present, refers to the area covered by the generator itself.",
    )

    area_center_lon: float | None = pt.Field(
        dtype=pl.Float32,
        allow_missing=True,
        description="Centroid longitude of the area polygon. For customer sites, the area, where present, refers to the area covered by the generator itself.",
    )

    h3_res_5: int = pt.Field(
        dtype=pl.UInt64,
        description="H3 discrete spatial index at resolution 5.",
    )


FoldId = str
"""Fold identifier for ``PowerForecast.fold_id``.

A CV fold's id is a short label defined in ``conf/cv/default.yaml`` (e.g.
``"mid_2025_to_mid_2026"``); fold identity is config-driven, never hard-coded here. ``"live"`` is
the reserved sentinel for a production forecast that belongs to no CV fold.
"""


class PowerForecast(pt.Model):
    """Forecast data schema for deterministic ensemble forecasts.

    Internal vs delivered schema (Milestone 1 report Table 1, p.28): the columns
    ``experiment_name``, ``fold_id``, and ``ml_flow_experiment_id`` are INTERNAL-ONLY —
    they exist on this schema and the internal ``power_forecasts`` Delta table to support
    cross-validation and the leaderboard, but they are NOT part of the ``power_forecast``
    table delivered to NGED.
    """

    valid_time: datetime = pt.Field(
        dtype=UTC_DATETIME_DTYPE,
        constraints=pl.col("valid_time") > pl.col("power_fcst_init_time"),
        description=(
            "The target time this forecast is valid for. Constrained to be strictly after"
            " power_fcst_init_time: a row targeting a valid time at or before its own"
            " initialisation is an undeliverable hindcast row — at power_fcst_init_time that"
            " valid time is already observed. The live service only forecasts strictly future"
            " valid times and bulk-mode feature engineering drops hindcast rows at source, so"
            " a constraint violation here indicates a pipeline regression."
        ),
    )

    time_series_id: int = _get_time_series_id_dtype()

    ensemble_member: int = pt.Field(
        dtype=pl.Int8, description="Ensemble member index. 0 is the control NWP ensemble member."
    )

    ml_flow_experiment_id: int | None = pt.Field(
        dtype=pl.Int32,
        allow_missing=True,
        description="MLflow experiment ID; links to the MLflow experiment that produced this forecast.",
    )

    nwp_init_time: datetime | None = pt.Field(
        dtype=UTC_DATETIME_DTYPE,
        allow_missing=True,
        description=(
            "The datetime that the underlying weather forecast was initialised. "
            "Null for models that do not use NWP (e.g. persistence baselines)."
        ),
    )

    power_fcst_model_name: str = pt.Field(
        dtype=pl.String,  # String, not Categorical — see experiment_name below.
        description="Identifier for our ML-based power forecasting model. Model-family identity set by the BaseForecaster subclass (MODEL_NAME).",
    )

    # String (not Categorical): experiment_name/fold_id are the Delta partition columns and delta-rs
    # stores dictionary-encoded columns as String anyway; String keeps them cast-free and lets
    # predicate pushdown work. See the "declare Delta filter/partition columns as String" gotcha:
    # ../../../../CLAUDE.md#delta-lake-dictionary-encoded-columns-declare-delta-filterpartition-columns-as-string
    experiment_name: str = pt.Field(
        dtype=pl.String,
        description=(
            "Per-experiment key identifying the experiment that produced this forecast."
            " Distinct from `power_fcst_model_name`, which is the model-family identity"
            " (`MODEL_NAME`); do not overload that with experiment identity."
            " Forecasts are partitioned in Delta by (experiment_name, fold_id)."
            " INTERNAL-ONLY: projected out of the `power_forecast` table delivered to NGED."
        ),
    )

    power_fcst_model_version: int = pt.Field(
        dtype=pl.Int16,
        description="Model version integer, bumped with each breaking change to the model implementation.",
    )

    power_fcst_init_time: datetime = pt.Field(
        dtype=UTC_DATETIME_DTYPE,
        description="The datetime that the power forecast was initialised. This might be called `t0` in some other OCF projects.",
    )

    power_fcst: float = pt.Field(
        dtype=pl.Float32,
        description=(
            "The power forecast itself in units of MW (active power) or MVA (apparent power)."
            " The unit is defined in the `TimeSeriesMetadata` for this `time_series_id`."
            """ Positive values mean "power sent to NGED's grid","""
            """ and negative values mean "power drawn from NGED's grid"."""
            " Rows read back from the internal `power_forecasts` Delta table carry reduced"
            " precision: values are rounded to a 13-bit significand at write time"
            " (max relative error 2^-13 ≈ 1.2e-4, far below forecast error) to aid compression;"
            " see `delta_store.power_forecasts`."
            # PLANNED: We intend to change `power_fcst` to a normalised value in the range
            # [-1, +1] (which NGED multiplies by a capacity to recover MW/MVA), per the
            # delivery-contract design agreed with NGED in the Milestone 1 report.
            # For this very early version we forecast raw MW/MVA because we are not yet
            # estimating capacity; we will switch to the scaled value once capacity estimation
            # lands (roadmap v0.7).
        ),
    )

    fold_id: FoldId = pt.Field(
        dtype=pl.String,  # String, not Categorical — see experiment_name above.
        description=(
            "Identifies the source of this forecast row.  "
            "For cross-validation runs, the value is the fold's label from conf/cv/default.yaml "
            "(e.g. 'mid_2025_to_mid_2026').  "
            "'live' means a production forecast with no associated CV fold.  "
            "All forecasts — CV and live — live in the same Delta table; "
            "filter on this column to select the population you need."
        ),
    )


class EffectiveCapacity(pt.Model):
    """Effective capacity of each time series at each half-hourly timestep.

    Delivered to NGED as ``effective_capacity`` Delta table (Table 4 in the Milestone 1 report).
    This table is backward-looking only — it does not cover the forecast period.

    **v0.1 implementation:** one row per ``time_series_id``, ``time`` set to the end of
    the available observation history, ``effective_capacity_mw`` = P99 of ``abs(power)`` over
    the full observed history. This is a static scalar per series.

    **Planned upgrade (v0.7):** replace the P99 scalar with a time-varying capacity estimate
    (see ``docs/techniques/convex-optimisation.md`` and
    ``docs/techniques/differentiable-physics.md`` for the candidate estimation methods),
    giving one row per ``(time_series_id, time)`` half-hourly timestep. This schema is unchanged;
    the ``effective_capacity`` asset body changes and the ``metrics`` pipeline swaps its
    ``time_series_id``-only NMAE-denominator join for a temporal as-of join. Do **not** pre-densify
    the v0.1 scalar into one row per half-hour — densifying a constant buys nothing, and the as-of
    join handles sparse capacity rows naturally.
    """

    time_series_id: int = _get_time_series_id_dtype()

    time: datetime = pt.Field(
        dtype=UTC_DATETIME_DTYPE,
        description=(
            "The half-hourly timestep this capacity estimate applies to. "
            "In v0.1, this is the end of the available observation history for that series."
        ),
    )

    effective_capacity_mw: float = pt.Field(
        dtype=pl.Float32,
        gt=0,
        description=(
            "OCF's estimate of the effective capacity (MW) of this asset at this timestep. "
            "For generators: absorbs PV panel degradation, partial inverter trips, etc., "
            "but ignores ANM curtailment — a wind farm ANM-capped at 5 MW with 10 MW physical "
            "capability has effective_capacity_mw = 10. "
            "For substations: the 99th percentile of observed load over a rolling time window, "
            "under normal running arrangement only. 'Switched' power (Table 5) should be "
            "added or subtracted when a switching event is in effect."
        ),
    )
