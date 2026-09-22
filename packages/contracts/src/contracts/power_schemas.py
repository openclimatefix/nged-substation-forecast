"""Contracts for NGED's power telemetry and for the forecasts we make from it.

The half-hourly ``PowerTimeSeries`` observations as they arrive from NGED, the
``TimeSeriesMetadata`` roster describing each series, the ``PowerForecast`` schema every model
emits, and the ``EffectiveCapacity`` estimate the metrics pipeline divides the mean absolute
error by, to express that error as a fraction of the series' capacity.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import ClassVar, Final, NamedTuple, Self

import patito as pt
import polars as pl

from .common import (
    MAX_PLAUSIBLE_DATETIME,
    MIN_PLAUSIBLE_DATETIME,
    UTC_DATETIME_DTYPE,
    _get_time_series_id_dtype,
    check_datetime_bounds,
    split_by_datetime_plausibility,
)


class DropImplausibleRowsResult(NamedTuple):
    """Result of ``PowerTimeSeries.drop_implausible_rows``."""

    survivors: pl.DataFrame
    n_dropped: int


POWER_TIMESTAMPS_CORRECTED_BEFORE: Final[datetime] = datetime(2026, 3, 26, 8, 30, tzinfo=UTC)
"""NGED's power timestamps are half an hour late before this instant, and correct from this instant
onwards.

A reading whose timestamp `T` falls before this instant is the mean over `(T - 60 min, T - 30 min]`,
not the `(T - 30 min, T]` the `time` field states. NGED reported the fault and corrected the feed at
this instant. Three independent measurements of when a solar farm's output peaks against the sun
agree with NGED's account:
<https://openclimatefix.github.io/nged-substation-forecast/results/beam-diffuse-split/#the-power-timestamps-before-26-march-2026-are-half-an-hour-late>.

The correction applies to every `time_series_id`. NGED report that they convert every series in the
trial area through one code path, so no series can have escaped the fault. That report is what the
fleet-wide scope rests on: the published measurements cover the six metered solar farms only,
because each measurement needs solar geometry, which a substation load profile has no equivalent
of.
"""


class PowerTimeSeries(pt.Model):
    """Half-hourly power observations (MW or MVA), one row per (time_series_id, time)."""

    time_series_id: int = _get_time_series_id_dtype()

    time: datetime = pt.Field(
        dtype=UTC_DATETIME_DTYPE,
        description=(
            "End time of the 30-minute observation period (all NGED data is already half-hourly)."
            " A value before `POWER_TIMESTAMPS_CORRECTED_BEFORE` is NGED's own timestamp moved 30"
            " minutes earlier by `correct_late_timestamps`: NGED's feed stamped every reading half"
            " an hour late until NGED corrected the feed at that instant, and the ingest repairs"
            " the late readings as they arrive."
            f" Must fall between {MIN_PLAUSIBLE_DATETIME:%Y-%m-%d} and"
            f" {MAX_PLAUSIBLE_DATETIME:%Y-%m-%d} (enforced by `validate`, not by the field, because"
            " Patito ignores `ge`/`le` on datetime fields — see `check_datetime_bounds`)."
        ),
    )

    power: float = pt.Field(
        dtype=pl.Float32,
        ge=-1000,
        le=1000,
        description=(
            "Average power (MW or MVA) over the preceding 30-minute period. Unit defined in "
            "TimeSeriesMetadata."
            " This is what the meter recorded, so an hour in which the network operator capped"
            " a generator's export reads as the capped value rather than as the power that was"
            " available. `PowerForecast.power_fcst` is defined the other way round."
            " Sign convention depends on `substation_type` in `TimeSeriesMetadata`, and describes"
            " a direction, so it applies only where `units` is `MW`. A series metered in `MVA`"
            " reports the magnitude of the flow and cannot see direction, so reverse power flow"
            " appears as a rise rather than as a change of sign. A negative value is then a fault"
            " between the meter and us rather than an export. See the Sign convention section in"
            " this package's README.md, also published at"
            " https://openclimatefix.github.io/nged-substation-forecast/roadmap/forecast-building-blocks/#sign-convention."
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
        """Validate the given dataframe, ensuring time is plausible, at :00 or :30, and unique."""
        validated_df = super().validate(
            dataframe=dataframe,
            columns=columns,
            allow_missing_columns=allow_missing_columns,
            allow_superfluous_columns=allow_superfluous_columns,
            drop_superfluous_columns=drop_superfluous_columns,
        )

        # Validate time falls in the plausible range (Patito ignores `ge`/`le` on datetime fields)
        check_datetime_bounds(validated_df, "time")

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

    @classmethod
    def correct_late_timestamps(cls, dataframe: pl.DataFrame) -> pl.DataFrame:
        """Move every `time` before ``POWER_TIMESTAMPS_CORRECTED_BEFORE`` 30 minutes earlier.

        NGED stamped this feed half an hour late until they corrected the feed; the constant's
        docstring holds the evidence and the scope. Correcting at ingestion is what lets the `time`
        field mean the same half-hour for every row, so no consumer has to know whether a row was
        stamped before or after NGED's correction.

        Call ``correct_late_timestamps`` BEFORE ``drop_implausible_rows``, and only at a boundary
        that receives NGED's raw JSON. ``drop_implausible_rows`` has to judge the timestamp that
        will actually be stored. In the reverse order, a reading whose corrected timestamp falls
        outside the plausible range survives the drop and then raises out of ``validate``. One
        malformed external reading would then fail the whole ingest run.

        The correction cannot collide with an existing row or disturb the sort order. The
        correction shifts a contiguous prefix of each series by a constant, and the shifted prefix
        ends 30 minutes before the unshifted remainder begins. A corrected series has no reading at
        ``POWER_TIMESTAMPS_CORRECTED_BEFORE - 30 min``, because NGED never published that
        half-hour.

        Args:
            dataframe: A frame with a `time` column already cast to ``UTC_DATETIME_DTYPE``; need
                not yet be validated.

        Returns:
            `dataframe` with `time` corrected, and every other column untouched.
        """
        return dataframe.with_columns(
            time=pl.when(pl.col("time") < POWER_TIMESTAMPS_CORRECTED_BEFORE)
            .then(pl.col("time").dt.offset_by("-30m"))
            .otherwise(pl.col("time"))
        )

    @classmethod
    def drop_implausible_rows(cls, dataframe: pl.DataFrame) -> DropImplausibleRowsResult:
        """Drop rows with a malformed ``time``, returning ``(survivors, n_dropped)``.

        A row is dropped when its ``time`` lies outside the plausible datetime range, is null, or
        does not fall on the top or bottom of the hour (minute 00 or 30). The schema declares
        ``time`` non-nullable, so a null this early is already malformed. All three conditions
        indicate a malformed upstream reading — not a bug in our own pipeline — so under [inherent
        stability](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/)
        an ingestion boundary should degrade the batch rather than abort it entirely.

        Dropping rows exists alongside ``validate``, which stays strict. ``validate`` raises on
        two of the three conditions above: a ``time`` outside the plausible range, and a ``time``
        that is not on the top or bottom of the hour. The third condition, a null ``time``, is
        rejected earlier still, by the non-nullable field. ``validate`` is also used as a hard
        assertion in tests and R&D code, where a raise-on-violation contract must not silently
        change.

        Call this method BEFORE ``validate``, and only at a boundary that receives data from
        outside our system (e.g. NGED's raw JSON feed). The uniqueness and sortedness checks in
        ``validate`` are NOT relaxed here, and should keep raising. A duplicate row or an unsorted
        column indicates a bug in OUR pipeline rather than malformed external data.

        Args:
            dataframe: An already-cast frame with a ``time`` column; need not yet be validated.

        Returns:
            ``(survivors, n_dropped)``. ``survivors`` keeps ``dataframe``'s row order.
        """
        survivors, _ = split_by_datetime_plausibility(dataframe, "time")
        # `dt.minute()` is null for a null `time` and `.filter()` drops a row on a null predicate,
        # so this also drops the null `time`s the non-nullable schema forbids.
        survivors = survivors.filter(pl.col("time").dt.minute().is_in([0, 30]))
        # The dropped-row count is computed as a difference in `height` (a frame's row count),
        # rather than by summing the rejected partitions. Whatever a filter does with a null
        # predicate, a height difference counts every row that left exactly once.
        return DropImplausibleRowsResult(survivors, dataframe.height - survivors.height)

    # Define columns_to_sort_by as a ClassVar so Patito/Pydantic knows it is not a data field
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
- PV: Photovoltaic (solar).
- Raw Flow: Used for bulk supply point (BSP) and grid supply point (GSP) substations.
"""


class TimeSeriesMetadata(pt.Model):
    """One row per time series — a substation or a customer meter.

    Carries the series' name, location, H3 index, and substation type.
    """

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
        description=(
            "Asset category (e.g. ‘PV’, ‘Wind’, ‘Disaggregated Demand’). See "
            "LIST_OF_TIME_SERIES_TYPES."
        ),
    )

    units: str = pt.Field(
        dtype=pl.Enum(["MW", "MVA"]),
        description=(
            "Power unit for this time series: ‘MW’ (active power) or ‘MVA’ (apparent power)."
        ),
    )

    licence_area: str = pt.Field(
        dtype=pl.Enum(["EMids"]),
        description="NGED licence area (for the trial area, this is always ‘EMids’).",
    )

    substation_number: int = pt.Field(
        dtype=pl.Int32,
        gt=0,
        lt=1_000_000,
        description=(
            "Each customer meter in the NGED trial area has its own "
            "substation_number (not one per physical substation)."
        ),
    )

    substation_type: str = pt.Field(
        dtype=pl.Enum(["BSP", "EHV Customer", "GSP", "HV Customer", "Primary"]),
        description=(
            "Substation voltage level / role: BSP, EHV Customer, GSP, HV Customer, or Primary."
            " BSP = bulk supply point. GSP = grid supply point. HV = high voltage."
            " EHV = extra high voltage."
        ),
    )

    latitude: float = pt.Field(
        dtype=pl.Float32,
        ge=49,
        le=61,  # UK latitude range
        description=(
            "Latitude in decimal degrees. For customer time series, gives the location of the "
            "substation, not the customer's site."
        ),
    )

    longitude: float = pt.Field(
        dtype=pl.Float32,
        ge=-9,
        le=2,  # UK longitude range
        description=(
            "Longitude in decimal degrees. For customer time series, gives the location of the "
            "substation, not the customer's site."
        ),
    )

    information: str | None = pt.Field(
        dtype=pl.String,
        allow_missing=True,
        description="Free-text NGED notes field; always null in the V1 trial area.",
    )

    area_wkt: str | None = pt.Field(
        dtype=pl.String,
        allow_missing=True,
        # Maps to the nested Area.WKT field in NGED's source JSON.
        description=(
            "Well-known text (WKT) polygon for the asset’s area. In the trial, only Primary"
            " substations have this. No customer site has a polygon yet. Where a customer site"
            " does have a polygon, the polygon refers to the area covered by the generator"
            " itself."
        ),
    )

    area_center_lat: float | None = pt.Field(
        dtype=pl.Float32,
        allow_missing=True,
        description=(
            "Centroid latitude of the area polygon. For customer sites, the area, where present, "
            "refers to the area covered by the generator itself."
        ),
    )

    area_center_lon: float | None = pt.Field(
        dtype=pl.Float32,
        allow_missing=True,
        description=(
            "Centroid longitude of the area polygon. For customer sites, the area, where present, "
            "refers to the area covered by the generator itself."
        ),
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
    """Forecast data schema for an ensemble of deterministic forecasts.

    Each ensemble member carries its own single-valued, or deterministic, forecast. The spread
    across the members is what expresses the forecast's uncertainty.

    One row per time series, per forecast run, per target time, per ensemble member — the four
    columns of ``PRIMARY_KEY``, declared below in that order.

    Internal vs delivered schema (Milestone 1 report Table 1, p.28): three columns are
    INTERNAL-ONLY — ``experiment_name``, ``fold_id``, and ``ml_flow_experiment_id``. They exist
    on this schema and on the internal ``power_forecasts`` Delta table, to support
    cross-validation and the leaderboard. They are NOT part of the ``power_forecast`` table
    delivered to NGED.
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
        dtype=pl.Int8,
        description=(
            "Ensemble member index. Member 0 is the control numerical weather prediction (NWP)"
            " ensemble member."
        ),
    )

    ml_flow_experiment_id: int | None = pt.Field(
        dtype=pl.Int32,
        allow_missing=True,
        description=(
            "MLflow experiment ID; links to the MLflow experiment that produced this forecast."
        ),
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
        description=(
            "Identifier for our ML-based power forecasting model. Model-family identity set by the "
            "BaseForecaster subclass (MODEL_NAME)."
        ),
    )

    # String (not Categorical): experiment_name/fold_id are the Delta partition columns, and
    # delta-rs stores dictionary-encoded columns as String anyway. String keeps those two columns
    # cast-free and lets predicate pushdown work. See the "Delta Lake dictionary-encoded columns"
    # section of the `polars-patito-gotchas` skill.
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
        description=(
            "Model version integer, bumped with each breaking change to the model implementation."
        ),
    )

    power_fcst_init_time: datetime = pt.Field(
        dtype=UTC_DATETIME_DTYPE,
        description=(
            "The datetime that the power forecast was initialised. This might be called `t0` in "
            "some other OCF projects."
        ),
    )

    power_fcst: float = pt.Field(
        dtype=pl.Float32,
        description=(
            "The power forecast itself in units of MW (active power) or MVA (apparent power)."
            " The unit is defined in the `TimeSeriesMetadata` for this `time_series_id`."
            " This forecasts the power *available*: what the series would have exported or drawn"
            " had no active-network-management (ANM) instruction been in force. A generator on a"
            " flexible connection can be told to turn down when the local electricity network"
            " fills, and that instruction is a network decision rather than a weather outcome, so"
            " it falls outside what this column describes. Holding a delivered forecast down to a"
            " cap the operator has set is the consumer's own step, and the same assumption is what"
            " the `power_forecast` delivery block means by a normal running arrangement — see"
            " https://openclimatefix.github.io/nged-substation-forecast/roadmap/forecast-building-blocks/#the-idea."
            " `PowerTimeSeries.power` is the opposite convention: a metered value that already"
            " carries any curtailment the operator instructed."
            " Sign convention depends on `substation_type` in `TimeSeriesMetadata`, and describes"
            " a direction, so it applies only where `units` is `MW` — see the Sign convention"
            " section in this package's README.md, also published at"
            " https://openclimatefix.github.io/nged-substation-forecast/roadmap/forecast-building-blocks/#sign-convention."
            " Rows read back from the internal `power_forecasts` Delta table carry reduced"
            " precision: values are rounded to a 13-bit significand at write time"
            " (max relative error 2^-13 ≈ 1.2e-4, far below forecast error) to aid compression;"
            " see `delta_store.power_forecasts`."
            # PLANNED: We intend to change `power_fcst` to a normalised value in the range
            # [-1, +1], which NGED multiplies by a capacity to recover MW/MVA. That change follows
            # the delivery-contract design agreed with NGED in the Milestone 1 report. The switch
            # is planned for v0.5. The switch will use the static P99 `effective_capacity`
            # estimate that already exists — the same scalar the `metrics` pipeline already divides
            # by for normalised mean absolute error (NMAE). The switch therefore no longer waits
            # for a time-varying capacity estimate.
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

    PRIMARY_KEY: ClassVar[tuple[str, ...]] = (
        "time_series_id",
        "power_fcst_init_time",
        "valid_time",
        "ensemble_member",
    )
    """At most one forecast per series, per init time, per target time, per ensemble member."""

    @classmethod
    def validate(  # ty: ignore[invalid-method-override]
        cls,
        dataframe: pl.DataFrame,
        columns: Sequence[str] | None = None,
        allow_missing_columns: bool = False,
        allow_superfluous_columns: bool = False,
        drop_superfluous_columns: bool = False,
    ) -> pt.DataFrame[Self]:
        """Validate the given dataframe, ensuring the primary key is unique.

        A duplicated primary key means either a join fanned out on the way here or the same rows
        were written twice, and both corrupt the metrics computed from them. It is our own bug
        rather than the outside world misbehaving, so this raises rather than degrading — see
        <https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/>.
        """
        validated_df = super().validate(
            dataframe=dataframe,
            columns=columns,
            allow_missing_columns=allow_missing_columns,
            allow_superfluous_columns=allow_superfluous_columns,
            drop_superfluous_columns=drop_superfluous_columns,
        )

        # `n_unique`, not `is_duplicated().any()`: the two expressions are equivalent here (every
        # primary-key column is non-nullable) but `is_duplicated` materialises a per-row mask,
        # costing ~5x the peak memory on a predict-sized frame.
        pk_cols = list(cls.PRIMARY_KEY)
        if validated_df.select(pk_cols).n_unique() != validated_df.height:
            raise ValueError(
                f"Duplicate entries found for primary key columns: {pk_cols}. "
                "Either an upstream join fanned out or these rows were written twice."
            )

        return validated_df


class EffectiveCapacity(pt.Model):
    """Effective capacity of each time series, at one or more half-hourly timesteps.

    Effective capacity is an estimate of the power a site actually reaches, derived from its own
    observed history. Effective capacity is not a nameplate, firm, or connection-agreement rating.

    Delivered to NGED as ``effective_capacity`` Delta table (Table 4 in the Milestone 1 report).
    This table is backward-looking only — it does not cover the forecast period.

    **v0.1 implementation:** one row per ``time_series_id``, ``time`` set to the end of the
    available observation history, ``effective_capacity_mw`` = P99 of ``abs(power)`` over the
    full observed history. The v0.1 estimate is a static scalar per series.

    **Planned upgrade (v0.7):** replace the P99 scalar with a time-varying capacity estimate,
    giving one row per ``(time_series_id, time)`` half-hourly timestep. The candidate estimation
    methods are described at
    <https://openclimatefix.github.io/nged-substation-forecast/techniques/convex-optimisation/>
    and
    <https://openclimatefix.github.io/nged-substation-forecast/techniques/differentiable-physics/>.
    This schema is unchanged. The ``effective_capacity`` asset body changes, and the ``metrics``
    pipeline swaps its ``time_series_id``-only normalised-mean-absolute-error-denominator join
    for a temporal as-of join, which matches each forecast row to the most recent capacity row at
    or before its timestamp. Do **not** pre-densify the v0.1 scalar into one row per half-hour —
    densifying a constant adds no information, and the as-of join handles sparse capacity rows
    naturally.
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
            "OCF's estimate of the effective capacity of this asset at this timestep. "
            "Despite the column name, the value carries the series' own unit from "
            "`TimeSeriesMetadata.units`, so a series metered in MVA has its effective capacity "
            "in MVA. "
            "For generators: absorbs any persistent loss of capability, such as photovoltaic "
            "(PV) panel degradation or a partial inverter trip. The estimate ignores Active "
            "Network Management (ANM) curtailment — a wind farm ANM-capped at 5 MW with 10 MW "
            "physical capability has effective_capacity_mw = 10. "
            "For substations: the 99th percentile of observed absolute power flow, under normal "
            "running arrangement only. In v0.1 that percentile is taken over the full observed "
            "history, giving one static scalar per series. 'Switched' power (Table 5) should be "
            "added or subtracted when a switching event is in effect."
        ),
    )
