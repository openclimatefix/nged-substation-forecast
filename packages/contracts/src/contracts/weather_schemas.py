"""Contracts for numerical weather prediction data.

The `Nwp` frame as stored and read, plus the run-completeness and data-quality reports that assess
an ingested ECMWF ENS run.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum, auto
from pathlib import Path
from typing import ClassVar, Final, Literal, Self

import patito as pt
import polars as pl

from contracts.settings import get_settings
from contracts.typing_utils import typeddict_to_dict
from contracts.uri import ObjectStoreOptions

from .common import UTC_DATETIME_DTYPE, check_datetime_bounds

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
    """The NWP models we ingest, used as the `nwp_model_id` column's vocabulary."""

    ECMWF_ENS_0_25_degree = auto()


ECMWF_ENS_ENSEMBLE_MEMBERS: Final[frozenset[int]] = frozenset(range(51))
"""The ensemble members every ECMWF IFS ENS run carries: the control member plus 50 perturbed
members, indexed 0-50 by Dynamical.org (matching `Nwp.ensemble_member`, which is 0-based).

The count is a fixed property of the ECMWF IFS ENS configuration, corroborated by our own data: the
2026-07-14 incident described in
<https://openclimatefix.github.io/nged-substation-forecast/architecture/ecmwf-ens-known-issues/>
lost a forecast step for "50 of 51 ensemble members", and the per-partition row arithmetic in
<https://openclimatefix.github.io/nged-substation-forecast/architecture/performance/> is
1671 H3 cells x 51 ensemble members x 85 native steps.
"""

ECMWF_ENS_LEAD_TIME_HOURS: Final[tuple[int, ...]] = tuple(range(0, 145, 3)) + tuple(
    range(150, 361, 6)
)
"""The 85 native forecast steps of the ECMWF IFS ENS 15-day 0.25-degree dataset, in hours after
`init_time`: 3-hourly out to 144 h (49 steps, lead-0 inclusive), then 6-hourly out to 360 h
(36 steps) — i.e. `init_time` to `init_time + 15 days` inclusive.

Lead-0 is included: it is a real step, distinguished only by the de-accumulated variables being
legitimately null there (see `Nwp.deaccumulated_var_names`).
"""

ECMWF_ENS_H3_RESOLUTION: Final[int] = 5
"""The H3 spatial resolution the `ecmwf_ens` ingest pipeline grids ECMWF ENS data to, and the
resolution `power_time_series_and_metadata` computes for every time series so the two line up on
one grid. Every NWP model shares this one resolution today; per-model resolution (so a future NWP
source can use a different one) is tracked in
<https://github.com/openclimatefix/nged-substation-forecast/issues/114>.
"""


class NwpVariableWhollyMissing(ValueError):
    """A de-accumulated NWP variable is null in *every* slice beyond lead-0 of a run.

    The field is wholesale absent rather than locally corrupt, so `Nwp.validate` rejects it: an
    all-null weather column would otherwise train and serve as a silently-degraded input for the
    full 15-day horizon, which is worse than falling back on the previous run.

    A distinct exception type rather than a bare `ValueError` because the `ecmwf_ens` asset
    retries it. An upstream publication still in progress can present this way — a variable's
    chunks read as fill-value null until the worker writing them commits — so waiting is a better
    first response than failing the partition. Note the limit of that: it only reaches this check
    when the unwritten variables are the de-accumulated ones. The nine instantaneous variables are
    non-nullable, so a frame missing one of those is rejected by base Patito validation first, with
    no retry — as is an all-null `categorical_precipitation_type_surface`, which is nullable but
    carries its own historical invariant. See
    <https://openclimatefix.github.io/nged-substation-forecast/architecture/ecmwf-ens-known-issues/>.
    """


class Nwp(pt.Model):
    """Weather data schema for NWP forecasts.

    Gridded ECMWF ENS ensemble weather, one row per
    (nwp_model_id, init_time, valid_time, ensemble_member, h3_index).

    Stored on disk as plain Float32, rounded to a significand-bit budget by
    `delta_store.nwp.write_nwp` — see
    <https://openclimatefix.github.io/nged-substation-forecast/architecture/overview/> for the
    physical format and measured numbers.

    `validate` is the fatal ingest gate; `assess_nwp_quality` reports the tolerated nulls in the
    de-accumulated variables (the known upstream ECMWF ENS corruption) and
    `assess_nwp_run_completeness` reports a run that is missing whole members, steps or cells —
    both as non-fatal checks, because both are the upstream provider misbehaving rather than a
    contract violation. Which patterns are fatal versus tolerated, and why, is documented at
    <https://openclimatefix.github.io/nged-substation-forecast/architecture/ecmwf-ens-known-issues/>.
    """

    # `dtype=pl.String` (not `Enum`, which Delta cannot store) means an out-of-vocabulary value is
    # no longer physically inexpressible the way it was under `Enum` — a plain `str` column can
    # hold any string at construction time, with no error. The `constraints=` check below is now
    # the *only* defence, and it fires at `Nwp.validate()`, not at construction: a frame built and
    # never validated can carry an unrecognised `nwp_model_id` all the way through. Every current
    # caller validates, so this is latent rather than live today.
    nwp_model_id: str = pt.Field(
        dtype=pl.String,
        constraints=pl.col("nwp_model_id").is_in([model.name for model in NwpModelId]),
        description="Which NWP model produced this row (e.g. 'ECMWF_ENS_0_25_degree').",
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
            "ECMWF's raw cumulative fields before we receive them. Resampling must honour that "
            "distinction — the step interval is 3 h out to lead 144 h and 6 h beyond it, so "
            "treating a period-ending value as instantaneous shifts it by up to 3 h. Conventions "
            "for every variable, and what the shift costs: "
            "<https://openclimatefix.github.io/nged-substation-forecast/architecture/nwp-variable-conventions/>"
        ),
    )

    ensemble_member: int = pt.Field(
        dtype=pl.Int8,
        description="Ensemble member index (0-based).",
    )

    # `Int64`, not `UInt64`, because Delta has no unsigned integer types: the column is stored as
    # `int64` whatever we declare, and declaring `UInt64` only bought a cast in `scan_delta` that
    # stopped Polars pushing `h3_index` filters into the Parquet scan.
    #
    # Signed is safe for the whole H3 index space, not just the values we happen to store. H3
    # reserves bit 63 of its 64-bit index as always zero — it is not a value bit in any index mode
    # (cell, directed edge, vertex) at any resolution — so every H3 index is below 2**63, which is
    # exactly the range a signed `Int64` covers. A cell index sets mode bits 59-62 to 1 and so tops
    # out around 2**60, well inside that. Sampling every resolution 0-15 agrees: the largest index
    # seen is ~6.5e17 against `Int64`'s ~9.2e18 ceiling, and bit 63 is never set.
    #
    # The guarantee does not depend on the resolution, so raising or lowering
    # `ECMWF_ENS_H3_RESOLUTION`, or giving a future NWP model its own resolution (issue 114),
    # cannot overflow this column. Resolution lives in bits 52-55 and leaves bit 63 alone.
    h3_index: int = pt.Field(
        dtype=pl.Int64,
        description=(
            "H3 cell index, at `ECMWF_ENS_H3_RESOLUTION` (every NWP model shares it today)."
        ),
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
        description=(
            "Wind speed at 10 m. Unit: meters per second. This is the magnitude of the cell's"
            " *vector* mean wind, not the mean of its grid points' scalar speeds. See"
            " <https://openclimatefix.github.io/nged-substation-forecast/architecture/nwp-variable-conventions/#wind-is-stored-as-speed-and-direction-and-why>"
        ),
        ge=0,
        le=200,  # Gemini says the highest non-tornadic surface wind speed recorded was 113 m/s
    )

    wind_direction_10m: float = pt.Field(
        dtype=pl.Float32,
        description=(
            "Wind direction at 10 m. The angle where the wind is coming from. Degrees."
            " 0° is North; 90° is East. This is a *circular* quantity: 359° and 1° are two degrees"
            " apart, so interpolating, averaging, differencing or taking a quantile of it as if it"
            " were an ordinary number is wrong wherever the values straddle North. Convert to"
            " components first. See"
            " <https://openclimatefix.github.io/nged-substation-forecast/architecture/nwp-variable-conventions/>"
        ),
        ge=0,
        le=360,
    )

    wind_speed_100m: float = pt.Field(
        dtype=pl.Float32,
        description=(
            "Wind speed at 100 m. Unit: meters per second. This is a vector mean, with the same"
            " caveat as `wind_speed_10m`. It is the height that matters most for wind generation,"
            " because a turbine power curve responds to the scalar speed at each grid point. See"
            " <https://openclimatefix.github.io/nged-substation-forecast/architecture/nwp-variable-conventions/#wind-is-stored-as-speed-and-direction-and-why>"
        ),
        ge=0,
        le=200,  # Gemini says the highest non-tornadic surface wind speed recorded was 113 m/s
    )

    wind_direction_100m: float = pt.Field(
        dtype=pl.Float32,
        description=(
            "Wind direction at 100 m. The angle where the wind is coming from. Degrees. 0° is "
            "North; 90° is East. This is circular, with the same handling requirement as "
            "`wind_direction_10m`."
        ),
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
        description=(
            "Downward long-wave radiation flux at surface. Note that this variable is all-null for "
            "lead time 0. Unit: W m-2."
        ),
        ge=0,
        le=1500,  # Max in 2 years of ECMWF ENS = 445
    )

    downward_short_wave_radiation_flux_surface: float | None = pt.Field(
        dtype=pl.Float32,
        description=(
            "Downward short-wave (solar) radiation flux at surface. Note that this variable is"
            " all-null for lead time 0. Unit: W m-2."
        ),
        ge=0,
        le=1500,  # Max in 2 years of ECMWF ENS = 892
    )

    precipitation_surface: float | None = pt.Field(
        dtype=pl.Float32,
        description=(
            "Total precipitation rate at surface, de-accumulated by Dynamical. Note that this "
            "variable is all-null for lead time 0. Unit: kg m-2 s-1."
        ),
        ge=0,
        le=0.01,  # Max in 2 years of ECMWF ENS = 0.006
    )

    categorical_precipitation_type_surface: int | None = pt.Field(
        dtype=pl.Int16,
        ge=0,
        le=255,
        description=(
            "This field is always NaN for init_times on and before 2024-11-12, and populated from"
            " the 2024-11-13 00Z run onwards (confirmed by direct inspection of the source data)."
            " Derived from ECMWF's `ptype` field. See https://codes.ecmwf.int/grib/param-db/260015"
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

    deaccumulated_var_names: ClassVar[frozenset[str]] = frozenset(
        {
            "precipitation_surface",
            "downward_short_wave_radiation_flux_surface",
            "downward_long_wave_radiation_flux_surface",
        }
    )
    """The variables Dynamical.org de-accumulates from ECMWF's cumulative source fields to rates.

    All three are legitimately null at lead-0, and they share a de-accumulation step whose known
    upstream corruption leaves further nulls beyond it. Those are tolerated at ingest and reported
    by :func:`assess_nwp_quality`; only a variable null in *every* slice beyond lead-0 is fatal.
    Which corruption patterns arrive, what the H3 aggregation absorbs before they reach a validated
    frame, and why the survivors are tolerated:
    <https://openclimatefix.github.io/nged-substation-forecast/architecture/ecmwf-ens-known-issues/#nulls-in-the-de-accumulated-variables-tolerated>.
    """

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
        """Validate the frame.

        Bounds `init_time`/`valid_time` to the plausible datetime range, rejects a de-accumulated
        variable that is wholly missing (smaller null patterns are tolerated), and enforces both
        uniqueness and the ptype-introduction invariant.
        """
        validated_df = super().validate(
            dataframe=dataframe,
            columns=columns,
            allow_missing_columns=allow_missing_columns,
            allow_superfluous_columns=allow_superfluous_columns,
            drop_superfluous_columns=drop_superfluous_columns,
        )

        # Patito ignores `ge`/`le` on datetime fields, so the range check lives here.
        check_datetime_bounds(validated_df, "init_time", "valid_time")
        cls._check_no_wholly_missing_deaccumulated_variable(validated_df)
        cls._check_unique(validated_df)
        cls._check_variables_that_were_introduced_after_start_of_dataset(validated_df)
        return validated_df

    @classmethod
    def _check_no_wholly_missing_deaccumulated_variable(cls, dataframe: pt.DataFrame[Self]) -> None:
        """Reject a de-accumulated variable that carries no weather at all.

        That is, one that is null in *every* (ensemble_member, valid_time) slice beyond lead-0 of
        a run.

        Every smaller null pattern is *tolerated* and reported by :func:`assess_nwp_quality`
        instead, so this is a cliff rather than a slope: a run one slice short of empty lands with
        a warning. There is no tunable fraction — the test is that *nothing* survives.

        Why an absent column is the one case worth discarding a run over:
        <https://openclimatefix.github.io/nged-substation-forecast/architecture/ecmwf-ens-known-issues/#a-wholly-missing-variable-and-instantaneous-nulls-fatal>.
        Why every smaller pattern is not — the slice arithmetic, and the interpolation argument
        that makes a tolerated slice survivable:
        <https://openclimatefix.github.io/nged-substation-forecast/architecture/ecmwf-ens-known-issues/#nulls-in-the-de-accumulated-variables-tolerated>.

        Two things a caller must not assume:

        - The judgement is made per `init_time`, so a run whose column is empty is caught even
          inside a frame holding other, healthy runs — but by the same token, a frame filtered down
          to nothing *but* a wholly-null slice is indistinguishable from an empty column and does
          raise, even though that same slice was deliberately landed when the whole run was
          validated. Latent rather than live today: the only production caller validates one whole
          run, and reads go through `scan_delta`/`set_model`, which do not validate.
        - Raising is not the end of the partition. :class:`NwpVariableWhollyMissing` is a distinct
          type because the `ecmwf_ens` asset retries it rather than failing outright.
        """
        slices_per_run = (
            dataframe.filter(pl.col("valid_time") > pl.col("init_time"))
            .group_by("init_time")
            .agg(n_slices=pl.struct("ensemble_member", "valid_time").n_unique())
        )
        wholly_missing = (
            _deaccumulated_null_breakdown(dataframe)
            .filter(pl.col("n_null") == pl.col("n_total"))
            .group_by("variable", "init_time")
            .agg(n_whole_null_slices=pl.len())
            # Per `init_time`: comparing a variable's wholly-null slice count against the frame's
            # *total* would let a wholly-missing run hide behind a healthy one in the same frame.
            .join(slices_per_run, on="init_time", how="inner")
            .filter(pl.col("n_whole_null_slices") == pl.col("n_slices"))
        )
        if wholly_missing.height:
            offenders = sorted(
                (row["variable"], str(row["init_time"]), row["n_slices"])
                for row in wholly_missing.iter_rows(named=True)
            )
            raise NwpVariableWhollyMissing(
                "De-accumulated NWP variable(s) null in every (ensemble_member, valid_time) slice "
                "beyond lead-0 of their run — the field is wholesale absent, not the "
                f"locally-corrupt scatter we tolerate. Offending (variable, init_time, n_slices): "
                f"{offenders}"
            )

    @classmethod
    def _check_unique(cls, dataframe: pt.DataFrame[Self]) -> None:
        if (
            dataframe.select(
                ["nwp_model_id", "init_time", "valid_time", "ensemble_member", "h3_index"]
            )
            .is_duplicated()
            .any()
        ):
            raise ValueError(
                "Duplicate entries found for "
                "(nwp_model_id, init_time, valid_time, ensemble_member, h3_index)."
            )

    @classmethod
    def _check_variables_that_were_introduced_after_start_of_dataset(
        cls, dataframe: pt.DataFrame[Self]
    ) -> None:
        """Check the introduction date of `categorical_precipitation_type_surface`.

        The column is all-null when init_time <= 2024-11-12, and is never null afterwards.

        Confirmed by direct inspection of the source data: the 2024-11-13 00Z run is the first
        run with `ptype` populated (0% null across all lead times), while 2024-11-12 and earlier
        are 100% null.
        """
        threshold_date = datetime(2024, 11, 12, tzinfo=UTC)

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
                "init_time <= 2024-11-12"
            )

        # Check after threshold
        if after["categorical_precipitation_type_surface"].is_null().any():
            raise ValueError(
                "categorical_precipitation_type_surface must not be null for init_time > 2024-11-12"
            )

    @classmethod
    def scan_delta(
        cls,
        path: str | Path | None = None,
        storage_options: ObjectStoreOptions | None = None,
    ) -> pt.LazyFrame[Self]:
        """Lazily scan the NWP Delta table, typed and cast to this contract's dtypes.

        The table stores physical-unit `Float32` directly (see `delta_store.nwp.write_nwp`),
        so no rescale step is needed.

        Args:
            path: Path or URI of the ``nwp`` Delta table; defaults to
                ``get_settings().nwp_data_path`` (resolved lazily so importing this module needs
                no ``.env``).
            storage_options: delta-rs object-store options (credentials/endpoint) for a remote
                ``path``; defaults to ``get_settings().storage_options`` (empty for a local
                ``path``).

        Returns:
            A lazy frame over the ``nwp`` Delta table, set to the ``Nwp`` model and cast to its
            declared dtypes — one row per (NWP model, init time, ensemble member, valid time,
            H3 cell), with the weather variables already in physical units.
        """
        settings = get_settings()
        if path is None:
            path = settings.nwp_data_path
        if storage_options is None:
            storage_options = settings.storage_options
        return (
            pt.LazyFrame.from_existing(
                pl.scan_delta(path, storage_options=typeddict_to_dict(storage_options))
            )
            .set_model(cls)
            .cast()
        )


def _deaccumulated_null_breakdown(dataframe: pl.DataFrame) -> pl.DataFrame:
    """Null- and total-cell counts per (variable, init_time, ensemble_member, valid_time).

    Covers every slice beyond lead-0.

    Returns only slices that have at least one null. Shared by the fatal wholly-missing-variable
    check and the non-fatal :func:`assess_nwp_quality`, so both agree on what a "null" is, and so
    the fatal case is exactly the extreme of what the warning reports. ``init_time`` is in the
    group key so the counts stay correct even on a multi-run frame (each grid cell is one row, so
    ``n_total`` is the slice's cell count). Operates on a single NWP run in practice (~1M rows), far
    below Polars' 2**32 row-count ceiling, so the counts are exact.
    """
    deaccumulated = sorted(Nwp.deaccumulated_var_names)
    beyond_lead0 = dataframe.filter(pl.col("valid_time") > pl.col("init_time"))
    return (
        beyond_lead0.unpivot(
            on=deaccumulated,
            index=["init_time", "ensemble_member", "valid_time"],
            variable_name="variable",
            value_name="value",
        )
        .group_by("variable", "init_time", "ensemble_member", "valid_time")
        .agg(n_null=pl.col("value").is_null().sum(), n_total=pl.len())
        .filter(pl.col("n_null") > 0)
    )


@dataclass(frozen=True)
class NwpQualityReport:
    """Non-fatal data-quality summary for one NWP run.

    Carries the *tolerated* nulls that the de-accumulated variables still hold after the H3 spatial
    aggregation, beyond lead-0. Usually that means whole (ensemble_member, valid_time) slices that
    arrived empty, plus the cells where upstream scatter happened to take out every contributing
    grid point; the two are counted separately because they warrant different responses, but
    neither fails the run. Only a variable that is null in *every* slice is fatal, and
    :meth:`Nwp.validate` rejects that before this runs.

    Read this as "how much did we lose", not as "how corrupt was the feed". The aggregation absorbs
    most per-pixel upstream corruption before it reaches a cell, so this is a poor proxy for the
    upstream null rate. That rate is measured where it lives, on the raw grid, by
    :class:`dynamical_data.ecmwf_ens.upstream_nulls.UpstreamNullRate`; the ``ecmwf_ens`` asset
    publishes both on one check, and they are not comparable as rates.
    """

    affected: pl.DataFrame
    """One row per affected (variable, init_time, ensemble_member, valid_time) slice, with
    ``n_null`` and ``n_total`` cell counts. Empty when the run is clean."""

    @property
    def n_null_cells(self) -> int:
        """Total null cells across all affected slices."""
        return int(self.affected["n_null"].sum()) if self.affected.height else 0

    @property
    def n_affected_slices(self) -> int:
        """Number of (variable, member, valid_time) slices carrying at least one null."""
        return self.affected.height

    @property
    def n_whole_null_slices(self) -> int:
        """Affected slices that arrived *entirely* null.

        The field is missing altogether for that one (variable, member, valid_time).

        Worth watching separately from the scattered case below: a rising count is the shape a
        partial upstream publication takes, and it is the number this report is best at, because a
        wholly-null slice reaches the cells intact however they are aggregated.
        """
        if not self.affected.height:
            return 0
        return int((self.affected["n_null"] == self.affected["n_total"]).sum())

    @property
    def n_scattered_slices(self) -> int:
        """Affected slices carrying only *some* null cells.

        Expect this to be small. Scattered upstream corruption is mostly absorbed by the H3
        aggregation, so a slice reaches this count only where the scatter took out every grid
        point of some cell. It is *not* a measure of the upstream per-pixel null rate.
        """
        return self.n_affected_slices - self.n_whole_null_slices

    @property
    def is_healthy(self) -> bool:
        """True when the run has no unexpected nulls."""
        return self.affected.height == 0

    @property
    def affected_variables(self) -> tuple[str, ...]:
        """The de-accumulated variables carrying unexpected nulls, sorted."""
        if not self.affected.height:
            return ()
        return tuple(sorted(self.affected["variable"].unique().to_list()))


def assess_nwp_quality(dataframe: pt.DataFrame[Nwp]) -> NwpQualityReport:
    """Summarise the tolerated-but-noteworthy nulls in a *validated* NWP run.

    Reports the nulls in the de-accumulated variables (precipitation/radiation) beyond lead-0 that
    :meth:`Nwp.validate` deliberately tolerates: whole (ensemble_member, valid_time) slices that
    arrived empty, and the cells where the upstream per-pixel corruption survived the H3
    aggregation by taking out every grid point of a cell. Pure and Dagster-free
    (unit-testable in isolation); the ``ecmwf_ens`` asset wraps the result into a WARN
    ``AssetCheckResult``. See
    <https://openclimatefix.github.io/nged-substation-forecast/architecture/ecmwf-ens-known-issues/>.
    """
    return NwpQualityReport(affected=_deaccumulated_null_breakdown(dataframe))


@dataclass(frozen=True)
class NwpRunCompletenessReport:
    """Run-level shape summary for one ingested NWP run.

    Answers a single question: is the whole (member x step x cell) grid there?

    Deliberately a *report* rather than an exception. A short run is the upstream provider
    misbehaving, not a contract violation, and the
    [never-raise-on-absent-input rule](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#the-rules)
    says we land what arrived and warn, rather than throwing away an otherwise-good run. The
    `ecmwf_ens` asset wraps this into a WARN, non-blocking `AssetCheckResult` and publishes the
    counts as materialisation metadata.
    """

    init_times: tuple[datetime, ...]
    """Distinct `init_time` values in the frame. A whole run has exactly one."""

    n_rows: int
    """Rows in the frame."""

    expected_n_rows: int
    """`len(expected_ensemble_members) * len(expected_lead_time_hours) * expected_n_h3_cells` —
    the size of the complete grid."""

    n_ensemble_members: int
    """Distinct `ensemble_member` values observed."""

    n_valid_times: int
    """Distinct `valid_time` values observed."""

    n_h3_cells: int
    """Distinct `h3_index` values observed."""

    expected_n_h3_cells: int
    """Distinct `h3_index` values the H3 grid weights say this run should cover."""

    valid_time_min: datetime | None
    """Earliest `valid_time`, or `None` for an empty frame."""

    valid_time_max: datetime | None
    """Latest `valid_time`, or `None` for an empty frame."""

    missing_ensemble_members: tuple[int, ...]
    """Expected members with no rows at all, sorted."""

    unexpected_ensemble_members: tuple[int, ...]
    """Observed members outside the expected set, sorted — the upstream ensemble changed shape."""

    missing_lead_time_hours: tuple[int, ...]
    """Expected forecast steps (hours after `init_time`) with no rows at all, sorted."""

    unexpected_valid_times: tuple[datetime, ...]
    """Observed `valid_time`s that are not on any expected forecast step, sorted."""

    @property
    def is_complete(self) -> bool:
        """True when the run carries the shape we expect.

        That is, when the frame is one run whose member set, forecast-step set, H3 cell *count*
        and row count all match the expectation.

        Cells are compared by count, not by set: the report never receives the expected `h3_index`
        values, only how many there should be. Substituting one cell for another would therefore
        pass. That is not a live gap, because the asset derives the expected count from the very H3
        grid weights the converter joins against — see
        <https://openclimatefix.github.io/nged-substation-forecast/architecture/ecmwf-ens-known-issues/>.
        """
        return (
            len(self.init_times) == 1
            and not self.missing_ensemble_members
            and not self.unexpected_ensemble_members
            and not self.missing_lead_time_hours
            and not self.unexpected_valid_times
            and self.n_h3_cells == self.expected_n_h3_cells
            and self.n_rows == self.expected_n_rows
        )

    @property
    def h3_cell_shortfall(self) -> int:
        """Expected H3 cells minus observed.

        The difference is *net*, so it is negative if the run carries extra cells and zero if one
        cell was swapped for another.
        """
        return self.expected_n_h3_cells - self.n_h3_cells

    def describe(self) -> str:
        """One human-readable sentence naming every gap, for the asset check's description."""
        gaps = list(self._describe_gaps())
        if not gaps:
            return (
                f"Complete run: {self.n_ensemble_members} ensemble members x "
                f"{self.n_valid_times} forecast steps x {self.n_h3_cells} H3 cells "
                f"= {self.n_rows} rows."
            )
        return "Incomplete NWP run — " + "; ".join(gaps) + "."

    def _describe_gaps(self) -> Iterator[str]:
        """Yield one clause per detected gap, in decreasing order of diagnostic value."""
        if len(self.init_times) != 1:
            yield f"expected exactly one init_time, found {len(self.init_times)}"
        if self.missing_ensemble_members:
            yield f"missing ensemble member(s) {_abbreviate(self.missing_ensemble_members)}"
        if self.unexpected_ensemble_members:
            yield f"unexpected ensemble member(s) {_abbreviate(self.unexpected_ensemble_members)}"
        if self.missing_lead_time_hours:
            yield f"missing lead time(s) in hours {_abbreviate(self.missing_lead_time_hours)}"
        if self.unexpected_valid_times:
            yield f"unexpected valid_time(s) {_abbreviate(self.unexpected_valid_times)}"
        if self.n_h3_cells != self.expected_n_h3_cells:
            yield f"{self.n_h3_cells} H3 cells, expected {self.expected_n_h3_cells}"
        if self.n_rows != self.expected_n_rows:
            yield f"{self.n_rows} rows, expected {self.expected_n_rows}"


_MAX_GAP_ITEMS_IN_DESCRIPTION: Final[int] = 10
"""Cap on how many missing members/steps `NwpRunCompletenessReport.describe` spells out.

A wholesale upstream outage can miss hundreds of steps; the exact counts stay in the report's
fields and the Dagster metadata, so the sentence only needs enough to start debugging.
"""


def _abbreviate(items: Sequence[object]) -> str:
    """Render a gap list, truncating past `_MAX_GAP_ITEMS_IN_DESCRIPTION` with a `+N more` tail."""
    shown = ", ".join(str(item) for item in items[:_MAX_GAP_ITEMS_IN_DESCRIPTION])
    overflow = len(items) - _MAX_GAP_ITEMS_IN_DESCRIPTION
    return f"[{shown}, +{overflow} more]" if overflow > 0 else f"[{shown}]"


def assess_nwp_run_completeness(
    dataframe: pt.DataFrame[Nwp],
    expected_n_h3_cells: int,
    expected_ensemble_members: frozenset[int] = ECMWF_ENS_ENSEMBLE_MEMBERS,
    expected_lead_time_hours: tuple[int, ...] = ECMWF_ENS_LEAD_TIME_HOURS,
) -> NwpRunCompletenessReport:
    """Summarise whether one ingested NWP run covers its full (member x step x cell) grid.

    Called from the `ecmwf_ens` asset, deliberately **not** from `Nwp.validate`: validation runs on
    arbitrary frames (filtered test fixtures, pruned scans, single-member training reads), whereas
    completeness is a property of one whole ingested run and would be false on every one of those.
    Pure and Dagster-free, so it is unit-testable in isolation.

    Never raises on a validated `Nwp` frame — not on an empty one, not on a multi-`init_time` one,
    not on one whose `valid_time`s are off-grid — so it cannot turn the warning path into a failure
    path (rule 7 of
    [Inherent Stability](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#the-rules)).
    The qualifier matters: the key columns being non-null and present is exactly what `Nwp.validate`
    guarantees, and the sole production caller validates before calling.

    Row counting is safe here despite Polars' 32-bit row index: this runs on a single in-memory run
    (at V1 scale 1671 cells x 51 members x 85 steps ~ 7.24M rows), four orders of magnitude below
    the 2**32 ceiling that the whole ~5.9B-row NWP Delta table would hit.

    Args:
        dataframe: One whole ingested NWP run, already through `Nwp.validate`.
        expected_n_h3_cells: Distinct H3 cells the run should cover — pass
            `h3_grid["h3_index"].n_unique()` so the expectation tracks the H3 grid weights rather
            than a hard-coded number.
        expected_ensemble_members: Members the run should carry. Defaults to the ECMWF ENS
            ensemble, the only `NwpModelId` we ingest today.
        expected_lead_time_hours: Forecast steps, in hours after `init_time`, the run should carry.
            Defaults to the ECMWF ENS native step structure.

    Returns:
        A `NwpRunCompletenessReport` giving the observed row count against the expected
        complete-grid row count, the observed versus expected ensemble members and H3 cell
        count, the earliest and latest `valid_time` (`None` for an empty frame), and which
        expected lead times are missing or which observed `valid_time`s are unexpected — the
        latter two both empty when the frame does not hold exactly one `init_time`.
    """
    init_times = tuple(sorted(dataframe["init_time"].unique().to_list()))
    observed_members = set(dataframe["ensemble_member"].unique().to_list())
    observed_valid_times = set(dataframe["valid_time"].unique().to_list())
    missing_lead_times, unexpected_valid_times = _lead_time_gaps(
        init_times, observed_valid_times, expected_lead_time_hours
    )
    return NwpRunCompletenessReport(
        init_times=init_times,
        n_rows=dataframe.height,
        expected_n_rows=(
            len(expected_ensemble_members) * len(expected_lead_time_hours) * expected_n_h3_cells
        ),
        n_ensemble_members=len(observed_members),
        n_valid_times=len(observed_valid_times),
        n_h3_cells=dataframe["h3_index"].n_unique(),
        expected_n_h3_cells=expected_n_h3_cells,
        valid_time_min=min(observed_valid_times, default=None),
        valid_time_max=max(observed_valid_times, default=None),
        missing_ensemble_members=tuple(sorted(expected_ensemble_members - observed_members)),
        unexpected_ensemble_members=tuple(sorted(observed_members - expected_ensemble_members)),
        missing_lead_time_hours=missing_lead_times,
        unexpected_valid_times=unexpected_valid_times,
    )


def _lead_time_gaps(
    init_times: tuple[datetime, ...],
    observed_valid_times: set[datetime],
    expected_lead_time_hours: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[datetime, ...]]:
    """Compare observed `valid_time`s against the expected forecast steps of a single run.

    Returns `(missing_lead_time_hours, unexpected_valid_times)`. Both are empty when the frame does
    not hold exactly one `init_time`, because there is then no single origin to measure lead times
    from; `NwpRunCompletenessReport.is_complete` reports that case through `init_times` instead.

    Compares whole `datetime`s rather than integer lead times, so a `valid_time` that is off-grid by
    minutes is reported rather than silently rounded onto a step.
    """
    if len(init_times) != 1:
        return (), ()
    init_time = init_times[0]
    expected_valid_times = {
        init_time + timedelta(hours=hours): hours for hours in expected_lead_time_hours
    }
    missing = tuple(
        sorted(
            hours
            for valid_time, hours in expected_valid_times.items()
            if valid_time not in observed_valid_times
        )
    )
    unexpected = tuple(sorted(observed_valid_times - set(expected_valid_times)))
    return missing, unexpected
