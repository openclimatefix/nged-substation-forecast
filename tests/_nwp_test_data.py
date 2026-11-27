"""Shared NWP test data and dtypes, for any test suite in the repo.

The several CV and metrics integration tests each build a synthetic ``Nwp``
frame. ``half_hours``, ``NWP_CONTINUOUS_COL_VALUES``, ``nwp_records`` and ``write_test_nwp`` live
here because they were byte-identical, or identical but for one parameter, across the test
modules that used to define them locally. What genuinely differs per test file is *which* cells,
days, member sets and (for a run initialised before the day it forecasts into) explicit
``init_time`` values get combined into one fixture — that selection is what stays local to each
test file, as a thin ``_write_nwp(path)`` wrapper around ``nwp_records`` and ``write_test_nwp``.
Importable by bare name via the ``pythonpath = ["tests"]`` pytest setting.

``cast_to_nwp_dtypes`` serves a second set of callers: the package test suites, whose fixtures
build a *partial* NWP frame that ``Nwp.validate`` cannot check.
"""

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Final

import polars as pl
from contracts.weather_schemas import ECMWF_ENS_LEAD_TIME_HOURS, Nwp, variables_known_empty_at
from deltalake import write_deltalake
from ml_core.features import NWP_PUBLICATION_DELAY_HOURS

NWP_CONTINUOUS_COL_VALUES: Final[dict[str, float]] = {
    "temperature_2m": 15.0,
    "dew_point_temperature_2m": 10.0,
    "wind_u_10m": 0.0,
    "wind_v_10m": 5.0,
    "wind_u_100m": 0.0,
    "wind_v_100m": 8.0,
    "pressure_surface": 101_000.0,
    "pressure_reduced_to_mean_sea_level": 101_500.0,
    "geopotential_height_500hpa": 5_500.0,
    "downward_long_wave_radiation_flux_surface": 300.0,
    "downward_short_wave_radiation_flux_surface": 200.0,
    "precipitation_surface": 0.001,
}
"""Physically plausible Float32 constants, one per continuous ``Nwp`` variable."""

_TOTAL_CLOUD_COVER_PERCENT: Final[float] = 50.0
"""The ``total_cloud_cover_atmosphere`` value that ``nwp_records`` gives a populated run."""


def cast_to_nwp_dtypes(frame: pl.DataFrame, *columns: str) -> pl.DataFrame:
    """Cast each named column of a partial NWP test fixture to the dtype `Nwp` declares for it.

    A test that feeds the feature pipeline a handful of NWP columns cannot call `Nwp.validate`,
    because the frame is missing most of the contract's fields. Spelling the dtypes out as literals
    instead lets a fixture drift from the contract in silence: three fixtures declared
    `categorical_precipitation_type_surface` as `UInt8` against the contract's `Int16`, and two
    declared `h3_index` as `UInt64` against `Int64`, so those tests exercised a frame the pipeline
    cannot produce. Casting through this helper makes the wrong dtype unwritable, and raises
    `KeyError` on a column `Nwp` does not declare, which also catches a typo in a column name.

    Name only the columns the fixture shares with `Nwp`. A column the contract does not declare —
    `nwp_init_time`, `time_series_id` — stays out of the call and is cast by the fixture instead.

    Args:
        frame: The fixture frame to cast.
        *columns: Column names to cast, each of which must be a field of `Nwp`.

    Returns:
        The frame, with each named column cast to `Nwp`'s dtype for it.
    """
    return frame.cast({column: Nwp.dtypes[column] for column in columns})


def half_hours(day: datetime) -> pl.Series:
    """Deliverable half-hours through the next native 3-hour boundary of a 00Z run.

    The start is derived from ``NWP_PUBLICATION_DELAY_HOURS`` rather than hard-coded: bulk mode
    stamps ``power_fcst_init_time = init_time + NWP_PUBLICATION_DELAY_HOURS`` and drops every
    earlier valid time as a hindcast, so a fixed window silently empties the fixture — and the
    asset then fails with "no usable rows" — if the delay ever grows.
    """
    init_time = day.replace(hour=0)
    first = init_time + timedelta(hours=NWP_PUBLICATION_DELAY_HOURS, minutes=30)
    end = init_time + timedelta(hours=(NWP_PUBLICATION_DELAY_HOURS // 3 + 1) * 3)
    return pl.datetime_range(first, end, interval="30m", time_zone="UTC", eager=True)


def nwp_records(
    cell: int,
    day: datetime,
    members: tuple[int, ...],
    init_time: datetime | None = None,
    valid_times: Sequence[datetime] | None = None,
) -> list[dict]:
    """Synthetic, contract-valid ``Nwp`` rows for one (cell, day, members) combination.

    One row per member and native forecast step, all sharing ``init_time``. The requested target
    times default to ``half_hours(day)``; native steps bracket that target interval using the
    ECMWF 3/6-hour schedule. Pass ``valid_times`` for a wider target interval. ``init_time``
    defaults to ``day`` at 00Z; an earlier initialisation exercises NWP lookback without inventing
    off-schedule native radiation windows.
    """
    resolved_init_time = init_time if init_time is not None else day.replace(hour=0)
    requested = list(valid_times if valid_times is not None else half_hours(day))
    schedule = [resolved_init_time + timedelta(hours=lead) for lead in ECMWF_ENS_LEAD_TIME_HOURS]
    first = max(time for time in schedule if time <= min(requested))
    last = min(time for time in schedule if time >= max(requested))
    resolved_valid_times = [time for time in schedule if first <= time <= last]
    # A variable is null inside its window in `NWP_VARIABLE_EMPTY_WINDOWS` and populated outside it,
    # which `Nwp.validate` enforces, so `nwp_records` sets each column to pass it.
    empty_at_init_time = variables_known_empty_at(resolved_init_time)
    ptype = None if "categorical_precipitation_type_surface" in empty_at_init_time else 0
    total_cloud = (
        None if "total_cloud_cover_atmosphere" in empty_at_init_time else _TOTAL_CLOUD_COVER_PERCENT
    )
    records = []
    for member in members:
        for valid_time in resolved_valid_times:
            record = {
                "nwp_model_id": "ECMWF_ENS_0_25_degree",
                "init_time": resolved_init_time,
                "valid_time": valid_time,
                "ensemble_member": member,
                "h3_index": cell,
                "categorical_precipitation_type_surface": ptype,
                "total_cloud_cover_atmosphere": total_cloud,
            }
            record.update(NWP_CONTINUOUS_COL_VALUES)
            if valid_time == resolved_init_time:
                record.update(dict.fromkeys(Nwp.deaccumulated_var_names))
            records.append(record)
    return records


def write_test_nwp(path: str, records: list[dict]) -> None:
    """Validate synthetic ``Nwp`` rows against the contract, then write them to a Delta table.

    Casts every column to the dtype the ``Nwp`` contract declares, continuous weather variables
    included (never the raw ints a hand-rolled sentinel could hide behind), and calls
    ``Nwp.validate`` so a dtype or range mistake in ``records`` raises here — loudly, and before
    the frame ever reaches the CV pipeline the caller's test is exercising — rather than silently
    training and scoring a model on data the contract would reject.

    This can't simply call ``delta_store.nwp.write_nwp``: that function also rounds every
    continuous variable to a 13-bit significand, which would silently perturb some of this
    fixture's hand-picked values (measured: ``pressure_surface`` 101000.0 -> 100992.0,
    ``pressure_reduced_to_mean_sea_level`` 101500.0 -> 101504.0, ``precipitation_surface`` rounds
    too), and applies writer properties (compression level, encoding) that exist to optimise the
    real table's on-disk size, not to serve a test fixture. Those two are deliberately *not*
    shared. What *is* shared with ``write_nwp`` — the ``(nwp_model_id, init_time)`` partitioning
    — is hand-copied below because ``write_nwp`` doesn't expose it as an importable constant. A
    change to the partition *column names* going stale here is caught by
    ``tests/test_nwp_test_data.py::test_partition_layout_matches_write_nwp``, which compares only
    ``partition_columns``.
    """
    df = pl.DataFrame(records)
    df = cast_to_nwp_dtypes(df, *df.columns)
    validated = Nwp.validate(df)
    # `partition_by` hand-copies `delta_store.nwp.write_nwp`'s layout — see the docstring above
    # for why this can't just call `write_nwp`, and `test_nwp_test_data.py` for the drift guard.
    write_deltalake(
        table_or_uri=path, data=validated.to_arrow(), partition_by=["nwp_model_id", "init_time"]
    )
