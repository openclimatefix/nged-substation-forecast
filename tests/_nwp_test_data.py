"""Shared NWP test data for the root integration tests.

The several ``live_forecasts`` / CV / metrics integration tests each build a synthetic ``Nwp``
frame. ``half_hours``, ``NWP_CONTINUOUS_COL_VALUES``, ``nwp_records`` and ``write_test_nwp`` live
here because they were byte-identical, or identical but for one parameter, across the test modules
that used to define them locally. What genuinely differs per test file is *which* cells, days,
member sets and (for a run initialised before the day it forecasts into) explicit ``init_time``
values get combined into one fixture — that selection is what stays local to each test file, as a
thin ``_write_nwp(path)`` wrapper around ``nwp_records`` and ``write_test_nwp``. Importable by bare
name via the ``pythonpath = ["tests"]`` pytest setting.
"""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Final

import polars as pl
from contracts.weather_schemas import Nwp
from deltalake import write_deltalake
from weather_utils import NWP_PUBLICATION_DELAY_HOURS

NWP_CONTINUOUS_COL_VALUES: Final[dict[str, float]] = {
    "temperature_2m": 15.0,
    "dew_point_temperature_2m": 10.0,
    "wind_speed_10m": 5.0,
    "wind_direction_10m": 180.0,
    "wind_speed_100m": 8.0,
    "wind_direction_100m": 180.0,
    "pressure_surface": 101_000.0,
    "pressure_reduced_to_mean_sea_level": 101_500.0,
    "geopotential_height_500hpa": 5_500.0,
    "downward_long_wave_radiation_flux_surface": 300.0,
    "downward_short_wave_radiation_flux_surface": 200.0,
    "precipitation_surface": 0.001,
}
"""Physically plausible Float32 constants, one per continuous ``Nwp`` variable."""

_PTYPE_INTRODUCED: Final[datetime] = datetime(2024, 11, 12, tzinfo=UTC)
"""``categorical_precipitation_type_surface`` must be null on or before this ``init_time`` and
populated after it — see ``Nwp._check_variables_that_were_introduced_after_start_of_dataset``.
``nwp_records`` sets the column to satisfy this so its output passes ``Nwp.validate``."""


def half_hours(day: datetime) -> pl.Series:
    """Half-hourly valid times inside a 00Z run's forecast window, after the publication delay.

    The start is derived from ``NWP_PUBLICATION_DELAY_HOURS`` rather than hard-coded: bulk mode
    stamps ``power_fcst_init_time = init_time + NWP_PUBLICATION_DELAY_HOURS`` and drops every
    earlier valid time as a hindcast, so a fixed window silently empties the fixture — and the
    asset then fails with "no usable rows" — if the delay ever grows.
    """
    first = day.replace(hour=0) + timedelta(hours=NWP_PUBLICATION_DELAY_HOURS + 1)
    return pl.datetime_range(
        first, first + timedelta(hours=2), interval="30m", time_zone="UTC", eager=True
    )


def nwp_records(
    cell: int,
    day: datetime,
    members: tuple[int, ...],
    init_time: datetime | None = None,
    valid_times: Sequence[datetime] | None = None,
) -> list[dict]:
    """Synthetic, contract-valid ``Nwp`` rows for one (cell, day, members) combination.

    One row per (member, valid_time), all sharing ``init_time``. ``valid_times`` defaults to
    ``half_hours(day)``; pass it explicitly for a fixture whose valid times need to span multiple
    days, e.g. to give a forecast a realistic multi-day horizon rather than the couple of hours
    ``half_hours`` produces. ``init_time`` defaults to ``day`` at 00Z; pass it explicitly for a run
    that was initialised on an earlier day and forecasts into ``day`` (e.g. to exercise an
    NWP-lookback window).
    """
    resolved_init_time = init_time if init_time is not None else day.replace(hour=0)
    resolved_valid_times = valid_times if valid_times is not None else half_hours(day)
    ptype = None if resolved_init_time <= _PTYPE_INTRODUCED else 0
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
            }
            record.update(NWP_CONTINUOUS_COL_VALUES)
            records.append(record)
    return records


def write_test_nwp(path: str, records: list[dict]) -> None:
    """Validate synthetic ``Nwp`` rows against the contract, then write them to a Delta table.

    Casts every continuous weather variable to the physical-unit ``Float32`` the ``Nwp`` contract
    declares (never the raw ints a hand-rolled sentinel could hide behind), and calls
    ``Nwp.validate`` so a dtype or range mistake in ``records`` raises here — loudly, and before
    the frame ever reaches the CV pipeline the caller's test is exercising — rather than silently
    training and scoring a model on data the contract would reject.

    This can't simply call ``delta_store.nwp.write_nwp``: that function also rounds every
    continuous variable to a 13-bit significand, which would silently perturb some of this
    fixture's hand-picked values (measured: ``pressure_surface`` 101000.0 -> 100992.0,
    ``pressure_reduced_to_mean_sea_level`` 101500.0 -> 101504.0, ``precipitation_surface`` rounds
    too), and applies writer properties (compression level, encoding) that exist to optimise the
    real table's on-disk size, not to serve a test fixture. Those two are deliberately *not*
    shared. What *is* shared with ``write_nwp`` — the ``(nwp_model_id, init_time)`` partitioning —
    is hand-copied below because ``write_nwp`` doesn't expose it as an importable constant. A
    change to the partition *column names* going stale here is caught by
    ``tests/test_nwp_test_data.py::test_partition_layout_matches_write_nwp``, which compares only
    ``partition_columns``.
    """
    df = pl.DataFrame(records).cast(
        {
            "nwp_model_id": pl.String,
            "init_time": pl.Datetime("us", "UTC"),
            "valid_time": pl.Datetime("us", "UTC"),
            "ensemble_member": pl.Int8,
            "h3_index": pl.Int64,
            "categorical_precipitation_type_surface": pl.Int16,
            **dict.fromkeys(NWP_CONTINUOUS_COL_VALUES, pl.Float32),
        }
    )
    validated = Nwp.validate(df)
    # `partition_by` hand-copies `delta_store.nwp.write_nwp`'s layout — see the docstring above
    # for why this can't just call `write_nwp`, and `test_nwp_test_data.py` for the drift guard.
    write_deltalake(
        table_or_uri=path, data=validated.to_arrow(), partition_by=["nwp_model_id", "init_time"]
    )
