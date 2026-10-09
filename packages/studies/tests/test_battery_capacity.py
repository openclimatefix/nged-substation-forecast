from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest
from studies.battery_capacity import calendar_replica, cell_energy_path, smallest_capacity

_WINDOW_START = datetime(2025, 9, 1, tzinfo=UTC)


def _grid(*, days: int, start: datetime = _WINDOW_START) -> pl.Series:
    return pl.datetime_range(
        start + timedelta(minutes=30),
        start + timedelta(days=days),
        interval="30m",
        time_unit="us",
        eager=True,
    ).alias("half_hour_end_time")


def test_calendar_replica_matches_values_recorded_from_the_function_before_it_moved() -> None:
    grid = _grid(days=365)
    rng = np.random.default_rng(7)
    output = rng.normal(10, 3, len(grid)) + np.arange(len(grid)) % 48
    output[rng.random(len(grid)) < 0.03] = np.nan

    replica = calendar_replica(output=output, half_hour_end_time=grid)

    assert len(grid) == 17520
    assert np.nansum(replica) == pytest.approx(568564.339179083, rel=1e-12)
    assert int(np.isnan(replica).sum()) == 533
    assert replica[:3] == pytest.approx(
        [10.395655939698992, 10.429364591710696, 11.667688899029706]
    )
    assert replica[-3:] == pytest.approx(
        [55.406543743991776, 55.64292261704903, 56.094666143538646]
    )


def _reference_replica(*, output: np.ndarray, grid: pl.Series) -> np.ndarray:
    """Average by (month, local half-hour, day type) with the standard library's time zones."""
    cells: dict[tuple[int, int, int], list[float]] = {}
    keys = []
    for end, value in zip(grid.to_list(), output, strict=True):
        local = (end - timedelta(minutes=15)).astimezone(ZoneInfo("Europe/London"))
        weekday = local.isoweekday()
        key = (local.month, local.hour * 2 + local.minute // 30, weekday if weekday >= 6 else 0)
        keys.append(key)
        if np.isfinite(value):
            cells.setdefault(key, []).append(value)
    return np.array(
        [
            np.mean(cells[key]) if np.isfinite(value) else np.nan
            for key, value in zip(keys, output, strict=True)
        ]
    )


def test_calendar_replica_equals_a_slow_reference_across_the_end_of_daylight_saving() -> None:
    # Eight weeks around 26 October 2025, when the UK clock moves back an hour: the same UTC
    # half-hour belongs to a different local slot before and after the change.
    grid = _grid(days=56, start=datetime(2025, 10, 6, tzinfo=UTC))
    rng = np.random.default_rng(3)
    output = rng.normal(5.0, 2.0, len(grid))
    output[rng.random(len(grid)) < 0.05] = np.nan

    replica = calendar_replica(output=output, half_hour_end_time=grid)

    assert replica == pytest.approx(_reference_replica(output=output, grid=grid), nan_ok=True)
    assert len(np.unique(np.round(replica[np.isfinite(replica)], 9))) > 200


def test_calendar_replica_keeps_saturday_and_sunday_apart_from_weekdays() -> None:
    grid = _grid(days=14)
    local = grid.dt.offset_by("-15m").dt.convert_time_zone("Europe/London")
    weekday = local.dt.weekday().to_numpy()
    output = np.where(weekday == 6, 10.0, np.where(weekday == 7, 20.0, 1.0))

    replica = calendar_replica(output=output, half_hour_end_time=grid)

    assert set(np.round(replica[weekday == 6], 9)) == {10.0}
    assert set(np.round(replica[weekday == 7], 9)) == {20.0}
    assert set(np.round(replica[weekday <= 5], 9)) == {1.0}


def test_calendar_replica_leaves_a_missing_half_hour_missing() -> None:
    grid = _grid(days=7)
    output = np.ones(len(grid))
    output[5] = np.nan

    replica = calendar_replica(output=output, half_hour_end_time=grid)

    assert np.isnan(replica[5])
    assert not np.isnan(np.delete(replica, 5)).any()


def test_cell_energy_path_charges_by_eta_and_discharges_by_one_over_eta() -> None:
    path = cell_energy_path(output_mwh=np.array([-1.0, 1.0]), eta=0.8)

    assert path.tolist() == pytest.approx([0.8, 0.8 - 1.0 / 0.8])


def test_smallest_capacity_is_the_range_of_the_path_and_starts_it_at_zero() -> None:
    output = np.array([-4.0, -2.0, 3.0, 5.0, -1.0])
    eta = 0.9

    capacity, soc = smallest_capacity(output_mwh=output, eta=eta)

    path = cell_energy_path(output_mwh=output, eta=eta)
    assert capacity == pytest.approx(path.max() - path.min())
    assert soc.min() == 0.0
    assert soc.max() == pytest.approx(capacity)
