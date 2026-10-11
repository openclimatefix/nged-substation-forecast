"""Tests for the ARCO-ERA5 reader's pure functions.

No test touches the network or `data/`. Each test fails on the bug it exists for: a cell written
under another cell's coordinates, a western longitude looked up on the wrong side of the meridian,
a month that loses or gains an hour at the span's ends, a table check that lets a NaN, a negative
accumulation, or a duplicate row through, and a resume that trusts a file with the wrong row count.
"""

from datetime import UTC, datetime
from pathlib import Path

import fetch_era5_solar_arco as arco
import numpy as np
import polars as pl
import pytest
from era5_solar_variables import VARIABLES_BY_NAME
from studies.era5_grid import GRID_LATITUDES, GRID_LONGITUDES


def test_fields_to_frame_puts_each_value_under_its_own_cell_and_hour() -> None:
    hours = [np.datetime64("2025-06-01T00", "h"), np.datetime64("2025-06-01T01", "h")]
    n_lat, n_lon = len(GRID_LATITUDES), len(GRID_LONGITUDES)
    fields = np.array(
        [[[h * 100 + i * 10 + j for j in range(n_lon)] for i in range(n_lat)] for h in range(2)],
        dtype=np.float32,
    )
    frame = arco.fields_to_frame(fields=fields, hours=hours)
    assert frame.height == 2 * arco.CELL_COUNT
    assert frame.schema["time"] == pl.Datetime("us", "UTC")
    for h, hour in enumerate(hours):
        for i, latitude in enumerate(GRID_LATITUDES):
            for j, longitude in enumerate(GRID_LONGITUDES):
                value = frame.filter(
                    pl.col("time") == datetime.fromisoformat(f"{hour}").replace(tzinfo=UTC),
                    pl.col("latitude") == latitude,
                    pl.col("longitude") == longitude,
                )["value"].item()
                assert value == h * 100 + i * 10 + j


def test_indices_finds_western_longitudes_on_a_0_to_360_axis() -> None:
    longitude_axis = np.arange(0.0, 360.0, 0.25)
    latitude_axis = np.arange(90.0, -90.25, -0.25)
    lon_index = arco.ArcoStore._indices(longitude_axis, tuple(x % 360 for x in GRID_LONGITUDES))
    lat_index = arco.ArcoStore._indices(latitude_axis, GRID_LATITUDES)
    assert longitude_axis[lon_index].tolist() == [359.5, 359.75, 0.0, 0.25, 0.5]
    assert latitude_axis[lat_index].tolist() == list(GRID_LATITUDES)
    with pytest.raises(ValueError, match="not on the ARCO grid"):
        arco.ArcoStore._indices(longitude_axis, (0.1,))


def test_month_hours_cover_the_span_exactly() -> None:
    months = arco.month_starts(first=arco.FIRST_HOUR, last=arco.LAST_HOUR)
    hours = [hour for month in months for hour in arco.month_hours(month_start=month)]
    assert hours[0] == np.datetime64(arco.FIRST_HOUR.replace(tzinfo=None), "h")
    assert hours[-1] == np.datetime64(arco.LAST_HOUR.replace(tzinfo=None), "h")
    assert len(hours) == len(set(hours))
    assert np.all(np.diff(np.array(hours)) == np.timedelta64(1, "h"))
    december = arco.month_hours(month_start=datetime(2019, 12, 1, tzinfo=UTC))
    assert len(december) == 31 * 24


def _table(*, values: list[float], n_hours: int = 1) -> pl.DataFrame:
    hours = [np.datetime64("2025-06-01T00", "h") + np.timedelta64(h, "h") for h in range(n_hours)]
    fields = np.array(values, dtype=np.float32).reshape(
        n_hours, len(GRID_LATITUDES), len(GRID_LONGITUDES)
    )
    return arco.fields_to_frame(fields=fields, hours=hours)


def test_check_table_passes_float_rounding_below_zero_on_an_accumulation() -> None:
    values = [1.0e6] * (arco.CELL_COUNT - 1) + [-0.25]
    record = arco.check_table(
        frame=_table(values=values), variable=VARIABLES_BY_NAME["ssrdc"], expected_rows=20
    )
    assert record["problems"] == []


def test_check_table_flags_a_real_negative_accumulation() -> None:
    values = [1.0e6] * (arco.CELL_COUNT - 1) + [-100.0]
    record = arco.check_table(
        frame=_table(values=values), variable=VARIABLES_BY_NAME["ssrdc"], expected_rows=20
    )
    assert any("below the limit" in problem for problem in list(record["problems"]))  # ty: ignore[invalid-argument-type]


def test_check_table_allows_nan_only_where_nan_is_a_value() -> None:
    values = [10.0] * (arco.CELL_COUNT - 1) + [float("nan")]
    cin = arco.check_table(
        frame=_table(values=values), variable=VARIABLES_BY_NAME["cin"], expected_rows=20
    )
    tcwv = arco.check_table(
        frame=_table(values=values), variable=VARIABLES_BY_NAME["tcwv"], expected_rows=20
    )
    assert cin["problems"] == []
    assert tcwv["problems"] == ["1 NaN values"]


def test_check_table_flags_duplicates_and_wrong_counts() -> None:
    frame = _table(values=[0.5] * arco.CELL_COUNT)
    record = arco.check_table(
        frame=pl.concat([frame, frame]), variable=VARIABLES_BY_NAME["tcc"], expected_rows=40
    )
    assert any("duplicate" in problem for problem in list(record["problems"]))  # ty: ignore[invalid-argument-type]
    assert any("distinct hours" in problem for problem in list(record["problems"]))  # ty: ignore[invalid-argument-type]


def test_check_month_nans_raises_on_a_missing_chunk_for_a_never_missing_variable() -> None:
    frame = _table(values=[float("nan")] * arco.CELL_COUNT)
    with pytest.raises(ValueError, match="NaN"):
        arco.check_month_nans(frame=frame, variable=VARIABLES_BY_NAME["tcwv"])
    arco.check_month_nans(frame=frame, variable=VARIABLES_BY_NAME["cin"])


def test_month_file_is_valid_rejects_wrong_row_counts_and_corrupt_files(tmp_path: Path) -> None:
    path = tmp_path / "2025-06.parquet"
    _table(values=[0.5] * arco.CELL_COUNT).write_parquet(path)
    assert arco.month_file_is_valid(path=path, expected_rows=20)
    assert not arco.month_file_is_valid(path=path, expected_rows=40)
    path.write_bytes(path.read_bytes()[:50])
    assert not arco.month_file_is_valid(path=path, expected_rows=20)
    assert not arco.month_file_is_valid(path=tmp_path / "absent.parquet", expected_rows=20)
