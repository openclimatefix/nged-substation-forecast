"""Tests for `studies/weather_downloads/era5_cells.py` and the 2019 to 2023 ERA5 wind validator.

No test touches the network or `data/`. Each test fails on the bug it exists for: a chunk that
crosses a year or requests months outside its range, an hour count that is not the exact length of
the product, a box that stops short of the grid line enclosing its edge, a block whose offsets
disagree with its cells, a cluster that splits a chain of nearby cells, and validator checks that
pass on a duplicated hour or a re-fetch that differs by one bit.
"""

import importlib
import sys
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import pytest
import xarray as xr

SCRIPT_DIR: Final[Path] = Path(__file__).parent.parent / "studies" / "weather_downloads"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validate_era5_wind_2019_2023 as validator  # noqa: E402
from era5_cells import (  # noqa: E402
    Chunk,
    block_cells,
    bounding_area,
    box_cells,
    cluster_cells,
    half_year_chunks,
    quarter_degree_index,
    request_body,
)


def test_half_year_chunks_cover_the_product_without_crossing_a_year() -> None:
    chunks = half_year_chunks(first=(2019, 9), last=(2023, 12))
    assert len(chunks) == 9
    assert chunks[0] == Chunk(year=2019, first_month=9, last_month=12)
    assert chunks[1] == Chunk(year=2020, first_month=1, last_month=6)
    assert chunks[-1] == Chunk(year=2023, first_month=7, last_month=12)
    assert sum(chunk.n_hours for chunk in chunks) == validator.EXPECTED_HOURS == 37_992


def test_chunk_hours_follow_the_calendar() -> None:
    assert Chunk(year=2020, first_month=2, last_month=2).n_hours == 29 * 24
    assert Chunk(year=2021, first_month=2, last_month=2).n_hours == 28 * 24


def test_request_body_lists_only_the_chunks_months() -> None:
    body = request_body(
        chunk=Chunk(year=2019, first_month=9, last_month=12),
        variables=["a"],
        area=[1.0, 0.0, 0.0, 1.0],
    )
    assert body["year"] == ["2019"]
    assert body["month"] == ["09", "10", "11", "12"]
    assert len(body["time"]) == 24


def test_box_cells_enclose_an_edge_that_is_off_the_grid() -> None:
    cells = box_cells(lat_min=52.9, lat_max=53.1, lon_min=-0.1, lon_max=0.1)
    assert {lat for lat, _ in cells} == {211, 212, 213}
    assert {lon for _, lon in cells} == {-1, 0, 1}


def test_box_cells_add_no_row_for_an_edge_on_a_grid_line() -> None:
    cells = box_cells(lat_min=53.0, lat_max=53.25, lon_min=0.0, lon_max=0.25)
    assert sorted(cells) == [(212, 0), (212, 1), (213, 0), (213, 1)]


def test_block_cells_offsets_match_the_cells() -> None:
    block = block_cells(centre=(200, -4))
    assert len(block) == 9
    assert all(cell == (200 + dy, -4 + dx) for dy, dx, cell in block)
    assert {dy for dy, _, _ in block} == {-1, 0, 1}


def test_quarter_degree_index_rounds_to_the_nearest_grid_line() -> None:
    assert quarter_degree_index(degrees=53.1) == 212
    assert quarter_degree_index(degrees=53.2) == 213
    assert quarter_degree_index(degrees=-0.1) == 0


def test_cluster_cells_joins_a_chain_and_splits_a_distant_cell() -> None:
    chain = {(0, 0), (0, 3), (0, 6)}
    far = {(100, 100)}
    groups = cluster_cells(cells=chain | far, link_cells=3)
    assert groups == [chain, far]


def test_bounding_area_is_north_west_south_east() -> None:
    assert bounding_area(cells={(4, 0), (8, 4), (6, -2)}) == [2.0, -0.5, 1.0, 1.0]


# --------------------------------------------------------------------------------------------------
# Validator checks, on frames small enough to read
# --------------------------------------------------------------------------------------------------


def _frame(*, cell_ids: tuple[int, ...] = (0, 1)) -> pl.DataFrame:
    hours = pl.datetime_range(validator.FIRST_HOUR, validator.LAST_HOUR, interval="1h", eager=True)
    rng = np.random.default_rng(0)
    frames = []
    for cell_id in cell_ids:
        values = {
            name: rng.normal(3.0, 1.0, hours.len()).astype(np.float32)
            for name in validator.VALUE_COLUMNS
        }
        frames.append(
            pl.DataFrame(
                {"cell_id": pl.Series([cell_id] * hours.len(), dtype=pl.Int32), "time": hours}
            ).with_columns(pl.Series(name, vals) for name, vals in values.items())
        )
    return pl.concat(frames)


def test_row_count_check_fails_when_one_hour_is_dropped() -> None:
    frame = _frame()
    cells = pl.DataFrame({"cell_id": [0, 1]})
    assert validator._check_rows(frame=frame, cells=cells)[0]
    assert not validator._check_rows(frame=frame.slice(1), cells=cells)[0]


def test_duplicate_check_fails_on_a_repeated_hour() -> None:
    frame = _frame()
    assert validator._check_duplicates(frame=frame)[0]
    assert not validator._check_duplicates(frame=pl.concat([frame, frame.head(1)]))[0]


def test_time_axis_check_fails_on_a_gap() -> None:
    frame = _frame()
    gapped = frame.filter(pl.col("time") != datetime(2021, 5, 5, 12, tzinfo=UTC))
    assert validator._check_time_axis(frame=frame)[0]
    assert not validator._check_time_axis(frame=gapped)[0]


def test_stuck_run_check_fails_on_a_frozen_value() -> None:
    frame = _frame()
    assert validator._check_stuck_runs(frame=frame)[0]
    mask = (pl.col("cell_id") == 0) & (pl.col("time").dt.date() == datetime(2020, 1, 1).date())
    frozen = frame.with_columns(
        pl.when(mask).then(1.0).otherwise(pl.col(name)).cast(pl.Float32).alias(name)
        for name in validator.VALUE_COLUMNS
    )
    assert not validator._check_stuck_runs(frame=frozen)[0]


def test_overlap_check_fails_on_a_one_bit_difference() -> None:
    january = _frame().filter(pl.col("time") < datetime(2020, 1, 1, tzinfo=UTC))
    old = january.clone()
    assert validator._check_overlap(overlap=january, old=old)[0]
    values = january["u10"].to_numpy().copy()
    values[5] = np.nextafter(values[5], np.float32(100.0))
    bumped = january.with_columns(u10=pl.Series(values))
    assert not validator._check_overlap(overlap=bumped, old=old)[0]


def test_orientation_check_fails_when_an_offset_disagrees_with_its_cell() -> None:
    good = pl.DataFrame(
        {
            "kind": ["wind", "wind"],
            "label": ["W1", "W1"],
            "dy": [0, 1],
            "dx": [0, 0],
            "lat_q": [200, 201],
            "lon_q": [0, 0],
        }
    )
    flipped = good.with_columns(lat_q=pl.Series([200, 199]))
    assert validator._check_orientation(groups=good)[0]
    assert not validator._check_orientation(groups=flipped)[0]


# --------------------------------------------------------------------------------------------------
# Reading a downloaded zip
# --------------------------------------------------------------------------------------------------


def _write_zip(*, path: Path, chunk: Chunk) -> xr.Dataset:
    times = pl.datetime_range(
        datetime(chunk.year, chunk.first_month, 1),
        datetime(chunk.year, chunk.first_month, 1) + timedelta(hours=chunk.n_hours - 1),
        interval="1h",
        eager=True,
    ).to_numpy()
    rng = np.random.default_rng(1)
    shape = (len(times), 3, 4)
    ds = xr.Dataset(
        {
            name: (
                ("valid_time", "latitude", "longitude"),
                rng.normal(size=shape).astype("float32"),
            )
            for name in ("u10", "v10", "u100", "v100")
        },
        coords={
            "valid_time": times,
            "latitude": [53.5, 53.25, 53.0],
            "longitude": [-0.5, -0.25, 0.0, 0.25],
        },
    )
    netcdf = path.parent / "a.nc"
    ds.to_netcdf(netcdf)
    with zipfile.ZipFile(path, "w") as archive:
        archive.write(netcdf, "data_stream-oper_stepType-instant.nc")
    return ds


@pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array:DeprecationWarning")
def test_read_zip_keeps_the_right_cell_and_value(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    pytest.importorskip("cdsapi")
    pytest.importorskip("netCDF4")
    fetch = importlib.import_module("fetch_era5_wind_2019_2023")
    monkeypatch.setattr(fetch, "SCRATCH_DIR", tmp_path)
    chunk = Chunk(year=2020, first_month=2, last_month=2)
    ds = _write_zip(path=tmp_path / "x.zip", chunk=chunk)
    cells = pl.DataFrame(
        {"cell_id": [0, 1], "lat_q": [214, 212], "lon_q": [-1, 1]},
        schema={"cell_id": pl.Int32, "lat_q": pl.Int16, "lon_q": pl.Int16},
    )
    kept = fetch._read_zip(zip_path=tmp_path / "x.zip", cells=cells, chunk=chunk)
    assert kept.height == 2 * chunk.n_hours
    first = kept.filter(pl.col("cell_id") == 0).row(0, named=True)
    assert first["time"] == datetime(2020, 2, 1, tzinfo=UTC)
    # Cell 0 is latitude 53.5 (first row), longitude -0.25 (second column).
    assert first["u10"] == ds["u10"].isel(valid_time=0, latitude=0, longitude=1).item()


@pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array:DeprecationWarning")
def test_read_zip_raises_when_a_requested_cell_is_outside_the_download(
    tmp_path: Path,
    monkeypatch,  # noqa: ANN001
) -> None:
    pytest.importorskip("cdsapi")
    pytest.importorskip("netCDF4")
    fetch = importlib.import_module("fetch_era5_wind_2019_2023")
    monkeypatch.setattr(fetch, "SCRATCH_DIR", tmp_path)
    chunk = Chunk(year=2020, first_month=2, last_month=2)
    _write_zip(path=tmp_path / "x.zip", chunk=chunk)
    cells = pl.DataFrame(
        {"cell_id": [0], "lat_q": [300], "lon_q": [0]},
        schema={"cell_id": pl.Int32, "lat_q": pl.Int16, "lon_q": pl.Int16},
    )
    with pytest.raises(RuntimeError, match="expected"):
        fetch._read_zip(zip_path=tmp_path / "x.zip", cells=cells, chunk=chunk)
