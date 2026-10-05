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
    cells = box_cells(lat_min=9.9, lat_max=10.1, lon_min=19.9, lon_max=20.1)
    assert {lat for lat, _ in cells} == {39, 40, 41}
    assert {lon for _, lon in cells} == {79, 80, 81}


def test_box_cells_add_no_row_for_an_edge_on_a_grid_line() -> None:
    cells = box_cells(lat_min=10.0, lat_max=10.25, lon_min=20.0, lon_max=20.25)
    assert sorted(cells) == [(40, 80), (40, 81), (41, 80), (41, 81)]


def test_block_cells_offsets_match_the_cells() -> None:
    block = block_cells(centre=(40, 80))
    assert len(block) == 9
    assert all(cell == (40 + dy, 80 + dx) for dy, dx, cell in block)
    assert {dy for dy, _, _ in block} == {-1, 0, 1}


def test_quarter_degree_index_rounds_to_the_nearest_grid_line() -> None:
    assert quarter_degree_index(degrees=10.1) == 40
    assert quarter_degree_index(degrees=10.2) == 41
    assert quarter_degree_index(degrees=19.9) == 80


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


def test_cell_plan_offsets_check_fails_when_an_offset_disagrees_with_its_cell() -> None:
    good = pl.DataFrame(
        {
            "kind": ["wind", "wind"],
            "label": ["W1", "W1"],
            "dy": [0, 1],
            "dx": [0, 0],
            "lat_q": [40, 41],
            "lon_q": [0, 0],
        }
    )
    flipped = good.with_columns(lat_q=pl.Series([40, 39]))
    assert validator._check_cell_plan_offsets(groups=good)[0]
    assert not validator._check_cell_plan_offsets(groups=flipped)[0]


def _old_file(*, path: Path, labels: tuple[str, ...]) -> pl.DataFrame:
    """Write a one-month old-style wind file whose series all hold the same values per hour."""
    hours = pl.datetime_range(
        datetime(2024, 1, 1), datetime(2024, 1, 31, 23), interval="1h", eager=True
    )
    rng = np.random.default_rng(2)
    values = {
        n: rng.normal(3.0, 1.0, hours.len()).astype(np.float32) for n in validator.VALUE_COLUMNS
    }
    frames = [
        pl.DataFrame({"site": [label] * hours.len(), "dy": 0, "dx": 0, "time": hours}).with_columns(
            pl.Series(n, v) for n, v in values.items()
        )
        for label in labels
    ]
    old = pl.concat(frames)
    old.write_parquet(path)
    return old


def _groups_sharing_a_cell() -> pl.DataFrame:
    """Two wind-farm labels whose centre cells are the same cell."""
    return pl.DataFrame(
        {
            "kind": ["wind", "wind"],
            "label": ["W1", "W2"],
            "dy": [0, 0],
            "dx": [0, 0],
            "cell_id": [7, 7],
        },
        schema_overrides={"cell_id": pl.Int32},
    )


def test_old_series_collapses_blocks_that_share_a_cell(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(validator, "OLD_WIND_PATH", tmp_path / "old.parquet")
    raw = _old_file(path=tmp_path / "old.parquet", labels=("W1", "W2"))
    old = validator._old_series(groups=_groups_sharing_a_cell())
    assert old.height == raw.height // 2
    assert old.select("cell_id", "time").is_duplicated().sum() == 0
    # An exact re-fetch holds each cell once, and must pass the bit-equality check.
    assert validator._check_overlap(overlap=old, old=old)[0]


def test_old_series_raises_when_blocks_sharing_a_cell_disagree(
    tmp_path: Path,
    monkeypatch,  # noqa: ANN001
) -> None:
    monkeypatch.setattr(validator, "OLD_WIND_PATH", tmp_path / "old.parquet")
    raw = _old_file(path=tmp_path / "old.parquet", labels=("W1", "W2"))
    raw.with_columns(
        pl.when(pl.col("site") == "W2")
        .then(pl.col("u10") + 1)
        .otherwise(pl.col("u10"))
        .alias("u10")
    ).write_parquet(tmp_path / "old.parquet")
    with pytest.raises(ValueError, match="different values"):
        validator._old_series(groups=_groups_sharing_a_cell())


def test_overlap_check_fails_on_a_duplicated_old_key() -> None:
    """Without de-duplication the old frame has twice the rows of a perfect re-fetch."""
    january = _frame(cell_ids=(7,)).filter(pl.col("time") < datetime(2020, 1, 1, tzinfo=UTC))
    doubled = pl.concat([january, january])
    assert not validator._check_overlap(overlap=january, old=doubled)[0]


def _station_inputs(
    *, lag: int, shuffle_cells: bool
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    frame = _frame(cell_ids=(0, 1, 2))
    groups = pl.DataFrame(
        {
            "kind": ["station"] * 3,
            "label": ["S0", "S1", "S2"],
            "dy": [0, 0, 0],
            "dx": [0, 0, 0],
            "cell_id": pl.Series([0, 1, 2], dtype=pl.Int32),
        }
    )
    truth = frame.select(
        "cell_id", "time", validator._speed(frame=frame, height="10").alias("wind_speed_m_s")
    )
    mapping = {0: "S1", 1: "S2", 2: "S0"} if shuffle_cells else {0: "S0", 1: "S1", 2: "S2"}
    observations = truth.with_columns(
        pl.col("cell_id").replace_strict(mapping, return_dtype=pl.String).alias("src_id"),
        pl.col("time") - pl.duration(hours=lag),
    ).select("src_id", "time", "wind_speed_m_s")
    return frame, groups, observations


def test_station_correlation_passes_on_matching_series_and_fails_on_a_shift_or_a_swap() -> None:
    frame, groups, observations = _station_inputs(lag=0, shuffle_cells=False)
    assert validator._check_station_correlation(
        frame=frame, groups=groups, observations=observations
    )[0]
    frame, groups, shifted = _station_inputs(lag=1, shuffle_cells=False)
    assert not validator._check_station_correlation(
        frame=frame, groups=groups, observations=shifted
    )[0]
    frame, groups, swapped = _station_inputs(lag=0, shuffle_cells=True)
    assert not validator._check_station_correlation(
        frame=frame, groups=groups, observations=swapped
    )[0]


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
            "latitude": [10.5, 10.25, 10.0],
            "longitude": [19.5, 19.75, 20.0, 20.25],
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
        {"cell_id": [0, 1], "lat_q": [42, 40], "lon_q": [79, 81]},
        schema={"cell_id": pl.Int32, "lat_q": pl.Int16, "lon_q": pl.Int16},
    )
    kept = fetch._read_zip(zip_path=tmp_path / "x.zip", cells=cells, chunk=chunk)
    assert kept.height == 2 * chunk.n_hours
    first = kept.filter(pl.col("cell_id") == 0).row(0, named=True)
    assert first["time"] == datetime(2020, 2, 1, tzinfo=UTC)
    # Cell 0 is latitude 10.5 (first row), longitude 19.75 (second column).
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
