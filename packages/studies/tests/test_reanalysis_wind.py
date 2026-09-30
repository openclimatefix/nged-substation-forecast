from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from pyproj import Transformer
from studies.reanalysis_wind import (
    CERRA_FILES,
    NORA3_GRID_SPACING_M,
    NORA3_GRID_X0_M,
    NORA3_GRID_Y0_M,
    NORA3_LAMBERT_PROJ4,
    centred_hourly_power,
    check_strictly_inside,
    derive_nearest_cells,
    derive_nearest_nora3_cells,
    join_centred_power,
    read_cerra_wind,
    read_nora3_wind,
)

# A crop of 5 rows (y 10 to 14) and 4 columns (x 20 to 23): only y 11 to 13 and x 21 to 22 are
# strictly inside it.
CROP_Y = range(10, 15)
CROP_X = range(20, 24)
DAY = datetime(2024, 1, 1)


def _crop() -> pl.DataFrame:
    return pl.DataFrame(
        [{"y_index": y, "x_index": x} for y in CROP_Y for x in CROP_X],
        schema={"y_index": pl.Int64, "x_index": pl.Int64},
    )


def _cells(*, rows: dict[str, tuple[int, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "site": list(rows),
            "y_index": [y for y, _ in rows.values()],
            "x_index": [x for _, x in rows.values()],
        }
    )


def _write_cerra(*, directory: Path, hours: list[int]) -> None:
    # A cell's speed is its height in metres plus its column index, so a value names its height.
    for height_m, name in CERRA_FILES.items():
        pl.DataFrame(
            [
                {
                    "valid_time": DAY + timedelta(hours=hour),
                    "y_index": y,
                    "x_index": x,
                    "wind_speed_m_s": float(height_m + x),
                }
                for hour in hours
                for y in CROP_Y
                for x in CROP_X
            ],
            schema={
                "valid_time": pl.Datetime("ns"),
                "y_index": pl.Int64,
                "x_index": pl.Int64,
                "wind_speed_m_s": pl.Float32,
            },
        ).write_parquet(directory / name)


def _write_nora3(*, path: Path, hours: list[int]) -> None:
    # A cell's speed is its height in metres plus its column index, and its direction is ten times
    # that, so a value names its height and its column.
    pl.DataFrame(
        [
            {
                "time": DAY + timedelta(hours=hour),
                "height_m": height_m,
                "y_index": y,
                "x_index": x,
                "wind_speed_m_s": float(height_m + x),
                "wind_direction_deg": float(10 * (height_m + x)),
            }
            for hour in hours
            for height_m in (50, 100)
            for y in CROP_Y
            for x in CROP_X
        ],
        schema={
            "time": pl.Datetime("ms"),
            "height_m": pl.Int16,
            "y_index": pl.Int16,
            "x_index": pl.Int16,
            "wind_speed_m_s": pl.Float32,
            "wind_direction_deg": pl.Float32,
        },
    ).write_parquet(path)


def _half_hourly(*, site: str, hours: int) -> pl.DataFrame:
    # Half-hours end at 00:30, 01:00, ..., and each reads its own index in megawatts.
    return pl.DataFrame(
        {
            "site": site,
            "time": [
                datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=30 * (i + 1))
                for i in range(2 * hours)
            ],
            "power_mw": [float(i) for i in range(2 * hours)],
        }
    )


def test_nearest_cells_pick_the_grid_cell_beside_each_site():
    grid = pl.DataFrame(
        {
            "y_index": [0, 0, 1, 1],
            "x_index": [0, 1, 0, 1],
            "latitude": [50.0, 50.0, 51.0, 51.0],
            "longitude": [355.0, 356.0, 355.0, 356.0],
        }
    )
    sites = pl.DataFrame(
        {"site": ["W1", "W2"], "latitude": [50.9, 50.1], "longitude": [-4.9, -3.9]}
    )

    nearest = derive_nearest_cells(grid=grid, sites=sites).sort("site")

    assert nearest["y_index"].to_list() == [1, 0]
    assert nearest["x_index"].to_list() == [0, 1]


def test_nora3_nearest_cell_recovers_the_cell_a_site_sits_beside():
    to_degrees = Transformer.from_crs(NORA3_LAMBERT_PROJ4, "EPSG:4326", always_xy=True)
    wanted = {"W1": (250, 620), "W2": (230, 640)}
    rows = []
    for site, (y, x) in wanted.items():
        longitude, latitude = to_degrees.transform(
            NORA3_GRID_X0_M + NORA3_GRID_SPACING_M * x + 400.0,
            NORA3_GRID_Y0_M + NORA3_GRID_SPACING_M * y - 300.0,
        )
        rows.append({"site": site, "latitude": latitude, "longitude": longitude})

    nearest = derive_nearest_nora3_cells(sites=pl.DataFrame(rows))

    assert list(zip(nearest["y_index"], nearest["x_index"], strict=True)) == list(wanted.values())
    assert nearest["distance_km"].max() == pytest.approx(0.5, abs=0.01)


def test_cells_strictly_inside_the_crop_pass():
    check_strictly_inside(cells=_cells(rows={"W1": (11, 21), "W2": (13, 22)}), crop=_crop())


def test_a_cell_on_the_crop_edge_raises_without_naming_the_site():
    cells = _cells(rows={"W1": (11, 21), "W2": (12, 23), "W3": (10, 21)})

    with pytest.raises(ValueError, match="only 1 of 3 sites") as error:
        check_strictly_inside(cells=cells, crop=_crop())

    assert "W2" not in str(error.value)


def test_a_cell_outside_the_crop_raises():
    with pytest.raises(ValueError, match="only 1 of 2 sites"):
        check_strictly_inside(cells=_cells(rows={"W1": (12, 22), "W2": (40, 90)}), crop=_crop())


def test_cerra_pivot_keeps_the_right_speed_per_height(tmp_path: Path):
    _write_cerra(directory=tmp_path, hours=[0])
    cells = _cells(rows={"W1": (11, 21), "W2": (12, 22)})

    wind = read_cerra_wind(directory=tmp_path, cells=cells).sort("site")

    assert wind.columns == [
        "site",
        "time",
        "wind_speed_10m",
        "wind_speed_50m",
        "wind_speed_75m",
        "wind_speed_100m",
        "wind_speed_150m",
    ]
    assert wind.row(0, named=True) | {"time": None} == {
        "site": "W1",
        "time": None,
        "wind_speed_10m": 31.0,
        "wind_speed_50m": 71.0,
        "wind_speed_75m": 96.0,
        "wind_speed_100m": 121.0,
        "wind_speed_150m": 171.0,
    }
    assert wind["wind_speed_100m"].to_list() == [121.0, 122.0]
    assert wind["time"].dtype == pl.Datetime("us", "UTC")


def test_cerra_reading_raises_when_a_cell_is_on_the_crop_edge(tmp_path: Path):
    _write_cerra(directory=tmp_path, hours=[0])

    with pytest.raises(ValueError, match="only 1 of 2 sites"):
        read_cerra_wind(directory=tmp_path, cells=_cells(rows={"W1": (11, 21), "W2": (14, 22)}))


def test_nora3_pivot_keeps_the_right_speed_and_direction_per_height(tmp_path: Path):
    path = tmp_path / "nora3.parquet"
    _write_nora3(path=path, hours=[0, 1])
    cells = _cells(rows={"W1": (11, 21)})

    wind = read_nora3_wind(path=path, cells=cells)

    assert wind.height == 2
    assert wind.row(0, named=True) | {"time": None} == {
        "site": "W1",
        "time": None,
        "wind_speed_50m": 71.0,
        "wind_direction_50m": 710.0,
        "wind_speed_100m": 121.0,
        "wind_direction_100m": 1210.0,
    }


def test_nora3_reading_raises_when_a_cell_is_outside_the_crop(tmp_path: Path):
    path = tmp_path / "nora3.parquet"
    _write_nora3(path=path, hours=[0])

    with pytest.raises(ValueError, match="only 0 of 1 sites"):
        read_nora3_wind(path=path, cells=_cells(rows={"W1": (200, 600)}))


def test_centred_hour_averages_the_half_hours_ending_at_the_label_and_thirty_minutes_after():
    hourly = centred_hourly_power(half_hourly=_half_hourly(site="W1", hours=3))

    # The hour labelled 01:00 holds the half-hours ending at 01:00 (index 1) and 01:30 (index 2).
    assert hourly.filter(pl.col("time") == datetime(2024, 1, 1, 1, tzinfo=UTC))[
        "power_mw"
    ].to_list() == [1.5]


def test_cerra_rows_keep_only_the_three_hourly_hours_with_a_full_centred_hour(tmp_path: Path):
    _write_cerra(directory=tmp_path, hours=[1, 4, 7, 20])
    wind = read_cerra_wind(directory=tmp_path, cells=_cells(rows={"W1": (11, 21)}))

    joined = join_centred_power(wind=wind, half_hourly=_half_hourly(site="W1", hours=8))

    # Hour 20 has wind but no power, and hours 2, 3, 5 and 6 have power but no wind.
    assert joined["time"].dt.hour().to_list() == [1, 4, 7]
    assert joined.columns[-2:] == ["power_mw", "has_zero_half_hour"]
