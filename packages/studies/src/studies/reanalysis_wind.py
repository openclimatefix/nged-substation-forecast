"""Load CERRA and NORA3 reanalysis wind at each wind farm's nearest cell, beside the power hour.

**CERRA's 10 m wind speed comes from the single-levels product, and its 50, 75, 100 and 150 m wind
speeds come from the height-levels product.** The two products are separate downloads, one parquet
file per height, and its wind direction is a separate set of files with the same keys. CERRA is a
3-hourly instantaneous analysis (00, 03, ..., 21 UTC). NORA3 is an hourly instantaneous product on a
3 km Lambert grid. Its main file holds 50 m and 100 m, each with a speed and a direction, and its
10 m level is a separate file.

**Each farm's nearest cell must lie strictly inside the downloaded crop.** A crop's edge row and
column have no neighbours beyond them in the file, and the download never fetched a cell outside the
crop, so a farm whose nearest cell is on the edge or absent is a farm the crop was too small for.
`check_strictly_inside` raises on that, naming counts only.

**Wind is an instantaneous value at the label, and the power hour is centred on the label.** The
hour labelled `T` averages the half-hours ending at `T` and at `T + 30 min`, which span `T - 30 min`
to `T + 30 min`. `centred_hourly_power` builds that hour, and `join_centred_power` keeps the hours
that also have a wind value, so a CERRA frame holds one hour in three.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from pyproj import Transformer

from studies.grid_sampling import nearest_cells
from studies.power import hourly_from_half_hourly

SITE_COLUMN: Final[str] = "site"
TIME_COLUMN: Final[str] = "time"
CELL_COLUMNS: Final[list[str]] = ["y_index", "x_index"]

CERRA_FILES: Final[Mapping[int, str]] = {
    10: "10m_wind_speed_surface.parquet",
    50: "wind_speed_50_m.parquet",
    75: "wind_speed_75_m.parquet",
    100: "wind_speed_100_m.parquet",
    150: "wind_speed_150_m.parquet",
}
"""CERRA's parquet file per height in metres: 10 m from the single-levels product, the rest from the
height-levels product."""

CERRA_DIRECTION_FILES: Final[Mapping[int, str]] = {
    10: "10m_wind_direction_surface.parquet",
    50: "wind_direction_50_m.parquet",
    75: "wind_direction_75_m.parquet",
    100: "wind_direction_100_m.parquet",
    150: "wind_direction_150_m.parquet",
}
"""CERRA's wind direction parquet file per height in metres, with the same keys as the speed files
and the value in `wind_direction_deg`."""

NORA3_HEIGHTS_M: Final[tuple[int, ...]] = (50, 100)
"""The heights NORA3's main file holds."""

NORA3_SURFACE_HEIGHT_M: Final[int] = 10
"""The height of NORA3's separate 10 m file."""

NORA3_LAMBERT_PROJ4: Final[str] = (
    "+proj=lcc +lat_1=66.3 +lat_2=66.3 +lat_0=66.3 +lon_0=-42.0 +R=6371000 +units=m +no_defs"
)
"""NORA3's Lambert conformal projection, from the dataset's own `projection_lambert` attributes."""

NORA3_GRID_X0_M: Final[float] = 778360.9
NORA3_GRID_Y0_M: Final[float] = -1270477.0
NORA3_GRID_SPACING_M: Final[float] = 3000.0
"""The grid's first `x` and `y` coordinate and its spacing, as `studies/weather_downloads/
fetch_nora3.py` reads them from the catalog."""


def derive_nearest_cells(*, grid: pl.DataFrame, sites: pl.DataFrame) -> pl.DataFrame:
    """Find each generator's nearest CERRA cell from the grid's latitude and longitude.

    Args:
        grid: One row per cell, with `y_index`, `x_index`, `latitude` and `longitude` in degrees.
            A longitude on 0 to 360 degrees is wrapped to -180 to 180.
        sites: The site list, with `site`, `latitude` and `longitude`.

    Returns:
        One row per generator with `site`, `y_index`, `x_index` and `distance_km`.
    """
    cells = grid.with_row_index("cell_id").with_columns(
        longitude=(pl.col("longitude") + 180.0) % 360.0 - 180.0
    )
    nearest = nearest_cells(sites=sites, cells=cells)
    return nearest.join(
        cells.select("cell_id", "y_index", "x_index"), on="cell_id", how="left"
    ).select("site", "y_index", "x_index", "distance_km")


def derive_nearest_nora3_cells(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Find each generator's nearest NORA3 cell on the Lambert grid.

    Args:
        sites: The site list, with `site`, `latitude` and `longitude`.

    Returns:
        One row per generator with `site`, `y_index`, `x_index` and `distance_km`, the projected
        distance to the cell's centre. The indices are not limited to the downloaded crop.
    """
    transformer = Transformer.from_crs("EPSG:4326", NORA3_LAMBERT_PROJ4, always_xy=True)
    easting, northing = transformer.transform(
        sites["longitude"].to_numpy(), sites["latitude"].to_numpy()
    )
    x_index = np.rint((easting - NORA3_GRID_X0_M) / NORA3_GRID_SPACING_M).astype(np.int64)
    y_index = np.rint((northing - NORA3_GRID_Y0_M) / NORA3_GRID_SPACING_M).astype(np.int64)
    distance_m = np.hypot(
        NORA3_GRID_X0_M + NORA3_GRID_SPACING_M * x_index - easting,
        NORA3_GRID_Y0_M + NORA3_GRID_SPACING_M * y_index - northing,
    )
    return pl.DataFrame(
        {
            SITE_COLUMN: sites[SITE_COLUMN],
            "y_index": y_index,
            "x_index": x_index,
            "distance_km": distance_m / 1000.0,
        }
    )


def check_strictly_inside(*, cells: pl.DataFrame, crop: pl.DataFrame) -> None:
    """Stop unless every site's cell is in the crop and not on its edge row or column.

    Args:
        cells: One row per site, with `site`, `y_index` and `x_index`.
        crop: The `y_index` and `x_index` of the cells the download holds. Its edge is the range of
            each index.

    Raises:
        ValueError: Naming how many sites fail, and never which site or which cell.
    """
    bounds = crop.select(
        y_min=pl.col("y_index").min(),
        y_max=pl.col("y_index").max(),
        x_min=pl.col("x_index").min(),
        x_max=pl.col("x_index").max(),
    )
    inside = (
        cells.join(crop.unique(subset=CELL_COLUMNS), on=CELL_COLUMNS, how="semi")
        .join(bounds, how="cross")
        .filter(
            (pl.col("y_index") > pl.col("y_min"))
            & (pl.col("y_index") < pl.col("y_max"))
            & (pl.col("x_index") > pl.col("x_min"))
            & (pl.col("x_index") < pl.col("x_max"))
        )
    )
    if inside.height != cells.height:
        msg = (
            f"only {inside.height} of {cells.height} sites have their nearest cell strictly "
            "inside the downloaded crop"
        )
        raise ValueError(msg)


def _crop_cells(*, path: Path) -> pl.DataFrame:
    """Return the distinct cells one parquet file holds.

    Args:
        path: A CERRA or NORA3 wind parquet file.

    Returns:
        One row per cell, with `y_index` and `x_index` as `Int64`.
    """
    return (
        pl.scan_parquet(path)
        .select(pl.col(CELL_COLUMNS).cast(pl.Int64))
        .unique()
        .collect(engine="streaming")
    )


def _site_rows(*, lazy: pl.LazyFrame, cells: pl.DataFrame) -> pl.LazyFrame:
    """Keep the rows at the sites' cells and label each with its site.

    Args:
        lazy: A wind parquet scan with `y_index` and `x_index`.
        cells: One row per site, with `site`, `y_index` and `x_index`.

    Returns:
        The scan's rows joined to `site`, so a cell shared by two sites gives each its own rows.
    """
    return lazy.with_columns(pl.col(CELL_COLUMNS).cast(pl.Int64)).join(
        cells.lazy().select(SITE_COLUMN, *CELL_COLUMNS), on=CELL_COLUMNS, how="inner"
    )


def _join_heights(*, per_height: Sequence[pl.DataFrame]) -> pl.DataFrame:
    """Join one frame per height into one row per (site, time).

    Args:
        per_height: Frames with `site`, `time` and that height's own columns.

    Returns:
        A row for every (site, time) in any frame, sorted by site then time, with a null where a
        height has no value.
    """
    joined = per_height[0]
    for frame in per_height[1:]:
        joined = joined.join(frame, on=[SITE_COLUMN, TIME_COLUMN], how="full", coalesce=True)
    return joined.sort(SITE_COLUMN, TIME_COLUMN)


def _utc(*, column: str) -> pl.Expr:
    """Return an expression labelling a naive UTC timestamp column as a UTC microsecond datetime.

    Args:
        column: The naive timestamp column.

    Returns:
        The column as `Datetime("us", "UTC")`, the dtype the power series uses.
    """
    return pl.col(column).cast(pl.Datetime("us")).dt.replace_time_zone("UTC").alias(TIME_COLUMN)


def read_cerra_wind(*, directory: Path, cells: pl.DataFrame) -> pl.DataFrame:
    """Read CERRA's wind speed at each site's cell, one column per height.

    Args:
        directory: The folder holding the files in `CERRA_FILES`.
        cells: One row per site, with `site`, `y_index` and `x_index`, from `derive_nearest_cells`.

    Returns:
        One row per (site, time) with `time` a UTC timestamp on the 3-hourly analysis hours and
        `wind_speed_10m`, `wind_speed_50m`, `wind_speed_75m`, `wind_speed_100m` and
        `wind_speed_150m` in metres per second.

    Raises:
        ValueError: If a site's cell is not strictly inside a file's crop.
    """
    per_height = []
    for height_m, name in CERRA_FILES.items():
        path = directory / name
        check_strictly_inside(cells=cells, crop=_crop_cells(path=path))
        per_height.append(
            _site_rows(lazy=pl.scan_parquet(path), cells=cells)
            .select(
                SITE_COLUMN,
                _utc(column="valid_time"),
                pl.col("wind_speed_m_s").alias(f"wind_speed_{height_m}m"),
            )
            .collect()
        )
    return _join_heights(per_height=per_height)


def read_cerra_direction(
    *, directory: Path, cells: pl.DataFrame, heights: Sequence[int]
) -> pl.DataFrame:
    """Read CERRA's wind direction at each site's cell, one column per height.

    Args:
        directory: The folder holding the files in `CERRA_DIRECTION_FILES`.
        cells: One row per site, with `site`, `y_index` and `x_index`, from `derive_nearest_cells`.
        heights: The heights in metres to read, keys of `CERRA_DIRECTION_FILES`.

    Returns:
        One row per (site, time) with `time` a UTC timestamp on the 3-hourly analysis hours and
        `wind_direction_{h}m` for each height `h`, in degrees clockwise from north, the direction
        the wind blows from.

    Raises:
        FileNotFoundError: If a height's file has not been downloaded.
        ValueError: If a site's cell is not strictly inside a file's crop.
    """
    per_height = []
    for height_m in heights:
        path = directory / CERRA_DIRECTION_FILES[height_m]
        if not path.exists():
            msg = f"CERRA's {height_m} m wind direction file {path.name} has not been downloaded"
            raise FileNotFoundError(msg)
        check_strictly_inside(cells=cells, crop=_crop_cells(path=path))
        per_height.append(
            _site_rows(lazy=pl.scan_parquet(path), cells=cells)
            .select(
                SITE_COLUMN,
                _utc(column="valid_time"),
                pl.col("wind_direction_deg").alias(f"wind_direction_{height_m}m"),
            )
            .collect()
        )
    return _join_heights(per_height=per_height)


def read_nora3_wind(
    *, path: Path, cells: pl.DataFrame, heights: Sequence[int] = NORA3_HEIGHTS_M
) -> pl.DataFrame:
    """Read NORA3's wind speed and direction at each site's cell, one column pair per height.

    Args:
        path: A NORA3 wind parquet file: the main file for 50 m and 100 m, or the 10 m file.
        cells: One row per site, with `site`, `y_index` and `x_index`, from
            `derive_nearest_nora3_cells`.
        heights: The heights in metres to read, each of which the file must hold.

    Returns:
        One row per (site, time) with `time` a UTC hourly timestamp, and `wind_speed_{h}m` and
        `wind_direction_{h}m` for each height `h`. Speeds are in metres per second and directions
        are degrees clockwise from north, the direction the wind blows from.

    Raises:
        ValueError: If a site's cell is not strictly inside the file's crop, or the file holds no
            row at one of the heights.
    """
    check_strictly_inside(cells=cells, crop=_crop_cells(path=path))
    rows = _site_rows(lazy=pl.scan_parquet(path), cells=cells)
    per_height = [
        rows.filter(pl.col("height_m") == height_m)
        .select(
            SITE_COLUMN,
            _utc(column="time"),
            pl.col("wind_speed_m_s").alias(f"wind_speed_{height_m}m"),
            pl.col("wind_direction_deg").alias(f"wind_direction_{height_m}m"),
        )
        .collect()
        for height_m in heights
    ]
    empty = [
        height_m for height_m, frame in zip(heights, per_height, strict=True) if frame.is_empty()
    ]
    if empty:
        msg = f"{path.name} holds no row at heights {empty} m"
        raise ValueError(msg)
    return _join_heights(per_height=per_height)


def centred_hourly_power(*, half_hourly: pl.DataFrame) -> pl.DataFrame:
    """Average half-hourly power onto an hour centred on each label.

    Shifting every stamp back 30 minutes before `hourly_from_half_hourly` means the hour labelled
    `T` holds the half-hours ending at `T` and at `T + 30 min`.

    Args:
        half_hourly: One row per `(site, time)`, carrying `power_mw`.

    Returns:
        One row per `(site, time)` with `power_mw` and `has_zero_half_hour`.
    """
    return hourly_from_half_hourly(
        half_hourly=half_hourly.with_columns(pl.col(TIME_COLUMN).dt.offset_by("-30m"))
    )


def join_centred_power(*, wind: pl.DataFrame, half_hourly: pl.DataFrame) -> pl.DataFrame:
    """Attach the centred power hour to the wind rows that have one.

    Args:
        wind: `read_cerra_wind`'s or `read_nora3_wind`'s result.
        half_hourly: One row per `(site, time)`, carrying `power_mw`.

    Returns:
        One row per (site, time) present in both, so a 3-hourly product gives one hour in three,
        with the wind columns, `power_mw` and `has_zero_half_hour`.
    """
    return wind.join(
        centred_hourly_power(half_hourly=half_hourly), on=[SITE_COLUMN, TIME_COLUMN], how="inner"
    ).sort(SITE_COLUMN, TIME_COLUMN)
