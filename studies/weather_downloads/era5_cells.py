"""Pure helpers for the 2019 to 2023 ERA5 wind download: grid cells, chunks, and request areas.

One-off throwaway module for the downloads in
<https://github.com/openclimatefix/nged-substation-forecast/issues/841>, shared by
`fetch_era5_wind_2019_2023.py` and `validate_era5_wind_2019_2023.py`. Nothing here reads private
data or the network, so every function is testable on its own. A cell is a pair of integer
quarter-degree indices `(lat_q, lon_q)`; a coordinate in degrees only appears in `bounding_area`,
whose result goes straight into a Climate Data Store request and is never printed.
"""

import calendar
import math
from dataclasses import dataclass
from typing import Any, Final

GRID_STEP_DEG: Final[float] = 0.25
"""ERA5's native grid spacing in degrees."""

Cell = tuple[int, int]
"""A grid cell as `(lat_q, lon_q)`: its latitude and longitude in whole quarter degrees."""


@dataclass(frozen=True)
class Chunk:
    """One request's calendar months, never crossing a year boundary."""

    year: int
    first_month: int
    last_month: int

    @property
    def months(self) -> tuple[int, ...]:
        """Return the chunk's months, first to last."""
        return tuple(range(self.first_month, self.last_month + 1))

    @property
    def name(self) -> str:
        """Return a filename-safe label such as `2019_09_12`."""
        return f"{self.year}_{self.first_month:02d}_{self.last_month:02d}"

    @property
    def n_hours(self) -> int:
        """Return how many hourly timestamps the chunk holds."""
        days = sum(calendar.monthrange(self.year, month)[1] for month in self.months)
        return days * 24


def half_year_chunks(*, first: tuple[int, int], last: tuple[int, int]) -> list[Chunk]:
    """Return calendar half-year chunks covering `first` to `last` months inclusive.

    A chunk is January to June, July to December, or the part of one of those inside the range.

    Args:
        first: The first `(year, month)` wanted.
        last: The last `(year, month)` wanted.

    Returns:
        The chunks in chronological order.
    """
    chunks = []
    for year in range(first[0], last[0] + 1):
        for half_first, half_last in ((1, 6), (7, 12)):
            low = max(half_first, first[1] if year == first[0] else 1)
            high = min(half_last, last[1] if year == last[0] else 12)
            if low <= high:
                chunks.append(Chunk(year=year, first_month=low, last_month=high))
    return chunks


def quarter_degree_index(*, degrees: float) -> int:
    """Return the index of the 0.25 degree grid line nearest `degrees`."""
    return round(degrees / GRID_STEP_DEG)


def block_cells(*, centre: Cell, half_width: int = 1) -> list[tuple[int, int, Cell]]:
    """Return a square block of cells around `centre` with each cell's offset from the centre.

    Args:
        centre: The block's middle cell.
        half_width: How many cells the block extends on each side; 1 gives a 3 x 3 block.

    Returns:
        `(dy, dx, cell)` per cell, where `dy` and `dx` are offsets in cells north and east.
    """
    return [
        (dy, dx, (centre[0] + dy, centre[1] + dx))
        for dy in range(-half_width, half_width + 1)
        for dx in range(-half_width, half_width + 1)
    ]


def box_cells(
    *, lat_min: float, lat_max: float, lon_min: float, lon_max: float, tolerance: float = 1e-9
) -> list[Cell]:
    """Return every grid cell from the grid line at or below each edge to the one at or above.

    The result is the smallest block of cells enclosing the box, so a point inside the box has all
    four surrounding grid nodes in the block.

    Args:
        lat_min: Southern edge in degrees.
        lat_max: Northern edge in degrees.
        lon_min: Western edge in degrees.
        lon_max: Eastern edge in degrees.
        tolerance: Slack in grid steps so that an edge exactly on a grid line adds no extra row.

    Returns:
        The cells, southern row first.
    """
    low_lat = math.floor(lat_min / GRID_STEP_DEG + tolerance)
    high_lat = math.ceil(lat_max / GRID_STEP_DEG - tolerance)
    low_lon = math.floor(lon_min / GRID_STEP_DEG + tolerance)
    high_lon = math.ceil(lon_max / GRID_STEP_DEG - tolerance)
    return [
        (lat_q, lon_q)
        for lat_q in range(low_lat, high_lat + 1)
        for lon_q in range(low_lon, high_lon + 1)
    ]


def cluster_cells(*, cells: set[Cell], link_cells: int) -> list[set[Cell]]:
    """Group cells so that two cells within `link_cells` grid steps of each other share a group.

    This is single-linkage clustering with the Chebyshev distance, so a chain of nearby cells forms
    one group. Each group is later requested as its own rectangle, which keeps a few distant sites
    from forcing one request over everything between them.

    Args:
        cells: The cells to group.
        link_cells: The largest gap, in grid steps along either axis, that still joins two cells.

    Returns:
        The groups, ordered by their southernmost then westernmost cell.
    """
    remaining = set(cells)
    groups: list[set[Cell]] = []
    while remaining:
        seed = min(remaining)
        group = {seed}
        frontier = [seed]
        remaining.discard(seed)
        while frontier:
            lat_q, lon_q = frontier.pop()
            near = {
                other
                for other in remaining
                if abs(other[0] - lat_q) <= link_cells and abs(other[1] - lon_q) <= link_cells
            }
            remaining -= near
            group |= near
            frontier.extend(near)
        groups.append(group)
    return sorted(groups, key=min)


def bounding_area(*, cells: set[Cell]) -> list[float]:
    """Return the Climate Data Store `area` value `[north, west, south, east]` covering `cells`.

    The result holds coordinates and must go straight into a request, never into a log line.
    """
    lat_qs = [cell[0] for cell in cells]
    lon_qs = [cell[1] for cell in cells]
    return [
        max(lat_qs) * GRID_STEP_DEG,
        min(lon_qs) * GRID_STEP_DEG,
        min(lat_qs) * GRID_STEP_DEG,
        max(lon_qs) * GRID_STEP_DEG,
    ]


def request_body(*, chunk: Chunk, variables: list[str], area: list[float]) -> dict[str, Any]:
    """Return the `reanalysis-era5-single-levels` request for one chunk and one area.

    Args:
        chunk: The calendar months to fetch.
        variables: The Climate Data Store variable names.
        area: `[north, west, south, east]` from `bounding_area`.

    Returns:
        The request dictionary, with all 24 hours and netCDF inside a zip.
    """
    return {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": [str(chunk.year)],
        "month": [f"{month:02d}" for month in chunk.months],
        "day": [f"{day:02d}" for day in range(1, 32)],
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "area": area,
        "data_format": "netcdf",
        "download_format": "zip",
    }
