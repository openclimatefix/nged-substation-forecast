"""The ERA5 grid and date range both downloads share.

One-off throwaway module for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784>.

The two sources have to land on the same cells and the same hours, or the replication would be
comparing the arms *and* the grid at once. Keeping the grid in one place is what makes "the same
cells" checkable rather than a coincidence of two literals.
"""

import os
from pathlib import Path
from typing import Final

AREA: Final[tuple[float, float, float, float]] = (53.50, -0.50, 52.75, 0.50)
"""The requested box as (north, west, south, east) in degrees.

Chosen to clear the six PV meters by at least a quarter of a degree on every side while landing on
the 0.25-degree ERA5 grid, so no interpolation happens anywhere. It is deliberately much coarser
than the meters' own footprint: their output is commercially sensitive and this repo is public.
"""

GRID_LATITUDES: Final[tuple[float, ...]] = (53.5, 53.25, 53.0, 52.75)
"""ERA5 grid latitudes inside `AREA`, in the descending order the archive stores them."""

GRID_LONGITUDES: Final[tuple[float, ...]] = (-0.5, -0.25, 0.0, 0.25, 0.5)
"""ERA5 grid longitudes inside `AREA`."""

PUBLISHED_FIRST_YEAR: Final[int] = 2019
PUBLISHED_LAST_DATE: Final[str] = "2026-09-10"
"""The last date of the datasets the published studies use. `build_dataset` always trims here."""

FIRST_DATE_OVERRIDE: Final[str | None] = os.environ.get("ERA5_FIRST_DATE")
"""If set (ISO date), the download starts here rather than at the start of the published range.

A refresh sets this, together with `ERA5_LAST_DATE` and `ERA5_OUTPUT_SUFFIX`, so it fetches only a
recent window and writes new files. With all three unset every value here is the published one.
"""

FIRST_YEAR: Final[int] = (
    int(FIRST_DATE_OVERRIDE[:4]) if FIRST_DATE_OVERRIDE else PUBLISHED_FIRST_YEAR
)
LAST_DATE: Final[str] = os.environ.get("ERA5_LAST_DATE", PUBLISHED_LAST_DATE)
LAST_YEAR: Final[int] = int(LAST_DATE[:4])
"""`LAST_DATE` is where the request stops (`ERA5_LAST_DATE` overrides the published value).

ERA5 runs a few days behind real time and Open-Meteo's mirror a few days further, so the published
value is the last date both sources could serve. The power readings run ten days past it, and those
ten days are dropped rather than letting the two sources cover different spans.
"""

OUTPUT_SUFFIX: Final[str] = os.environ.get("ERA5_OUTPUT_SUFFIX", "")
"""Appended to every download's file stem, so a refresh never overwrites a published file."""

FIRST_MONTH_OF_FIRST_YEAR: Final[int] = 9
"""The earliest usable PV series starts in September 2019, so earlier months are not requested."""


def first_date_of(*, year: int) -> str:
    """Return the first date to request in `year`, as `YYYY-MM-DD`."""
    if FIRST_DATE_OVERRIDE and year == FIRST_YEAR:
        return FIRST_DATE_OVERRIDE
    month = FIRST_MONTH_OF_FIRST_YEAR if year == FIRST_YEAR else 1
    return f"{year}-{month:02d}-01"


def suffixed(path: Path) -> Path:
    """Append `OUTPUT_SUFFIX` to a file stem.

    Args:
        path: A download's published output path.

    Returns:
        `path` itself when the suffix is empty, else `path` with the suffix before the extension.
    """
    return path.with_name(f"{path.stem}{OUTPUT_SUFFIX}{path.suffix}")
