"""The ERA5 grid and date range both downloads share.

One-off throwaway module for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784>.

The two sources have to land on the same cells and the same hours, or the replication would be
comparing the arms *and* the grid at once. Keeping the grid in one place is what makes "the same
cells" checkable rather than a coincidence of two literals.
"""

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

FIRST_YEAR: Final[int] = 2019
LAST_YEAR: Final[int] = 2026

FIRST_MONTH_OF_FIRST_YEAR: Final[int] = 9
"""The earliest usable PV series starts in September 2019, so earlier months are not requested."""

LAST_DATE: Final[str] = "2026-09-10"
"""Where the request stops.

ERA5 runs a few days behind real time and Open-Meteo's mirror a few days further, so this is the
last date both sources can serve. The power readings run ten days past it, and those ten days are
dropped rather than letting the two sources cover different spans.
"""


def first_date_of(*, year: int) -> str:
    """Return the first date to request in `year`, as `YYYY-MM-DD`."""
    month = FIRST_MONTH_OF_FIRST_YEAR if year == FIRST_YEAR else 1
    return f"{year}-{month:02d}-01"
