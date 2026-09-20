"""Download the ERA5 fields the beam/diffuse experiment needs, one request per half-year.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784>. It is deliberately outside
the Dagster asset graph and writes nothing the rest of the repo reads.

Three fields are fetched over a box covering the six Lincolnshire PV sites: `ssrd` (global
horizontal irradiance), `fdir` (the direct beam's flux onto a horizontal plane), and `t2m`. `ssrd`
and `fdir` come from the same IFS radiation scheme on the same grid at the same time, which is the
whole point — it lets one arm of the experiment read the model's own beam/diffuse split and another
arm read only the global irradiance, with nothing else different between them.

Run it with `uv run --no-project --with cdsapi python
scripts/experiments/beam_diffuse_split/fetch_era5.py`.

Requires a Copernicus Climate Data Store token in `~/.cdsapirc`, and the account must have accepted
both the "licence to use Copernicus products" and the CC-BY licence.

**Requests are as large as the Climate Data Store allows, because it runs one of an account's jobs
at a time.** Queue waiting, not data volume, is what sets the wall-clock time: eighty-five monthly
requests spend hours in the queue to fetch a few megabytes, while fifteen half-yearly requests for
the same bytes spend almost none. Six months is the largest chunk that fits: the store prices a
request by how many fields it returns, and twelve months of three hourly fields comes to 157,680
against a limit of 121,000. Each chunk is skipped when its output file already exists, so an
interrupted run resumes by being started again.
"""

import concurrent.futures
import logging
import sys
from pathlib import Path
from typing import Final

# cdsapi is not a workspace dependency; this throwaway script is run with `uv run --with cdsapi`.
import cdsapi  # ty: ignore[unresolved-import]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("fetch_era5")

OUTPUT_DIR: Final[Path] = Path("/home/jack/dev/nged-substation-forecast/data/ERA5/beam_diffuse")
"""Where the downloads land.

Outside the repo working tree on purpose: these are data, and `data/` is git-ignored.
"""

AREA: Final[tuple[float, float, float, float]] = (53.50, -0.50, 52.75, 0.50)
"""CDS `area` as (north, west, south, east) in degrees.

The six PV sites span 52.92–53.15 N and -0.10–0.24 E, so this box clears them by at least a quarter
of a degree on every side and lands on the 0.25-degree ERA5 grid without interpolation.
"""

VARIABLES: Final[tuple[str, ...]] = (
    "surface_solar_radiation_downwards",
    "total_sky_direct_solar_radiation_at_surface",
    "2m_temperature",
)

FIRST_YEAR: Final[int] = 2019
LAST_YEAR: Final[int] = 2026

FIRST_MONTH_OF_FIRST_YEAR: Final[int] = 9
"""The earliest usable PV series starts in September 2019, so earlier months are not requested."""

LAST_MONTH_OF_LAST_YEAR: Final[int] = 9
"""The power data currently runs to September 2026."""

MONTHS_PER_REQUEST: Final[int] = 6
"""The largest chunk that stays inside the Climate Data Store's per-request field limit."""

MAX_CONCURRENT_REQUESTS: Final[int] = 4
"""How many CDS jobs to have in flight at once.

The Climate Data Store runs one of an account's jobs at a time and queues the rest, so this only
controls how deep the queue is kept, not how fast it drains.
"""


def _chunks() -> list[tuple[int, tuple[int, ...]]]:
    """Return every (year, months) chunk to request, in chronological order."""
    chunks: list[tuple[int, tuple[int, ...]]] = []
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        first = FIRST_MONTH_OF_FIRST_YEAR if year == FIRST_YEAR else 1
        last = LAST_MONTH_OF_LAST_YEAR if year == LAST_YEAR else 12
        months = list(range(first, last + 1))
        chunks.extend(
            (year, tuple(months[start : start + MONTHS_PER_REQUEST]))
            for start in range(0, len(months), MONTHS_PER_REQUEST)
        )
    return chunks


def _fetch_chunk(*, year: int, months: tuple[int, ...]) -> Path:
    """Download one chunk of ERA5, or return the existing file untouched.

    Args:
        year: The year the chunk sits in.
        months: The months of that year to fetch.

    Returns:
        The path the chunk's archive now sits at.
    """
    path = OUTPUT_DIR / f"era5_{year}_{months[0]:02d}_{months[-1]:02d}.zip"
    if path.exists():
        _LOG.info("%s already downloaded, skipping", path.name)
        return path

    client = cdsapi.Client(quiet=True, progress=False)
    request = {
        "product_type": ["reanalysis"],
        "variable": list(VARIABLES),
        "year": [f"{year}"],
        "month": [f"{month:02d}" for month in months],
        "day": [f"{day:02d}" for day in range(1, 32)],
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "area": list(AREA),
        "data_format": "netcdf",
        "download_format": "zip",
    }
    # Download to a temporary name and rename on success, so an interrupted run never leaves a
    # truncated archive that the resume logic would mistake for a finished one.
    partial = path.with_suffix(".zip.partial")
    client.retrieve("reanalysis-era5-single-levels", request, str(partial))
    partial.rename(path)
    _LOG.info("%s downloaded (%.1f MB)", path.name, path.stat().st_size / 1e6)
    return path


def main() -> int:
    """Fetch every chunk and report how many failed."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chunks = _chunks()
    _LOG.info("fetching %d chunks into %s", len(chunks), OUTPUT_DIR)

    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as pool:
        futures = {
            pool.submit(_fetch_chunk, year=year, months=months): (year, months)
            for year, months in chunks
        }
        for future in concurrent.futures.as_completed(futures):
            year, months = futures[future]
            try:
                future.result()
            # One chunk failing must not abandon the rest, and cdsapi surfaces network, HTTP and
            # server-side job failures as several unrelated exception types.
            except Exception:  # noqa: BLE001
                failures += 1
                _LOG.exception("%d months %s failed", year, months)

    _LOG.info("done: %d of %d chunks failed", failures, len(chunks))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
