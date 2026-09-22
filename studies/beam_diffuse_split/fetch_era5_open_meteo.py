"""Download the same ERA5 fields from Open-Meteo's mirror, as an independent replication.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784>.

**Open-Meteo serves ERA5's own `fdir`, not a separation model's estimate of it, which is the only
reason this script is usable here.** A mirror that reconstructed the beam from global irradiance
would make arm C a copy of arm B and void the experiment, so the claim is checked rather than
trusted: `verify_era5_sources.py` compares this frame against the Copernicus download field by
field, and on the test day 2024-06-20 at one grid cell the two agree to Open-Meteo's 1 W m⁻²
rounding.

What it buys is time. The Climate Data Store runs one of an account's jobs at a time and takes
around five minutes per month of hourly fields, so seven years is most of a night; Open-Meteo
returns the same seven years on the same grid in about a minute. The Copernicus download stays the
source of the headline result, and this one is reported beside it.

Requests are made at the ERA5 grid-cell centres rather than at the sites, with
`cell_selection=nearest`, so both sources feed the same cells to the same join.

Run it with `uv run python
studies/beam_diffuse_split/fetch_era5_open_meteo.py`.
"""

import itertools
import json
import logging
import sys
import time
import urllib.request
from pathlib import Path
from typing import Final

import polars as pl
from era5_grid import (
    FIRST_YEAR,
    GRID_LATITUDES,
    GRID_LONGITUDES,
    LAST_DATE,
    LAST_YEAR,
    first_date_of,
)
from sources import WEATHER_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("fetch_era5_open_meteo")

OUTPUT_PATH: Final[Path] = WEATHER_DATA_DIR / "ERA5" / "beam_diffuse_open_meteo.parquet"

ARCHIVE_URL: Final[str] = "https://archive-api.open-meteo.com/v1/archive"
REQUEST_TIMEOUT_SECONDS: Final[float] = 300.0
MAX_ATTEMPTS: Final[int] = 5


def _last_date_of(*, year: int) -> str:
    """Return the last date to request in `year`, as `YYYY-MM-DD`."""
    return LAST_DATE if year == LAST_YEAR else f"{year}-12-31"


def _fetch_year(*, year: int) -> list[pl.DataFrame]:
    """Fetch one year for every grid cell, retrying on a network failure.

    Args:
        year: The year to fetch.

    Returns:
        One frame per grid cell.

    Raises:
        RuntimeError: If every attempt failed.
    """
    latitudes = ",".join(
        str(latitude) for latitude, _ in itertools.product(GRID_LATITUDES, GRID_LONGITUDES)
    )
    longitudes = ",".join(
        str(longitude) for _, longitude in itertools.product(GRID_LATITUDES, GRID_LONGITUDES)
    )
    url = (
        f"{ARCHIVE_URL}?latitude={latitudes}&longitude={longitudes}"
        f"&start_date={first_date_of(year=year)}&end_date={_last_date_of(year=year)}"
        "&hourly=shortwave_radiation,direct_radiation,temperature_2m"
        "&models=era5&timezone=UTC&cell_selection=nearest"
    )
    for attempt in range(MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read())
            break
        # urllib raises several unrelated types for a transient failure, and a retry is the right
        # response to all of them.
        except Exception:  # noqa: BLE001
            _LOG.warning("%d attempt %d failed, retrying", year, attempt + 1)
            time.sleep(5.0 * (attempt + 1))
    else:
        msg = f"Open-Meteo failed for {year} after {MAX_ATTEMPTS} attempts"
        raise RuntimeError(msg)

    blocks = payload if isinstance(payload, list) else [payload]
    return [
        pl.DataFrame(
            {
                "time": block["hourly"]["time"],
                "latitude": round(block["latitude"], 2),
                "longitude": round(block["longitude"], 2),
                "ghi_w_m2": block["hourly"]["shortwave_radiation"],
                "bhi_w_m2": block["hourly"]["direct_radiation"],
                "temp_c": block["hourly"]["temperature_2m"],
            },
            schema_overrides={"ghi_w_m2": pl.Float64, "bhi_w_m2": pl.Float64, "temp_c": pl.Float64},
        )
        for block in blocks
    ]


def main() -> int:
    """Fetch every year and write the long frame."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frames: list[pl.DataFrame] = []
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        frames += _fetch_year(year=year)
        _LOG.info("%d fetched", year)

    era5 = (
        pl.concat(frames)
        .with_columns(pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M").dt.replace_time_zone("UTC"))
        .drop_nulls()
        .unique(subset=["time", "latitude", "longitude"], keep="first")
        .sort("time", "latitude", "longitude")
    )
    era5.write_parquet(OUTPUT_PATH)
    _LOG.info(
        "wrote %d rows covering %s to %s across %d cells",
        era5.height,
        era5["time"].min(),
        era5["time"].max(),
        era5.select("latitude", "longitude").n_unique(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
