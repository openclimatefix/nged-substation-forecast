"""Download CAMS satellite-derived irradiance at each PV site, as a third irradiance source.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784>.

**CAMS is the source that tests whether ERA5's null result is a property of the beam/diffuse split
or a property of ERA5's 31 km grid.** ERA5 averages the cloud field over roughly 31 km, which blunts
the beam far more than the diffuse component, so a split that carries real information at a site
could still read as no information at ERA5's resolution. The CAMS radiation service infers cloud
from Meteosat at around 5 km and publishes the global, beam and diffuse horizontal irradiances from
that one retrieval, which is the same three-field structure the arms already consume.

Requests are made at each meter's own coordinates, read at run time from the private roster. **No
coordinate and no identifier reaches the written frame**: rows are keyed by the anonymised site
label `build_dataset._pv_sites` assigns, and the downloaded CSVs stay in the git-ignored data
directory.

The service publishes irradiation in Wh m⁻² summed over each step, so at a 1-hour step the number is
also the mean flux in W m⁻², and no conversion is needed. Each row's `Observation period` names the
interval's start and end; the end is kept, which is the period-ending convention ERA5 uses.

Run it with `uv run --with cdsapi python studies/beam_diffuse_split/fetch_cams.py`. The extra
dependencies are `build_dataset`'s, which this script imports the site roster from.
"""

import concurrent.futures
import logging
import sys
from pathlib import Path
from typing import Final, NamedTuple

import cdsapi  # ty: ignore[unresolved-import]
import polars as pl
from build_dataset import _pv_sites
from era5_grid import FIRST_YEAR, LAST_DATE, LAST_YEAR, first_date_of
from sources import STUDY_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("fetch_cams")

CAMS_DIR: Final[Path] = STUDY_DATA_DIR / "CAMS"
OUTPUT_PATH: Final[Path] = CAMS_DIR / "beam_diffuse_cams.parquet"

ADS_URL: Final[str] = "https://ads.atmosphere.copernicus.eu/api"
DATASET: Final[str] = "cams-solar-radiation-timeseries"

MAX_CONCURRENT_REQUESTS: Final[int] = 3
"""How many requests are in flight at once.

The Atmosphere Data Store queues an account's jobs, so this buys pipelining of the wait rather than
parallel computation, and a small number keeps the account well inside its fair-use limits.
"""

UNKNOWN_ALTITUDE: Final[str] = "-999."
"""What the service wants when the caller does not supply a site altitude.

It then reads the elevation out of its own terrain model, which is what we want: a wrong altitude
would change the clear-sky irradiance.
"""

COLUMN_NAMES: Final[tuple[str, ...]] = (
    "observation_period",
    "toa_wh_m2",
    "clear_sky_ghi_wh_m2",
    "clear_sky_bhi_wh_m2",
    "clear_sky_dhi_wh_m2",
    "clear_sky_bni_wh_m2",
    "ghi_wh_m2",
    "bhi_wh_m2",
    "dhi_wh_m2",
    "bni_wh_m2",
    "reliability",
)
"""The 11 columns the service writes, in order, under its commented header."""


class SiteYear(NamedTuple):
    """One download: a site's coordinates and the year to request for it.

    Attributes:
        site: The anonymised site label, which is all that reaches a written file.
        latitude: The meter's latitude, sent to the service and not recorded.
        longitude: The meter's longitude, sent to the service and not recorded.
        year: The year to request.
    """

    site: str
    latitude: float
    longitude: float
    year: int


def _api_key() -> str:
    """Read the Copernicus API key, which both data stores accept."""
    for line in Path.home().joinpath(".cdsapirc").read_text().splitlines():
        if line.startswith("key:"):
            return line.split(":", 1)[1].strip()
    msg = "no key found in ~/.cdsapirc"
    raise RuntimeError(msg)


def _last_date_of(*, year: int) -> str:
    """Return the last date to request in `year`, as `YYYY-MM-DD`."""
    return LAST_DATE if year == LAST_YEAR else f"{year}-12-31"


def _csv_path(*, site: str, year: int) -> Path:
    """Return where one site-year's download lands."""
    return CAMS_DIR / f"cams_site_{site}_{year}.csv"


def _fetch_one(*, job: SiteYear) -> Path:
    """Download one site-year, skipping the request if the file is already there.

    Args:
        job: Which site and year to fetch.

    Returns:
        The path the CSV landed at.
    """
    destination = _csv_path(site=job.site, year=job.year)
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    client = cdsapi.Client(url=ADS_URL, key=_api_key(), quiet=True)
    partial = destination.with_suffix(".csv.partial")
    client.retrieve(
        DATASET,
        {
            "sky_type": "observed_cloud",
            "location": {"latitude": job.latitude, "longitude": job.longitude},
            "altitude": UNKNOWN_ALTITUDE,
            "date": [f"{first_date_of(year=job.year)}/{_last_date_of(year=job.year)}"],
            "time_step": "1hour",
            "time_reference": "universal_time",
            "data_format": "csv",
        },
        str(partial),
    )
    _strip_coordinates(path=partial)
    partial.rename(destination)
    _LOG.info("site %s %d downloaded", job.site, job.year)
    return destination


COORDINATE_HEADER_PREFIXES: Final[tuple[str, ...]] = (
    "# Latitude",
    "# Longitude",
    "# Altitude",
)
"""Header lines the service writes that name where the request was made."""


def _strip_coordinates(*, path: Path) -> None:
    """Remove the header lines that pair a site's anonymised label with its coordinates.

    **The downloaded file is the one artefact in this experiment that would carry both halves of the
    mapping at once**, because its name holds the published label and the service writes the
    requested latitude and longitude into its header. Nothing under the data directory is committed,
    but a file a human might paste into an issue to show what the service returned should not carry
    a metered generator's location.

    Args:
        path: The downloaded CSV, rewritten in place.
    """
    kept = [
        line
        for line in path.read_text().splitlines(keepends=True)
        if not line.startswith(COORDINATE_HEADER_PREFIXES)
    ]
    path.write_text("".join(kept))


def _read_one(*, path: Path, site: str) -> pl.DataFrame:
    """Parse one downloaded CSV into the columns the experiment uses.

    Args:
        path: The downloaded CSV.
        site: The anonymised label to stamp on every row.

    Returns:
        One row per hour with `site`, `time`, the three all-sky horizontal fluxes, the clear-sky
        global flux and the service's reliability flag.
    """
    frame = pl.read_csv(
        path,
        separator=";",
        comment_prefix="#",
        has_header=False,
        new_columns=list(COLUMN_NAMES),
    )
    return frame.select(
        site=pl.lit(site),
        time=pl.col("observation_period")
        .str.split("/")
        .list.last()
        .str.to_datetime("%Y-%m-%dT%H:%M:%S%.f")
        .dt.replace_time_zone("UTC"),
        ghi_w_m2=pl.col("ghi_wh_m2"),
        bhi_w_m2=pl.col("bhi_wh_m2"),
        dhi_direct_w_m2=pl.col("dhi_wh_m2"),
        clear_sky_ghi_w_m2=pl.col("clear_sky_ghi_wh_m2"),
        reliability=pl.col("reliability"),
    )


def main() -> int:
    """Download every site-year and write the one long frame `build_dataset` reads."""
    CAMS_DIR.mkdir(parents=True, exist_ok=True)
    sites = _pv_sites()
    jobs = [
        SiteYear(
            site=str(row["site"]),
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            year=year,
        )
        for row in sites.to_dicts()
        for year in range(FIRST_YEAR, LAST_YEAR + 1)
    ]
    _LOG.info("%d site-years to fetch for %d sites", len(jobs), sites.height)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as pool:
        futures = [pool.submit(_fetch_one, job=job) for job in jobs]
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            future.result()
            _LOG.info("%d of %d site-years ready", done, len(jobs))

    cams = (
        pl.concat(
            [_read_one(path=_csv_path(site=job.site, year=job.year), site=job.site) for job in jobs]
        )
        .drop_nulls()
        .unique(subset=["site", "time"], keep="first")
        .sort("site", "time")
    )
    cams.write_parquet(OUTPUT_PATH)
    _LOG.info(
        "wrote %d rows covering %s to %s for %d sites",
        cams.height,
        cams["time"].min(),
        cams["time"].max(),
        cams["site"].n_unique(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
