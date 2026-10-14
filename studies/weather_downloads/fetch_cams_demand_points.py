"""Download CAMS satellite-derived irradiance at the locations of the 22 demand series.

One-off throwaway script for experiment 00003 (ERA5-trained stage 1a against ENS-trained, for
demand series). It uses the same Atmosphere Data Store dataset (`cams-solar-radiation-timeseries`),
the same `~/.cdsapirc` key, and the same request as `studies/beam_diffuse_split/fetch_cams.py`, at
the locations of the demand series instead of the solar farms.

**No coordinate and no series identifier reaches a written file.** The series are read from a
private table (`--series`) holding `site` (the label `D1` to `D22`), `latitude`, and `longitude`.
Rows are keyed by the label, the service's header lines that name the requested location are
stripped from each CSV, and the README lists no point.

The service publishes irradiation in Wh m⁻² summed over each step, so at a 1-hour step the number is
also the mean flux in W m⁻². Each row's `Observation period` names the interval's start and end;
the end is kept, which is the period-ending convention ERA5 uses.

The period is `studies.era5_grid`'s: 2019-09-01 to `LAST_DATE`. Each site is requested one
calendar year at a time, three requests at a time. A failed request is retried with a longer wait
each time. A year whose raw CSV exists is skipped, so a crashed run resumes where it stopped.

Output under `CAMS_PRODUCT_DIR / demand_points`: raw CSVs under `raw/`,
`cams_demand_points.parquet`, `lineage.json`, and `README.md`.

    uv run --with cdsapi python studies/weather_downloads/fetch_cams_demand_points.py \
        --series <private series table> [--dry-run] [--first-only]
"""

import argparse
import concurrent.futures
import datetime
import hashlib
import logging
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Final, NamedTuple

import polars as pl
from lineage import write_lineage_note, write_readme
from studies.era5_grid import (
    FIRST_DATE_OVERRIDE,
    FIRST_YEAR,
    LAST_DATE,
    LAST_YEAR,
    OUTPUT_SUFFIX,
    PUBLISHED_LAST_DATE,
    first_date_of,
)
from studies.sources import CAMS_PRODUCT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("fetch_cams_demand_points")
for _name in ("cdsapi", "ecmwf.datastores"):
    logging.getLogger(_name).setLevel(logging.WARNING)  # keep request bodies out of the log

SCRIPT_PATH: Final[str] = "studies/weather_downloads/fetch_cams_demand_points.py"
OUTPUT_DIR: Final[Path] = CAMS_PRODUCT_DIR / "demand_points"
RAW_DIR: Final[Path] = OUTPUT_DIR / "raw"
OUTPUT_PATH: Final[Path] = OUTPUT_DIR / "cams_demand_points.parquet"

ADS_URL: Final[str] = "https://ads.atmosphere.copernicus.eu/api"
SOURCE_WEB_PAGE: Final[str] = (
    "https://ads.atmosphere.copernicus.eu/datasets/cams-solar-radiation-timeseries"
)
DATASET: Final[str] = "cams-solar-radiation-timeseries"
MAX_CONCURRENT_REQUESTS: Final[int] = 3
"""The Atmosphere Data Store queues an account's jobs, so this pipelines the wait.

A client-side failure does not cancel the server-side job, so a retry can leave more than this many
jobs queued on the account."""
RETRY_WAITS_SECONDS: Final[tuple[int, ...]] = (60, 300, 900)
"""How long to wait before each retry of a failed request; a fourth failure is raised."""
UNKNOWN_ALTITUDE: Final[str] = "-999."
"""The service then reads the elevation from its own terrain model."""

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
COORDINATE_HEADER_PREFIXES: Final[tuple[str, ...]] = ("# Latitude", "# Longitude", "# Altitude")
"""Header lines the service writes that name where the request was made."""
DECIMAL_TOKEN: Final[re.Pattern[str]] = re.compile(r"(?<![\d.])-?\d+\.\d{2,}(?!\d)")
"""A number with at least two decimals; version strings such as 4.6 and 1.6 have one."""
LOCATION_TOLERANCE_DEGREES: Final[float] = 0.01
VALUE_COLUMNS: Final[tuple[str, ...]] = ("ghi_w_m2", "clear_sky_ghi_w_m2", "reliability")


class SiteYear(NamedTuple):
    """One download: a site and the calendar year to request for it.

    Attributes:
        site: The anonymous label, `D1` to `D22`.
        latitude: Sent to the service and never written.
        longitude: Sent to the service and never written.
        year: The calendar year to request.
    """

    site: str
    latitude: float
    longitude: float
    year: int


def _api_key() -> str:
    """Return the Copernicus key from `~/.cdsapirc`, or raise if it holds none."""
    for line in Path.home().joinpath(".cdsapirc").read_text().splitlines():
        if line.startswith("key:"):
            return line.split(":", 1)[1].strip()
    msg = "no key found in ~/.cdsapirc"
    raise RuntimeError(msg)


def date_range_of(*, year: int) -> str:
    """Return the `first/last` date range to request in `year`."""
    last = LAST_DATE if year == LAST_YEAR else f"{year}-12-31"
    return f"{first_date_of(year=year)}/{last}"


def request_body(*, job: SiteYear) -> dict[str, object]:
    """Return the request body for one site-year."""
    return {
        "sky_type": "observed_cloud",
        "location": {"latitude": job.latitude, "longitude": job.longitude},
        "altitude": UNKNOWN_ALTITUDE,
        "date": [date_range_of(year=job.year)],
        "time_step": "1hour",
        "time_reference": "universal_time",
        "data_format": "csv",
    }


def csv_path(*, site: str, year: int) -> Path:
    """Return where one site-year's download lands; the name holds the whole requested range."""
    first, last = date_range_of(year=year).split("/")
    return RAW_DIR / f"cams_{site}_{first}_to_{last}.csv"


def is_downloaded(*, path: Path) -> bool:
    """Return whether a complete, non-empty CSV is at `path` (a download is renamed on success)."""
    return path.exists() and path.stat().st_size > 0


def strip_coordinates(*, path: Path) -> None:
    """Remove the header lines that name the requested location, rewriting the file in place."""
    kept = [
        line
        for line in path.read_text().splitlines(keepends=True)
        if not line.startswith(COORDINATE_HEADER_PREFIXES)
    ]
    path.write_text("".join(kept))


def assert_no_coordinates(*, path: Path, job: SiteYear) -> None:
    """Delete the file and raise if a header line still names the requested location.

    Args:
        path: The stripped CSV.
        job: The request the CSV answers.

    Raises:
        RuntimeError: If any `#` line holds the latitude or longitude.
    """
    lines = path.read_text().splitlines()
    named = [
        float(token)
        for line in lines
        if line.startswith("#")
        for token in DECIMAL_TOKEN.findall(line)
    ]
    if any(
        abs(value) - abs(v) < LOCATION_TOLERANCE_DEGREES
        and abs(value) - abs(v) > -LOCATION_TOLERANCE_DEGREES
        for value in named
        for v in (job.latitude, job.longitude)
    ):
        path.unlink()
        msg = "a header line still names the requested location"
        raise RuntimeError(msg)
    data = [line for line in lines if line and not line.startswith("#")]
    if not data:
        path.unlink()
        msg = "the download holds no data line"
        raise RuntimeError(msg)


def fetch_one(*, job: SiteYear) -> Path:
    """Download one site-year with retries, skipping the request if the file is already there.

    Args:
        job: Which site and year to fetch.

    Returns:
        The path the CSV landed at.
    """
    destination = csv_path(site=job.site, year=job.year)
    if is_downloaded(path=destination):
        return destination

    import cdsapi  # ty: ignore[unresolved-import]  # `uv run --with cdsapi` only

    client = cdsapi.Client(url=ADS_URL, key=_api_key(), quiet=True)
    partial = destination.with_suffix(".csv.partial")
    for attempt in range(len(RETRY_WAITS_SECONDS) + 1):
        partial.unlink(missing_ok=True)
        started = time.monotonic()
        try:
            client.retrieve(DATASET, request_body(job=job), str(partial))
            strip_coordinates(path=partial)
            assert_no_coordinates(path=partial, job=job)
        except Exception as error:  # noqa: BLE001  # any failure of the remote call is retried, then raised
            partial.unlink(missing_ok=True)
            _LOG.warning("site %s %d: %s", job.site, job.year, type(error).__name__)
            if attempt == len(RETRY_WAITS_SECONDS):
                msg = f"site {job.site} {job.year} failed after {attempt + 1} attempts"
                raise RuntimeError(msg) from None
            wait = RETRY_WAITS_SECONDS[attempt]
            _LOG.warning(
                "site %s %d failed (attempt %d); retrying in %d s",
                job.site,
                job.year,
                attempt + 1,
                wait,
            )
            time.sleep(wait)
        else:
            partial.rename(destination)
            _LOG.info(
                "site %s %d downloaded in %.0f s", job.site, job.year, time.monotonic() - started
            )
            break
    return destination


def read_one(*, path: Path, site: str) -> pl.DataFrame:
    """Parse one downloaded CSV into the tidy columns.

    Args:
        path: The downloaded CSV.
        site: The label stamped on every row.

    Returns:
        One row per hour with `site`, `time` (UTC, the end of the hour), the all-sky and clear-sky
        global fluxes as `Float32` W m⁻², and the reliability flag.
    """
    frame = pl.read_csv(
        path, separator=";", comment_prefix="#", has_header=False, new_columns=list(COLUMN_NAMES)
    )
    ends = pl.col("observation_period").str.split("/")
    parse = "%Y-%m-%dT%H:%M:%S%.f"
    lengths = frame.select(
        (ends.list.last().str.to_datetime(parse) - ends.list.first().str.to_datetime(parse))
        .dt.total_minutes()
        .unique()
    )["observation_period"].to_list()
    if lengths != [60]:
        msg = f"observation periods are not all one hour: {lengths}"
        raise ValueError(msg)
    return frame.select(
        site=pl.lit(site),
        time=ends.list.last().str.to_datetime(parse).dt.replace_time_zone("UTC"),
        ghi_w_m2=pl.when(pl.col("ghi_wh_m2") < 0)
        .then(None)
        .otherwise(pl.col("ghi_wh_m2"))
        .cast(pl.Float32),
        clear_sky_ghi_w_m2=pl.when(pl.col("clear_sky_ghi_wh_m2") < 0)
        .then(None)
        .otherwise(pl.col("clear_sky_ghi_wh_m2"))
        .cast(pl.Float32),
        reliability=pl.col("reliability").cast(pl.Float32),
    )


def expected_hours(*, year: int) -> int:
    """Return how many hourly rows the request for `year` should yield, per site."""
    first, last = (datetime.date.fromisoformat(d) for d in date_range_of(year=year).split("/"))
    return ((last - first).days + 1) * 24


def count_bad_values(*, frame: pl.DataFrame) -> dict[str, int]:
    """Count the null and NaN values in each value column, before any row is dropped."""
    counts: dict[str, int] = {}
    for column in VALUE_COLUMNS:
        counts[f"{column}_null"] = frame[column].null_count()
        counts[f"{column}_nan"] = int(frame[column].is_nan().sum())
    return counts


def find_gaps(*, tidy: pl.DataFrame, jobs: list[SiteYear]) -> dict[str, int]:
    """Return rows found minus rows expected for each site-year where that is not zero.

    A row belongs to the year in which its hour starts.
    """
    found = {
        (row["site"], row["year"]): row["len"]
        for row in tidy.group_by("site", year=(pl.col("time") - pl.duration(hours=1)).dt.year())
        .len()
        .to_dicts()
    }
    gaps = {}
    for job in jobs:
        difference = found.get((job.site, job.year), 0) - expected_hours(year=job.year)
        if difference != 0:
            gaps[f"{job.site}/{job.year}"] = difference
    return gaps


def _git_commit() -> str:
    """Return the repository's current commit hash, or `unknown` if git cannot say."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()
    except OSError, subprocess.CalledProcessError:
        return "unknown"


def build_jobs(*, series: pl.DataFrame) -> list[SiteYear]:
    """Return one job per site per calendar year, in site order then year order."""
    return [
        SiteYear(
            site=str(row["site"]),
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            year=year,
        )
        for row in series.to_dicts()
        for year in range(FIRST_YEAR, LAST_YEAR + 1)
    ]


def write_notes(
    *,
    tidy: pl.DataFrame,
    jobs: list[SiteYear],
    n_sites: int,
    bad_values: dict[str, int],
    gaps: dict[str, int],
) -> None:
    """Write `lineage.json` and `README.md` from the measured frame. No location is written."""
    first_time, last_time = tidy["time"].min(), tidy["time"].max()
    write_lineage_note(
        product_dir=OUTPUT_DIR,
        source_address=ADS_URL,
        request_description=(
            f"{DATASET}, observed-cloud sky type, 1-hour step, universal time, CSV, one request "
            f"per site per calendar year for {first_date_of(year=FIRST_YEAR)} to {LAST_DATE}"
        ),
        variables=["ghi", "clear_sky_ghi", "reliability"],
        extra={
            "dataset": DATASET,
            "script": SCRIPT_PATH,
            "git_commit": _git_commit(),
            "sites": n_sites,
            "sites_with_rows": tidy["site"].n_unique(),
            "ghi_w_m2_min_max": [tidy["ghi_w_m2"].min(), tidy["ghi_w_m2"].max()],
            "clear_sky_ghi_w_m2_max": tidy["clear_sky_ghi_w_m2"].max(),
            "share_ghi_above_1p2_clear_sky": float(
                (tidy["ghi_w_m2"] > 1.2 * tidy["clear_sky_ghi_w_m2"]).mean()  # ty: ignore[invalid-argument-type]
            ),
            "site_years": len(jobs),
            "rows": tidy.height,
            "first_time_utc": first_time,
            "last_time_utc": last_time,
            "null_and_nan_counts_before_dropping": bad_values,
            "site_years_with_row_count_gap": gaps,
        },
    )
    write_readme(
        product_dir=OUTPUT_DIR,
        product_name="CAMS solar radiation time-series at the 22 demand series' locations",
        source_web_page=SOURCE_WEB_PAGE,
        script_path=SCRIPT_PATH,
        lineage_filenames=["lineage.json"],
        columns={
            "site": "Anonymous label `D1` to `D22`. Never an identifier or a coordinate.",
            "time": "UTC, timezone-aware. The END of the hour each value averages over, from "
            f"{first_time} to {last_time}.",
            "ghi_w_m2": "Global horizontal irradiance, W/m², mean over the hour ending at `time`. "
            "`Float32`.",
            "clear_sky_ghi_w_m2": "The service's clear-sky global horizontal irradiance, W/m². "
            "`Float32`.",
            "reliability": "The service's reliability flag, from "
            f"{tidy['reliability'].min()} to {tidy['reliability'].max()} in this file. `Float32`.",
        },
        missing_value_convention=(
            f"No null or NaN is written. {sum(bad_values.values())} null or NaN values were found "
            f"before rows were dropped ({bad_values}). A dropped hour is an absent row: "
            f"{len(gaps)} site-years differ from one row per requested hour ({gaps or 'none'})."
        ),
        gotchas=[
            f"The last requested date is `studies.era5_grid.LAST_DATE`, {LAST_DATE}.",
            (
                "The raw CSVs in `raw/` have the service's location header lines stripped. The "
                "mapping from label to series is in the private demand-experiment series table."
            ),
        ],
        external_docs={
            "CAMS solar radiation time-series (Atmosphere Data Store)": SOURCE_WEB_PAGE,
        },
    )


def check_series(*, series: pl.DataFrame) -> None:
    """Raise unless the table holds 22 rows labelled D1 to D22 with no nulls.

    Args:
        series: The private series table, with a `site` label column.

    Raises:
        SystemExit: If the labels or the shape are wrong, so that no identifier can reach a file
            name. The message names no value.
    """
    labels = series["site"].to_list()
    if (
        series.height != 22
        or series["site"].n_unique() != 22
        or not all(re.fullmatch(r"D([1-9]|1\d|2[0-2])", str(label)) for label in labels)
        or series.null_count().sum_horizontal().item() != 0
    ):
        msg = "series table must hold 22 rows, labelled D1 to D22, with no nulls"
        raise SystemExit(msg)


def main() -> int:
    """Download every site-year and write the tidy frame, `lineage.json`, and `README.md`."""
    if FIRST_DATE_OVERRIDE or OUTPUT_SUFFIX or LAST_DATE != PUBLISHED_LAST_DATE:
        msg = "this script writes only the published window; unset the ERA5_* overrides"
        raise SystemExit(msg)
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--series", type=Path, required=True, help="Private series table.")
    parser.add_argument("--dry-run", action="store_true", help="List the requests and stop.")
    parser.add_argument("--first-only", action="store_true", help="Fetch one site-year and stop.")
    args = parser.parse_args()
    series = pl.read_parquet(args.series).select("site", "latitude", "longitude")
    check_series(series=series)
    jobs = build_jobs(series=series)
    _LOG.info("%d site-years for %d sites", len(jobs), series.height)
    if args.dry_run:
        for job in jobs:
            state = (
                "skip (downloaded)"
                if is_downloaded(path=csv_path(site=job.site, year=job.year))
                else "request"
            )
            _LOG.info("%s %s %d (%s)", state, job.site, job.year, date_range_of(year=job.year))
        return 0
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(series.sort("site").write_csv().encode()).hexdigest()
    marker = OUTPUT_DIR / ".series_fingerprint"
    if marker.exists() and marker.read_text() != fingerprint:
        msg = "series table differs from the one the raw files were fetched with"
        raise SystemExit(msg)
    marker.write_text(fingerprint)
    if args.first_only:
        fetch_one(job=jobs[0])
        return 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as pool:
        futures = [pool.submit(fetch_one, job=job) for job in jobs]
        try:
            for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                future.result()
                _LOG.info("%d of %d site-years ready", done, len(jobs))
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    frame = pl.concat(
        [read_one(path=csv_path(site=j.site, year=j.year), site=j.site) for j in jobs]
    )
    duplicates = frame.height - frame.unique(subset=["site", "time"]).height
    if duplicates:
        msg = f"{duplicates} duplicate (site, time) rows across the raw files"
        raise RuntimeError(msg)
    bad_values = count_bad_values(frame=frame)
    tidy = (
        frame.drop_nulls(subset=["time", *VALUE_COLUMNS])
        .filter(~pl.any_horizontal(pl.col(*VALUE_COLUMNS).is_nan()))
        .sort("site", "time")
    )
    gaps = find_gaps(tidy=tidy, jobs=jobs)
    for key, difference in gaps.items():
        _LOG.warning("site-year %s has %+d rows against one per requested hour", key, difference)
    partial = OUTPUT_PATH.with_suffix(".parquet.partial")
    tidy.write_parquet(partial)
    partial.rename(OUTPUT_PATH)
    write_notes(tidy=tidy, jobs=jobs, n_sites=series.height, bad_values=bad_values, gaps=gaps)
    _LOG.info(
        "wrote %d rows covering %s to %s", tidy.height, tidy["time"].min(), tidy["time"].max()
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
