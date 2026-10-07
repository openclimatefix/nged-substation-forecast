"""Download CAMS satellite-derived irradiance at public points: solar BMUs and a coarse GB grid.

One-off throwaway script. It uses the same Atmosphere Data Store dataset
(`cams-solar-radiation-timeseries`), the same `~/.cdsapirc` key, and the same columns as
`studies/beam_diffuse_split/fetch_cams.py`, but at points whose coordinates are public, so the
latitude and longitude are written to the output. The private-site CAMS download is not read or
changed, and nothing here reads the private site roster.

The points are the 10 single-site solar BMUs in the solar-BMU census (two BMUs that share one
coordinate are requested once and labelled with both ids), and 18 hard-coded points that cover Great
Britain coarsely.

The service publishes irradiation in Wh m⁻² summed over each step, so at a 1-hour step the number is
also the mean flux in W m⁻², and no conversion is needed. Each row's `Observation period` names the
interval's start and end; the end is kept, which is the period-ending convention ERA5 uses.

The period runs from 2025-01-01 to `studies.era5_grid.LAST_DATE`, the same last date
`fetch_cams.py` uses. Each point is requested one calendar year at a time, and a year whose raw CSV
already exists is skipped, so a crashed run resumes where it stopped. Output goes to
`CAMS_public_points/` next to `CAMS/`: raw CSVs under `raw/`, the tidy
`cams_public_points.parquet`, a `lineage.json`, and a `README.md` written from the measured frame.

Run it with `uv run --with cdsapi python studies/weather_downloads/fetch_cams_public_points.py`.
Add `--dry-run` to list the requests without making any.
"""

import argparse
import concurrent.futures
import datetime
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Final, NamedTuple

import polars as pl
from lineage import write_lineage_note, write_readme
from studies.era5_grid import LAST_DATE, LAST_YEAR
from studies.sources import CAMS_PUBLIC_POINTS_DIR, SOLAR_BMU_CENSUS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("fetch_cams_public_points")

SCRIPT_PATH: Final[str] = "studies/weather_downloads/fetch_cams_public_points.py"
OUTPUT_DIR: Final[Path] = CAMS_PUBLIC_POINTS_DIR
RAW_DIR: Final[Path] = OUTPUT_DIR / "raw"
OUTPUT_PATH: Final[Path] = OUTPUT_DIR / "cams_public_points.parquet"
CENSUS_PATH: Final[Path] = SOLAR_BMU_CENSUS_DIR / "solar_bmus.parquet"

ADS_URL: Final[str] = "https://ads.atmosphere.copernicus.eu/api"
SOURCE_WEB_PAGE: Final[str] = (
    "https://ads.atmosphere.copernicus.eu/datasets/cams-solar-radiation-timeseries"
)
DATASET: Final[str] = "cams-solar-radiation-timeseries"

FIRST_DATE: Final[str] = "2025-01-01"
FIRST_YEAR: Final[int] = int(FIRST_DATE[:4])

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

GB_GRID: Final[tuple[tuple[str, float, float], ...]] = (
    ("northern Highlands", 57.5, -4.0),
    ("southern Highlands", 56.4, -3.4),
    ("south-west Scotland", 55.5, -4.2),
    ("Scottish Borders", 55.2, -2.6),
    ("Tyneside", 54.8, -1.6),
    ("north-west England", 53.9, -2.5),
    ("Lincolnshire", 53.4, -0.8),
    ("mid Wales", 52.5, -3.5),
    ("south Wales", 51.7, -3.5),
    ("West Midlands", 52.5, -1.8),
    ("East Midlands", 52.9, -0.7),
    ("Norfolk", 52.6, 0.9),
    ("London", 51.6, -0.3),
    ("Kent", 51.1, 0.6),
    ("Hampshire coast", 50.9, -1.4),
    ("Somerset", 51.0, -2.8),
    ("Cornwall", 50.5, -4.3),
    ("Oxfordshire", 51.8, -1.2),
)
"""The coarse GB grid as (region, latitude, longitude), in the order of `gb_01` to `gb_18`."""


class Point(NamedTuple):
    """One place to request, with the label its rows carry.

    Attributes:
        point_id: The label written to the output, such as `bmu_T_BURWS-1` or `gb_07`.
        bmu_ids: The BMU ids at this coordinate, or `None` for a grid point.
        region: A short place name for a grid point, or `None` for a BMU point.
        latitude: Latitude in degrees north.
        longitude: Longitude in degrees east.
    """

    point_id: str
    bmu_ids: tuple[str, ...] | None
    region: str | None
    latitude: float
    longitude: float


class PointYear(NamedTuple):
    """One download: a point and the calendar year to request for it.

    Attributes:
        point: The point to request.
        year: The year to request.
    """

    point: Point
    year: int


def bmu_points(*, census: pl.DataFrame) -> list[Point]:
    """Build one point per distinct coordinate among the single-site solar BMUs.

    Args:
        census: The solar-BMU census, with `scope`, `elexon_bmu_id`, `latitude` and `longitude`.

    Returns:
        One point per coordinate, sorted by BMU id. A coordinate shared by several BMUs is one point
        carrying every id, and its `point_id` names the first id in sorted order.
    """
    single_site = census.filter(pl.col("scope") == "single-site").select(
        "elexon_bmu_id", "latitude", "longitude"
    )
    grouped = (
        single_site.group_by("latitude", "longitude")
        .agg(bmu_ids=pl.col("elexon_bmu_id").sort())
        .with_columns(first_id=pl.col("bmu_ids").list.first())
        .sort("first_id")
    )
    return [
        Point(
            point_id=f"bmu_{row['first_id']}",
            bmu_ids=tuple(row["bmu_ids"]),
            region=None,
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
        )
        for row in grouped.to_dicts()
    ]


def grid_points() -> list[Point]:
    """Return the coarse GB grid as points labelled `gb_01` to `gb_18`."""
    return [
        Point(
            point_id=f"gb_{number:02d}",
            bmu_ids=None,
            region=region,
            latitude=latitude,
            longitude=longitude,
        )
        for number, (region, latitude, longitude) in enumerate(GB_GRID, start=1)
    ]


def _api_key() -> str:
    """Read the Copernicus API key, which both data stores accept.

    Returns:
        The key from `~/.cdsapirc`.

    Raises:
        RuntimeError: If the file holds no `key:` line.
    """
    for line in Path.home().joinpath(".cdsapirc").read_text().splitlines():
        if line.startswith("key:"):
            return line.split(":", 1)[1].strip()
    msg = "no key found in ~/.cdsapirc"
    raise RuntimeError(msg)


def _date_range_of(*, year: int) -> str:
    """Return the `first/last` date range to request in `year`."""
    first = FIRST_DATE if year == FIRST_YEAR else f"{year}-01-01"
    last = LAST_DATE if year == LAST_YEAR else f"{year}-12-31"
    return f"{first}/{last}"


def _request_body(*, point: Point, year: int) -> dict[str, object]:
    """Return the request body for one point-year."""
    return {
        "sky_type": "observed_cloud",
        "location": {"latitude": point.latitude, "longitude": point.longitude},
        "altitude": UNKNOWN_ALTITUDE,
        "date": [_date_range_of(year=year)],
        "time_step": "1hour",
        "time_reference": "universal_time",
        "data_format": "csv",
    }


def csv_path(*, raw_dir: Path, point_id: str, year: int) -> Path:
    """Return where one point-year's download lands.

    Args:
        raw_dir: The folder of raw CSVs.
        point_id: The point's label.
        year: The calendar year requested.

    Returns:
        The CSV's path. The final year's name carries the requested end date, so a run with a
        different `LAST_DATE` does not reuse a file that stops earlier or later.
    """
    suffix = f"_to_{LAST_DATE}" if year == LAST_YEAR else ""
    return raw_dir / f"cams_{point_id}_{year}{suffix}.csv"


def _fetch_one(*, job: PointYear, raw_dir: Path) -> Path:
    """Download one point-year, skipping the request if the file is already there.

    Args:
        job: Which point and year to fetch.
        raw_dir: The folder of raw CSVs.

    Returns:
        The path the CSV landed at.
    """
    destination = csv_path(raw_dir=raw_dir, point_id=job.point.point_id, year=job.year)
    if is_downloaded(path=destination):
        return destination

    import cdsapi  # ty: ignore[unresolved-import]  # `uv run --with cdsapi` only

    client = cdsapi.Client(url=ADS_URL, key=_api_key(), quiet=True)
    partial = destination.with_suffix(".csv.partial")
    client.retrieve(DATASET, _request_body(point=job.point, year=job.year), str(partial))
    partial.rename(destination)
    _LOG.info("point %s %d downloaded", job.point.point_id, job.year)
    return destination


def is_downloaded(*, path: Path) -> bool:
    """Return whether a complete, non-empty CSV is already at `path`.

    A download is written to a `.partial` name and renamed on success, so the final name only ever
    holds a finished file.

    Args:
        path: The CSV's final path.

    Returns:
        `True` if the file exists and is not empty.
    """
    return path.exists() and path.stat().st_size > 0


def read_one(*, path: Path, point: Point) -> pl.DataFrame:
    """Parse one downloaded CSV into the tidy columns.

    Args:
        path: The downloaded CSV.
        point: The point the CSV was requested for.

    Returns:
        One row per hour with the point's labels and coordinates, `time` (UTC, the end of the hour),
        the all-sky and clear-sky global fluxes as `Float32` W m⁻², and the reliability flag.
    """
    frame = pl.read_csv(
        path,
        separator=";",
        comment_prefix="#",
        has_header=False,
        new_columns=list(COLUMN_NAMES),
    )
    return frame.select(
        point_id=pl.lit(point.point_id),
        bmu_ids=pl.lit(
            None if point.bmu_ids is None else list(point.bmu_ids), dtype=pl.List(pl.String)
        ),
        latitude=pl.lit(point.latitude),
        longitude=pl.lit(point.longitude),
        time=pl.col("observation_period")
        .str.split("/")
        .list.last()
        .str.to_datetime("%Y-%m-%dT%H:%M:%S%.f")
        .dt.replace_time_zone("UTC"),
        ghi_w_m2=pl.col("ghi_wh_m2").cast(pl.Float32),
        clear_sky_ghi_w_m2=pl.col("clear_sky_ghi_wh_m2").cast(pl.Float32),
        reliability=pl.col("reliability").cast(pl.Float32),
    )


VALUE_COLUMNS: Final[tuple[str, ...]] = ("ghi_w_m2", "clear_sky_ghi_w_m2", "reliability")
"""The columns that must hold a finite number on every written row."""


def expected_hours(*, year: int) -> int:
    """Return how many hourly rows the request for `year` should yield, per point.

    Args:
        year: The calendar year requested.

    Returns:
        24 times the number of days in the requested date range.
    """
    first, last = (datetime.date.fromisoformat(day) for day in _date_range_of(year=year).split("/"))
    return ((last - first).days + 1) * 24


def count_bad_values(*, frame: pl.DataFrame) -> dict[str, int]:
    """Count the null and NaN values in the value columns.

    Args:
        frame: The concatenated frame from `read_one`, before any row is dropped.

    Returns:
        A count per `<column>_null` and `<column>_nan`.
    """
    counts: dict[str, int] = {}
    for column in VALUE_COLUMNS:
        counts[f"{column}_null"] = frame[column].null_count()
        counts[f"{column}_nan"] = int(frame[column].is_nan().sum())
    return counts


def drop_bad_rows(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Drop rows whose time or any value column is null or NaN.

    Args:
        frame: The concatenated frame from `read_one`.

    Returns:
        The frame without those rows.
    """
    return frame.drop_nulls(subset=["time", *VALUE_COLUMNS]).filter(
        ~pl.any_horizontal(pl.col(*VALUE_COLUMNS).is_nan())
    )


def find_gaps(*, tidy: pl.DataFrame, jobs: list[PointYear]) -> dict[str, int]:
    """Find point-years whose row count differs from one row per requested hour.

    Args:
        tidy: The written frame.
        jobs: The point-years that were requested.

    Returns:
        A map from `<point_id>/<year>` to rows found minus rows expected, for each point-year where
        that is not zero. A row belongs to the year in which its hour starts.
    """
    found = {
        (row["point_id"], row["year"]): row["len"]
        for row in tidy.group_by("point_id", year=(pl.col("time") - pl.duration(hours=1)).dt.year())
        .len()
        .to_dicts()
    }
    gaps = {}
    for job in jobs:
        difference = found.get((job.point.point_id, job.year), 0) - expected_hours(year=job.year)
        if difference != 0:
            gaps[f"{job.point.point_id}/{job.year}"] = difference
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


def _points_section(*, points: list[Point]) -> str:
    """Return the README section that lists every point."""
    lines = ["", "## Points", ""]
    for point in points:
        label = ", ".join(point.bmu_ids) if point.bmu_ids else point.region
        lines.append(f"- `{point.point_id}` ({label}): {point.latitude}, {point.longitude}")
    return "\n".join(lines) + "\n"


def _write_notes(
    *,
    points: list[Point],
    tidy: pl.DataFrame,
    jobs: list[PointYear],
    bad_values: dict[str, int],
    gaps: dict[str, int],
) -> None:
    """Write `lineage.json` and `README.md` from the measured frame."""
    first_time = tidy["time"].min()
    last_time = tidy["time"].max()
    reliability_min = tidy["reliability"].min()
    reliability_max = tidy["reliability"].max()
    bad_total = sum(bad_values.values())
    write_lineage_note(
        product_dir=OUTPUT_DIR,
        source_address=ADS_URL,
        request_description=(
            f"{DATASET}, observed-cloud sky type, 1-hour step, universal time, CSV, one request "
            f"per point per calendar year for {FIRST_DATE} to {LAST_DATE}"
        ),
        variables=["ghi", "clear_sky_ghi", "reliability"],
        extra={
            "dataset": DATASET,
            "request_body_example": _request_body(point=points[0], year=FIRST_YEAR),
            "first_date_requested": FIRST_DATE,
            "last_date_requested": LAST_DATE,
            "first_time_utc": first_time,
            "last_time_utc": last_time,
            "script": SCRIPT_PATH,
            "git_commit": _git_commit(),
            "points": len(points),
            "point_years": len(jobs),
            "rows": tidy.height,
            "null_and_nan_counts_before_dropping": bad_values,
            "point_years_with_row_count_gap": gaps,
        },
    )
    readme_path = write_readme(
        product_dir=OUTPUT_DIR,
        product_name="CAMS solar radiation time-series at public points",
        source_web_page=SOURCE_WEB_PAGE,
        script_path=SCRIPT_PATH,
        lineage_filenames=["lineage.json"],
        columns={
            "point_id": "`bmu_<first BMU id>` for a solar-BMU point, `gb_01` to `gb_18` for a grid "
            "point.",
            "bmu_ids": "The BMU ids at the point, a list of strings. Null for a grid point.",
            "latitude": "Latitude of the request, degrees north.",
            "longitude": "Longitude of the request, degrees east.",
            "time": "UTC, timezone-aware. The END of the hour each value averages over (the end of "
            f"the service's `Observation period`), from {first_time} to {last_time}.",
            "ghi_w_m2": "Global horizontal irradiance, W/m², mean over the hour ending at `time`. "
            "The service publishes Wh/m² per 1-hour step, which is numerically the mean flux in "
            "W/m². `Float32`.",
            "clear_sky_ghi_w_m2": "The service's clear-sky global horizontal irradiance, W/m², for "
            "the same hour. `Float32`.",
            "reliability": "The service's reliability flag for the hour, measured from "
            f"{reliability_min} to {reliability_max} in this file: the share of the hour's "
            "satellite data judged reliable. `Float32`.",
        },
        missing_value_convention=(
            f"No null or NaN is written. Rows with a null or NaN in `time`, `ghi_w_m2`, "
            f"`clear_sky_ghi_w_m2` or `reliability` are dropped, and {bad_total} null or NaN "
            f"values were found before dropping ({bad_values}). A dropped or never-served hour is "
            f"an absent row: {len(gaps)} point-years differ from one row per requested hour "
            f"({gaps or 'none'})."
        ),
        gotchas=[
            (
                f"`cams_public_points.parquet` holds {tidy.height:,} rows for {len(points)} "
                f"points from {first_time} to {last_time}."
            ),
            (
                "The raw per-point-year CSVs in `raw/` keep the service's commented header, which "
                "names the requested coordinates. The points are public, so nothing is stripped."
            ),
            f"The last requested date is `studies.era5_grid.LAST_DATE`, {LAST_DATE}.",
        ],
        external_docs={
            "CAMS solar radiation time-series (Atmosphere Data Store)": SOURCE_WEB_PAGE,
            "Qu et al. (2017), the Heliosat-4 method behind the service": (
                "https://doi.org/10.1127/metz/2016/0781"
            ),
        },
    )
    with readme_path.open("a") as readme:
        readme.write(_points_section(points=points))


def build_jobs(*, points: list[Point]) -> list[PointYear]:
    """Return one job per point per calendar year in the period.

    Args:
        points: The points to request.

    Returns:
        The jobs, in point order and then year order.
    """
    return [
        PointYear(point=point, year=year)
        for point in points
        for year in range(FIRST_YEAR, LAST_YEAR + 1)
    ]


def main() -> int:
    """Download every point-year and write the tidy frame, `lineage.json`, and `README.md`."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument(
        "--dry-run", action="store_true", help="List the requests that would be made and stop."
    )
    arguments = parser.parse_args()

    points = bmu_points(census=pl.read_parquet(CENSUS_PATH)) + grid_points()
    jobs = build_jobs(points=points)
    if arguments.dry_run:
        for job in jobs:
            path = csv_path(raw_dir=RAW_DIR, point_id=job.point.point_id, year=job.year)
            action = "skip (already downloaded)" if is_downloaded(path=path) else "request"
            body = json.dumps(_request_body(point=job.point, year=job.year))
            print(f"{action}: {job.point.point_id} {job.year} {body}")
        print(f"{len(jobs)} point-years for {len(points)} points")
        return 0

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    _LOG.info("%d point-years to fetch for %d points", len(jobs), len(points))
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as pool:
        futures = [pool.submit(_fetch_one, job=job, raw_dir=RAW_DIR) for job in jobs]
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            future.result()
            _LOG.info("%d of %d point-years ready", done, len(jobs))

    combined = pl.concat(
        [
            read_one(
                path=csv_path(raw_dir=RAW_DIR, point_id=job.point.point_id, year=job.year),
                point=job.point,
            )
            for job in jobs
        ]
    )
    bad_values = count_bad_values(frame=combined)
    tidy = (
        drop_bad_rows(frame=combined)
        .unique(subset=["point_id", "time"], keep="first")
        .sort("point_id", "time")
    )
    tidy.write_parquet(OUTPUT_PATH)
    gaps = find_gaps(tidy=tidy, jobs=jobs)
    for key, difference in gaps.items():
        _LOG.warning("point-year %s has %+d rows against one per requested hour", key, difference)
    _write_notes(points=points, tidy=tidy, jobs=jobs, bad_values=bad_values, gaps=gaps)
    _LOG.info(
        "wrote %d rows covering %s to %s for %d points",
        tidy.height,
        tidy["time"].min(),
        tidy["time"].max(),
        tidy["point_id"].n_unique(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
