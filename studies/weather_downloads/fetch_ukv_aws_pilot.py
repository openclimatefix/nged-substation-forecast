r"""Pilot download of the Met Office UKV from its AWS open-data bucket, cropped at read time.

One-off throwaway script for the comparison of the UKV that CEDA archives against the UKV that the
Met Office serves live. The bucket `met-office-atmospheric-model-data` is public and anonymous, and
holds a rolling two-year window of the 2 km deterministic UKV
(<https://registry.opendata.aws/met-office-uk-deterministic/>).

**Layout.** Each hourly run has its own prefix `uk-deterministic-2km/<run>Z/`, and holds one NetCDF4
(HDF5) file per variable per valid time: `<valid>Z-PT<lead>H00M-<variable>.nc`. Each file covers the
whole 970 by 1042 grid on a Lambert azimuthal equal-area projection (not CEDA's national grid).
Variables on height levels hold 56 levels in one file.

**Cropping at read time.** The files are chunked 128 by 128 cells (one level per chunk) and
zlib-compressed, so an HTTP range read fetches only the chunks that overlap the private trial-area
box. The script therefore never transfers a whole file, and the per-object wire size is a few
hundred kilobytes instead of 1 to 60 MB. The cropped rectangle (bounding rectangle of every cell
inside the box) is saved with each run. **The saved coordinates and the rectangle file
`_crop_rectangle.json` reveal the private box, so they stay in the private data folder, and no log
line, README or lineage note carries a coordinate or a cell count.**

**What is fetched.** Screen temperature, 10 m wind speed and direction, the total, direct, and
diffuse downward short-wave, and wind speed and direction at the hub heights in
`HUB_HEIGHTS_M`, for leads 0 to 5 of every requested run, on two days: one near the start of the
bucket's window and one recent (override with `--days`). One compressed `.npz` per run holds every
variable as `(lead, [height,] row, column)`. Each run is written to a `.partial` file and renamed,
so a re-run skips the runs already done. The output folder is write-once: a run whose file exists is
never rewritten.

Run `--dry-run` first: it lists the bucket and prints what it would fetch and the byte totals,
without reading any data file apart from one small file's coordinate axes.

    uv run --with h5netcdf --with fsspec --with aiohttp --with requests \\
        python studies/weather_downloads/fetch_ukv_aws_pilot.py --dry-run
"""

import argparse
import datetime as dt
import json
import re
import shutil
import time
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from xml.etree import ElementTree

import fsspec
import h5netcdf
import numpy as np
import requests
from lineage import write_lineage_note, write_readme
from pyproj import CRS, Transformer
from studies.sources import NWP_DOWNLOADS_DIR
from studies.trial_area import load_trial_area_box

CODE_VERSION: Final[str] = "fetch_ukv_aws_pilot-1"
BUCKET_URL: Final[str] = "https://met-office-atmospheric-model-data.s3-eu-west-2.amazonaws.com"
PREFIX: Final[str] = "uk-deterministic-2km/"
REGISTRY_URL: Final[str] = "https://registry.opendata.aws/met-office-uk-deterministic/"
PRODUCT_DIR: Final[Path] = NWP_DOWNLOADS_DIR / "UKV-AWS_pilot"
"""The write-once output folder."""

LEADS_HOURS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4, 5)
HUB_HEIGHTS_M: Final[tuple[float, ...]] = (50.0, 75.0, 100.0, 125.0, 150.0)
"""The heights, in metres above ground, kept from the 56-level wind files."""

SURFACE_FILES: Final[tuple[str, ...]] = (
    "temperature_at_screen_level",
    "wind_speed_at_10m",
    "wind_direction_at_10m",
    "radiation_flux_in_shortwave_total_downward_at_surface",
    "radiation_flux_in_shortwave_direct_downward_at_surface",
    "radiation_flux_in_shortwave_diffuse_downward_at_surface",
)
LEVEL_FILES: Final[tuple[str, ...]] = (
    "wind_speed_on_height_levels",
    "wind_direction_on_height_levels",
)
ALL_FILES: Final[tuple[str, ...]] = SURFACE_FILES + LEVEL_FILES

GRID_SHAPE: Final[tuple[int, int]] = (970, 1042)
RUNS_PER_DAY: Final[int] = 24
EARLY_MARGIN_DAYS: Final[int] = 2
"""The early day sits this many days after the window's first full day, because the oldest runs are
deleted object by object and may be part-gone."""
WIRE_BYTES_PER_OBJECT_ESTIMATE: Final[int] = 600_000
"""Rough bytes on the wire for one cropped read, from one test (about 120 KB read from a one-chunk
crop of two files, plus HTTP block rounding). Replace with the figure measured on the first run.
"""

WORKERS: Final[int] = 8
MAX_ATTEMPTS: Final[int] = 5
BLOCK_BYTES: Final[int] = 256 * 1024
DEFAULT_MIN_FREE_GB: Final[float] = 20.0
EPOCH: Final[dt.datetime] = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)

_S3_NAMESPACE: Final[str] = "{http://s3.amazonaws.com/doc/2006-03-01/}"
_KEY: Final[re.Pattern[str]] = re.compile(r"^(\d{8}T\d{4}Z)-PT(\d{4})H00M-(.+)\.nc$")


@dataclass(frozen=True)
class CropRectangle:
    """The bounding rectangle, in grid indices, of every cell inside the private box.

    Attributes:
        row_start: First row (counting along `projection_y_coordinate`).
        row_stop: One past the last row.
        col_start: First column (counting along `projection_x_coordinate`).
        col_stop: One past the last column.
    """

    row_start: int
    row_stop: int
    col_start: int
    col_stop: int


def _list_page(*, prefix: str, delimiter: str | None, token: str | None) -> ElementTree.Element:
    """Fetch one page of an anonymous ListObjectsV2 listing."""
    params = {"list-type": "2", "prefix": prefix}
    if delimiter is not None:
        params["delimiter"] = delimiter
    if token is not None:
        params["continuation-token"] = token
    response = requests.get(BUCKET_URL, params=params, timeout=60)
    response.raise_for_status()
    return ElementTree.fromstring(response.content)


def list_keys(*, prefix: str) -> dict[str, int]:
    """List every object under `prefix` with its size in bytes."""
    sizes: dict[str, int] = {}
    token: str | None = None
    while True:
        page = _list_page(prefix=prefix, delimiter=None, token=token)
        for item in page.iter(f"{_S3_NAMESPACE}Contents"):
            sizes[item.findtext(f"{_S3_NAMESPACE}Key", "")] = int(
                item.findtext(f"{_S3_NAMESPACE}Size", "0")
            )
        if page.findtext(f"{_S3_NAMESPACE}IsTruncated") != "true":
            return sizes
        token = page.findtext(f"{_S3_NAMESPACE}NextContinuationToken")


def list_runs() -> list[str]:
    """List every run prefix in the bucket, such as `20261002T0300Z`, oldest first."""
    runs: list[str] = []
    token: str | None = None
    while True:
        page = _list_page(prefix=PREFIX, delimiter="/", token=token)
        runs.extend(
            item.findtext(f"{_S3_NAMESPACE}Prefix", "").removeprefix(PREFIX).strip("/")
            for item in page.iter(f"{_S3_NAMESPACE}CommonPrefixes")
        )
        if page.findtext(f"{_S3_NAMESPACE}IsTruncated") != "true":
            return sorted(runs)
        token = page.findtext(f"{_S3_NAMESPACE}NextContinuationToken")


def choose_days(*, runs: Sequence[str]) -> tuple[dt.date, dt.date]:
    """Pick the early day and the recent day: full days of 24 runs only."""
    per_day: dict[str, int] = defaultdict(int)
    for run in runs:
        per_day[run[:8]] += 1
    full_days = sorted(day for day, count in per_day.items() if count == RUNS_PER_DAY)
    if len(full_days) <= EARLY_MARGIN_DAYS:
        message = "the bucket holds too few full days to choose from"
        raise RuntimeError(message)
    early = dt.datetime.strptime(full_days[EARLY_MARGIN_DAYS], "%Y%m%d").date()  # noqa: DTZ007
    recent = dt.datetime.strptime(full_days[-1], "%Y%m%d").date()  # noqa: DTZ007
    return early, recent


def wanted_keys(
    *, run: str, available: dict[str, int]
) -> tuple[list[tuple[str, int, str]], list[str]]:
    """Choose the objects of one run to fetch.

    Args:
        run: The run prefix, such as `20261002T0300Z`.
        available: Key to size for the run, from `list_keys`.

    Returns:
        The `(key, lead, file name)` triples that exist, and the names of those that do not.
    """
    run_time = dt.datetime.strptime(run, "%Y%m%dT%H%MZ").replace(tzinfo=dt.UTC)
    found: list[tuple[str, int, str]] = []
    missing: list[str] = []
    for lead in LEADS_HOURS:
        valid = run_time + dt.timedelta(hours=lead)
        for name in ALL_FILES:
            key = f"{PREFIX}{run}/{valid:%Y%m%dT%H%MZ}-PT{lead:04d}H00M-{name}.nc"
            if key in available:
                found.append((key, lead, name))
            else:
                missing.append(f"{lead}:{name}")
    return found, missing


def _data_variable(*, dataset: h5netcdf.File) -> str:
    """Return the one gridded data variable in a file, ignoring axes and bounds."""
    candidates = [
        name
        for name, variable in dataset.variables.items()
        if len(variable.dimensions) >= 2
        and variable.dimensions[-2:] == ("projection_y_coordinate", "projection_x_coordinate")
        and not name.endswith("_bnds")
    ]
    if len(candidates) != 1:
        message = f"expected one gridded variable, found {candidates}"
        raise ValueError(message)
    return candidates[0]


def _open(*, key: str) -> h5netcdf.File:
    """Open one remote file for range reads."""
    handle = fsspec.open(f"{BUCKET_URL}/{key}", "rb", block_size=BLOCK_BYTES).open()
    return h5netcdf.File(handle, "r")


def compute_crop_rectangle(*, dataset: h5netcdf.File) -> CropRectangle:
    """Find the cells of this file's grid that lie inside the private box.

    Raises:
        RuntimeError: If the grid shape is not `GRID_SHAPE` or no cell lies inside the box.
    """
    x = np.asarray(dataset["projection_x_coordinate"][:], dtype=np.float64)
    y = np.asarray(dataset["projection_y_coordinate"][:], dtype=np.float64)
    if (len(y), len(x)) != GRID_SHAPE:
        message = f"unexpected grid shape {(len(y), len(x))}"
        raise RuntimeError(message)
    crs = CRS.from_cf(dict(dataset["lambert_azimuthal_equal_area"].attrs))
    xs, ys = np.meshgrid(x, y)
    longitude, latitude = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(xs, ys)
    box = load_trial_area_box()
    inside = (
        (latitude >= box.lat_min)
        & (latitude <= box.lat_max)
        & (longitude >= box.lon_min)
        & (longitude <= box.lon_max)
    )
    rows = np.flatnonzero(inside.any(axis=1))
    cols = np.flatnonzero(inside.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        message = "no grid cell lies inside the trial-area box"
        raise RuntimeError(message)
    return CropRectangle(int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1)


def load_or_make_rectangle(*, first_key: str, write: bool) -> CropRectangle:
    """Read the saved rectangle, or compute it from one file's axes and save it if `write`."""
    path = PRODUCT_DIR / "_crop_rectangle.json"
    if path.exists():
        return CropRectangle(**json.loads(path.read_text()))
    with _open(key=first_key) as dataset:
        rectangle = compute_crop_rectangle(dataset=dataset)
    if write:
        PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rectangle.__dict__))
    return rectangle


def _check_times(*, dataset: h5netcdf.File, run: str, lead: int) -> None:
    """Check that the file's own reference time and forecast period match its name.

    Raises:
        ValueError: If either differs from the name's, which would mean a mislabelled file.
    """
    reference = int(dataset["forecast_reference_time"][()])
    period = int(dataset["forecast_period"][()])
    expected = int(dt.datetime.strptime(run, "%Y%m%dT%H%MZ").replace(tzinfo=dt.UTC).timestamp())
    if reference != expected or period != lead * 3600:
        message = f"{run} lead {lead}: file says reference {reference}, period {period}"
        raise ValueError(message)


def _check_axes(*, key: str, found: tuple[float, float], expected: tuple[float, float]) -> None:
    """Raise if a file's grid origin differs from the first file's, which would shift every cell."""
    if found != expected:
        message = f"{key}: grid axes differ from the first file's"
        raise ValueError(message)


def _read_object(
    *, key: str, run: str, lead: int, rectangle: CropRectangle, expected_axes: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray | None]:
    """Range-read the cropped field of one object, retrying transient failures.

    Returns:
        The cropped values, `(row, column)` or `(height, row, column)`, and the kept heights for a
        file on height levels, else `None`.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with _open(key=key) as dataset:
                name = _data_variable(dataset=dataset)
                _check_times(dataset=dataset, run=run, lead=lead)
                x0 = float(dataset["projection_x_coordinate"][rectangle.col_start])
                y0 = float(dataset["projection_y_coordinate"][rectangle.row_start])
                _check_axes(key=key, found=(x0, y0), expected=expected_axes)
                rows = slice(rectangle.row_start, rectangle.row_stop)
                cols = slice(rectangle.col_start, rectangle.col_stop)
                variable = dataset[name]
                if "height" not in variable.dimensions:
                    return np.asarray(variable[rows, cols], dtype=np.float32), None
                heights = np.asarray(dataset["height"][:], dtype=np.float64)
                keep = [int(np.flatnonzero(heights == h)[0]) for h in HUB_HEIGHTS_M]
                stack = [np.asarray(variable[i, rows, cols], dtype=np.float32) for i in keep]
                return np.stack(stack), heights[keep]
        except FileNotFoundError, ValueError:
            raise  # A listed object that vanishes, or a wrong file, is a fault, never retried.
        except OSError, requests.RequestException:
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(5 * 2 ** (attempt - 1))
    message = "unreachable"
    raise AssertionError(message)


def fetch_run(*, run: str, found: list[tuple[str, int, str]], rectangle: CropRectangle) -> Path:
    """Fetch one run's cropped fields and write them atomically to `runs/<run>.npz`."""
    runs_dir = PRODUCT_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    first = next(key for key, lead, _ in found if lead == 0)
    with _open(key=first) as dataset:
        x = np.asarray(dataset["projection_x_coordinate"][rectangle.col_start : rectangle.col_stop])
        y = np.asarray(dataset["projection_y_coordinate"][rectangle.row_start : rectangle.row_stop])
    axes = (float(x[0]), float(y[0]))
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            (name, lead): pool.submit(
                _read_object, key=key, run=run, lead=lead, rectangle=rectangle, expected_axes=axes
            )
            for key, lead, name in found
        }
        results = {item: future.result() for item, future in futures.items()}
    arrays: dict[str, np.ndarray] = {"x": x, "y": y, "lead_hours": np.array(LEADS_HOURS)}
    for name in ALL_FILES:
        per_lead = [results.get((name, lead)) for lead in LEADS_HOURS]
        template = next((item for item in per_lead if item is not None), None)
        if template is None:
            continue
        blank = np.full_like(template[0], np.nan)
        arrays[name] = np.stack([blank if item is None else item[0] for item in per_lead])
        if template[1] is not None:
            arrays["height_m"] = template[1]
    final = runs_dir / f"{run}.npz"
    partial = final.with_name(final.name + ".partial")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, allow_pickle=False, **arrays)
    partial.rename(final)
    return final


def dry_run(*, runs_by_day: dict[dt.date, list[str]], rectangle: CropRectangle | None) -> None:
    """List what would be fetched and print byte totals; reads no data."""
    grand_objects = 0
    grand_full = 0
    for day, runs in runs_by_day.items():
        objects = 0
        full_bytes = 0
        missing_total = 0
        for run in runs:
            available = list_keys(prefix=f"{PREFIX}{run}/")
            found, missing = wanted_keys(run=run, available=available)
            objects += len(found)
            missing_total += len(missing)
            full_bytes += sum(available[key] for key, _, _ in found)
        print(
            f"{day}: {len(runs)} runs, {objects} objects, {missing_total} expected objects absent, "
            f"{full_bytes / 1e9:.2f} GB if fetched whole"
        )
        grand_objects += objects
        grand_full += full_bytes
    wire = grand_objects * WIRE_BYTES_PER_OBJECT_ESTIMATE
    print(f"TOTAL: {grand_objects} objects; {grand_full / 1e9:.2f} GB whole-file equivalent.")
    print(f"Range-read estimate on the wire: about {wire / 1e9:.2f} GB (rough constant).")
    print("Crop rectangle computed." if rectangle else "Crop rectangle not computed.")
    print(f"Output folder (write-once): {PRODUCT_DIR}")


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the arguments, list the bucket, and either print the plan or fetch."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="list and print; fetch nothing")
    parser.add_argument("--days", nargs="+", type=dt.date.fromisoformat, help="UTC dates to fetch")
    parser.add_argument("--run-hours", nargs="+", type=int, default=list(range(24)))
    parser.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB)
    args = parser.parse_args(argv)

    all_runs = list_runs()
    days = args.days or list(choose_days(runs=all_runs))
    runs_by_day = {
        day: [r for r in all_runs if r[:8] == f"{day:%Y%m%d}" and int(r[9:11]) in args.run_hours]
        for day in days
    }
    for day, runs in runs_by_day.items():
        if not runs:
            message = f"no runs in the bucket for {day}"
            raise RuntimeError(message)
    first_run = next(iter(runs_by_day.values()))[0]
    first_key = f"{PREFIX}{first_run}/{first_run}-PT0000H00M-{SURFACE_FILES[0]}.nc"
    if args.dry_run:
        rectangle = load_or_make_rectangle(first_key=first_key, write=False)
        dry_run(runs_by_day=runs_by_day, rectangle=rectangle)
        return

    if shutil.disk_usage(NWP_DOWNLOADS_DIR.parent).free < args.min_free_gb * 1e9:
        message = f"less than {args.min_free_gb} GB free"
        raise RuntimeError(message)
    rectangle = load_or_make_rectangle(first_key=first_key, write=True)
    for runs in runs_by_day.values():
        for run in runs:
            if (PRODUCT_DIR / "runs" / f"{run}.npz").exists():
                print(f"{run}: already written, skipping")
                continue
            found, missing = wanted_keys(run=run, available=list_keys(prefix=f"{PREFIX}{run}/"))
            if missing:
                print(f"{run}: {len(missing)} expected objects absent from the listing")
            fetch_run(run=run, found=found, rectangle=rectangle)
            print(f"{run}: written")
    write_lineage_note(
        product_dir=PRODUCT_DIR,
        source_address=f"{BUCKET_URL}/{PREFIX}",
        request_description=(
            "Met Office UKV 2 km, leads 0 to 5, cropped at read time to the private trial-area box"
        ),
        variables=list(ALL_FILES),
        extra={"code_version": CODE_VERSION, "days": [str(day) for day in runs_by_day]},
    )
    write_readme(
        product_dir=PRODUCT_DIR,
        product_name="Met Office UKV 2 km from AWS (pilot)",
        source_web_page=REGISTRY_URL,
        script_path="studies/weather_downloads/fetch_ukv_aws_pilot.py",
        columns=dict.fromkeys(
            ALL_FILES,
            "One array per run file, (lead, [height,] row, column), NaN where not fetched",
        ),
        missing_value_convention="NaN where an object listed as absent was not fetched",
        gotchas=["Grid is a Lambert azimuthal equal-area grid, not CEDA's national grid."],
        external_docs={"AWS registry entry": REGISTRY_URL},
        lineage_filenames=["lineage.json"],
    )


if __name__ == "__main__":
    main()
