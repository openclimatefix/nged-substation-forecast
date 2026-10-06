r"""Pilot download of the Met Office UKV from its AWS open-data bucket, cropped at read time.

One-off throwaway script for the comparison of the UKV that CEDA archives against the UKV that the
Met Office serves live. The bucket `met-office-atmospheric-model-data` is public and anonymous, and
holds a rolling two-year window of the 2 km deterministic UKV
(<https://registry.opendata.aws/met-office-uk-deterministic/>).

**Layout.** Each hourly run has its own prefix `uk-deterministic-2km/<run>/`, and holds one NetCDF4
(HDF5) file per variable per valid time: `<valid>-PT<lead>H00M-<variable>.nc`. Each file covers the
whole 970 by 1042 grid on a Lambert azimuthal equal-area projection (not CEDA's national grid).
Variables on height levels hold 56 levels in one file.

**Cropping at read time.** The files are chunked 128 by 128 cells (one level per chunk) and
zlib-compressed, so an HTTP range read fetches only the chunks that overlap the private trial-area
box. The cropped rectangle (bounding rectangle of every cell inside the box) is saved in
`_crop_rectangle.json`. **That file and the `x` and `y` arrays in each run file reveal the private
box, so they stay in the private data folder, and no log line, README or lineage note carries a
coordinate or a cell count.** The bytes actually requested from the bucket are summed per object
and printed, and the run stops once `--max-wire-gb` is exceeded.

**What is fetched.** Screen temperature, 10 m wind speed and direction, the total, direct, and
diffuse downward short-wave, and wind speed and direction at the hub heights in `HUB_HEIGHTS_M`,
for leads 0 to 5 of every requested run, one UTC day at a time. The default range is the first
day with 24 runs to yesterday (UTC); `--start` and `--end` override it, and a range over `MAX_DAYS`
is refused. One compressed `.npz` per run holds each variable as `(lead, [height,] row, column)`,
decoded with the file's own `scale_factor`, `add_offset` and fill value, together with the file's
units, cell methods, time value and time bounds. Each run is written to a `.partial` file and
renamed. After a day's runs are all on disk, `ledger/<day>.json` is written the same way, and a
re-run skips every day that has a ledger entry and every run that has a file, so a restart costs
at most one day. The output folder (`UKV-AWS`) is write-once and never touches `UKV-AWS_pilot`.

**Concurrency.** `h5py` serialises every call under one global lock and the network read happens
inside it, so threads would give no concurrency. The script uses a process pool.

Run `--dry-run` first: it lists the bucket, prints the object count, the whole-file bytes and the
grid projection, and reads only the axes of one small file.

    uv run python studies/weather_downloads/fetch_ukv_aws_pilot.py --dry-run
"""

import argparse
import contextlib
import datetime as dt
import json
import shutil
import time
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
from xml.etree import ElementTree

import aiohttp
import fsspec
import h5py
import numpy as np
import requests
from lineage import write_lineage_note, write_readme
from pyproj import CRS, Transformer
from studies.sources import NWP_DOWNLOADS_DIR
from studies.trial_area import load_trial_area_box

CODE_VERSION: Final[str] = "fetch_ukv_aws_pilot-2"
BUCKET_URL: Final[str] = "https://met-office-atmospheric-model-data.s3-eu-west-2.amazonaws.com"
PREFIX: Final[str] = "uk-deterministic-2km/"
REGISTRY_URL: Final[str] = "https://registry.opendata.aws/met-office-uk-deterministic/"
PRODUCT_DIR: Final[Path] = NWP_DOWNLOADS_DIR / "UKV-AWS"
"""The write-once output folder. The two-day pilot lives in `UKV-AWS_pilot`, which this script never
writes."""

LEADS_HOURS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4, 5)
HUB_HEIGHTS_M: Final[tuple[float, ...]] = (50.0, 75.0, 100.0, 150.0)
"""The heights, in metres above ground, kept from the wind files on height levels.

The 2024 files hold 33 levels and the 2026 files 56, and 125 m is in the 2026 files only, so the
kept heights are the ones every file holds.
"""

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
MAX_DAYS: Final[int] = 800
"""The most days one invocation covers. The bucket holds 733 or so, so a typo is bounded."""
DEFAULT_MAX_WIRE_GB: Final[float] = 600.0
"""Stop once this many gigabytes have been requested in one invocation. The pilot measured 0.43 MB
per object, so the whole window needs about 360 GB."""
LISTING_THREADS: Final[int] = 4

WORKERS: Final[int] = 8
MAX_ATTEMPTS: Final[int] = 5
BLOCK_BYTES: Final[int] = 64 * 1024
MAX_BLOCKS: Final[int] = 64
DEFAULT_MIN_FREE_GB: Final[float] = 20.0
NO_BOUNDS: Final[int] = int(np.iinfo(np.int64).min)
"""Stored in a time-bounds array where the file has no bounds."""

_S3_NAMESPACE: Final[str] = "{http://s3.amazonaws.com/doc/2006-03-01/}"
_ATTRIBUTES_KEPT: Final[tuple[str, ...]] = (
    "units",
    "standard_name",
    "long_name",
    "cell_methods",
    "scale_factor",
    "add_offset",
    "_FillValue",
    "missing_value",
)


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


@dataclass(frozen=True)
class FieldRead:
    """One object's cropped field and what the file says about it.

    Attributes:
        values: Decoded `float32` values, `(row, column)` or `(height, row, column)`.
        heights_m: The kept heights of a height-level file, else `None`.
        attributes: The kept attributes (units, cell methods, packing) of the data variable.
        time_s: The file's `time` value, seconds since 1970-01-01 UTC.
        time_bounds_s: The file's time bounds in the same unit, or `None`.
        wire_bytes: Bytes requested from the bucket to read this object.
    """

    values: np.ndarray
    heights_m: np.ndarray | None
    attributes: dict[str, str]
    time_s: int
    time_bounds_s: tuple[int, int] | None
    wire_bytes: int


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


def _next_token(*, page: ElementTree.Element) -> str | None:
    """Return the continuation token, `None` on the last page.

    Raises:
        RuntimeError: If the page says it is truncated but gives no token, which would loop forever.
    """
    if page.findtext(f"{_S3_NAMESPACE}IsTruncated") != "true":
        return None
    token = page.findtext(f"{_S3_NAMESPACE}NextContinuationToken")
    if token is None:
        message = "listing is truncated but gives no continuation token"
        raise RuntimeError(message)
    return token


def list_keys(*, prefix: str, until: str | None = None) -> dict[str, int]:
    """List objects under `prefix` with their sizes in bytes.

    Args:
        prefix: The key prefix to list.
        until: If given, stop after the first page whose last key sorts after this key. S3 lists
            keys in order, so every key at or before `until` has been seen by then.
    """
    sizes: dict[str, int] = {}
    token: str | None = None
    while True:
        page = _list_page(prefix=prefix, delimiter=None, token=token)
        for item in page.iter(f"{_S3_NAMESPACE}Contents"):
            sizes[item.findtext(f"{_S3_NAMESPACE}Key", "")] = int(
                item.findtext(f"{_S3_NAMESPACE}Size", "0")
            )
        token = _next_token(page=page)
        if token is None or (until is not None and max(sizes, default="") > until):
            return sizes


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
        token = _next_token(page=page)
        if token is None:
            return sorted(runs)


def default_range(*, runs: Sequence[str], today: dt.date) -> tuple[dt.date, dt.date]:
    """Return the first day with all 24 runs, and yesterday (UTC), so that every day is complete."""
    per_day: dict[str, int] = defaultdict(int)
    for run in runs:
        per_day[run[:8]] += 1
    full_days = sorted(day for day, count in per_day.items() if count == RUNS_PER_DAY)
    if not full_days:
        message = "the bucket holds no full day"
        raise RuntimeError(message)
    first = dt.datetime.strptime(full_days[0], "%Y%m%d").date()  # noqa: DTZ007
    return first, today - dt.timedelta(days=1)


def day_range(*, start: dt.date, end: dt.date) -> list[dt.date]:
    """List every date from `start` to `end` inclusive.

    Raises:
        ValueError: If `end` is before `start`, or the range is longer than `MAX_DAYS`.
    """
    count = (end - start).days + 1
    if count < 1 or count > MAX_DAYS:
        message = f"{start} to {end} is {count} days; the limit is 1 to {MAX_DAYS}"
        raise ValueError(message)
    return [start + dt.timedelta(days=offset) for offset in range(count)]


def ledger_path(*, product_dir: Path, day: dt.date) -> Path:
    """The file whose existence says the day was fetched completely."""
    return product_dir / "ledger" / f"{day:%Y%m%d}.json"


def commit_day(*, product_dir: Path, day: dt.date, record: Mapping[str, Any]) -> None:
    """Record a finished day, via a `.partial` file and a rename so it is never half-written."""
    path = ledger_path(product_dir=product_dir, day=day)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(record, indent=2))
    partial.rename(path)


def listing_bound(*, run: str) -> str:
    """The largest key that can belong to a wanted object of `run`: its last lead's valid time."""
    last = dt.datetime.strptime(run, "%Y%m%dT%H%MZ").replace(tzinfo=dt.UTC) + dt.timedelta(
        hours=max(LEADS_HOURS)
    )
    return f"{PREFIX}{run}/{last:%Y%m%dT%H%MZ}-PT9999"


def wanted_keys(
    *, run: str, available: Mapping[str, int]
) -> tuple[list[tuple[str, int, str]], list[str]]:
    """Choose the objects of one run to fetch.

    Args:
        run: The run prefix, such as `20261002T0300Z`.
        available: Key to size for the run, from `list_keys`.

    Returns:
        The `(key, lead, file name)` triples that exist, and the `lead:file name` of those that do
        not.
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


def _text(value: Any) -> str:
    """Render an HDF5 attribute value as a plain string."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        return " ".join(_text(item) for item in value.ravel())
    return str(value)


def _attributes(*, variable: h5py.Dataset) -> dict[str, str]:
    """The kept attributes of a variable, as strings."""
    return {
        name: _text(variable.attrs[name]) for name in _ATTRIBUTES_KEPT if name in variable.attrs
    }


def decode_values(*, raw: np.ndarray, attributes: Mapping[str, str]) -> np.ndarray:
    """Apply CF unpacking and fill values to raw stored values, returning `float32`.

    Args:
        raw: The values as stored.
        attributes: The variable's attributes as strings, from `_attributes`.

    Returns:
        Values with `_FillValue` and `missing_value` set to NaN, then `raw * scale_factor +
        add_offset`.

    Raises:
        TypeError: If the stored values are neither floating point nor integer.
    """
    if raw.dtype.kind not in "fiu":
        message = f"unexpected stored dtype {raw.dtype}"
        raise TypeError(message)
    values = raw.astype(np.float64)
    for fill_name in ("_FillValue", "missing_value"):
        if fill_name in attributes:
            values[raw == float(attributes[fill_name])] = np.nan
    scale = float(attributes.get("scale_factor", 1.0))
    offset = float(attributes.get("add_offset", 0.0))
    return (values * scale + offset).astype(np.float32)


def _gridded_variable(*, dataset: h5py.File) -> h5py.Dataset:
    """Return the one gridded data variable in a file, ignoring axes and bounds."""
    candidates = [
        variable
        for name, variable in dataset.items()
        if isinstance(variable, h5py.Dataset)
        and variable.ndim >= 2
        and variable.shape[-2:] == GRID_SHAPE
        and not name.endswith("_bnds")
    ]
    if len(candidates) != 1:
        message = f"expected one gridded variable, found {[c.name for c in candidates]}"
        raise ValueError(message)
    return candidates[0]


@contextlib.contextmanager
def _open(*, key: str) -> Iterator[tuple[h5py.File, Any]]:
    """Open one remote file for cached range reads; yield the HDF5 file and the HTTP file."""
    with (
        fsspec.open(
            f"{BUCKET_URL}/{key}",
            "rb",
            block_size=BLOCK_BYTES,
            cache_type="blockcache",
            cache_options={"maxblocks": MAX_BLOCKS},
        ) as handle,
        h5py.File(handle, "r") as dataset,
    ):
        yield dataset, handle


def _check_axes(*, dataset: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    """Read both axes, and check their lengths and that each is strictly monotonic.

    Raises:
        ValueError: If an axis has the wrong length or is not strictly monotonic.
    """
    x = np.asarray(dataset["projection_x_coordinate"][:], dtype=np.float64)
    y = np.asarray(dataset["projection_y_coordinate"][:], dtype=np.float64)
    if (len(y), len(x)) != GRID_SHAPE:
        message = f"unexpected axis lengths {(len(y), len(x))}"
        raise ValueError(message)
    for axis in (x, y):
        steps = np.diff(axis)
        if not (np.all(steps > 0) or np.all(steps < 0)):
            message = "a grid axis is not strictly monotonic"
            raise ValueError(message)
    return x, y


def compute_crop_rectangle(
    *, x: np.ndarray, y: np.ndarray, crs: CRS, box_bounds: tuple[float, float, float, float]
) -> CropRectangle:
    """Find the cells of a projected grid that lie inside a latitude-longitude box.

    Args:
        x: The `projection_x_coordinate` axis in metres.
        y: The `projection_y_coordinate` axis in metres.
        crs: The grid's projection.
        box_bounds: `(lat_min, lat_max, lon_min, lon_max)` in degrees.

    Returns:
        The bounding rectangle of the cells inside the box, in grid indices.

    Raises:
        RuntimeError: If no cell lies inside the box.
    """
    lat_min, lat_max, lon_min, lon_max = box_bounds
    xs, ys = np.meshgrid(x, y)
    longitude, latitude = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(xs, ys)
    inside = (
        (latitude >= lat_min)
        & (latitude <= lat_max)
        & (longitude >= lon_min)
        & (longitude <= lon_max)
    )
    rows = np.flatnonzero(inside.any(axis=1))
    cols = np.flatnonzero(inside.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        message = "no grid cell lies inside the trial-area box"
        raise RuntimeError(message)
    return CropRectangle(int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1)


def _plain(value: Any) -> Any:
    """Turn an HDF5 attribute value into a plain Python value, which `CRS.from_cf` needs."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray | np.generic):
        return value.item() if value.size == 1 else value.tolist()
    return value


def _grid_crs(*, dataset: h5py.File) -> CRS:
    """The grid's projection from the file's own CF grid-mapping attributes."""
    attributes = {
        name: _plain(value) for name, value in dataset["lambert_azimuthal_equal_area"].attrs.items()
    }
    return CRS.from_cf(attributes)


def load_or_make_rectangle(*, first_key: str, write: bool) -> tuple[CropRectangle, str]:
    """Read the saved rectangle, or compute it from one file's axes and save it if `write`.

    Returns:
        The rectangle, and the grid's PROJ string (public: the whole-domain projection).
    """
    with _open(key=first_key) as (dataset, _):
        crs = _grid_crs(dataset=dataset)
        path = PRODUCT_DIR / "_crop_rectangle.json"
        if path.exists():
            return CropRectangle(**json.loads(path.read_text())), crs.to_proj4()
        x, y = _check_axes(dataset=dataset)
    box = load_trial_area_box()
    rectangle = compute_crop_rectangle(
        x=x, y=y, crs=crs, box_bounds=(box.lat_min, box.lat_max, box.lon_min, box.lon_max)
    )
    if write:
        PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rectangle.__dict__))
    return rectangle, crs.to_proj4()


def _check_times(*, dataset: h5py.File, run: str, lead: int) -> int:
    """Check that the file's own reference time, period, and time match its name.

    Returns:
        The file's `time` in seconds since 1970-01-01 UTC.

    Raises:
        ValueError: If a unit is not seconds, or a time differs from the name's.
    """
    for name, prefix in (
        ("forecast_reference_time", "seconds since 1970-01-01"),
        ("time", "seconds since 1970-01-01"),
        ("forecast_period", "seconds"),
    ):
        if not _text(dataset[name].attrs["units"]).startswith(prefix):
            message = f"{run}: {name} is not in '{prefix}'"
            raise ValueError(message)
    run_s = int(dt.datetime.strptime(run, "%Y%m%dT%H%MZ").replace(tzinfo=dt.UTC).timestamp())
    reference = int(dataset["forecast_reference_time"][()])
    period = int(dataset["forecast_period"][()])
    valid = int(dataset["time"][()])
    if (reference, period, valid) != (run_s, lead * 3600, run_s + lead * 3600):
        message = (
            f"{run} lead {lead}: file says reference {reference}, period {period}, time {valid}"
        )
        raise ValueError(message)
    return valid


def _time_bounds(*, dataset: h5py.File) -> tuple[int, int] | None:
    """The file's time bounds, if it has any."""
    for name in ("time_bnds", "time_bounds"):
        if name in dataset:
            low, high = (int(value) for value in np.asarray(dataset[name][()]).ravel()[:2])
            return low, high
    return None


def _read_once(
    *,
    key: str,
    run: str,
    lead: int,
    rectangle: CropRectangle,
    expected_axes: tuple[np.ndarray, np.ndarray],
) -> FieldRead:
    """Range-read the cropped, decoded field of one object, once."""
    with _open(key=key) as (dataset, handle):
        x, y = _check_axes(dataset=dataset)
        cropped_x = x[rectangle.col_start : rectangle.col_stop]
        cropped_y = y[rectangle.row_start : rectangle.row_stop]
        if not (
            np.array_equal(cropped_x, expected_axes[0])
            and np.array_equal(cropped_y, expected_axes[1])
        ):
            message = f"{key}: grid axes differ from the first file's"
            raise ValueError(message)
        valid = _check_times(dataset=dataset, run=run, lead=lead)
        variable = _gridded_variable(dataset=dataset)
        attributes = _attributes(variable=variable)
        rows = slice(rectangle.row_start, rectangle.row_stop)
        cols = slice(rectangle.col_start, rectangle.col_stop)
        heights: np.ndarray | None = None
        if variable.ndim == 2:
            raw = variable[rows, cols]
        else:
            all_heights = np.asarray(dataset["height"][:], dtype=np.float64)
            keep = [int(np.flatnonzero(all_heights == h)[0]) for h in HUB_HEIGHTS_M]
            raw = np.stack([variable[i, rows, cols] for i in keep])
            heights = all_heights[keep]
        bounds = _time_bounds(dataset=dataset)
        wire_bytes = int(handle.cache.total_requested_bytes)
    return FieldRead(
        values=decode_values(raw=np.asarray(raw), attributes=attributes),
        heights_m=heights,
        attributes=attributes,
        time_s=valid,
        time_bounds_s=bounds,
        wire_bytes=wire_bytes,
    )


def read_object(
    *,
    key: str,
    run: str,
    lead: int,
    rectangle: CropRectangle,
    expected_axes: tuple[np.ndarray, np.ndarray],
) -> FieldRead:
    """Read one object, retrying transient network faults with backoff.

    A listed object that has vanished (`FileNotFoundError`) and a wrong file (`ValueError`) are
    faults, so they are never retried. A 5xx at open time, such as S3's `SlowDown`, arrives as an
    `aiohttp.ClientError` and is retried.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return _read_once(
                key=key, run=run, lead=lead, rectangle=rectangle, expected_axes=expected_axes
            )
        except FileNotFoundError, ValueError:
            raise
        except OSError, aiohttp.ClientError, TimeoutError:
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(5 * 2 ** (attempt - 1))
    message = "unreachable"
    raise AssertionError(message)


def _read_object_task(arguments: dict[str, Any]) -> FieldRead:
    """Process-pool entry point: unpack keyword arguments for `read_object`."""
    return read_object(**arguments)


def assemble_run_arrays(
    *,
    results: Mapping[tuple[str, int], FieldRead],
    missing: Sequence[str],
    axes: tuple[np.ndarray, np.ndarray],
) -> dict[str, np.ndarray]:
    """Stack one run's reads into the arrays of its `.npz`, with the file metadata beside each."""
    wire_bytes = sum(item.wire_bytes for item in results.values())
    arrays: dict[str, np.ndarray] = {
        "x": axes[0],
        "y": axes[1],
        "lead_hours": np.array(LEADS_HOURS),
        "missing": np.array(json.dumps(list(missing))),
        "wire_bytes": np.array(wire_bytes),
    }
    no_bounds = (NO_BOUNDS, NO_BOUNDS)
    for name in ALL_FILES:
        per_lead = [results.get((name, lead)) for lead in LEADS_HOURS]
        template = next((item for item in per_lead if item is not None), None)
        if template is None:
            continue
        blank = np.full_like(template.values, np.nan)
        arrays[name] = np.stack([blank if item is None else item.values for item in per_lead])
        arrays[f"{name}__time_s"] = np.array(
            [NO_BOUNDS if item is None else item.time_s for item in per_lead], dtype=np.int64
        )
        arrays[f"{name}__time_bounds_s"] = np.array(
            [
                no_bounds if item is None or item.time_bounds_s is None else item.time_bounds_s
                for item in per_lead
            ],
            dtype=np.int64,
        )
        arrays[f"{name}__attributes"] = np.array(json.dumps(template.attributes))
        if template.heights_m is not None:
            arrays["height_m"] = template.heights_m
    return arrays


def fetch_run(
    *,
    run: str,
    found: list[tuple[str, int, str]],
    missing: list[str],
    rectangle: CropRectangle,
    pool: ProcessPoolExecutor,
) -> tuple[Path, int]:
    """Fetch one run's cropped fields and write them atomically to `runs/<run>.npz`.

    Returns:
        The path written and the bytes requested from the bucket for this run.
    """
    runs_dir = PRODUCT_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    first = next(key for key, lead, _ in found if lead == 0)
    with _open(key=first) as (dataset, _):
        x, y = _check_axes(dataset=dataset)
    axes = (
        x[rectangle.col_start : rectangle.col_stop],
        y[rectangle.row_start : rectangle.row_stop],
    )
    futures = {
        (name, lead): pool.submit(
            _read_object_task,
            {"key": key, "run": run, "lead": lead, "rectangle": rectangle, "expected_axes": axes},
        )
        for key, lead, name in found
    }
    results = {item: future.result() for item, future in futures.items()}
    arrays = assemble_run_arrays(results=results, missing=missing, axes=axes)
    final = runs_dir / f"{run}.npz"
    partial = final.with_name(final.name + ".partial")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, allow_pickle=False, **arrays)
    partial.rename(final)
    return final, int(arrays["wire_bytes"])


RunPlan = tuple[list[tuple[str, int, str]], list[str], int]
"""For one run: the objects to fetch, the `lead:file` names absent from the listing, and the
whole-file bytes of the objects to fetch."""


def plan_run(run: str) -> RunPlan:
    """List one run's first hours and choose its objects."""
    available = list_keys(prefix=f"{PREFIX}{run}/", until=listing_bound(run=run))
    found, missing = wanted_keys(run=run, available=available)
    return found, missing, sum(available[key] for key, _, _ in found)


def plan_day(*, day_runs: Sequence[str]) -> dict[str, RunPlan]:
    """Plan every run of one day, listing the runs concurrently."""
    with ThreadPoolExecutor(max_workers=LISTING_THREADS) as listing_pool:
        return dict(zip(day_runs, listing_pool.map(plan_run, day_runs), strict=True))


def _summarise_folder() -> dict[str, Any]:
    """Describe the run files in the folder: count, span, variable attributes, absences, bytes."""
    runs: list[str] = []
    attributes: dict[str, dict[str, str]] = {}
    missing: dict[str, list[str]] = {}
    wire_bytes = 0
    for path in sorted((PRODUCT_DIR / "runs").glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            runs.append(path.stem)
            wire_bytes += int(archive["wire_bytes"])
            run_missing = json.loads(str(archive["missing"]))
            if run_missing:
                missing[path.stem] = run_missing
            if not attributes:
                for key in archive.files:
                    if key.endswith("__attributes"):
                        attributes[key.removesuffix("__attributes")] = json.loads(str(archive[key]))
    return {"runs": runs, "attributes": attributes, "missing": missing, "wire_bytes": wire_bytes}


GOTCHAS: Final[tuple[str, ...]] = (
    (
        "The grid is a Lambert azimuthal equal-area grid (GRS80, latitude of origin 54.9, "
        "longitude of origin -2.5), not CEDA's national grid."
    ),
    (
        "Row 0 is the southernmost row: `projection_y_coordinate` increases with the row "
        "index. CEDA's rows run north to south."
    ),
    (
        "Every field is an instantaneous value at its valid time, shortwave included. No file"
        " carries `cell_methods` or time bounds for any kept variable, so that rests on the "
        "absence of bounds."
    ),
    (
        "The wind files on height levels hold 33 levels in 2024 and 56 in 2026, and 125 m "
        "exists only in 2026. The kept heights, 50, 75, 100, and 150 m, are in every file "
        "checked."
    ),
    (
        "In the two pilot days, total shortwave equalled direct plus diffuse to rounding on "
        "2026-10-05 but not on 2024-10-08, where the daytime mean absolute residual was 8 to "
        "42 W m-2 per run hour. Check the residual month by month before relying on the three"
        " components in 2024."
    ),
    (
        "The files hold no `scale_factor`, `add_offset` or fill value in the pilot days; the "
        "script decodes them if present."
    ),
    ("The bucket is a rolling two-year window, so an early day can no longer be fetched later."),
)


def write_notes(*, summary: Mapping[str, Any]) -> None:
    """Write the lineage note and README from what the run files hold."""
    runs = summary["runs"]
    write_lineage_note(
        product_dir=PRODUCT_DIR,
        source_address=f"{BUCKET_URL}/{PREFIX}",
        request_description=(
            "Met Office UKV 2 km, leads 0 to 5, cropped at read time to the private trial-area box"
        ),
        variables=list(ALL_FILES),
        extra={
            "code_version": CODE_VERSION,
            "run_files": len(runs),
            "first_run": runs[0] if runs else None,
            "last_run": runs[-1] if runs else None,
            "attributes_per_variable": summary["attributes"],
            "missing_objects_per_run": summary["missing"],
            "wire_bytes_requested": summary["wire_bytes"],
        },
    )
    columns = {
        name: "(lead, [height,] row, column) float32, decoded; "
        + ", ".join(f"{key}={value}" for key, value in summary["attributes"].get(name, {}).items())
        for name in ALL_FILES
    }
    columns |= {
        "x, y": "Cropped grid axes in metres on the Lambert azimuthal equal-area grid (private)",
        "lead_hours": "Forecast lead in hours of axis 0 of every field",
        "height_m": "Height above ground in metres of axis 1 of the height-level fields",
        "<variable>__time_s": "The file's valid time per lead, seconds since 1970-01-01 UTC",
        "<variable>__time_bounds_s": "Time bounds per lead, same unit; int64 minimum if none",
        "<variable>__attributes": "JSON of the file's units, cell methods and packing attributes",
        "missing": "JSON list of 'lead:variable' objects absent from the bucket listing",
        "wire_bytes": "Bytes requested from the bucket to build this run's file",
    }
    write_readme(
        product_dir=PRODUCT_DIR,
        product_name="Met Office UKV 2 km from AWS",
        source_web_page=REGISTRY_URL,
        script_path="studies/weather_downloads/fetch_ukv_aws_pilot.py",
        columns=columns,
        missing_value_convention=(
            "NaN where the file holds a fill value or where an object was absent from the listing "
            "(the run file's `missing` entry names it); a variable absent at every lead is not kept"
        ),
        gotchas=list(GOTCHAS),
        external_docs={"AWS registry entry": REGISTRY_URL},
        lineage_filenames=["lineage.json"],
    )


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", maxsplit=1)[0])
    parser.add_argument("--dry-run", action="store_true", help="list and print; fetch nothing")
    parser.add_argument("--start", type=dt.date.fromisoformat, help="first UTC day to fetch")
    parser.add_argument("--end", type=dt.date.fromisoformat, help="last UTC day to fetch")
    parser.add_argument("--run-hours", nargs="+", type=int, default=list(range(24)))
    parser.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB)
    parser.add_argument("--max-wire-gb", type=float, default=DEFAULT_MAX_WIRE_GB)
    return parser.parse_args(argv)


def _fetch_day(
    *,
    day: dt.date,
    plan: Mapping[str, RunPlan],
    rectangle: CropRectangle,
    pool: ProcessPoolExecutor,
    max_wire_bytes: float,
    wire_total: int,
) -> int:
    """Fetch every run of one day that is not on disk, then commit the day to the ledger.

    Returns:
        The bytes requested for this day.
    """
    day_bytes = 0
    for run, (found, missing, _) in plan.items():
        if (PRODUCT_DIR / "runs" / f"{run}.npz").exists():
            continue
        if wire_total + day_bytes > max_wire_bytes:
            message = f"stopping before {run}: over --max-wire-gb; a re-run resumes here"
            raise SystemExit(message)
        _, run_bytes = fetch_run(
            run=run, found=found, missing=missing, rectangle=rectangle, pool=pool
        )
        day_bytes += run_bytes
    commit_day(
        product_dir=PRODUCT_DIR,
        day=day,
        record={
            "day": str(day),
            "runs": len(plan),
            "objects": sum(len(found) for found, _, _ in plan.values()),
            "absent_objects": sum(len(missing) for _, missing, _ in plan.values()),
            "wire_bytes_this_invocation": day_bytes,
            "code_version": CODE_VERSION,
            "committed_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        },
    )
    return day_bytes


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the arguments, then print the plan or fetch the days that the ledger lacks."""
    args = _parse_arguments(argv)
    all_runs = list_runs()
    first, last = default_range(runs=all_runs, today=dt.datetime.now(dt.UTC).date())
    days = day_range(start=args.start or first, end=args.end or last)
    runs_by_day: dict[dt.date, list[str]] = defaultdict(list)
    for run in all_runs:
        day = dt.datetime.strptime(run[:8], "%Y%m%d").date()  # noqa: DTZ007
        if day in days and int(run[9:11]) in args.run_hours:
            runs_by_day[day].append(run)
    if not runs_by_day:
        message = "no runs in the bucket for the requested days and hours"
        raise SystemExit(message)
    first_run = runs_by_day[min(runs_by_day)][0]
    first_key = f"{PREFIX}{first_run}/{first_run}-PT0000H00M-{SURFACE_FILES[0]}.nc"
    rectangle, proj4 = load_or_make_rectangle(first_key=first_key, write=not args.dry_run)
    print(f"Grid projection: {proj4}")
    print(f"Days {days[0]} to {days[-1]}: {len(runs_by_day)} with runs; output {PRODUCT_DIR}")
    if args.dry_run:
        objects = whole_bytes = absent = 0
        for day in sorted(runs_by_day):
            plan = plan_day(day_runs=runs_by_day[day])
            objects += sum(len(found) for found, _, _ in plan.values())
            absent += sum(len(missing) for _, missing, _ in plan.values())
            whole_bytes += sum(size for _, _, size in plan.values())
        print(f"{objects} objects, {absent} absent, {whole_bytes / 1e9:.1f} GB if fetched whole")
        return

    if shutil.disk_usage(NWP_DOWNLOADS_DIR.parent).free < args.min_free_gb * 1e9:
        message = f"less than {args.min_free_gb} GB free"
        raise SystemExit(message)
    wire_total = 0
    started = time.monotonic()
    try:
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            for number, day in enumerate(sorted(runs_by_day), start=1):
                if ledger_path(product_dir=PRODUCT_DIR, day=day).exists():
                    print(f"{day}: committed already, skipping", flush=True)
                    continue
                plan = plan_day(day_runs=runs_by_day[day])
                day_bytes = _fetch_day(
                    day=day,
                    plan=plan,
                    rectangle=rectangle,
                    pool=pool,
                    max_wire_bytes=args.max_wire_gb * 1e9,
                    wire_total=wire_total,
                )
                wire_total += day_bytes
                print(
                    f"{day}: day {number}/{len(runs_by_day)}, {len(plan)} runs, "
                    f"{day_bytes / 1e6:.0f} MB, {wire_total / 1e9:.2f} GB in total, "
                    f"{(time.monotonic() - started) / 60:.1f} min",
                    flush=True,
                )
    finally:
        write_notes(summary=_summarise_folder())


if __name__ == "__main__":
    main()
