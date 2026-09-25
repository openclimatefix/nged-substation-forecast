"""Download WeatherNext 3 ensemble-mean runs into an Icechunk store of a wide United Kingdom crop.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/934>, the WeatherNext 3 arm of
the matched-lead comparison. The source is the Requester Pays bucket
`weathernext3_statistics_spatial` in region US-EAST1, which holds one Zarr store per run, at 0.1
degrees over the globe, with the ensemble mean of each variable. The archive starts on 2026-01-01
and has a run every hour. The runs at 00, 06, 12, and 18 UTC have 360 hourly lead times, and the
other hours only 48. This script fetches only the runs at 00, 06, 12, and 18 UTC.

**The Zarr chunks are whole-globe, so cropping to the box saves no bytes.** Each chunk holds one
variable at one lead time over the whole globe, about 20 MB compressed. The script therefore
transfers about 50 GB per run (360 lead times, 7 variables) and keeps about 15 MB of it. The box
(`LAT_MIN` to `LAT_MAX` degrees north, `LON_MIN` to `LON_MAX` degrees east) is wide enough to reach
offshore wind farms, and is not private.

**The script must run on a Compute Engine machine in us-east1.** Reads inside the same region cost
nothing, while reads from anywhere else are billed as internet egress at $0.12 per GB. One run is
about 50 GB of reads, which costs about £4.50 ($6) from outside Google Cloud. The four 360-hour runs
of one day are about 200 GB, and the 267 days in the archive so far would be about 53 TB. The script
detects whether it is on a Compute Engine machine in us-east1 through the metadata server. Outside
us-east1 it refuses to start when the estimated transfer exceeds `--max-external-gb` (default 5.0),
which is less than one run. `--dry-run` prints the number of runs and the estimated transfer, and
touches neither the network nor the store.

**Credentials and the billing project come from the environment.** `GOOGLE_CLOUD_PROJECT` names the
project that pays for the reads, and the script exits with a message if that variable is unset.
Authentication uses Google application default credentials: `GOOGLE_APPLICATION_CREDENTIALS` on a
workstation, or the machine's own service account on Compute Engine. The script never prints or
writes the project name, the bucket name, the credentials, or an account name.

**The output is one Icechunk repository with a `main` branch and a `staging` branch.** `--bucket`
names a Cloud Storage bucket (in us-east1) that holds the repository under `STORE_PREFIX`, and
`--local-store` names a directory instead, for tests. Exactly one is required. The store has seven
`Float32` arrays, one per variable, with dimensions `(init_time, lead_time, latitude, longitude)`
in the source's native units, one shard per run, and values rounded to `KEEP_BITS` significand
bits. `init_time` is preallocated with every 00, 06, 12, and 18 UTC run from `--start-date` to
`--end-date`, given when the repository is created and never changed, so runs can be written in
any order. A later invocation clips its request to the stored range and prints how many requested
runs fall outside it. Two more arrays record which runs are present: `run_written` and
`source_init_time`. The group attributes hold the lineage and the runs skipped.

**Each run is one commit on `staging`, and the script never moves `main`.** A killed process leaves
at most an uncommitted session, so `staging` never holds a partial run. `validate_weathernext3.py
--publish` moves `main` to the validated `staging` snapshot. Re-running the script skips every run
whose `run_written` flag is set.

**A failed run does not stop the job.** Each (variable, lead time) chunk is read by one task in a
pool of `--workers` threads (default 16), retried with exponential backoff on any error. A missing
chunk does not raise in Zarr version 3 (it reads as the fill value, NaN), so a slice that is
entirely NaN is treated as a failed read. A run whose `success` marker object is absent, and a
wanted run the bucket does not list, are skipped and recorded in the group attributes rather than
raised. The script records a failed run's label, continues with the other runs, prints the failed
labels at the end, and exits non-zero. An uncaught exception prints only the run label and the
exception's type name, because an exception message can carry an account name, the billing project,
or the bucket name.

Run it on the machine with `GOOGLE_CLOUD_PROJECT=<project> uv run python
studies/weather_downloads/fetch_weathernext3.py --bucket <bucket> --start-date 2026-01-01
--end-date 2026-09-24`. Then check the output with `validate_weathernext3.py`.
"""

import argparse
import os
import sys
import time
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

# The Rust core of Icechunk reads this variable when it is imported, so it must be set first.
os.environ.setdefault("ICECHUNK_LOG", "error")

import gcsfs
import icechunk
import numpy as np
import polars as pl
import xarray as xr
import zarr
from delta_store.precision import round_to_significand_bits
from zarr.codecs import BloscCodec
from zarr.core.common import JSON

BUCKET_PREFIX: Final[str] = (
    "weathernext3_statistics_spatial/weathernext_3_0_0_statistics/zarr/2026_to_present"
)
"""Where the run stores live, without the `gs://` scheme, as `gcsfs` expects."""

STORE_PREFIX: Final[str] = "weathernext3_statistics_uk"
"""The directory of the Icechunk repository inside the output bucket."""

MAIN_BRANCH: Final[str] = "main"
"""The branch colleagues read. Only `validate_weathernext3.py --publish` moves it."""

STAGING_BRANCH: Final[str] = "staging"
"""The branch this script commits to, one commit per run."""

ARCHIVE_START: Final[date] = date(2026, 1, 1)

LAT_MIN: Final[float] = 49.0
LAT_MAX: Final[float] = 61.5
LON_MIN: Final[float] = -10.0
LON_MAX: Final[float] = 3.5
"""The crop, in degrees north and degrees east (signed, west negative), inclusive at both ends.
The box is wide enough to reach offshore wind farms."""

GRID_STEPS_PER_DEGREE: Final[int] = 10
"""The source grid is 0.1 degrees."""

LATITUDES: Final[np.ndarray] = (
    np.arange(round(LAT_MIN * GRID_STEPS_PER_DEGREE), round(LAT_MAX * GRID_STEPS_PER_DEGREE) + 1)
    / GRID_STEPS_PER_DEGREE
)
"""The stored latitudes, ascending, rounded to 0.1 degrees."""

LONGITUDES: Final[np.ndarray] = (
    np.arange(round(LON_MIN * GRID_STEPS_PER_DEGREE), round(LON_MAX * GRID_STEPS_PER_DEGREE) + 1)
    / GRID_STEPS_PER_DEGREE
)
"""The stored longitudes in signed degrees, ascending, rounded to 0.1 degrees."""

# The source axes are float32, which is off by up to about 2e-5 degrees near 350; a tenth of a cell
# is far tighter than the 0.1 degree spacing and still far looser than that float32 error.
AXIS_TOLERANCE: Final[float] = 1e-2
"""The largest difference, in degrees, between a source coordinate and the stored value."""

VARIABLES: Final[tuple[str, ...]] = (
    "temperature_2m_mean",
    "u_component_of_wind_10m_mean",
    "v_component_of_wind_10m_mean",
    "u_component_of_wind_100m_mean",
    "v_component_of_wind_100m_mean",
    "surface_solar_radiation_downwards_1hr_mean",
    "total_sky_direct_solar_radiation_at_surface_1hr_mean",
)
"""The ensemble means the matched-lead comparison needs: temperature, winds, and total and direct
radiation. Radiation is in J/m2, accumulated over the hour ending at the valid time."""

UNITS: Final[dict[str, str]] = {
    "temperature_2m_mean": "K",
    "u_component_of_wind_10m_mean": "m s-1",
    "v_component_of_wind_10m_mean": "m s-1",
    "u_component_of_wind_100m_mean": "m s-1",
    "v_component_of_wind_100m_mean": "m s-1",
    "surface_solar_radiation_downwards_1hr_mean": "J m-2",
    "total_sky_direct_solar_radiation_at_surface_1hr_mean": "J m-2",
}
"""Each variable's unit in the source store, which the output keeps unchanged."""

RADIATION_CONVENTION: Final[str] = (
    "J m-2 accumulated over the hour ending at the valid time (init_time + lead_time); divide by "
    "3600 for the mean W m-2 over that hour"
)

LATITUDE_DIM: Final[str] = "lat_0p1"
LONGITUDE_DIM: Final[str] = "lon_0p1"
"""The names of the source store's latitude and longitude dimensions."""

INIT_TIME: Final[str] = "init_time"
LEAD_TIME: Final[str] = "lead_time"
LATITUDE: Final[str] = "latitude"
LONGITUDE: Final[str] = "longitude"
RUN_WRITTEN: Final[str] = "run_written"
SOURCE_INIT_TIME: Final[str] = "source_init_time"
"""Array names in the output. The last two hold one value per `init_time` slot."""

DIMENSIONS: Final[tuple[str, ...]] = (INIT_TIME, LEAD_TIME, LATITUDE, LONGITUDE)

HOURS_SINCE_EPOCH_UNITS: Final[str] = "hours since 1970-01-01T00:00:00"
"""The CF unit of the stored `init_time` and `source_init_time` arrays, which are `int64`, so that
`xarray` decodes them to datetimes."""

SKIPPED_ATTRIBUTE: Final[str] = "runs_skipped_missing_success_marker"
NOT_PUBLISHED_ATTRIBUTE: Final[str] = "runs_skipped_not_published"
FIRST_DAY_ATTRIBUTE: Final[str] = "first_init_day"
LAST_DAY_ATTRIBUTE: Final[str] = "last_init_day"

LEADS: Final[int] = 360
"""Lead times stored: every hour from 1 to `LEADS`."""

SHARD_SHAPE: Final[tuple[int, int, int, int]] = (1, LEADS, 128, 136)
"""One run, all leads and the whole box: one object in storage per run and variable."""

CHUNK_SHAPE: Final[tuple[int, int, int, int]] = (1, LEADS, 8, 8)
"""Inner chunk: a point read fetches about 100 KB."""

DEFAULT_WORKERS: Final[int] = 16
"""Concurrent chunk reads within one run. Each read is a network wait for about 20 MB."""

KEEP_BITS: Final[int] = 13
"""Significand bits kept in every value, as in production NWP storage."""

MAX_ATTEMPTS: Final[int] = 5
"""Attempts at one chunk read before the error is raised."""

BACKOFF_SECONDS: Final[float] = 5.0
"""Base of the exponential backoff: the first retry sleeps twice this (10 s), the next four
times."""

NON_RETRYABLE_ERRORS: Final[tuple[type[Exception], ...]] = (
    AssertionError,
    KeyError,
    FileNotFoundError,
    ValueError,
)
"""Errors that repeating the call cannot fix: a failed store assertion, a missing key or file, and
this module's own `ValueError`s (an all-NaN slice, a mismatched grid)."""

INTERRUPTED_EXIT_CODE: Final[int] = 130
"""The shell's conventional exit code for a process ended by Ctrl-C."""

INIT_HOURS: Final[tuple[int, ...]] = (0, 6, 12, 18)
"""Init hours (UTC) of the runs with `LEADS` lead times, the only runs this script fetches."""

RUNS_PER_DAY: Final[int] = len(INIT_HOURS)
HOURS_PER_SLOT: Final[int] = 6

GB_PER_CHUNK: Final[float] = 0.0197
"""Measured compressed size of one (variable, lead time) chunk, GB."""

DEFAULT_MAX_EXTERNAL_GB: Final[float] = 5.0
"""Largest estimated transfer, GB, the script accepts outside us-east1: less than one run."""

EGRESS_USD_PER_GB: Final[float] = 0.12
"""Internet egress price, dollars per GB, for reads from outside the bucket's region."""

_METADATA_ZONE_URL: Final[str] = "http://metadata.google.internal/computeMetadata/v1/instance/zone"
_METADATA_TIMEOUT_SECONDS: Final[float] = 2.0
_SECONDS_PER_HOUR: Final[int] = 3600


def _describe(*, error: BaseException) -> str:
    """Return what may be printed about `error`: its type name, plus the message of an assertion.

    The message of an `AssertionError` is a fixed string written in this module, with no data
    value, coordinate, or count. The message of any other exception can carry an account name, the
    billing project, or the bucket name.
    """
    if isinstance(error, AssertionError):
        return f"AssertionError: {error}"
    return type(error).__name__


def _estimated_gb() -> float:
    """Return the estimated GB transferred to read every variable at every lead of one run."""
    return LEADS * len(VARIABLES) * GB_PER_CHUNK


def _in_us_east1() -> bool:
    """Return whether this machine is a Compute Engine instance in a us-east1 zone.

    Any failure to reach the metadata server, including running on a workstation, means no.
    """
    request = urllib.request.Request(_METADATA_ZONE_URL, headers={"Metadata-Flavor": "Google"})
    try:
        with urllib.request.urlopen(request, timeout=_METADATA_TIMEOUT_SECONDS) as response:
            zone = response.read().decode()
    except Exception:  # noqa: BLE001  # any failure means this is not a Compute Engine machine
        return False
    return zone.rsplit("/", maxsplit=1)[-1].startswith("us-east1-")


def _run_name(*, day: date, hour: int) -> str:
    """Return the store directory name of the run initialised at `hour` UTC on `day`."""
    return f"{day:%Y%m%d}_{hour:02d}hr_01_preds"


def _parse_run_name(*, name: str) -> tuple[date, int] | None:
    """Return the (day, hour) of a store directory name, or `None` if it is not a run name."""
    parts = name.split("_")
    if len(parts) != 4 or parts[2:] != ["01", "preds"] or not parts[1].endswith("hr"):
        return None
    try:
        return date(int(parts[0][:4]), int(parts[0][4:6]), int(parts[0][6:])), int(parts[1][:-2])
    except ValueError:
        return None


def _filesystem() -> gcsfs.GCSFileSystem:
    """Return a Requester Pays Cloud Storage filesystem, billed to `GOOGLE_CLOUD_PROJECT`.

    Returns:
        A filesystem whose reads are billed to the project named by `GOOGLE_CLOUD_PROJECT`.

    Raises:
        SystemExit: If `GOOGLE_CLOUD_PROJECT` is unset.
    """
    billing_project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not billing_project:
        sys.exit("GOOGLE_CLOUD_PROJECT is not set: it names the project billed for the reads.")
    return gcsfs.GCSFileSystem(
        token="google_default", requester_pays=billing_project, project=billing_project
    )


def _with_retries[T](*, action: Callable[[], T], label: str, what: str) -> T:
    """Call `action`, retrying with exponential backoff on any failure.

    A transient network error partway through a run would otherwise abandon 50 GB of reads. An error
    in `NON_RETRYABLE_ERRORS` is deterministic, so it is raised at once. The first sleep is 10 s and
    each later one doubles. Each retry prints the run label, what was being done, the attempt
    number, and the exception's type name only, because an exception message can carry a request
    URL, an account name, or the billing project.

    Args:
        action: The call to make.
        label: The run's `YYYYMMDD_HH` label, or `bucket`, for the retry log line.
        what: Short description of the call, for the retry log line.

    Returns:
        Whatever `action` returns.

    Raises:
        Exception: The last error, after `MAX_ATTEMPTS` failed attempts, or at once for an error in
            `NON_RETRYABLE_ERRORS`.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return action()
        except NON_RETRYABLE_ERRORS:
            raise
        except Exception as error:
            if attempt == MAX_ATTEMPTS:
                raise
            error_name = type(error).__name__
        # The backoff sleeps outside the `except` block, so a Ctrl-C during the sleep does not
        # chain the original error, whose message can carry the billing project or an account.
        print(f"{label}: {what} attempt {attempt} failed ({error_name}); retrying", flush=True)
        time.sleep(BACKOFF_SECONDS * 2**attempt)
    message = "unreachable: the loop returns or raises"
    raise AssertionError(message)


def _source_positions(*, source_axis: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """Return, for each `wanted` coordinate, the position of the source cell nearest to it.

    Args:
        source_axis: The source store's latitude or longitude axis.
        wanted: The coordinates to select, in the source's own convention.

    Returns:
        One position per wanted coordinate, in the order of `wanted`.
    """
    positions = np.abs(source_axis[:, np.newaxis] - wanted[np.newaxis, :]).argmin(axis=0)
    assert np.all(np.abs(source_axis[positions] - wanted) < AXIS_TOLERANCE), (
        "the source axis has no cell at a wanted coordinate"
    )
    return positions


def _open_cropped(*, fs: gcsfs.GCSFileSystem, run_name: str) -> xr.Dataset:
    """Open one run's store lazily and crop it to the box, keeping the seven `VARIABLES`.

    Asserts, before any chunk is read, that every variable's fill value is NaN, that `lead_time`
    decodes to a timedelta and is `1..LEADS` hours, that latitude ascends, that the store's
    `datetime` coordinate equals `init_time + lead_time` at every lead, and that every wanted
    latitude and longitude has a source cell within `AXIS_TOLERANCE`. The store has no consolidated
    metadata.

    Args:
        fs: A Requester Pays filesystem.
        run_name: The run's store directory name.

    Returns:
        The lazy, cropped dataset, in the store's own coordinates.
    """
    path = f"{BUCKET_PREFIX}/{run_name}/predictions.zarr"
    dataset = xr.open_zarr(
        fs.get_mapper(path), chunks=None, consolidated=False, decode_timedelta=True
    )
    for variable in VARIABLES:
        fill_value = dataset[variable].encoding.get("_FillValue")
        assert fill_value is not None, "no fill value"
        assert np.isnan(fill_value), "fill value is not NaN"
    assert dataset[LEAD_TIME].dtype.kind == "m", "lead_time did not decode to a timedelta"
    leads = dataset[LEAD_TIME].to_numpy()
    assert np.array_equal(leads, np.arange(1, LEADS + 1) * np.timedelta64(1, "h")), (
        "lead_time is not 1 to 360 hours"
    )
    assert np.all(dataset["datetime"].to_numpy() == dataset[INIT_TIME].to_numpy() + leads), (
        "datetime is not init_time + lead_time"
    )
    source_latitudes = dataset[LATITUDE_DIM].to_numpy()
    assert np.all(np.diff(source_latitudes) > 0), "latitude does not ascend"
    return dataset[list(VARIABLES)].isel(
        {
            LATITUDE_DIM: _source_positions(source_axis=source_latitudes, wanted=LATITUDES),
            # The source's longitude axis runs 0 to 360, so signed degrees are wrapped onto it.
            LONGITUDE_DIM: _source_positions(
                source_axis=dataset[LONGITUDE_DIM].to_numpy(), wanted=LONGITUDES % 360.0
            ),
        }
    )


def _read_chunk(
    *, dataset: xr.Dataset, variable: str, lead_position: int, label: str
) -> np.ndarray:
    """Read one cropped (variable, lead time) chunk, retrying with exponential backoff.

    Args:
        dataset: The lazy, cropped store.
        variable: One of `VARIABLES`.
        lead_position: Position on the `lead_time` axis.
        label: The run's `YYYYMMDD_HH` label, for the retry log line.

    Returns:
        A `(latitude, longitude)` `Float32` array.

    Raises:
        ValueError: If the slice is entirely NaN, after `MAX_ATTEMPTS` attempts.
        Exception: The last read error, after `MAX_ATTEMPTS` failed attempts.
    """

    def read() -> np.ndarray:
        slice_ = (
            dataset[variable]
            .isel(lead_time=lead_position)
            .transpose(LATITUDE_DIM, LONGITUDE_DIM)
            .to_numpy()
        )
        # Zarr version 3 reads a missing chunk as the fill value (NaN) instead of raising, and an
        # ensemble mean has no legitimate all-NaN slice, so an all-NaN slice means a missing chunk.
        if np.isnan(slice_).all():
            message = "all-NaN slice"
            raise ValueError(message)
        return slice_.astype(np.float32)

    return _with_retries(action=read, label=label, what="read")


def _load_run(*, dataset: xr.Dataset, label: str, workers: int) -> np.ndarray:
    """Read every variable at every lead time of one run, in a thread pool.

    Args:
        dataset: The lazy, cropped store.
        label: The run's `YYYYMMDD_HH` label.
        workers: Concurrent chunk reads.

    Returns:
        A `(variable, lead_time, latitude, longitude)` `Float32` array. A single failed read raises,
        so a returned array is complete.
    """
    n_leads = dataset.sizes["lead_time"]
    shape = (len(VARIABLES), n_leads, dataset.sizes[LATITUDE_DIM], dataset.sizes[LONGITUDE_DIM])
    values = np.empty(shape, dtype=np.float32)

    def read(task: tuple[int, int]) -> None:
        variable_position, lead_position = task
        values[variable_position, lead_position] = _read_chunk(
            dataset=dataset,
            variable=VARIABLES[variable_position],
            lead_position=lead_position,
            label=label,
        )

    tasks = [(v, lead) for v in range(len(VARIABLES)) for lead in range(n_leads)]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Consuming the iterator re-raises the first failed read.
        list(executor.map(read, tasks))
    return values


def _hours_since_epoch(*, moment: datetime) -> int:
    """Return `moment`, a UTC datetime on the hour, as whole hours since 1970-01-01T00:00 UTC."""
    return int(moment.timestamp()) // _SECONDS_PER_HOUR


def _init_time(*, day: date, hour: int) -> datetime:
    """Return the init time of the run at `hour` UTC on `day`."""
    return datetime(day.year, day.month, day.day, hour, tzinfo=UTC)


def _slot(*, day: date, hour: int, first_day: date) -> int:
    """Return the position of a run on the `init_time` axis of a store starting at `first_day`."""
    return (day - first_day).days * RUNS_PER_DAY + hour // HOURS_PER_SLOT


def _label(*, day: date, hour: int) -> str:
    """Return the `YYYYMMDD_HH` label of a run."""
    return f"{day:%Y%m%d}_{hour:02d}"


def _storage(*, bucket: str | None, local_store: Path | None) -> icechunk.Storage:
    """Return the storage of the output repository: a Cloud Storage bucket or a local directory.

    Args:
        bucket: The output bucket, in us-east1. Credentials come from the environment.
        local_store: A local directory, used by tests instead of a bucket.

    Returns:
        The storage of the repository at `STORE_PREFIX`, or of the directory itself.

    Raises:
        ValueError: If neither or both of `bucket` and `local_store` are given.
    """
    if (bucket is None) == (local_store is None):
        message = "give exactly one of bucket and local_store"
        raise ValueError(message)
    if local_store is not None:
        return icechunk.local_filesystem_storage(str(local_store))
    assert bucket is not None, "unreachable: exactly one of bucket and local_store is given"
    return icechunk.gcs_storage(bucket=bucket, prefix=STORE_PREFIX, from_env=True)


def _create_arrays(
    *, root: zarr.Group, first_day: date, last_day: date, retrieved_at: datetime
) -> None:
    """Create the empty arrays and the lineage attributes of a new repository.

    Every array is created and its coordinate arrays are filled, so a run is written by setting
    one slice of each of the seven variable arrays.

    Args:
        root: The writable root group of `STAGING_BRANCH`.
        first_day: The first init day, which fixes the start of the `init_time` axis for good.
        last_day: The last init day, which fixes the end of the `init_time` axis for good.
        retrieved_at: The retrieval time recorded in the attributes.
    """
    days = [first_day + timedelta(days=offset) for offset in range((last_day - first_day).days + 1)]
    init_hours = np.array(
        [
            _hours_since_epoch(moment=_init_time(day=day, hour=hour))
            for day in days
            for hour in INIT_HOURS
        ],
        dtype=np.int64,
    )
    n_runs = len(init_hours)
    axes: dict[str, tuple[np.ndarray, dict[str, JSON]]] = {
        INIT_TIME: (init_hours, {"units": HOURS_SINCE_EPOCH_UNITS}),
        LEAD_TIME: (np.arange(1, LEADS + 1, dtype=np.int64), {"units": "hours"}),
        LATITUDE: (LATITUDES, {"units": "degrees_north"}),
        LONGITUDE: (LONGITUDES, {"units": "degrees_east"}),
    }
    for name, (values, attributes) in axes.items():
        array = root.create_array(
            name,
            shape=values.shape,
            dtype=values.dtype,
            dimension_names=(name,),
            attributes=attributes,
        )
        array[:] = values
    root.create_array(
        SOURCE_INIT_TIME,
        shape=(n_runs,),
        dtype=np.int64,
        dimension_names=(INIT_TIME,),
        attributes={"units": HOURS_SINCE_EPOCH_UNITS},
    )
    root.create_array(RUN_WRITTEN, shape=(n_runs,), dtype=np.bool_, dimension_names=(INIT_TIME,))
    shape = (n_runs, LEADS, len(LATITUDES), len(LONGITUDES))
    for variable in VARIABLES:
        root.create_array(
            variable,
            shape=shape,
            dtype=np.float32,
            chunks=CHUNK_SHAPE,
            shards=SHARD_SHAPE,
            compressors=BloscCodec(cname="zstd", shuffle="shuffle"),
            fill_value=float("nan"),
            dimension_names=DIMENSIONS,
            attributes={"units": UNITS[variable]},
        )
    root.attrs.update(
        {
            "source_address": f"gs://{BUCKET_PREFIX}/<YYYYMMDD>_<HH>hr_01_preds/predictions.zarr",
            "retrieved_at_utc": retrieved_at.isoformat(),
            "box": {
                "lat_min": LAT_MIN,
                "lat_max": LAT_MAX,
                "lon_min": LON_MIN,
                "lon_max": LON_MAX,
            },
            "significand_bits_kept": KEEP_BITS,
            "units": UNITS,
            "radiation_convention": RADIATION_CONVENTION,
            FIRST_DAY_ATTRIBUTE: first_day.isoformat(),
            LAST_DAY_ATTRIBUTE: last_day.isoformat(),
            SKIPPED_ATTRIBUTE: [],
            NOT_PUBLISHED_ATTRIBUTE: [],
        }
    )


def _open_or_create_repository(
    *, storage: icechunk.Storage, first_day: date, last_day: date | None
) -> icechunk.Repository:
    """Open the repository, or create it with its arrays if `storage` holds none.

    Creating puts the arrays on `STAGING_BRANCH` and leaves `MAIN_BRANCH` at the empty initial
    snapshot, so a colleague reading `main` never sees an unvalidated run.

    Args:
        storage: Where the repository lives.
        first_day: The first init day, used only when creating.
        last_day: The last init day, required when creating.

    Returns:
        The repository, with both branches present.

    Raises:
        SystemExit: If the repository must be created and `last_day` is `None`.
    """
    if icechunk.Repository.exists(storage):
        return icechunk.Repository.open(storage)
    if last_day is None:
        sys.exit("--end-date is required when creating the repository.")
    repo = icechunk.Repository.create(storage)
    repo.create_branch(STAGING_BRANCH, repo.lookup_branch(MAIN_BRANCH))
    session = repo.writable_session(STAGING_BRANCH)
    _create_arrays(
        root=zarr.open_group(session.store, mode="a"),
        first_day=first_day,
        last_day=last_day,
        retrieved_at=datetime.now(UTC),
    )
    session.commit("create the arrays and coordinates")
    return repo


def _array(*, root: zarr.Group, name: str) -> zarr.Array:
    """Return the array called `name` in `root`."""
    return root.get_array(name)


def _staging_root(*, repo: icechunk.Repository) -> zarr.Group:
    """Return the root group of the current tip of `STAGING_BRANCH`, read-only."""
    return zarr.open_group(repo.readonly_session(branch=STAGING_BRANCH).store, mode="r")


def _stored_range(*, root: zarr.Group) -> tuple[date, date]:
    """Return the first and last init day the store was created with."""
    return (
        date.fromisoformat(str(root.attrs[FIRST_DAY_ATTRIBUTE])),
        date.fromisoformat(str(root.attrs[LAST_DAY_ATTRIBUTE])),
    )


def _is_written(*, repo: icechunk.Repository, slot: int) -> bool:
    """Return whether the tip of `STAGING_BRANCH` has the run in `slot` marked as written."""
    return bool(_array(root=_staging_root(repo=repo), name=RUN_WRITTEN)[slot])


def _round_values(*, values: np.ndarray) -> np.ndarray:
    """Round every value to `KEEP_BITS` significand bits, one variable at a time."""
    rounded = np.empty_like(values)
    for position in range(values.shape[0]):
        flat = pl.select(
            round_to_significand_bits(
                pl.lit(pl.Series(values[position].reshape(-1))), keep_bits=KEEP_BITS
            )
        ).to_series()
        rounded[position] = flat.to_numpy().reshape(values.shape[1:])
    return rounded


def _write_run(
    *, session: icechunk.Session, slot: int, values: np.ndarray, source_init_hours: int
) -> None:
    """Set one run's slice of every array in `session`, writing `run_written` last."""
    root = zarr.open_group(session.store, mode="r+")
    for position, variable in enumerate(VARIABLES):
        _array(root=root, name=variable)[slot] = values[position]
    _array(root=root, name=SOURCE_INIT_TIME)[slot] = source_init_hours
    _array(root=root, name=RUN_WRITTEN)[slot] = True


def _commit_run(
    *, repo: icechunk.Repository, session: icechunk.Session, slot: int, label: str
) -> None:
    """Commit `session`; after a commit error, keep going only if the run is on `staging` anyway.

    The commit is not retried, because a retry after a lost response would fail as a conflict. The
    tip of `STAGING_BRANCH` is read instead.

    Args:
        repo: The output repository.
        session: The writable session that holds the run's data.
        slot: The run's position on the `init_time` axis.
        label: The run's `YYYYMMDD_HH` label, for the log line and the commit message.

    Raises:
        Exception: The commit error, if the run is not on `STAGING_BRANCH` afterwards.
    """
    try:
        session.commit(label)
    except Exception as error:
        if not _is_written(repo=repo, slot=slot):
            raise
        print(
            f"{label}: commit reported {type(error).__name__} but the run is on staging",
            flush=True,
        )


def _fetch_run(
    *,
    fs: gcsfs.GCSFileSystem,
    repo: icechunk.Repository,
    day: date,
    hour: int,
    first_day: date,
    workers: int,
) -> None:
    """Read one run, then write it to `STAGING_BRANCH` in one commit.

    A fresh session is opened for each run, and a session that raised is dropped, because chunks
    reach storage as soon as they are set and a reused session would carry a failed run into the
    next commit.

    Args:
        fs: A Requester Pays filesystem for the source bucket.
        repo: The output repository.
        day: The run's init day.
        hour: The run's init hour, UTC.
        first_day: The first init day of the store, which fixes the run's slot.
        workers: Concurrent chunk reads.

    Raises:
        AssertionError: If the source store's `init_time` differs from the run's slot.
    """
    label = _label(day=day, hour=hour)
    dataset = _with_retries(
        action=lambda: _open_cropped(fs=fs, run_name=_run_name(day=day, hour=hour)),
        label=label,
        what="open",
    )
    expected = _hours_since_epoch(moment=_init_time(day=day, hour=hour))
    source_init = dataset[INIT_TIME].to_numpy().astype("datetime64[h]").astype(np.int64)
    assert int(source_init) == expected, "source init_time differs from the run's slot"
    values = _round_values(values=_load_run(dataset=dataset, label=label, workers=workers))
    slot = _slot(day=day, hour=hour, first_day=first_day)
    session = repo.writable_session(STAGING_BRANCH)
    _write_run(session=session, slot=slot, values=values, source_init_hours=expected)
    _commit_run(repo=repo, session=session, slot=slot, label=label)
    print(f"{label}: fetched", flush=True)


def _candidate_runs(
    *, start: date, end: date, init_hours: list[int], listed: set[str]
) -> list[tuple[date, int]]:
    """Return the (day, hour) of every listed run inside the window at the wanted hours.

    Args:
        start: First init date.
        end: Last init date.
        init_hours: Wanted init hours, UTC.
        listed: Every run directory name found in the bucket.

    Returns:
        The sorted (day, hour) pairs of the runs to fetch.
    """
    parsed = [run for name in listed if (run := _parse_run_name(name=name)) is not None]
    return sorted((day, hour) for day, hour in parsed if start <= day <= end and hour in init_hours)


def _unlisted_runs(
    *, start: date, end: date, init_hours: list[int], listed: set[str]
) -> list[tuple[date, int]]:
    """Return the (day, hour) of every wanted run in the window that the bucket does not list."""
    parsed = {run for name in listed if (run := _parse_run_name(name=name)) is not None}
    return [
        (day, hour)
        for day in (start + timedelta(days=offset) for offset in range((end - start).days + 1))
        for hour in sorted(init_hours)
        if (day, hour) not in parsed
    ]


def _record_invocation(
    *,
    repo: icechunk.Repository,
    skipped: dict[str, list[str]],
    examined: set[str],
) -> None:
    """Record the skipped runs in the group attributes, with the retrieval time, in one commit.

    For each attribute, the labels stored earlier stay unless this invocation examined the run
    again, so the stored lists are the union of the earlier and the new lists across invocations.

    Args:
        repo: The output repository.
        skipped: The new labels to record, keyed by attribute name.
        examined: Labels of the runs whose presence this invocation checked.
    """
    session = repo.writable_session(STAGING_BRANCH)
    root = zarr.open_group(session.store, mode="r+")
    for attribute, new_labels in skipped.items():
        stored_labels = root.attrs[attribute]
        assert isinstance(stored_labels, list), "a skipped-runs attribute is not a list"
        stored = {str(label) for label in stored_labels}
        root.attrs[attribute] = sorted((stored - examined) | set(new_labels))
    root.attrs["retrieved_at_utc"] = datetime.now(UTC).isoformat()
    session.commit("record skipped runs and the retrieval time")


def _dry_run(*, arguments: argparse.Namespace) -> int:
    """Print the number of runs and the estimated transfer, touching neither network nor store."""
    # No network: the newest run is assumed to be yesterday's, and every run to exist.
    last = arguments.end_date or datetime.now(UTC).date() - timedelta(days=1)
    days = max((last - arguments.start_date).days + 1, 0)
    estimate = days * len(arguments.init_hours) * _estimated_gb()
    print(f"dry run: {days * len(arguments.init_hours)} runs, about {estimate:.0f} GB to read")
    print(f"about ${estimate * EGRESS_USD_PER_GB:.0f} if read from outside us-east1")
    return 0


def _clip_to_stored_range(
    *, arguments: argparse.Namespace, first_day: date, last_day: date
) -> tuple[date, date]:
    """Return the requested window clipped to the stored range; print how many runs fall outside."""
    requested_end = arguments.end_date or last_day
    start = max(arguments.start_date, first_day)
    end = min(requested_end, last_day)
    requested = max((requested_end - arguments.start_date).days + 1, 0)
    inside = max((end - start).days + 1, 0)
    if requested > inside:
        print(
            f"{(requested - inside) * len(arguments.init_hours)} requested runs fall outside the "
            "stored init_time range and are ignored"
        )
    return start, end


def _run(*, arguments: argparse.Namespace) -> int:
    """Run the fetch for parsed `arguments`; return the process exit code."""
    if arguments.dry_run:
        return _dry_run(arguments=arguments)

    repo = _open_or_create_repository(
        storage=_storage(bucket=arguments.bucket, local_store=arguments.local_store),
        first_day=arguments.start_date,
        last_day=arguments.end_date,
    )
    first_day, last_day = _stored_range(root=_staging_root(repo=repo))
    start, end = _clip_to_stored_range(arguments=arguments, first_day=first_day, last_day=last_day)

    fs = _filesystem()
    listed = {
        Path(entry).name
        for entry in _with_retries(
            action=lambda: fs.ls(BUCKET_PREFIX, detail=False), label="bucket", what="listing"
        )
    }
    written = np.asarray(_array(root=_staging_root(repo=repo), name=RUN_WRITTEN)[:])
    unwritten = [
        run
        for run in _candidate_runs(
            start=start, end=end, init_hours=arguments.init_hours, listed=listed
        )
        if not written[_slot(day=run[0], hour=run[1], first_day=first_day)]
    ]
    not_published = [
        run
        for run in _unlisted_runs(
            start=start, end=end, init_hours=arguments.init_hours, listed=listed
        )
        if not written[_slot(day=run[0], hour=run[1], first_day=first_day)]
    ]
    no_marker: list[tuple[date, int]] = []
    to_fetch: list[tuple[date, int]] = []
    for day, hour in unwritten:
        label = _label(day=day, hour=hour)
        marker = f"{BUCKET_PREFIX}/{_run_name(day=day, hour=hour)}/success"
        if _with_retries(action=lambda m=marker: fs.exists(m), label=label, what="marker check"):
            to_fetch.append((day, hour))
        else:
            no_marker.append((day, hour))
            print(f"{label}: no success marker, skipping", flush=True)
    estimate = len(to_fetch) * _estimated_gb()
    print(f"{len(to_fetch)} runs to fetch, about {estimate:.0f} GB to read")
    if to_fetch and not _in_us_east1() and estimate > arguments.max_external_gb:
        print(
            f"Refusing to read about {estimate:.0f} GB from outside us-east1: it would be billed "
            f"as egress at ${EGRESS_USD_PER_GB}/GB, about ${estimate * EGRESS_USD_PER_GB:.0f}. "
            "Run this script on a Compute Engine machine in us-east1, or raise "
            "--max-external-gb.",
            file=sys.stderr,
        )
        return 1

    _record_invocation(
        repo=repo,
        skipped={
            SKIPPED_ATTRIBUTE: [_label(day=day, hour=hour) for day, hour in no_marker],
            NOT_PUBLISHED_ATTRIBUTE: [_label(day=day, hour=hour) for day, hour in not_published],
        },
        examined={_label(day=day, hour=hour) for day, hour in [*unwritten, *not_published]},
    )
    failed: list[str] = []
    for day, hour in to_fetch:
        try:
            _fetch_run(
                fs=fs, repo=repo, day=day, hour=hour, first_day=first_day, workers=arguments.workers
            )
        except Exception as error:  # noqa: BLE001  # one failed run must not stop the job
            label = _label(day=day, hour=hour)
            failed.append(label)
            print(f"{label}: FAILED ({_describe(error=error)})", file=sys.stderr, flush=True)
    if failed:
        print(f"Failed runs: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    """Fetch WeatherNext 3 ensemble-mean runs into an Icechunk repository, one commit per run.

    Any uncaught exception prints only its type name, because a message can carry an account name,
    the billing project, or the bucket name. The message of an `AssertionError` is printed too,
    because every assertion in this module has a fixed message. Ctrl-C prints only `interrupted`.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bucket", help="Output Cloud Storage bucket, in us-east1 (repository at STORE_PREFIX)."
    )
    parser.add_argument("--local-store", type=Path, help="Output directory, instead of a bucket.")
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=ARCHIVE_START,
        help="First init date, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        help=(
            "Last init date, YYYY-MM-DD. Required when creating the repository, where it fixes "
            "the end of the init_time axis for good. Default afterwards: the stored last day."
        ),
    )
    parser.add_argument(
        "--init-hours",
        type=int,
        nargs="+",
        choices=INIT_HOURS,
        default=list(INIT_HOURS),
        help="Init hours, UTC.",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Concurrent reads.")
    parser.add_argument("--max-external-gb", type=float, default=DEFAULT_MAX_EXTERNAL_GB)
    parser.add_argument("--dry-run", action="store_true", help="Print the estimate and exit.")
    arguments = parser.parse_args()
    if not arguments.dry_run and (arguments.bucket is None) == (arguments.local_store is None):
        parser.error("give exactly one of --bucket and --local-store")
    try:
        return _run(arguments=arguments)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return INTERRUPTED_EXIT_CODE
    except BaseException as error:  # noqa: BLE001  # a panic in the Rust core is not an Exception
        print(f"FAILED ({_describe(error=error)})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
