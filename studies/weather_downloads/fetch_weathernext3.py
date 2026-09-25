"""Download WeatherNext 3 ensemble-mean runs, cropped to the trial-area box.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/934>, the WeatherNext 3 arm of
the matched-lead comparison. The source is the Requester Pays bucket
`weathernext3_statistics_spatial` in region US-EAST1, which holds one Zarr store per run, at 0.1
degrees over the globe, with the ensemble mean of each variable. The archive starts on 2026-01-01
and has a run every hour. The runs at 00, 06, 12, and 18 UTC have 360 hourly lead times, and the
other hours only 48.

**The Zarr chunks are whole-globe, so cropping to the box saves no bytes.** Each chunk holds one
variable at one lead time over the whole globe, about 20 MB compressed. The script therefore
transfers about 50 GB per run (360 lead times, 7 variables) and keeps about 15 MB of it. The crop is
applied lazily with one `.isel()` on the store's latitude and longitude axes, and the box appears
only in that call in this process.

**The script must run on a Compute Engine machine in us-east1.** Reads inside the same region cost
nothing, while reads from anywhere else are billed as internet egress at $0.12 per GB. One run is
about 50 GB of reads, which costs about £4.50 ($6) from outside Google Cloud. The four 360-hour runs
of one day are about 200 GB, and the 267 days in the archive so far would be about 53 TB. The script
detects whether it is on a Compute Engine machine in us-east1 through the metadata server. Outside
us-east1 it refuses to start when the estimated transfer exceeds `--max-external-gb` (default 5.0),
which is less than one run. `--dry-run` prints the number of runs and the estimated transfer, and
touches neither the network nor the disk.

**Credentials and the billing project come from the environment.** `GOOGLE_CLOUD_PROJECT` names the
project that pays for the reads, and the script exits with a message if that variable is unset.
Authentication uses Google application default credentials: `GOOGLE_APPLICATION_CREDENTIALS` on a
workstation, or the machine's own service account on Compute Engine. The script reads the project
name from the environment and never prints or writes it, and it never reads, prints, or writes the
credentials or an account name.

**The script checkpoints one run per file.** Each run is written to `_run_cache/` as soon as every
one of its reads has succeeded, and a re-run skips every run already cached. One run is about 50 GB
of reads, so a month per file would put 1.5 TB at risk. Each (variable, lead time) chunk is read by
one task in a pool of `--workers` threads (default 16), retried with exponential backoff on any
error. A missing chunk does not raise in Zarr version 3 (it reads as the fill value, NaN), so a
slice that is entirely NaN is treated as a failed read. A run whose `success` marker object is
absent is skipped and recorded in the lineage note rather than raised. The final file is built from
this invocation's runs with `scan_parquet` and `sink_parquet`, after checking that every run file
records the same grid hash.

**A failed run does not stop the job.** The script records the run's label, continues with the other
runs, prints the failed labels at the end, exits non-zero, and skips the combine step. An uncaught
exception prints only the run label and the exception's type name, because an exception message can
carry an account name, the billing project, or the crop's shape.

**Row counts, cell counts, and the crop's hash go only to the private lineage note and the parquet
metadata, never to stdout,** because they reveal the size of the trial-area box. The `cell_hash` in
the parquet metadata is as sensitive as the box itself, because anyone holding the grid can
recompute the hash and confirm a guessed box.

Run it on the machine with `GOOGLE_CLOUD_PROJECT=<project> uv run python
studies/weather_downloads/fetch_weathernext3.py`, adding `--start-date` and `--end-date` (both
`YYYY-MM-DD`, inclusive) to fetch a window into its own directory, and `--init-hours`
(default `0 6 12 18`, the four runs with 360 lead times) to choose which runs of each day. Then
check the output with `validate_weathernext3.py`.
"""

import argparse
import hashlib
import os
import sys
import time
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

import gcsfs
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import xarray as xr
from delta_store.precision import round_to_significand_bits
from fetch_dynamical_zarr import _axis_column
from lineage import write_lineage_note, write_readme
from paths import WEATHER_DOWNLOADS_DIR, load_trial_area_box

BUCKET_PREFIX: Final[str] = (
    "weathernext3_statistics_spatial/weathernext_3_0_0_statistics/zarr/2026_to_present"
)
"""Where the run stores live, without the `gs://` scheme, as `gcsfs` expects."""

ARCHIVE_START: Final[date] = date(2026, 1, 1)

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

LATITUDE_DIM: Final[str] = "lat_0p1"
LONGITUDE_DIM: Final[str] = "lon_0p1"

DEFAULT_WORKERS: Final[int] = 16
"""Concurrent chunk reads within one run. Each read is a network wait for about 20 MB."""

KEEP_BITS: Final[int] = 13
"""Significand bits kept in every value column, as in production NWP storage."""

MAX_ATTEMPTS: Final[int] = 5
"""Attempts at one chunk read before the error is raised."""

BACKOFF_SECONDS: Final[float] = 5.0
"""Base of the exponential backoff: the first retry sleeps twice this (10 s), the next four
times."""

LONG_RUN_LEADS: Final[int] = 360
SHORT_RUN_LEADS: Final[int] = 48
LONG_RUN_HOURS: Final[frozenset[int]] = frozenset({0, 6, 12, 18})
"""Init hours (UTC) whose runs have `LONG_RUN_LEADS` lead times; the other hours have
`SHORT_RUN_LEADS`."""

GB_PER_CHUNK: Final[float] = 0.0197
"""Measured compressed size of one (variable, lead time) chunk, GB."""

DEFAULT_MAX_EXTERNAL_GB: Final[float] = 5.0
"""Largest estimated transfer, GB, the script accepts outside us-east1: less than one run."""

EGRESS_USD_PER_GB: Final[float] = 0.12
"""Internet egress price, dollars per GB, for reads from outside the bucket's region."""

_METADATA_ZONE_URL: Final[str] = "http://metadata.google.internal/computeMetadata/v1/instance/zone"
_METADATA_TIMEOUT_SECONDS: Final[float] = 2.0
_BYTES_PER_MB: Final[float] = 1e6


def _estimated_gb(*, init_hour: int) -> float:
    """Return the estimated GB transferred to read every variable at every lead of one run."""
    leads = LONG_RUN_LEADS if init_hour in LONG_RUN_HOURS else SHORT_RUN_LEADS
    return leads * len(VARIABLES) * GB_PER_CHUNK


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

    A transient network error partway through a run would otherwise abandon 50 GB of reads. The
    first sleep is 10 s and each later one doubles. Each retry prints the run label, what was being
    done, the attempt number, and the exception's type name only, because an exception message can
    carry a request URL, an account name, or the billing project.

    Args:
        action: The call to make.
        label: The run's `YYYYMMDD_HH` label, or `bucket`, for the retry log line.
        what: Short description of the call, for the retry log line.

    Returns:
        Whatever `action` returns.

    Raises:
        Exception: The last error, after `MAX_ATTEMPTS` failed attempts.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return action()
        except Exception as error:
            if attempt == MAX_ATTEMPTS:
                raise
            print(
                f"{label}: {what} attempt {attempt} failed ({type(error).__name__}); retrying",
                flush=True,
            )
            time.sleep(BACKOFF_SECONDS * 2**attempt)
    raise AssertionError  # unreachable: the loop returns or raises


def _crop_indices(*, dataset: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Return the latitude and longitude positions inside the trial-area box.

    The store's longitude axis is 0 to 360, so it is converted to signed degrees before the
    comparison, and the longitude positions are ordered by signed longitude, ascending. Latitude
    ascends in this store, so its positions are already ascending.

    Args:
        dataset: A run store, still lazy.

    Returns:
        The latitude positions and the longitude positions to pass to one `.isel()`.
    """
    box = load_trial_area_box()
    latitudes = dataset[LATITUDE_DIM].to_numpy()
    signed_longitudes = (dataset[LONGITUDE_DIM].to_numpy() + 180.0) % 360.0 - 180.0
    lat_positions = np.flatnonzero((latitudes >= box.lat_min) & (latitudes <= box.lat_max))
    lon_inside = np.flatnonzero(
        (signed_longitudes >= box.lon_min) & (signed_longitudes <= box.lon_max)
    )
    lon_positions = lon_inside[np.argsort(signed_longitudes[lon_inside])]
    return lat_positions, lon_positions


def _open_cropped(*, fs: gcsfs.GCSFileSystem, run_name: str) -> xr.Dataset:
    """Open one run's store lazily and crop it to the box, keeping the seven `VARIABLES`.

    Asserts, before any chunk is read, that every variable's fill value is NaN, that `lead_time`
    decodes to a timedelta, that latitude ascends, and that the store's `datetime` coordinate
    equals `init_time + lead_time` at every lead.
    """
    path = f"{BUCKET_PREFIX}/{run_name}/predictions.zarr"
    dataset = xr.open_zarr(
        fs.get_mapper(path), chunks=None, consolidated=None, decode_timedelta=True
    )
    for variable in VARIABLES:
        fill_value = dataset[variable].encoding.get("_FillValue")
        assert fill_value is not None, "no fill value"
        assert np.isnan(fill_value), "fill value is not NaN"
    assert dataset["lead_time"].dtype.kind == "m", "lead_time did not decode to a timedelta"
    assert np.all(
        dataset["datetime"].to_numpy()
        == dataset["init_time"].to_numpy() + dataset["lead_time"].to_numpy()
    ), "datetime is not init_time + lead_time"
    lat_positions, lon_positions = _crop_indices(dataset=dataset)
    cropped = dataset[list(VARIABLES)].isel(
        {LATITUDE_DIM: lat_positions, LONGITUDE_DIM: lon_positions}
    )
    assert np.all(np.diff(cropped[LATITUDE_DIM].to_numpy()) > 0), "latitude does not ascend"
    # Cell coordinates in signed degrees, so `_grid_cells.parquet` and the hash match across runs.
    return cropped.assign_coords(
        {LONGITUDE_DIM: (cropped[LONGITUDE_DIM].to_numpy() + 180.0) % 360.0 - 180.0}
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


def _to_long_frame(*, dataset: xr.Dataset, values: np.ndarray) -> pl.DataFrame:
    """Flatten one loaded run to a long frame with no coordinate column.

    One row per (init_time, lead_time, lat_index, lon_index). `lat_index` and `lon_index` are the
    rank of each cell's latitude and signed longitude within the crop, ascending, replacing the
    grid's own coordinates, which locate a generator to within one grid cell. Value columns are
    `Float32` rounded to `KEEP_BITS` significand bits.

    Args:
        dataset: The cropped dataset the values were read from.
        values: The `(variable, lead_time, latitude, longitude)` array from `_load_run`.

    Returns:
        A long frame keyed by integer grid indices rather than coordinates.
    """
    shape = values.shape[1:]
    leads = dataset["lead_time"].to_numpy()
    columns: dict[str, np.ndarray] = {
        "init_time": np.full(int(np.prod(shape)), dataset["init_time"].to_numpy()),
        "lead_time": _axis_column(values=leads, axis=0, shape=shape),
        "lat_index": _axis_column(values=np.arange(shape[1], dtype=np.int16), axis=1, shape=shape),
        "lon_index": _axis_column(values=np.arange(shape[2], dtype=np.int16), axis=2, shape=shape),
    }
    for position, variable in enumerate(VARIABLES):
        columns[variable] = values[position].reshape(-1)
    frame = pl.DataFrame(columns)
    return frame.with_columns(
        round_to_significand_bits(pl.col(variable), keep_bits=KEEP_BITS) for variable in VARIABLES
    )


def _write_grid_cells(*, dataset: xr.Dataset, path: Path, fingerprint: dict[str, str]) -> None:
    """Write the private lookup from `lat_index`/`lon_index` to the grid cell's coordinates.

    The file stays on the private data disk and is never printed, published, or quoted. It is
    written through a `.tmp` rename, only when it is absent or its recorded hash differs from
    `fingerprint`.
    """
    if (
        path.exists()
        and pl.read_parquet_metadata(path).get("cell_hash") == fingerprint["cell_hash"]
    ):
        return
    latitudes = dataset[LATITUDE_DIM].to_numpy()
    longitudes = dataset[LONGITUDE_DIM].to_numpy()
    pl.DataFrame(
        {
            "lat_index": np.repeat(np.arange(len(latitudes), dtype=np.int16), len(longitudes)),
            "lon_index": np.tile(np.arange(len(longitudes), dtype=np.int16), len(latitudes)),
            "latitude": np.repeat(latitudes, len(longitudes)),
            "longitude": np.tile(longitudes, len(latitudes)),
        }
    ).write_parquet(temporary := path.with_suffix(".parquet.tmp"), metadata=fingerprint)
    temporary.rename(path)


def _cell_fingerprint(*, dataset: xr.Dataset) -> dict[str, str]:
    """Return the crop's cell count and a hash of its cell coordinates, as parquet metadata.

    The values stay in the private parquet metadata and are never printed.
    """
    digest = hashlib.sha256()
    digest.update(dataset[LATITUDE_DIM].to_numpy().tobytes())
    digest.update(dataset[LONGITUDE_DIM].to_numpy().tobytes())
    n_cells = dataset.sizes[LATITUDE_DIM] * dataset.sizes[LONGITUDE_DIM]
    return {"cell_count": str(n_cells), "cell_hash": digest.hexdigest()}


def _candidate_runs(
    *, start: date, end: date | None, init_hours: list[int], listed: set[str]
) -> list[tuple[date, int]]:
    """Return the (day, hour) of every listed run inside the window at the wanted hours.

    Args:
        start: First init date.
        end: Last init date, or `None` for the newest listed run's date.
        init_hours: Wanted init hours, UTC.
        listed: Every run directory name found in the bucket.

    Returns:
        The sorted (day, hour) pairs of the runs to fetch.
    """
    parsed = [run for name in listed if (run := _parse_run_name(name=name)) is not None]
    last = end or max(day for day, _ in parsed)
    return sorted(
        (day, hour) for day, hour in parsed if start <= day <= last and hour in init_hours
    )


def _write_documentation(
    *,
    output_dir: Path,
    first_init: str,
    last_init: str,
    used_paths: list[Path],
    skipped: list[str],
    fingerprint: dict[str, str],
    rows: int,
    size_mb: float,
) -> None:
    """Write the lineage note and the README next to the combined file.

    Args:
        output_dir: The product directory.
        first_init: The earliest `init_time` fetched, ISO format.
        last_init: The latest `init_time` fetched, ISO format.
        used_paths: The run files combined.
        skipped: Labels of runs skipped because their `success` marker is missing.
        fingerprint: The crop's cell count and hash (private, lineage only).
        rows: Row count of the combined file (private, lineage only).
        size_mb: Size of the combined file, MB.
    """
    lineage_filename = "lineage.json"
    write_lineage_note(
        product_dir=output_dir,
        source_address=f"gs://{BUCKET_PREFIX}/<YYYYMMDD>_<HH>hr_01_preds/predictions.zarr",
        request_description=(
            f"WeatherNext 3 ensemble-mean runs, variables {', '.join(VARIABLES)}, every lead_time, "
            f"cropped to the trial-area box with one .isel() on {LATITUDE_DIM}/{LONGITUDE_DIM}"
        ),
        variables=list(VARIABLES),
        extra={
            "first_init_time": first_init,
            "last_init_time": last_init,
            "runs_cached": [path.stem for path in used_paths],
            "runs_skipped_missing_success_marker": skipped,
            "rows": rows,
            "output_size_mb": round(size_mb, 2),
            "cell_count": fingerprint["cell_count"],
            "significand_bits_kept": KEEP_BITS,
        },
        filename=lineage_filename,
    )
    write_readme(
        product_dir=output_dir,
        product_name="WeatherNext 3 ensemble-mean runs from Google Cloud Storage",
        source_web_page="https://cloud.google.com/blog/products/ai-machine-learning/weathernext",
        script_path="studies/weather_downloads/fetch_weathernext3.py",
        lineage_filenames=[lineage_filename],
        columns={
            "init_time": "Timezone-naive (implicitly UTC) start time of the model run.",
            "lead_time": "Duration since init_time. The valid time is init_time + lead_time.",
            "lat_index": "Rank of the cell's latitude in the crop, ascending. Not a coordinate.",
            "lon_index": (
                "Rank of the cell's signed longitude in the crop, ascending. Not a coordinate."
            ),
            "temperature_2m_mean": "Ensemble-mean air temperature at 2 m, Kelvin.",
            "u_component_of_wind_10m_mean": "Ensemble-mean eastward wind at 10 m, m/s.",
            "v_component_of_wind_10m_mean": "Ensemble-mean northward wind at 10 m, m/s.",
            "u_component_of_wind_100m_mean": "Ensemble-mean eastward wind at 100 m, m/s.",
            "v_component_of_wind_100m_mean": "Ensemble-mean northward wind at 100 m, m/s.",
            "surface_solar_radiation_downwards_1hr_mean": (
                "Ensemble-mean downward shortwave radiation at the surface, J/m2, accumulated "
                "over the hour ending at init_time + lead_time."
            ),
            "total_sky_direct_solar_radiation_at_surface_1hr_mean": (
                "Ensemble-mean direct (beam) solar radiation on a horizontal surface, J/m2, "
                "accumulated over the hour ending at init_time + lead_time."
            ),
        },
        missing_value_convention=(
            "A missing value would be `NaN` in a `Float32` value column. A run is written only "
            "if every read succeeded, and `validate_weathernext3.py` reports any `NaN` or null."
        ),
        gotchas=[
            (
                "Radiation is in the store's native J/m2, accumulated over the hour ending at the "
                "valid time. Divide by 3600 for the mean W/m2 over that hour. No unit conversion "
                "is applied to any column."
            ),
            (
                "Every value is the ensemble mean, so a mean of a non-linear field such as "
                "radiation is smoother than any single member. There is no `ensemble_member` "
                "column."
            ),
            (
                "The archive starts on 2026-01-01 and holds one model version label, "
                "`weathernext_3_0_0`. No model-version boundary is known."
            ),
            (
                "Runs at 00, 06, 12, and 18 UTC have 360 hourly lead times, and the other hours "
                "have 48. `--init-hours` chooses which runs are fetched."
            ),
            (
                "A run whose `success` marker object is missing in the bucket is skipped and "
                "listed in the lineage note."
            ),
            f"Value columns are rounded to {KEEP_BITS} significand bits (relative error 1.2e-4).",
            (
                "`_grid_cells.parquet` maps `lat_index` and `lon_index` to coordinates. It is "
                "private and must never be published. No value here is per-generator: the "
                "columns are gridded weather only."
            ),
            (
                "The data is licensed CC BY 4.0 once it is more than 1 hour old. Data fresher "
                "than that falls under experimental terms."
            ),
        ],
        external_docs={
            "WeatherNext": "https://cloud.google.com/blog/products/ai-machine-learning/weathernext",
        },
    )


def _fetch_run(
    *, fs: gcsfs.GCSFileSystem, run_name: str, label: str, output_dir: Path, workers: int
) -> None:
    """Read one run, then checkpoint it to `_run_cache/<label>.parquet` through a `.tmp` rename."""
    dataset = _with_retries(
        action=lambda: _open_cropped(fs=fs, run_name=run_name), label=label, what="open"
    )
    fingerprint = _cell_fingerprint(dataset=dataset)
    values = _load_run(dataset=dataset, label=label, workers=workers)
    frame = _to_long_frame(dataset=dataset, values=values)
    _write_grid_cells(
        dataset=dataset, path=output_dir / "_grid_cells.parquet", fingerprint=fingerprint
    )
    target = output_dir / "_run_cache" / f"{label}.parquet"
    temporary = target.with_suffix(".parquet.tmp")
    frame.write_parquet(temporary, compression="zstd", metadata=fingerprint)
    temporary.rename(target)
    print(f"{label}: fetched", flush=True)


def _combine_runs(*, used_paths: list[Path], output_dir: Path) -> tuple[dict[str, str], int, float]:
    """Combine run files into `WeatherNext3.parquet` after checking they share one grid hash.

    Args:
        used_paths: The per-run Parquet files to combine.
        output_dir: The directory that receives `WeatherNext3.parquet`.

    Returns:
        The shared cell fingerprint, the combined row count, and the combined file's size in MB.

    Raises:
        ValueError: If a run file was cropped to different grid cells.
    """
    fingerprint = {
        key: pl.read_parquet_metadata(used_paths[0]).get(key, "")
        for key in ("cell_count", "cell_hash")
    }
    for path in used_paths:
        stored = pl.read_parquet_metadata(path)
        if {key: stored.get(key) for key in fingerprint} != fingerprint:
            message = f"{path.name} was cropped to different grid cells; delete it and re-run"
            raise ValueError(message)
    output_path = output_dir / "WeatherNext3.parquet"
    temporary = output_path.with_suffix(".parquet.tmp")
    pl.scan_parquet(used_paths).sink_parquet(temporary, compression="zstd", metadata=fingerprint)
    temporary.rename(output_path)
    rows = pq.ParquetFile(output_path).metadata.num_rows
    print(f"WeatherNext3: wrote {output_path}")
    return fingerprint, rows, output_path.stat().st_size / _BYTES_PER_MB


def _run(*, arguments: argparse.Namespace) -> int:
    """Run the fetch for parsed `arguments`; return the process exit code."""
    is_window = arguments.start_date != ARCHIVE_START or arguments.end_date is not None

    if arguments.dry_run:
        # No network: the newest run is assumed to be yesterday's, and every run to exist.
        last = arguments.end_date or datetime.now(UTC).date() - timedelta(days=1)
        days = max((last - arguments.start_date).days + 1, 0)
        estimate = days * sum(_estimated_gb(init_hour=hour) for hour in arguments.init_hours)
        print(f"dry run: {days * len(arguments.init_hours)} runs, about {estimate:.0f} GB to read")
        print(f"about ${estimate * EGRESS_USD_PER_GB:.0f} if read from outside us-east1")
        return 0

    fs = _filesystem()
    entries = _with_retries(
        action=lambda: fs.ls(BUCKET_PREFIX, detail=False), label="bucket", what="listing"
    )
    candidates = _candidate_runs(
        start=arguments.start_date,
        end=arguments.end_date,
        init_hours=arguments.init_hours,
        listed={Path(entry).name for entry in entries},
    )
    directory_name = (
        f"WeatherNext3_window_{arguments.start_date}_{arguments.end_date or 'latest'}"
        if is_window
        else "WeatherNext3"
    )
    output_dir = WEATHER_DOWNLOADS_DIR / directory_name
    run_cache_dir = output_dir / "_run_cache"
    run_cache_dir.mkdir(parents=True, exist_ok=True)

    skipped: list[str] = []
    runs: list[tuple[str, str, int]] = []
    for day, hour in candidates:
        run_name = _run_name(day=day, hour=hour)
        label = f"{day:%Y%m%d}_{hour:02d}"
        marker = f"{BUCKET_PREFIX}/{run_name}/success"
        if not _with_retries(
            action=lambda m=marker: fs.exists(m), label=label, what="marker check"
        ):
            skipped.append(label)
            print(f"{label}: no success marker, skipping", flush=True)
            continue
        runs.append((label, run_name, hour))
    to_fetch = [run for run in runs if not (run_cache_dir / f"{run[0]}.parquet").exists()]
    estimate = sum(_estimated_gb(init_hour=hour) for _, _, hour in to_fetch)
    print(f"{len(runs)} runs found, {len(to_fetch)} to fetch, about {estimate:.0f} GB to read")
    if to_fetch and not _in_us_east1() and estimate > arguments.max_external_gb:
        print(
            f"Refusing to read about {estimate:.0f} GB from outside us-east1: it would be billed "
            f"as egress at ${EGRESS_USD_PER_GB}/GB, about ${estimate * EGRESS_USD_PER_GB:.0f}. "
            "Run this script on a Compute Engine machine in us-east1, or raise "
            "--max-external-gb.",
            file=sys.stderr,
        )
        return 1

    failed: list[str] = []
    for label, run_name, _ in to_fetch:
        try:
            _fetch_run(
                fs=fs,
                run_name=run_name,
                label=label,
                output_dir=output_dir,
                workers=arguments.workers,
            )
        except Exception as error:  # noqa: BLE001  # one failed run must not stop the job
            failed.append(label)
            print(f"{label}: FAILED ({type(error).__name__})", file=sys.stderr, flush=True)
    if failed:
        print(f"Failed runs, combine skipped: {', '.join(failed)}", file=sys.stderr)
        return 1

    used_paths = [run_cache_dir / f"{label}.parquet" for label, _, _ in runs]
    if not used_paths:
        print("No runs to combine.", file=sys.stderr)
        return 1
    fingerprint, rows, size_mb = _combine_runs(used_paths=used_paths, output_dir=output_dir)

    inits = [datetime.strptime(path.stem, "%Y%m%d_%H").replace(tzinfo=UTC) for path in used_paths]
    _write_documentation(
        output_dir=output_dir,
        first_init=min(inits).isoformat(),
        last_init=max(inits).isoformat(),
        used_paths=used_paths,
        skipped=skipped,
        fingerprint=fingerprint,
        rows=rows,
        size_mb=size_mb,
    )
    return 0


def main() -> int:
    """Fetch WeatherNext 3 ensemble-mean runs, cropped to the trial-area box, one run per file.

    Any uncaught exception prints only its type name, because a message can carry an account name,
    the billing project, or the crop's shape.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=ARCHIVE_START,
        help="First init date, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        help="Last init date, YYYY-MM-DD. Default: the newest run found.",
    )
    parser.add_argument(
        "--init-hours", type=int, nargs="+", default=[0, 6, 12, 18], help="Init hours, UTC."
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Concurrent reads.")
    parser.add_argument("--max-external-gb", type=float, default=DEFAULT_MAX_EXTERNAL_GB)
    parser.add_argument("--dry-run", action="store_true", help="Print the estimate and exit.")
    arguments = parser.parse_args()
    try:
        return _run(arguments=arguments)
    except Exception as error:  # noqa: BLE001  # the message can carry the project or an account
        print(f"FAILED ({type(error).__name__})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
