"""Download whole GFS/GEFS runs from Dynamical.org's Zarr catalog, cropped to the trial-area box.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/841>, covering the "GFS and GEFS
whole runs" row: the forecast study (#810) needs every lead time of every run, from 2021-05 for GFS
and 2020-10 for GEFS. Both datasets are opened lazily through `dynamical_catalog.open`, the same
entry point `dynamical_data.ecmwf_ens.download` uses for production ECMWF ENS, then cropped by a
`.sel()` on `latitude`/`longitude` before any array chunk is requested. The box appears only in the
arguments to that one `.sel()` call in this process.

**No lead-time or ensemble-member subsetting is applied.** The output is every `init_time`, every
ensemble member (GEFS and ECMWF AIFS ENS only), every lead time, and the eight variables in
`VARIABLES`, for the grid cells inside the box.

**The script fetches and checkpoints one calendar month of `init_time` at a time.** Each month is
written to `_month_cache/` as soon as it lands, and a re-run skips every month already cached as
complete. A month counts as complete only if its last day is earlier than the newest `init_time` in
the store minus `PUBLICATION_LAG_DAYS`; any other month is written as `<month>.partial.parquet` and
re-fetched on the next run. The final file is built from this run's months with `scan_parquet` and
`sink_parquet`, so its peak memory does not depend on the length of the archive. One month of GEFS
(30 runs, 31 members, 181 lead times) held in memory needs a few GB of RAM, well within this
workstation's 61 GB, where a year would need tens of GB. Every month file records a hash of the
crop's grid cells in its parquet metadata, and the combine step refuses to mix months whose hash
differs.

**The Zarr stores are chunked far larger than the box, so the bytes transferred exceed the bytes
kept.** GFS stores 105 lead times by 121 by 121 grid cells per chunk, and GEFS stores 64 lead times
by 17 by 16 grid cells (all 31 members in one chunk). Every chunk the box touches is transferred
whole.

**Row counts, cell counts, and the crop's hash go only to the private lineage note and the parquet
metadata, never to stdout,** because they reveal the size of the trial-area box.

Run it with `uv run python studies/weather_downloads/fetch_dynamical_zarr.py --dataset
noaa-gfs-forecast`, `--dataset noaa-gefs-forecast-35-day`, `--dataset ecmwf-aifs-single-forecast`,
or `--dataset ecmwf-aifs-ens-forecast` (the last two land in `ECMWF-AIFS/` and `ECMWF-AIFS-ENS/`).
`--workers` sets how many months are fetched concurrently. Passing `--start-date` and `--end-date`
(both `YYYY-MM-DD`, inclusive) fetches only that window, into its own directory, for a trial run.
Then check the output with `validate_dynamical_zarr.py`.
"""

import argparse
import hashlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import dynamical_catalog
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import xarray as xr
from delta_store.precision import round_to_significand_bits
from lineage import write_lineage_note, write_readme
from paths import WEATHER_DOWNLOADS_DIR, load_trial_area_box

VARIABLES: Final[tuple[str, ...]] = (
    "downward_short_wave_radiation_flux_surface",
    "downward_long_wave_radiation_flux_surface",
    "temperature_2m",
    "wind_u_10m",
    "wind_v_10m",
    "wind_u_100m",
    "wind_v_100m",
    "pressure_surface",
)
"""The solar and wind fields both candidate studies need, plus surface pressure and 2 m temperature
for the physical checks every arm in `studies/beam_diffuse_split` runs."""

DATASETS: Final[dict[str, str]] = {
    "noaa-gfs-forecast": "GFS",
    "noaa-gefs-forecast-35-day": "GEFS",
    "ecmwf-aifs-single-forecast": "ECMWF-AIFS",
    "ecmwf-aifs-ens-forecast": "ECMWF-AIFS-ENS",
}

DEFAULT_WORKERS: Final[int] = 3
"""Months fetched concurrently. Each month is a separate set of Zarr chunk requests, so threads
overlap the network waits."""

KEEP_BITS: Final[int] = 13
"""Significand bits kept in every value column, as in production NWP storage."""

PUBLICATION_LAG_DAYS: Final[int] = 2
"""A month is complete once its last day is this many days older than the newest `init_time`."""

MAX_ATTEMPTS: Final[int] = 5
BACKOFF_SECONDS: Final[float] = 5.0

_BYTES_PER_MB: Final[float] = 1e6


def _cropped_dataset(*, dataset_id: str) -> xr.Dataset:
    """Open one Dynamical.org catalog entry and crop it to the trial-area box.

    Args:
        dataset_id: A key of `DATASETS`.

    Returns:
        The catalog's dataset, sliced to `VARIABLES` and to the box's lat/lon extent, still lazy.
    """
    box = load_trial_area_box()
    dataset = dynamical_catalog.open(dataset_id, chunks=None)[list(VARIABLES)]
    # Latitude is stored descending (90 to -90), so the slice bounds are given high-to-low.
    return dataset.sel(
        latitude=slice(box.lat_max, box.lat_min), longitude=slice(box.lon_min, box.lon_max)
    )


def _load_with_retries(*, dataset: xr.Dataset, month: str) -> xr.Dataset:
    """Load a lazy dataset, retrying with exponential backoff on any failure.

    A transient network error partway through a month would otherwise abandon the whole run. The
    first sleep is 10 s and each later one doubles. Each retry prints the month, the attempt
    number, and the exception's type name only, because an exception message can carry a request
    URL.

    Args:
        dataset: The lazy, cropped slice to load.
        month: The `YYYY-MM` label, for the retry log line.

    Returns:
        The same slice, in memory.

    Raises:
        Exception: The last error, after `MAX_ATTEMPTS` failed attempts.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return dataset.load()
        except Exception as error:
            if attempt == MAX_ATTEMPTS:
                raise
            print(
                f"{month}: load attempt {attempt} failed ({type(error).__name__}); retrying",
                flush=True,
            )
            time.sleep(BACKOFF_SECONDS * 2**attempt)
    raise AssertionError  # unreachable: the loop returns or raises


def _axis_column(*, values: np.ndarray, axis: int, shape: tuple[int, ...]) -> np.ndarray:
    """Broadcast a 1-D coordinate array along `axis` of a dense array and flatten it in C order."""
    view_shape = [1] * len(shape)
    view_shape[axis] = -1
    return np.broadcast_to(values.reshape(view_shape), shape).ravel()


def _to_long_frame(*, dataset: xr.Dataset) -> pl.DataFrame:
    """Flatten a loaded, cropped Dynamical.org dataset to a long frame with no coordinate column.

    One row per (init_time, [ensemble_member,] lead_time, lat_index, lon_index). `lat_index` and
    `lon_index` are the rank of each cell's latitude and longitude within the crop, ascending,
    replacing the grid's own coordinates, which locate a generator to within one grid cell.
    Coordinate columns use narrow dtypes (`Int8`, `Int16`, `Datetime`, `Duration`) so that a month
    of GEFS stays within a few GB. Value columns are `Float32` rounded to `KEEP_BITS` significand
    bits.

    Args:
        dataset: The already-cropped, already-loaded dataset.

    Returns:
        A long frame keyed by integer grid indices rather than coordinates.
    """
    dims = tuple(str(dim) for dim in dataset[VARIABLES[0]].dims)
    shape = tuple(dataset.sizes[dim] for dim in dims)
    columns: dict[str, np.ndarray] = {}
    for axis, dim in enumerate(dims):
        values = dataset[dim].to_numpy()
        if dim in ("latitude", "longitude"):
            # The rank of each coordinate within the crop, ascending.
            ranks = np.argsort(np.argsort(values)).astype(np.int16)
            columns[f"{dim[:3]}_index"] = _axis_column(values=ranks, axis=axis, shape=shape)
        elif dim == "ensemble_member":
            columns[dim] = _axis_column(values=values.astype(np.int8), axis=axis, shape=shape)
        else:
            columns[dim] = _axis_column(values=values, axis=axis, shape=shape)
    for variable in VARIABLES:
        columns[variable] = (
            dataset[variable].transpose(*dims).to_numpy().astype(np.float32).reshape(-1)
        )
    frame = pl.DataFrame(columns)
    return frame.with_columns(
        round_to_significand_bits(pl.col(variable), keep_bits=KEEP_BITS) for variable in VARIABLES
    )


def _write_grid_cells(*, dataset: xr.Dataset, path: Path) -> None:
    """Write the private lookup from `lat_index`/`lon_index` to the grid cell's coordinates.

    The file stays on the private data disk and is never printed, published, or quoted.
    """
    latitudes = dataset["latitude"].to_numpy()
    longitudes = dataset["longitude"].to_numpy()
    lat_ranks = np.argsort(np.argsort(latitudes)).astype(np.int16)
    lon_ranks = np.argsort(np.argsort(longitudes)).astype(np.int16)
    pl.DataFrame(
        {
            "lat_index": np.repeat(lat_ranks, len(lon_ranks)),
            "lon_index": np.tile(lon_ranks, len(lat_ranks)),
            "latitude": np.repeat(latitudes, len(lon_ranks)),
            "longitude": np.tile(longitudes, len(lat_ranks)),
        }
    ).write_parquet(path)


def _cell_fingerprint(*, dataset: xr.Dataset) -> dict[str, str]:
    """Return the crop's cell count and a hash of its cell coordinates, as parquet metadata.

    The values stay in the private parquet metadata and are never printed.
    """
    digest = hashlib.sha256()
    digest.update(dataset["latitude"].to_numpy().tobytes())
    digest.update(dataset["longitude"].to_numpy().tobytes())
    n_cells = dataset.sizes["latitude"] * dataset.sizes["longitude"]
    return {"cell_count": str(n_cells), "cell_hash": digest.hexdigest()}


def _months(*, init_times: np.ndarray) -> list[str]:
    """Return the distinct `YYYY-MM` labels of `init_times`, in order."""
    return sorted({str(np.datetime64(t, "M")) for t in init_times})


def _month_paths(*, month_cache_dir: Path, month: str) -> tuple[Path, Path]:
    """Return the (complete, partial) cache paths of one month."""
    return month_cache_dir / f"{month}.parquet", month_cache_dir / f"{month}.partial.parquet"


def _is_complete(*, month: str, newest_init_time: np.datetime64) -> bool:
    """Return whether every run of `month` should already be published."""
    next_month = np.datetime64(month, "M") + np.timedelta64(1, "M")
    last_day = next_month.astype("datetime64[D]") - np.timedelta64(1, "D")
    newest_day = newest_init_time.astype("datetime64[D]")
    return bool(last_day < newest_day - np.timedelta64(PUBLICATION_LAG_DAYS, "D"))


LEAD_STEP_NOTES: Final[dict[str, str]] = {
    "GFS": "GFS lead times are hourly to 120 h, then 3-hourly to 384 h.",
    "GEFS": "GEFS lead times are 3-hourly to 240 h, then 6-hourly to 840 h.",
    "ECMWF-AIFS": "AIFS Single lead times are 6-hourly from 0 h to 360 h.",
    "ECMWF-AIFS-ENS": "AIFS ENS lead times are 6-hourly from 0 h to 360 h.",
}

AVERAGING_NOTES: Final[dict[str, str]] = {
    "GFS": (
        "GFS radiation is the average since the last 6-hourly reset (00, 06, 12, 18 UTC), so a "
        "lead time labels the END of an averaging window of 1 to 6 hours."
    ),
    "GEFS": (
        "GEFS radiation is the average over the preceding 6-hour period (00, 06, 12, 18 UTC "
        "valid times) or 3-hour period (03, 09, 15, 21 UTC), and a lead time labels the END of "
        "that window."
    ),
    "ECMWF-AIFS": (
        "AIFS Single radiation is the average flux since the previous forecast step, so a lead "
        "time labels the END of a 6-hour averaging window (00, 06, 12, 18 UTC valid times)."
    ),
    "ECMWF-AIFS-ENS": (
        "AIFS ENS radiation is the average flux since the previous forecast step, so a lead time "
        "labels the END of a 6-hour averaging window (00, 06, 12, 18 UTC valid times)."
    ),
}

STEP_CHANGES: Final[frozenset[str]] = frozenset({"GFS", "GEFS"})
"""Labels whose lead-time axis changes step width part-way. The AIFS axes do not."""

EXTERNAL_DOCS: Final[dict[str, dict[str, str]]] = {
    "ECMWF-AIFS": {
        "ECMWF AIFS Single v1.1 paper": "https://gmd.copernicus.org/articles/19/4703/2026/",
        "ECMWF IFS Cycle 50r1 and AIFS v2 announcement": (
            "https://forum.ecmwf.int/t/confirmation-ifs-cycle-50r1-and-aifs-v2-joint-implementation-on-12-may-2026/14937"
        ),
    },
    "ECMWF-AIFS-ENS": {
        "Implementation of AIFS ENS v2": (
            "https://confluence.ecmwf.int/display/FCST/Implementation+of+AIFS+ENS+v2"
        ),
    },
}

_VERSION_READ_NOT_VERIFIED: Final[str] = (
    "The dates are read from ECMWF release pages, not verified in the data. A level shift in "
    "these fields at a version change was not tested."
)

VERSION_NOTES: Final[dict[str, list[str]]] = {
    "ECMWF-AIFS": [
        "Model versions inside this history: AIFS Single v1 became operational with the "
        "2025-02-25 06 UTC run, v1.1 on 2025-08-27, and v2 on 2026-05-12 (the same day as IFS "
        "Cycle 50r1). Runs before 2025-02-25 06 UTC pre-date the operational v1; which model "
        "version produced them was not checked. " + _VERSION_READ_NOT_VERIFIED,
        (
            "The store starts on 2024-04-01, but `downward_short_wave_radiation_flux_surface`, "
            "`downward_long_wave_radiation_flux_surface`, `wind_u_100m` and `wind_v_100m` are "
            "`NaN` in every run before the 2025-02-24 06 UTC run: the store holds no values "
            "for those fields before then. That is one day before the operational v1 date "
            "(2025-02-25 06 UTC) read from ECMWF's pages, so the store's start of these four "
            "fields does not coincide with that date. Those rows are kept as `NaN`; "
            "`validate_dynamical_zarr.py` treats exactly those `NaN`s as expected."
        ),
    ],
    "ECMWF-AIFS-ENS": [
        "Model versions inside this history: AIFS ENS v1 became operational on 2025-07-01 (the "
        "store starts on 2025-07-02) and v2 on 2026-05-12. " + _VERSION_READ_NOT_VERIFIED,
    ],
}

SOURCE_GAP_NOTES: Final[dict[str, str]] = {
    "GEFS": (
        "`validate_dynamical_zarr.py` on 2026-09-25 found `NaN` in nine months of the full "
        "2020-10 to 2026-09 run. Most are gaps in `wind_u_100m` and `wind_v_100m`: 2020-10 (1 "
        "init_time, from lead 8 d 15 h), 2020-11 (3, from 4 d 3 h), 2020-12 (2, from 8 d 21 h), "
        "2021-01 (1, from 9 d 3 h), 2021-02 (4, from 1 d 15 h), 2023-05 (1, from 16 d 6 h), "
        "2024-10 (1, 9 rows, from 34 d 18 h), and 2026-01 (8, from 16 d 6 h; the 2026-01-25 run "
        "only from 31 d 6 h and only for 6 members). In 2021-10 one init_time has 9 rows that "
        "are `NaN` in every variable except the 100 m winds, from lead 24 d 12 h. The 2026-01 "
        "gap was confirmed in the Dynamical.org store itself, so it is not a crop artefact. "
        "Missing values were kept as `NaN`, not masked or dropped."
    ),
    "ECMWF-AIFS": "Missing values were kept as `NaN`, not masked or dropped.",
    "ECMWF-AIFS-ENS": "Missing values were kept as `NaN`, not masked or dropped.",
    "GFS": (
        "`validate_dynamical_zarr.py` on 2026-09-25 found `NaN` in one month of the full "
        "2020-10 to 2026-09 run: 2022-11, on four init_times. The 2022-11-29 12:00 UTC run is "
        "`NaN` from lead 4 d 22 h (153 rows), and the 2022-11-29 18:00, 2022-11-30 00:00, and "
        "2022-11-30 06:00 UTC runs from lead 2 h (522, 612, and 477 rows), in every variable. "
        "Missing values were kept as `NaN`, not masked or dropped."
    ),
}
"""Source gaps found by validation of the full run, added to each README's gotchas."""


def _write_documentation(
    *,
    output_dir: Path,
    label: str,
    dataset_id: str,
    init_times: np.ndarray,
    is_window: bool,
    has_members: bool,
    used_paths: list[Path],
    fingerprint: dict[str, str],
    rows: int,
    size_mb: float,
) -> None:
    """Write the lineage note and the README next to the combined file.

    Args:
        output_dir: The product directory.
        label: `GFS` or `GEFS`.
        dataset_id: The Dynamical.org catalog key.
        init_times: The `init_time`s this run fetched.
        is_window: Whether a start or end date restricted the run.
        has_members: Whether the dataset has an `ensemble_member` dimension.
        used_paths: The month files combined.
        fingerprint: The crop's cell count and hash (private, lineage only).
        rows: Row count of the combined file (private, lineage only).
        size_mb: Size of the combined file, MB.
    """
    first_init = str(init_times.min())
    last_init = str(init_times.max())
    lineage_filename = "lineage.json"
    lead_step_note = LEAD_STEP_NOTES[label]
    averaging_note = AVERAGING_NOTES[label]
    init_scope = (
        f"every init_time from {first_init} to {last_init}"
        if is_window
        else f"every init_time in the store ({first_init} to {last_init})"
    )
    write_lineage_note(
        product_dir=output_dir,
        source_address=f"dynamical_catalog dataset '{dataset_id}'",
        request_description=(
            f"{label} whole runs, variables {', '.join(VARIABLES)}, {init_scope}"
            + (", every ensemble_member" if has_members else "")
            + ", every lead_time, cropped to the trial-area box via .sel() before any array chunk "
            "is requested"
        ),
        variables=list(VARIABLES),
        extra={
            "first_init_time": first_init,
            "last_init_time": last_init,
            "months_used": [path.name for path in used_paths],
            "rows": rows,
            "output_size_mb": round(size_mb, 2),
            "cell_count": fingerprint["cell_count"],
            "significand_bits_kept": KEEP_BITS,
            "note": f"{lead_step_note} {averaging_note}",
        },
        filename=lineage_filename,
    )
    write_readme(
        product_dir=output_dir,
        product_name=f"{label} whole runs from Dynamical.org",
        source_web_page="https://dynamical.org/catalog/",
        script_path="studies/weather_downloads/fetch_dynamical_zarr.py",
        lineage_filenames=[lineage_filename],
        columns={
            "init_time": "Timezone-naive (implicitly UTC) start time of the model run.",
            "ensemble_member": (
                "Ensemble member number (GEFS and ECMWF AIFS ENS only; 0 is the control run)."
            ),
            "lead_time": "Duration since init_time. The valid time is init_time + lead_time.",
            "lat_index": "Rank of the cell's latitude in the crop, ascending. Not a coordinate.",
            "lon_index": "Rank of the cell's longitude in the crop, ascending. Not a coordinate.",
            "downward_short_wave_radiation_flux_surface": "Downward shortwave flux at the "
            "surface, W/m2, averaged over a window ending at init_time + lead_time (see gotchas).",
            "downward_long_wave_radiation_flux_surface": "Downward longwave flux at the "
            "surface, W/m2, averaged over a window ending at init_time + lead_time (see gotchas).",
            "temperature_2m": "Air temperature at 2 m, degrees Celsius.",
            "wind_u_10m": "Eastward wind at 10 m, m/s.",
            "wind_v_10m": "Northward wind at 10 m, m/s.",
            "wind_u_100m": "Eastward wind at 100 m, m/s.",
            "wind_v_100m": "Northward wind at 100 m, m/s.",
            "pressure_surface": "Surface pressure, Pa.",
        },
        missing_value_convention=(
            "A missing value is `NaN` in a `Float32` value column. The averaged radiation fields "
            "have no value at lead_time 0, because no time has elapsed."
        ),
        gotchas=[
            averaging_note,
            lead_step_note
            + (
                " The step width changes inside the lead-time axis."
                if label in STEP_CHANGES
                else ""
            ),
            f"Value columns are rounded to {KEEP_BITS} significand bits (relative error 1.2e-4).",
            (
                "`_grid_cells.parquet` maps `lat_index` and `lon_index` to coordinates. It is "
                "private and must never be published."
            ),
            (
                "A month whose last day is less than "
                f"{PUBLICATION_LAG_DAYS} days older than the newest run in the store is cached as "
                "`<month>.partial.parquet` and re-fetched on the next run, so the newest runs of "
                "the final month may be missing."
            ),
            SOURCE_GAP_NOTES[label],
            *VERSION_NOTES.get(label, []),
        ],
        external_docs={
            f"Dynamical.org {dataset_id}": f"https://dynamical.org/catalog/{dataset_id}/",
            **EXTERNAL_DOCS.get(label, {}),
        },
    )


def main() -> int:
    """Fetch one Dynamical.org dataset, cropped to the trial-area box, one month at a time."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--start-date", help="First init date to fetch, YYYY-MM-DD (trial run).")
    parser.add_argument("--end-date", help="Last init date to fetch, YYYY-MM-DD (trial run).")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Concurrent months.")
    arguments = parser.parse_args()
    label = DATASETS[arguments.dataset]
    is_window = arguments.start_date is not None or arguments.end_date is not None

    full = _cropped_dataset(dataset_id=arguments.dataset)
    newest_init_time = full["init_time"].to_numpy().max()
    cropped = full.sel(init_time=slice(arguments.start_date, arguments.end_date))
    init_times = cropped["init_time"].to_numpy()
    months = _months(init_times=init_times)
    print(f"{label}: {len(months)} months to fetch, month by month")

    directory_name = (
        f"{label}_window_{arguments.start_date}_{arguments.end_date}" if is_window else label
    )
    output_dir = WEATHER_DOWNLOADS_DIR / directory_name
    month_cache_dir = output_dir / "_month_cache"
    month_cache_dir.mkdir(parents=True, exist_ok=True)
    _write_grid_cells(dataset=cropped, path=output_dir / "_grid_cells.parquet")
    fingerprint = _cell_fingerprint(dataset=cropped)

    def fetch_month(month: str) -> Path:
        """Fetch, or skip if already cached, one month; return the file holding it."""
        complete_path, partial_path = _month_paths(month_cache_dir=month_cache_dir, month=month)
        complete = _is_complete(month=month, newest_init_time=newest_init_time)
        if complete and complete_path.exists():
            print(f"{label} {month}: already cached, skipping", flush=True)
            return complete_path
        # A string bound is expanded by xarray to the whole month; a `np.datetime64` bound would
        # be an exact instant and drop later init_times on the last day.
        month_slice = _load_with_retries(
            dataset=cropped.sel(init_time=slice(month, month)), month=month
        )
        frame = _to_long_frame(dataset=month_slice)
        target = complete_path if complete else partial_path
        temporary = target.with_suffix(".parquet.tmp")
        frame.write_parquet(temporary, compression="zstd", metadata=fingerprint)
        temporary.rename(target)
        (partial_path if complete else complete_path).unlink(missing_ok=True)
        print(
            f"{label} {month}: fetched" + ("" if complete else " (partial, will be re-fetched)"),
            flush=True,
        )
        return target

    with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        used_paths = list(executor.map(fetch_month, months))

    for path in used_paths:
        stored = pl.read_parquet_metadata(path)
        if {key: stored.get(key) for key in fingerprint} != fingerprint:
            message = f"{path.name} was cropped to different grid cells; delete it and re-run"
            raise ValueError(message)
    output_path = output_dir / f"{label}.parquet"
    pl.scan_parquet(used_paths).sink_parquet(output_path, compression="zstd", metadata=fingerprint)
    rows = pq.ParquetFile(output_path).metadata.num_rows
    size_mb = output_path.stat().st_size / _BYTES_PER_MB
    print(f"{label}: wrote {output_path}")

    _write_documentation(
        output_dir=output_dir,
        label=label,
        dataset_id=arguments.dataset,
        init_times=init_times,
        is_window=is_window,
        has_members="ensemble_member" in cropped.sizes,
        used_paths=used_paths,
        fingerprint=fingerprint,
        rows=rows,
        size_mb=size_mb,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
