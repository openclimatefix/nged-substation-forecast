"""Download whole GFS/GEFS runs from Dynamical.org's Zarr catalog, cropped to the trial-area box.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/841>, covering the "GFS and GEFS
whole runs" row: the forecast study (#810) needs every lead time of every run, from 2021-05 for GFS
and 2020-10 for GEFS. Both datasets are opened lazily through `dynamical_catalog.open`, the same
entry point `dynamical_data.ecmwf_ens.download` uses for production ECMWF ENS, then cropped by a
`.sel()` on `latitude`/`longitude` before any array chunk is requested. The box appears only in the
arguments to that one `.sel()` call in this process.

**No lead-time or ensemble-member subsetting is applied.** The output is every `init_time`, every
ensemble member (GEFS only), every lead time, and the eight variables in `VARIABLES`, for the grid
cells inside the box.

**The script fetches and checkpoints one calendar month of `init_time` at a time.** Each month is
written to `_month_cache/` as soon as it lands, and a re-run skips every month already cached as
complete. A month counts as complete only if its last day is earlier than the newest `init_time` in
the store minus `PUBLICATION_LAG_DAYS`; any other month is written as `<month>.partial.parquet` and
re-fetched on the next run. The final file is built from this run's months with `scan_parquet` and
`sink_parquet`, so its peak memory does not depend on the length of the archive. One month of GEFS
(30 runs, 31 members, 181 lead times) held in memory is under 1 GB, where a year would need tens of
GB. Every month file records a hash of the crop's grid cells in its parquet metadata, and the
combine step refuses to mix months whose hash differs.

**The Zarr stores are chunked far larger than the box, so the bytes transferred exceed the bytes
kept.** GFS stores 105 lead times by 121 by 121 grid cells per chunk, and GEFS stores 64 lead times
by 17 by 16 grid cells (all 31 members in one chunk). Every chunk the box touches is transferred
whole.

**Row counts, cell counts, and the crop's hash go only to the private lineage note and the parquet
metadata, never to stdout,** because they reveal the size of the trial-area box.

Run it with `uv run python studies/weather_downloads/fetch_dynamical_zarr.py --dataset
noaa-gfs-forecast` or `--dataset noaa-gefs-forecast-35-day`. Passing `--start-date` and `--end-date`
(both `YYYY-MM-DD`, inclusive) fetches only that window, into its own directory, for a trial run.
Then check the output with `validate_dynamical_zarr.py`.
"""

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Final

import dynamical_catalog
import numpy as np
import polars as pl
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
}

KEEP_BITS: Final[int] = 13
"""Significand bits kept in every value column, as in production NWP storage."""

PUBLICATION_LAG_DAYS: Final[int] = 2
"""A month is complete once its last day is this many days older than the newest `init_time`."""

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
    of GEFS stays under 1 GB. Value columns are `Float32` rounded to `KEEP_BITS` significand bits.

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
    lead_step_note = (
        "GFS lead times are hourly to 120 h, then 3-hourly to 384 h."
        if label == "GFS"
        else "GEFS lead times are 3-hourly to 240 h, then 6-hourly to 840 h."
    )
    averaging_note = (
        "GFS radiation is the average since the last 6-hourly reset (00, 06, 12, 18 UTC), so a "
        "lead time labels the END of an averaging window of 1 to 6 hours."
        if label == "GFS"
        else "GEFS radiation is the average over the preceding 6-hour period (00, 06, 12, 18 UTC "
        "valid times) or 3-hour period (03, 09, 15, 21 UTC), and a lead time labels the END of "
        "that window."
    )
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
            "ensemble_member": "Ensemble member number (GEFS only; 0 is the control run).",
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
            f"{lead_step_note} The step width changes inside the lead-time axis.",
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
        ],
        external_docs={
            "Dynamical.org GFS forecast": "https://dynamical.org/catalog/noaa-gfs-forecast/",
            "Dynamical.org GEFS 35-day forecast": (
                "https://dynamical.org/catalog/noaa-gefs-forecast-35-day/"
            ),
        },
    )


def main() -> int:
    """Fetch one Dynamical.org dataset, cropped to the trial-area box, one month at a time."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--start-date", help="First init date to fetch, YYYY-MM-DD (trial run).")
    parser.add_argument("--end-date", help="Last init date to fetch, YYYY-MM-DD (trial run).")
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

    used_paths: list[Path] = []
    for month in months:
        complete_path, partial_path = _month_paths(month_cache_dir=month_cache_dir, month=month)
        complete = _is_complete(month=month, newest_init_time=newest_init_time)
        if complete and complete_path.exists():
            print(f"{label} {month}: already cached, skipping")
            used_paths.append(complete_path)
            continue
        # A string bound is expanded by xarray to the whole month; a `np.datetime64` bound would
        # be an exact instant and drop later init_times on the last day.
        month_slice = cropped.sel(init_time=slice(month, month)).load()
        frame = _to_long_frame(dataset=month_slice)
        target = complete_path if complete else partial_path
        temporary = target.with_suffix(".parquet.tmp")
        frame.write_parquet(temporary, compression="zstd", metadata=fingerprint)
        temporary.rename(target)
        (partial_path if complete else complete_path).unlink(missing_ok=True)
        used_paths.append(target)
        print(f"{label} {month}: fetched" + ("" if complete else " (partial, will be re-fetched)"))

    for path in used_paths:
        stored = pl.read_parquet_metadata(path)
        if {key: stored.get(key) for key in fingerprint} != fingerprint:
            message = f"{path.name} was cropped to different grid cells; delete it and re-run"
            raise ValueError(message)
    output_path = output_dir / f"{label}.parquet"
    pl.scan_parquet(used_paths).sink_parquet(output_path, compression="zstd", metadata=fingerprint)
    rows = pl.scan_parquet(output_path).select(pl.len()).collect().item()
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
