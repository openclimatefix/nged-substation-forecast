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
written to `_month_cache/<year>-<month>.parquet` as soon as it lands, and a re-run skips every month
already cached, except the current month, which is re-fetched because its newest runs may not have
been published. The final file is built from the cached months with `scan_parquet` and
`sink_parquet`, so its peak memory does not depend on the length of the archive. One month of GEFS
(30 runs, 31 members, 181 lead times) held in memory is under 1 GB, where a year would need tens of
GB.

**The Zarr stores are chunked far larger than the box, so the bytes transferred exceed the bytes
kept.** GFS stores 105 lead times by 121 by 121 grid cells per chunk, and GEFS stores 64 lead times
by 17 by 16 grid cells (all 31 members in one chunk). Every chunk the box touches is transferred
whole. `--measure` prints the bytes received during a fetch, for extrapolating the full run.

Run it with `uv run python studies/weather_downloads/fetch_dynamical_zarr.py --dataset
noaa-gfs-forecast` or `--dataset noaa-gefs-forecast-35-day`. Passing `--start-date` and `--end-date`
(both `YYYY-MM-DD`, inclusive) fetches only that window, into its own directory, for a trial run.
Then check the output with `validate_dynamical_zarr.py`.
"""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
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


def _received_bytes() -> int:
    """Return the bytes received so far on all network interfaces, from `/proc/net/dev`."""
    total = 0
    for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
        name, counters = line.split(":", maxsplit=1)
        if name.strip() != "lo":
            total += int(counters.split()[0])
    return total


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
        columns[variable] = dataset[variable].to_numpy().astype(np.float32).reshape(-1)
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


def _months(*, init_times: np.ndarray) -> list[str]:
    """Return the distinct `YYYY-MM` labels of `init_times`, in order."""
    return sorted({str(np.datetime64(t, "M")) for t in init_times})


def main() -> int:
    """Fetch one Dynamical.org dataset, cropped to the trial-area box, one month at a time."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--start-date", help="First init date to fetch, YYYY-MM-DD (trial run).")
    parser.add_argument("--end-date", help="Last init date to fetch, YYYY-MM-DD (trial run).")
    parser.add_argument(
        "--measure", action="store_true", help="Print seconds and bytes received per month."
    )
    arguments = parser.parse_args()
    label = DATASETS[arguments.dataset]
    is_window = arguments.start_date is not None or arguments.end_date is not None

    cropped = _cropped_dataset(dataset_id=arguments.dataset)
    init_slice = slice(arguments.start_date, arguments.end_date)
    cropped = cropped.sel(init_time=init_slice)
    months = _months(init_times=cropped["init_time"].to_numpy())
    print(f"{label}: {len(months)} months to fetch, month by month")

    directory_name = (
        f"{label}_window_{arguments.start_date}_{arguments.end_date}" if is_window else label
    )
    output_dir = WEATHER_DOWNLOADS_DIR / directory_name
    month_cache_dir = output_dir / "_month_cache"
    month_cache_dir.mkdir(parents=True, exist_ok=True)
    _write_grid_cells(dataset=cropped, path=output_dir / "_grid_cells.parquet")

    current_month = datetime.now(UTC).strftime("%Y-%m")
    for month in months:
        month_path = month_cache_dir / f"{month}.parquet"
        # The current month is still being ingested, so it is re-fetched on every run.
        if month_path.exists() and month != current_month:
            print(f"{label} {month}: already cached, skipping")
            continue
        started = monotonic()
        received_before = _received_bytes()
        # A string bound is expanded by xarray to the whole month; a `np.datetime64` bound would
        # be an exact instant and drop later init_times on the last day.
        month_slice = cropped.sel(init_time=slice(month, month)).load()
        frame = _to_long_frame(dataset=month_slice)
        partial = month_path.with_suffix(".parquet.partial")
        frame.write_parquet(partial, compression="zstd")
        partial.rename(month_path)
        message = f"{label} {month}: {frame.height} rows"
        if arguments.measure:
            received_mb = (_received_bytes() - received_before) / _BYTES_PER_MB
            message += f", {monotonic() - started:.0f} s, {received_mb:.0f} MB received"
            message += f", {month_path.stat().st_size / _BYTES_PER_MB:.1f} MB kept"
        print(message)

    cached_paths = sorted(month_cache_dir.glob("*.parquet"))
    output_path = output_dir / f"{label}.parquet"
    pl.scan_parquet(cached_paths).sink_parquet(output_path, compression="zstd")
    rows = pl.scan_parquet(output_path).select(pl.len()).collect().item()
    size_mb = output_path.stat().st_size / _BYTES_PER_MB
    print(f"{label}: wrote {rows} rows to {output_path}, {size_mb:.1f} MB")

    lineage_filename = "lineage.json"
    lead_step_note = (
        "GFS lead times are hourly to 120 h, then 3-hourly to 384 h."
        if label == "GFS"
        else "GEFS lead times are 3-hourly to 240 h, then 6-hourly to 840 h."
    )
    write_lineage_note(
        product_dir=output_dir,
        source_address=f"dynamical_catalog dataset '{arguments.dataset}'",
        request_description=(
            f"{label} whole runs, variables {', '.join(VARIABLES)}, every init_time"
            + (", every ensemble_member" if "ensemble_member" in cropped.sizes else "")
            + ", every lead_time, cropped to the trial-area box via .sel() before any array chunk "
            "is requested"
        ),
        variables=list(VARIABLES),
        extra={
            "months_cached": [path.stem for path in cached_paths],
            "rows": rows,
            "significand_bits_kept": KEEP_BITS,
            "note": lead_step_note
            + " The shortwave and longwave fields are averages over the preceding step, so a "
            "lead time labels the END of its averaging window.",
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
            "surface, W/m2, mean over the step ending at init_time + lead_time.",
            "downward_long_wave_radiation_flux_surface": "Downward longwave flux at the "
            "surface, W/m2, mean over the step ending at init_time + lead_time.",
            "temperature_2m": "Air temperature at 2 m, degrees Celsius.",
            "wind_u_10m": "Eastward wind at 10 m, m/s.",
            "wind_v_10m": "Northward wind at 10 m, m/s.",
            "wind_u_100m": "Eastward wind at 100 m, m/s.",
            "wind_v_100m": "Northward wind at 100 m, m/s.",
            "pressure_surface": "Surface pressure, Pa.",
        },
        missing_value_convention=(
            "A missing value is `NaN` in a `Float32` value column. The averaged radiation fields "
            "have no value at lead_time 0, because no step has elapsed."
        ),
        gotchas=[
            "The radiation fields are averages over the preceding step, not instantaneous values.",
            lead_step_note + " The step width changes inside the lead-time axis.",
            f"Value columns are rounded to {KEEP_BITS} significand bits (relative error 1.2e-4).",
            (
                "`_grid_cells.parquet` maps `lat_index` and `lon_index` to coordinates. It is "
                "private and must never be published."
            ),
            (
                "The current month is re-fetched on every run of the script, so its newest runs "
                "may be missing."
            ),
        ],
        external_docs={
            "Dynamical.org GFS forecast": "https://dynamical.org/catalog/noaa-gfs-forecast/",
            "Dynamical.org GEFS 35-day forecast": (
                "https://dynamical.org/catalog/noaa-gefs-forecast-35-day/"
            ),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
