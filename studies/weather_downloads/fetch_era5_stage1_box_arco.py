"""Read the ERA5 variables stage 1 needs, at every cell of the trial-area box plus a margin.

The staged probabilistic forecast trains a weather-to-power model ("stage 1") on ERA5 from 2019-09,
so it needs the hourly ERA5 variables that the ECMWF ENS `Nwp` contract carries, at the cells of
every series in scope. The values come from Google's public ARCO-ERA5 zarr store, read without
credentials, as in `fetch_era5_solar_arco.py`.

**The cell set is the smallest block of 0.25 degree cells enclosing the trial-area box, widened by
`MARGIN_CELLS` in every direction.** The box comes from the private list of series
(`studies.trial_area.load_trial_area_box`). No coordinate is printed, logged, or written to a
filename. The parquet files under `data/` hold the cells' coordinates, and `data/` is never
published.

Each hourly chunk of a single-level variable covers the whole globe, so a month of one variable
reads about 720 chunks and keeps the cells. The geopotential array stores all 37 pressure levels in
one chunk, so reading the 500 hPa level costs about 15 times a single-level variable. Each
variable-month is checkpointed to its own parquet file under `_month_cache/<variable>/`, written
through a `.partial` file, and a re-run skips every month whose file holds the expected rows.

ARCO carries no `expver` label. The `expver` column is the constant "0001" (final ERA5) throughout;
`validate_era5_stage1_box.py` compares the span with the held CDS tables, which do carry it.

Run commands (from `studies/weather_downloads/`):

    uv run --with zarr --with gcsfs python fetch_era5_stage1_box_arco.py --cell-count
    uv run --with zarr --with gcsfs python fetch_era5_stage1_box_arco.py --only-month 2025-06
    uv run --with zarr --with gcsfs python fetch_era5_stage1_box_arco.py
"""

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import polars as pl
import xarray as xr
import zarr
from era5_cells import box_cells
from fetch_era5_solar_arco import (
    ARCO_BUCKET_PATH,
    DEFAULT_THREADS,
    FIRST_HOUR,
    LAST_HOUR,
    READ_ATTEMPTS,
    month_hours,
    month_starts,
)
from lineage import write_lineage_note
from studies.sources import ERA5_PRODUCT_DIR
from studies.trial_area import load_trial_area_box

log = logging.getLogger(__name__)

OUTPUT_DIR: Final[Path] = ERA5_PRODUCT_DIR / "stage1_box"
GRID_STEP_DEG: Final[float] = 0.25
MARGIN_CELLS: Final[int] = 3
"""How many cells the block enclosing the trial-area box is widened by in every direction."""

EXPVER: Final[str] = "0001"
GEOPOTENTIAL_LEVEL_HPA: Final[int] = 500
ROUNDING_TOLERANCE: Final[float] = 1e-6
"""The share of a variable's upper limit by which float32 rounding may cross a limit.

ERA5 accumulations carry values such as -0.125 J m-2 where the true total is zero.
"""

VariableKindType = Literal["instantaneous", "accumulation"]
"""An accumulation is a total over the hour ending at the time stamp; every other field is a
snapshot at the time stamp."""


@dataclass(frozen=True)
class Stage1Variable:
    """One ERA5 variable stage 1 needs.

    Attributes:
        short_name: The ERA5 short name, which is also the parquet file stem.
        kind: Whether the field is an accumulation over the hour ending at the stamp or a snapshot.
        units: The unit ARCO documents for the array, which must match what is read.
        minimum: The physical lower limit in `units`.
        maximum: The physical upper limit in `units`.
        level_hpa: The pressure level kept, for a variable stored on pressure levels, else `None`.
    """

    short_name: str
    kind: VariableKindType
    units: str
    minimum: float
    maximum: float
    level_hpa: int | None = None


VARIABLES: Final[tuple[Stage1Variable, ...]] = (
    Stage1Variable("t2m", "instantaneous", "K", 200.0, 340.0),
    Stage1Variable("d2m", "instantaneous", "K", 200.0, 340.0),
    Stage1Variable("u10", "instantaneous", "m s**-1", -80.0, 80.0),
    Stage1Variable("v10", "instantaneous", "m s**-1", -80.0, 80.0),
    Stage1Variable("u100", "instantaneous", "m s**-1", -100.0, 100.0),
    Stage1Variable("v100", "instantaneous", "m s**-1", -100.0, 100.0),
    Stage1Variable("sp", "instantaneous", "Pa", 40_000.0, 110_000.0),
    Stage1Variable("msl", "instantaneous", "Pa", 90_000.0, 109_000.0),
    Stage1Variable("tcc", "instantaneous", "(0 - 1)", 0.0, 1.0),
    Stage1Variable("ptype", "instantaneous", "code table (4.201)", 0.0, 12.5),
    Stage1Variable("ssrd", "accumulation", "J m**-2", 0.0, 4.0e6),
    Stage1Variable("fdir", "accumulation", "J m**-2", 0.0, 4.0e6),
    Stage1Variable("strd", "accumulation", "J m**-2", 0.0, 2.5e6),
    Stage1Variable("tp", "accumulation", "m", 0.0, 0.2),
    Stage1Variable("z", "instantaneous", "m**2 s**-2", 40_000.0, 60_000.0, GEOPOTENTIAL_LEVEL_HPA),
)
"""The variables to fetch, cheapest first so that the geopotential is read last.

`ptype` is ERA5's precipitation type, a code from table 4.201 (0 none, 1 rain, 3 freezing rain,
5 snow, 6 wet snow, 7 mixture of rain and snow, 8 ice pellets, 12 freezing drizzle).
"""

VARIABLES_BY_NAME: Final[dict[str, Stage1Variable]] = {v.short_name: v for v in VARIABLES}


def cell_axes() -> tuple[list[int], list[int]]:
    """Return the box's latitude and longitude grid lines as whole quarter degrees.

    Returns:
        The latitudes, northernmost first (the order ARCO stores them), then the longitudes,
        westernmost first. Both are integers in units of 0.25 degrees, so neither is a coordinate
        that could be printed by accident.
    """
    box = load_trial_area_box()
    cells = box_cells(
        lat_min=box.lat_min, lat_max=box.lat_max, lon_min=box.lon_min, lon_max=box.lon_max
    )
    lat_q = sorted({cell[0] for cell in cells})
    lon_q = sorted({cell[1] for cell in cells})
    lat_q = list(range(lat_q[0] - MARGIN_CELLS, lat_q[-1] + MARGIN_CELLS + 1))
    lon_q = list(range(lon_q[0] - MARGIN_CELLS, lon_q[-1] + MARGIN_CELLS + 1))
    return lat_q[::-1], lon_q


class ArcoStore:
    """An open ARCO-ERA5 store with the box's array indices and the variable-name lookup."""

    def __init__(self, *, threads: int = DEFAULT_THREADS) -> None:
        """Open the store and index the cells, failing unless each lies exactly on the ARCO grid."""
        self.threads = threads
        options = {"token": "anon"}
        self.group = zarr.open_group(f"gs://{ARCO_BUCKET_PATH}", mode="r", storage_options=options)
        dataset = xr.open_zarr(f"gs://{ARCO_BUCKET_PATH}", storage_options=options, chunks=None)
        self.times: np.ndarray = dataset["time"].values
        # `geopotential_at_surface` shares the short name "z" with the pressure-level array.
        keys = [str(key) for key in dataset.data_vars if str(key) != "geopotential_at_surface"]
        self.names: dict[str, str] = {
            str(dataset[key].attrs.get("short_name", key)): key for key in keys
        }
        self.units: dict[str, str] = {
            str(dataset[key].attrs.get("short_name", key)): str(dataset[key].attrs.get("units", ""))
            for key in keys
        }
        self.levels: np.ndarray = dataset["level"].values
        self.latitude_q, self.longitude_q = cell_axes()
        self.lat_index = self._indices(dataset["latitude"].values, self.latitude_q)
        self.lon_index = self._indices(dataset["longitude"].values, self.longitude_q)

    @staticmethod
    def _indices(axis: np.ndarray, wanted_quarter_degrees: list[int]) -> list[int]:
        """Find each wanted grid line on `axis` (0 to 360 for longitude), failing unless exact."""
        indices: list[int] = []
        for quarter_degrees in wanted_quarter_degrees:
            value = quarter_degrees * GRID_STEP_DEG
            if axis.max() > 180.0:
                value %= 360.0
            nearest = int(np.argmin(np.abs(axis - value)))
            if abs(float(axis[nearest]) - value) > 1e-6:
                raise ValueError("a box cell is not on the ARCO grid")
            indices.append(nearest)
        return indices

    def time_position(self, *, hour: np.datetime64) -> int:
        """Return the array position of one hourly stamp, failing if the stamp is missing."""
        position = int(np.searchsorted(self.times, hour))
        if position >= len(self.times) or self.times[position] != hour:
            raise ValueError(f"hour {hour} is not in the ARCO time axis")
        return position

    def read_month(self, *, variable: Stage1Variable, hours: list[np.datetime64]) -> pl.DataFrame:
        """Read the box's cells for every hour in `hours` and return them as a tidy frame."""
        array = self.group[self.names[variable.short_name]]
        assert isinstance(array, zarr.Array)
        oindex: Any = array.oindex
        level_index = (
            int(np.flatnonzero(self.levels == variable.level_hpa)[0])
            if variable.level_hpa is not None
            else None
        )
        positions = [self.time_position(hour=hour) for hour in hours]

        def read_one(position: int) -> np.ndarray:
            for attempt in range(READ_ATTEMPTS):
                try:
                    if level_index is None:
                        return np.asarray(oindex[position, self.lat_index, self.lon_index])
                    return np.asarray(oindex[position, level_index, self.lat_index, self.lon_index])
                except Exception:
                    if attempt == READ_ATTEMPTS - 1:
                        raise
                    log.warning("retrying %s position %d", variable.short_name, position)
                    time.sleep(2**attempt)
            raise AssertionError("unreachable")

        with ThreadPoolExecutor(max_workers=self.threads) as pool:
            fields = np.stack(list(pool.map(read_one, positions)))
        if variable.short_name == "ptype":
            # ARCO stores the integer codes with float noise of about 5e-5, such as 1.0000458.
            fields = np.rint(fields)
        return fields_to_frame(
            fields=fields,
            hours=hours,
            latitude_q=self.latitude_q,
            longitude_q=self.longitude_q,
        )


def fields_to_frame(
    *,
    fields: np.ndarray,
    hours: list[np.datetime64],
    latitude_q: list[int],
    longitude_q: list[int],
) -> pl.DataFrame:
    """Turn an `(hours, latitudes, longitudes)` array into the tidy table the CDS files use.

    Args:
        fields: The values, shaped `(len(hours), len(latitude_q), len(longitude_q))`.
        hours: The hourly stamps, naive UTC.
        latitude_q: The latitudes in quarter degrees, in the order of the array's second axis.
        longitude_q: The longitudes in quarter degrees, in the order of the array's third axis.

    Returns:
        One row per hour and cell, with columns `time` (UTC `Datetime[us]`), `latitude` and
        `longitude` (`Float64` degrees), `value` (`Float32`), and `expver` (the constant "0001").
        The rows are ordered by hour, then latitude, then longitude, as the axes are.
    """
    cell_count = len(latitude_q) * len(longitude_q)
    n_hours = len(hours)
    times = np.repeat(np.array(hours, dtype="datetime64[us]"), cell_count)
    latitudes = np.tile(np.repeat(np.array(latitude_q) * GRID_STEP_DEG, len(longitude_q)), n_hours)
    longitudes = np.tile(np.array(longitude_q) * GRID_STEP_DEG, n_hours * len(latitude_q))
    return pl.DataFrame(
        {
            "time": pl.Series(times).dt.replace_time_zone("UTC"),
            "latitude": latitudes,
            "longitude": longitudes,
            "value": fields.reshape(-1).astype(np.float32),
            "expver": [EXPVER] * (n_hours * cell_count),
        }
    )


def month_file_is_valid(*, path: Path, expected_rows: int) -> bool:
    """Say whether a checkpoint file exists and holds exactly the expected number of rows."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        return pl.scan_parquet(path).select(pl.len()).collect().item() == expected_rows
    except pl.exceptions.PolarsError, OSError:
        return False


def check_month_values(*, frame: pl.DataFrame, variable: Stage1Variable) -> None:
    """Raise if a month holds a NaN, which a chunk never written reads back as, or a bad value."""
    nans = int(frame["value"].is_nan().sum())
    if nans:
        raise ValueError(f"{variable.short_name}: {nans} NaN values in one month")
    low, high = float(frame["value"].min()), float(frame["value"].max())  # ty: ignore[invalid-argument-type]
    slack = ROUNDING_TOLERANCE * variable.maximum
    if low < variable.minimum - slack or high > variable.maximum + slack:
        raise ValueError(f"{variable.short_name}: values {low}..{high} outside the physical limits")


def expected_hour_count() -> int:
    """Return how many hourly stamps the whole span holds, both ends included."""
    first = np.datetime64(FIRST_HOUR.replace(tzinfo=None), "h")
    last = np.datetime64(LAST_HOUR.replace(tzinfo=None), "h")
    return int((last - first).astype(int)) + 1


def checkpoint_variable_month(
    *, store: ArcoStore, variable: Stage1Variable, month_start: datetime, chunk_dir: Path
) -> Path:
    """Fetch one variable-month unless its checkpoint is complete; return the checkpoint path."""
    hours = month_hours(month_start=month_start)
    path = chunk_dir / f"{month_start:%Y-%m}.parquet"
    expected_rows = len(hours) * len(store.latitude_q) * len(store.longitude_q)
    if month_file_is_valid(path=path, expected_rows=expected_rows):
        return path
    started = time.monotonic()
    frame = store.read_month(variable=variable, hours=hours)
    check_month_values(frame=frame, variable=variable)
    partial = path.with_suffix(".parquet.partial")
    frame.write_parquet(partial)
    partial.replace(path)
    log.info(
        "%s %s: %d rows, %.0f s",
        variable.short_name,
        f"{month_start:%Y-%m}",
        frame.height,
        time.monotonic() - started,
    )
    return path


def check_table(
    *, frame: pl.DataFrame, variable: Stage1Variable, expected_rows: int, cell_count: int
) -> dict[str, object]:
    """Check one combined table and return the record, listing every hard-check problem found."""
    keys = frame.select(pl.struct("time", "latitude", "longitude").is_duplicated().sum()).item()
    hours = frame["time"].n_unique()
    values = frame["value"]
    problems: list[str] = []
    if frame.height != expected_rows:
        problems.append(f"{frame.height} rows, expected {expected_rows}")
    if keys:
        problems.append(f"{keys} duplicate (time, latitude, longitude) rows")
    if hours != expected_rows // cell_count:
        problems.append(f"{hours} distinct hours, expected {expected_rows // cell_count}")
    if int(values.is_nan().sum()) or values.null_count():
        problems.append("NaN or null values")
    return {
        "variable": variable.short_name,
        "rows": frame.height,
        "hours": hours,
        "cells": cell_count,
        "minimum": float(values.min()),  # ty: ignore[invalid-argument-type]
        "maximum": float(values.max()),  # ty: ignore[invalid-argument-type]
        "problems": problems,
    }


def fetch_variable(
    *, store: ArcoStore, variable: Stage1Variable, output_dir: Path, only_month: str | None
) -> dict[str, object] | None:
    """Fetch every month of one variable, combine them, and return the check record.

    Args:
        store: The open ARCO store.
        variable: The variable to fetch.
        output_dir: The folder holding the combined parquet files and `_month_cache/`.
        only_month: `YYYY-MM` to fetch just that month's checkpoint, or `None` for the whole span.

    Returns:
        The check record, or `None` when only one month was fetched and nothing was combined.
    """
    chunk_dir = output_dir / "_month_cache" / variable.short_name
    chunk_dir.mkdir(parents=True, exist_ok=True)
    months = month_starts(first=FIRST_HOUR, last=LAST_HOUR)
    if only_month:
        months = [m for m in months if f"{m:%Y-%m}" == only_month]
        if not months:
            raise SystemExit(f"{only_month} is outside the span")
    paths = [
        checkpoint_variable_month(
            store=store, variable=variable, month_start=month, chunk_dir=chunk_dir
        )
        for month in months
    ]
    if only_month:
        return None
    cell_count = len(store.latitude_q) * len(store.longitude_q)
    table = pl.concat([pl.read_parquet(path) for path in paths]).sort(
        "time", "latitude", "longitude", descending=[False, True, False]
    )
    record = check_table(
        frame=table,
        variable=variable,
        expected_rows=expected_hour_count() * cell_count,
        cell_count=cell_count,
    )
    combined_path = output_dir / f"{variable.short_name}.parquet"
    combined_path.unlink(missing_ok=True)  # never leave a file from an earlier run beside a failure
    if record["problems"]:
        return record
    partial = combined_path.with_suffix(".parquet.partial")
    table.write_parquet(partial)
    partial.replace(combined_path)
    write_lineage_note(
        product_dir=output_dir,
        source_address=f"gs://{ARCO_BUCKET_PATH}",
        request_description=(
            f"ARCO-ERA5 array {store.names[variable.short_name]}"
            + (f" at {variable.level_hpa} hPa" if variable.level_hpa else "")
            + f", hourly {FIRST_HOUR:%Y-%m-%d %H:%M} to {LAST_HOUR:%Y-%m-%d %H:%M} UTC, on the "
            f"{len(store.latitude_q)} x {len(store.longitude_q)} cell block enclosing the "
            f"trial-area box, widened by {MARGIN_CELLS} cells in every direction"
        ),
        variables=[variable.short_name],
        extra=record
        | {
            "arco_units": store.units[variable.short_name],
            "units_match_expected": store.units[variable.short_name] == variable.units,
            "kind": variable.kind,
            "span": [FIRST_HOUR.isoformat(), LAST_HOUR.isoformat()],
        },
        filename=f"lineage_{variable.short_name}.json",
    )
    return record


def write_manifest(*, output_dir: Path, records: list[dict[str, object]]) -> None:
    """Write `manifest.json`, listing every variable's check record."""
    manifest = {
        "written_at_utc": datetime.now(UTC).isoformat(),
        "span": [FIRST_HOUR.isoformat(), LAST_HOUR.isoformat()],
        "variables": records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")


def main() -> None:
    """Parse the command line and run the fetch."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variables", nargs="*", default=None, help="Fetch only these names.")
    parser.add_argument("--only-month", default=None, help="YYYY-MM: fetch just this month.")
    parser.add_argument(
        "--cell-count", action="store_true", help="Print the cell count and size estimate."
    )
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="Parallel reads.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.cell_count:
        latitude_q, longitude_q = cell_axes()
        print(
            f"{len(latitude_q)} x {len(longitude_q)} = {len(latitude_q) * len(longitude_q)} cells"
        )
        return
    selected = (
        [VARIABLES_BY_NAME[name] for name in args.variables] if args.variables else list(VARIABLES)
    )
    store = ArcoStore(threads=args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    failures: list[str] = []
    for variable in selected:
        try:
            record = fetch_variable(
                store=store,
                variable=variable,
                output_dir=args.output_dir,
                only_month=args.only_month,
            )
        except Exception:
            log.exception("=== FAILED %s", variable.short_name)
            failures.append(variable.short_name)
            continue
        if record is None:
            continue
        records.append(record)
        log.info("=== DONE %s problems=%s", variable.short_name, record["problems"])
        if record["problems"]:
            failures.append(variable.short_name)
    if records:
        write_manifest(output_dir=args.output_dir, records=records)
    if failures:
        raise SystemExit(f"FAILED: {failures}")


if __name__ == "__main__":
    main()
