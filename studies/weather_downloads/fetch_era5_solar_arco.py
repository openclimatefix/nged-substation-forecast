"""Read the ERA5 solar variables for the 20-cell box from Google's ARCO-ERA5 zarr store.

ARCO-ERA5 is a copy of ERA5 on a public Google Cloud Storage bucket, read without credentials.
Each hourly chunk covers the whole globe, so a month of one variable reads about 720 chunks of
1 to 3 MB each and keeps 20 cells. Each month is checkpointed to its own parquet file under
`_arco_chunks/<variable>/`, and a re-run skips every month whose file holds the expected rows.

ARCO carries no `expver` label. The `expver` column of every table written here is the constant
"0001" (final ERA5), and the span ends on 2026-07-31 23:00 UTC, inside the final period.

Run commands (from `studies/weather_downloads/`):

    uv run --with zarr --with gcsfs python fetch_era5_solar_arco.py --check-month 2025-06
    uv run --with zarr --with gcsfs python fetch_era5_solar_arco.py --tier tier1b
    uv run --with zarr --with gcsfs python fetch_era5_solar_arco.py --tier tier2
    uv run --with zarr --with gcsfs python fetch_era5_solar_arco.py --tier tier3
"""

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import gcsfs
import numpy as np
import polars as pl
import xarray as xr
import zarr
from era5_solar_variables import TIERS, VARIABLES_BY_NAME, TierType, Variable, variables_in
from studies.era5_grid import GRID_LATITUDES, GRID_LONGITUDES
from studies.sources import ERA5_SOLAR_VARIABLES_DIR

log = logging.getLogger(__name__)

ARCO_BUCKET_PATH: Final[str] = "gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
FIRST_HOUR: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
LAST_HOUR: Final[datetime] = datetime(2026, 7, 31, 23, tzinfo=UTC)
"""The first and last hourly stamps read. The last is the final hour of the last final month."""

EXPVER: Final[str] = "0001"
CELL_COUNT: Final[int] = len(GRID_LATITUDES) * len(GRID_LONGITUDES)
DEFAULT_THREADS: Final[int] = 16
READ_ATTEMPTS: Final[int] = 5
EXISTENCE_SAMPLE_STEP: Final[int] = 24
"""Every this-many-th hour of a month is checked for a stored chunk when NaN is a valid value."""
ROUNDING_TOLERANCE: Final[float] = 1e-6
"""The share of a variable's maximum by which float32 rounding may cross a limit."""

CHECK_VARIABLES: Final[tuple[str, ...]] = (
    "tcc", "lcc", "mcc", "hcc", "ssrdc", "cdir", "strd", "sd", "cape", "fal", "asn",
)  # fmt: skip
"""Variables with a held CDS table for 2025-06 in the main folder or `_pilot/`."""


def month_starts(*, first: datetime, last: datetime) -> list[datetime]:
    """List the first instant of every calendar month from `first`'s month to `last`'s month."""
    months: list[datetime] = []
    year, month = first.year, first.month
    while (year, month) <= (last.year, last.month):
        months.append(datetime(year, month, 1, tzinfo=UTC))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def month_hours(*, month_start: datetime) -> list[np.datetime64]:
    """List the hourly stamps of one month that fall inside the span."""
    next_year, next_month = (
        (month_start.year + 1, 1)
        if month_start.month == 12
        else (month_start.year, month_start.month + 1)
    )
    start = max(month_start, FIRST_HOUR)
    stop = min(datetime(next_year, next_month, 1, tzinfo=UTC), LAST_HOUR + timedelta(hours=1))
    return list(
        np.arange(
            np.datetime64(start.replace(tzinfo=None), "h"),
            np.datetime64(stop.replace(tzinfo=None), "h"),
            np.timedelta64(1, "h"),
        )
    )


class ArcoStore:
    """An open ARCO-ERA5 store with the cell indices and the variable-name lookup."""

    def __init__(self, *, threads: int = DEFAULT_THREADS) -> None:
        """Open the store, read the final-ERA5 end date from its attributes, and index the cells."""
        self.threads = threads
        self.filesystem = gcsfs.GCSFileSystem(token="anon")
        options = {"token": "anon"}
        self.group = zarr.open_group(f"gs://{ARCO_BUCKET_PATH}", mode="r", storage_options=options)
        dataset = xr.open_zarr(f"gs://{ARCO_BUCKET_PATH}", storage_options=options, chunks=None)
        self.times: np.ndarray = dataset["time"].values
        self.names: dict[str, str] = {
            str(dataset[key].attrs.get("short_name", key)): str(key) for key in dataset.data_vars
        }
        self.units: dict[str, str] = {
            str(dataset[key].attrs.get("short_name", key)): str(dataset[key].attrs.get("units", ""))
            for key in dataset.data_vars
        }
        # ARCO's `valid_time_stop` ends final ERA5 on 2026-06-30, but July 2026 matched the final
        # (expver 0001) CDS tables exactly for tcc, lcc, mcc and hcc, so the span counts as final.
        self.final_stop = np.datetime64(LAST_HOUR.replace(tzinfo=None), "h")
        self.lat_index = self._indices(dataset["latitude"].values, GRID_LATITUDES)
        self.lon_index = self._indices(
            dataset["longitude"].values, tuple(x % 360 for x in GRID_LONGITUDES)
        )

    @staticmethod
    def _indices(axis: np.ndarray, wanted: tuple[float, ...]) -> list[int]:
        """Find each wanted coordinate on `axis`, failing unless it matches exactly."""
        indices: list[int] = []
        for value in wanted:
            nearest = int(np.argmin(np.abs(axis - value)))
            if abs(float(axis[nearest]) - value) > 1e-6:
                raise ValueError(f"coordinate {value} is not on the ARCO grid")
            indices.append(nearest)
        return indices

    def time_position(self, *, hour: np.datetime64) -> int:
        """Return the array position of one hourly stamp, failing if the stamp is missing."""
        position = int(np.searchsorted(self.times, hour))
        if position >= len(self.times) or self.times[position] != hour:
            raise ValueError(f"hour {hour} is not in the ARCO time axis")
        return position

    def check_chunks_exist(self, *, short_name: str, hours: list[np.datetime64]) -> None:
        """Raise if a sampled hour has no stored chunk, which would read back as all NaN."""
        name = self.names[short_name]
        array = self.group[name]
        assert isinstance(array, zarr.Array)
        for hour in hours[::EXISTENCE_SAMPLE_STEP]:
            key = array.metadata.encode_chunk_key((self.time_position(hour=hour), 0, 0))
            if not self.filesystem.exists(f"{ARCO_BUCKET_PATH}/{name}/{key}"):
                raise ValueError(f"{short_name}: no stored chunk for {hour}")

    def read_month(self, *, short_name: str, hours: list[np.datetime64]) -> pl.DataFrame:
        """Read the 20 cells for every hour in `hours` and return them as a tidy frame.

        Hours after the end of the final ERA5 period are labelled `expver` "0005" (ERA5T).
        """
        array = self.group[self.names[short_name]]
        assert isinstance(array, zarr.Array)
        oindex: Any = array.oindex
        positions = [self.time_position(hour=hour) for hour in hours]

        def read_one(position: int) -> np.ndarray:
            for attempt in range(READ_ATTEMPTS):
                try:
                    return np.asarray(oindex[position, self.lat_index, self.lon_index])
                except Exception:
                    if attempt == READ_ATTEMPTS - 1:
                        raise
                    log.warning("retrying %s position %d", short_name, position, exc_info=True)
                    time.sleep(2**attempt)
            raise AssertionError("unreachable")

        with ThreadPoolExecutor(max_workers=self.threads) as pool:
            fields = np.stack(list(pool.map(read_one, positions)))
        frame = fields_to_frame(fields=fields, hours=hours)
        late = [hour > self.final_stop for hour in hours]
        if any(late):
            labels = np.repeat(np.where(late, "0005", EXPVER), CELL_COUNT)
            frame = frame.with_columns(expver=pl.Series(labels))
        return frame


def fields_to_frame(*, fields: np.ndarray, hours: list[np.datetime64]) -> pl.DataFrame:
    """Turn an `(hours, latitudes, longitudes)` array into the tidy table the CDS files use."""
    n_hours = len(hours)
    times = np.repeat(np.array(hours, dtype="datetime64[us]"), CELL_COUNT)
    latitudes = np.tile(np.repeat(np.array(GRID_LATITUDES), len(GRID_LONGITUDES)), n_hours)
    longitudes = np.tile(np.array(GRID_LONGITUDES), n_hours * len(GRID_LATITUDES))
    return pl.DataFrame(
        {
            "time": pl.Series(times).dt.replace_time_zone("UTC"),
            "latitude": latitudes,
            "longitude": longitudes,
            "value": fields.reshape(-1).astype(np.float32),
            "expver": [EXPVER] * (n_hours * CELL_COUNT),
        }
    )


def check_month_nans(*, frame: pl.DataFrame, variable: Variable) -> None:
    """Raise if a variable that is never missing has NaN values in one month.

    ARCO returns NaN, not an error, for a chunk that was never written. Variables that are NaN
    where the quantity is undefined (`cbh`, `cin`) are checked by `ArcoStore.check_chunks_exist`.
    """
    nans = int(frame["value"].is_nan().sum())
    if nans and not variable.nan_means_no_cloud:
        raise ValueError(f"{variable.short_name}: {nans} NaN values in one month")


def month_file_is_valid(*, path: Path, expected_rows: int) -> bool:
    """Say whether a checkpoint file exists and holds exactly the expected number of rows."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        return pl.scan_parquet(path).select(pl.len()).collect().item() == expected_rows
    except pl.exceptions.PolarsError, OSError:
        return False


def check_table(
    *, frame: pl.DataFrame, variable: Variable, expected_rows: int
) -> dict[str, object]:
    """Check one finished table and return the record; raise if any hard check fails."""
    duplicates = frame.select(
        pl.struct("time", "latitude", "longitude").is_duplicated().sum()
    ).item()
    hours = frame["time"].n_unique()
    expected_hours = expected_rows // CELL_COUNT
    values = frame["value"].to_numpy()
    minimum = float(np.nanmin(values)) if not np.isnan(values).all() else None
    nans = int(frame["value"].is_nan().sum()) + int(frame["value"].null_count())
    problems: list[str] = []
    if frame.height != expected_rows:
        problems.append(f"{frame.height} rows, expected {expected_rows}")
    if duplicates:
        problems.append(f"{duplicates} duplicate (time, latitude, longitude) rows")
    if hours != expected_hours:
        problems.append(f"{hours} distinct hours, expected {expected_hours}")
    maximum = float(np.nanmax(values)) if not np.isnan(values).all() else None
    tolerance = ROUNDING_TOLERANCE * abs(maximum) if maximum is not None else 0.0
    lower = 0.0 if variable.kind == "accumulation" else variable.minimum
    if lower is not None and minimum is not None and minimum < lower - tolerance:
        problems.append(f"minimum {minimum} below the limit {lower}")
    if (
        variable.maximum is not None
        and maximum is not None
        and maximum > variable.maximum + tolerance
    ):
        problems.append(f"maximum {maximum} above physical limit {variable.maximum}")
    if nans and not variable.nan_means_no_cloud:
        problems.append(f"{nans} NaN values")
    return {
        "variable": variable.short_name,
        "rows": frame.height,
        "hours": hours,
        "nan_values": nans,
        "minimum": minimum,
        "maximum": maximum,
        "problems": problems,
    }


def fetch_variable(*, store: ArcoStore, variable: Variable, output_dir: Path) -> dict[str, object]:
    """Fetch every month of one variable, assemble its parquet, and return the check record."""
    chunk_dir = output_dir / "_arco_chunks" / variable.short_name
    chunk_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pl.DataFrame] = []
    for month_start in month_starts(first=FIRST_HOUR, last=LAST_HOUR):
        hours = month_hours(month_start=month_start)
        path = chunk_dir / f"{month_start:%Y-%m}.parquet"
        if not month_file_is_valid(path=path, expected_rows=len(hours) * CELL_COUNT):
            started = time.monotonic()
            frame = store.read_month(short_name=variable.short_name, hours=hours)
            check_month_nans(frame=frame, variable=variable)
            if variable.nan_means_no_cloud:
                store.check_chunks_exist(short_name=variable.short_name, hours=hours)
            frame.write_parquet(path.with_suffix(".partial"))
            path.with_suffix(".partial").replace(path)
            log.info(
                "%s %s: %d rows, %.0f s",
                variable.short_name,
                f"{month_start:%Y-%m}",
                frame.height,
                time.monotonic() - started,
            )
        frames.append(pl.read_parquet(path))
    table = pl.concat(frames).sort("time", "latitude", "longitude", descending=[False, True, False])
    expected_rows = (
        int(
            (
                np.datetime64(LAST_HOUR.replace(tzinfo=None), "h")
                - np.datetime64(FIRST_HOUR.replace(tzinfo=None), "h")
            ).astype(int)
            + 1
        )
        * CELL_COUNT
    )
    record = check_table(frame=table, variable=variable, expected_rows=expected_rows)
    if not record["problems"]:
        table.write_parquet(output_dir / f"{variable.short_name}.parquet")
    record["source"] = (
        f"ARCO-ERA5, gs://{ARCO_BUCKET_PATH}, array {store.names[variable.short_name]}"
    )
    record["arco_units"] = store.units[variable.short_name]
    record["expver_note"] = (
        "ARCO has no expver; every hour is labelled 0001 (final ERA5) because the whole span "
        "matched the final CDS tables where checked"
    )
    record["expver_counts"] = table["expver"].value_counts().to_dicts()
    record["units_match_cds_documentation"] = store.units[variable.short_name] == (
        variable.expected_units
    )
    record["span"] = [FIRST_HOUR.isoformat(), LAST_HOUR.isoformat()]
    (output_dir / f"{variable.short_name}.arco_lineage.json").write_text(
        json.dumps(record, indent=2, default=str) + "\n"
    )
    return record


def check_month(*, store: ArcoStore, month: str, output_dir: Path) -> dict[str, object]:
    """Compare ARCO with the held CDS files for one month, and with one-hour shifts."""
    year, month_number = (int(part) for part in month.split("-"))
    month_start = datetime(year, month_number, 1, tzinfo=UTC)
    hours = month_hours(month_start=month_start)
    report: dict[str, object] = {}
    for short_name in CHECK_VARIABLES:
        held_path = output_dir / f"{short_name}.parquet"
        if not held_path.is_file():
            held_path = output_dir / "_pilot" / f"{short_name}.parquet"
        if not held_path.is_file() or short_name not in store.names:
            report[short_name] = "held file or ARCO variable missing"
            continue
        arco = store.read_month(short_name=short_name, hours=hours)
        held = pl.read_parquet(held_path).filter(
            pl.col("time").is_between(
                pl.lit(month_start), pl.lit(month_start).dt.offset_by("1mo"), closed="left"
            )
        )
        keys = ["time", "latitude", "longitude"]
        joined = arco.join(held, on=keys, suffix="_cds")
        difference = (joined["value"] - joined["value_cds"]).abs()
        shifts: dict[str, float | None] = {}
        for shift in (-1, 1):
            shifted = arco.with_columns(pl.col("time").dt.offset_by(f"{shift}h")).join(
                held, on=keys, suffix="_cds"
            )
            shifts[f"{shift:+d}h"] = float(
                np.mean(np.abs(shifted["value"].to_numpy() - shifted["value_cds"].to_numpy()))
            )
        report[short_name] = {
            "rows_compared": joined.height,
            "rows_expected": len(hours) * CELL_COUNT,
            "max_abs_diff": difference.max(),
            "mean_abs_diff": difference.mean(),
            "mean_abs_cds": joined["value_cds"].abs().mean(),
            "mean_abs_diff_if_shifted": shifts,
        }
    return report


def main() -> None:
    """Parse the command line and run the check or the fetch."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tier", choices=[t for t in TIERS if t != "tier1a"], default=None)
    parser.add_argument(
        "--variables", nargs="*", default=None, help="Fetch only these short names."
    )
    parser.add_argument(
        "--check-month", default=None, help="YYYY-MM to compare with held CDS files."
    )
    parser.add_argument(
        "--threads", type=int, default=DEFAULT_THREADS, help="Parallel chunk reads."
    )
    parser.add_argument("--output-root", type=Path, default=ERA5_SOLAR_VARIABLES_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    store = ArcoStore(threads=args.threads)
    if args.check_month:
        print(
            json.dumps(
                check_month(store=store, month=args.check_month, output_dir=args.output_root),
                indent=2,
                default=str,
            )
        )
        return
    selected: list[Variable]
    if args.variables:
        selected = [VARIABLES_BY_NAME[name] for name in args.variables]
        held_by_cds = [v.short_name for v in selected if v.tier == "tier1a"]
        if held_by_cds:
            raise SystemExit(f"refusing to overwrite the held CDS tables: {held_by_cds}")
    else:
        tier: TierType = args.tier or "tier1b"
        selected = variables_in(tiers=[tier])
    failures: list[str] = []
    for variable in selected:
        if variable.short_name not in store.names:
            log.error("%s is missing from ARCO", variable.short_name)
            failures.append(variable.short_name)
            continue
        try:
            record = fetch_variable(store=store, variable=variable, output_dir=args.output_root)
        except Exception:
            log.exception("=== FAILED %s", variable.short_name)
            failures.append(variable.short_name)
            continue
        log.info("=== DONE %s problems=%s", variable.short_name, record["problems"])
        if record["problems"]:
            failures.append(variable.short_name)
    if failures:
        raise SystemExit(f"FAILED: {failures}")


if __name__ == "__main__":
    main()
