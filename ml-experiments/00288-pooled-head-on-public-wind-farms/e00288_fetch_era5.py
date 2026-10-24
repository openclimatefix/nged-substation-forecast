"""Read ERA5 wind at the experiment 00288 farms' cells from Google's ARCO-ERA5 store.

ARCO-ERA5 is a copy of ERA5 on a public Google Cloud Storage bucket, read without credentials. Each
hourly chunk covers the whole globe (1 to 3 MB), so a month of one variable reads about 720 chunks
and keeps a few dozen cells. Each month of each variable is written atomically to its own parquet
file under `_chunks/<variable>/`, and a re-run skips the months already on disk. The span is
2019-09-01 to 2024-03-31, the base models' training window (`BASE_END` in the base script).

The cells are the 0.25 degree ERA5 grid points nearest each candidate farm in
`work/farm_candidates.parquet`. Cell coordinates stay in the private data folder and are never
printed. The combined file is `era5_farm_cells.parquet`: `cell_id`, `time`, and the four variables
`u10`, `v10`, `u100`, `v100` (m/s); `farm_cells.parquet` maps each BMU to its
`cell_id`.

    uv run --no-sync python ml-experiments/00288-*/e00288_fetch_era5.py [--month 2023-01]
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import Final

import numpy as np
import polars as pl
import xarray as xr
import zarr
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

WORK_DIR: Final = ML_EXPERIMENTS_DATA_DIR / "00288-pooled-head-on-public-wind-farms" / "work"
OUT_DIR: Final = REPO_DATA_DIR / "studies/downloads/reanalysis/ERA5/farm_cells_00288"
BUCKET: Final[str] = "gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
FIRST_HOUR: Final = datetime(2019, 9, 1, tzinfo=UTC)
LAST_HOUR: Final = datetime(2024, 3, 31, 23, tzinfo=UTC)
VARIABLES: Final[dict[str, str]] = {
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "u100": "100m_u_component_of_wind",
    "v100": "100m_v_component_of_wind",
}
THREADS: Final[int] = 16
ATTEMPTS: Final[int] = 5


def farm_cells() -> pl.DataFrame:
    """Return each BMU's nearest cell: `cell_id` and its `lat_q`, `lon_q` in quarter degrees."""
    farms = pl.read_parquet(WORK_DIR / "farm_candidates.parquet").filter(pl.col("in_ens"))
    cells = farms.select(
        "elexon_bmu_id",
        lat_q=(pl.col("lat") * 4).round().cast(pl.Int32),
        lon_q=(pl.col("lon") * 4).round().cast(pl.Int32),
    )
    ids = cells.select("lat_q", "lon_q").unique().sort("lat_q", "lon_q").with_row_index("cell_id")
    return cells.join(ids, on=["lat_q", "lon_q"]).select(
        "elexon_bmu_id", "cell_id", "lat_q", "lon_q"
    )


def month_starts() -> list[datetime]:
    """Return the first instant of each calendar month from `FIRST_HOUR` to `LAST_HOUR`."""
    out, year, month = [], FIRST_HOUR.year, FIRST_HOUR.month
    while (current := datetime(year, month, 1, tzinfo=UTC)) <= LAST_HOUR:
        out.append(current)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def read_month(
    *, array: zarr.Array, positions: list[int], lat_index: list[int], lon_index: list[int]
) -> np.ndarray:
    """Return shape (hours, cells) values of one variable for the given time positions."""

    def one(position: int) -> np.ndarray:
        for attempt in range(ATTEMPTS):
            try:
                field = np.asarray(array[position])
                return field[lat_index, lon_index]
            except Exception:
                if attempt == ATTEMPTS - 1:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        return np.stack(list(pool.map(one, positions)))


def main() -> None:
    """Fetch every missing month of every variable, then combine."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", default=None, help="fetch only this month, e.g. 2023-01")
    args = parser.parse_args()
    cells = farm_cells()
    unique = cells.select("cell_id", "lat_q", "lon_q").unique().sort("cell_id")
    options = {"token": "anon"}
    group = zarr.open_group(f"gs://{BUCKET}", mode="r", storage_options=options)
    dataset = xr.open_zarr(f"gs://{BUCKET}", storage_options=options, chunks=None)
    times, latitudes, longitudes = (dataset[k].values for k in ("time", "latitude", "longitude"))
    lat_index = [int(np.argmin(np.abs(latitudes - q / 4))) for q in unique["lat_q"]]
    lon_index = [int(np.argmin(np.abs(longitudes - (q / 4) % 360))) for q in unique["lon_q"]]
    for index, q in zip(lat_index, unique["lat_q"], strict=True):
        if abs(float(latitudes[index]) - q / 4) > 1e-6:
            raise ValueError("a farm cell latitude is not on the ARCO grid")
    for index, q in zip(lon_index, unique["lon_q"], strict=True):
        if abs(float(longitudes[index]) - (q / 4) % 360) > 1e-6:
            raise ValueError("a farm cell longitude is not on the ARCO grid")
    months = [m for m in month_starts() if args.month is None or f"{m:%Y-%m}" == args.month]
    began = time.monotonic()
    for variable, name in VARIABLES.items():
        directory = OUT_DIR / "_chunks" / variable
        directory.mkdir(parents=True, exist_ok=True)
        array = group[name]
        assert isinstance(array, zarr.Array)
        for start in months:
            path = directory / f"{start:%Y-%m}.parquet"
            if path.exists():
                continue
            stop = datetime(start.year + (start.month == 12), start.month % 12 + 1, 1, tzinfo=UTC)
            stop_hour = min(stop, LAST_HOUR + timedelta(hours=1))
            hours = np.arange(
                np.datetime64(start.replace(tzinfo=None), "h"),
                np.datetime64(stop_hour.replace(tzinfo=None), "h"),
                np.timedelta64(1, "h"),
            )
            positions = [int(np.searchsorted(times, hour)) for hour in hours]
            if any(times[p] != h for p, h in zip(positions, hours, strict=True)):
                raise ValueError(f"{variable} {start:%Y-%m}: an hour is missing from the store")
            fields = read_month(
                array=array, positions=positions, lat_index=lat_index, lon_index=lon_index
            )
            if not np.isfinite(fields).all():
                raise ValueError(f"{variable} {start:%Y-%m}: NaN or infinite values")
            frame = pl.DataFrame(
                {
                    "cell_id": np.tile(unique["cell_id"].to_numpy(), len(hours)).astype(np.int32),
                    "time": pl.Series(
                        np.repeat(hours.astype("datetime64[us]"), unique.height)
                    ).dt.replace_time_zone("UTC"),
                    variable: fields.reshape(-1).astype(np.float32),
                }
            )
            partial = path.with_suffix(".parquet.partial")
            frame.write_parquet(partial)
            partial.rename(path)
            print(
                f"{variable} {start:%Y-%m} {frame.height} rows {time.monotonic() - began:.0f} s",
                flush=True,
            )
    if args.month is not None:
        return
    combined = None
    for variable in VARIABLES:
        frames = [
            pl.read_parquet(OUT_DIR / "_chunks" / variable / f"{m:%Y-%m}.parquet")
            for m in month_starts()
        ]
        part = pl.concat(frames)
        combined = (
            part
            if combined is None
            else combined.join(part, on=["cell_id", "time"], validate="1:1")
        )
    assert combined is not None
    combined.sort("cell_id", "time").write_parquet(OUT_DIR / "era5_farm_cells.parquet")
    cells.write_parquet(WORK_DIR / "farm_cells.parquet")
    print(f"wrote {combined.height} rows for {unique.height} cells")


if __name__ == "__main__":
    main()
