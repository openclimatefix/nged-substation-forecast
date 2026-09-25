"""Download native ERA5 10 m and 100 m wind from the Climate Data Store at the wind-farm cells.

One-off throwaway script for the downloads in
<https://github.com/openclimatefix/nged-substation-forecast/issues/841>. It checks Open-Meteo's
ERA5 wind (speed and direction, km/h) against the native Copernicus fields, including whether
Open-Meteo applies a scaling factor.

**Each wind site gets its own 3 x 3 block of 0.25 degree ERA5 cells.** The block is centred on the
cell nearest the site, so the comparison can test both the nearest cell and the land-cell choice
Open-Meteo makes. One pooled box covering the three blocks is requested, and the download is then
cut down to the blocks. Rows are keyed by the anonymous wind label (`W1` to `W3`) and the cell's
offset from the nearest cell, never by a coordinate. The pooled request box is never printed or
logged.

**Requests are six months each.** The Climate Data Store runs one job per account at a time and
rejects requests over its cost limit (a two-year request was refused). Each chunk is skipped when
its output exists.

**The data on disk covers only 2024-01-01 to 2026-09-20.** That is 3 years, the newest 6 six-month
chunks, fetched with `--chunks 6`, because the Open-Meteo comparison needs only a couple of years.
Running without `--chunks` fetches the older chunks back to 2019-09 as well; cached chunks are
skipped.

Run with `uv run --with cdsapi --with netCDF4 python studies/weather_downloads/fetch_era5_wind.py`,
adding `--chunks 6` for the newest three years only.
Credentials are read by `cdsapi` from `~/.cdsapirc`; nothing is stored here.
"""

import argparse
import logging
import sys
import zipfile
from pathlib import Path
from typing import Final

import cdsapi  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import xarray as xr
from lineage import write_lineage_note, write_readme
from paths import WEATHER_DOWNLOADS_DIR

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "beam_diffuse_split"))
from build_dataset import _wind_sites

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("fetch_era5_wind")

OUTPUT_DIR: Final[Path] = WEATHER_DOWNLOADS_DIR / "ERA5"
CHUNK_DIR: Final[Path] = OUTPUT_DIR / "wind_native_chunks"
OUTPUT_PATH: Final[Path] = OUTPUT_DIR / "wind_native_cds.parquet"
VARIABLES: Final[dict[str, str]] = {
    "100m_u_component_of_wind": "u100",
    "100m_v_component_of_wind": "v100",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
}
FIRST_DATE: Final[tuple[int, int]] = (2019, 9)
LAST_DATE: Final[tuple[int, int]] = (2026, 9)
GRID_STEP: Final[float] = 0.25
MONTHS_PER_REQUEST: Final[int] = 6
"""The Climate Data Store rejected two-year requests (cost limit); six months of 4 fields fits."""


def _site_blocks() -> pl.DataFrame:
    """Return one row per (site, dy, dx): the cell's grid coordinates in quarter degrees.

    Coordinates are held in memory only, as integer quarter-degree indices.
    """
    sites = _wind_sites()
    rows = []
    for site, lat, lon in sites.select("site", "latitude", "longitude").iter_rows():
        centre_lat, centre_lon = round(lat / GRID_STEP), round(lon / GRID_STEP)
        rows.extend(
            {"site": site, "dy": dy, "dx": dx, "lat_q": centre_lat + dy, "lon_q": centre_lon + dx}
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
        )
    return pl.DataFrame(rows)


def _chunks() -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Return (years, months) per request: at most six calendar months, never crossing a year."""
    chunks = []
    for year in range(FIRST_DATE[0], LAST_DATE[0] + 1):
        first = FIRST_DATE[1] if year == FIRST_DATE[0] else 1
        last = LAST_DATE[1] if year == LAST_DATE[0] else 12
        months = list(range(first, last + 1))
        chunks.extend(
            ((year,), tuple(months[i : i + MONTHS_PER_REQUEST]))
            for i in range(0, len(months), MONTHS_PER_REQUEST)
        )
    return chunks


def _fetch_chunk(*, years: tuple[int, ...], months: tuple[int, ...], blocks: pl.DataFrame) -> Path:
    """Download one chunk to a zip, or return the existing file."""
    path = CHUNK_DIR / f"era5_wind_{years[0]}_{months[0]:02d}_{months[-1]:02d}.zip"
    if path.exists():
        _LOG.info("%s exists, skipping", path.name)
        return path
    lat_q, lon_q = blocks["lat_q"].to_list(), blocks["lon_q"].to_list()
    request = {
        "product_type": ["reanalysis"],
        "variable": list(VARIABLES),
        "year": [str(y) for y in years],
        "month": [f"{m:02d}" for m in months],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [
            max(lat_q) * GRID_STEP,
            min(lon_q) * GRID_STEP,
            min(lat_q) * GRID_STEP,
            max(lon_q) * GRID_STEP,
        ],
        "data_format": "netcdf",
        "download_format": "zip",
    }
    partial = path.with_suffix(".zip.partial")
    cdsapi.Client(quiet=True, progress=False).retrieve(
        "reanalysis-era5-single-levels", request, str(partial)
    )
    partial.rename(path)
    _LOG.info("%s downloaded (%.1f MB)", path.name, path.stat().st_size / 1e6)
    return path


def _read_chunk(*, path: Path, blocks: pl.DataFrame) -> pl.DataFrame:
    """Return the block cells of one chunk as a long frame keyed by site, offsets and time."""
    frames = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            extracted = Path(archive.extract(name, CHUNK_DIR / "_tmp"))
            with xr.open_dataset(extracted) as ds:
                time_name = "valid_time" if "valid_time" in ds.coords else "time"
                frame = ds.to_dataframe().reset_index()
                frame = frame.rename(columns={time_name: "time"})
                frames.append(pl.from_pandas(frame))
            extracted.unlink()
    long = frames[0]
    for other in frames[1:]:
        keep = [c for c in other.columns if c in ("u100", "v100", "u10", "v10")]
        long = long.join(
            other.select("time", "latitude", "longitude", *keep),
            on=["time", "latitude", "longitude"],
            how="full",
            coalesce=True,
        )
    long = long.with_columns(
        (pl.col("latitude") / GRID_STEP).round().cast(pl.Int64).alias("lat_q"),
        (pl.col("longitude") / GRID_STEP).round().cast(pl.Int64).alias("lon_q"),
    )
    return long.join(blocks, on=["lat_q", "lon_q"]).select(
        "site", "dy", "dx", "time", "u100", "v100", "u10", "v10"
    )


def main() -> int:
    """Fetch every chunk, then assemble the parquet, lineage and README."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, default=None, help="only the newest N chunks")
    args = parser.parse_args()
    blocks = _site_blocks()
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    chunks = _chunks()[-args.chunks :] if args.chunks else _chunks()
    paths = [
        _fetch_chunk(years=years, months=months, blocks=blocks)
        for years, months in reversed(chunks)
    ]
    frame = (
        pl.concat([_read_chunk(path=p, blocks=blocks) for p in paths])
        .unique(subset=["site", "dy", "dx", "time"])
        .sort("site", "dy", "dx", "time")
    )
    frame = frame.with_columns(pl.col(c).cast(pl.Float32) for c in ("u100", "v100", "u10", "v10"))
    frame.write_parquet(OUTPUT_PATH)
    _LOG.info("wrote %d rows to %s", frame.height, OUTPUT_PATH.name)

    write_lineage_note(
        product_dir=OUTPUT_DIR,
        filename="lineage_wind_native_cds.json",
        source_address="https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels",
        request_description=(
            "ERA5 hourly single levels, 100 m and 10 m u/v wind, netCDF, requested in "
            "six-month chunks over one pooled box covering a 3 x 3 block of 0.25 "
            "degree cells around each of the three wind sites, then cut to those blocks."
        ),
        variables=list(VARIABLES),
        extra={
            "first_time": str(frame["time"].min()),
            "last_time": str(frame["time"].max()),
            "rows": frame.height,
        },
    )
    write_readme(
        product_dir=OUTPUT_DIR,
        filename="README_wind_native.md",
        product_name="ERA5 native wind from the Climate Data Store, at the wind-farm cells",
        source_web_page="https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels",
        script_path="studies/weather_downloads/fetch_era5_wind.py",
        lineage_filenames=["lineage_wind_native_cds.json"],
        columns={
            "site": "Anonymised wind-generator label, W1 to W3. Never a name, ID or coordinate.",
            "dy": "Cell offset in 0.25 degree steps north (+1) or south (-1) of the ERA5 cell "
            "nearest the site; 0 is the nearest cell.",
            "dx": "Cell offset in 0.25 degree steps east (+1) or west (-1) of the nearest cell.",
            "time": "UTC instant of the value (ERA5 winds are instantaneous, not averaged).",
            "u100": "Eastward wind at 100 m, m/s.",
            "v100": "Northward wind at 100 m, m/s.",
            "u10": "Eastward wind at 10 m, m/s.",
            "v10": "Northward wind at 10 m, m/s.",
        },
        missing_value_convention="No nulls expected; a row missing from a (site, dy, dx) series "
        "means the hour was absent from the download.",
        gotchas=[
            (
                "Speeds are in m/s, whereas Open-Meteo's ERA5 columns in `wind_era5.parquet` are "
                "km/h (multiply m/s by 3.6). Speed is sqrt(u^2 + v^2); direction the wind blows "
                "from is (270 - atan2(v, u) in degrees) mod 360."
            ),
            (
                "The request covered one pooled box around the three sites, cut to a 3 x 3 block "
                "of 0.25 degree cells per site. The pooled box is not published because per-site "
                "cells could locate the generators."
            ),
            (
                "The file on disk covers only 2024-01-01 to 2026-09-20, the newest 6 six-month "
                "chunks (`--chunks 6`). Running the script without `--chunks` adds older chunks "
                "back to 2019-09."
            ),
            (
                "The newest chunk asks for whole months up to 2026-09, so it ends a few days short "
                "of the end of the month (ERA5 lags real time by about 5 days)."
            ),
        ],
        external_docs={
            "ERA5 hourly data on single levels": "https://cds.climate.copernicus.eu/datasets/"
            "reanalysis-era5-single-levels",
            "Hersbach et al. (2020)": "https://doi.org/10.1002/qj.3803",
        },
    )
    return 0


if __name__ == "__main__":
    np.seterr(all="raise")
    sys.exit(main())
