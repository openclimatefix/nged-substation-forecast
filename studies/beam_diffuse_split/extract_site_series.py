"""Cut each solar farm's hourly irradiance from a gridded download, as a per-site frame.

One-off throwaway script for the second round of the study in
<https://github.com/openclimatefix/nged-substation-forecast/issues/809>. It writes the same frame
`fetch_open_meteo_point.py` writes, one row per (site, hour) with `ghi_w_m2` and `bhi_w_m2`, so
`build_dataset.py` reads the result as it reads any other per-site source.

**Each product reaches the mean over the hour ending at the label, the object every other source
serves.** The two products store something else:

- **SARAH-3** stores instantaneous satellite snapshots every 30 minutes. The hour labelled `T` is
  the mean of the snapshots stamped `T - 60 min` and `T - 30 min` (`SARAH_SLOT_OFFSETS_MINUTES`).
- **ICON-DREAM-EU** stores the mean since the most recent 3-hourly forecast start, which
  `studies.hourly_means.hourly_from_running_means` de-averages. The global flux is the direct plus
  the diffuse flux, which is how DWD splits it.

`check_new_products.py` checks both conventions against the sun before any build reads them.

**Each site reads its nearest grid cell by great-circle distance**, and the script logs that
distance for each site: 0.8 km to 2.8 km on SARAH-3's 0.05° grid, and 1.6 km to 5.0 km on
ICON-DREAM's triangles of about 6.5 km. No coordinate is written or logged, only the anonymised
site label.

Run it with `uv run --with netcdf4 python studies/beam_diffuse_split/extract_site_series.py
--product sarah-3`, and again with `--product icon-dream-eu`. `--first-date`, `--last-date` and
`--output` exist for a smoke test that reads one month and writes outside `data/`.
"""

import argparse
import logging
import sys
import urllib.request
from datetime import date
from pathlib import Path
from typing import Final, Literal

import numpy as np
import polars as pl
import xarray as xr
from build_dataset import _pv_sites
from sources import WEATHER_DATA_DIR, point_output_path_for
from studies.grid_sampling import nearest_cells
from studies.hourly_means import KEY_COLUMN, hourly_from_running_means, hourly_from_snapshots

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("extract_site_series")

ExtractedProductType = Literal["sarah-3", "icon-dream-eu"]
"""The gridded products this script cuts per-site frames from."""

SARAH_DIR: Final[Path] = WEATHER_DATA_DIR / "SARAH-3"
"""Holds `SIS/` (global) and `SID/` (direct) daily files from the CM SAF orders."""

SARAH_SLOT_OFFSETS_MINUTES: Final[tuple[int, int]] = (-60, -30)
"""Which snapshots, relative to the label, make up the hour ending at the label.

SEVIRI scans the Earth from south to north over about 12 minutes from each slot's nominal time, so
the trial area is observed some minutes after its stamp. The snapshots stamped 60 and 30 minutes
before the label therefore straddle the hour's midpoint more closely than the snapshots at 30 and 0
minutes do. Two measurements agree, on 48 days spread through 2025 at the six solar farms. Against
the sun (`studies.timestamp_checks`), the hourly means from the earlier pair track it best 30
minutes before the label at every farm, as a mean over the hour ending at the label should and as
CAMS's hourly values do at 35 minutes; the later pair peaks at the label itself. Against CAMS, the
earlier pair correlates at 0.977 and the later at 0.961. `check_new_products.py` repeats the first
measurement on the whole record.
"""

SARAH_GOOD_RECORD: Final[int] = 0
"""The `record_status` of a usable slot; 1 is void and 2 is bad quality."""

ICON_DREAM_DIR: Final[Path] = WEATHER_DATA_DIR / "ICON-DREAM-EU"
"""Holds one parquet per variable, keyed by ICON grid cell, from the issue #841 download."""

ICON_DREAM_FILES: Final[dict[str, str]] = {
    "direct": "ASWDIR_S_201909_202608.parquet",
    "diffuse": "ASWDIFD_S_201909_202608.parquet",
}
"""The direct and the diffuse downward shortwave flux at the surface."""

ICON_DREAM_VALUE_COLUMNS: Final[dict[str, str]] = {
    "direct": "aswdir_s_w_m2",
    "diffuse": "aswdifd_s_w_m2",
}

ICON_DREAM_CYCLE_HOURS: Final[int] = 3
"""ICON-DREAM's forecasts start every 3 hours from 00 UTC, and its radiation averages since then."""

ICON_DREAM_GRID_URL: Final[str] = (
    "https://opendata.dwd.de/climate_environment/REA/ICON-DREAM-EU/invariant/ICON-DREAM-EU_grid.nc"
)
"""The grid description holding each triangle's centre, about 400 MB, read once and deleted."""

ICON_DREAM_CELL_CENTRES: Final[Path] = ICON_DREAM_DIR / "cell_centres.parquet"
"""The centres of the cells the download holds, cached so the grid file is fetched only once."""


def _sarah_files(*, variable: str, first: date, last: date) -> dict[date, Path]:
    """Return SARAH-3's daily files for one variable, keyed by day.

    Args:
        variable: `SIS` or `SID`.
        first: The first day wanted.
        last: The last day wanted.

    Returns:
        Each day's file, for every day in the range the download holds.
    """
    files: dict[date, Path] = {}
    for path in sorted((SARAH_DIR / variable).glob(f"{variable}in*.nc")):
        stamp = path.name.removeprefix(f"{variable}in")[:8]
        day = date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8]))
        if first <= day <= last:
            files[day] = path
    return files


def _sarah_cells(*, path: Path, sites: pl.DataFrame) -> pl.DataFrame:
    """Return each site's nearest SARAH-3 cell, as indices into the latitude and longitude axes.

    Args:
        path: Any one daily file, for the grid's axes.
        sites: The roster, carrying `site`, `latitude` and `longitude`.

    Returns:
        One row per site with `site`, `lat_index`, `lon_index` and `distance_km`.
    """
    with xr.open_dataset(path) as dataset:
        latitudes = dataset["lat"].to_numpy().astype(np.float64)
        longitudes = dataset["lon"].to_numpy().astype(np.float64)
    lat_grid, lon_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
    cells = pl.DataFrame(
        {"latitude": lat_grid.ravel(), "longitude": lon_grid.ravel()}
    ).with_row_index(name="cell_id")
    nearest = nearest_cells(sites=sites, cells=cells)
    return nearest.with_columns(
        lat_index=pl.col("cell_id") // len(longitudes),
        lon_index=pl.col("cell_id") % len(longitudes),
    ).drop("cell_id")


def _sarah_day(*, sis: Path, sid: Path, cells: pl.DataFrame) -> pl.DataFrame:
    """Return one day's snapshots at each site's cell, with unusable slots set to not-a-number.

    Args:
        sis: The day's global irradiance file.
        sid: The day's direct irradiance file.
        cells: The output of `_sarah_cells`.

    Returns:
        One row per (site, slot) with `ghi_w_m2` and `bhi_w_m2`.

    Raises:
        ValueError: If the two files hold different slots.
    """
    lat_index = xr.DataArray(cells["lat_index"].to_numpy(), dims="site")
    lon_index = xr.DataArray(cells["lon_index"].to_numpy(), dims="site")
    values: dict[str, np.ndarray] = {}
    stamps: dict[str, np.ndarray] = {}
    for column, path, variable in (("ghi_w_m2", sis, "SIS"), ("bhi_w_m2", sid, "SID")):
        with xr.open_dataset(path) as dataset:
            usable = dataset["record_status"].to_numpy() == SARAH_GOOD_RECORD
            field = dataset[variable].isel(lat=lat_index, lon=lon_index).to_numpy()
            stamps[column] = dataset["time"].to_numpy().astype("datetime64[us]")
        values[column] = np.where(usable[:, None], field, np.nan).astype(np.float64)
    times = stamps["ghi_w_m2"]
    if not np.array_equal(times, stamps["bhi_w_m2"]):
        msg = f"{sis.name} and {sid.name} hold different slots"
        raise ValueError(msg)
    n_slots, n_sites = values["ghi_w_m2"].shape
    return pl.DataFrame(
        {
            KEY_COLUMN: np.tile(cells["site"].to_numpy(), n_slots),
            "time": np.repeat(times, n_sites),
            "ghi_w_m2": values["ghi_w_m2"].ravel(),
            "bhi_w_m2": values["bhi_w_m2"].ravel(),
        }
    ).with_columns(pl.col("time").dt.replace_time_zone("UTC"))


def extract_sarah(*, sites: pl.DataFrame, first: date, last: date) -> pl.DataFrame:
    """Return SARAH-3's hourly global and direct irradiance at each site's nearest cell.

    Args:
        sites: The roster, carrying `site`, `latitude` and `longitude`.
        first: The first day to read.
        last: The last day to read.

    Returns:
        One row per (site, hour) with `ghi_w_m2` and `bhi_w_m2`, each the mean over the hour ending
        at `time`.

    Raises:
        ValueError: If the global and the direct downloads do not hold the same days.
    """
    sis_files = _sarah_files(variable="SIS", first=first, last=last)
    sid_files = _sarah_files(variable="SID", first=first, last=last)
    if sis_files.keys() != sid_files.keys():
        msg = "the SIS and SID downloads do not hold the same days"
        raise ValueError(msg)
    cells = _sarah_cells(path=next(iter(sis_files.values())), sites=sites)
    _log_distances(product="SARAH-3", nearest=cells)
    snapshots = pl.concat(
        _sarah_day(sis=sis_files[day], sid=sid_files[day], cells=cells) for day in sorted(sis_files)
    )
    return hourly_from_snapshots(
        frame=snapshots,
        value_columns=("ghi_w_m2", "bhi_w_m2"),
        slot_offsets_minutes=SARAH_SLOT_OFFSETS_MINUTES,
    ).rename({KEY_COLUMN: "site"})


def _icon_dream_cell_centres(*, cache: Path, cell_ids: list[int]) -> pl.DataFrame:
    """Return the centres of the ICON-DREAM cells the download holds, from a cache or the grid file.

    Args:
        cache: Where the centres are kept once read.
        cell_ids: The cells the download holds.

    Returns:
        One row per cell with `cell_id`, `latitude` and `longitude` in degrees.
    """
    if not cache.exists():
        grid_path = cache.with_name("ICON-DREAM-EU_grid.nc")
        _LOG.info("reading ICON-DREAM's grid description once, about 400 MB")
        urllib.request.urlretrieve(ICON_DREAM_GRID_URL, grid_path)
        try:
            with xr.open_dataset(grid_path) as grid:
                latitudes = np.degrees(grid["clat"].to_numpy()[cell_ids])
                longitudes = np.degrees(grid["clon"].to_numpy()[cell_ids])
        finally:
            grid_path.unlink(missing_ok=True)
        pl.DataFrame(
            {"cell_id": cell_ids, "latitude": latitudes, "longitude": longitudes}
        ).write_parquet(cache)
    return pl.read_parquet(cache)


def extract_icon_dream(
    *, sites: pl.DataFrame, first: date, last: date, cell_centres: Path
) -> pl.DataFrame:
    """Return ICON-DREAM-EU's hourly global and direct irradiance at each site's nearest cell.

    Args:
        sites: The roster, carrying `site`, `latitude` and `longitude`.
        first: The first day to keep.
        last: The last day to keep.
        cell_centres: Where the cells' centres are cached.

    Returns:
        One row per (site, hour) with `ghi_w_m2` and `bhi_w_m2`, each the mean over the hour ending
        at `time`.
    """
    raw = {
        part: pl.read_parquet(ICON_DREAM_DIR / name).select(
            time=pl.col("valid_time").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"),
            value=pl.col(ICON_DREAM_VALUE_COLUMNS[part]).cast(pl.Float64),
            **{KEY_COLUMN: pl.col("cell_id")},
        )
        for part, name in ICON_DREAM_FILES.items()
    }
    cell_ids = sorted(raw["direct"][KEY_COLUMN].unique().to_list())
    nearest = nearest_cells(
        sites=sites, cells=_icon_dream_cell_centres(cache=cell_centres, cell_ids=cell_ids)
    )
    _log_distances(product="ICON-DREAM-EU", nearest=nearest)
    wanted = nearest["cell_id"].unique().to_list()
    hourly = {
        part: hourly_from_running_means(
            frame=frame.filter(
                pl.col(KEY_COLUMN).is_in(wanted),
                pl.col("time").dt.date().is_between(first, last),
            ),
            value_column="value",
            cycle_hours=ICON_DREAM_CYCLE_HOURS,
        )
        for part, frame in raw.items()
    }
    fluxes = hourly["direct"].join(hourly["diffuse"], on=[KEY_COLUMN, "time"], suffix="_diffuse")
    return (
        nearest.select("site", **{KEY_COLUMN: pl.col("cell_id")})
        .join(fluxes, on=KEY_COLUMN)
        .select(
            "site",
            "time",
            ghi_w_m2=pl.col("value") + pl.col("value_diffuse"),
            bhi_w_m2=pl.col("value"),
        )
        .sort("site", "time")
    )


def _log_distances(*, product: str, nearest: pl.DataFrame) -> None:
    """Log how far each site sits from the centre of the cell it reads, never where either is.

    Args:
        product: The product's name, for the log line.
        nearest: One row per site with `site` and `distance_km`.
    """
    for row in nearest.sort("site").iter_rows(named=True):
        _LOG.info("%s: site %s reads a cell %.1f km away", product, row["site"], row["distance_km"])


def main() -> int:
    """Cut the per-site frame for the product named on the command line, and write it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", choices=("sarah-3", "icon-dream-eu"), required=True)
    parser.add_argument("--first-date", type=date.fromisoformat, default=date(2019, 1, 1))
    parser.add_argument("--last-date", type=date.fromisoformat, default=date(2026, 12, 31))
    parser.add_argument(
        "--output", type=Path, default=None, help="Write here instead of the product's own path."
    )
    parser.add_argument(
        "--cell-centres",
        type=Path,
        default=ICON_DREAM_CELL_CENTRES,
        help="Where ICON-DREAM's cell centres are cached.",
    )
    arguments = parser.parse_args()
    product: ExtractedProductType = arguments.product

    sites = _pv_sites().select("site", "latitude", "longitude")
    frame = (
        extract_sarah(sites=sites, first=arguments.first_date, last=arguments.last_date)
        if product == "sarah-3"
        else extract_icon_dream(
            sites=sites,
            first=arguments.first_date,
            last=arguments.last_date,
            cell_centres=arguments.cell_centres,
        )
    )
    output = arguments.output or point_output_path_for(source=product)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output)
    _LOG.info(
        "wrote %d rows covering %s to %s for %d sites to %s",
        frame.height,
        frame["time"].min(),
        frame["time"].max(),
        frame["site"].n_unique(),
        output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
