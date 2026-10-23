"""Extract the extra ECMWF ensemble variables for each farm's rows (experiment 270).

For each farm, take the ensemble-mean of dew point, downward longwave, precipitation, surface
pressure, mean-sea-level pressure, 10 m and 100 m wind speed, and 500 hPa geopotential height at the
store's 3-hourly period that holds each half-hour of experiment 224's band-0 rows (00 UTC run, leads
12 to 36 hours). All exist at forecast time. Relative humidity is derived from temperature and dew
point with the Magnus formula. Writes `extra_<farm>.parquet` under the experiment's data folder.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00270-*/e00270_weather.py
"""

import e00270_common as common
import polars as pl
from studies.pv_dataset import pv_sites
from studies.sources import REPO_DATA_DIR

NWP_URI = str(REPO_DATA_DIR / "NWP")
VARIABLES = {
    "dew_point_temperature_2m": "dew_point",
    "downward_long_wave_radiation_flux_surface": "longwave",
    "precipitation_surface": "precipitation",
    "pressure_surface": "pressure_surface",
    "pressure_reduced_to_mean_sea_level": "pressure_msl",
    "geopotential_height_500hpa": "z500",
    "temperature_2m": "temp_2m",
}


def nearest_cell(*, latitude: float, longitude: float) -> int:
    """Return the store's H3 cell nearest the farm (the rule of experiment 224)."""
    import e00270_cells as cells

    return cells.nearest_cell(latitude=latitude, longitude=longitude, uri=NWP_URI)


def extract(site: str) -> str:
    """Write one farm's extra variables per (init_time, valid_time)."""
    row = pv_sites().filter(pl.col("site") == site).row(0, named=True)
    cell = nearest_cell(latitude=row["latitude"], longitude=row["longitude"])
    columns = dict(VARIABLES)
    raw = (
        pl.scan_delta(NWP_URI)
        .filter(
            (pl.col("h3_index") == cell)
            & (pl.col("init_time") >= common.ENS_START)
            & (pl.col("init_time") < pl.lit("2026-09-01").str.to_datetime(time_zone="UTC"))
        )
        .select(
            "init_time",
            "valid_time",
            "ensemble_member",
            *[pl.col(k).alias(v) for k, v in columns.items()],
            wind10=(pl.col("wind_u_10m") ** 2 + pl.col("wind_v_10m") ** 2).sqrt(),
            wind100=(pl.col("wind_u_100m") ** 2 + pl.col("wind_v_100m") ** 2).sqrt(),
        )
        .group_by("init_time", "valid_time")
        .agg(pl.len().alias("n_members"), pl.exclude("ensemble_member").mean())
        .filter(pl.col("n_members") == common.N_MEMBERS)
        .collect()
        .with_columns(
            relative_humidity=(
                100.0
                * (
                    (17.625 * pl.col("dew_point") / (243.04 + pl.col("dew_point"))).exp()
                    / (17.625 * pl.col("temp_2m") / (243.04 + pl.col("temp_2m"))).exp()
                )
            ).clip(0.0, 100.0)
        )
        .drop("n_members", "temp_2m", "dew_point")
    )
    path = common.DATA_DIR / f"extra_{site}.parquet"
    raw.write_parquet(path)
    return f"{site} {raw.height} rows"


def main() -> None:
    """Extract every farm."""
    common.DATA_DIR.mkdir(parents=True, exist_ok=True)
    for site in common.FARMS:
        print(extract(site), flush=True)


if __name__ == "__main__":
    main()
