"""Shared constants and the farm-row builder for experiment 218 (per-farm plant fit).

Farms appear only as labels A to F. The rows hold the metered output over capacity next to the sun
and sky (CAMS irradiance split into beam and diffuse, ERA5 air temperature) at every daylight
half-hour, and are written under `data/ml-experiments/00218-*/`, never committed.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import polars as pl
from studies.pv_dataset import nearest_era5_cell, pv_sites, read_cams, read_era5
from studies.pv_physics import MIN_ELEVATION_DEG, half_hour_sun_and_sky
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00218-metered-solar-farm-plant-fit"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
CLEANED: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00179-metered-solar-farm-exploration" / "cleaned_solar_farms.parquet"
)
ROWS_PATH: Final[Path] = DATA_DIR / "farm_rows_cams.parquet"
FARMS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E", "F")
FLAT_TOP_FARMS: Final[tuple[str, ...]] = ("C", "D", "E", "F")
MIN_CAMS_RELIABILITY: Final[float] = 0.9
MIN_TUNING_MONTHS: Final[int] = 10
FOLD_SCORED_STARTS: Final[dict[int, datetime]] = {
    1: datetime(2025, 9, 1, tzinfo=UTC),
    2: datetime(2024, 9, 1, tzinfo=UTC),
}
"""Each fold scores the 12 months from this instant and tunes on every earlier month."""
FLAT_TOP_CLEAR_SKY_W_M2: Final[float] = 650.0
FLAT_TOP_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8)
SHOULDER_MONTHS: Final[tuple[int, ...]] = (3, 4, 9, 10)
BOOTSTRAP_DRAWS: Final[int] = 10_000
BOOTSTRAP_SEED: Final[int] = 218
BOOTSTRAP_LEVEL: Final[float] = 0.9875


def scored_end(*, fold: int) -> datetime:
    """Return the exclusive end of a fold's scored year."""
    start = FOLD_SCORED_STARTS[fold]
    return start.replace(year=start.year + 1)


def build_rows() -> pl.DataFrame:
    """Join the cleaned farm output to CAMS irradiance and ERA5 temperature.

    Returns:
        One row per daylight half-hour with `site`, `time` (end of the half-hour, UTC), `output`
        (power over effective capacity), sky columns, `temperature_c`, and `clear_sky_ghi_w_m2`.
    """
    sites = pv_sites()
    cleaned = pl.read_parquet(CLEANED).filter(pl.col("drop_reason").is_null())
    cams = read_cams(min_reliability=MIN_CAMS_RELIABILITY)
    era5 = read_era5(source="open-meteo")
    sites = nearest_era5_cell(sites=sites, era5=era5)
    parts = []
    for row in sites.sort("site").iter_rows(named=True):
        hourly = cams.filter(pl.col("site") == row["site"]).select("time", "ghi_w_m2").drop_nulls()
        sky = half_hour_sun_and_sky(
            hourly=hourly, latitude=row["latitude"], longitude=row["longitude"]
        )
        sky_frame = pl.DataFrame(
            {
                "time": pl.Series(sky.half_hour_end_time).dt.replace_time_zone("UTC"),
                "ghi_w_m2": sky.ghi_w_m2,
                "dni_w_m2": sky.dni_w_m2,
                "dhi_w_m2": sky.dhi_w_m2,
                "zenith_deg": sky.zenith_deg,
                "sun_azimuth_deg": sky.sun_azimuth_deg,
            }
        )
        temperature = (
            era5.filter(
                (pl.col("latitude") == row["cell_latitude"])
                & (pl.col("longitude") == row["cell_longitude"])
            )
            .select(hour_end="time", temperature_c="temp_c")
            .drop_nulls()
        )
        power = cleaned.filter(pl.col("time_series_id") == row["time_series_id"]).select(
            "time",
            output=pl.col("power") / row["effective_capacity_mw"],
            clear_sky="clear_sky_ghi_w_m2",
        )
        joined = (
            power.join(sky_frame, on="time", how="inner")
            .with_columns(
                hour_end=(pl.col("time") - timedelta(microseconds=1)).dt.truncate("1h")
                + timedelta(hours=1)
            )
            .join(temperature, on="hour_end", how="inner")
            .drop("hour_end")
            .filter((90.0 - pl.col("zenith_deg")) > MIN_ELEVATION_DEG)
            .with_columns(site=pl.lit(row["site"]))
        )
        parts.append(joined)
    rows = pl.concat(parts).with_columns(
        month=pl.col("time").dt.month(),
        month_index=pl.col("time").dt.year() * 12 + pl.col("time").dt.month(),
        elevation_deg=90.0 - pl.col("zenith_deg"),
    )
    return rows.sort("site", "time")


def main() -> None:
    """Write the farm rows and print a summary per farm."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    rows.write_parquet(ROWS_PATH)
    print(
        rows.group_by("site")
        .agg(
            pl.len().alias("n"),
            pl.col("time").min().alias("first"),
            pl.col("time").max().alias("last"),
            pl.col("output").quantile(0.99).alias("p99_output"),
            pl.col("temperature_c").mean().alias("mean_temp"),
        )
        .sort("site")
    )


if __name__ == "__main__":
    main()
