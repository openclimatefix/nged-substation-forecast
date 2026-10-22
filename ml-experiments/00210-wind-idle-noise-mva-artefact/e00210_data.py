"""Join the three metered wind series to ERA5 100 m wind speed, one row per half-hour."""

from pathlib import Path
from typing import Final

import polars as pl
from studies.power import scan_power
from studies.pv_dataset import wind_sites

ERA5_PATH: Final[Path] = Path("data/handover/era5_wind_at_nged_wind_farms_anonymised.parquet")
OUT_DIR: Final[Path] = Path("data/ml-experiments/00210-wind-idle-noise-mva-artefact")


def half_hourly_wind_and_power() -> pl.DataFrame:
    """Return `site`, `time_series_id`, `valid_time`, `power`, `wind_speed`, `capacity_mw`.

    `valid_time` is the end of the half-hour as stored. The wind is ERA5 100 m speed linearly
    interpolated to the half-hour's midpoint (`valid_time` minus 15 minutes).
    """
    sites = wind_sites()
    power = (
        scan_power()
        .filter(pl.col("time_series_id").is_in(sites["time_series_id"].to_list()))
        .select("time_series_id", "time", "power")
        .collect()
        .rename({"time": "valid_time"})
        .join(sites.select("time_series_id", "site", "effective_capacity_mw"), on="time_series_id")
    )
    era5 = pl.read_parquet(ERA5_PATH).select("site", "time", "wind_speed_100m").sort("site", "time")
    parts = []
    for site in ("W1", "W2", "W3"):
        hourly = era5.filter(pl.col("site") == site)
        sub = power.filter(pl.col("site") == site).with_columns(
            mid=pl.col("valid_time") - pl.duration(minutes=15)
        )
        # Linear interpolation between the two hourly instants either side of the midpoint.
        before = sub.sort("mid").join_asof(
            hourly.rename({"time": "t0", "wind_speed_100m": "w0"}).sort("t0"),
            left_on="mid",
            right_on="t0",
            by="site",
            strategy="backward",
        )
        after = before.join_asof(
            hourly.rename({"time": "t1", "wind_speed_100m": "w1"}).sort("t1"),
            left_on="mid",
            right_on="t1",
            by="site",
            strategy="forward",
        )
        frac = (pl.col("mid") - pl.col("t0")).dt.total_seconds() / (
            pl.col("t1") - pl.col("t0")
        ).dt.total_seconds().replace(0, 1)
        parts.append(
            after.with_columns(
                wind_speed=pl.when(pl.col("t0") == pl.col("t1"))
                .then(pl.col("w0"))
                .otherwise(pl.col("w0") + frac * (pl.col("w1") - pl.col("w0")))
            )
        )
    return (
        pl.concat(parts)
        .drop_nulls("wind_speed")
        .select(
            "site",
            "time_series_id",
            "valid_time",
            "power",
            "wind_speed",
            pl.col("effective_capacity_mw").alias("capacity_mw"),
        )
        .sort("site", "valid_time")
    )


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = half_hourly_wind_and_power()
    frame.write_parquet(OUT_DIR / "joined.parquet")
    print(
        frame.group_by("site")
        .agg(
            pl.len(),
            pl.col("valid_time").min().alias("t0"),
            pl.col("valid_time").max().alias("t1"),
            pl.col("power").min().alias("pmin"),
            pl.col("power").max().alias("pmax"),
            pl.col("power").median().alias("pmed"),
            pl.col("capacity_mw").first(),
            pl.col("power").is_null().sum().alias("nulls"),
            (pl.col("valid_time").is_duplicated().sum()).alias("dups"),
        )
        .sort("site")
    )
