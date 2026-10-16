"""Build `work/frame.parquet`, one row per site and half-hour, and print the validation checks.

Columns: `site`, `kind` (substation or farm), `time` (UTC, period-ending), `u` (unit PV shape from
CAMS global horizontal irradiance), `u_clear` (the same from clear-sky irradiance), `ghi_u` and
`ghi_u_clear` (the irradiances in W/m²), `zenith`, `power`, and `temperature_2m` (substations
only). Only hours whose CAMS reliability is at least 0.9 are present.

    uv run --with holidays python \
    ml-experiments/00039-pv-disaggregation-cams-injection/e00039_inputs.py
"""

import e00039_common as common
import numpy as np
import polars as pl
from studies.power import scan_power
from studies.pv_dataset import pv_sites
from studies.pv_physics import half_hour_sun_and_sky, plant_power_mw


def unit_frame(*, cams: pl.DataFrame, latitude: float, longitude: float) -> pl.DataFrame:
    """Return the unit PV shape and irradiance at one CAMS point, in half-hours.

    Args:
        cams: Hourly CAMS rows of one site with `time`, `ghi_w_m2`, `clear_sky_ghi_w_m2`,
            and `reliability`.
        latitude: Degrees north.
        longitude: Degrees east.

    Returns:
        One row per half-hour of each reliable hour.
    """
    reliable = cams.filter(pl.col("reliability") >= common.MIN_RELIABILITY).sort("time")
    out: dict[str, np.ndarray] = {}
    for column, name in (("ghi_w_m2", "u"), ("clear_sky_ghi_w_m2", "u_clear")):
        sky = half_hour_sun_and_sky(
            hourly=reliable.select("time", pl.col(column).cast(pl.Float64).alias("ghi_w_m2")),
            latitude=latitude,
            longitude=longitude,
        )
        out[name] = plant_power_mw(sky=sky, parameters=common.FLEET)
        out["ghi_" + name] = sky.ghi_w_m2
    out["zenith"] = sky.zenith_deg
    out["time"] = sky.half_hour_end_time
    return pl.DataFrame(out).with_columns(
        pl.col("time").cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    )


def build() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the half-hourly frame and the site table (coordinates and effective capacity)."""
    series = pl.read_parquet(common.SERIES_PATH).filter(
        pl.col("group").is_in(["primary demand", "grid flow"])
    )
    farms = pv_sites()
    cams_demand = pl.read_parquet(common.CAMS_DEMAND_POINTS_PATH)
    cams_farm = pl.read_parquet(common.CAMS_FARM_POINTS_PATH)
    power = scan_power().collect()
    era5 = pl.read_parquet(common.ERA5_PATH).select(
        "time_series_id", "valid_time", "temperature_2m"
    )
    frames = []
    for row in series.iter_rows(named=True):
        u = unit_frame(
            cams=cams_demand.filter(pl.col("site") == row["site"]),
            latitude=row["latitude"],
            longitude=row["longitude"],
        )
        p = power.filter(pl.col("time_series_id") == row["time_series_id"]).select("time", "power")
        t = era5.filter(pl.col("time_series_id") == row["time_series_id"]).select(
            pl.col("valid_time").alias("hour"), "temperature_2m"
        )
        frames.append(
            u.join(p, on="time", how="left")
            .with_columns(hour=pl.col("time").dt.offset_by("29m").dt.truncate("1h"))
            .join(t, on="hour", how="left")
            .drop("hour")
            .with_columns(site=pl.lit(row["site"]), kind=pl.lit("substation"))
        )
    for row in farms.iter_rows(named=True):
        u = unit_frame(
            cams=cams_farm.filter(pl.col("site") == row["site"]),
            latitude=row["latitude"],
            longitude=row["longitude"],
        )
        p = power.filter(pl.col("time_series_id") == row["time_series_id"]).select("time", "power")
        frames.append(
            u.join(p, on="time", how="left").with_columns(
                temperature_2m=pl.lit(None, pl.Float32),
                site=pl.lit(row["site"]),
                kind=pl.lit("farm"),
            )
        )
    frame = pl.concat(frames, how="vertical_relaxed")
    columns = ("site", "latitude", "longitude", "effective_capacity_mw")
    sites = pl.concat(
        [
            series.select(*columns, "time_series_id"),
            farms.select(*columns, "time_series_id").cast(
                {"effective_capacity_mw": pl.Float32, "time_series_id": pl.Int32}
            ),
        ]
    )
    return frame, sites


def validate(*, frame: pl.DataFrame) -> None:
    """Print the checks of the `data-validation` skill that apply to this frame."""
    print("\nRows, non-null power share, and first and last time per site")
    with pl.Config(tbl_rows=60, tbl_width_chars=200):
        print(
            frame.group_by("site")
            .agg(
                pl.len().alias("rows"),
                pl.col("power").is_not_null().mean().alias("power_share"),
                pl.col("time").min().alias("first"),
                pl.col("time").max().alias("last"),
            )
            .sort("site")
        )
    duplicates = frame.group_by("site", "time").len().filter(pl.col("len") > 1).height
    print(f"duplicate (site, time) keys: {duplicates}")
    assert duplicates == 0
    print("P99 of u over daylight (design: 1.0):")
    print(
        frame.filter(pl.col("zenith") < common.MAX_ZENITH_DEG)
        .group_by("kind")
        .agg(pl.col("u").quantile(0.99).alias("p99_u"))
    )
    print("\nHour-of-day (UTC) profile of corr(power, u), directional Primaries: peak near 12")
    profile = (
        frame.filter(
            pl.col("site").is_in(common.DIRECTIONAL_PRIMARIES)
            & pl.col("power").is_not_null()
            & (pl.col("zenith") < common.MAX_ZENITH_DEG)
            & (pl.col("time").dt.month().is_in(common.SUMMER_MONTHS))
        )
        .with_columns(hour=pl.col("time").dt.hour())
        .group_by("site", "hour")
        .agg(pl.corr("power", "u").alias("corr"))
        .group_by("hour")
        .agg(pl.col("corr").median())
        .sort("hour")
    )
    with pl.Config(tbl_rows=30):
        print(profile)
    print("\nFarm checks: share of non-null power at or below zero while GHI > 100 W/m²")
    print(
        frame.filter((pl.col("kind") == "farm") & pl.col("power").is_not_null())
        .group_by("site")
        .agg(
            ((pl.col("power") <= 0.0) & (pl.col("ghi_u") > common.DROPOUT_GHI_W_M2))
            .mean()
            .alias("dropout_share"),
            pl.col("power").min().alias("min_power"),
            pl.col("power").quantile(0.99).alias("p99_power"),
        )
        .sort("site")
    )


def main() -> None:
    """Build the frame, write it, and validate it."""
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    frame, sites = build()
    frame.write_parquet(common.FRAME_PATH)
    sites.write_parquet(common.SITES_PATH)
    validate(frame=frame)


if __name__ == "__main__":
    main()
