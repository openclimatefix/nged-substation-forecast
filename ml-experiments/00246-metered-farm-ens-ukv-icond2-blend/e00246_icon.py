"""Build the per-row ICON-D2 tables for experiment 246 from Open-Meteo's Previous Runs file.

`previous_day1` is the run made 24 hours before the hour, so every row has a lead of 24 hours.
The irradiance columns are means over the hour ending at the stamp. Each half-hour row takes the
hour that holds its midpoint, scaled by the cosine of the solar zenith at the midpoint over the
mean cosine of the hour (the mean of the two half-hour midpoints). Temperature is interpolated
linearly to the midpoint between the two hourly stamps, and wind speed is converted from km/h
to m/s.

    uv run --no-sync python ml-experiments/00246-*/e00246_icon.py
"""

from datetime import timedelta

import e00246_common as common
import numpy as np
import pandas as pd  # noqa: TID251  (pvlib's API takes a pandas index)
import polars as pl
import pvlib
from studies.pv_dataset import pv_sites

MIN_COSINE = 0.02
KMH_PER_MS = 3.6


def cosine(*, site: str, times: pl.Series) -> np.ndarray:
    """Return the clipped cosine of the apparent zenith at a farm for UTC instants."""
    row = pv_sites().filter(pl.col("site") == site).row(0, named=True)
    index = pd.DatetimeIndex(times.to_numpy(), tz="UTC")
    position = pvlib.location.Location(row["latitude"], row["longitude"]).get_solarposition(index)
    return np.clip(np.cos(np.radians(position["apparent_zenith"].to_numpy())), 0.0, None)


def build(site: str) -> pl.DataFrame:
    """Return the ICON-D2 columns for one farm's band 0 rows."""
    rows = pl.read_parquet(common.members_table(site=site, band=0)).select("time", "init_time")
    icon = (
        pl.read_parquet(common.ICON_PATH)
        .filter(pl.col("site") == site)
        .select("time", *common.ICON_COLUMNS)
    )
    midpoint = rows["time"] - timedelta(minutes=15)
    hour_end = midpoint.dt.truncate("1h") + timedelta(hours=1)
    frame = rows.with_columns(midpoint=midpoint, hour_end=hour_end).with_columns(
        hour_start=pl.col("hour_end") - timedelta(hours=1)
    )
    frame = frame.join(icon.rename({"time": "hour_end"}), on="hour_end", how="left").join(
        icon.select(
            hour_start=pl.col("time"), temperature_prev=pl.col("temperature_2m_previous_day1")
        ),
        on="hour_start",
        how="left",
    )
    cos_mid = cosine(site=site, times=frame["midpoint"])
    mean_cos = 0.5 * (
        cosine(site=site, times=frame["hour_end"] - timedelta(minutes=45))
        + cosine(site=site, times=frame["hour_end"] - timedelta(minutes=15))
    )
    scale = np.where(mean_cos > MIN_COSINE, cos_mid / np.maximum(mean_cos, MIN_COSINE), 1.0)
    fraction = (frame["midpoint"] - frame["hour_start"]).dt.total_minutes().to_numpy() / 60.0
    temperature = frame["temperature_2m_previous_day1"].to_numpy() * fraction + frame[
        "temperature_prev"
    ].to_numpy() * (1.0 - fraction)
    return frame.select(
        "time",
        "init_time",
        icon_ghi=pl.Series(frame["shortwave_radiation_previous_day1"].to_numpy() * scale),
        icon_direct=pl.Series(frame["direct_radiation_previous_day1"].to_numpy() * scale),
        icon_temperature=pl.Series(temperature),
        icon_wind_ms=frame["wind_speed_10m_previous_day1"] / KMH_PER_MS,
    ).with_columns(
        icon_usable=pl.all_horizontal(
            pl.col("icon_ghi").is_finite(), pl.col("icon_temperature").is_finite()
        )
    )


def main() -> None:
    """Write every farm's ICON-D2 table."""
    for site in common.FARMS:
        table = build(site)
        table.write_parquet(common.UKV_DIR / f"{site}_icon.parquet")
        print("built", site, int(table["icon_usable"].sum()), "of", table.height, flush=True)


if __name__ == "__main__":
    main()
