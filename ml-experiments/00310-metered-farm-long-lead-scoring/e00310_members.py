# ruff: noqa: E501
"""Pass each of the 51 ECMWF ENS members through the fleet-shape plant for every farm and day 1 to 10.

Copied from experiment 224's member builder and extended to leads up to 252 hours. The ensemble
store holds one run per day (00 UTC) with 3-hourly stamps up to lead 144 hours and 6-hourly stamps
after. Downward shortwave is a mean over the period that ends at the stamp (3 or 6 hours), spread
over the period's half-hours by the cosine of the zenith and split into beam and diffuse with the
Erbs correlation, member by member. Air temperature is an instant, interpolated linearly between
stamps. Day `d` takes, for a half-hour ending `T`, the run whose lead is in
`[12 + 24(d-1), 36 + 24(d-1))` hours, so each half-hour has one run per day.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00310-*/e00310_members.py
"""

import multiprocessing
import os
from datetime import timedelta
from typing import Final

import e00310_common as common
import h3
import numpy as np
import pandas as pd  # noqa: TID251  (pvlib's API takes a pandas index)
import polars as pl
import pvlib
from studies.pv_dataset import pv_sites
from studies.pv_physics import SunAndSky, plane_of_array_w_m2
from studies.sources import REPO_DATA_DIR

NWP_URI: Final[str] = str(REPO_DATA_DIR / "NWP")
N_PROCESSES: Final[int] = 4


def nearest_cell(*, latitude: float, longitude: float, cells: list[int]) -> int:
    """Return the store's H3 cell whose centre is nearest the farm."""
    centres = np.array([h3.cell_to_latlng(h3.int_to_str(c)) for c in cells])
    distance = (centres[:, 0] - latitude) ** 2 + (
        (centres[:, 1] - longitude) * np.cos(np.radians(latitude))
    ) ** 2
    return int(cells[int(np.argmin(distance))])


def read_members(*, cell: int) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Return the (init, valid) keys, and (stamps, 51) arrays of shortwave and temperature."""
    raw = (
        pl.scan_delta(NWP_URI)
        .filter(
            (pl.col("h3_index") == cell)
            & (pl.col("init_time") >= common.ENS_START)
            & (pl.col("init_time") < pl.lit(common.GRID_END).str.to_datetime(time_zone="UTC"))
        )
        .select(
            "init_time",
            "valid_time",
            "ensemble_member",
            ghi="downward_short_wave_radiation_flux_surface",
            temperature_c="temperature_2m",
        )
        .collect()
        .sort("init_time", "valid_time", "ensemble_member")
    )
    counts = raw.group_by("init_time", "valid_time").len()
    complete = counts.filter(pl.col("len") == common.N_MEMBERS).select("init_time", "valid_time")
    raw = raw.join(complete, on=["init_time", "valid_time"], how="semi").sort(
        "init_time", "valid_time", "ensemble_member"
    )
    keys = raw.select("init_time", "valid_time").unique(maintain_order=True)
    n = keys.height
    ghi = raw["ghi"].to_numpy().reshape(n, common.N_MEMBERS).astype(np.float64)
    temperature = raw["temperature_c"].to_numpy().reshape(n, common.N_MEMBERS).astype(np.float64)
    return keys.with_row_index("idx"), ghi, temperature


def sun_grid(*, row: dict[str, float | str]) -> pl.DataFrame:
    """Return zenith, azimuth, and the mean cosine of each half-hour's 3-hour and 6-hour periods."""
    start = common.ENS_START - timedelta(hours=6)
    stop = pl.Series([common.GRID_END]).str.to_datetime(time_zone="UTC")[0] + timedelta(hours=6)
    times = pl.datetime_range(
        start, stop, interval="30m", time_zone="UTC", eager=True, closed="left"
    ).alias("time")
    index = pd.DatetimeIndex((times - timedelta(minutes=15)).to_numpy(), tz="UTC")
    position = pvlib.location.Location(row["latitude"], row["longitude"]).get_solarposition(index)
    zenith = position["apparent_zenith"].to_numpy()
    grid = pl.DataFrame(
        {
            "time": times,
            "zenith_deg": zenith,
            "sun_azimuth_deg": position["azimuth"].to_numpy(),
            "cosine": np.clip(np.cos(np.radians(zenith)), 0.0, None),
        }
    )
    for hours in (3, 6):
        grid = grid.with_columns(
            **{
                f"end{hours}": (pl.col("time") - timedelta(microseconds=1)).dt.truncate(f"{hours}h")
                + timedelta(hours=hours)
            }
        ).with_columns(**{f"mean_cosine{hours}": pl.col("cosine").mean().over(f"end{hours}")})
    return grid


def unit_response(
    *,
    ghi: np.ndarray,
    temperature_c: np.ndarray,
    zenith: np.ndarray,
    azimuth: np.ndarray,
    doy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the fleet-shape unit response, beam, and diffuse for arrays shaped (rows, k)."""
    flat_zenith = np.broadcast_to(zenith[:, None], ghi.shape).ravel()
    flat_doy = np.broadcast_to(doy[:, None], ghi.shape).ravel()
    split = pvlib.irradiance.erbs(ghi=ghi.ravel(), zenith=flat_zenith, datetime_or_doy=flat_doy)
    dni = np.nan_to_num(np.asarray(split["dni"]))
    dhi = np.nan_to_num(np.asarray(split["dhi"]))
    sky = SunAndSky(
        half_hour_end_time=np.zeros(ghi.size),
        ghi_w_m2=ghi.ravel(),
        dni_w_m2=dni,
        dhi_w_m2=dhi,
        zenith_deg=flat_zenith,
        sun_azimuth_deg=np.broadcast_to(azimuth[:, None], ghi.shape).ravel(),
    )
    poa = plane_of_array_w_m2(
        sky=sky,
        tilt_deg=common.FLEET_TILT_DEG,
        azimuth_deg=common.FLEET_AZIMUTH_DEG,
        tracker=False,
    )
    unit = poa / 1000.0
    cell = temperature_c.ravel() + common.CELL_TEMPERATURE_RISE_K * unit
    derating = np.clip(
        1.0 - common.HEAT_PER_KELVIN * (cell - common.REFERENCE_CELL_TEMPERATURE_C), 0.0, None
    )
    return (
        (unit * derating).reshape(ghi.shape),
        dni.reshape(ghi.shape),
        dhi.reshape(ghi.shape),
    )


def build_farm(site: str) -> str:
    """Write every day's table and member array for one farm."""
    row = pv_sites().filter(pl.col("site") == site).row(0, named=True)
    store_cells = (
        pl.scan_delta(NWP_URI)
        .filter(pl.col("init_time") == common.ENS_START)
        .select("h3_index")
        .unique()
        .collect()["h3_index"]
        .to_list()
    )
    cell = nearest_cell(latitude=row["latitude"], longitude=row["longitude"], cells=store_cells)
    keys, ghi_p, temp_p = read_members(cell=cell)
    grid = sun_grid(row=row)
    output = (
        pl.read_parquet(common.FARM_ROWS).filter(pl.col("site") == site).select("time", "output")
    )
    stamp_keys = keys.rename({"valid_time": "period_end"})
    common.MEMBERS_DIR.mkdir(parents=True, exist_ok=True)
    for day in common.DAYS:
        low = 12 + 24 * (day - 1)
        rows = (
            output.join(grid, on="time", how="inner")
            .with_columns(init_time=(pl.col("time") - timedelta(hours=low)).dt.truncate("1d"))
            .with_columns(
                lead3=(pl.col("end3") - pl.col("init_time")).dt.total_hours(),
            )
            .with_columns(
                fine=pl.col("lead3") <= common.THREE_HOURLY_UNTIL_HOURS,
            )
            .with_columns(
                period_end=pl.when(pl.col("fine")).then(pl.col("end3")).otherwise(pl.col("end6")),
                mean_cos=pl.when(pl.col("fine"))
                .then(pl.col("mean_cosine3"))
                .otherwise(pl.col("mean_cosine6")),
                step_hours=pl.when(pl.col("fine")).then(3).otherwise(6),
            )
            .with_columns(
                previous_end=pl.col("period_end") - pl.duration(hours=1) * pl.col("step_hours")
            )
            .join(stamp_keys, on=["init_time", "period_end"], how="inner")
            .join(
                stamp_keys.select("init_time", previous_end="period_end", previous_idx="idx"),
                on=["init_time", "previous_end"],
                how="left",
            )
            .filter(pl.col("mean_cos") > 0)
            .sort("time")
        )
        idx = rows["idx"].to_numpy()
        prev = rows["previous_idx"].fill_null(rows["idx"]).to_numpy()
        step_minutes = rows["step_hours"].to_numpy() * 60.0
        weight = (
            (rows["period_end"] - rows["time"]).dt.total_minutes().to_numpy() + 15.0
        ) / step_minutes
        scale = rows["cosine"].to_numpy() / rows["mean_cos"].to_numpy()
        ghi = ghi_p[idx] * scale[:, None]
        temperature = temp_p[idx] * (1 - weight[:, None]) + temp_p[prev] * weight[:, None]
        zenith = rows["zenith_deg"].to_numpy()
        azimuth = rows["sun_azimuth_deg"].to_numpy()
        doy = rows["time"].dt.ordinal_day().to_numpy()
        unit, _, _ = unit_response(
            ghi=ghi, temperature_c=temperature, zenith=zenith, azimuth=azimuth, doy=doy
        )
        ghi_mean = ghi.mean(axis=1)
        temp_mean = temperature.mean(axis=1)
        unit_mean, dni_mean, dhi_mean = unit_response(
            ghi=ghi_mean[:, None],
            temperature_c=temp_mean[:, None],
            zenith=zenith,
            azimuth=azimuth,
            doy=doy,
        )
        table = rows.select(
            "time",
            "init_time",
            "period_end",
            "output",
            lead_hours=(pl.col("time") - pl.col("init_time")).dt.total_minutes() / 60.0,
            zenith_deg="zenith_deg",
            sun_azimuth_deg="sun_azimuth_deg",
        ).with_columns(
            ghi_mean=pl.Series(ghi_mean),
            ghi_std=pl.Series(ghi.std(axis=1)),
            temperature_mean=pl.Series(temp_mean),
            dni_mean=pl.Series(dni_mean[:, 0]),
            dhi_mean=pl.Series(dhi_mean[:, 0]),
            unit_mean=pl.Series(unit_mean[:, 0]),
            site=pl.lit(site),
            day=pl.lit(day),
        )
        table.write_parquet(common.table_path(site=site, day=day))
        np.save(common.unit_path(site=site, day=day), unit.astype(np.float32))
        print("built", site, day, table.height, flush=True)
    return site


def main() -> None:
    """Build every farm in parallel."""
    sites = list(common.FARMS)
    with multiprocessing.Pool(N_PROCESSES) as pool:
        for site in pool.imap_unordered(build_farm, sites):
            print("done", site, flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
