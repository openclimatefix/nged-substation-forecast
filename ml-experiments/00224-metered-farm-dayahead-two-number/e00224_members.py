"""Pass each of the 51 ECMWF ENS members through the fleet-shape plant for every farm and lead band.

The ensemble store holds one run per day (00:00 UTC) with 3-hourly stamps. Downward shortwave is a
mean over the 3 hours that end at the stamp, so it is spread over the six half-hours by the cosine
of the zenith and split into beam and diffuse with the Erbs correlation, member by member. Air
temperature is an instant, interpolated linearly between stamps. Band `j` takes, for a half-hour
ending `T`, the run whose lead is in `[12 + 24j, 36 + 24j)` hours, so each half-hour has one run.

The output per (farm, band) is a parquet table of the ensemble-mean weather and the plant's unit
response to the mean, plus an array of the unit response to each member, written under
`data/ml-experiments/00224-*/members/`. "Unit response" is plant power over capacity per unit scale
(plane-of-array irradiance over 1000 W/m² times the heat derating); the scale is fitted later.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00224-*/e00224_members.py
"""

import multiprocessing
from datetime import timedelta
from pathlib import Path
from typing import Final

import e00224_common as common
import h3
import numpy as np
import pandas as pd  # noqa: TID251  (pvlib's API takes a pandas index)
import polars as pl
import pvlib
from studies.pv_dataset import pv_sites
from studies.pv_physics import SunAndSky, plane_of_array_w_m2
from studies.sources import REPO_DATA_DIR

NWP_URI: Final[str] = str(REPO_DATA_DIR / "NWP")
PERIOD_HOURS: Final[int] = 3
HALF_HOURS_PER_PERIOD: Final[int] = 6
N_PROCESSES: Final[int] = 4
GRID_END: Final[str] = "2026-09-01"


def table_path(*, site: str, band: int) -> Path:
    """Return the parquet path of one (farm, band) table."""
    return common.MEMBERS_DIR / f"{site}_b{band}.parquet"


def unit_path(*, site: str, band: int) -> Path:
    """Return the npy path of one (farm, band) member-response array."""
    return common.MEMBERS_DIR / f"{site}_b{band}_unit.npy"


def nearest_cell(*, latitude: float, longitude: float, cells: list[int]) -> int:
    """Return the store's H3 cell whose centre is nearest the farm."""
    centres = np.array([h3.cell_to_latlng(h3.int_to_str(c)) for c in cells])
    distance = (centres[:, 0] - latitude) ** 2 + (
        (centres[:, 1] - longitude) * np.cos(np.radians(latitude))
    ) ** 2
    return int(cells[int(np.argmin(distance))])


def read_members(*, cell: int) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Return the (init, valid) keys, and (periods, 51) arrays of shortwave and temperature."""
    raw = (
        pl.scan_delta(NWP_URI)
        .filter(
            (pl.col("h3_index") == cell)
            & (pl.col("init_time") >= common.ENS_START)
            & (pl.col("init_time") < pl.lit(GRID_END).str.to_datetime(time_zone="UTC"))
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


def sun_grid(*, row: dict[str, float | str], first: pl.Series) -> pl.DataFrame:
    """Return zenith, azimuth, and the mean cosine of each half-hour's 3-hour period."""
    start = common.ENS_START - timedelta(hours=PERIOD_HOURS)
    stop = pl.Series([GRID_END]).str.to_datetime(time_zone="UTC")[0] + timedelta(hours=PERIOD_HOURS)
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
    ).with_columns(
        period_end=(pl.col("time") - timedelta(microseconds=1)).dt.truncate("3h")
        + timedelta(hours=PERIOD_HOURS)
    )
    return grid.with_columns(mean_cosine=pl.col("cosine").mean().over("period_end"))


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
    """Write every band's table and member array for one farm."""
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
    grid = sun_grid(row=row, first=keys["init_time"])
    output = common.farm_output().filter(pl.col("site") == site).select("time", "output")
    previous = keys.select(
        "init_time",
        period_end=pl.col("valid_time") + timedelta(hours=PERIOD_HOURS),
        previous_idx="idx",
    )
    common.MEMBERS_DIR.mkdir(parents=True, exist_ok=True)
    for band in range(len(common.BAND_SHIFT_DAYS)):
        shift = common.BAND_SHIFT_DAYS[band]
        low = 12 + 24 * shift
        rows = (
            output.join(grid, on="time", how="inner")
            .with_columns(
                init_time=(pl.col("time") - timedelta(hours=low)).dt.truncate("1d"),
            )
            .join(
                keys.rename({"valid_time": "period_end"}),
                on=["init_time", "period_end"],
                how="inner",
            )
            .join(previous, on=["init_time", "period_end"], how="left")
            .filter(pl.col("mean_cosine") > 0)
            .sort("time")
        )
        idx = rows["idx"].to_numpy()
        prev = rows["previous_idx"].fill_null(rows["idx"]).to_numpy()
        weight = (
            (rows["period_end"] - rows["time"]).dt.total_minutes().to_numpy() / 30.0 + 0.5
        ) / HALF_HOURS_PER_PERIOD
        scale = rows["cosine"].to_numpy() / rows["mean_cosine"].to_numpy()
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
            band=pl.lit(band),
        )
        table.write_parquet(table_path(site=site, band=band))
        np.save(unit_path(site=site, band=band), unit.astype(np.float32))
    return site


def main() -> None:
    """Build every farm in parallel."""
    sites = list(common.FARMS)
    with multiprocessing.Pool(N_PROCESSES) as pool:
        for site in pool.imap_unordered(build_farm, sites):
            print("done", site, flush=True)


if __name__ == "__main__":
    main()
