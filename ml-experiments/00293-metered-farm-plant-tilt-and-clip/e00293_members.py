# ruff: noqa: E501
"""Recompute each farm's member responses at tilts 35 and 20 degrees (experiment 293).

Copied from experiment 224's `e00224_members.py` for band 0 only. For each farm it rebuilds the 51
members' shortwave and temperature, checks that the rows and the tilt-35 responses match experiment
224's stored table and array, and writes one npz per farm under `data/ml-experiments/00293-*/shapes/`
holding the ensemble-mean unit response, the standard deviation of the members' unit responses
at both tilts, and the members' shortwave (for the pooled shape).

    systemd-run --user --scope -p MemoryMax=12G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00293-*/e00293_members.py
"""

import multiprocessing
from datetime import timedelta
from typing import Final

import e00293_common as common
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
TILTS_DEG: Final[tuple[float, ...]] = (35.0, 20.0)


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


def sun_grid(*, row: dict[str, float | str]) -> pl.DataFrame:
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
    tilt_deg: float,
) -> np.ndarray:
    """Return the unit response at a tilt (south-facing) for arrays shaped (rows, k)."""
    flat_zenith = np.broadcast_to(zenith[:, None], ghi.shape).ravel()
    flat_doy = np.broadcast_to(doy[:, None], ghi.shape).ravel()
    split = pvlib.irradiance.erbs(ghi=ghi.ravel(), zenith=flat_zenith, datetime_or_doy=flat_doy)
    sky = SunAndSky(
        half_hour_end_time=np.zeros(ghi.size),
        ghi_w_m2=ghi.ravel(),
        dni_w_m2=np.nan_to_num(np.asarray(split["dni"])),
        dhi_w_m2=np.nan_to_num(np.asarray(split["dhi"])),
        zenith_deg=flat_zenith,
        sun_azimuth_deg=np.broadcast_to(azimuth[:, None], ghi.shape).ravel(),
    )
    poa = plane_of_array_w_m2(
        sky=sky, tilt_deg=tilt_deg, azimuth_deg=common.FLEET_AZIMUTH_DEG, tracker=False
    )
    unit = poa / 1000.0
    cell = temperature_c.ravel() + common.CELL_TEMPERATURE_RISE_K * unit
    derating = np.clip(
        1.0 - common.HEAT_PER_KELVIN * (cell - common.REFERENCE_CELL_TEMPERATURE_C), 0.0, None
    )
    return (unit * derating).reshape(ghi.shape)


def build_farm(site: str) -> str:
    """Write the shapes file for one farm and check it against experiment 224's band-0 table."""
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
        pl.read_parquet(common.FARM_ROWS)
        .filter(pl.col("site") == site)
        .select("time", "output")
    )
    previous = keys.select(
        "init_time",
        period_end=pl.col("valid_time") + timedelta(hours=PERIOD_HOURS),
        previous_idx="idx",
    )
    rows = (
        output.join(grid, on="time", how="inner")
        .with_columns(init_time=(pl.col("time") - timedelta(hours=12)).dt.truncate("1d"))
        .join(keys.rename({"valid_time": "period_end"}), on=["init_time", "period_end"], how="inner")
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
    ghi_mean, temp_mean = ghi.mean(axis=1), temperature.mean(axis=1)
    arrays: dict[str, np.ndarray] = {"ghi_members": ghi.astype(np.float32)}
    for tilt in TILTS_DEG:
        members = unit_response(
            ghi=ghi, temperature_c=temperature, zenith=zenith, azimuth=azimuth, doy=doy, tilt_deg=tilt
        )
        mean_weather = unit_response(
            ghi=ghi_mean[:, None],
            temperature_c=temp_mean[:, None],
            zenith=zenith,
            azimuth=azimuth,
            doy=doy,
            tilt_deg=tilt,
        )
        arrays[f"unit_mean_{int(tilt)}"] = mean_weather[:, 0]
        arrays[f"unit_std_{int(tilt)}"] = members.std(axis=1)
    stored = pl.read_parquet(common.E224_DIR / "members" / f"{site}_b0.parquet")
    if not (
        stored["time"].equals(rows["time"])
        and np.allclose(stored["unit_mean"].to_numpy(), arrays["unit_mean_35"], atol=1e-6)
    ):
        msg = f"farm {site}: rebuilt rows or tilt-35 mean response differ from experiment 224"
        raise ValueError(msg)
    stored_members = np.load(common.E224_DIR / "members" / f"{site}_b0_unit.npy")
    drift = float(np.abs(stored_members.std(axis=1) - arrays["unit_std_35"]).max())
    if drift > 1e-4:
        msg = f"farm {site}: tilt-35 member spread differs from experiment 224 by {drift}"
        raise ValueError(msg)
    common.SHAPES_DIR.mkdir(parents=True, exist_ok=True)
    path = common.SHAPES_DIR / f"{site}.npz"
    np.savez(path.with_suffix(".tmp.npz"), **arrays)
    path.with_suffix(".tmp.npz").replace(path)
    return f"{site} rows={rows.height} max tilt-35 spread drift={drift:.2e}"


def main() -> None:
    """Build every farm in parallel."""
    with multiprocessing.Pool(N_PROCESSES) as pool:
        for line in pool.imap_unordered(build_farm, common.FARMS):
            print("done", line, flush=True)


if __name__ == "__main__":
    main()
