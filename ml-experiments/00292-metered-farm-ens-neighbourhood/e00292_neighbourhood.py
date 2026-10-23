# ruff: noqa: E501
"""Summarise the ensemble's irradiance on the hexagons around each farm for every band-0 half-hour.

The ensemble store holds H3 resolution-5 hexagons. Ring 1 is the 6 neighbours of the farm's own
cell and rings 1 and 2 together are 18 neighbours. Each half-hour ending `T` takes the run whose lead
is 12 to 36 hours, as in experiment 224, with the 3-hour mean irradiance rescaled by the cosine of
the zenith. Per farm the script writes a parquet table of 11 summaries (`SUMMARY_FEATURES`), the
cell counts, and an npy array (rows, 3, 5, 5) of member-mean, member-spread, and presence on the
5 by 5 axial layout of the 19 cells. A farm lacking 5 of its 6 ring-1 cells or 15 of its 19 cells
is reported and skipped.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00292-*/e00292_neighbourhood.py
"""

import os
from datetime import UTC, datetime, timedelta
from typing import Final

os.environ.setdefault("POLARS_MAX_THREADS", "2")

import e00292_common as common
import h3
import numpy as np
import pandas as pd  # noqa: TID251  (pvlib's API takes a pandas index)
import polars as pl
import pvlib
from studies.pv_dataset import pv_sites
from studies.sources import REPO_DATA_DIR

NWP_URI: Final[str] = str(REPO_DATA_DIR / "NWP")
GRID_END: Final[str] = "2026-09-01"
PERIOD_HOURS: Final[int] = 3
MAX_LEAD_HOURS: Final[int] = 42
CLEAR_INDEX: Final[float] = 0.8
N_MEMBERS: Final[int] = 51
ENS_START: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)


def store_cells() -> set[int]:
    """Return every H3 cell in the ensemble store."""
    return set(
        pl.scan_delta(NWP_URI)
        .filter(pl.col("init_time") == ENS_START)
        .select("h3_index")
        .unique()
        .collect()["h3_index"]
        .to_list()
    )


def nearest_cell(*, latitude: float, longitude: float, cells: set[int]) -> int:
    """Return the store's cell whose centre is nearest the farm (experiment 224's rule)."""
    ordered = sorted(cells)
    centres = np.array([h3.cell_to_latlng(h3.int_to_str(c)) for c in ordered])
    distance = (centres[:, 0] - latitude) ** 2 + (
        (centres[:, 1] - longitude) * np.cos(np.radians(latitude))
    ) ** 2
    return int(ordered[int(np.argmin(distance))])


def read_block(*, cells: list[int]) -> pl.DataFrame:
    """Return shortwave and 10 m wind for the cells, all members, leads up to MAX_LEAD_HOURS."""
    return (
        pl.scan_delta(NWP_URI)
        .filter(
            pl.col("h3_index").is_in(cells)
            & (pl.col("init_time") >= ENS_START)
            & (pl.col("init_time") < pl.lit(GRID_END).str.to_datetime(time_zone="UTC"))
            & (pl.col("valid_time") <= pl.col("init_time") + pl.duration(hours=MAX_LEAD_HOURS))
        )
        .select(
            "h3_index",
            "init_time",
            "valid_time",
            "ensemble_member",
            ghi="downward_short_wave_radiation_flux_surface",
            u10="wind_u_10m",
            v10="wind_v_10m",
        )
        .collect()
    )


def sun_grid(*, latitude: float, longitude: float) -> pl.DataFrame:
    """Return zenith and the cosine rescaling of each half-hour, as experiment 224 builds it."""
    start = ENS_START - timedelta(hours=PERIOD_HOURS)
    stop = pl.Series([GRID_END]).str.to_datetime(time_zone="UTC")[0] + timedelta(hours=PERIOD_HOURS)
    times = pl.datetime_range(
        start, stop, interval="30m", time_zone="UTC", eager=True, closed="left"
    ).alias("time")
    index = pd.DatetimeIndex((times - timedelta(minutes=15)).to_numpy(), tz="UTC")
    zenith = pvlib.location.Location(latitude, longitude).get_solarposition(index)[
        "apparent_zenith"
    ]
    grid = pl.DataFrame(
        {"time": times, "cosine": np.clip(np.cos(np.radians(zenith.to_numpy())), 0.0, None)}
    ).with_columns(
        period_end=(pl.col("time") - timedelta(microseconds=1)).dt.truncate("3h")
        + timedelta(hours=PERIOD_HOURS)
    )
    return grid.with_columns(mean_cosine=pl.col("cosine").mean().over("period_end"))


def build_farm(*, site: str, row: dict, cells_in_store: set[int]) -> dict[str, object]:  # noqa: PLR0915
    """Write one farm's summaries and return its cell-availability report."""
    centre = nearest_cell(
        latitude=row["latitude"], longitude=row["longitude"], cells=cells_in_store
    )
    centre_str = h3.int_to_str(centre)
    ring1 = {h3.str_to_int(c) for c in h3.grid_ring(centre_str, 1)}
    disk = {h3.str_to_int(c) for c in h3.grid_disk(centre_str, 2)}
    have_ring1 = sorted(ring1 & cells_in_store)
    have_disk = sorted(disk & cells_in_store)
    report = {"site": site, "ring1_cells": len(have_ring1), "disk_cells": len(have_disk)}
    if len(have_ring1) < common.MIN_RING1_CELLS or len(have_disk) < common.MIN_DISK_CELLS:
        return {**report, "kept": False}
    raw = read_block(cells=have_disk)
    keys = raw.select("init_time", "valid_time").unique().sort("init_time", "valid_time")
    keys = keys.with_row_index("key")
    cell_index = {c: i for i, c in enumerate(have_disk)}
    raw = raw.join(keys, on=["init_time", "valid_time"]).with_columns(
        cell=pl.col("h3_index").replace_strict(cell_index, return_dtype=pl.Int64)
    )
    ghi = np.full((keys.height, len(have_disk), N_MEMBERS), np.nan, dtype=np.float32)
    u10 = np.full_like(ghi, np.nan)
    v10 = np.full_like(ghi, np.nan)
    k = raw["key"].to_numpy()
    c = raw["cell"].to_numpy()
    m = raw["ensemble_member"].to_numpy().astype(np.int64)
    ghi[k, c, m] = raw["ghi"].to_numpy()
    u10[k, c, m] = raw["u10"].to_numpy()
    v10[k, c, m] = raw["v10"].to_numpy()
    del raw
    base = pl.read_parquet(common.E224_DIR / "members" / f"{site}_b0.parquet").select(
        "time", "init_time", "ghi_mean"
    )
    sky = (
        pl.read_parquet(common.FARM_ROWS).filter(pl.col("site") == site).select("time", "clear_sky")
    )
    rows = (
        base.join(sky, on="time", how="left")
        .join(sun_grid(latitude=row["latitude"], longitude=row["longitude"]), on="time", how="left")
        .with_columns(
            period_end=(pl.col("time") - timedelta(microseconds=1)).dt.truncate("3h")
            + timedelta(hours=PERIOD_HOURS)
        )
        .join(keys.rename({"valid_time": "period_end"}), on=["init_time", "period_end"], how="left")
    )
    assert rows["key"].null_count() == 0, "a band-0 row has no ensemble run"
    idx = rows["key"].to_numpy()
    scale = (rows["cosine"] / rows["mean_cosine"]).to_numpy().astype(np.float32)
    g = ghi[idx] * scale[:, None, None]  # (rows, cells, members)
    sky_w = np.clip(rows["clear_sky"].fill_null(1.0).to_numpy(), 1.0, None).astype(np.float32)
    centre_i = cell_index[centre]
    # The centre cell must reproduce experiment 224's ensemble-mean irradiance.
    difference = np.abs(np.nanmean(g[:, centre_i, :], axis=1) - rows["ghi_mean"].to_numpy())
    assert float(np.nanmax(difference)) < 0.5, (
        f"centre cell differs from experiment 224: {np.nanmax(difference)}"
    )
    cell_mean = np.nanmean(g, axis=2)  # (rows, cells)
    cell_std = np.nanstd(g, axis=2)
    in_k1 = np.array([c in ring1 or c == centre for c in have_disk])
    out: dict[str, np.ndarray] = {}
    for name, sel in (("k1", in_k1), ("k2", np.ones(len(have_disk), dtype=bool))):
        block = cell_mean[:, sel]
        out[f"{name}_mean"] = np.nanmean(block, axis=1)
        out[f"{name}_std"] = np.nanstd(block, axis=1)
        out[f"{name}_min"] = np.nanmin(block, axis=1)
        out[f"{name}_max"] = np.nanmax(block, axis=1)
        clear = (g[:, sel, :] / sky_w[:, None, None]) > CLEAR_INDEX
        valid = ~np.isnan(g[:, sel, :])
        out[f"{name}_clear_share"] = (clear & valid).sum(axis=(1, 2)) / np.maximum(
            valid.sum(axis=(1, 2)), 1
        )
    # Upwind: the ring-1 cell in the direction the 10 m wind comes from.
    wind_u = np.nanmean(u10[idx][:, centre_i, :], axis=1)
    wind_v = np.nanmean(v10[idx][:, centre_i, :], axis=1)
    centre_lat, centre_lon = h3.cell_to_latlng(centre_str)
    ring_cells = list(have_ring1)
    vectors = []
    for cell in ring_cells:
        lat, lon = h3.cell_to_latlng(h3.int_to_str(cell))
        vectors.append([(lon - centre_lon) * np.cos(np.radians(centre_lat)), lat - centre_lat])
    vectors = np.array(vectors)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    from_direction = np.column_stack([-wind_u, -wind_v])
    from_direction /= np.maximum(np.linalg.norm(from_direction, axis=1, keepdims=True), 1e-9)
    chosen = np.argmax(from_direction @ vectors.T, axis=1)
    ring_positions = np.array([cell_index[cell] for cell in ring_cells])
    out["upwind_ghi"] = cell_mean[np.arange(len(chosen)), ring_positions[chosen]]
    table = pl.DataFrame(
        {"time": rows["time"], **{f: out[f] for f in common.SUMMARY_FEATURES}}
    ).with_columns(ring1_cells=pl.lit(len(have_ring1)), disk_cells=pl.lit(len(have_disk)))
    common.NEIGHBOUR_DIR.mkdir(parents=True, exist_ok=True)
    table.write_parquet(common.NEIGHBOUR_DIR / f"{site}.parquet")
    # The 5 by 5 axial layout for the convolution arm: channels member mean, member spread, presence.
    image = np.zeros((len(rows), 3, 5, 5), dtype=np.float32)
    origin = h3.cell_to_local_ij(centre_str, centre_str)
    for cell, position in cell_index.items():
        i, j = h3.cell_to_local_ij(centre_str, h3.int_to_str(cell))
        i, j = i - origin[0], j - origin[1]
        image[:, 0, i + 2, j + 2] = cell_mean[:, position]
        image[:, 1, i + 2, j + 2] = cell_std[:, position]
        image[:, 2, i + 2, j + 2] = 1.0
    np.save(common.NEIGHBOUR_DIR / f"{site}_image.npy", np.nan_to_num(image))
    np.save(common.NEIGHBOUR_DIR / f"{site}_image_time.npy", rows["time"].dt.epoch("us").to_numpy())
    # Spatial sanity: nearer cells should agree better with the centre cell.
    near = np.nanmean(
        np.abs(cell_mean[:, [cell_index[c] for c in have_ring1]] - cell_mean[:, [centre_i]])
    )
    far_cells = [cell_index[c] for c in have_disk if c not in ring1 and c != centre]
    far = np.nanmean(np.abs(cell_mean[:, far_cells] - cell_mean[:, [centre_i]]))
    return {
        **report,
        "kept": True,
        "rows": table.height,
        "mean_abs_gap_ring1": float(near),
        "mean_abs_gap_ring2": float(far),
        "null_summaries": int(table.select(pl.all().is_null().sum()).sum_horizontal().item()),
    }


def main() -> None:
    """Build every farm and print the availability report."""
    cells_in_store = store_cells()
    sites = pv_sites()
    reports = []
    for site in common.FARMS:
        row = sites.filter(pl.col("site") == site).row(0, named=True)
        reports.append(build_farm(site=site, row=row, cells_in_store=cells_in_store))
        print(reports[-1], flush=True)
    pl.DataFrame(reports).write_csv(common.DATA_DIR / "neighbourhood_report.csv")


if __name__ == "__main__":
    main()
