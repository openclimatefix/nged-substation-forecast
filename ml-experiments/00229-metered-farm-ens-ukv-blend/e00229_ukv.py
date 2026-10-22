"""Read the UKV-CEDA variables at each farm's nearest cell and build the per-row UKV tables.

Step 1 reads every complete 00 UTC UKV run that a farm row needs, for the 22 variables of
`e00229_common.UKV_VARIABLES`, at the six farms' nearest 2 km cells. Step 2 interpolates each
variable to the midpoint of each half-hour row, with shortwave scaled by the cosine of the solar
zenith so that the sun's movement between steps is not read as cloud. Wind direction is
interpolated as a vector.

    systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00229-*/e00229_ukv.py
"""

import sys
from datetime import UTC, datetime, timedelta
from typing import Final

import e00229_common as common
import icechunk
import numpy as np
import pandas as pd  # noqa: TID251  (pvlib's API takes a pandas index)
import polars as pl
import pvlib
import zarr
from studies.grid_sampling import distance_matrix_km
from studies.pv_dataset import pv_sites
from studies.ukv_ceda_profiles import STATUS_COMPLETE
from studies.ukv_ceda_stores import Store, make_stores, store_index

RAW_PATH: Final = common.UKV_DIR / "raw_runs.npz"
MIN_COSINE: Final[float] = 0.02
MAX_CELL_DISTANCE_KM: Final[float] = 2.0
FEATURE_PERIOD_END: Final[datetime] = datetime(2026, 9, 2, tzinfo=UTC)


def open_stores() -> tuple[tuple[Store, ...], zarr.Group]:
    """Open the three UKV-CEDA stores read-only; return them and the first store's group."""
    groups, statuses = [], []
    for name in common.UKV_STORES:
        repository = icechunk.Repository.open(
            icechunk.local_filesystem_storage(str(common.NWP_DIR / name / "store"))
        )
        group = zarr.open_group(repository.readonly_session(branch="main").store, mode="r")
        groups.append(group)
        status = group["status"]
        assert isinstance(status, zarr.Array)
        statuses.append(np.asarray(status[:], dtype=np.int8))
    return make_stores(groups=groups, statuses=statuses), groups[0]


def farm_cells(*, group: zarr.Group) -> np.ndarray:
    """Return each farm's nearest cell index in `FARMS` order; coordinates are never printed."""
    latitude, longitude = group["cell_latitude"], group["cell_longitude"]
    assert isinstance(latitude, zarr.Array)
    assert isinstance(longitude, zarr.Array)
    cells = pl.DataFrame(
        {"latitude": np.asarray(latitude[:]), "longitude": np.asarray(longitude[:])}
    )
    sites = pv_sites().filter(pl.col("site").is_in(common.FARMS)).sort("site")
    distance = distance_matrix_km(sites=sites, cells=cells)
    if float(distance.min(axis=1).max()) > MAX_CELL_DISTANCE_KM:
        msg = "a farm lies more than 2 km from every UKV cell"
        raise ValueError(msg)
    return distance.argmin(axis=1)


def needed_runs() -> list[datetime]:
    """Return the 00 UTC init times of every ensemble run that a farm row uses."""
    runs: set[datetime] = set()
    for site in common.FARMS:
        for band in common.BANDS:
            table = pl.read_parquet(
                common.members_table(site=site, band=band), columns=["init_time"]
            )
            runs.update(table["init_time"].unique().to_list())
    return sorted(runs)


def read_raw() -> None:
    """Read every needed complete run at the six farm cells and save it."""
    stores, group = open_stores()
    cells = farm_cells(group=group)
    runs = needed_runs()
    values = {
        name: np.full((len(runs), 55, len(cells)), np.nan, dtype=np.float32)
        for name in common.UKV_VARIABLES
    }
    complete = np.zeros(len(runs), dtype=bool)
    for i, run in enumerate(runs):
        slot = common.UKV_SLOTS_PER_DAY * (run - common.UKV_START).days
        position = store_index(stores=stores, slot=slot)
        if position is None or stores[position].statuses[slot] != STATUS_COMPLETE:
            continue
        complete[i] = True
        for name in common.UKV_VARIABLES:
            block = stores[position].group[name]
            assert isinstance(block, zarr.Array)
            values[name][i] = np.asarray(block[slot])[:, cells]
        if i % 100 == 0:
            print("read", i, "of", len(runs), flush=True)
    common.UKV_DIR.mkdir(parents=True, exist_ok=True)
    stamps = np.array([r.timestamp() for r in runs])
    arrays: dict[str, np.ndarray] = {"runs": stamps, "complete": complete, **values}
    np.savez(RAW_PATH, **arrays)  # ty: ignore[invalid-argument-type]
    print("complete runs", int(complete.sum()), "of", len(runs))


def zenith_cosine(*, site: str, instants: pd.DatetimeIndex) -> np.ndarray:
    """Return the cosine of the apparent solar zenith at a farm for the given instants."""
    row = pv_sites().filter(pl.col("site") == site).row(0, named=True)
    position = pvlib.location.Location(row["latitude"], row["longitude"]).get_solarposition(
        instants
    )
    return np.cos(np.radians(position["apparent_zenith"].to_numpy()))


def build_tables(*, only: str | None = None) -> None:
    """Write one parquet of interpolated UKV columns per farm and band (one farm if `only`)."""
    raw = np.load(RAW_PATH)
    run_stamps = raw["runs"]
    complete = raw["complete"]
    arrays = {name: raw[name] for name in common.UKV_VARIABLES}
    native = np.array(common.NATIVE_STEPS)
    for farm_index, site in enumerate(common.FARMS):
        if only is not None and site != only:
            continue
        for band in common.BANDS:
            table = pl.read_parquet(common.members_table(site=site, band=band)).select(
                "time", "init_time"
            )
            run_index = np.searchsorted(run_stamps, table["init_time"].dt.epoch("s").to_numpy())
            midpoint = (
                table["time"] - timedelta(minutes=15) - table["init_time"]
            ).dt.total_minutes().to_numpy() / 60.0
            usable = complete[run_index] & (midpoint >= 0) & (midpoint <= native[-1])
            # Cosine of the zenith at the midpoint and at every native step of the row's run.
            midpoint_times = pd.DatetimeIndex(
                (table["time"] - timedelta(minutes=15)).to_numpy(), tz="UTC"
            )
            cos_mid = np.clip(zenith_cosine(site=site, instants=midpoint_times), 0.0, None)
            unique_runs = np.unique(run_index)
            cos_steps = {
                int(r): np.clip(
                    zenith_cosine(
                        site=site,
                        instants=pd.DatetimeIndex(
                            [
                                pd.Timestamp(run_stamps[r], unit="s", tz="UTC")
                                + pd.Timedelta(hours=int(h))
                                for h in native
                            ]
                        ),
                    ),
                    0.0,
                    None,
                )
                for r in unique_runs
            }
            columns: dict[str, np.ndarray] = {}
            for name in common.UKV_VARIABLES:
                out = np.full(len(table), np.nan)
                for r in unique_runs:
                    rows = np.flatnonzero((run_index == r) & usable)
                    if len(rows) == 0:
                        continue
                    series = arrays[name][r, :, farm_index]
                    base = np.interp(midpoint[rows], native, series[native])
                    out[rows] = base
                columns[name] = out
            # Shortwave rescaled by the cosine of the zenith (the sun moves between hourly steps).
            sw = np.full(len(table), np.nan)
            for r in unique_runs:
                rows = np.flatnonzero((run_index == r) & usable)
                if len(rows) == 0:
                    continue
                series = arrays["shortwave_down"][r, :, farm_index]
                kt = np.where(
                    cos_steps[int(r)] > MIN_COSINE,
                    series[native] / np.maximum(cos_steps[int(r)], MIN_COSINE),
                    np.nan,
                )
                # Clear-sky-index-like ratio, filled across the night by the nearest daytime ratio.
                good = np.isfinite(kt)
                if good.sum() < 2:
                    sw[rows] = columns["shortwave_down"][rows]
                    continue
                kt_mid = np.interp(midpoint[rows], native[good], kt[good])
                sw[rows] = kt_mid * cos_mid[rows]
            columns["shortwave_down"] = sw
            # Wind direction as a vector with the speed.
            speed, direction = columns["wind_speed_10m"], np.radians(columns["wind_direction_10m"])
            columns["wind_u"] = -speed * np.sin(direction)
            columns["wind_v"] = -speed * np.cos(direction)
            del columns["wind_direction_10m"]
            frame = pl.DataFrame(
                {"time": table["time"], "init_time": table["init_time"]}
            ).with_columns(
                site=pl.lit(site),
                band=pl.lit(band),
                ukv_usable=pl.Series(usable),
                **{f"ukv_{k}": pl.Series(v) for k, v in columns.items()},
            )
            frame.write_parquet(common.UKV_DIR / f"{site}_b{band}.parquet")
            print("built", site, band, int(usable.sum()), "of", len(table), flush=True)


if __name__ == "__main__":
    if not RAW_PATH.exists():
        read_raw()
    build_tables(only=sys.argv[1] if len(sys.argv) > 1 else None)
