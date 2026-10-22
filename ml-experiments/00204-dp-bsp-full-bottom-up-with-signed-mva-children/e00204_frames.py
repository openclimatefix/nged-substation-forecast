"""Cut the 51-member ensemble weather at the two BSP locations into one array file per series.

For each BSP the file holds every daily run from 2024-04-01 to 2026-06-30: the run's origin, and the
51 members' temperature and shortwave flux at the 702 half-hours after the origin, in the layout of
experiment 111's store. The remainder series of a BSP reuse its file under their own labels.

Runs of the scored year already have frames (experiments 48 and the stage 1 pilot), which hold both
BSPs. The 456 earlier runs are cut from the archive of ensemble forecasts at the BSPs' H3 cells with
production's half-hourly upsampling, as experiment 111's frame builder does for the Primaries.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00161-*/e00204_frames.py [--workers 3]
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Final

import e00204_common as common
import h3
import numpy as np
import polars as pl
from ml_core.features._nwp import _upsample_nwp_to_half_hourly
from studies.dp_features import default_input_paths
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

N_MEMBERS: Final[int] = 51
N_TARGETS: Final[int] = 702
"""Half-hours in a frame: 351 hours after the origin."""
H3_RESOLUTION: Final[int] = 5
HORIZON_HOURS: Final[int] = 351
FRAME_FOLDERS: Final[tuple[Path, ...]] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00048-weekly-retraining-confirmation-and-extension-demand-generators"
    / "work"
    / "frames_demand",
    REPO_DATA_DIR / "studies/per_study/stage1_pilot/demand/ens_every5d/val",
)
ARCHIVE_DIR: Final[Path] = REPO_DATA_DIR / "NWP" / "nwp_model_id=ECMWF_ENS_0_25_degree"
SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "h3_index",
    "ensemble_member",
    "valid_time",
    "temperature_2m",
    "downward_short_wave_radiation_flux_surface",
    "downward_long_wave_radiation_flux_surface",
    "pressure_surface",
    "pressure_reduced_to_mean_sea_level",
    "precipitation_surface",
    "categorical_precipitation_type_surface",
    "dew_point_temperature_2m",
    "wind_u_10m",
    "wind_v_10m",
    "wind_u_100m",
    "wind_v_100m",
)
RENAMES: Final[dict[str, str]] = {
    "downward_short_wave_radiation_flux_surface": "shortwave",
}


def bsp_table() -> pl.DataFrame:
    """Return the BSPs' `site`, `time_series_id`, coordinates, and H3 cell."""
    series = pl.read_parquet(default_input_paths().series_table).filter(
        pl.col("site").is_in(list(common.BSPS))
    )
    return series.select("site", "time_series_id", "latitude", "longitude").with_columns(
        h3_res_5=pl.struct("latitude", "longitude").map_elements(
            lambda row: h3.str_to_int(
                h3.latlng_to_cell(row["latitude"], row["longitude"], H3_RESOLUTION)
            ),
            return_dtype=pl.Int64,
        ),
        time_series_id=pl.col("time_series_id").cast(pl.Int32),
    )


def stored_frame(*, run: datetime) -> Path | None:
    """Return the stored frame of a run, or `None` when the run is not in a stored folder."""
    name = f"init={run:%Y-%m-%d}.parquet"
    return next((f / name for f in FRAME_FOLDERS if (f / name).exists()), None)


def archive_frame(*, run: datetime, sites: pl.DataFrame) -> pl.DataFrame:
    """Return a run's weather at the BSPs from the archive, on the half-hourly grid."""
    stamp = f"{run:%Y-%m-%d} 00:00:00.000000".replace(" ", "%20").replace(":", "%3A")
    cells = sites["h3_res_5"].to_list()
    native = (
        pl.scan_parquet(ARCHIVE_DIR / f"init_time={stamp}" / "*.parquet")
        .select(SOURCE_COLUMNS)
        .filter(pl.col("h3_index").is_in(cells))
        .join(
            sites.lazy().select("time_series_id", "h3_res_5"),
            left_on="h3_index",
            right_on="h3_res_5",
            how="inner",
        )
        .drop("h3_index")
        .with_columns(nwp_init_time=pl.lit(run).cast(pl.Datetime("us", "UTC")))
    )
    coordinates = sites.lazy().select("time_series_id", "latitude", "longitude")
    half_hourly = (
        _upsample_nwp_to_half_hourly(native, solar_coordinates=coordinates)
        .collect()
        .rename(RENAMES)
    )
    origin = run + common.PUBLICATION_DELAY
    return half_hourly.with_columns(
        lead_hours=(
            pl.col("valid_time") - pl.lit(origin).cast(pl.Datetime("us", "UTC"))
        ).dt.total_minutes()
        / 60.0
    ).filter((pl.col("lead_hours") > 0.0) & (pl.col("lead_hours") <= HORIZON_HOURS))


def read_run(*, run: datetime, sites: pl.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return one run's temperature and shortwave per BSP, shape (members, targets) each."""
    path = stored_frame(run=run)
    ids = dict(zip(sites["time_series_id"].to_list(), sites["site"].to_list(), strict=True))
    if path is not None:
        frame = pl.read_parquet(
            path,
            columns=[
                "time_series_id",
                "ensemble_member",
                "valid_time",
                "temperature_2m",
                "shortwave",
            ],
        ).filter(pl.col("time_series_id").is_in(list(ids)))
    else:
        frame = archive_frame(run=run, sites=sites)
    out = {}
    for series_id, label in ids.items():
        rows = frame.filter(pl.col("time_series_id") == series_id).sort(
            "ensemble_member", "valid_time"
        )
        if rows.height != N_MEMBERS * N_TARGETS:
            msg = f"{label} on {run:%Y-%m-%d} has {rows.height} rows, not {N_MEMBERS * N_TARGETS}"
            raise ValueError(msg)
        if rows.select("ensemble_member", "valid_time").is_duplicated().any():
            msg = f"{label} on {run:%Y-%m-%d} repeats a key"
            raise ValueError(msg)
        out[label] = (
            rows["temperature_2m"].to_numpy().astype(np.float32).reshape(N_MEMBERS, N_TARGETS),
            rows["shortwave"].to_numpy().astype(np.float32).reshape(N_MEMBERS, N_TARGETS),
        )
    return out


_SITES: dict[str, pl.DataFrame] = {}


def _read(run: datetime) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if "sites" not in _SITES:
        _SITES["sites"] = bsp_table()
    return read_run(run=run, sites=_SITES["sites"])


def main() -> None:
    """Read every run and write one array file per BSP and per remainder label."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--smoke", action="store_true", help="read two archive runs and print")
    args = parser.parse_args()
    os.environ["POLARS_MAX_THREADS"] = str(max(1, 6 // args.workers))
    runs = common.run_dates()
    if args.smoke:
        started = time.monotonic()
        result = _read(runs[0])
        print(
            {
                k: (v[0].shape, float(np.nanmean(v[0])), float(np.nanmean(v[1])))
                for k, v in result.items()
            }
        )
        print(f"{time.monotonic() - started:.0f} s for one archive run")
        return
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_read, runs, chunksize=8))
    common.STORE_DIR.mkdir(parents=True, exist_ok=True)
    origins = np.array(
        [np.datetime64((r + common.PUBLICATION_DELAY).replace(tzinfo=None), "us") for r in runs]
    )
    for bsp in common.BSPS:
        temperature = np.stack([result[bsp][0] for result in results])
        shortwave = np.stack([result[bsp][1] for result in results])
        for suffix in ("", *common.VARIANT_SUFFIX.values()):
            target = common.STORE_DIR / f"{bsp}{suffix}.npz"
            temporary = target.with_name(f"{target.stem}.tmp.npz")
            np.savez(temporary, origins=origins, temperature=temperature, shortwave=shortwave)
            temporary.rename(target)
        print(
            f"{bsp}: {temperature.shape}, shortwave nulls {int(np.isnan(shortwave).sum())}",
            flush=True,
        )
    print(f"done in {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()
