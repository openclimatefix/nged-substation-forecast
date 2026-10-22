"""Add the network's inputs to each substation's joined arm D frame.

Reads `work/gam_d/<site>.parquet` (from `e00201_join.py`) and writes `work/features/<site>.parquet`
with these extra float32 columns, all known at the forecast origin:

- `drv_temp_c`, `drv_sw_w_m2`: the weather the convex model was driven by. For `era = past_weather`
  this is ERA5 temperature and CAMS irradiance at the valid time (the analysis proxy). For
  `era = ens` it is the mean over the 51 ensemble members of the raw 2-metre temperature and
  shortwave flux at the valid time (experiment 111's store).
- `drv_temp_prev_c`: the same temperature 24 hours before the valid time, `NaN` when that is before
  the run's first forecast step.
- `elevation_deg`, `dark`: the sun's geometry at the valid time (no weather).
- `lag_power`, `lag_hours`: the actual power (points) at `valid_time - lag_hours`, where `lag_hours`
  is the smallest whole number of days that is at least the lead plus the telemetry delay, so the
  reading exists at the origin; `NaN` where there is no reading.
- `true_temp_c`: ERA5 temperature at the valid time, used only by the known-answer gate.

Wind and humidity are not in the stored ensemble arrays, so the weather arms use temperature and
shortwave only.

    uv run --no-sync python ml-experiments/00201-*/e00201_features.py
"""

import os
from pathlib import Path
from typing import Final

import e00201_common as common
import numpy as np
import polars as pl
from studies.dp_assimilation import DEFAULT_DELAY_HOURS

IN_DIR: Final[Path] = common.WORK_DIR / "gam_d"
OUT_DIR: Final[Path] = common.WORK_DIR / "features"
HALF_HOURS_PER_DAY: Final[int] = 48


def ens_means(*, site: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the ensemble-mean temperature and shortwave, shape (runs, 702), of a site."""
    with np.load(common.STORE_DIR / f"{site}.npz") as stored:
        return np.nanmean(stored["temperature"], axis=1), np.nanmean(stored["shortwave"], axis=1)


def add_features(*, site: str) -> pl.DataFrame:
    """Return the joined frame of a site with the network's input columns."""
    cache = common.load_cache()
    features = cache.sites[site]
    frame = pl.read_parquet(IN_DIR / f"{site}.parquet")
    valid = frame["valid_time"].dt.replace_time_zone(None).dt.cast_time_unit("us").to_numpy()
    pos = np.searchsorted(cache.end_times, valid)
    if not (cache.end_times[pos] == valid).all():
        msg = f"{site}: a valid time is not on the cache grid"
        raise ValueError(msg)
    ens = (frame["era"] == "ens").to_numpy()
    step = np.rint(frame["lead_hours"].to_numpy() * 2).astype(np.int64) - 1
    run = frame["run"].to_numpy().astype(np.int64)
    temp_cache, sw_cache = features.temperature_c[pos], features.ghi_w_m2[pos]
    temp_prev = np.full(len(pos), np.nan)
    drv_temp, drv_sw = temp_cache.copy(), sw_cache.copy()
    back = pos - HALF_HOURS_PER_DAY
    temp_prev[:] = np.where(back >= 0, features.temperature_c[np.maximum(back, 0)], np.nan)
    if ens.any():
        ens_t, ens_sw = ens_means(site=site)
        drv_temp[ens] = ens_t[run[ens], step[ens]]
        drv_sw[ens] = ens_sw[run[ens], step[ens]]
        before = step[ens] - HALF_HOURS_PER_DAY
        temp_prev[ens] = np.where(before >= 0, ens_t[run[ens], np.maximum(before, 0)], np.nan)
    # Past-weather rows: yesterday's temperature is known at the origin only beyond 24 h of lead.
    lead = frame["lead_hours"].to_numpy()
    temp_prev[~ens & (lead <= 24.0)] = np.nan
    lag_days = np.ceil((lead + DEFAULT_DELAY_HOURS) / 24.0).astype(np.int64)
    lag_pos = pos - lag_days * HALF_HOURS_PER_DAY
    lag_power = np.where(lag_pos >= 0, features.y[np.maximum(lag_pos, 0)], np.nan)
    lag_power = np.where(features.present[np.maximum(lag_pos, 0)], lag_power, np.nan)
    return frame.with_columns(
        pl.Series("drv_temp_c", drv_temp.astype(np.float32)),
        pl.Series("drv_sw_w_m2", drv_sw.astype(np.float32)),
        pl.Series("drv_temp_prev_c", temp_prev.astype(np.float32)),
        pl.Series("elevation_deg", features.elevation_deg[pos].astype(np.float32)),
        pl.Series("dark", features.dark[pos]),
        pl.Series("lag_power", lag_power.astype(np.float32)),
        pl.Series("lag_hours", (lag_days * 24).astype(np.float32)),
        pl.Series("true_temp_c", temp_cache.astype(np.float32)),
    )


def main() -> None:
    """Write the feature frame of every substation."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for site in common.SCORED_SITES:
        frame = add_features(site=site)
        target = OUT_DIR / f"{site}.parquet"
        temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        frame.write_parquet(temporary)
        temporary.rename(target)
        nulls = {
            c: int(frame[c].null_count()) + int(frame[c].is_nan().sum())
            for c in ("drv_temp_c", "drv_sw_w_m2", "lag_power")
        }
        print(f"{site}: {frame.height:,} rows, non-finite {nulls}", flush=True)


if __name__ == "__main__":
    main()
