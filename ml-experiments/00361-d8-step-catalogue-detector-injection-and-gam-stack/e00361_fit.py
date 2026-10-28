"""Step 3: refit D8's monthly stack with detected steps removed from the training target.

Copies experiment 354's `e00354_fit.py` fit loop (fit (b): from 2023-09 with a level step per hour
band at 2025-04-01) and edits the training target at each cut. Arms:

| Arm | Training target at each cut |
|---|---|
| `ctl` | raw (reproduces experiment 354's fit (b)) |
| `t1` | each detected segment's level added back to the rows of the segment |
| `t0` | the rows of each detected segment masked out of the fit |

The detector sees only rows ending at or before the cut (`e00361_detect`). A drop's level is used
where a drop covers the row, and a step's level elsewhere.

    PYTHONPATH=<the folders of experiments 252, 264, 354, and 361> \
        OMP_NUM_THREADS=4 systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python -W ignore ml-experiments/00361-*/e00361_fit.py --arms ctl t1 t0
"""

import argparse
import os

os.environ.setdefault("E00252_SOURCE", "mva")
os.environ.setdefault("E00252_ARM", "a1")

import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import e00252_common as common  # ty: ignore[unresolved-import]
import e00252_tall as tall  # ty: ignore[unresolved-import]
import e00252s2_corrections as corr  # ty: ignore[unresolved-import]
import e00252s2_run as stage2  # ty: ignore[unresolved-import]
import e00354_fit as f354  # ty: ignore[unresolved-import]
import e00354_step as step  # ty: ignore[unresolved-import]
import e00361_detect as detect
import numpy as np
import polars as pl
import torch
from studies.dp_ens_weather import replace_site
from studies.dp_features import naive_utc
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SITE: Final[str] = "D8"
WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00361-d8-step-catalogue-detector-injection-and-gam-stack" / "work"
)
ARMS: Final[tuple[str, ...]] = ("ctl", "t1", "t0")


def arm_dir(*, arm: str) -> Path:
    """Return the folder of an arm's fits and forecasts."""
    return WORK / f"fits_{arm}"


def removal(*, cache: Any, cut: datetime) -> tuple[np.ndarray, np.ndarray]:
    """Return, for each grid row, whether a segment covers it and the segment's level in points.

    Detection sees only rows ending at or before `cut`, on D8's power in MW (NaN before the fit
    history starts).
    """
    features = cache.sites[SITE]
    capacity = float(features.table.effective_capacity_mw)
    known = cache.end_times <= naive_utc(moment=cut)
    n = int(known.sum())
    power = (features.y[:n] * capacity / 100.0).astype(np.float64)
    flag = np.zeros(len(cache.end_times), dtype=bool)
    level_mw = np.zeros(len(cache.end_times))
    steps, drops = detect.detect(power_mw=power), detect.detect_drops(power_mw=power)
    for seg in (steps, drops):  # drops last, so a drop's level wins where both cover a row
        for s, e, lv in zip(seg.start, seg.end, seg.level_mw, strict=True):
            flag[s:e] = True
            level_mw[s:e] = lv
    return flag, level_mw * 100.0 / capacity


def edit_cache(*, cache, arm: str, cut):  # noqa: ANN001, ANN201
    """Return the cache with the arm's training-target edit at `cut`."""
    if arm == "ctl":
        return cache
    flag, level_points = removal(cache=cache, cut=cut)
    features = cache.sites[SITE]
    if arm == "t1":
        return replace_site(
            cache=cache, site=SITE, y=np.where(flag, features.y - level_points, features.y)
        )
    return replace_site(
        cache=cache,
        site=SITE,
        y=np.where(flag, np.nan, features.y),
        present=features.present & ~flag,
        default_rows=features.default_rows & ~flag,
    )


def fit_arm_cuts(*, arm: str) -> list:
    """Fit T-all at every stored monthly cut with the arm's target edit."""
    settings = f354.ARMS["b"]
    common.D8_FIRST_DAY = settings["start"]
    common.WINDOW_MONTHS = settings["window"]
    base_cache = common.variant_cache(cache=common.load_cache(), site=SITE)
    vintages = common.vintages_for(cache=base_cache)
    cuts = [f.cut for f in pickle.loads(tall.fit_path(cadence="monthly", site=SITE).read_bytes())]
    fits = []
    for cut in cuts:
        cache = edit_cache(cache=base_cache, arm=arm, cut=cut)
        base = tall.design.build_base(
            cache=cache, vintages=vintages, site=SITE, cut=cut, fitted=common.FITTED
        )
        spec = step.step_arm(
            base=tall.design.interaction_arm(name="t_all", penalty_weight=common.T_ALL_LAMBDA),
            step_from=settings["step"],
        )
        fits.append(
            tall.TallFit(
                site=SITE,
                cut=cut,
                layout=base.layout,
                pinned=base.pinned,
                arm=spec,
                fit=tall.design.fit_arm(cache=cache, base=base, arm=spec),
            )
        )
    return fits


def main() -> None:
    """Fit and forecast every requested arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    args = parser.parse_args()
    torch.set_num_threads(4)
    step.install()
    f354.WORK = WORK
    f354.ARMS["b"] = f354.ARMS["b"]
    for arm in args.arms:
        started = time.monotonic()
        folder = arm_dir(arm=arm)
        folder.mkdir(parents=True, exist_ok=True)
        fits = fit_arm_cuts(arm=arm)
        (folder / "tall_fits.pkl").write_bytes(pickle.dumps(fits))
        print(arm, "fits", len(fits), f"{time.monotonic() - started:.0f} s", flush=True)
        tall.FITS_DIR = folder / "tall_fits_root"
        target = tall.fit_path(cadence="monthly", site=SITE)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(pickle.dumps(fits))
        common.D8_FIRST_DAY = f354.ARMS["b"]["start"]
        frame = f354.forecast_scored(arm="b")
        base = frame.with_columns(period=pl.lit("scored"))
        base.write_parquet(folder / "base_raw_scored_D8.parquet")
        arms = stage2.build_arms(site=SITE, frame=corr.add_calendar(frame=base))
        for name in ("a", "d"):
            arms[name].write_parquet(folder / f"{name}_scored_D8.parquet")
        print(arm, "forecast", frame.height, f"{time.monotonic() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
