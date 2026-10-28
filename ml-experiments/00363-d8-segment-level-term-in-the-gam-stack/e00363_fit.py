"""Fit D8's (or a placebo Primary's) monthly stack with a penalised level per detected stretch.

Arms (all monthly, at the cuts of experiment 252's stored fits, T-all with a fitted solar capacity):

| Arm | Design |
|---|---|
| `ctl` | experiment 354's fit (b) at D8 (history from 2023-09, level step per hour band at 2025-04-01); the stored settings at D2 and D4 |
| `seg_weak` | `ctl` plus one ridge-penalised level per stretch between detected steps, lambda 0.1 |
| `seg_mid` | the same with lambda 1 |
| `seg_strong` | the same with lambda 10 |

Stretch boundaries are every start and end of the segments that experiment 361's step and
multi-day-drop rules find on the power up to the cut. Rows after the cut take the latest stretch's
level. Frames are written under `work/<site>/fits_<arm>/`.

    PYTHONPATH=$E252:$E264:$E354:$E361:$E363 \
        OMP_NUM_THREADS=4 systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python -W ignore ml-experiments/00363-*/e00363_fit.py --site D8 --arms ctl seg_mid
"""

import argparse
import os

os.environ.setdefault("E00252_SOURCE", "mva")
os.environ.setdefault("E00252_ARM", "a1")

import pickle
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00252_common as common  # ty: ignore[unresolved-import]
import e00252_run as run  # ty: ignore[unresolved-import]
import e00252_tall as tall  # ty: ignore[unresolved-import]
import e00252s2_corrections as corr  # ty: ignore[unresolved-import]
import e00252s2_run as stage2  # ty: ignore[unresolved-import]
import e00354_fit as f354  # ty: ignore[unresolved-import]
import e00354_step as step  # ty: ignore[unresolved-import]
import e00361_detect as detect  # ty: ignore[unresolved-import]
import e00363_segment as segment
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.dp_assimilation import DEFAULT_DELAY_HOURS
from studies.dp_features import naive_utc
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00363-d8-segment-level-term-in-the-gam-stack" / "work"
LAMBDAS: Final[dict[str, float]] = {
    "ctl": 0.0,
    "seg_weak": 0.1,
    "seg_mid": 1.0,
    "seg_strong": 10.0,
}
STORED_START: Final[datetime] = datetime(2025, 4, 1, tzinfo=UTC)
STORED_WINDOW_MONTHS: Final[int] = 24


def arm_dir(*, site: str, arm: str) -> Path:
    """Return the folder of an arm's fits and forecasts at a site."""
    return WORK / site / f"fits_{arm}"


def stretches(*, cache, site: str, cut: datetime) -> dict:  # noqa: ANN001
    """Return the stretch boundaries and the fall statistics, from power up to `cut` only."""
    features = cache.sites[site]
    capacity = float(features.table.effective_capacity_mw)
    n = int((cache.end_times <= naive_utc(moment=cut)).sum())
    power = (features.y[:n] * capacity / 100.0).astype(np.float64)
    flag = np.zeros(n, dtype=bool)
    edges: list[int] = []
    depths: list[float] = []
    for seg in (detect.detect(power_mw=power), detect.detect_drops(power_mw=power)):
        for s, e, level in zip(seg.start, seg.end, seg.level_mw, strict=True):
            flag[s:e] = True
            edges += [int(s), int(e)]
            depths.append(abs(float(level)))
    edges = sorted({edge for edge in edges if 0 < edge < n})
    boundaries = tuple(int(np.datetime64(cache.end_times[edge], "ns").astype(np.int64)) for edge in edges)
    finite = int(np.isfinite(power).sum())
    return {
        "boundaries": boundaries,
        "last_in_fall": bool(edges) and bool(flag[edges[-1]]),
        "fall_share": float(flag.sum() / max(finite, 1)),
        "fall_depth_points": float(np.median(depths) * 100.0 / capacity) if depths else 0.0,
    }


def fit_arm_cuts(*, site: str, arm: str) -> list:
    """Fit T-all at every stored monthly cut of the site for one arm and return the fits."""
    if site == "D8":
        settings = f354.ARMS["b"]
        common.D8_FIRST_DAY, common.WINDOW_MONTHS = settings["start"], settings["window"]
        step_from = settings["step"]
    else:
        common.D8_FIRST_DAY, common.WINDOW_MONTHS = STORED_START, STORED_WINDOW_MONTHS
        step_from = None
    cache = common.variant_cache(cache=common.load_cache(), site=site)
    vintages = common.vintages_for(cache=cache)
    cuts = [f.cut for f in pickle.loads(tall.fit_path(cadence="monthly", site=site).read_bytes())]
    fits = []
    for cut in cuts:
        base = tall.design.build_base(
            cache=cache, vintages=vintages, site=site, cut=cut, fitted=common.FITTED
        )
        plain = step.step_arm(
            base=tall.design.interaction_arm(name="t_all", penalty_weight=common.T_ALL_LAMBDA),
            step_from=step_from,
        )
        if LAMBDAS[arm] > 0.0:
            plain = segment.SegmentArm(
                **{f: getattr(plain, f) for f in plain.__dataclass_fields__},
                segment_lambda=LAMBDAS[arm],
                **stretches(cache=cache, site=site, cut=cut),
            )
        fits.append(
            tall.TallFit(
                site=site,
                cut=cut,
                layout=base.layout,
                pinned=base.pinned,
                arm=plain,
                fit=tall.design.fit_arm(cache=cache, base=base, arm=plain),
            )
        )
    return fits


def forecast_frame(*, site: str) -> pl.DataFrame:
    """Forecast from the stored fits with the donated spread: D8's scored runs, or every period."""
    context = run.load_site(site=site)
    forecaster = run.ArmForecaster(arm="a2", context=context)
    first = forecaster.fits[0].cut + timedelta(hours=DEFAULT_DELAY_HOURS)
    periods = ["scored"] if site == "D8" else list(run.PERIODS)
    frames = []
    for period in periods:
        runs = [
            i
            for i in context.setup.indices(periods={period})
            if context.setup.origins[i] >= first
        ]
        if runs:
            frames.append(forecaster.forecast(runs=runs).with_columns(period=pl.lit(period)))
    return pl.concat(frames)


def main() -> None:
    """Fit and forecast every requested arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default="D8")
    parser.add_argument("--arms", nargs="+", default=list(LAMBDAS))
    args = parser.parse_args()
    torch.set_num_threads(4)
    step.install()
    segment.install()
    site = args.site
    for arm in args.arms:
        started = time.monotonic()
        folder = arm_dir(site=site, arm=arm)
        folder.mkdir(parents=True, exist_ok=True)
        fits = fit_arm_cuts(site=site, arm=arm)
        (folder / "tall_fits.pkl").write_bytes(pickle.dumps(fits))
        print(site, arm, "fits", len(fits), f"{time.monotonic() - started:.0f} s", flush=True)
        tall.FITS_DIR = folder / "tall_fits_root"
        target = tall.fit_path(cadence="monthly", site=site)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(pickle.dumps(fits))
        base = forecast_frame(site=site)
        frame = base
        base.write_parquet(folder / f"base_raw_scored_{site}.parquet")
        arms = stage2.build_arms(site=site, frame=corr.add_calendar(frame=base))
        for name in ("a", "d"):
            arms[name].write_parquet(folder / f"{name}_scored_{site}.parquet")
        print(site, arm, "forecast", frame.height, f"{time.monotonic() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
