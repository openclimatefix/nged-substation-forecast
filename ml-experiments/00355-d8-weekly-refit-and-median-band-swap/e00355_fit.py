# ruff: noqa: E501
"""Fit D8's stack (experiment 354's fit `b`) at monthly, weekly, and quarterly cuts, and forecast.

Fit `b` is T-all from 2023-09-01 with a level step per hour band at 2025-04-01 (experiment 354's
best fit). Cadence `monthly` uses the stored monthly cuts, `weekly` a cut every 7 days from the first
monthly cut to 2026-06-29 (experiment 252's weekly cuts), and `quarterly` every third monthly cut.
Each cadence's frames are written under `work/<site>_<cadence>/` as `d_scored_<site>.parquet`
(night offset and recipe R, the card's stack) and `a_scored_<site>.parquet` (no corrections).
Cuts are fitted by a spawned process pool, one thread per worker, one file per cut.

    PYTHONPATH=ml-experiments/00252-gam-stack-at-eight-more-primaries:ml-experiments/00354-d8-own-residual-bands-and-longer-fit-history:ml-experiments/00355-d8-weekly-refit-and-median-band-swap \
        systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00355-*/e00355_fit.py --cadences monthly weekly
"""

import argparse
import multiprocessing
import os

os.environ.setdefault("E00252_SOURCE", "mva")
os.environ.setdefault("E00252_ARM", "a1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00252_common as common  # ty: ignore[unresolved-import]
import e00252_run as run  # ty: ignore[unresolved-import]
import e00252_tall as tall  # ty: ignore[unresolved-import]
import e00252s2_corrections as corr  # ty: ignore[unresolved-import]
import e00252s2_run as stage2  # ty: ignore[unresolved-import]
import e00354_step as step  # ty: ignore[unresolved-import]
import polars as pl
import torch
from studies.dp_assimilation import DEFAULT_DELAY_HOURS
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00355-d8-weekly-refit-and-median-band-swap" / "work"
LONG_START: Final[datetime] = datetime(2023, 9, 1, tzinfo=UTC)
STEP_AT: Final[datetime] = datetime(2025, 4, 1, tzinfo=UTC)
LONG_WINDOW_MONTHS: Final[int] = 48
CADENCES: Final[tuple[str, ...]] = ("monthly", "weekly", "quarterly")


def fit_settings(*, site: str) -> dict:
    """Return D8's fit `b` settings, or the stored default window with no step at a placebo."""
    if site == "D8":
        return {"start": LONG_START, "window": LONG_WINDOW_MONTHS, "step": STEP_AT}
    return {"start": common.D8_FIRST_DAY, "window": common.WINDOW_MONTHS, "step": None}


def cuts_for(*, site: str, cadence: str) -> list[datetime]:
    """Return a cadence's cuts at a site."""
    if cadence == "quarterly":
        return tall.cuts_of(site=site, cadence="monthly")[::3]
    return tall.cuts_of(site=site, cadence=cadence)


def folder_of(*, site: str, cadence: str) -> Path:
    """Return the folder of one site and cadence's fits and forecasts."""
    return WORK / f"{site}_{cadence}"


def _fit_cut(job: tuple[str, datetime]):  # noqa: ANN202
    site, cut = job
    torch.set_num_threads(1)
    step.install()
    settings = fit_settings(site=site)
    common.D8_FIRST_DAY = settings["start"]
    common.WINDOW_MONTHS = settings["window"]
    cache = common.variant_cache(cache=common.load_cache(), site=site)
    vintages = common.vintages_for(cache=cache)
    base = tall.design.build_base(
        cache=cache, vintages=vintages, site=site, cut=cut, fitted=common.FITTED
    )
    spec = step.step_arm(
        base=tall.design.interaction_arm(name="t_all", penalty_weight=common.T_ALL_LAMBDA),
        step_from=settings["step"],
    )
    return tall.TallFit(
        site=site,
        cut=cut,
        layout=base.layout,
        pinned=base.pinned,
        arm=spec,
        fit=tall.design.fit_arm(cache=cache, base=base, arm=spec),
    )


def fit_cadence(*, site: str, cadence: str, workers: int) -> list:
    """Fit every cut of a cadence in a spawned pool and return the fits ascending by cut."""
    cuts = cuts_for(site=site, cadence=cadence)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        return list(pool.map(_fit_cut, [(site, cut) for cut in cuts]))


def forecast_scored(*, site: str) -> pl.DataFrame:
    """Forecast the scored runs from the fits stored under `tall.FITS_DIR`."""
    context = run.load_site(site=site)
    forecaster = run.ArmForecaster(arm="a2", context=context)
    first = forecaster.fits[0].cut + timedelta(hours=DEFAULT_DELAY_HOURS)
    runs = [
        i for i in context.setup.indices(periods={"scored"}) if context.setup.origins[i] >= first
    ]
    return forecaster.forecast(runs=runs)


def main() -> None:
    """Fit and forecast every requested cadence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default="D8")
    parser.add_argument("--cadences", nargs="+", default=list(CADENCES))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(4)
    step.install()
    site = args.site
    for cadence in args.cadences:
        started = time.monotonic()
        folder = folder_of(site=site, cadence=cadence)
        folder.mkdir(parents=True, exist_ok=True)
        fits = fit_cadence(site=site, cadence=cadence, workers=args.workers)
        (folder / "tall_fits.pkl").write_bytes(pickle.dumps(fits))
        print(site, cadence, "fits", len(fits), f"{time.monotonic() - started:.0f} s", flush=True)
        tall.FITS_DIR = folder / "tall_fits_root"
        target = tall.fit_path(cadence="monthly", site=site)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(pickle.dumps(fits))
        common.D8_FIRST_DAY = fit_settings(site=site)["start"]
        frame = forecast_scored(site=site)
        base = frame.with_columns(period=pl.lit("scored"))
        base.write_parquet(folder / f"base_raw_scored_{site}.parquet")
        arms = stage2.build_arms(site=site, frame=corr.add_calendar(frame=base))
        for name in ("a", "d"):
            arms[name].write_parquet(folder / f"{name}_scored_{site}.parquet")
        print(
            site,
            cadence,
            "forecast",
            frame.height,
            f"{time.monotonic() - started:.0f} s",
            flush=True,
        )


if __name__ == "__main__":
    main()
