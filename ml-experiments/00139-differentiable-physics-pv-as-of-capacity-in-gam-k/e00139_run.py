"""Forecast every capacity arm of experiment 139 at every substation.

Every arm is experiment 136's `a0` (experiment 111's GAM-K with filter noise FIX) driven by the
calibrated ensemble mean, with one change: the capacity the fits subtract (`e00139_fits.py`).
The filter runs on the arm's own fits' residuals against the analysis proxy; the quantiles are the
corrected mean plus a residual table fitted on the tuning months from the same arm, with every
spread factor 1.

    uv run --no-sync python ml-experiments/00139-*/e00139_run.py [--workers 6] [--arms pin asof]
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Final

import e00139_ens as ens
import e00139_filters as filters
import e00139_fits as fits_module
import e00139_run_common as common
import numpy as np
import polars as pl
import torch
from studies.dp_assimilation import DEFAULT_DELAY_HOURS
from studies.dp_ens import proxy_residual_stream
from studies.dp_ens_weather import Calibration, proxy_cache
from studies.dp_features import FeatureCache

ARM_NAMES: Final[tuple[str, ...]] = fits_module.ARMS
PERIODS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")
OUT_DIR: Final[Path] = common.WORK_DIR / "dp_preds"


def out_path(*, arm: str, period: str, site: str) -> Path:
    """Return the file that holds an arm's forecasts for one period at one site."""
    return OUT_DIR / f"{arm}_{period}_{site}.parquet"


def read_tuned() -> dict:
    """Return experiment 111's tuned settings (solar shrink, filter values, spread factors)."""
    return json.loads((common.SOURCE_WORK_DIR / "tuned.json").read_text())


def load_calibration(*, site: str) -> Calibration:
    """Return experiment 111's stored calibration at a site, without its solar shrink."""
    with np.load(common.SOURCE_WORK_DIR / "calibration" / f"{site}.npz") as stored:
        return Calibration(
            temperature_shift_c=stored["shift"],
            irradiance_ratio=stored["ratio"],
            solar_shrink=stored["shrink"],
        )


def noise_hyper(*, site: str) -> np.ndarray:
    """Return the four unconstrained filter values of experiment 111 at a site (FIX)."""
    return np.array(read_tuned()["filter_hyper"]["gamk"][site])


def settings_of_hyper(*, hyper: np.ndarray) -> filters.FilterSettings:
    """Return the filter setting of four unconstrained values (log q, logit phi, log sigma)."""
    return filters.FilterSettings(
        q=float(np.exp(hyper[0])),
        phi=float(1.0 / (1.0 + np.exp(-hyper[1]))),
        sigma_z=float(np.exp(hyper[2])),
        sigma_eps=float(np.exp(hyper[3])),
        rho_a=0.0,
    )


@dataclass(frozen=True)
class SiteContext:
    """A substation's stored inputs: the ensemble setup and the cache."""

    site: str
    cache: FeatureCache
    setup: ens.SiteSetup


def load_site(*, site: str, cache: FeatureCache | None = None) -> SiteContext:
    """Load a substation's ensemble setup with experiment 111's calibration and solar shrink."""
    cache = cache if cache is not None else common.load_cache()
    calibration = load_calibration(site=site).with_solar_shrink(
        solar_shrink=read_tuned()["solar_shrink"]
    )
    setup = ens.load_setup(cache=cache, site=site).with_calibration(calibration=calibration)
    return SiteContext(site=site, cache=cache, setup=setup)


class ArmForecaster:
    """Forecasts one arm at one substation, with the spread fitted on the tuning months."""

    def __init__(self, *, arm: str, context: SiteContext) -> None:
        """Run the filter on the arm's residual stream and fit the spread on the tuning months."""
        self.arm = arm
        self.context = context
        self.fits = fits_module.load_fits(arm=arm, site=context.site)
        proxy = proxy_cache(cache=context.cache, context=context.setup.context)
        stream = proxy_residual_stream(cache=proxy, fits=self.fits, site=context.site)
        self.run = filters.filter_stream(
            residual=stream,
            band=context.cache.calendar.band,
            settings=settings_of_hyper(hyper=noise_hyper(site=context.site)),
        )
        first = self.fits[0].cut + timedelta(hours=DEFAULT_DELAY_HOURS)
        self.tuning = [
            i
            for i in context.setup.indices(periods={"tuning"})
            if context.setup.origins[i] >= first
        ]
        self.spread = ens.fit_spread_from_frame(
            frame=self._frame(runs=self.tuning, spread=ens.ZERO_SPREAD)
        )

    def _frame(self, *, runs: list[int], spread) -> pl.DataFrame:  # noqa: ANN001
        return ens.forecast_frame(
            cache=self.context.cache,
            setup=self.context.setup,
            fits=self.fits,
            spread=spread,
            trace=self.run.trace,
            run_indices=runs,
            arm=self.arm,
        )

    def forecast(self, *, runs: list[int]) -> pl.DataFrame:
        """Forecast the runs with the fitted spread, every factor 1."""
        return self._frame(runs=runs, spread=self.spread)


def _job(job: tuple[str, str]) -> str:
    torch.set_num_threads(1)
    arm, site = job
    started = time.monotonic()
    context = load_site(site=site)
    forecaster = ArmForecaster(arm=arm, context=context)
    first = forecaster.fits[0].cut + timedelta(hours=DEFAULT_DELAY_HOURS)
    rows = 0
    for period in PERIODS:
        runs = [
            i for i in context.setup.indices(periods={period}) if context.setup.origins[i] >= first
        ]
        if not runs:
            continue
        frame = forecaster.forecast(runs=runs)
        target = out_path(arm=arm, period=period, site=site)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        frame.write_parquet(temporary)
        temporary.rename(target)
        rows += frame.height
    return f"{arm} {site}: {rows:,} rows, {time.monotonic() - started:.0f} s"


def main() -> None:
    """Forecast every arm and substation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--arms", nargs="+", default=list(ARM_NAMES))
    parser.add_argument("--sites", nargs="+", default=list(common.SCORED_SITES))
    args = parser.parse_args()
    jobs = fits_module.jobs_for(arms=args.arms, sites=args.sites)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(_job, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
