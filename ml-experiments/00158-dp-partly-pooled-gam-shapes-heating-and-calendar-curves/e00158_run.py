"""Forecast every arm of experiment 158 at every substation and write the prediction frames.

An arm is a history (`full` or `3m`) and a pooling configuration of `e00158_fits.py`. Every arm is
driven by the calibrated ensemble mean, runs experiment 111's per-substation filter noise (FIX) on
its own fits' residuals against the analysis proxy, and takes the quantiles of the corrected mean
plus a residual table fitted on the tuning months from the same arm, every spread factor 1, as
experiment 136 does. The arms therefore differ only in their monthly fits.

    uv run python ml-experiments/00158-*/e00158_run.py --history full --configs k0 ph300 \
        --periods tuning [--workers 6]
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

import e00158_common as common
import e00158_ens as ens
import e00158_filters as filters
import e00158_fits as fits_module
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.dp_assimilation import DEFAULT_DELAY_HOURS
from studies.dp_ens import proxy_residual_stream
from studies.dp_ens_weather import Calibration, proxy_cache
from studies.dp_features import FeatureCache

PERIODS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")
OUT_DIR: Final[Path] = common.WORK_DIR / "dp_preds"


def arm_name(*, history: str, config: str) -> str:
    """Return the arm's name: `<history>_<config>`."""
    return f"{history}_{config}"


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

    def __init__(self, *, history: str, config: str, context: SiteContext) -> None:
        """Run the filter on the arm's residual stream and fit the spread on the tuning months."""
        self.arm = arm_name(history=history, config=config)
        self.context = context
        self.fits = fits_module.load_fits(config=config, history=history, site=context.site)
        proxy = proxy_cache(cache=context.cache, context=context.setup.context)
        stream = proxy_residual_stream(cache=proxy, fits=self.fits, site=context.site)
        hyper = np.array(read_tuned()["filter_hyper"]["gamk"][context.site])
        self.run = filters.filter_stream(
            residual=stream,
            band=context.cache.calendar.band,
            settings=settings_of_hyper(hyper=hyper),
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


def _job(job: tuple[str, str, str, tuple[str, ...]]) -> str:
    torch.set_num_threads(1)
    history, config, site, periods = job
    started = time.monotonic()
    context = load_site(site=site)
    forecaster = ArmForecaster(history=history, config=config, context=context)
    first = forecaster.fits[0].cut + timedelta(hours=DEFAULT_DELAY_HOURS)
    rows = 0
    for period in periods:
        runs = [
            i for i in context.setup.indices(periods={period}) if context.setup.origins[i] >= first
        ]
        if not runs:
            continue
        frame = forecaster.forecast(runs=runs)
        target = out_path(arm=forecaster.arm, period=period, site=site)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        frame.write_parquet(temporary)
        temporary.rename(target)
        rows += frame.height
    return f"{forecaster.arm} {site}: {rows:,} rows, {time.monotonic() - started:.0f} s"


def main() -> None:
    """Forecast every arm and substation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--history", choices=fits_module.HISTORIES, default="full")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--periods", nargs="+", default=list(PERIODS))
    parser.add_argument("--sites", nargs="+", default=list(common.FLEET))
    args = parser.parse_args()
    jobs = [
        (args.history, config, site, tuple(args.periods))
        for config in args.configs
        for site in args.sites
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(_job, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
