"""Forecast every stage 1 arm of experiment 170 at every substation and write the prediction frames.

| Arm | Fits | Drift | Window | Filter noise | Cadence |
|---|---|---|---|---|---|
| `a2_repro` | T-all | none | last 24 months | POOL (experiment 133) | monthly |
| `m24_d1` | T-all | D1 | last 24 months | POOL | monthly |
| `all_d1` | T-all | D1 | all years | POOL | monthly |

`a2_repro` repeats experiment 136's arm A2 with this folder's fitting code, as a reproduction
control. Every arm is driven by the calibrated ensemble mean. The filter runs on the arm's own
fits' residuals against the analysis proxy; the quantiles are the corrected mean plus a residual
table fitted on the tuning months from the same arm, with every spread factor 1.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00170-*/e00170_run.py [--workers 6]
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from itertools import pairwise
from pathlib import Path
from typing import Final

import e00170_common as common
import e00170_ens as ens
import e00170_filters as filters
import e00170_tall as tall
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.dp_assimilation import DEFAULT_DELAY_HOURS
from studies.dp_ens_weather import Calibration, proxy_cache
from studies.dp_features import FeatureCache, naive_utc

ARM_NAMES: Final[tuple[str, ...]] = ("a2_repro", "m24_d1", "all_d1")
NO_FILTER_SUFFIX: Final[str] = "_nf"
NO_FILTER_ARMS: Final[tuple[str, ...]] = tuple(f"{a}{NO_FILTER_SUFFIX}" for a in ARM_NAMES)
"""Diagnostic arms: the same fits with the Kalman filter's correction left out of the forecast
(post hoc, run to test why a drift term helps without the filter and not with it)."""
PERIODS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")
OUT_DIR: Final[Path] = common.WORK_DIR / "stage1_preds"
ARM_FITS: Final[dict[str, str]] = {"a2_repro": "m24", "m24_d1": "m24_d1", "all_d1": "all_d1"}
ARM_FITS.update({f"{a}{NO_FILTER_SUFFIX}": ARM_FITS[a] for a in ARM_NAMES})


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


def noise_hyper() -> np.ndarray:
    """Return the four unconstrained filter values of experiment 133's POOL."""
    return np.array(json.loads(common.POOL_PATH.read_text())["hyper"])


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


def load_arm_fits(*, arm: str, site: str) -> list:
    """Return an arm's monthly or weekly fits at a site."""
    return tall.load_fits(fit_set=ARM_FITS[arm], site=site)


def tall_residual_stream(*, cache: FeatureCache, fits: list[tall.TallFit], site: str) -> np.ndarray:
    """Return observed minus the chained T-all fits' forecast on a cache, `NaN` where none.

    Fit `i` supplies the rows after its cut and up to the next cut, as `proxy_residual_stream`
    does for `GamFit`, so no residual is in-sample.
    """
    if not fits or any(later.cut <= earlier.cut for earlier, later in pairwise(fits)):
        msg = "the residual stream needs fits with strictly ascending cuts"
        raise ValueError(msg)
    features = cache.sites[site]
    stream = np.full(len(cache.end_times), np.nan)
    cuts = [
        int(np.searchsorted(cache.end_times, naive_utc(moment=fit.cut), side="right")) - 1
        for fit in fits
    ]
    ends = [*cuts[1:], len(stream) - 1]
    for fit, cut, end in zip(fits, cuts, ends, strict=True):
        rows = np.arange(cut + 1, end + 1)
        usable = (
            np.isfinite(features.smoothed_temperature_c[rows]).all(axis=1)
            & np.isfinite(features.pv_unit_shape[rows])
            & np.isfinite(features.temperature_c[rows])
            & np.isfinite(features.y[rows])
        )
        rows = rows[usable]
        if len(rows):
            stream[rows] = features.y[rows] - tall.predict_tall(cache=cache, fit=fit, index=rows)
    return stream


class ArmForecaster:
    """Forecasts one arm at one substation, with the spread fitted on the tuning months."""

    def __init__(self, *, arm: str, context: SiteContext) -> None:
        """Run the filter on the arm's residual stream and fit the spread on the tuning months."""
        self.arm = arm
        self.context = context
        self.fits = load_arm_fits(arm=arm, site=context.site)
        proxy = proxy_cache(cache=context.cache, context=context.setup.context)
        stream = tall_residual_stream(cache=proxy, fits=self.fits, site=context.site)
        self.run = filters.filter_stream(
            residual=stream,
            band=context.cache.calendar.band,
            settings=settings_of_hyper(hyper=noise_hyper()),
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
            trace=None if self.arm.endswith(NO_FILTER_SUFFIX) else self.run.trace,
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
    jobs = [(arm, site) for arm in args.arms for site in args.sites]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(_job, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
