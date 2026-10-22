"""Forecast the chain's arm A2 at each BSP and each remainder series, and write the frames.

Arm A2 is experiment 136's: T-all monthly fits, the shared filter noise (POOL), the calibrated
ensemble-mean weather, and a spread fitted on the tuning months. The BSPs and their remainders have
no capacity vintage, so their fits carry a fitted PV capacity (`e00204_tall.py`). The weather
calibration is fitted on the tuning months for each BSP and shared by its remainder series.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00161-*/e00204_run.py [--workers 6]
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from itertools import pairwise
from pathlib import Path
from typing import Final

import e00204_common as common
import e00204_ens as ens
import e00204_filters as filters
import e00204_tall as tall
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from e00204_ens import SiteSetup
from studies.dp_assimilation import DEFAULT_DELAY_HOURS
from studies.dp_ens_weather import Calibration, proxy_cache
from studies.dp_features import FeatureCache, naive_utc

PERIODS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")
OUT_DIR: Final[Path] = common.OUT_DIR


def out_path(*, arm: str, period: str, site: str) -> Path:
    """Return the file that holds an arm's forecasts for one period at one series."""
    return OUT_DIR / f"{arm}_{period}_{site}.parquet"


def bsp_of(*, site: str) -> str:
    """Return the BSP a chain series belongs to: the label without its remainder suffix."""
    return site[:3]


def calibration_path(*, bsp: str) -> Path:
    """Return the file that holds a BSP's weather calibration."""
    return common.CALIBRATION_DIR / f"{bsp}.npz"


def read_tuned() -> dict:
    """Return experiment 111's tuned settings (the solar shrink per lead band)."""
    return json.loads((common.SOURCE_WORK_DIR / "tuned.json").read_text())


def calibrate(*, bsp: str, cache: FeatureCache) -> None:
    """Fit a BSP's weather calibration on the tuning months and store it."""
    setup = ens.load_setup(cache=cache, site=bsp)
    calibration = ens.fit_calibration(cache=cache, setup=setup)
    path = calibration_path(bsp=bsp)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        shift=calibration.temperature_shift_c,
        ratio=calibration.irradiance_ratio,
        shrink=calibration.solar_shrink,
    )


def load_calibration(*, bsp: str) -> Calibration:
    """Return a BSP's stored calibration with experiment 111's solar shrink."""
    with np.load(calibration_path(bsp=bsp)) as stored:
        calibration = Calibration(
            temperature_shift_c=stored["shift"],
            irradiance_ratio=stored["ratio"],
            solar_shrink=stored["shrink"],
        )
    return calibration.with_solar_shrink(solar_shrink=read_tuned()["solar_shrink"])


def settings_of_hyper(*, hyper: np.ndarray) -> filters.FilterSettings:
    """Return the filter setting of four unconstrained values (log q, logit phi, log sigma)."""
    return filters.FilterSettings(
        q=float(np.exp(hyper[0])),
        phi=float(1.0 / (1.0 + np.exp(-hyper[1]))),
        sigma_z=float(np.exp(hyper[2])),
        sigma_eps=float(np.exp(hyper[3])),
        rho_a=0.0,
    )


def tall_residual_stream(*, cache: FeatureCache, fits: list[tall.TallFit], site: str) -> np.ndarray:
    """Return observed minus the chained T-all fits' forecast on a cache, `NaN` where none.

    Fit `i` supplies the rows after its cut and up to the next cut, so no residual is in-sample.
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


class SiteForecaster:
    """Forecasts arm A2 at one series, with the spread fitted on the tuning months."""

    def __init__(self, *, site: str, cache: FeatureCache) -> None:
        """Run the filter on the series' residual stream and fit the spread on the tuning months."""
        self.site = site
        self.cache = cache
        calibration = load_calibration(bsp=bsp_of(site=site))
        self.setup: SiteSetup = ens.load_setup(cache=cache, site=site).with_calibration(
            calibration=calibration
        )
        self.fits = tall.load_fits(site=site)
        proxy = proxy_cache(cache=cache, context=self.setup.context)
        stream = tall_residual_stream(cache=proxy, fits=self.fits, site=site)
        hyper = np.array(json.loads(common.POOL_PATH.read_text())["hyper"])
        self.run = filters.filter_stream(
            residual=stream,
            band=cache.calendar.band,
            settings=settings_of_hyper(hyper=hyper),
        )
        first = self.fits[0].cut + timedelta(hours=DEFAULT_DELAY_HOURS)
        self.tuning = [
            i for i in self.setup.indices(periods={"tuning"}) if self.setup.origins[i] >= first
        ]
        self.spread = ens.fit_spread_from_frame(
            frame=self._frame(runs=self.tuning, spread=ens.ZERO_SPREAD)
        )

    def _frame(self, *, runs: list[int], spread) -> pl.DataFrame:  # noqa: ANN001
        return ens.forecast_frame(
            cache=self.cache,
            setup=self.setup,
            fits=self.fits,
            spread=spread,
            trace=self.run.trace,
            run_indices=runs,
            arm="a2",
        )

    def forecast(self, *, runs: list[int]) -> pl.DataFrame:
        """Forecast the runs with the fitted spread, every factor 1."""
        return self._frame(runs=runs, spread=self.spread)


def _job(site: str) -> str:
    torch.set_num_threads(1)
    started = time.monotonic()
    cache = common.load_cache()
    forecaster = SiteForecaster(site=site, cache=cache)
    first = forecaster.fits[0].cut + timedelta(hours=DEFAULT_DELAY_HOURS)
    rows = 0
    for period in PERIODS:
        runs = [
            i
            for i in forecaster.setup.indices(periods={period})
            if forecaster.setup.origins[i] >= first
        ]
        if not runs:
            continue
        frame = forecaster.forecast(runs=runs)
        target = out_path(arm="a2", period=period, site=site)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        frame.write_parquet(temporary)
        temporary.rename(target)
        rows += frame.height
    return f"a2 {site}: {rows:,} rows, {time.monotonic() - started:.0f} s"


def main() -> None:
    """Calibrate each BSP's weather, then forecast every chain series."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--sites", nargs="+", default=list(common.CHAIN_SITES))
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()
    if args.calibrate:
        cache = common.load_cache()
        for bsp in common.BSPS:
            calibrate(bsp=bsp, cache=cache)
            print(f"calibrated {bsp}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(_job, args.sites):
            print(line, flush=True)


if __name__ == "__main__":
    main()
