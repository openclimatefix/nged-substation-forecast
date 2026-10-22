"""Fit each arm's monthly T-all fits and forecast arm D from them, at the six substations.

An arm is T-all (`r0`, rebuilt as the pipeline control) or T-all with the tail columns of F1 or F2
at a tail ridge ratio. The pipeline is experiment 136's arm `a2` (monthly fits, pooled filter noise,
ensemble-mean weather, spread fitted on the tuning months) followed by experiment 155's arm `d`
(winter night offset, recipe R, spread refitted). Fits are cached under `work/fits/<arm>/<site>.pkl`
and forecasts under `work/dp_preds/<arm>_<period>_<site>.parquet`.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00216-*/e00216_run.py --arms r0 f1_0.1 [--workers 4]

An arm name is `r0`, or `f1_<ratio>` or `f2_<ratio>`.
"""

import argparse
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Final

import e00216_common as common
import e00216_corrections as corr
import e00216_design as design
import e00216_ens as ens
import e00216_filters as filters
import e00216_tall as tall
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.dp_assimilation import DEFAULT_DELAY_HOURS
from studies.dp_ens import proxy_residual_stream  # noqa: F401
from studies.dp_ens_weather import Calibration, proxy_cache
from studies.dp_features import FeatureCache, naive_utc
from studies.dp_pinned_capacity import read_vintages

PERIODS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")
OUT_DIR: Final[Path] = common.WORK_DIR / "dp_preds"
FITS_DIR: Final[Path] = common.WORK_DIR / "fits"
OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "time_series_id", "run", "power_fcst_init_time", "valid_time", "lead_hours", "seed", "arm",
    *common.QUANTILES, "actual", "mean", "hour_band", "quarter", "switching_mask",
)  # fmt: skip


def arm_spec(*, arm: str) -> design.ArmSpec | None:
    """Return the design of an arm name: `None` for T-all (`r0`), else F1 or F2 at a tail ratio."""
    if arm == "r0":
        return None
    kind, ratio = arm.split("_")
    if kind not in ("f1", "f2"):
        msg = f"unknown arm {arm!r}"
        raise ValueError(msg)
    return design.tail_arm(
        name=arm, penalty_weight=common.T_ALL_LAMBDA, tail_ratio=float(ratio), lag=kind == "f2"
    )


def fits_path(*, arm: str, site: str) -> Path:
    """Return the file that holds an arm's monthly fits at a site."""
    return FITS_DIR / arm / f"{site}.pkl"


def write_atomic_bytes(*, target: Path, content: bytes) -> None:
    """Write a file through a temporary name in the same folder."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    temporary.rename(target)


def fit_site(*, arm: str, site: str, cache: FeatureCache) -> list[tall.TallFit]:
    """Return an arm's monthly fits at a site, from disk if stored."""
    path = fits_path(arm=arm, site=site)
    if path.exists():
        return pickle.loads(path.read_bytes())
    vintages = read_vintages()
    spec = arm_spec(arm=arm)
    fits = [
        tall.fit_tall(
            cache=cache,
            vintages=vintages,
            site=site,
            cut=cut,
            penalty_weight=common.T_ALL_LAMBDA,
            arm=spec,
        )
        for cut in tall.cuts_of(site=site, cadence="monthly")
    ]
    write_atomic_bytes(target=path, content=pickle.dumps(fits))
    return fits


def load_calibration(*, site: str) -> Calibration:
    """Return experiment 111's stored calibration at a site, without its solar shrink."""
    with np.load(common.SOURCE_WORK_DIR / "calibration" / f"{site}.npz") as stored:
        return Calibration(
            temperature_shift_c=stored["shift"],
            irradiance_ratio=stored["ratio"],
            solar_shrink=stored["shrink"],
        )


def filter_settings() -> filters.FilterSettings:
    """Return experiment 133's pooled filter noise setting (POOL)."""
    hyper = np.array(json.loads(common.POOL_PATH.read_text())["hyper"])
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
    tuned = json.loads((common.SOURCE_WORK_DIR / "tuned.json").read_text())
    calibration = load_calibration(site=site).with_solar_shrink(solar_shrink=tuned["solar_shrink"])
    setup = ens.load_setup(cache=cache, site=site).with_calibration(calibration=calibration)
    return SiteContext(site=site, cache=cache, setup=setup)


class ArmForecaster:
    """Forecasts one arm at one substation: filter on the arm's residuals, spread on tuning months."""

    def __init__(self, *, fits: list[tall.TallFit], context: SiteContext, arm: str) -> None:
        """Run the filter on the fits' residual stream and fit the spread on the tuning months."""
        self.arm = arm
        self.context = context
        self.fits = fits
        proxy = proxy_cache(cache=context.cache, context=context.setup.context)
        stream = tall_residual_stream(cache=proxy, fits=fits, site=context.site)
        self.run = filters.filter_stream(
            residual=stream, band=context.cache.calendar.band, settings=filter_settings()
        )
        first = fits[0].cut + timedelta(hours=DEFAULT_DELAY_HOURS)
        self.first_origin = first
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

    def forecast_period(self, *, period: str) -> pl.DataFrame:
        """Forecast every run of a period whose origin is after the first fit's cut."""
        setup = self.context.setup
        runs = [i for i in setup.indices(periods={period}) if setup.origins[i] >= self.first_origin]
        return self.forecast(runs=runs)


def tall_residual_stream(*, cache: FeatureCache, fits: list[tall.TallFit], site: str) -> np.ndarray:
    """Return observed minus the chained fits' forecast on a cache, `NaN` where none.

    Fit `i` supplies the rows after its cut and up to the next cut, so no residual is in-sample.
    """
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


def base_frame(*, forecaster: ArmForecaster) -> pl.DataFrame:
    """Return the arm's `a2`-style frame of all three periods with the calendar columns."""
    pieces = [
        forecaster.forecast_period(period=period).with_columns(period=pl.lit(period))
        for period in PERIODS
    ]
    frame = pl.concat(pieces).sort(["power_fcst_init_time", "valid_time"])
    return corr.add_calendar(frame=frame)


def arm_d(*, base: pl.DataFrame) -> pl.DataFrame:
    """Return experiment 155's arm D built on a base frame: offset, then recipe R, spread refitted."""
    mean_a = base["mean"].to_numpy().astype(np.float64)
    offset, _ = corr.winter_offset_shift(frame=base, variant="winter")
    mean_b = mean_a + offset
    shift_d = offset + corr.recipe_r_shift(frame=base, forecast=mean_b)
    return corr.apply_shift_and_refit_spread(frame=base, shift=shift_d)


def out_path(*, arm: str, period: str, site: str) -> Path:
    """Return the file that holds an arm's arm-D forecasts for one period at one site."""
    return OUT_DIR / f"{arm}_{period}_{site}.parquet"


def write_arm_d(*, arm: str, site: str, frame: pl.DataFrame) -> int:
    """Write arm D's rows per period, atomically, and return the number of rows written."""
    rows = 0
    for period in PERIODS:
        part = frame.filter(pl.col("period") == period).with_columns(arm=pl.lit(arm))
        target = out_path(arm=arm, period=period, site=site)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        part.select(list(OUTPUT_COLUMNS)).write_parquet(temporary)
        temporary.rename(target)
        rows += part.height
    return rows


def _job(job: tuple[str, str]) -> str:
    torch.set_num_threads(1)
    arm, site = job
    started = time.monotonic()
    context = load_site(site=site)
    fits = fit_site(arm=arm, site=site, cache=context.cache)
    base = base_frame(forecaster=ArmForecaster(fits=fits, context=context, arm=arm))
    rows = write_arm_d(arm=arm, site=site, frame=arm_d(base=base))
    return f"{arm} {site}: {rows:,} rows, {time.monotonic() - started:.0f} s"


def main() -> None:
    """Fit and forecast every arm at every substation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--sites", nargs="+", default=list(common.SCORED_SITES))
    args = parser.parse_args()
    jobs = [(arm, site) for arm in args.arms for site in args.sites]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(_job, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
