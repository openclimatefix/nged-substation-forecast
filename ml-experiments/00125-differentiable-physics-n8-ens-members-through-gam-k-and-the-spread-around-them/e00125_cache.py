"""Cache the corrected member means and the proxy-driven forecast of every run, once per substation.

Every arm of experiment 125 is a post-processing step on this cache: no arm refits a GAM-K or
re-runs the weather layer. For each run of a period the cache holds

- the corrected mean forecast of each usable member (the member's GAM-K mean plus the filter's
  level and anomaly, in points of effective capacity), exactly as `studies.dp_ens.forecast_run`
  builds them;
- the proxy-driven forecast of the same window with the same filter correction, for the residual
  table of arm MP;
- the observed power, valid time, lead, hour band, quarter, and switching mask of every scored
  half-hour.

The filter is experiment 111's level-and-anomaly filter (`gamk`), unchanged, with 111's stored
hyperparameters.

Build it with `e00125_build_cache.py`, which keeps `RunCache` importable by module name so the
pickled cache loads in every other script.
"""

import pickle
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Final

import e00125_arms as arms
import e00125_common as common
import e00125_ens as ens
import numpy as np
from studies.dp_assimilation import CorrectionTrace, correction_at, usable_cutoff_positions
from studies.dp_ens import choose_fit, member_means
from studies.dp_ens_weather import proxy_cache, slice_cache
from studies.dp_features import FeatureCache, naive_utc
from studies.dp_gam import GamFit, predict_gam
from studies.power import HALF_HOURS_PER_HOUR

CUTOFF: Final[np.datetime64] = np.datetime64("2026-07-01T00:00:00", "us")
"""No valid time on or after this instant is read."""
EXCLUDED_RUN: Final[datetime] = common.LAST_RUN
"""Experiment 111 excluded the last run; so does this experiment."""


@dataclass(frozen=True)
class RunCache:
    """One run's cached forecasts at the scored half-hours.

    Attributes:
        run: Index of the run in the setup.
        origin: The run's `power_fcst_init_time` as naive UTC.
        members: The ensemble members that entered the run.
        corrected: Each usable member's corrected mean, shape (members, rows), in points.
        proxy: The proxy-driven forecast plus the same correction, shape (rows,); `NaN` where the
            proxy covers no weather.
        level: The filter's persistent correction at each row.
        anomaly: The filter's decaying correction at each row.
        valid_time: Each row's valid time as naive UTC.
        lead_hours: Each row's lead from the origin.
        hour_band: Each row's hour band.
        quarter: Each row's calendar quarter.
        actual: The observed power.
        switching_mask: True inside a hand-filtered switching event.
    """

    run: int
    origin: np.datetime64
    members: list[int]
    corrected: np.ndarray
    proxy: np.ndarray
    level: np.ndarray
    anomaly: np.ndarray
    valid_time: np.ndarray
    lead_hours: np.ndarray
    hour_band: np.ndarray
    quarter: np.ndarray
    actual: np.ndarray
    switching_mask: np.ndarray


def cache_path(*, site: str, period: str) -> Path:
    """Return the file that holds a site's cache for a period."""
    return common.WORK_DIR / "cache" / f"{site}_{period}.pkl"


def read_cache(*, site: str, period: str) -> list[RunCache]:
    """Return a site's cached runs for a period."""
    return pickle.loads(cache_path(site=site, period=period).read_bytes())


@dataclass(frozen=True)
class SiteInputs:
    """Everything needed to cache a run of one site.

    Attributes:
        site: The substation's label.
        cache: The feature cache.
        setup: The site's runs and weather context, with the tuned calibration and shrink.
        fits: The monthly fits of `gamk`.
        trace: The filter's trace on the proxy-driven residuals.
        driven_by_proxy: The cache with the calibrated analysis proxy as its weather.
    """

    site: str
    cache: FeatureCache
    setup: ens.SiteSetup
    fits: list[GamFit]
    trace: CorrectionTrace
    driven_by_proxy: FeatureCache


def prepare_site(
    *, site: str, cache: FeatureCache | None = None, setup: ens.SiteSetup | None = None
) -> SiteInputs:
    """Load a site's fits, calibration, and filter trace.

    Args:
        site: The substation's label.
        cache: A feature cache to use instead of the stored one (for the causality test).
        setup: A setup to use instead of the stored one (for the causality test); it must carry
            the tuned calibration already.

    Returns:
        The inputs.
    """
    cache = common.load_cache() if cache is None else cache
    tuned = arms.read_tuned()
    if setup is None:
        setup = ens.load_setup(cache=cache, site=site)
        calibration = replace(
            arms.load_calibration(site=site), solar_shrink=np.array(tuned["solar_shrink"])
        )
        setup = setup.with_calibration(calibration=calibration)
    arm = arms.ARMS["gamk"]
    fits = arms.load_arm_fits(arm=arm, site=site)
    trace = arms.build_trace(
        cache=cache,
        setup=setup,
        arm=arm,
        fits=fits,
        hyper=np.array(tuned["filter_hyper"]["gamk"][site]),
    )
    return SiteInputs(
        site=site,
        cache=cache,
        setup=setup,
        fits=fits,
        trace=trace,
        driven_by_proxy=proxy_cache(cache=cache, context=setup.context),
    )


def build_run(*, inputs: SiteInputs, index: int) -> RunCache | None:
    """Cache one run, or return `None` if it has no scored half-hour.

    Args:
        inputs: The site's inputs.
        index: The run's index in the setup.

    Returns:
        The cached run.
    """
    cache, setup, site = inputs.cache, inputs.setup, inputs.site
    features = cache.sites[site]
    run = setup.runs[index]
    origin = setup.origins[index]
    if setup.run_dates[index] == EXCLUDED_RUN:
        return None
    if len(cache.end_times) - run.origin_position - 1 < ens.MIN_WINDOW_HALF_HOURS:
        return None
    result = member_means(
        cache=cache, context=setup.context, run=run, origin=origin, fits=inputs.fits
    )
    target = np.arange(result.window.start, result.window.stop)
    band = cache.calendar.band[target].astype(np.int64)
    cutoff = usable_cutoff_positions(
        end_times=cache.end_times,
        initialisation=np.array([naive_utc(moment=origin)]),
        delay_hours=3.5,
    )
    correction = correction_at(
        trace=inputs.trace, cutoff=np.full(len(target), cutoff[0]), target=target, band=band
    )
    shift = correction.level + correction.anomaly
    proxy_window = slice_cache(
        cache=inputs.driven_by_proxy,
        start=result.window.start,
        stop=result.window.stop,
        sites=[site],
    )
    fit = choose_fit(fits=inputs.fits, origin=origin)
    proxy_forecast = predict_gam(cache=proxy_window, fit=fit, index=np.arange(len(target)))
    valid = cache.end_times[target]
    keep = features.present[target] & np.isfinite(features.y[target]) & (valid < CUTOFF)
    if not keep.any():
        return None
    lead_hours = np.arange(1, len(target) + 1) / HALF_HOURS_PER_HOUR
    return RunCache(
        run=index,
        origin=np.datetime64(origin.replace(tzinfo=None), "us"),
        members=result.members,
        corrected=(result.means + shift[None, :])[:, keep],
        proxy=(proxy_forecast + shift)[keep],
        level=correction.level[keep],
        anomaly=correction.anomaly[keep],
        valid_time=valid[keep],
        lead_hours=lead_hours[keep],
        hour_band=band[keep].astype(np.int8),
        quarter=cache.calendar.quarter[target][keep].astype(np.int8),
        actual=features.y[target][keep],
        switching_mask=features.switching_mask[target][keep],
    )


def build_site(*, site: str, period: str) -> list[RunCache]:
    """Build a site's cache for one period.

    Args:
        site: The substation's label.
        period: `tuning`, `replicate`, or `scored`.

    Returns:
        The cached runs.
    """
    inputs = prepare_site(site=site)
    built = (
        build_run(inputs=inputs, index=index) for index in inputs.setup.indices(periods={period})
    )
    return [run for run in built if run is not None]
