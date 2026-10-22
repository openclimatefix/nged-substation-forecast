"""The filter's residual stream of experiment 136's arm A2, and the filter run on any stream.

`site_stream` returns the observed minus the chained monthly T-all fits' forecast on the analysis
proxy, on the cache's half-hour grid, exactly as experiment 136's `ArmForecaster` builds it for
arm A2 (the convex model with POOL's filter noise). `corrections_for_rows` reads a filter trace at
each stored forecast row's cut-off, so a stream edited by a network gives the level and anomaly
that a forecast row would have received.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import pairwise
from typing import Final

import e00222_common as common
import e00222_ens as ens
import e00222_filters as filters
import e00222_tall as tall
import numpy as np
import polars as pl
from studies.dp_assimilation import (
    DEFAULT_DELAY_HOURS,
    CorrectionTrace,
    correction_at,
    usable_cutoff_positions,
)
from studies.dp_ens_weather import Calibration, proxy_cache
from studies.dp_features import FeatureCache, naive_utc

ARM: Final[str] = "a2"


def read_tuned() -> dict:
    """Return experiment 111's tuned settings (solar shrink)."""
    return json.loads((common.SOURCE_WORK_DIR / "tuned.json").read_text())


def pool_settings() -> filters.FilterSettings:
    """Return experiment 133's shared filter cell (POOL) as a filter setting."""
    hyper = np.array(json.loads(common.POOL_PATH.read_text())["hyper"])
    return filters.FilterSettings(
        q=float(np.exp(hyper[0])),
        phi=float(1.0 / (1.0 + np.exp(-hyper[1]))),
        sigma_z=float(np.exp(hyper[2])),
        sigma_eps=float(np.exp(hyper[3])),
        rho_a=0.0,
    )


def load_calibration(*, site: str) -> Calibration:
    """Return experiment 111's stored calibration at a site, with its tuned solar shrink."""
    with np.load(common.SOURCE_WORK_DIR / "calibration" / f"{site}.npz") as stored:
        calibration = Calibration(
            temperature_shift_c=stored["shift"],
            irradiance_ratio=stored["ratio"],
            solar_shrink=stored["shrink"],
        )
    return calibration.with_solar_shrink(solar_shrink=read_tuned()["solar_shrink"])


@dataclass(frozen=True)
class SiteStream:
    """A substation's cache, proxy cache, fits and residual stream.

    Attributes:
        site: The substation's label.
        cache: The feature cache.
        proxy: The analysis-proxy cache the stream is built on.
        fits: Arm A2's monthly T-all fits.
        stream: Observed minus the chained fits' forecast, on the cache grid, `NaN` where none.
    """

    site: str
    cache: FeatureCache
    proxy: FeatureCache
    fits: list[tall.TallFit]
    stream: np.ndarray


def residual_stream(
    *, cache: FeatureCache, fits: list[tall.TallFit], site: str, last: datetime | None = None
) -> np.ndarray:
    """Return observed minus the chained T-all fits' forecast on a cache, `NaN` where none.

    Args:
        cache: The cache the fits are applied to.
        fits: The fits, ascending by cut.
        site: The site's label.
        last: The time the last fit stops at, or `None` for the end of the grid.

    Returns:
        The residual on the cache's grid.

    Raises:
        ValueError: If the fits are empty or their cuts do not strictly ascend.
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
    stop = (
        len(stream) - 1
        if last is None
        else int(np.searchsorted(cache.end_times, naive_utc(moment=last)))
    )
    ends = [*cuts[1:], stop]
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


def load_site_stream(*, site: str, cache: FeatureCache | None = None) -> SiteStream:
    """Build a substation's arm A2 residual stream as experiment 136 does."""
    cache = cache if cache is not None else common.load_cache()
    setup = ens.load_setup(cache=cache, site=site).with_calibration(
        calibration=load_calibration(site=site)
    )
    fits = tall.load_fits(cadence="monthly", site=site)
    proxy = proxy_cache(cache=cache, context=setup.context)
    return SiteStream(
        site=site,
        cache=cache,
        proxy=proxy,
        fits=fits,
        stream=residual_stream(cache=proxy, fits=fits, site=site),
    )


def run_filter(
    *,
    cache: FeatureCache,
    stream: np.ndarray,
    settings: filters.FilterSettings | None = None,
) -> CorrectionTrace:
    """Run a filter (POOL's by default) over a residual stream and return its trace."""
    return filters.filter_stream(
        residual=stream, band=cache.calendar.band, settings=settings or pool_settings()
    ).trace


def first_origin(*, fits: list[tall.TallFit]):  # noqa: ANN201
    """Return the first forecast origin of arm A2, as experiment 136 does."""
    return fits[0].cut + timedelta(hours=DEFAULT_DELAY_HOURS)


def corrections_for_rows(
    *, cache: FeatureCache, trace: CorrectionTrace, frame: pl.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Return the trace's level and anomaly at each row's cut-off and target.

    Args:
        cache: The feature cache.
        trace: The filter's trace.
        frame: Rows with `power_fcst_init_time` and `valid_time` (UTC).

    Returns:
        The level and the anomaly per row, in points.
    """
    origin = frame["power_fcst_init_time"].dt.replace_time_zone(None).dt.cast_time_unit("us")
    valid = frame["valid_time"].dt.replace_time_zone(None).dt.cast_time_unit("us").to_numpy()
    target = np.searchsorted(cache.end_times, valid)
    band = cache.calendar.band[target].astype(np.int64)
    cutoff = np.empty(len(target), dtype=np.int64)
    origins = origin.to_numpy()
    for stamp in np.unique(origins):
        rows = origins == stamp
        cutoff[rows] = usable_cutoff_positions(
            end_times=cache.end_times,
            initialisation=np.array([stamp]),
            delay_hours=DEFAULT_DELAY_HOURS,
        )[0]
    correction = correction_at(trace=trace, cutoff=cutoff, target=target, band=band)
    return correction.level, correction.anomaly
