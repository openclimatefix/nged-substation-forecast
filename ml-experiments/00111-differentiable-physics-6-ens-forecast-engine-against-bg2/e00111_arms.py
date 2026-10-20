"""The DP arms of experiment 111: which fits, which filter, and which residuals feed it.

| Arm | Fits | Correction | Filter's residuals |
|---|---|---|---|
| `gam0` | `gam0` (drift D1) | none | |
| `gamk` | `gamk` (no drift) | Kalman | observed minus the proxy-driven fits |
| `game` | `gamk` | EWMA, 14-day half-life | the same |
| `gamk_era5` | `gamk` | Kalman | observed minus the ERA5- and CAMS-driven fits |

`gamk_era5` is the filter on estimates of past weather, which a live service would not have; it
uses Part 4's fitted hyperparameters, because its residuals are Part 4's kind.
"""

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final, Literal

import e00111_common as common
import e00111_ens as ens
import e00111_fits as fits_module
import numpy as np
from studies.dp_assimilation import (
    MIN_PRIOR_SPREAD,
    CorrectionTrace,
    FilterHyperparameters,
    fit_site_hyperparameters,
    run_ewma,
    run_kalman,
)
from studies.dp_ens import LeadHourSpread, proxy_residual_stream
from studies.dp_ens_weather import N_LEAD_BANDS, Calibration, proxy_cache
from studies.dp_features import FeatureCache
from studies.dp_gam import GamFit, ResidualSpread

FilterType = Literal["none", "kalman", "ewma"]
ResidualSourceType = Literal["proxy", "era5"]
EWMA_HALF_LIFE_DAYS: Final[float] = 14.0


@dataclass(frozen=True)
class ArmConfig:
    """One DP arm.

    Attributes:
        name: The arm's name.
        fits: Which stored fits it uses (`gam0` or `gamk`).
        correction: The recent-level correction.
        residuals: Where the correction's residuals come from.
    """

    name: str
    fits: str
    correction: FilterType
    residuals: ResidualSourceType


ARMS: Final[dict[str, ArmConfig]] = {
    "gam0": ArmConfig("gam0", "gam0", "none", "proxy"),
    "gamk": ArmConfig("gamk", "gamk", "kalman", "proxy"),
    "game": ArmConfig("game", "gamk", "ewma", "proxy"),
    "gamk_era5": ArmConfig("gamk_era5", "gamk", "kalman", "era5"),
}


def in_months(*, cache: FeatureCache, months: frozenset[tuple[int, int]]) -> np.ndarray:
    """Return a mask of the grid rows whose half-hour starts in one of the (year, month) pairs."""
    starts = cache.end_times - np.timedelta64(30, "m")
    year = starts.astype("datetime64[Y]").astype(np.int64) + 1970
    month = starts.astype("datetime64[M]").astype(np.int64) % 12 + 1
    mask = np.zeros(len(starts), dtype=bool)
    for y, m in months:
        mask |= (year == y) & (month == m)
    return mask


def part_4_hyper(*, site: str) -> np.ndarray | None:
    """Return Part 4's fitted hyperparameters of the no-drift model at a site, if it has any."""
    tuned = json.loads(common.PART_4_TUNED.read_text())["filter_hyper"]["a0"]
    return np.array(tuned[site]) if site in tuned else None


def part_4_prior() -> tuple[np.ndarray, np.ndarray]:
    """Return the mean and spread of Part 4's per-site hyperparameters of the no-drift model."""
    tuned = json.loads(common.PART_4_TUNED.read_text())["filter_hyper"]["a0"]
    stacked = np.array(list(tuned.values()))
    return stacked.mean(axis=0), np.maximum(stacked.std(axis=0, ddof=1), MIN_PRIOR_SPREAD)


def residual_stream(
    *, cache: FeatureCache, setup: ens.SiteSetup, fits: list[GamFit], source: ResidualSourceType
) -> np.ndarray:
    """Return a site's residual stream: observed minus the chained fits' forecast.

    Args:
        cache: The feature cache.
        setup: The substation's setup, with its calibration.
        fits: The monthly fits.
        source: `proxy` for the analysis proxy's weather, `era5` for ERA5 and CAMS.

    Returns:
        A stream on the grid in points.
    """
    driven = proxy_cache(cache=cache, context=setup.context) if source == "proxy" else cache
    return proxy_residual_stream(cache=driven, fits=fits, site=setup.site)


def fit_filter_hyper(
    *, cache: FeatureCache, setup: ens.SiteSetup, fits: list[GamFit]
) -> np.ndarray:
    """Fit a site's Kalman hyperparameters on the tuning months' proxy-driven residuals.

    The prior is Part 4's fitted prior and the start is Part 4's fit at the site. Only residuals
    of half-hours in the tuning months enter the likelihood.

    Args:
        cache: The feature cache.
        setup: The substation's setup, with its calibration.
        fits: The monthly fits of `gamk`.

    Returns:
        The four unconstrained values.
    """
    stream = residual_stream(cache=cache, setup=setup, fits=fits, source="proxy")
    stream = np.where(in_months(cache=cache, months=common.TUNING_MONTHS), stream, np.nan)
    prior_mean, prior_spread = part_4_prior()
    start = part_4_hyper(site=setup.site)
    result = fit_site_hyperparameters(
        residual=stream,
        band=cache.calendar.band,
        prior_mean=prior_mean,
        prior_spread=prior_spread,
        start=prior_mean if start is None else start,
    )
    return result.as_array()


def build_trace(
    *,
    cache: FeatureCache,
    setup: ens.SiteSetup,
    arm: ArmConfig,
    fits: list[GamFit],
    hyper: np.ndarray | None,
) -> CorrectionTrace | None:
    """Return an arm's correction trace for a site, or `None` for an arm with no correction.

    Args:
        cache: The feature cache.
        setup: The substation's setup, with its calibration.
        arm: The arm.
        fits: The arm's monthly fits.
        hyper: The Kalman hyperparameters at this site, for a Kalman arm.

    Returns:
        The trace.
    """
    if arm.correction == "none":
        return None
    stream = residual_stream(cache=cache, setup=setup, fits=fits, source=arm.residuals)
    if arm.correction == "ewma":
        return run_ewma(
            residual=stream, band=cache.calendar.band, half_life_days=EWMA_HALF_LIFE_DAYS
        )
    if hyper is None:
        msg = f"the Kalman arm {arm.name} needs hyperparameters at {setup.site}"
        raise ValueError(msg)
    return partial(run_kalman, hyper=FilterHyperparameters.from_array(values=hyper))(
        residual=stream, band=cache.calendar.band
    )


def spread_path(*, arm: str, site: str) -> Path:
    """Return the file that holds an arm's spread at a site."""
    return common.WORK_DIR / "spread" / f"{arm}_{site}.npz"


def save_spread(*, arm: str, site: str, spread: LeadHourSpread) -> None:
    """Store a spread's tables and factors."""
    path = spread_path(arm=arm, site=site)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        tables=np.stack([band.table for band in spread.by_lead_band]),
        factors=spread.factors,
    )


def load_spread(*, arm: str, site: str) -> LeadHourSpread:
    """Return a stored spread."""
    with np.load(spread_path(arm=arm, site=site)) as stored:
        tables, factors = stored["tables"], stored["factors"]
    return LeadHourSpread(
        by_lead_band=tuple(ResidualSpread(table=table) for table in tables),
        factors=factors,
    )


def calibration_path(*, site: str) -> Path:
    """Return the file that holds a site's calibration."""
    return common.WORK_DIR / "calibration" / f"{site}.npz"


def save_calibration(*, site: str, calibration: Calibration) -> None:
    """Store a site's calibration."""
    path = calibration_path(site=site)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        shift=calibration.temperature_shift_c,
        ratio=calibration.irradiance_ratio,
        shrink=calibration.solar_shrink,
    )


def load_calibration(*, site: str) -> Calibration:
    """Return a stored calibration."""
    with np.load(calibration_path(site=site)) as stored:
        return Calibration(
            temperature_shift_c=stored["shift"],
            irradiance_ratio=stored["ratio"],
            solar_shrink=stored["shrink"],
        )


def tuned_path() -> Path:
    """Return the file that holds the tuned settings shared by the sites."""
    return common.WORK_DIR / "tuned.json"


def read_tuned() -> dict:
    """Return the tuned settings (solar shrink, filter hyperparameters, spread factors)."""
    return json.loads(tuned_path().read_text())


def spread_with_factors(*, arm: str, site: str, tuned: dict) -> LeadHourSpread:
    """Return an arm's stored spread with the tuned factors of the arm."""
    factors = tuned["spread_factors"].get(arm, [1.0] * N_LEAD_BANDS)
    return load_spread(arm=arm, site=site).with_factors(factors=factors)


def load_arm_fits(*, arm: ArmConfig, site: str) -> list[GamFit]:
    """Return the stored monthly fits of an arm at a site."""
    return fits_module.load_fits(arm=arm.fits, site=site)
