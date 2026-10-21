"""The DP arms of experiment 135: which fits, which filter, and which residuals feed it.

| Arm | Fits | Correction | Filter's residuals |
|---|---|---|---|
| `ts` | `gamk` (experiment 111's two-stage fits) | Kalman | observed minus the proxy-driven fits |
| `j1` | `j1` (joint fit, filter in the loss) | Kalman | the same, from the joint fits |
| `j2_ridge_*` | the same name (joint fit with a ridge towards `ts`) | Kalman | the same |

Every arm uses the same Kalman settings per substation (`hyper`), so the arms differ only in the
fitted coefficients. The settings are experiment 133's chosen values when `work/hyper.json` exists,
and experiment 111's fixed values otherwise.
"""

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final, Literal

import e00135_common as common
import e00135_ens as ens
import e00135_fits as fits_module
import numpy as np
from studies.dp_assimilation import (
    CorrectionTrace,
    FilterHyperparameters,
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
        fits: Which stored fits it uses (`gamk`, `j1`, or `j2`).
        correction: The recent-level correction.
        residuals: Where the correction's residuals come from.
    """

    name: str
    fits: str
    correction: FilterType
    residuals: ResidualSourceType


ARMS: Final[dict[str, ArmConfig]] = {
    "ts": ArmConfig("ts", "gamk", "kalman", "proxy"),
    "j1": ArmConfig("j1", "j1", "kalman", "proxy"),
    **{name: ArmConfig(name, name, "kalman", "proxy") for name in fits_module.J2_RIDGES},
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
    return common.SHARED_WORK_DIR / "calibration" / f"{site}.npz"


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
    """Return the file of experiment 111's tuned settings (solar shrink and the fixed filter)."""
    return common.SHARED_WORK_DIR / "tuned.json"


def read_tuned() -> dict:
    """Return experiment 111's tuned settings (solar shrink, fixed filter settings)."""
    return json.loads(tuned_path().read_text())


def hyper_path() -> Path:
    """Return the file that holds the filter settings every arm uses (experiment 133's values)."""
    return common.WORK_DIR / "hyper.json"


def read_hyper(*, site: str) -> np.ndarray:
    """Return the four unconstrained filter settings of a site, shared by every arm.

    The settings are experiment 133's chosen values when `work/hyper.json` exists, and experiment
    111's fixed values otherwise.
    """
    if hyper_path().exists():
        return np.array(json.loads(hyper_path().read_text())[site])
    return np.array(read_tuned()["filter_hyper"]["gamk"][site])


def factors_path() -> Path:
    """Return the file that holds each arm's spread factor per lead band."""
    return common.WORK_DIR / "factors.json"


def spread_with_factors(*, arm: str, site: str) -> LeadHourSpread:
    """Return an arm's stored spread with the tuned factors of the arm."""
    stored = json.loads(factors_path().read_text()) if factors_path().exists() else {}
    return load_spread(arm=arm, site=site).with_factors(
        factors=stored.get(arm, [1.0] * N_LEAD_BANDS)
    )


def load_arm_fits(*, arm: ArmConfig, site: str) -> list[GamFit]:
    """Return the stored monthly fits of an arm at a site."""
    return fits_module.load_fits(arm=arm.fits, site=site)
