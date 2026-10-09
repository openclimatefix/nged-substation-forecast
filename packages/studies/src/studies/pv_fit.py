"""Fit a plant's physical parameters to its output, with the capacity left free.

The fit minimises a robust loss between the forward model's power and the plant's measured power
over the daytime half-hours. It runs from several starting orientations and keeps the lowest loss,
because the loss has local minima in azimuth and tilt. The clip at the AC capacity is smoothed
during the fit, so that the optimiser sees a gradient on both sides of it, and is hard again when
the fitted plant is scored.
"""

from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from typing import Final, Literal

import numpy as np
from scipy.optimize import least_squares

from studies.pv_physics import (
    PlantParameters,
    SunAndSky,
    daylight,
    plane_of_array_w_m2,
)

OrientationType = Literal["free", "fixed", "tracker"]

FIXED_TILT_DEG: Final[float] = 30.0
"""The tilt of a `fixed` fit (also the middle starting tilt of `free`)."""
FIXED_AZIMUTH_DEG: Final[float] = 180.0
"""The azimuth of a fit whose orientation is `fixed`: due south."""
START_TILTS_DEG: Final[tuple[float, ...]] = (15.0, 30.0, 45.0)
START_AZIMUTHS_DEG: Final[tuple[float, ...]] = (150.0, 180.0, 210.0)
"""The orientations a `free` fit starts from."""
TILT_BOUNDS_DEG: Final[tuple[float, float]] = (0.0, 70.0)
AZIMUTH_BOUNDS_DEG: Final[tuple[float, float]] = (90.0, 270.0)
"""Panels face between due east and due west. Wider azimuths are not physical for a plant."""
DC_AC_RATIO_BOUNDS: Final[tuple[float, float]] = (1.0, 2.2)
START_DC_AC_RATIO: Final[float] = 1.4
CLIP_SHARPNESS: Final[float] = 50.0
"""The smooth clip's sharpness as a multiple of 1 / AC capacity: its rounding spans ~2% of AC."""
ROBUST_SCALE_SHARE: Final[float] = 0.1
"""The robust loss treats residuals under this share of the output's 99th percentile as inliers."""


@dataclass(frozen=True)
class PlantPriors:
    """Gaussian priors on a plant's orientation and DC:AC ratio, for a maximum a posteriori fit.

    Attributes:
        tilt_mean_deg: The prior mean of the panels' tilt.
        tilt_sd_deg: The prior standard deviation of the tilt.
        azimuth_mean_deg: The prior mean of the panels' azimuth, clockwise from north.
        azimuth_sd_deg: The prior standard deviation of the azimuth.
        ratio_mean: The prior mean of the DC:AC ratio.
        ratio_sd: The prior standard deviation of the DC:AC ratio.
    """

    tilt_mean_deg: float
    tilt_sd_deg: float
    azimuth_mean_deg: float
    azimuth_sd_deg: float
    ratio_mean: float
    ratio_sd: float


LOOSE_PRIORS: Final[PlantPriors] = PlantPriors(
    tilt_mean_deg=30.0,
    tilt_sd_deg=20.0,
    azimuth_mean_deg=180.0,
    azimuth_sd_deg=40.0,
    ratio_mean=1.3,
    ratio_sd=0.3,
)
"""Priors from general knowledge, not from any fit: Great Britain's fixed-tilt plants face roughly
south at 25 to 35 degrees, and the median DC:AC ratio of utility-scale plants is about 1.3 (Lawrence
Berkeley National Laboratory's Utility-Scale Solar report). The widths let a plant face well away
from south."""
TIGHT_PRIORS: Final[PlantPriors] = PlantPriors(
    tilt_mean_deg=30.0,
    tilt_sd_deg=10.0,
    azimuth_mean_deg=180.0,
    azimuth_sd_deg=15.0,
    ratio_mean=1.3,
    ratio_sd=0.15,
)
"""The same means as `LOOSE_PRIORS` with about half the widths."""


@dataclass(frozen=True)
class FitResult:
    """A fitted plant and how well it fitted.

    Attributes:
        parameters: The fitted plant.
        loss: The robust loss at the optimum, divided by the number of half-hours fitted.
        half_hours_fitted: The number of daytime half-hours the fit used.
    """

    parameters: PlantParameters
    loss: float
    half_hours_fitted: int


def _smooth_min(*, power: np.ndarray, limit: float) -> np.ndarray:
    """Return a differentiable version of `min(power, limit)`, linear well below the limit."""
    beta = CLIP_SHARPNESS / limit
    return limit - np.logaddexp(0.0, beta * (limit - power)) / beta


def _unpack(*, theta: np.ndarray, orientation: OrientationType) -> PlantParameters:
    """Turn the optimiser's vector into a plant: (log AC, DC:AC ratio[, tilt, azimuth])."""
    ac = float(np.exp(theta[0]))
    dc = ac * float(theta[1])
    if orientation == "free":
        return PlantParameters(
            tilt_deg=float(theta[2]),
            azimuth_deg=float(theta[3]),
            dc_capacity_mw=dc,
            ac_capacity_mw=ac,
        )
    if orientation == "tracker":
        return PlantParameters(
            tilt_deg=0.0, azimuth_deg=180.0, dc_capacity_mw=dc, ac_capacity_mw=ac, tracker=True
        )
    return PlantParameters(
        tilt_deg=FIXED_TILT_DEG, azimuth_deg=FIXED_AZIMUTH_DEG, dc_capacity_mw=dc, ac_capacity_mw=ac
    )


def _starts(*, orientation: OrientationType, ac_guess_mw: float) -> list[np.ndarray]:
    """Return the optimiser's starting vectors."""
    if orientation != "free":
        return [np.array([np.log(ac_guess_mw), START_DC_AC_RATIO])]
    return [
        np.array([np.log(ac_guess_mw), START_DC_AC_RATIO, tilt, azimuth])
        for tilt, azimuth in product(START_TILTS_DEG, START_AZIMUTHS_DEG)
    ]


def _bounds(
    *, orientation: OrientationType, ac_guess_mw: float, ratio_bounds: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Return the lower and upper bounds that go with `_starts`."""
    low = [np.log(ac_guess_mw / 5.0), ratio_bounds[0]]
    high = [np.log(ac_guess_mw * 5.0), ratio_bounds[1]]
    if orientation == "free":
        low += [TILT_BOUNDS_DEG[0], AZIMUTH_BOUNDS_DEG[0]]
        high += [TILT_BOUNDS_DEG[1], AZIMUTH_BOUNDS_DEG[1]]
    return np.array(low), np.array(high)


def fit_plant(
    *,
    sky: SunAndSky,
    output_mw: np.ndarray,
    orientation: OrientationType,
    usable: np.ndarray | None = None,
    ratio_bounds: tuple[float, float] = DC_AC_RATIO_BOUNDS,
) -> FitResult | None:
    """Fit a plant to its measured power over the daytime half-hours.

    The fit never sees a registered capacity. It starts from the output's 99th percentile and may
    move the AC capacity within a factor of five of that start.

    Args:
        sky: The sun and sky, one entry per half-hour.
        output_mw: The measured power at the same half-hours, in megawatts. NaN marks a missing
            half-hour.
        orientation: `free` fits tilt and azimuth, `fixed` uses `FIXED_TILT_DEG` and
            `FIXED_AZIMUTH_DEG`, and `tracker` models a single-axis tracker.
        usable: An optional mask of the half-hours the fit may use, on top of daylight and the
            non-missing output.
        ratio_bounds: The lowest and highest DC:AC ratio the fit may take. Bounds of (1.0, 1.001)
            switch clipping off, up to the plane irradiance passing 1000 W/m².

    Returns:
        The best fit over the starting points, or None when fewer than 100 half-hours are usable
        or the output's 99th percentile is not above zero.
    """
    mask = daylight(sky=sky) & np.isfinite(output_mw)
    if usable is not None:
        mask &= usable
    if mask.sum() < 100:
        return None
    measured = output_mw[mask]
    p99 = float(np.quantile(measured, 0.99))
    if p99 <= 0:
        return None
    # Each starting orientation changes the plane irradiance, so the optimiser recomputes it.
    masked_sky = SunAndSky(
        half_hour_end_time=sky.half_hour_end_time[mask],
        ghi_w_m2=sky.ghi_w_m2[mask],
        dni_w_m2=sky.dni_w_m2[mask],
        dhi_w_m2=sky.dhi_w_m2[mask],
        zenith_deg=sky.zenith_deg[mask],
        sun_azimuth_deg=sky.sun_azimuth_deg[mask],
    )

    def residuals(theta: np.ndarray) -> np.ndarray:
        plant = _unpack(theta=theta, orientation=orientation)
        poa = plane_of_array_w_m2(
            sky=masked_sky,
            tilt_deg=plant.tilt_deg,
            azimuth_deg=plant.azimuth_deg,
            tracker=plant.tracker,
        )
        dc_power = plant.dc_capacity_mw * poa / 1000.0
        return _smooth_min(power=dc_power, limit=plant.ac_capacity_mw) - measured

    low, high = _bounds(orientation=orientation, ac_guess_mw=p99, ratio_bounds=ratio_bounds)
    best: tuple[float, np.ndarray] | None = None
    for start in _starts(orientation=orientation, ac_guess_mw=p99):
        solution = least_squares(
            residuals,
            np.clip(start, low, high),
            bounds=(low, high),
            loss="soft_l1",
            f_scale=ROBUST_SCALE_SHARE * p99,
            x_scale="jac",
        )
        if best is None or solution.cost < best[0]:
            best = (float(solution.cost), solution.x)
    if best is None:
        return None
    return FitResult(
        parameters=_unpack(theta=best[1], orientation=orientation),
        loss=best[0] / int(mask.sum()),
        half_hours_fitted=int(mask.sum()),
    )


def fit_plant_to_changes(
    *,
    sky: SunAndSky,
    output_mw: np.ndarray,
    orientation: OrientationType,
    ac_guess_mw: float,
    lag_half_hours: int,
    ratio_bounds: tuple[float, float] = DC_AC_RATIO_BOUNDS,
    priors: PlantPriors | None = None,
) -> FitResult | None:
    """Fit a plant to the change in an aggregate's output over `lag_half_hours`.

    An aggregate holds generation other than the plant, so its level cannot be fitted. The change in
    its output over a few hours is dominated by the plant when the sun rises or sets or cloud
    passes, because slower parts of the output drop out of the difference. The fit minimises the
    robust loss between the plant's change in power and the aggregate's change in output, over pairs
    of half-hours at least one of which is in daylight.

    Args:
        sky: The sun and sky, one entry per half-hour.
        output_mw: The aggregate's output at the same half-hours. NaN marks a missing half-hour.
        orientation: `free`, `fixed`, or `tracker`, as for `fit_plant`.
        ac_guess_mw: A starting AC capacity, for example from a separation. The fit may move the
            capacity within a factor of five of it.
        lag_half_hours: The lag of the differences.
        ratio_bounds: The lowest and highest DC:AC ratio the fit may take.
        priors: Gaussian priors on tilt, azimuth, and DC:AC ratio, or None for no prior. With
            priors the fit is a maximum a posteriori fit: it first fits without them to estimate
            the noise scale of the changes (1.4826 times the median absolute residual), then adds
            each prior as a residual of `(parameter - mean) / sd` noise scales. A `fixed` or
            `tracker` orientation has no tilt or azimuth to constrain, so only the ratio prior
            applies there.

    Returns:
        The best fit over the starting points, or None when fewer than 500 pairs are usable.
    """
    later = lagged_pairs(sky=sky, output_mw=output_mw, lag_half_hours=lag_half_hours)
    if later.size < 500 or ac_guess_mw <= 0:
        return None
    measured_change = output_mw[later] - output_mw[later - lag_half_hours]

    def residuals(theta: np.ndarray) -> np.ndarray:
        plant = _unpack(theta=theta, orientation=orientation)
        poa = plane_of_array_w_m2(
            sky=sky,
            tilt_deg=plant.tilt_deg,
            azimuth_deg=plant.azimuth_deg,
            tracker=plant.tracker,
        )
        power = _smooth_min(power=plant.dc_capacity_mw * poa / 1000.0, limit=plant.ac_capacity_mw)
        return power[later] - power[later - lag_half_hours] - measured_change

    low, high = _bounds(orientation=orientation, ac_guess_mw=ac_guess_mw, ratio_bounds=ratio_bounds)
    f_scale = ROBUST_SCALE_SHARE * ac_guess_mw
    best = _best_start(
        residuals=residuals,
        starts=_starts(orientation=orientation, ac_guess_mw=ac_guess_mw),
        bounds=(low, high),
        f_scale=f_scale,
    )
    if best is None:
        return None
    if priors is not None:
        noise = 1.4826 * float(np.median(np.abs(residuals(best[1]))))
        noise = max(noise, 1e-6 * ac_guess_mw)

        def residuals_with_priors(theta: np.ndarray) -> np.ndarray:
            return np.concatenate(
                [
                    residuals(theta),
                    _prior_residuals(
                        theta=theta, orientation=orientation, priors=priors, noise=noise
                    ),
                ]
            )

        best = _best_start(
            residuals=residuals_with_priors,
            starts=[best[1], *_starts(orientation=orientation, ac_guess_mw=ac_guess_mw)],
            bounds=(low, high),
            f_scale=f_scale,
        )
        if best is None:
            return None
    return FitResult(
        parameters=_unpack(theta=best[1], orientation=orientation),
        loss=best[0] / later.size,
        half_hours_fitted=int(later.size),
    )


@dataclass(frozen=True)
class RegressorFitResult:
    """A plant fitted together with some free linear regressors.

    Attributes:
        fit: The fitted plant and the loss.
        coefficients: One signed coefficient per regressor column, in megawatts per unit of the
            regressor.
    """

    fit: FitResult
    coefficients: np.ndarray


def fit_plant_to_changes_with_regressors(
    *,
    sky: SunAndSky,
    output_mw: np.ndarray,
    regressors: np.ndarray,
    orientation: OrientationType,
    ac_guess_mw: float,
    lag_half_hours: int,
    ratio_bounds: tuple[float, float] = DC_AC_RATIO_BOUNDS,
) -> RegressorFitResult | None:
    """Fit a plant plus linear regressors to the change in an aggregate's output.

    The fitted output is the plant's power plus `regressors @ coefficients`. The fit is
    `fit_plant_to_changes` (same pairs, same soft-L1 loss, same starting orientations) with the
    coefficients as extra free parameters. They start at zero and have no bounds or priors. With
    no regressor columns the fit is `fit_plant_to_changes` without priors.

    Args:
        sky: The sun and sky, one entry per half-hour.
        output_mw: The aggregate's output at the same half-hours. NaN marks a missing half-hour.
        regressors: An array with one row per half-hour of `sky` and one column per regressor. It
            must be finite wherever a pair of half-hours is used.
        orientation: `free`, `fixed`, or `tracker`, as for `fit_plant`.
        ac_guess_mw: A starting AC capacity. The fit may move the capacity within a factor of five.
        lag_half_hours: The lag of the differences.
        ratio_bounds: The lowest and highest DC:AC ratio the fit may take.

    Returns:
        The fit, or None when fewer than 500 pairs are usable.

    Raises:
        ValueError: If `regressors` has the wrong number of rows, or a non-finite value at a used
            half-hour.
    """
    if regressors.ndim != 2 or regressors.shape[0] != len(output_mw):
        raise ValueError(
            f"regressors must have shape ({len(output_mw)}, K), not {regressors.shape}"
        )
    later = lagged_pairs(sky=sky, output_mw=output_mw, lag_half_hours=lag_half_hours)
    if later.size < 500 or ac_guess_mw <= 0:
        return None
    regressor_changes = regressors[later] - regressors[later - lag_half_hours]
    if not np.isfinite(regressor_changes).all():
        raise ValueError("regressors are not finite at every half-hour of a used pair")
    measured_change = output_mw[later] - output_mw[later - lag_half_hours]
    n_plant = len(_starts(orientation=orientation, ac_guess_mw=ac_guess_mw)[0])
    n_regressors = regressors.shape[1]

    def residuals(theta: np.ndarray) -> np.ndarray:
        plant = _unpack(theta=theta[:n_plant], orientation=orientation)
        poa = plane_of_array_w_m2(
            sky=sky,
            tilt_deg=plant.tilt_deg,
            azimuth_deg=plant.azimuth_deg,
            tracker=plant.tracker,
        )
        power = _smooth_min(power=plant.dc_capacity_mw * poa / 1000.0, limit=plant.ac_capacity_mw)
        plant_change = power[later] - power[later - lag_half_hours]
        return plant_change + regressor_changes @ theta[n_plant:] - measured_change

    low, high = _bounds(orientation=orientation, ac_guess_mw=ac_guess_mw, ratio_bounds=ratio_bounds)
    low = np.concatenate([low, np.full(n_regressors, -np.inf)])
    high = np.concatenate([high, np.full(n_regressors, np.inf)])
    starts = [
        np.concatenate([start, np.zeros(n_regressors)])
        for start in _starts(orientation=orientation, ac_guess_mw=ac_guess_mw)
    ]
    best = _best_start(
        residuals=residuals,
        starts=starts,
        bounds=(low, high),
        f_scale=ROBUST_SCALE_SHARE * ac_guess_mw,
    )
    if best is None:
        return None
    return RegressorFitResult(
        fit=FitResult(
            parameters=_unpack(theta=best[1][:n_plant], orientation=orientation),
            loss=best[0] / later.size,
            half_hours_fitted=int(later.size),
        ),
        coefficients=best[1][n_plant:],
    )


def lagged_pairs(*, sky: SunAndSky, output_mw: np.ndarray, lag_half_hours: int) -> np.ndarray:
    """Return the indices `i` whose output change from `i - lag_half_hours` the fit uses.

    A pair is kept when both outputs are finite, the two stamps are exactly `lag_half_hours`
    half-hours apart (the sky has gaps where the irradiance was unreliable, so a lag in rows is
    not always a lag in time), and either end is in daylight.

    Args:
        sky: The sun and sky, with one entry per row of `output_mw`.
        output_mw: The output at each half-hour of `sky`. NaN marks a missing half-hour.
        lag_half_hours: The lag, in half-hours.

    Returns:
        The later index of each usable pair.
    """
    if lag_half_hours < 1:
        raise ValueError(f"lag_half_hours must be at least 1, not {lag_half_hours}")
    usable = np.isfinite(output_mw)
    later = np.flatnonzero(usable[lag_half_hours:] & usable[:-lag_half_hours]) + lag_half_hours
    stamps = sky.half_hour_end_time
    spacing = stamps[later] - stamps[later - lag_half_hours]
    later = later[spacing == np.timedelta64(30 * lag_half_hours, "m")]
    day = daylight(sky=sky)
    return later[day[later] | day[later - lag_half_hours]]


def change_variance_explained(
    *, sky: SunAndSky, output_mw: np.ndarray, power_mw: np.ndarray, lag_half_hours: int
) -> float:
    """Return the share of the variance of the output's changes that a plant's changes explain.

    Uses the pairs of half-hours `lagged_pairs` selects, the same pairs `fit_plant_to_changes`
    fits. The share is one minus the sum of squared residuals over the sum of squared changes.

    Args:
        sky: The sun and sky, with one entry per row of `output_mw`.
        output_mw: The output at each half-hour of `sky`. NaN marks a missing half-hour.
        power_mw: The plant's power at each half-hour of `sky`.
        lag_half_hours: The lag of the differences.

    Returns:
        The share of the changes' variance explained.
    """
    pairs = lagged_pairs(sky=sky, output_mw=output_mw, lag_half_hours=lag_half_hours)
    change = output_mw[pairs] - output_mw[pairs - lag_half_hours]
    explained = power_mw[pairs] - power_mw[pairs - lag_half_hours]
    return 1.0 - float(((change - explained) ** 2).sum() / (change**2).sum())


def _prior_residuals(
    *, theta: np.ndarray, orientation: OrientationType, priors: PlantPriors, noise: float
) -> np.ndarray:
    """Return each prior as a residual in the units of the data's residuals.

    The data residuals pass through the fit's robust loss, which is linear beyond `f_scale`, and so
    do these. Where the data's noise is larger than `f_scale`, the prior is therefore weaker than
    a Gaussian prior of the stated standard deviation.

    A Gaussian prior contributes `((parameter - mean) / sd) ** 2` to the negative log posterior,
    and each data residual contributes `(residual / noise) ** 2`, so a prior residual of
    `noise * (parameter - mean) / sd` carries the prior's weight against the data.
    """
    terms = [noise * (float(theta[1]) - priors.ratio_mean) / priors.ratio_sd]
    if orientation == "free":
        terms.append(noise * (float(theta[2]) - priors.tilt_mean_deg) / priors.tilt_sd_deg)
        terms.append(noise * (float(theta[3]) - priors.azimuth_mean_deg) / priors.azimuth_sd_deg)
    return np.array(terms)


def _best_start(
    *,
    residuals: Callable[[np.ndarray], np.ndarray],
    starts: list[np.ndarray],
    bounds: tuple[np.ndarray, np.ndarray],
    f_scale: float,
) -> tuple[float, np.ndarray] | None:
    """Return the lowest robust-loss solution over the starting vectors, or None for no start."""
    low, high = bounds
    best: tuple[float, np.ndarray] | None = None
    for start in starts:
        solution = least_squares(
            residuals,
            np.clip(start, low, high),
            bounds=(low, high),
            loss="soft_l1",
            f_scale=f_scale,
            x_scale="jac",
        )
        if best is None or solution.cost < best[0]:
            best = (float(solution.cost), solution.x)
    return best
