"""Separate the solar part of an aggregate's output from its calendar-driven part.

An aggregate's half-hourly output `y(t)` is modelled as `sum_k w_k * B_k(t) + baseline(t)`. The
solar curves `B_k` come from irradiance: one per fleet orientation (east, south, west, and a
single-axis tracker), each the output of one megawatt of AC capacity. The weights `w_k` are
non-negative AC capacities. The baseline is a linear function of the calendar (half-hour of day,
day type, and optionally the season), so it can follow demand's regular pattern but cannot follow
the day-to-day swings of cloud. Cloud is therefore what identifies the solar part. Because every
term is linear in its coefficients, the fit is a bounded least-squares problem with one global
optimum. It is the convex twin of the fleet node in
[Differentiable physics](https://openclimatefix.github.io/nged-substation-forecast/techniques/differentiable-physics/#scaling-to-aggregate-fleets-universalsolarfleetnode).
"""

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import polars as pl
from scipy.optimize import lsq_linear

from studies.pv_physics import SunAndSky, plane_of_array_w_m2

BaselineFlexibilityType = Literal["daily", "seasonal", "monthly"]

FLEET_TILT_DEG: Final[float] = 25.0
"""The tilt of the east-, south-, and west-facing basis curves."""
FLEET_ORIENTATIONS: Final[tuple[tuple[str, float], ...]] = (
    ("east", 90.0),
    ("south", 180.0),
    ("west", 270.0),
    ("tracker", 0.0),
)
"""The basis curves, in column order: a name and the panels' azimuth (unused for the tracker)."""
SEASONAL_HARMONICS: Final[int] = 2
"""The number of annual harmonics that `seasonal` adds to every half-hour of the day."""
DIFFERENCE_LAG_HALF_HOURS: Final[int] = 8
"""The lag, in half-hours, of the difference separation: four hours."""
RIDGE_PER_COLUMN: Final[float] = 1e-3
"""The ridge penalty on the baseline's coefficients, relative to the mean squared column norm."""
LOCAL_TIME_ZONE: Final[str] = "Europe/London"
"""The clock that demand follows, so the baseline's half-hour of day tracks daylight saving."""


@dataclass(frozen=True)
class SeparationResult:
    """The outcome of one separation.

    Attributes:
        capacities_mw: The fitted AC capacity of each orientation, in `FLEET_ORIENTATIONS` order.
        solar_mw: The solar part of the output at each half-hour.
        baseline_mw: The calendar part at each half-hour.
        residual_mw: The measured output minus both parts.
    """

    capacities_mw: np.ndarray
    solar_mw: np.ndarray
    baseline_mw: np.ndarray
    residual_mw: np.ndarray

    @property
    def total_capacity_mw(self) -> float:
        """The sum of the orientations' AC capacities."""
        return float(self.capacities_mw.sum())


def solar_basis(*, sky: SunAndSky, dc_ac_ratio: float) -> np.ndarray:
    """Return the output of one megawatt of AC capacity at each half-hour, per fleet orientation.

    Each column is `min(r * poa / 1000, 1)`: a plant with DC:AC ratio `r` and one megawatt of AC
    capacity clips at that megawatt.

    Args:
        sky: The regional sun and sky.
        dc_ac_ratio: The DC:AC ratio `r` of the fleet's plants.

    Returns:
        An array with one row per half-hour and one column per entry of `FLEET_ORIENTATIONS`.
    """
    columns = []
    for name, azimuth in FLEET_ORIENTATIONS:
        poa = plane_of_array_w_m2(
            sky=sky, tilt_deg=FLEET_TILT_DEG, azimuth_deg=azimuth, tracker=name == "tracker"
        )
        columns.append(np.minimum(dc_ac_ratio * poa / 1000.0, 1.0))
    return np.column_stack(columns)


def baseline_design(
    *, half_hour_end_time: pl.Series, flexibility: BaselineFlexibilityType
) -> np.ndarray:
    """Return the calendar design matrix of the baseline.

    `daily` is one indicator per local half-hour of the day and day type (weekday, Saturday,
    Sunday). `seasonal` adds, for every half-hour of the day, `SEASONAL_HARMONICS` annual cosine and
    sine terms. `monthly` replaces the seasonal terms with an indicator per month and half-hour.

    Args:
        half_hour_end_time: UTC datetimes at the end of each half-hour.
        flexibility: How freely the baseline follows the season.

    Returns:
        An array with one row per half-hour.
    """
    local = half_hour_end_time.dt.replace_time_zone("UTC").dt.convert_time_zone(LOCAL_TIME_ZONE)
    start = local.dt.offset_by("-30m")
    # Polars returns 8-bit integers for these parts, and `day_type * 48 + slot` would overflow them.
    slot = (start.dt.hour().cast(pl.Int64) * 2 + start.dt.minute().cast(pl.Int64) // 30).to_numpy()
    weekday = start.dt.weekday().cast(pl.Int64).to_numpy()
    day_type = np.where(weekday <= 5, 0, weekday - 5)
    n = len(slot)
    daily = np.zeros((n, 48 * 3))
    daily[np.arange(n), day_type * 48 + slot] = 1.0
    if flexibility == "daily":
        return daily
    slot_indicator = np.zeros((n, 48))
    slot_indicator[np.arange(n), slot] = 1.0
    if flexibility == "monthly":
        month = start.dt.month().cast(pl.Int64).to_numpy() - 1
        by_month = np.zeros((n, 48 * 12))
        by_month[np.arange(n), month * 48 + slot] = 1.0
        return np.hstack([daily, by_month])
    phase = 2.0 * np.pi * (start.dt.ordinal_day().cast(pl.Int64).to_numpy() - 1) / 365.25
    terms = [daily]
    for harmonic in range(1, SEASONAL_HARMONICS + 1):
        terms.append(slot_indicator * np.cos(harmonic * phase)[:, None])
        terms.append(slot_indicator * np.sin(harmonic * phase)[:, None])
    return np.hstack(terms)


def separate(
    *,
    output_mw: np.ndarray,
    basis: np.ndarray,
    design: np.ndarray,
    ridge: float = RIDGE_PER_COLUMN,
) -> SeparationResult | None:
    """Fit the aggregate's output as non-negative solar capacities plus a calendar baseline.

    Half-hours with a missing output or a missing basis value are left out of the fit, and the two
    parts are still reported at every half-hour whose basis and design rows exist.

    Args:
        output_mw: The measured output, one entry per half-hour. NaN marks a missing half-hour.
        basis: `solar_basis`'s output, with NaN where the irradiance is missing.
        design: `baseline_design`'s output.
        ridge: The penalty on the baseline's coefficients, relative to the mean squared norm of
            the design's columns. A larger penalty makes the baseline harder to move, so the solar
            curves absorb more of the output's regular daily pattern.

    Returns:
        The fit, or None when fewer than 1,000 half-hours are usable.
    """
    usable = np.isfinite(output_mw) & np.isfinite(basis).all(axis=1)
    if usable.sum() < 1000:
        return None
    n_basis = basis.shape[1]
    penalty = np.sqrt(ridge * float((design[usable] ** 2).sum(axis=0).mean()))
    matrix = np.vstack(
        [
            np.hstack([basis[usable], design[usable]]),
            np.hstack([np.zeros((design.shape[1], n_basis)), penalty * np.eye(design.shape[1])]),
        ]
    )
    target = np.concatenate([output_mw[usable], np.zeros(design.shape[1])])
    lower = np.concatenate([np.zeros(n_basis), np.full(design.shape[1], -np.inf)])
    solution = lsq_linear(
        matrix, target, bounds=(lower, np.full(matrix.shape[1], np.inf)), method="bvls"
    )
    capacities = solution.x[:n_basis]
    solar = np.nan_to_num(basis) @ capacities
    baseline = design @ solution.x[n_basis:]
    return SeparationResult(
        capacities_mw=capacities,
        solar_mw=solar,
        baseline_mw=baseline,
        residual_mw=output_mw - solar - baseline,
    )


def separate_by_differences(
    *,
    output_mw: np.ndarray,
    basis: np.ndarray,
    design: np.ndarray,
    lag_half_hours: int = DIFFERENCE_LAG_HALF_HOURS,
) -> SeparationResult | None:
    """Fit the solar capacities to the change in output over `lag_half_hours`, then the baseline.

    The separation of `separate` reads the solar level from the day-to-day swings in cloud that
    the calendar cannot explain. Other generation that follows the weather, such as wind, varies
    with those swings too, and biases the level. Differencing over a few hours removes every part
    of the output that changes more slowly than that, so the solar capacities are fitted only to
    changes that happen within hours, when the sun rises or sets or cloud passes. The baseline is
    then fitted to the output minus the solar part, so the three parts still sum to the output.

    Args:
        output_mw: The measured output, one entry per half-hour. NaN marks a missing half-hour.
        basis: `solar_basis`'s output, with NaN where the irradiance is missing.
        design: `baseline_design`'s output.
        lag_half_hours: The lag of the differences, in half-hours.

    Returns:
        The fit, or None when fewer than 1,000 pairs of half-hours are usable.
    """
    usable = np.isfinite(output_mw) & np.isfinite(basis).all(axis=1)
    later = np.flatnonzero(usable[lag_half_hours:] & usable[:-lag_half_hours]) + lag_half_hours
    if later.size < 1000:
        return None
    changes = output_mw[later] - output_mw[later - lag_half_hours]
    basis_changes = basis[later] - basis[later - lag_half_hours]
    weights = lsq_linear(basis_changes, changes, bounds=(0.0, np.inf), method="bvls").x
    solar = np.nan_to_num(basis) @ weights
    remainder = np.where(usable, output_mw - solar, np.nan)
    baseline_fit = separate(output_mw=remainder, basis=np.zeros((len(output_mw), 1)), design=design)
    baseline = np.zeros(len(output_mw)) if baseline_fit is None else baseline_fit.baseline_mw
    return SeparationResult(
        capacities_mw=weights,
        solar_mw=solar,
        baseline_mw=baseline,
        residual_mw=output_mw - solar - baseline,
    )
