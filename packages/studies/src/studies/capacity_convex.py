"""Estimate a plant's capacity with linear programs, for the study in issue #1143.

**Each estimate is one linear program over the latest 60 days, solved by `scipy.optimize.linprog`
(HiGHS).** The program fits a daily direct-current capacity trace to the plant's hourly power, with
a fused-lasso penalty on the day-to-day change so that the trace is piecewise constant, and the
estimate for day `d` is the trace's value on day `d`. A program reads days up to `d` only, so the
estimate is causal.

**CX1 fits one trace to the clear-sky envelope.** Power is the trace times the plant's modelled
output under a clear sky, and a pinball loss at a high quantile makes the trace the upper envelope
of the ratio, so that cloudy hours fall below it.

**CX2 fits a direct-current and an alternating-current trace to satellite-driven output.** It
follows the roadmap's Candidate A with two departures: the alternating-current fit uses absolute
error, so that the whole program stays linear, and the tilt and azimuth are fixed from the plant's
orientation fit rather than grid-searched. A classifier marks the hours near the window's highest
power as clipped. Clipped hours fit the alternating-current trace and carry a hinge that requires
the direct-current output to have reached the alternating-current limit. All other hours fit the
direct-current trace by pinball loss.

Power is divided by the plant's static power before the program, so the traces are dimensionless and
the penalty weights do not depend on the plant's size.
"""

from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from studies.capacity_estimators import (
    CapacityRecord,
    PlantEstimate,
    RawEstimate,
    valid_days,
)

WINDOW_DAYS: Final[int] = 60
"""How many days, ending at the estimate's day, one program reads."""

MIN_UNIT_OUTPUT: Final[float] = 0.15
"""Hours whose modelled output per megawatt of direct-current capacity is below this are left
out."""

MIN_DAYS_WITH_POINTS: Final[int] = 15
"""A program needs this many days that each hold at least one usable hour."""

MIN_POINTS: Final[int] = 100
"""A program needs this many usable hours in all."""

CLIP_SHARE: Final[float] = 0.97
"""An hour is classified as clipped when its power is at least this share of the window's 99th
percentile."""

CONVEX_DEFAULT_PARAMS: Final[Mapping[str, Mapping[str, float]]] = {
    "CX1": {"quantile": 0.9, "fused_weight": 10.0},
    "CX2": {"quantile": 0.7, "fused_weight": 10.0, "hinge_weight": 1.0},
}
"""The settings the tuning draw starts from."""


class WindowPoints:
    """The usable hours of one window: which day, the power, and the modelled output."""

    def __init__(self, *, day: np.ndarray, power: np.ndarray, unit_output: np.ndarray) -> None:
        """Hold the points.

        Args:
            day: Each point's day within the window, from 0.
            power: Each point's power over the static power.
            unit_output: Each point's modelled output per unit of direct-current capacity.
        """
        self.day = day
        self.power = power
        self.unit_output = unit_output


def window_points(
    *,
    record: CapacityRecord,
    plant: int,
    end_day: int,
    unit_output: np.ndarray,
    valid: np.ndarray,
) -> WindowPoints | None:
    """Collect the usable hours of the `WINDOW_DAYS` days ending at `end_day`.

    Args:
        record: The record.
        plant: The plant's index.
        end_day: The last day of the window.
        unit_output: The modelled output per unit of capacity, shape (days, 24) for this plant.
        valid: The record's valid days for this plant, shape (days,).

    Returns:
        The points, or `None` if the window holds fewer than `MIN_DAYS_WITH_POINTS` days with
        points or `MIN_POINTS` points. Days before the record starts are not in the window.
    """
    first = end_day - WINDOW_DAYS + 1
    days = np.arange(max(first, 0), end_day + 1)
    power = record.power_mw[plant, days] / record.static_p99_mw[plant]
    shape = unit_output[days]
    usable = (
        valid[days][:, None] & np.isfinite(power) & np.isfinite(shape) & (shape >= MIN_UNIT_OUTPUT)
    )
    day_index, hour_index = np.nonzero(usable)
    if len(day_index) < MIN_POINTS or len(np.unique(day_index)) < MIN_DAYS_WITH_POINTS:
        return None
    return WindowPoints(
        day=day_index + (days[0] - first),
        power=power[day_index, hour_index],
        unit_output=shape[day_index, hour_index],
    )


def _solve(
    *,
    cost: np.ndarray,
    upper: sparse.spmatrix,
    upper_rhs: np.ndarray,
    equal: sparse.spmatrix,
    equal_rhs: np.ndarray,
    plant: str,
    end_day: int,
) -> np.ndarray:
    """Solve a linear program with non-negative variables, raising unless it is optimal.

    Args:
        cost: The objective's coefficients.
        upper: The inequality matrix `A x <= b`.
        upper_rhs: The inequality right-hand side.
        equal: The equality matrix.
        equal_rhs: The equality right-hand side.
        plant: The plant's label, for the error message.
        end_day: The window's last day, for the error message.

    Returns:
        The optimal variables.

    Raises:
        RuntimeError: If HiGHS does not report an optimal solution.
    """
    result = linprog(
        cost,
        A_ub=upper,
        b_ub=upper_rhs,
        A_eq=equal,
        b_eq=equal_rhs,
        bounds=(0, None),
        method="highs",
    )
    if result.status != 0:
        msg = (
            f"the linear program for plant {plant} ending on day index {end_day} was not solved "
            f"to optimality: {result.message}"
        )
        raise RuntimeError(msg)
    return result.x


def _fused_rows(
    *, offset: int, tv_offset: int, n_days: int, total: int
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Return the inequality rows tying a trace's day-to-day change to its fused-lasso variables.

    Args:
        offset: The column of the trace's first day.
        tv_offset: The column of the first change variable.
        n_days: The number of days in the window.
        total: The program's number of variables.

    Returns:
        The matrix (two rows per change) and its zero right-hand side.
    """
    steps = np.arange(n_days - 1)
    rows = np.concatenate(
        [2 * steps, 2 * steps, 2 * steps, 2 * steps + 1, 2 * steps + 1, 2 * steps + 1]
    )
    cols = np.concatenate(
        [
            offset + steps + 1,
            offset + steps,
            tv_offset + steps,
            offset + steps,
            offset + steps + 1,
            tv_offset + steps,
        ]
    )
    data = np.concatenate([np.ones(len(steps)), -np.ones(len(steps)), -np.ones(len(steps))] * 2)
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(2 * (n_days - 1), total)).tocsr()
    return matrix, np.zeros(2 * (n_days - 1))


def cx1_trace(
    *,
    points: WindowPoints,
    quantile: float,
    fused_weight: float,
    plant: str,
    end_day: int,
    n_days: int = WINDOW_DAYS,
) -> np.ndarray:
    """Solve CX1's program and return the daily trace.

    Variables: the trace `c` (one per day), the positive and negative residuals of each point, and
    the absolute day-to-day changes. The objective is the pinball loss at `quantile` of each
    point's power minus its unit output times the trace, plus `fused_weight` times the sum of the
    changes.

    Args:
        points: The window's usable hours.
        quantile: The pinball loss's quantile.
        fused_weight: The weight of the fused-lasso penalty.
        plant: The plant's label, for an error message.
        end_day: The window's last day, for an error message.
        n_days: The window's length in days (a short window is for tests).

    Returns:
        The trace, one value per day of the window.
    """
    n = len(points.day)
    total = n_days + 2 * n + (n_days - 1)
    cost = np.concatenate(
        [
            np.zeros(n_days),
            np.full(n, quantile),
            np.full(n, 1.0 - quantile),
            np.full(n_days - 1, fused_weight),
        ]
    )
    point = np.arange(n)
    equal = sparse.coo_matrix(
        (
            np.concatenate([points.unit_output, np.ones(n), -np.ones(n)]),
            (
                np.concatenate([point, point, point]),
                np.concatenate([points.day, n_days + point, n_days + n + point]),
            ),
        ),
        shape=(n, total),
    ).tocsr()
    upper, upper_rhs = _fused_rows(offset=0, tv_offset=n_days + 2 * n, n_days=n_days, total=total)
    return _solve(
        cost=cost,
        upper=upper,
        upper_rhs=upper_rhs,
        equal=equal,
        equal_rhs=points.power,
        plant=plant,
        end_day=end_day,
    )[:n_days]


def cx1_objective(
    *, trace: np.ndarray, points: WindowPoints, quantile: float, fused_weight: float
) -> float:
    """Return CX1's objective at a trace, for checking the program against brute force.

    Args:
        trace: One value per day.
        points: The usable hours.
        quantile: The pinball loss's quantile.
        fused_weight: The fused-lasso weight.

    Returns:
        The pinball loss plus the penalty.
    """
    residual = points.power - points.unit_output * trace[points.day]
    pinball = np.where(residual >= 0, quantile * residual, (quantile - 1.0) * residual).sum()
    return float(pinball + fused_weight * np.abs(np.diff(trace)).sum())


def classify_clipped(*, points: WindowPoints) -> np.ndarray:
    """Mark the hours at or near the window's highest power as clipped.

    Args:
        points: The usable hours.

    Returns:
        A boolean array, one entry per point.
    """
    return points.power >= CLIP_SHARE * np.quantile(points.power, 0.99)


def cx2_traces(
    *,
    points: WindowPoints,
    quantile: float,
    fused_weight: float,
    hinge_weight: float,
    plant: str,
    end_day: int,
    n_days: int = WINDOW_DAYS,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve CX2's program and return the direct-current and alternating-current traces.

    Unclipped hours fit the direct-current trace `c_dc` by pinball loss. Clipped hours fit the
    alternating-current trace `c_ac` by absolute error and add the hinge
    `max(0, c_ac - unit_output * c_dc)`. Both traces carry the fused-lasso penalty.

    Args:
        points: The window's usable hours.
        quantile: The pinball loss's quantile.
        fused_weight: The fused-lasso weight.
        hinge_weight: The weight of the censoring hinge.
        plant: The plant's label, for an error message.
        end_day: The window's last day, for an error message.
        n_days: The window's length in days (a short window is for tests).

    Returns:
        The two traces, one value per day of the window.
    """
    clipped = classify_clipped(points=points)
    free = ~clipped
    n_free, n_clip = int(free.sum()), int(clipped.sum())
    # Columns: c_dc, c_ac, free residuals (+, -), clipped residuals (+, -), hinge, changes (dc, ac).
    offsets = {
        "dc": 0,
        "ac": n_days,
        "up": 2 * n_days,
        "un": 2 * n_days + n_free,
        "ap": 2 * n_days + 2 * n_free,
        "an": 2 * n_days + 2 * n_free + n_clip,
        "hinge": 2 * n_days + 2 * n_free + 2 * n_clip,
        "tv_dc": 2 * n_days + 2 * n_free + 3 * n_clip,
        "tv_ac": 2 * n_days + 2 * n_free + 3 * n_clip + n_days - 1,
    }
    total = offsets["tv_ac"] + n_days - 1
    cost = np.zeros(total)
    cost[offsets["up"] : offsets["up"] + n_free] = quantile
    cost[offsets["un"] : offsets["un"] + n_free] = 1.0 - quantile
    cost[offsets["ap"] : offsets["ap"] + 2 * n_clip] = 1.0
    cost[offsets["hinge"] : offsets["hinge"] + n_clip] = hinge_weight
    cost[offsets["tv_dc"] :] = fused_weight
    free_index, clip_index = np.arange(n_free), np.arange(n_clip)
    free_days, clip_days = points.day[free], points.day[clipped]
    free_output, clip_output = points.unit_output[free], points.unit_output[clipped]
    equal = sparse.coo_matrix(
        (
            np.concatenate(
                [
                    free_output,
                    np.ones(n_free),
                    -np.ones(n_free),
                    np.ones(n_clip),
                    np.ones(n_clip),
                    -np.ones(n_clip),
                ]
            ),
            (
                np.concatenate([free_index] * 3 + [n_free + clip_index] * 3),
                np.concatenate(
                    [
                        free_days,
                        offsets["up"] + free_index,
                        offsets["un"] + free_index,
                        n_days + clip_days,
                        offsets["ap"] + clip_index,
                        offsets["an"] + clip_index,
                    ]
                ),
            ),
        ),
        shape=(n_free + n_clip, total),
    ).tocsr()
    equal_rhs = np.concatenate([points.power[free], points.power[clipped]])
    hinge_rows = sparse.coo_matrix(
        (
            np.concatenate([np.ones(n_clip), -clip_output, -np.ones(n_clip)]),
            (
                np.concatenate([clip_index] * 3),
                np.concatenate([n_days + clip_days, clip_days, offsets["hinge"] + clip_index]),
            ),
        ),
        shape=(n_clip, total),
    ).tocsr()
    fused_dc, _ = _fused_rows(
        offset=offsets["dc"], tv_offset=offsets["tv_dc"], n_days=n_days, total=total
    )
    fused_ac, _ = _fused_rows(
        offset=offsets["ac"], tv_offset=offsets["tv_ac"], n_days=n_days, total=total
    )
    upper = sparse.vstack([hinge_rows, fused_dc, fused_ac]).tocsr()
    upper_rhs = np.zeros(upper.shape[0])
    solution = _solve(
        cost=cost,
        upper=upper,
        upper_rhs=upper_rhs,
        equal=equal,
        equal_rhs=equal_rhs,
        plant=plant,
        end_day=end_day,
    )
    return solution[:n_days], solution[n_days : 2 * n_days]


def cx2_objective(
    *,
    dc_trace: np.ndarray,
    ac_trace: np.ndarray,
    points: WindowPoints,
    quantile: float,
    fused_weight: float,
    hinge_weight: float,
) -> float:
    """Return CX2's objective at a pair of traces, for checking the program against brute force.

    Args:
        dc_trace: The direct-current trace.
        ac_trace: The alternating-current trace.
        points: The usable hours.
        quantile: The pinball loss's quantile.
        fused_weight: The fused-lasso weight.
        hinge_weight: The hinge's weight.

    Returns:
        The full objective.
    """
    clipped = classify_clipped(points=points)
    free = ~clipped
    residual = points.power[free] - points.unit_output[free] * dc_trace[points.day[free]]
    pinball = np.where(residual >= 0, quantile * residual, (quantile - 1.0) * residual).sum()
    plateau = np.abs(points.power[clipped] - ac_trace[points.day[clipped]]).sum()
    hinge = np.maximum(
        0.0,
        ac_trace[points.day[clipped]] - points.unit_output[clipped] * dc_trace[points.day[clipped]],
    ).sum()
    penalty = np.abs(np.diff(dc_trace)).sum() + np.abs(np.diff(ac_trace)).sum()
    return float(pinball + plateau + hinge_weight * hinge + fused_weight * penalty)


def estimate_convex_plant(
    *,
    name: str,
    record: CapacityRecord,
    plant: int,
    params: Mapping[str, float],
    days: Sequence[int] | np.ndarray,
) -> PlantEstimate:
    """Estimate one plant's capacity on the given days with CX1 or CX2.

    Args:
        name: `CX1` or `CX2`.
        record: The record.
        plant: The plant's index.
        params: The estimator's settings (see `CONVEX_DEFAULT_PARAMS`).
        days: The day indices to estimate. Every other day is NaN.

    Returns:
        The level (the trace's value on each requested day, dimensionless) with no standard
        deviation.

    Raises:
        ValueError: If `name` is not a convex estimator.
    """
    if name not in CONVEX_DEFAULT_PARAMS:
        msg = f"{name} is not a convex estimator"
        raise ValueError(msg)
    valid = valid_days(record=record)[plant]
    unit_output = record.clear_shape[plant] if name == "CX1" else record.cams_shape[plant]
    level = np.full(len(record.dates), np.nan)
    for day in days:
        points = window_points(
            record=record, plant=plant, end_day=int(day), unit_output=unit_output, valid=valid
        )
        if points is None:
            continue
        if name == "CX1":
            trace = cx1_trace(
                points=points,
                quantile=params["quantile"],
                fused_weight=params["fused_weight"],
                plant=record.sites[plant],
                end_day=int(day),
            )
        else:
            trace, _ = cx2_traces(
                points=points,
                quantile=params["quantile"],
                fused_weight=params["fused_weight"],
                hinge_weight=params["hinge_weight"],
                plant=record.sites[plant],
                end_day=int(day),
            )
        level[day] = trace[-1]
    return PlantEstimate(level=level, sd=None)


def run_convex_estimator(
    *,
    name: str,
    record: CapacityRecord,
    params: Mapping[str, float] | None,
    days_by_plant: Sequence[Sequence[int] | np.ndarray],
) -> RawEstimate:
    """Run CX1 or CX2 on every plant at that plant's own days.

    Args:
        name: `CX1` or `CX2`.
        record: The record.
        params: The settings, or `None` for `CONVEX_DEFAULT_PARAMS`.
        days_by_plant: The day indices to estimate, one list per plant.

    Returns:
        The raw level for every plant and day, NaN on days not estimated.
    """
    settings = CONVEX_DEFAULT_PARAMS[name] if params is None else params
    parts = [
        estimate_convex_plant(name=name, record=record, plant=plant, params=settings, days=days)
        for plant, days in enumerate(days_by_plant)
    ]
    return RawEstimate(level=np.stack([part.level for part in parts]), sd=None)
