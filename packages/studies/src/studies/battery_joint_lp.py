"""Fit solar plus a battery to an aggregate with one linear programme.

The aggregate `y(t)` is modelled as `sum_k w_k * basis_k(t) + u(t)` plus a residual, and
optionally plus `sum_j b_j * column_j(t)`: free, signed columns such as a calendar baseline or a
price regressor. The solar
weights `w_k` are non-negative megawatts of AC capacity for each fleet orientation's one-megawatt
output curve. The battery's output `u(t) = d(t) - c(t)` is free, subject to the physical limits
`0 <= d, c <= P` and a state of charge `SoC` that obeys

    SoC(t) = SoC(t - 1) + eta * c(t) * dt - d(t) * dt / eta,    0 <= SoC <= E.

The linear programme minimises the sum of absolute residuals, plus `smoothness_penalty` times the
sum of absolute half-hour changes of `u`, plus `throughput_penalty` times the sum of `c + d`.

Simultaneous charging and discharging only wastes energy, so without a cost on throughput it is a
free way to dump energy, and it would let an empty battery (`E = 0`) act as a load of up to
`(1 - eta**2)` times its power. Dumping one megawatt of net output costs `(1 + eta**2) /
(1 - eta**2)` megawatts of throughput, so a throughput penalty above `(1 - eta**2) / (1 + eta**2)`
(0.08 at `eta = 0.92`) makes dumping cost more than the residual it removes, which is 1 per
megawatt. The default is 0.1.

The signed coefficients `b_j` are free, with a penalty of `signed_column_penalty` per unit of
`|b_j|`. A linear programme cannot hold the ridge penalty of a least-squares fit, so this small
absolute penalty plays its part: it picks one answer among equally good ones and keeps the
coefficients finite.

The state of charge is free at the start of each window, and the weights are shared by all windows.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

HOURS_PER_HALF_HOUR: Final[float] = 0.5
DEFAULT_ONE_WAY_EFFICIENCY: Final[float] = 0.92
DEFAULT_THROUGHPUT_PENALTY: Final[float] = 0.1
"""Cost per megawatt of charging or discharging, against 1 per megawatt of residual."""
DEFAULT_SIGNED_COLUMN_PENALTY: Final[float] = 1e-3
"""Cost per unit of the absolute value of a signed column's coefficient, against 1 per megawatt of
residual."""


@dataclass(frozen=True)
class JointFit:
    """The solution of `fit_joint_solar_battery`.

    Attributes:
        solar_weights: The fitted AC capacity of each fleet orientation, in megawatts.
        solar_mw: The fitted solar output at every half-hour.
        battery_mw: The fitted battery output, discharge minus charge, at every half-hour.
        soc_mwh: The state of charge at the end of every half-hour.
        residual_mw: The aggregate minus the solar and the battery; NaN where the aggregate is.
        window_starts: The first half-hour of each window.
        signed_coefficients: The coefficient of each signed column; empty when there are none.
        signed_mw: The signed columns' contribution at every half-hour; zero when there are none.
    """

    solar_weights: np.ndarray
    solar_mw: np.ndarray
    battery_mw: np.ndarray
    soc_mwh: np.ndarray
    residual_mw: np.ndarray
    window_starts: np.ndarray
    signed_coefficients: np.ndarray
    signed_mw: np.ndarray


class _Layout:
    """Column positions of each variable block of the linear programme."""

    def __init__(
        self, *, n_basis: int, n_times: int, n_windows: int, smooth: bool, n_signed: int
    ) -> None:
        self.n_times = n_times
        self.weights = 0
        self.discharge = n_basis
        self.charge = self.discharge + n_times
        self.soc = self.charge + n_times
        self.soc_start = self.soc + n_times
        self.residual_up = self.soc_start + n_windows
        self.residual_down = self.residual_up + n_times
        self.step_up = self.residual_down + n_times
        self.step_down = self.step_up + (n_times if smooth else 0)
        self.signed_up = self.step_down + (n_times if smooth else 0)
        self.signed_down = self.signed_up + n_signed
        self.size = self.signed_down + n_signed


def _window_starts(*, n_times: int, window_half_hours: int | None) -> np.ndarray:
    """Return the first half-hour of each window; the windows have near-equal lengths."""
    if window_half_hours is None or window_half_hours >= n_times:
        return np.array([0])
    n_windows = max(1, round(n_times / window_half_hours))
    return np.array([chunk[0] for chunk in np.array_split(np.arange(n_times), n_windows)])


def _equality_system(
    *,
    layout: _Layout,
    basis: np.ndarray,
    aggregate_mw: np.ndarray,
    starts: np.ndarray,
    one_way_efficiency: float,
    smooth: bool,
    signed: sparse.csr_matrix,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Build the equality constraints: the residual identity, the energy balance, and the steps.

    Args:
        layout: The variable positions.
        basis: The solar basis with NaN replaced by 0.
        aggregate_mw: The aggregate, with NaN half-hours that get no residual identity.
        starts: The first half-hour of each window.
        one_way_efficiency: The efficiency of charging and, separately, of discharging.
        smooth: Whether to add the rows that define the half-hour steps of the battery's output.
        signed: The signed columns, one row per half-hour; no columns when there are none.

    Returns:
        The constraint matrix and its right-hand side.
    """
    n_times, n_basis = basis.shape
    times = np.arange(n_times)
    finite = np.flatnonzero(np.isfinite(aggregate_mw))
    n_fit = len(finite)
    fit_rows = np.arange(n_fit)
    entries: list[tuple[np.ndarray, np.ndarray, np.ndarray | float]] = []
    # The residual identity at each finite half-hour:
    # solar + signed + discharge - charge + up - down = y.
    entries += [
        (fit_rows, np.full(n_fit, layout.weights + k), basis[finite, k]) for k in range(n_basis)
    ]
    signed_rows = signed[finite].tocoo()
    entries += [
        (signed_rows.row, layout.signed_up + signed_rows.col, signed_rows.data),
        (signed_rows.row, layout.signed_down + signed_rows.col, -signed_rows.data),
    ]
    entries += [
        (fit_rows, layout.discharge + finite, 1.0),
        (fit_rows, layout.charge + finite, -1.0),
        (fit_rows, layout.residual_up + finite, 1.0),
        (fit_rows, layout.residual_down + finite, -1.0),
    ]
    # The energy balance: SoC(t) - SoC(t - 1) - eta * dt * c(t) + dt / eta * d(t) = 0.
    balance = n_fit + times
    inside = times[~np.isin(times, starts)]
    entries += [
        (balance, layout.soc + times, 1.0),
        (n_fit + inside, layout.soc + inside - 1, -1.0),
        (n_fit + starts, layout.soc_start + np.arange(len(starts)), -1.0),
        (balance, layout.charge + times, -one_way_efficiency * HOURS_PER_HALF_HOUR),
        (balance, layout.discharge + times, HOURS_PER_HALF_HOUR / one_way_efficiency),
    ]
    rhs = [aggregate_mw[finite], np.zeros(n_times)]
    n_rows = n_fit + n_times
    if smooth:
        # u(t) - u(t - 1) - step_up(t) + step_down(t) = 0, for t >= 1.
        later = times[1:]
        step_rows = n_rows + later - 1
        entries += [
            (step_rows, layout.discharge + later, 1.0),
            (step_rows, layout.charge + later, -1.0),
            (step_rows, layout.discharge + later - 1, -1.0),
            (step_rows, layout.charge + later - 1, 1.0),
            (step_rows, layout.step_up + later, -1.0),
            (step_rows, layout.step_down + later, 1.0),
        ]
        rhs.append(np.zeros(n_times - 1))
        n_rows += n_times - 1
    rows = np.concatenate([row for row, _, _ in entries])
    cols = np.concatenate([col for _, col, _ in entries])
    vals = np.concatenate(
        [np.broadcast_to(np.asarray(v, dtype=float), np.shape(row)) for row, _, v in entries]
    )
    matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(n_rows, layout.size))
    return matrix, np.concatenate(rhs)


def fit_joint_solar_battery(
    *,
    aggregate_mw: np.ndarray,
    solar_basis: np.ndarray,
    power_mw: float,
    energy_mwh: float,
    one_way_efficiency: float = DEFAULT_ONE_WAY_EFFICIENCY,
    smoothness_penalty: float = 0.0,
    throughput_penalty: float = DEFAULT_THROUGHPUT_PENALTY,
    window_half_hours: int | None = None,
    signed_columns: np.ndarray | None = None,
    signed_column_penalty: float = DEFAULT_SIGNED_COLUMN_PENALTY,
    solar_weight_penalty: float = 0.0,
    method: str = "highs",
) -> JointFit:
    """Fit solar weights and a battery schedule to a half-hourly aggregate.

    Args:
        aggregate_mw: The aggregate output in megawatts at every half-hour; NaN half-hours add no
            residual term, and the battery idles through them.
        solar_basis: The one-megawatt output of each fleet orientation, one row per half-hour and
            one column per orientation; NaN counts as 0.
        power_mw: The battery's power limit in megawatts for charging and for discharging.
        energy_mwh: The battery's energy capacity in megawatt-hours.
        one_way_efficiency: The efficiency of charging and, separately, of discharging.
        smoothness_penalty: The cost per megawatt of half-hour-to-half-hour change in the
            battery's output, against 1 per megawatt of residual.
        throughput_penalty: The cost per megawatt of charging or discharging.
        window_half_hours: The target length of each window, whose starting state of charge is
            free; None for one window.
        signed_columns: Columns whose coefficients are free and signed, one row per half-hour; NaN
            counts as 0. None for no such columns.
        signed_column_penalty: The cost per unit of each signed coefficient's absolute value.
        solar_weight_penalty: The cost per megawatt of fitted solar AC capacity, against 1 per
            megawatt of residual. Zero leaves solar capacity free of cost.
        method: The `scipy.optimize.linprog` method. `highs-ipm` solves a problem with hundreds of
            signed columns far faster than the default `highs`, which picks a simplex method.

    Returns:
        The fit.

    Raises:
        ValueError: If the shapes disagree, the efficiency is outside (0, 1], or a limit is
            negative.
        RuntimeError: If the solver finds no solution.
    """
    n_times, n_basis = solar_basis.shape
    if aggregate_mw.shape != (n_times,):
        raise ValueError(f"aggregate has shape {aggregate_mw.shape}, basis has {n_times} rows")
    if not 0.0 < one_way_efficiency <= 1.0:
        raise ValueError(f"one_way_efficiency must be in (0, 1], got {one_way_efficiency}")
    if power_mw < 0.0 or energy_mwh < 0.0:
        raise ValueError("power_mw and energy_mwh must not be negative")
    signed_dense = np.nan_to_num(
        np.empty((n_times, 0)) if signed_columns is None else signed_columns
    )
    if signed_dense.shape[0] != n_times:
        raise ValueError(f"signed_columns has {signed_dense.shape[0]} rows, basis has {n_times}")
    signed = sparse.csr_matrix(signed_dense)
    n_signed = signed.shape[1]
    basis = np.nan_to_num(solar_basis)
    starts = _window_starts(n_times=n_times, window_half_hours=window_half_hours)
    smooth = smoothness_penalty > 0.0
    layout = _Layout(
        n_basis=n_basis,
        n_times=n_times,
        n_windows=len(starts),
        smooth=smooth,
        n_signed=n_signed,
    )
    equality, rhs = _equality_system(
        layout=layout,
        basis=basis,
        aggregate_mw=aggregate_mw,
        starts=starts,
        one_way_efficiency=one_way_efficiency,
        smooth=smooth,
        signed=signed,
    )
    cost = np.zeros(layout.size)
    cost[layout.discharge : layout.soc] = throughput_penalty
    cost[layout.residual_up : layout.step_up] = 1.0
    cost[layout.signed_up : layout.size] = signed_column_penalty
    cost[layout.weights : layout.weights + n_basis] = solar_weight_penalty
    bounds = np.zeros((layout.size, 2))
    bounds[:, 1] = np.inf
    bounds[layout.discharge : layout.soc, 1] = power_mw
    bounds[layout.soc : layout.residual_up, 1] = energy_mwh
    # Half-hours with no aggregate have no residual identity, so their residual variables stay 0.
    unobserved = np.flatnonzero(~np.isfinite(aggregate_mw))
    bounds[layout.residual_up + unobserved, 1] = 0.0
    bounds[layout.residual_down + unobserved, 1] = 0.0
    if smooth:
        cost[layout.step_up : layout.signed_up] = smoothness_penalty
    result = linprog(c=cost, A_eq=equality, b_eq=rhs, bounds=bounds, method=method)
    if not result.success:
        raise RuntimeError(f"Joint solar and battery fit failed: {result.message}")
    x = result.x
    weights = x[layout.weights : layout.weights + n_basis]
    solar = basis @ weights
    battery = x[layout.discharge : layout.charge] - x[layout.charge : layout.soc]
    coefficients = x[layout.signed_up : layout.signed_down] - x[layout.signed_down : layout.size]
    signed_mw = signed_dense @ coefficients
    return JointFit(
        solar_weights=weights,
        solar_mw=solar,
        battery_mw=battery,
        soc_mwh=x[layout.soc : layout.soc_start],
        residual_mw=aggregate_mw - solar - battery - signed_mw,
        window_starts=starts,
        signed_coefficients=coefficients,
        signed_mw=signed_mw,
    )
