"""Paths and the speed-dependent offset fits shared by the scripts of experiment 00344.

The scripts import experiment 00340's and 00336's modules (which import experiments 00326, 00303,
00285, 00259, and 00230), so run them with
`PYTHONPATH=ml-experiments/00303-wind-adapter-in-production-setting:ml-experiments/00326-wind-research-default-1-change-arms:ml-experiments/00336-wind-research-default-1-below-cut-in-zeroing:ml-experiments/00340-wind-research-default-1-constant-shift-then-low-wind-drop`.
Everything read from other experiments' folders is read only. The key of every arm is the mean
over the ensemble members of the 100 m wind speed at the valid time (forecast-time data only).
"""

from pathlib import Path
from typing import Final

import e00340_common as c340  # isort: skip  # ty: ignore[unresolved-import]
import numpy as np
from scipy.optimize import minimize
from studies.sources import ML_EXPERIMENTS_DATA_DIR

c336 = c340.c336
c303 = c340.c303
EXPERIMENT_ID: Final[int] = 344
SLUG: Final[str] = "wind-research-default-1-speed-dependent-offsets"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
"""Intermediate tables; private and never committed."""
FINAL_DIR: Final[Path] = WORK_DIR / "final"
FARMS: Final[tuple[str, ...]] = c340.FARMS
SEEDS: Final[tuple[int, ...]] = c340.SEEDS
REPLICATE_START = c340.REPLICATE_START
SPEED_EDGES: Final[tuple[float, ...]] = (3.0, 6.0, 9.0, 12.0, 15.0)
"""Bin edges (m/s) of the ensemble-mean 100 m speed that key the x1 offsets: six bins."""
REPORT_EDGES: Final[tuple[float, ...]] = (5.0, 8.0, 15.0)
"""Edges of the reported speed groups: below 5, 5 to 8, 8 to 15, and 15 m/s or more."""
MIN_CELL_ROWS: Final[int] = 200
MIDPOINTS: Final[tuple[float, ...]] = (5.0, 6.5, 8.0, 9.5, 11.0)
"""Candidate midpoints (m/s) of the smooth step of x2 and x3."""
STEP_WIDTH: Final[float] = 6.0
"""Width (m/s) of the smooth step: it rises from 0 to 1 over this range around the midpoint."""
SCALE_GRID: Final[np.ndarray] = np.arange(0.0, 3.0001, 0.05)
LEVELS: Final[np.ndarray] = c303.LEVELS


def smooth_step(*, speed: np.ndarray, midpoint: float) -> np.ndarray:
    """Return the smoothstep (0 to 1) of `speed`, rising over `STEP_WIDTH` around `midpoint`."""
    t = np.clip((speed - (midpoint - STEP_WIDTH / 2)) / STEP_WIDTH, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def pinball(*, residual: np.ndarray, level: float) -> float:
    """Return the mean pinball loss of `residual` (observed minus forecast) at `level`."""
    return float(np.mean(np.maximum(level * residual, (level - 1.0) * residual)))


def speed_bin(*, speed: np.ndarray) -> np.ndarray:
    """Return the x1 speed bin (0 to 5) of each case's ensemble-mean speed."""
    return np.searchsorted(np.array(SPEED_EDGES), speed, side="right")


def cell_ids(*, sites: np.ndarray, lead_day: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """Return an integer id of each case's (farm, lead day, speed bin) cell."""
    farm = np.array([FARMS.index(s) for s in sites])
    return farm * 1000 + lead_day.astype(int) * 10 + speed_bin(speed=speed)


def fit_x1(
    *, quantiles: np.ndarray, actual: np.ndarray, cells: np.ndarray
) -> dict[int, np.ndarray]:
    """Return the 11 offsets of every cell with at least `MIN_CELL_ROWS` rows.

    Each offset is the pinball-optimal one: the level's quantile of (observed minus quantile).
    """
    table = {}
    for cell in np.unique(cells):
        rows = cells == cell
        if rows.sum() < MIN_CELL_ROWS:
            continue
        residual = actual[rows, None] - quantiles[rows].astype(np.float64)
        table[int(cell)] = np.array([np.quantile(residual[:, k], p) for k, p in enumerate(LEVELS)])
    return table


def apply_x1(
    *, quantiles: np.ndarray, cells: np.ndarray, table: dict[int, np.ndarray]
) -> np.ndarray:
    """Return `quantiles` plus the cell offsets, clipped and sorted along the levels."""
    out = quantiles.astype(np.float64).copy()
    for cell, offset in table.items():
        out[cells == cell] += offset
    return np.sort(np.clip(out, 0.0, 1.0), axis=1)


def fit_curve(
    *, residual: np.ndarray, speed: np.ndarray, level: float
) -> tuple[float, float, float]:
    """Return (c_low, c_high, midpoint) of the monotone curve that minimises the pinball loss.

    The correction is c_low + (c_high - c_low) * smooth_step(speed). For each candidate midpoint
    the two levels are fitted by Nelder-Mead from the level's plain quantile, and the midpoint with
    the lowest loss wins.
    """
    start = float(np.quantile(residual, level))
    best = (np.inf, start, start, MIDPOINTS[0])
    for midpoint in MIDPOINTS:
        step = smooth_step(speed=speed, midpoint=midpoint)

        def loss(params: np.ndarray, step: np.ndarray = step) -> float:
            fitted = params[0] + (params[1] - params[0]) * step
            return pinball(residual=residual - fitted, level=level)

        fit = minimize(
            loss, x0=[start, start], method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-9}
        )
        if fit.fun < best[0]:
            best = (float(fit.fun), float(fit.x[0]), float(fit.x[1]), midpoint)
    return best[1], best[2], best[3]


def curve_value(*, params: tuple[float, float, float], speed: np.ndarray) -> np.ndarray:
    """Return the correction of a fitted curve at `speed`."""
    c_low, c_high, midpoint = params
    return c_low + (c_high - c_low) * smooth_step(speed=speed, midpoint=midpoint)


def fit_x2(
    *, quantiles: np.ndarray, actual: np.ndarray, sites: np.ndarray, speed: np.ndarray
) -> dict[str, list[tuple[float, float, float]]]:
    """Return one curve per quantile level for each farm."""
    out = {}
    for farm in FARMS:
        rows = sites == farm
        out[farm] = [
            fit_curve(
                residual=actual[rows] - quantiles[rows, k].astype(np.float64),
                speed=speed[rows],
                level=p,
            )
            for k, p in enumerate(LEVELS)
        ]
    return out


def apply_x2(
    *, quantiles: np.ndarray, sites: np.ndarray, speed: np.ndarray, curves: dict
) -> np.ndarray:
    """Return `quantiles` plus each farm's curves, clipped and sorted along the levels."""
    out = quantiles.astype(np.float64).copy()
    for farm in FARMS:
        rows = sites == farm
        for k, params in enumerate(curves[farm]):
            out[rows, k] += curve_value(params=params, speed=speed[rows])
    return np.sort(np.clip(out, 0.0, 1.0), axis=1)


def fit_x3(
    *, quantiles: np.ndarray, actual: np.ndarray, sites: np.ndarray, speed: np.ndarray
) -> dict:
    """Return the shared shape (one curve per level, fitted on the pooled farms) and farm scales."""
    shape = [
        fit_curve(residual=actual - quantiles[:, k].astype(np.float64), speed=speed, level=p)
        for k, p in enumerate(LEVELS)
    ]
    scales = {}
    for farm in FARMS:
        rows = sites == farm
        base = [curve_value(params=shape[k], speed=speed[rows]) for k in range(len(LEVELS))]
        residual = [
            actual[rows] - quantiles[rows, k].astype(np.float64) for k in range(len(LEVELS))
        ]
        costs = [
            sum(
                pinball(residual=residual[k] - scale * base[k], level=p)
                for k, p in enumerate(LEVELS)
            )
            for scale in SCALE_GRID
        ]
        scales[farm] = float(SCALE_GRID[int(np.argmin(costs))])
    return {"shape": shape, "scales": scales}


def apply_x3(
    *, quantiles: np.ndarray, sites: np.ndarray, speed: np.ndarray, fit: dict
) -> np.ndarray:
    """Return `quantiles` plus scale times shared shape, clipped and sorted along the levels."""
    out = quantiles.astype(np.float64).copy()
    for farm in FARMS:
        rows = sites == farm
        for k, params in enumerate(fit["shape"]):
            out[rows, k] += fit["scales"][farm] * curve_value(params=params, speed=speed[rows])
    return np.sort(np.clip(out, 0.0, 1.0), axis=1)


def shuffled_speed(
    *, speed: np.ndarray, sites: np.ndarray, lead_day: np.ndarray, seed: int
) -> np.ndarray:
    """Return `speed` permuted among the cases of the same farm and lead day (the placebo key)."""
    rng = np.random.default_rng(seed)
    out = speed.copy()
    for farm in FARMS:
        for day in np.unique(lead_day):
            rows = np.flatnonzero((sites == farm) & (lead_day == day))
            out[rows] = speed[rng.permutation(rows)]
    return out


def known_answer_test() -> dict[str, float]:
    """Check that x2 and x3 recover a known speed-dependent bias.

    Synthetic cases of the three farms: ensemble-mean speed uniform on 0 to 20 m/s; observed power
    is a rising curve of the speed plus Gaussian noise of 0.08. The "forecast" quantiles are the
    true quantiles plus a bias of -0.02 below the midpoint and +0.05 above it (a smooth step at
    8 m/s). Each fit must return the opposite correction, +0.02 at low speed and -0.05 at high
    speed, within 20%.
    """
    rng = np.random.default_rng(344)
    size = 60_000
    speed = rng.uniform(0.0, 20.0, size=size)
    sites = np.array(FARMS)[np.arange(size) % len(FARMS)]
    mean = 0.5 * smooth_step(speed=speed, midpoint=9.0)
    actual = mean + rng.normal(0.0, 0.08, size=size)
    from scipy.stats import norm

    true_quantiles = mean[:, None] + 0.08 * norm.ppf(LEVELS)[None, :]
    bias = -0.02 + 0.07 * smooth_step(speed=speed, midpoint=8.0)
    quantiles = true_quantiles + bias[:, None]
    fits = {
        "x2": fit_x2(quantiles=quantiles, actual=actual, sites=sites, speed=speed)[FARMS[0]],
        "x3": fit_x3(quantiles=quantiles, actual=actual, sites=sites, speed=speed)["shape"],
    }
    out = {"injected_low": 0.02, "injected_high": -0.05}
    for name, curves in fits.items():
        low = np.mean([c[0] for c in curves[1:-1]])
        high = np.mean([c[1] for c in curves[1:-1]])
        out[f"{name}_low"] = float(low)
        out[f"{name}_high"] = float(high)
        out[f"{name}_max_relative_error"] = float(
            max(abs(low - 0.02) / 0.02, abs(high + 0.05) / 0.05)
        )
    return out
