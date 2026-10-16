"""The fitting machinery of experiment 00073: designs, orientation arms, and the ratio search.

Every fit is least squares within calendar cells, as in experiments 00039 and 00040:

    power = cell + nuisance - sum_o c_o * u_o + error

where the cell is (month, local half-hour, day type), the nuisance terms are heating degree-days and
the cloud response in three hour bands (experiment 00040's arm M2h), and `u_o` is the AC output of
1 MW of panels facing orientation `o`. The nuisance terms are free in sign, so they are projected
out first (Frisch-Waugh) and the capacities `c_o >= 0` are fitted to what is left.
"""

from __future__ import annotations

from dataclasses import dataclass

import e00073_common as common
import holidays
import numpy as np
import polars as pl
from scipy.optimize import nnls


def bank_holidays() -> pl.Series:
    """Return the English bank holidays of 2018 to 2027 as a date series."""
    return pl.Series(
        sorted(holidays.country_holidays("GB", subdiv="ENG", years=range(2018, 2028)))
    ).cast(pl.Date)


def prepare(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Keep daylight half-hours and add the calendar, temperature, and cloud columns."""
    return (
        frame.filter(pl.col("zenith") < common.MAX_ZENITH_DEG)
        .with_columns(
            local=pl.col("time").dt.offset_by("-15m").dt.convert_time_zone("Europe/London")
        )
        .with_columns(
            half_hour=(
                pl.col("local").dt.hour().cast(pl.Int64) * 2
                + pl.col("local").dt.minute().cast(pl.Int64) // 30
            ),
            off=(
                (pl.col("local").dt.weekday() >= 6)
                | pl.col("local").dt.date().is_in(bank_holidays().implode())
            ).cast(pl.Int64),
            month=pl.col("local").dt.month().cast(pl.Int64),
            date=pl.col("local").dt.date(),
            season_year=(
                pl.col("local").dt.year() - (pl.col("local").dt.month() < 9).cast(pl.Int32)
            ),
            hdd=(common.HDD_BASE_C - pl.col("temperature_2m")).clip(lower_bound=0.0),
            cloud=1.0
            - (pl.col("ghi_u") / pl.col("ghi_u_clear")).clip(upper_bound=1.0).fill_nan(0.0),
        )
        .with_columns(day_cloud=pl.col("cloud").mean().over("site", "date"))
    )


def demean(*, values: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Subtract each cell's mean from its rows (works column-wise on a 2-D array)."""
    _, inverse = np.unique(cells, return_inverse=True)
    counts = np.bincount(inverse)
    if values.ndim == 1:
        return values - (np.bincount(inverse, weights=values) / counts)[inverse]
    out = np.empty_like(values, dtype=float)
    for k in range(values.shape[1]):
        out[:, k] = demean(values=values[:, k], cells=cells)
    return out


def clip_shape(*, g: np.ndarray, ratio: float) -> np.ndarray:
    """Return the AC output per MW of AC capacity: `min(ratio * g, 1)`."""
    return np.minimum(ratio * g, 1.0)


@dataclass(frozen=True)
class Design:
    """A set of rows with its calendar cells, and the projection that removes the nuisance terms.

    Attributes:
        cells: Integer code of (month, half-hour, day type) per row.
        basis: Orthonormal basis of the demeaned nuisance columns, one column each.
    """

    cells: np.ndarray
    basis: np.ndarray

    @staticmethod
    def build(*, cells: np.ndarray, nuisance: np.ndarray | None) -> Design:
        """Demean the nuisance columns within cells and orthonormalise them."""
        if nuisance is None or nuisance.shape[1] == 0:
            return Design(cells=cells, basis=np.zeros((len(cells), 0)))
        x = demean(values=nuisance, cells=cells)
        u, s, _ = np.linalg.svd(x, full_matrices=False)
        keep = s > 1e-9 * max(s.max(), 1e-12)
        return Design(cells=cells, basis=u[:, keep])

    def residual(self, *, values: np.ndarray) -> np.ndarray:
        """Return `values` with the cell means and the nuisance terms removed."""
        v = demean(values=values, cells=self.cells)
        return v - self.basis @ (self.basis.T @ v)


def fit_orientation(*, design: Design, y: np.ndarray, shapes: np.ndarray) -> np.ndarray:
    """Return the non-negative capacities `c` (one per column of `shapes`), MW."""
    c, _ = nnls(design.residual(values=-shapes), design.residual(values=y))
    return c


def fit_pooled_detail(
    *,
    designs: list[Design],
    ys: list[np.ndarray],
    shapes: list[np.ndarray],
    ridge: float,
) -> tuple[np.ndarray, list[float]]:
    """Return a Primary's shared orientation shares `m` and each year's total capacity `s_y`, MW.

    Each year's total capacity `s_y` comes from a south-only fit. The stack fits shares `m >= 0` to
    `y_y = ... - s_y * (u_y @ m)`, with a ridge penalty pulling `m` to the fleet mix. The returned
    capacity of year `y` in orientation `o` is `s_y * m_o`.
    """
    prior = np.array(common.FLEET_MIX)
    scale, blocks, targets = [], [], []
    for design, y, u in zip(designs, ys, shapes, strict=True):
        my = design.residual(values=y)
        mu = design.residual(values=-u)
        south, _ = nnls(mu[:, [1]], my)
        s = max(float(south[0]), 0.1)
        scale.append(s)
        blocks.append(s * mu)
        targets.append(my)
    a = np.vstack(blocks)
    b = np.concatenate(targets)
    if ridge > 0.0:
        sigma2 = float((a**2).sum() / 3.0)
        root = np.sqrt(ridge * sigma2)
        a = np.vstack([a, root * np.eye(3)])
        b = np.concatenate([b, root * prior])
    m, _ = nnls(a, b)
    return m, scale


def fit_pooled(
    *,
    designs: list[Design],
    ys: list[np.ndarray],
    shapes: list[np.ndarray],
    ridge: float,
) -> np.ndarray:
    """Return the pooled fit as capacities, MW: the mean of the years' `s_y` times `m`."""
    m, scale = fit_pooled_detail(designs=designs, ys=ys, shapes=shapes, ridge=ridge)
    return float(np.mean(scale)) * m


def tv_error(*, increment: np.ndarray, truth: np.ndarray) -> tuple[float, bool]:
    """Return the total-variation error of the shares in an increment, and whether a rule bound.

    Negative increments are set to zero. A total under `MIN_INCREMENT_MW` counts as an error of
    `NO_INCREMENT_ERROR`.
    """
    positive = np.clip(increment, 0.0, None)
    bound = bool((increment < -1e-6 * max(positive.sum(), 1.0)).any())
    if positive.sum() < common.MIN_INCREMENT_MW:
        return common.NO_INCREMENT_ERROR, True
    return float(0.5 * np.abs(positive / positive.sum() - truth).sum()), bound


def ratio_profile(
    *, design: Design | None, y: np.ndarray, g: np.ndarray, grid: tuple[float, ...]
) -> np.ndarray:
    """Return the residual sum of squares of the best capacity at each ratio on the grid.

    With no design the fit is a regression through the origin (a solar farm alone).
    """
    out = []
    for ratio in grid:
        u = clip_shape(g=g, ratio=ratio)
        if design is None:
            c = float(u @ y) / float(u @ u)
            out.append(float(((y - c * u) ** 2).sum()))
            continue
        my, mu = design.residual(values=y), design.residual(values=-u)
        c = max(float(mu @ my) / float(mu @ mu), 0.0)
        out.append(float(((my - c * mu) ** 2).sum()))
    return np.array(out)


def best_ratio(*, profile: np.ndarray, grid: tuple[float, ...]) -> float:
    """Return the ratio with the lowest residual sum of squares."""
    return float(grid[int(np.argmin(profile))])
