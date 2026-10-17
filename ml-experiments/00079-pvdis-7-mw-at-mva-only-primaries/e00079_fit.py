"""The fitting machinery of experiment 00079: the signed fit, the naive fit, and the unfolding fit.

Within one window the model is, as in experiment 00073,

    m = cell + nuisance - c * u,      c >= 0

where the cell is (month, local half-hour, day type), the nuisance terms are heating degree-days and
the cloud response in three hour bands, and `u` is the AC output of 1 MW of the fleet's panels. The
signed fit sees `power = m + error`. The naive fit sees `|power|` and fits it as if it were signed.
The unfolding fit sees `|power|` and fits `|m|`: it alternates between fitting the signed model to
`sign x |power|` and setting each sign to the sign of the fitted `m`.
"""

from __future__ import annotations

from dataclasses import dataclass

import e00079_common as common
import holidays
import numpy as np
import polars as pl


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


def clip_shape(*, g: np.ndarray, ratio: float) -> np.ndarray:
    """Return the AC output per MW of AC capacity: `min(ratio * g, 1)`."""
    return np.minimum(ratio * g, 1.0)


def fleet_shape(*, g_e: np.ndarray, g_s: np.ndarray, g_w: np.ndarray) -> np.ndarray:
    """Return the fleet-mix unit shape: the 0.2, 0.6, 0.2 blend of east, south, and west output."""
    mix = common.FLEET_MIX
    return sum(
        mix[o] * clip_shape(g=g, ratio=common.RATIO)
        for o, g in (("E", g_e), ("S", g_s), ("W", g_w))
    )


@dataclass(frozen=True)
class Design:
    """The rows of a window with their calendar cells and the projection that removes the nuisance.

    Attributes:
        inverse: Index of each row's calendar cell, `0 .. n_cells - 1`.
        counts: Rows per cell.
        basis: Orthonormal basis of the demeaned nuisance columns, one column each.
    """

    inverse: np.ndarray
    counts: np.ndarray
    basis: np.ndarray

    @staticmethod
    def build(*, cells: np.ndarray, nuisance: np.ndarray) -> Design:
        """Demean the nuisance columns within cells and orthonormalise them."""
        _, inverse = np.unique(cells, return_inverse=True)
        counts = np.bincount(inverse)
        design = Design(inverse=inverse, counts=counts, basis=np.zeros((len(cells), 0)))
        x = np.column_stack(
            [design.demean(values=nuisance[:, k]) for k in range(nuisance.shape[1])]
        )
        u, s, _ = np.linalg.svd(x, full_matrices=False)
        keep = s > 1e-9 * max(s.max(), 1e-12)
        return Design(inverse=inverse, counts=counts, basis=u[:, keep])

    def cell_mean(self, *, values: np.ndarray) -> np.ndarray:
        """Return each row's cell mean of `values`."""
        return (np.bincount(self.inverse, weights=values) / self.counts)[self.inverse]

    def demean(self, *, values: np.ndarray) -> np.ndarray:
        """Subtract each cell's mean from its rows."""
        return values - self.cell_mean(values=values)

    def residual(self, *, values: np.ndarray) -> np.ndarray:
        """Return `values` with the cell means and the nuisance terms removed."""
        v = self.demean(values=values)
        return v - self.basis @ (self.basis.T @ v)


def signed_fit(*, design: Design, y: np.ndarray, u: np.ndarray) -> tuple[float, np.ndarray]:
    """Return the capacity `c >= 0` and the fitted mean `m` of the model fitted to signed `y`."""
    mu = design.residual(values=-u)
    my = design.residual(values=y)
    c = max(float(mu @ my) / float(mu @ mu), 0.0)
    error = my - c * mu
    return c, y - error


def abs_loss(*, a: np.ndarray, m: np.ndarray) -> float:
    """Return the sum of squared differences between the reading and the magnitude of the fit."""
    return float(((a - np.abs(m)) ** 2).sum())


def naive_fit(*, design: Design, a: np.ndarray, u: np.ndarray) -> tuple[float, np.ndarray]:
    """Return the capacity and fitted mean of the magnitude fitted as if it were signed."""
    return signed_fit(design=design, y=a, u=u)


@dataclass(frozen=True)
class Unfolded:
    """The result of the unfolding fit.

    Attributes:
        capacity: Fitted capacity, MW.
        mean: Fitted signed mean `m` per row.
        loss: Sum of squared differences between the reading and `|m|`.
        n_starts_distinct: Number of distinct capacities the starts reach.
        loss_gap: Loss of the second-best start minus that of the best start.
    """

    capacity: float
    mean: np.ndarray
    loss: float
    n_starts_distinct: int
    loss_gap: float


def _run(
    *, design: Design, a: np.ndarray, u: np.ndarray, sign: np.ndarray
) -> tuple[float, np.ndarray, float]:
    """Alternate the signed fit and the sign update from `sign` until the signs stop changing."""
    best: tuple[float, np.ndarray, float] | None = None
    for _ in range(common.MAX_ITERATIONS):
        c, m = signed_fit(design=design, y=sign * a, u=u)
        loss = abs_loss(a=a, m=m)
        if best is None or loss < best[2]:
            best = (c, m, loss)
        new = np.where(m >= 0.0, 1.0, -1.0)
        if np.array_equal(new, sign):
            break
        sign = new
    assert best is not None
    return best


def start_signs(*, design: Design, a: np.ndarray, u: np.ndarray, scale: float) -> list[np.ndarray]:
    """Return the starting sign patterns: all positive, and one per capacity multiple of `scale`."""
    cell = design.cell_mean(values=a)
    out = [np.ones_like(a)]
    out.extend(np.where(cell - k * scale * u >= 0.0, 1.0, -1.0) for k in common.START_MULTIPLES)
    return out


def unfold_fit(*, design: Design, a: np.ndarray, u: np.ndarray, scale: float) -> Unfolded:
    """Fit `|m|` to the reading `a`, from several starting sign patterns, keeping the best.

    Args:
        design: The window's design.
        a: The magnitude readings, MW.
        u: The unit solar shape.
        scale: The window's median absolute midday power, MW, which sizes the starts.

    Returns:
        The start with the lowest loss, unless it lowers the all-positive start's loss by less than
        `TIE_MARGIN`, in which case the all-positive start.
    """
    results = [
        _run(design=design, a=a, u=u, sign=s)
        for s in start_signs(design=design, a=a, u=u, scale=scale)
    ]
    positive = results[0]
    best = min(results, key=lambda r: r[2])
    if best[2] > (1.0 - common.TIE_MARGIN) * positive[2]:
        best = positive
    losses = sorted(r[2] for r in results)
    distinct = len({round(r[0], 6) for r in results})
    return Unfolded(
        capacity=best[0],
        mean=best[1],
        loss=best[2],
        n_starts_distinct=distinct,
        loss_gap=losses[1] - losses[0],
    )


def balanced_accuracy(*, true_sign: np.ndarray, fitted_mean: np.ndarray, floor: float) -> float:
    """Return the balanced accuracy of the fitted flow direction, or NaN if a class is too small.

    Rows with `|true flow| < floor` are dropped. The score is the mean of the share of
    reverse-flow rows whose fitted mean is negative and the share of forward-flow rows whose fitted
    mean is non-negative.
    """
    keep = np.abs(true_sign) >= floor
    truth = true_sign[keep] < 0.0
    predicted = fitted_mean[keep] < 0.0
    if truth.sum() < common.MIN_CLASS_ROWS or (~truth).sum() < common.MIN_CLASS_ROWS:
        return float("nan")
    return float(0.5 * (predicted[truth].mean() + (~predicted[~truth]).mean()))
