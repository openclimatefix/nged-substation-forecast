"""One substation's convex design, its monthly Gram matrices, and the fits that subset its columns.

**Every fit of experiment 113 is a column subset of one design.** The design is `studies.dp_gam`'s
layout with fitted capacity and the D1 drift. A month-block bootstrap resample is a vector of
month counts, so a resample's Gram matrix is the count-weighted sum of per-month Gram matrices,
and so is its right-hand side. One pass over the rows therefore serves every resample, every
target, and every arm.
"""

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Final

import e00113_common as common
import numpy as np
import torch
from studies.dp_calendar import SMOOTHING_HOURS
from studies.dp_features import FeatureCache
from studies.dp_gam import (
    N_BANDS,
    GamLayout,
    GamSpec,
    TrainingWindow,
    design_matrix,
    make_layout,
    penalty_matrix,
    solve_bounded_quadratic,
    training_rows,
)

N_FUNCTIONALS: Final[int] = N_BANDS + 1
"""The scored functionals: each band's total heating slope, then the latest capacity."""
CAPACITY_FUNCTIONAL: Final[int] = N_BANDS
FUNCTIONAL_NAMES: Final[tuple[str, ...]] = ("night", "morning", "midday", "evening", "capacity")
MIN_SEASON_ROWS: Final[int] = 1000
"""A season-year with fewer training rows gets no capacity increment (the cut can leave 2 rows)."""
WINDOW_MONTHS: Final[dict[str, int | None]] = {"all": None, "2y": 24}


@dataclass(frozen=True)
class Arm:
    """One fit: which terms it drops, its drift penalty, and whether it pins the capacity.

    Attributes:
        name: The arm's name.
        drop: The terms of the full layout that the fit leaves out.
        drift_step: The weight of the drift's second-difference penalty.
        pin: The share of the truth's capacity increments that the fit takes as known (1.0 is the
            truth), or `None` to fit the capacity.
        bounded: False to solve without the non-negativity bounds.
    """

    name: str
    drop: tuple[str, ...] = ()
    drift_step: float = common.DRIFT_STEP_PART_2
    pin: float | None = None
    bounded: bool = True


ARMS: Final[dict[str, Arm]] = {
    arm.name: arm
    for arm in (
        Arm("g_full"),
        Arm("g_full_30", drift_step=common.DRIFT_STEP_PART_4),
        Arm("g_full_pin", pin=1.0),
        Arm("g_full_pin80", pin=0.8),
        Arm("g_full_unbounded", bounded=False),
        Arm("g_0", drop=("harmonics", "drift")),
        Arm("g_0_pin", drop=("harmonics", "drift"), pin=1.0),
        Arm("g_h", drop=("drift",)),
        Arm("g_d", drop=("harmonics",)),
    )
}


@dataclass
class Design:
    """A substation's design at one cut and window.

    Attributes:
        site: The substation's label.
        cut: The fit's cut.
        window: `all` for all years before the cut, or `2y` for the last 24 months.
        layout: The full layout (fitted capacity, D1 drift).
        index: The training rows (grid positions).
        x: The design matrix, shape (n, p).
        months: The calendar months the rows span, ascending.
        month_of_row: For each row, its position in `months`.
        quarter_of_month: The calendar quarter (0 to 3) of each month.
        month_gram: `X_m^T X_m` for each month, shape (M, p, p), float64.
        penalties: The penalty matrix for each drift step.
        functionals: The (p, 5) matrix whose columns are the scored linear functionals.
    """

    site: str
    cut: datetime
    window: str
    layout: GamLayout
    index: np.ndarray
    x: np.ndarray
    months: np.ndarray
    month_of_row: np.ndarray
    quarter_of_month: np.ndarray
    month_gram: torch.Tensor
    penalties: dict[float, np.ndarray]
    functionals: np.ndarray

    @property
    def n(self) -> int:
        """Return the number of training rows."""
        return len(self.index)

    @property
    def p(self) -> int:
        """Return the number of columns."""
        return self.layout.n_columns


def full_spec(*, drift_step: float = common.DRIFT_STEP_PART_2) -> GamSpec:
    """Return the spec of the full layout: D1 drift and fitted capacity."""
    return GamSpec(ridge=common.RIDGE, drift="d1", drift_step=drift_step, capacity="fitted")


def build_design(*, cache: FeatureCache, site: str, cut: datetime, window: str) -> Design:
    """Build the design, the monthly Gram matrices, and the penalties of one substation and window.

    Args:
        cache: The feature cache.
        site: The substation's label.
        cut: The cut; no training row ends after it.
        window: A key of `WINDOW_MONTHS`.

    Returns:
        The design.
    """
    features = cache.sites[site]
    months_back = WINDOW_MONTHS[window]
    rule = TrainingWindow(months=months_back) if months_back else TrainingWindow()
    index = training_rows(features=features, cache=cache, cut=cut, window=rule)
    seasons, season_rows = np.unique(cache.calendar.season_year[index], return_counts=True)
    season_years = [
        int(s) for s, rows in zip(seasons, season_rows, strict=True) if rows >= MIN_SEASON_ROWS
    ]
    layout = make_layout(
        spec=full_spec(),
        season_years=season_years,
        cut=cut,
        earliest=cache.end_times[index[0]],
    )
    x = design_matrix(cache=cache, features=features, layout=layout, index=index)
    month_keys = cache.end_times[index].astype("datetime64[M]")
    months, month_of_row = np.unique(month_keys, return_inverse=True)
    quarter = ((months.astype(np.int64) % 12) // 3).astype(np.int64)
    p = layout.n_columns
    grams = torch.zeros((len(months), p, p), dtype=torch.float64)
    for position in range(len(months)):
        block = torch.as_tensor(x[month_of_row == position])
        grams[position] = block.T @ block
    penalties = {
        step: penalty_matrix(
            layout=replace(layout, spec=full_spec(drift_step=step)), n_rows=len(index)
        )[0]
        for step in (common.DRIFT_STEP_PART_2, common.DRIFT_STEP_PART_4)
    }
    functionals = np.zeros((p, N_FUNCTIONALS))
    heating = layout.slices["heating"].start
    for band in range(N_BANDS):
        for smoothing in range(len(SMOOTHING_HOURS)):
            functionals[heating + smoothing * N_BANDS + band, band] = 1.0
    functionals[layout.slices["capacity"], CAPACITY_FUNCTIONAL] = 1.0
    return Design(
        site=site,
        cut=cut,
        window=window,
        layout=layout,
        index=index,
        x=x,
        months=months,
        month_of_row=month_of_row,
        quarter_of_month=quarter,
        month_gram=grams,
        penalties=penalties,
        functionals=functionals,
    )


def stratified_counts(*, design: Design, n_resamples: int, rng: np.random.Generator) -> np.ndarray:
    """Return month counts of a month-block bootstrap that keeps every calendar quarter's size.

    Each resample draws, with replacement and within each calendar quarter, as many of the
    quarter's months as the quarter holds.

    Args:
        design: The design, for its months and their quarters.
        n_resamples: The number of resamples.
        rng: The random generator.

    Returns:
        Counts of shape (n_resamples, number of months).
    """
    counts = np.zeros((n_resamples, len(design.months)), dtype=np.float64)
    for quarter in range(4):
        members = np.flatnonzero(design.quarter_of_month == quarter)
        if len(members) == 0:
            continue
        draws = rng.integers(0, len(members), size=(n_resamples, len(members)))
        for resample in range(n_resamples):
            counts[resample, members] = np.bincount(draws[resample], minlength=len(members))
    return counts


def unstratified_counts(
    *, design: Design, n_resamples: int, rng: np.random.Generator
) -> np.ndarray:
    """Return month counts of `studies.dp_gam.month_block_weights`' scheme: months drawn freely."""
    n_months = len(design.months)
    draws = rng.integers(0, n_months, size=(n_resamples, n_months))
    return np.stack([np.bincount(row, minlength=n_months) for row in draws]).astype(np.float64)


def weighting_set(*, design: Design, rng: np.random.Generator) -> np.ndarray:
    """Return the 1 + 2 x `N_BOOTSTRAP` month-count rows: the full sample, I1's, then I1u's."""
    full = np.ones((1, len(design.months)))
    return np.concatenate(
        [
            full,
            stratified_counts(design=design, n_resamples=common.N_BOOTSTRAP, rng=rng),
            unstratified_counts(design=design, n_resamples=common.N_BOOTSTRAP, rng=rng),
        ]
    )


def weighted_gram(*, design: Design, counts: np.ndarray) -> torch.Tensor:
    """Return `X^T W X` for each row of month counts, shape (B, p, p)."""
    return torch.einsum("bm,mpq->bpq", torch.as_tensor(counts), design.month_gram)


def month_cross_products(*, design: Design, target: np.ndarray) -> torch.Tensor:
    """Return `X_m^T y_m` for each month, shape (M, p)."""
    out = torch.zeros((len(design.months), design.p), dtype=torch.float64)
    for position in range(len(design.months)):
        rows = design.month_of_row == position
        out[position] = torch.as_tensor(design.x[rows].T @ target[rows])
    return out


def weighted_rhs(*, counts: np.ndarray, month_rhs: torch.Tensor) -> torch.Tensor:
    """Return `X^T W y` for each row of month counts, shape (B, p)."""
    return torch.as_tensor(counts) @ month_rhs


def selection(*, design: Design, arm: Arm) -> np.ndarray:
    """Return the boolean mask of the columns an arm fits."""
    keep = np.ones(design.p, dtype=bool)
    for term in arm.drop:
        keep[design.layout.slices[term]] = False
    if arm.pin is not None:
        keep[design.layout.slices["capacity"]] = False
    return keep


def solve_arm(
    *,
    design: Design,
    arm: Arm,
    gram: torch.Tensor,
    rhs: torch.Tensor,
    pinned_increments: np.ndarray | None,
) -> np.ndarray:
    """Solve an arm for every weighting of a batch.

    A pinned arm drops the capacity columns and subtracts the Gram block times the pinned
    increments from the right-hand side, which equals adding the pinned PV to net power (the
    capacity columns carry a minus sign).

    Args:
        design: The design.
        arm: The arm.
        gram: `X^T W X`, shape (B, p, p).
        rhs: `X^T W y`, shape (B, p).
        pinned_increments: The truth's capacity increments, for a pinned arm.

    Returns:
        The coefficients, shape (B, p), zero in the columns the arm leaves out and, for a pinned
        arm, holding the pinned increments in the capacity columns.
    """
    keep = selection(design=design, arm=arm)
    chosen = torch.as_tensor(np.flatnonzero(keep))
    penalty = torch.as_tensor(design.penalties[arm.drift_step])
    matrix = gram[:, chosen][:, :, chosen] + penalty[chosen][:, chosen]
    right = rhs[:, chosen]
    capacity = design.layout.slices["capacity"]
    pinned_theta = np.zeros(design.p)
    if arm.pin is not None:
        if pinned_increments is None:
            msg = "a pinned arm needs the pinned increments"
            raise ValueError(msg)
        pinned_theta[capacity] = arm.pin * pinned_increments
        shift = gram[:, chosen][
            :, :, torch.arange(capacity.start, capacity.stop)
        ] @ torch.as_tensor(pinned_theta[capacity])
        right = right - shift
    constrained = torch.as_tensor(design.layout.constrained[keep] & arm.bounded)
    solution, _ = solve_bounded_quadratic(matrix=matrix, rhs=right, constrained=constrained)
    theta = np.tile(pinned_theta, (gram.shape[0], 1))
    theta[:, np.flatnonzero(keep)] = solution.numpy()
    return theta


def restrict_theta(*, theta: np.ndarray, full: Design, sub: Design) -> np.ndarray:
    """Map coefficients of the all-years layout onto the layout of a shorter window.

    The two layouts count their drift knots back from the same cut, so the sub-window's knots are
    the last knots of the full layout. The sub-window's first capacity increment is the capacity of
    its first season-year, and later increments are differences.

    Args:
        theta: Coefficients of `full`'s layout.
        full: The all-years design.
        sub: The shorter window's design.

    Returns:
        Coefficients of `sub`'s layout.

    Raises:
        ValueError: If the layouts' last knots differ.
    """
    out = np.zeros(sub.p)
    for term in ("profile", "holiday", "tent", "harmonics", "heating", "lighting"):
        out[sub.layout.slices[term]] = theta[full.layout.slices[term]]
    full_capacity = np.cumsum(theta[full.layout.slices["capacity"]])
    by_year = dict(zip(full.layout.season_years, full_capacity, strict=True))
    levels = np.array([by_year[year] for year in sub.layout.season_years])
    out[sub.layout.slices["capacity"]] = np.diff(levels, prepend=0.0)
    n_full = len(full.layout.knot_days)
    n_sub = len(sub.layout.knot_days)
    if not np.allclose(full.layout.knot_days[-n_sub:], sub.layout.knot_days):
        msg = "the layouts' drift knots do not line up"
        raise ValueError(msg)
    for band in range(N_BANDS):
        source = full.layout.slices["drift"].start + band * n_full
        target = sub.layout.slices["drift"].start + band * n_sub
        out[target : target + n_sub] = theta[source + n_full - n_sub : source + n_full]
    return out


def analytic_bias(*, design: Design, arm: Arm, theta_truth: np.ndarray) -> np.ndarray:
    """Return `-v^T (G + P)^-1 P theta_truth` for each functional, for an unbounded fit of `arm`.

    The bias is the deterministic shrinkage of the noise-free penalised fit, for the arm's columns
    only, with the full sample's Gram matrix. A column the arm leaves out is treated as absent from
    the truth; for a misspecified arm the result is not the fit's bias.

    Args:
        design: The design.
        arm: The arm.
        theta_truth: The truth's coefficients.

    Returns:
        The bias of each functional, shape (5,).
    """
    keep = selection(design=design, arm=arm)
    chosen = np.flatnonzero(keep)
    gram = design.month_gram.sum(dim=0).numpy()
    penalty = design.penalties[arm.drift_step]
    matrix = gram[np.ix_(chosen, chosen)] + penalty[np.ix_(chosen, chosen)]
    shrink = np.linalg.solve(matrix, (penalty[np.ix_(chosen, chosen)] @ theta_truth[chosen]))
    return -(design.functionals[chosen].T @ shrink)


def functional_values(*, design: Design, theta: np.ndarray) -> np.ndarray:
    """Return the scored functionals of coefficient rows, shape (..., 5)."""
    return theta @ design.functionals


def rows_of(*, big: Design, small: Design) -> np.ndarray:
    """Return, for each training row of `small`, its position among `big`'s rows."""
    positions = np.searchsorted(big.index, small.index)
    if not np.array_equal(big.index[positions], small.index):
        msg = "the smaller window's rows are not a subset of the larger window's"
        raise ValueError(msg)
    return positions


def band_supporting_months(*, design: Design, functional: int) -> np.ndarray:
    """Return the months that carry a functional, as a boolean mask over `design.months`.

    A month carries a functional if the column `X v` has at least a quarter of the largest
    month's absolute sum there. For the latest capacity, only months of the latest season-year
    count.
    """
    column = design.x @ design.functionals[:, functional]
    sums = np.bincount(design.month_of_row, weights=np.abs(column), minlength=len(design.months))
    mask = sums >= 0.25 * sums.max()
    if functional == CAPACITY_FUNCTIONAL:
        last = max(design.layout.season_years)
        months_int = design.months.astype(np.int64)
        starts = (last - 1970) * 12 + 6
        mask &= (months_int >= starts) & (months_int < starts + 12)
    return mask


def summed_gram(*, design: Design) -> torch.Tensor:
    """Return the full-sample `X^T X`, shape (p, p)."""
    return design.month_gram.sum(dim=0)
