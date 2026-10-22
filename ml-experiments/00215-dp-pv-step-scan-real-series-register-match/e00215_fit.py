"""The step path of experiment 207, copied, with its Gram, penalty, and level helpers.

Experiment 215 reuses the month design, the step parametrisation, and the total-variation penalty of
experiment 207 unchanged. The injection and read-out code is not copied.
"""

from dataclasses import dataclass
from typing import Final

import e00215_common as common
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.dp_features import FeatureCache
from studies.dp_gam import (
    DEFAULT_MAX_ITERATIONS,
    GamLayout,
    design_matrix,
    penalty_matrix,
    row_chunks,
    solve_bounded_quadratic,
)

LEVEL_RIDGE_SHARE: Final[float] = 1e-6
"""The path's ridge on the level itself, as a share of the growth penalty, so that a month with no
training row (every month outside the summer) stays well posed."""


@dataclass(frozen=True)
class MonthDesign:
    """The monthly capacity columns of one Primary's training rows.

    Attributes:
        first_month: The first month as months since 1970-01.
        n_months: The number of months from the first to the last month with a training row.
        season_year_of_month: The season-year (July to June, named by the July's year) of each.
    """

    first_month: int
    n_months: int
    season_year_of_month: np.ndarray


def row_months(*, cache: FeatureCache, rows: np.ndarray) -> np.ndarray:
    """Return each row's calendar month, as months since 1970-01, from the half-hour's start."""
    start = cache.end_times[rows] - np.timedelta64(30, "m")
    return start.astype("datetime64[M]").astype(np.int64)


def summer_rows(*, cache: FeatureCache, index: np.ndarray) -> np.ndarray:
    """Keep the rows whose calendar month is in `SUMMER_MONTHS`."""
    months = row_months(cache=cache, rows=index) % 12 + 1
    return index[np.isin(months, common.SUMMER_MONTHS)]


def month_design(*, cache: FeatureCache, index: np.ndarray) -> MonthDesign:
    """Return the span of months the training rows cover, including months with no row."""
    months = row_months(cache=cache, rows=index)
    first, last = int(months.min()), int(months.max())
    all_months = np.arange(first, last + 1)
    calendar_year = 1970 + all_months // 12
    calendar_month = all_months % 12 + 1
    season = np.where(calendar_month >= 7, calendar_year, calendar_year - 1)
    return MonthDesign(first_month=first, n_months=last - first + 1, season_year_of_month=season)


def month_columns(
    *, cache: FeatureCache, rows: np.ndarray, design: MonthDesign, shape: np.ndarray
) -> np.ndarray:
    """Return the minus-signed month columns: `-shape` in the column of the row's month."""
    columns = np.zeros((len(rows), design.n_months))
    columns[np.arange(len(rows)), row_months(cache=cache, rows=rows) - design.first_month] = -shape
    return columns


def accumulate_month_gram(
    *,
    cache: FeatureCache,
    site: str,
    layout: GamLayout,
    index: np.ndarray,
    design: MonthDesign,
    targets: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum `X^T X` and `X^T y` for the month design over the rows `index`.

    Args:
        cache: The cache.
        site: The substation's label.
        layout: The season-year layout, whose capacity columns the month columns replace.
        index: The training rows.
        design: The months.
        targets: The targets, shape (n_targets, len(index)).

    Returns:
        `(gram, rhs)` of shapes (p', p') and (n_targets, p') in float64.

    Raises:
        ValueError: If a training row precedes the first season-year with a capacity increment.
    """
    features = cache.sites[site]
    cap = layout.slices["capacity"]
    p_new = layout.n_columns - (cap.stop - cap.start) + design.n_months
    gram = torch.zeros((p_new, p_new), dtype=torch.float64)
    rhs = torch.zeros((targets.shape[0], p_new), dtype=torch.float64)
    target_tensor = torch.as_tensor(targets, dtype=torch.float64)
    for chunk in row_chunks(index=index):
        rows = index[chunk]
        x = design_matrix(cache=cache, features=features, layout=layout, index=rows)
        years = np.array(layout.season_years)
        season = cache.calendar.season_year[rows]
        shape = -x[:, cap][:, 0] if season.min() >= years[0] else None
        if shape is None:
            msg = "a training row precedes the first season-year with a capacity increment"
            raise ValueError(msg)
        x_new = np.concatenate(
            [
                x[:, : cap.start],
                month_columns(cache=cache, rows=rows, design=design, shape=shape),
                x[:, cap.stop :],
            ],
            axis=1,
        )
        x_tensor = torch.as_tensor(x_new, dtype=torch.float64)
        gram += x_tensor.T @ x_tensor
        rhs += target_tensor[:, chunk] @ x_tensor
    return gram, rhs


def step_transform(*, layout: GamLayout, design: MonthDesign) -> torch.Tensor:
    """Return `T` with `X_step = X_month @ T` for the step parametrisation of the monthly path.

    The capacity block of the step design is `[l0, u_1 .. u_(M-1), d_1 .. d_(M-1)]`: the level of
    the first month, then an up-step and a down-step at each month boundary, all non-negative.
    The level of month `m` is `l0 + sum_(k<=m) (u_k - d_k)`.
    """
    cap = layout.slices["capacity"]
    n_months = design.n_months
    p_month = layout.n_columns - (cap.stop - cap.start) + n_months
    n_steps = 1 + 2 * (n_months - 1)
    p_step = p_month - n_months + n_steps
    transform = torch.zeros((p_month, p_step), dtype=torch.float64)
    before = np.arange(cap.start)
    transform[before, before] = 1.0
    block = cap.start
    transform[block : block + n_months, block] = 1.0
    for k in range(1, n_months):
        transform[block + k : block + n_months, block + k] = 1.0
        transform[block + k : block + n_months, block + n_months - 1 + k] = -1.0
    after_month = np.arange(cap.start + n_months, p_month)
    after_step = np.arange(block + n_steps, p_step)
    transform[after_month, after_step] = 1.0
    return transform


def solve(
    *,
    gram: torch.Tensor,
    rhs: torch.Tensor,
    penalty: np.ndarray,
    cost: np.ndarray,
    constrained: np.ndarray,
) -> np.ndarray:
    """Solve the penalised, bounded least squares for each right-hand side; return theta."""
    batch = rhs.shape[0]
    matrix = gram[None] + torch.as_tensor(penalty, dtype=torch.float64)[None]
    theta, _ = solve_bounded_quadratic(
        matrix=matrix.expand(batch, -1, -1),
        rhs=rhs - 0.5 * torch.as_tensor(cost, dtype=torch.float64)[None],
        constrained=torch.as_tensor(constrained),
        max_iterations=DEFAULT_MAX_ITERATIONS,
    )
    return theta.numpy()


def step_penalty(
    *,
    layout: GamLayout,
    design: MonthDesign,
    n_rows: int,
    gram: torch.Tensor,
    share: float,
    transform: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the total-variation path's penalty matrix, linear cost, and bound mask.

    The linear cost on every up-step and down-step is `share` times the mean positive diagonal of
    the month Gram matrix, per point of step. The first-month level has the small ridge only.
    """
    cap = layout.slices["capacity"]
    n_years = cap.stop - cap.start
    old_matrix, old_cost = penalty_matrix(layout=layout, n_rows=n_rows)
    p_month = layout.n_columns - n_years + design.n_months
    p_step = transform.shape[1]
    n_steps = 1 + 2 * (design.n_months - 1)
    keep_old = np.r_[0 : cap.start, cap.stop : layout.n_columns]
    keep_new = np.r_[0 : cap.start, cap.start + n_steps : p_step]
    matrix = np.zeros((p_step, p_step))
    matrix[np.ix_(keep_new, keep_new)] = old_matrix[np.ix_(keep_old, keep_old)]
    cost = np.zeros(p_step)
    cost[keep_new] = old_cost[keep_old]
    months = slice(cap.start, cap.start + design.n_months)
    diagonal = torch.diagonal(gram)[months].numpy()
    unit = float(diagonal[diagonal > 0].mean())
    block = slice(cap.start, cap.start + n_steps)
    matrix[block, block] = LEVEL_RIDGE_SHARE * unit * np.eye(n_steps)
    cost[cap.start + 1 : cap.start + n_steps] = share * unit
    constrained = np.zeros(p_step, dtype=bool)
    constrained[keep_new] = layout.constrained[keep_old]
    constrained[block] = True
    del p_month
    return matrix, cost, constrained


def levels_from(
    *, arm: str, theta: np.ndarray, layout: GamLayout, design: MonthDesign
) -> np.ndarray:
    """Return the monthly levels (points) of each target, shape (n_targets, n_months)."""
    cap = layout.slices["capacity"]
    n_months = design.n_months
    if arm == "two_stage":
        years = list(layout.season_years)
        by_year = np.cumsum(theta[:, cap], axis=1)
        position = np.array([years.index(int(s)) for s in design.season_year_of_month])
        return by_year[:, position]
    if arm == "smooth":
        return theta[:, cap.start : cap.start + n_months]
    first = theta[:, cap.start]
    ups = theta[:, cap.start + 1 : cap.start + n_months]
    downs = theta[:, cap.start + n_months : cap.start + 2 * n_months - 1]
    return first[:, None] + np.concatenate(
        [np.zeros((theta.shape[0], 1)), np.cumsum(ups - downs, axis=1)], axis=1
    )
