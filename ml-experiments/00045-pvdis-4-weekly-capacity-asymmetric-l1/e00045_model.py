"""The weekly-capacity design and the five fits of experiment 00045.

Each Primary's usable daylight half-hours are modelled as

    power = cell(season-year, month, local half-hour, day type) + b * HDD - c(w) * u + error,

where `u` is the unit PV shape, `HDD` is heating degree-days, and `c(w)` is the capacity of the
Monday-start week `w` that the half-hour falls in. The cells and the HDD term are projected out
once, which leaves a quadratic in the weekly capacities with a (weeks by weeks) normal matrix. The
penalised fits then minimise

    0.5 * c' S c / sigma^2 - q' c / sigma^2
        + a * s_P * sum(rises) + a * r * s_P * sum(falls),   c >= 0,

where `s_P` is the median over weeks of the free weekly fit's standard error, so `a` is in units of
one weekly standard error. The solver is a short ADMM routine in numpy.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal

import e00045_common as common
import holidays
import numpy as np
import polars as pl
import scipy.linalg
import scipy.optimize
import scipy.sparse

HDD_BASE_C: Final[float] = 15.5
"""Heating degree-days are the degrees that the temperature falls below this base."""
ADMM_MAX_ITERATIONS: Final[int] = 5000
ADMM_TOLERANCE: Final[float] = 1e-7
"""Relative and absolute tolerance of the ADMM stopping rule, in units of `s_P`."""
RIDGE: Final[float] = 1e-10
"""Added to the diagonal of the normal matrix of observed weeks so that it can be factorised."""
FitName = Literal["free", "symmetric", "asymmetric", "non_decreasing", "season_year"]
"""The five fits."""


@dataclass(frozen=True)
class Rows:
    """The usable half-hours of one Primary, as aligned arrays.

    Attributes:
        power: Observed power, MW.
        u: Unit PV shape.
        hdd: Heating degree-days, degrees C.
        cells: Integer code of the demand cell of each row.
        week: Week index of each row (0 to `n_weeks - 1`).
    """

    power: np.ndarray
    u: np.ndarray
    hdd: np.ndarray
    cells: np.ndarray
    week: np.ndarray

    def before(self, *, origin_week: int) -> Rows:
        """Return the rows of weeks strictly before `origin_week`."""
        keep = self.week < origin_week
        return Rows(
            self.power[keep], self.u[keep], self.hdd[keep], self.cells[keep], self.week[keep]
        )


@dataclass(frozen=True)
class Problem:
    """The projected, scaled least-squares problem of one design at one origin.

    Attributes:
        span: First and last week with data (inclusive); every array below covers this span.
        gram: The normal matrix `S` of the weekly capacities, after the cells and HDD are removed.
        rhs: The vector `q`.
        observed: True for weeks that hold rows.
        sigma2: Residual variance of the free weekly fit.
        scale: `s_P`, the median free-fit standard error of an observed week, MW.
        free: The non-negative least-squares weekly capacities, MW (NaN in unobserved weeks).
        season_years: Season-year of each week in the span.
    """

    span: tuple[int, int]
    gram: np.ndarray
    rhs: np.ndarray
    observed: np.ndarray
    sigma2: float
    scale: float
    free: np.ndarray
    season_years: np.ndarray


def _nnls(*, gram: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Minimise `0.5 x' G x - rhs' x` subject to `x >= 0` through its Cholesky factor."""
    factor = scipy.linalg.cholesky(gram, lower=False)
    target = scipy.linalg.solve_triangular(factor, rhs, trans="T", lower=False)
    return scipy.optimize.nnls(factor, target)[0]


def _project(*, rows: Rows, n_weeks: int) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    """Remove cells and HDD from the weekly regression.

    Args:
        rows: The half-hours to use.
        n_weeks: Number of weeks in the full record.

    Returns:
        The weekly normal matrix `S`, the vector `q`, the residual sum of squares at zero capacity,
        the number of cells, and the number of rows, in that order.
    """
    n = len(rows.power)
    _, cell_index = np.unique(rows.cells, return_inverse=True)
    n_cells = int(cell_index.max()) + 1
    counts = np.bincount(cell_index).astype(float)
    members = scipy.sparse.csr_matrix((np.ones(n), (np.arange(n), cell_index)), shape=(n, n_cells))
    # The regressor of week w is -u on that week's rows, so a positive c lowers power.
    weekly = scipy.sparse.csr_matrix((-rows.u, (np.arange(n), rows.week)), shape=(n, n_weeks))
    hdd = scipy.sparse.csr_matrix(rows.hdd.reshape(-1, 1))
    design = scipy.sparse.hstack([weekly, hdd], format="csr")
    to_cells = (design.T @ members).toarray()
    y_cells = members.T @ rows.power
    gram_full = (design.T @ design).toarray() - (to_cells / counts) @ to_cells.T
    rhs_full = design.T @ rows.power - (to_cells / counts) @ y_cells
    rss0 = float(rows.power @ rows.power - (y_cells**2 / counts).sum())
    g_hh = gram_full[-1, -1]
    if g_hh > 1e-12:
        g_ch = gram_full[:-1, -1]
        gram = gram_full[:-1, :-1] - np.outer(g_ch, g_ch) / g_hh
        rhs = rhs_full[:-1] - g_ch * rhs_full[-1] / g_hh
        rss0 -= rhs_full[-1] ** 2 / g_hh
    else:
        gram, rhs = gram_full[:-1, :-1], rhs_full[:-1]
    return gram, rhs, rss0, float(n_cells), n


def build_problem(*, rows: Rows, n_weeks: int, week_season_year: np.ndarray) -> Problem:
    """Project out the demand cells and HDD and fit the free weekly capacities.

    Args:
        rows: The half-hours to use (already truncated at an origin, if any).
        n_weeks: Number of weeks in the full record.
        week_season_year: Season-year of each of the `n_weeks` weeks.

    Returns:
        The problem restricted to the span from the first to the last week with rows.
    """
    gram, rhs, rss0, n_cells, n = _project(rows=rows, n_weeks=n_weeks)
    observed_all = np.bincount(rows.week, minlength=n_weeks) > 0
    first, last = (int(i) for i in np.nonzero(observed_all)[0][[0, -1]])
    span = slice(first, last + 1)
    gram, rhs, observed = gram[span, span], rhs[span], observed_all[span]
    sub = np.nonzero(observed)[0]
    gram_obs = gram[np.ix_(sub, sub)] + RIDGE * np.eye(len(sub))
    free_obs = _nnls(gram=gram_obs, rhs=rhs[sub])
    rss = rss0 - 2 * rhs[sub] @ free_obs + free_obs @ gram_obs @ free_obs
    sigma2 = max(rss, 1e-12) / max(n - n_cells - len(sub) - 1, 1)
    errors = np.sqrt(sigma2 * np.diag(np.linalg.inv(gram_obs)))
    free = np.full(last - first + 1, np.nan)
    free[sub] = free_obs
    return Problem(
        span=(first, last),
        gram=gram,
        rhs=rhs,
        observed=observed,
        sigma2=float(sigma2),
        scale=float(np.median(errors)),
        free=free,
        season_years=week_season_year[span],
    )


class Design:
    """One Primary's rows, with the projected problem cached for each origin.

    Args:
        rows: The usable half-hours (weeks with fewer usable days than the minimum already dropped).
        n_weeks: Number of weeks in the full record.
        week_season_year: Season-year of each week, from the week's Monday.
    """

    def __init__(self, *, rows: Rows, n_weeks: int, week_season_year: np.ndarray) -> None:  # noqa: D107
        self.rows = rows
        self.n_weeks = n_weeks
        self.week_season_year = week_season_year
        self._problems: dict[int | None, Problem] = {}

    def problem(self, *, origin_week: int | None = None) -> Problem:
        """Return the problem using only weeks before `origin_week` (all weeks when `None`)."""
        if origin_week not in self._problems:
            rows = self.rows if origin_week is None else self.rows.before(origin_week=origin_week)
            self._problems[origin_week] = build_problem(
                rows=rows, n_weeks=self.n_weeks, week_season_year=self.week_season_year
            )
        return self._problems[origin_week]


def design_from_half_hourly(*, half_hourly: pl.DataFrame, weeks: pl.DataFrame, site: str) -> Design:
    """Build one Primary's design from the U1 outputs.

    Args:
        half_hourly: `inputs_half_hourly.parquet`.
        weeks: `inputs_primary_weeks.parquet`.
        site: Frame site code, such as `"D16"`.

    Returns:
        The design over the `common.N_WEEKS` weeks, with unobserved weeks left without rows.
    """
    unobserved = weeks.filter((pl.col("site") == site) & pl.col("unobserved"))["week"]
    holiday_dates = pl.Series(
        sorted(holidays.country_holidays("GB", subdiv="ENG", years=range(2018, 2028)))
    ).cast(pl.Date)
    rows = (
        half_hourly.filter(
            (pl.col("site") == site)
            & pl.col("usable")
            & ~pl.col("week").is_in(unobserved.implode())
        )
        .with_columns(
            local=pl.col("time").dt.offset_by("-15m").dt.convert_time_zone("Europe/London")
        )
        .with_columns(
            cell=(
                (pl.col("local").dt.year() - (pl.col("local").dt.month() < 9).cast(pl.Int32)).cast(
                    pl.Int64
                )
                * 1_000_000
                + pl.col("local").dt.month().cast(pl.Int64) * 10_000
                + (
                    pl.col("local").dt.hour().cast(pl.Int64) * 2
                    + pl.col("local").dt.minute().cast(pl.Int64) // 30
                )
                * 10
                + (
                    (pl.col("local").dt.weekday() >= 6)
                    | pl.col("local").dt.date().is_in(holiday_dates.implode())
                ).cast(pl.Int64)
            ),
            hdd=(HDD_BASE_C - pl.col("temperature_2m")).clip(lower_bound=0.0),
        )
        .sort("time")
    )
    season_years = np.array(
        [
            (start := common.week_start(week=w)).year - (start.month < 9)
            for w in range(common.N_WEEKS)
        ]
    )
    return Design(
        rows=Rows(
            power=rows["power"].to_numpy().astype(float),
            u=rows["u"].to_numpy().astype(float),
            hdd=rows["hdd"].to_numpy().astype(float),
            cells=rows["cell"].to_numpy(),
            week=rows["week"].to_numpy().astype(np.int64),
        ),
        n_weeks=common.N_WEEKS,
        week_season_year=season_years,
    )


def _increments(*, n: int) -> scipy.sparse.csr_matrix:
    """Return the (n - 1) by n matrix whose product with a path is its weekly increments."""
    return scipy.sparse.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n)).tocsr()


def asymmetric_l1_admm(
    *,
    gram: np.ndarray,
    rhs: np.ndarray,
    rise_weight: float,
    fall_weight: float,
    start: np.ndarray | None = None,
    tolerance: float = ADMM_TOLERANCE,
    max_iterations: int = ADMM_MAX_ITERATIONS,
) -> tuple[np.ndarray, int]:
    """Minimise `0.5 x' G x - q' x + rise_weight * sum(rises) + fall_weight * sum(falls)`, `x >= 0`.

    A rise is a positive weekly increment and a fall is a negative one. A fall weight of infinity
    forbids falls.

    Args:
        gram: Positive semi-definite matrix `G` (it needs only to be positive definite on the
            directions the penalty does not control).
        rhs: The vector `q`.
        rise_weight: Penalty per unit rise.
        fall_weight: Penalty per unit fall (may be `math.inf`).
        start: Initial path (defaults to zeros).
        tolerance: Absolute and relative tolerance on the primal and dual residuals.
        max_iterations: Iteration cap.

    Returns:
        The path and the number of iterations used.
    """
    n = len(rhs)
    diff = _increments(n=n)
    dtd = (diff.T @ diff).toarray() + np.eye(n)
    x = np.zeros(n) if start is None else np.maximum(start, 0.0)
    w_inc, w_lvl = diff @ x, x.copy()
    u_inc, u_lvl = np.zeros(n - 1), np.zeros(n)
    rho = 1.0
    factor = scipy.linalg.cho_factor(gram + rho * dtd)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        x = scipy.linalg.cho_solve(factor, rhs + rho * (diff.T @ (w_inc - u_inc) + (w_lvl - u_lvl)))
        d_x = diff @ x
        t_inc, t_lvl = d_x + u_inc, x + u_lvl
        old_inc, old_lvl = w_inc, w_lvl
        w_inc = np.where(
            t_inc > rise_weight / rho,
            t_inc - rise_weight / rho,
            np.where(t_inc < -fall_weight / rho, t_inc + fall_weight / rho, 0.0),
        )
        w_lvl = np.maximum(t_lvl, 0.0)
        u_inc += d_x - w_inc
        u_lvl += x - w_lvl
        primal = math.hypot(np.linalg.norm(d_x - w_inc), np.linalg.norm(x - w_lvl))
        dual = rho * np.linalg.norm(diff.T @ (w_inc - old_inc) + (w_lvl - old_lvl))
        size = math.sqrt(2 * n)
        scale_p = max(
            np.linalg.norm(d_x), np.linalg.norm(x), np.linalg.norm(w_inc), np.linalg.norm(w_lvl)
        )
        scale_d = rho * math.hypot(np.linalg.norm(diff.T @ u_inc), np.linalg.norm(u_lvl))
        if primal <= tolerance * (size + scale_p) and dual <= tolerance * (size + scale_d):
            break
        if iterations % 20 == 0 and (primal > 10 * dual or dual > 10 * primal):
            factor_up = 2.0 if primal > 10 * dual else 0.5
            rho *= factor_up
            u_inc /= factor_up
            u_lvl /= factor_up
            factor = scipy.linalg.cho_factor(gram + rho * dtd)
    return np.maximum(w_lvl, 0.0), iterations


def scale_p(*, design: Design, origin_week: int | None = None) -> float:
    """Return `s_P`, the median free-fit standard error of an observed week, before the origin."""
    return design.problem(origin_week=origin_week).scale


def _pad(
    *, design: Design, problem: Problem, span_path: np.ndarray, recovery_ratio: float
) -> np.ndarray:
    """Place a span's path in the full-length vector (NaN outside) and divide by the ratio."""
    path = np.full(design.n_weeks, np.nan)
    path[problem.span[0] : problem.span[1] + 1] = span_path / recovery_ratio
    return path


def fit_path(
    *,
    design: Design,
    a: float,
    r: float,
    origin_week: int | None = None,
    recovery_ratio: float = 1.0,
) -> np.ndarray:
    """Fit the weekly capacity path with rise weight `a` and fall weight `a * r`.

    `a = 0` is the free weekly fit (NaN in unobserved weeks), `r = 1` the symmetric fit, `r >= 10`
    the asymmetric fit, and `r = inf` the strictly non-decreasing fit. With `origin_week`, only
    weeks before it are used, and `s_P` is computed from them alone; weeks outside the span of
    those weeks are NaN.

    Args:
        design: The Primary's design.
        a: Rise penalty in units of `s_P` (one weekly standard error).
        r: Ratio of the fall penalty to the rise penalty.
        origin_week: First week not to use, or `None` for all weeks.
        recovery_ratio: The recovery ratio at the origin; the path is divided by it.

    Returns:
        The capacity of every week, MW, length `design.n_weeks`.
    """
    problem = design.problem(origin_week=origin_week)
    if a == 0.0:
        return _pad(
            design=design, problem=problem, span_path=problem.free, recovery_ratio=recovery_ratio
        )
    # Work in units of s_P: x = c / s_P, so G' = s_P^2 G / sigma^2 and q' = s_P q / sigma^2.
    gram = problem.gram * problem.scale**2 / problem.sigma2
    rhs = problem.rhs * problem.scale / problem.sigma2
    gram = gram + RIDGE * np.eye(len(rhs))
    start = np.where(np.isnan(problem.free), np.nanmean(problem.free), problem.free) / problem.scale
    x, _ = asymmetric_l1_admm(gram=gram, rhs=rhs, rise_weight=a, fall_weight=a * r, start=start)
    if math.isinf(r):
        x = np.maximum.accumulate(x)
    return _pad(
        design=design, problem=problem, span_path=x * problem.scale, recovery_ratio=recovery_ratio
    )


def fit_season_year_constant(
    *, design: Design, origin_week: int | None = None, recovery_ratio: float = 1.0
) -> np.ndarray:
    """Fit one capacity per season-year (Sept to Aug) with `c >= 0`; weeks of the span get it."""
    problem = design.problem(origin_week=origin_week)
    years, inverse = np.unique(problem.season_years, return_inverse=True)
    pooling = scipy.sparse.csr_matrix(
        (np.ones(len(inverse)), (np.arange(len(inverse)), inverse)),
        shape=(len(inverse), len(years)),
    )
    pooled_gram = (pooling.T @ problem.gram @ pooling) + RIDGE * np.eye(len(years))
    pooled = _nnls(gram=np.asarray(pooled_gram), rhs=np.asarray(pooling.T @ problem.rhs).ravel())
    return _pad(
        design=design, problem=problem, span_path=pooled[inverse], recovery_ratio=recovery_ratio
    )


FITS: Final[dict[FitName, Callable[..., np.ndarray]]] = {
    "free": lambda *, design, origin_week=None, recovery_ratio=1.0, **_: fit_path(
        design=design, a=0.0, r=1.0, origin_week=origin_week, recovery_ratio=recovery_ratio
    ),
    "symmetric": lambda *, design, a, origin_week=None, recovery_ratio=1.0, **_: fit_path(
        design=design, a=a, r=1.0, origin_week=origin_week, recovery_ratio=recovery_ratio
    ),
    "asymmetric": fit_path,
    "non_decreasing": lambda *, design, a, origin_week=None, recovery_ratio=1.0, **_: fit_path(
        design=design, a=a, r=math.inf, origin_week=origin_week, recovery_ratio=recovery_ratio
    ),
    "season_year": lambda *, design, origin_week=None, recovery_ratio=1.0, **_: (
        fit_season_year_constant(
            design=design, origin_week=origin_week, recovery_ratio=recovery_ratio
        )
    ),
}
"""The five fits by name. All take `design`, `origin_week`, and `recovery_ratio` by keyword; the
penalised ones also take `a` (and `r` for the asymmetric fit)."""
