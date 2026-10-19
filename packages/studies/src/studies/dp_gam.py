"""The convex model of the differentiable-physics (DP) series: demand minus solar, fitted per site.

**Net power in points of effective capacity is a sum of terms that are linear in their
parameters**, so the fit is a penalised least-squares problem with sign constraints on some
parameters, and it has one global optimum:

    net power = profile(day type, half-hour) + holiday periods + Christmas tent + day-of-year
                + heating slopes * relu(15.5 C - smoothed temperature) + lighting ramps
                - capacity(season-year) * derated unit PV shape + drift + steps

- **Profile:** one level per day type (weekday, Saturday, Sunday, bank holiday) and local
  half-hour.
- **Holiday, tent, harmonics:** one coefficient per hour band for each named holiday period, for
  the tent and the signed tent around Christmas, and for two day-of-year harmonics.
- **Heating:** a non-negative slope per hour band on the hinge `relu(15.5 - T_k)` of temperature
  averaged over k = 6, 24, 72, and 168 hours. Fixing the threshold keeps the fit convex.
- **Lighting:** four coefficients on ramps of how dark it is (zero at midday).
- **Solar:** the capacity of season-year `s` is the sum of non-negative increments up to `s`, so it
  never decreases. The unit PV shape is derated by 0.4% per kelvin of cell temperature. The
  penalty acts on each season-year's capacity, so a window that holds one season-year shrinks
  that one capacity as a single ridge parameter would.
- **Drift D1:** a level per hour band on monthly knots counted back from the end of the training
  window, linear between knots, with a second-difference penalty. A forecast holds the last knot's
  value.
- **Steps D3:** a second, piecewise-constant level path per hour band whose monthly first
  differences carry an L1 penalty, written as non-negative positive and negative parts.

**The solver works on the Gram matrix.** `accumulate_gram` builds the design in chunks and sums
`X^T W X` and `X^T W y`, with a leading batch dimension of weightings (a month-block bootstrap
resample is a row of weights). `solve_bounded_quadratic` then minimises `theta^T A theta - 2 b^T
theta` over the box `theta_j >= 0` for the constrained parameters with a projected-Newton
active-set method, batched over the leading dimension, in float64 by default.

**The spread is a residual quantile model fitted on a replica of the scored forecast.**
`replica_residuals` fits the arm 12 months before the cut, with the arm's own window rule, holds its
drift, and predicts the 12 months that follow. Those residuals are as stale as the scored forecast
(0 to 12 months), because the scored fit is frozen for the whole year. `ResidualSpread` turns them
into quantiles per hour band and calendar quarter.
"""

import calendar
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Final, Literal

import numpy as np
import torch

from studies.dp_calendar import (
    BAND_NAMES,
    DAY_TYPES,
    HALF_HOURS_PER_DAY,
    HEATING_THRESHOLD_C,
    HOLIDAY_PERIODS,
    SMOOTHING_HOURS,
    lighting_ramps,
)
from studies.dp_features import FeatureCache, SiteFeatures
from studies.experiment_store import HOLDOUT_START


def naive_utc(*, moment: datetime) -> np.datetime64:
    """Return a timezone-aware datetime as a naive UTC `datetime64[us]`.

    Args:
        moment: An aware datetime in any time zone.

    Returns:
        The same instant as UTC microseconds.

    Raises:
        ValueError: If `moment` has no time zone, since a naive wall time could be local.
    """
    if moment.tzinfo is None:
        msg = "a cut or window bound must carry a time zone"
        raise ValueError(msg)
    return np.datetime64(moment.astimezone(UTC).replace(tzinfo=None), "us")


DriftType = Literal["none", "d1", "d1_steps"]
"""The drift form: no drift, the smooth level path, or the smooth path plus sparse steps."""

QUANTILE_LEVELS: Final[tuple[float, ...]] = (
    0.01,
    0.05,
    0.10,
    0.25,
    0.40,
    0.50,
    0.60,
    0.75,
    0.90,
    0.95,
    0.99,
)
"""The 11 quantile levels of experiment 87."""

QUANTILE_COLUMNS: Final[tuple[str, ...]] = (
    "q01",
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
    "q99",
)
"""The prediction column of each level."""

N_BANDS: Final[int] = len(BAND_NAMES)
KNOT_SPACING_DAYS: Final[float] = 30.4375
"""One month, the spacing of the drift knots."""
SITE_YEAR_ROWS: Final[int] = 17532
"""Half-hours in a site-year; the drift and step penalties scale with training site-years."""
HEATING_SCALE_C: Final[float] = 10.0
"""The heating hinge is divided by this, so its coefficient is points per 10 degrees."""
DERATING_PER_KELVIN: Final[float] = 0.004
CELL_TEMPERATURE_RISE_K: Final[float] = 25.0
"""Cell temperature is the air temperature plus this many kelvin times the unit shape."""
REFERENCE_CELL_TEMPERATURE_C: Final[float] = 25.0
"""The cell temperature at which the unit PV shape is not derated."""
CHUNK_ROWS: Final[int] = 4000
"""Rows of the design built at once."""
MIN_SPREAD_GROUP_ROWS: Final[int] = 200
"""A (band, quarter) group with fewer residuals falls back to its band, then to all residuals."""
REPLICA_MONTHS: Final[int] = 12
"""The scored forecast is frozen for this many months, and so is its replica."""
LAST_TRAINED_MIN_SHAPE: Final[float] = 1e-3
"""A training row carries solar signal when its derated unit shape is above this."""


@dataclass(frozen=True)
class GamSpec:
    """The penalties and the drift form of one fit.

    Attributes:
        drift: The drift form.
        ridge: The ridge penalty on the holiday, tent, harmonic, heating, lighting, and capacity
            terms.
        profile_ridge: The ridge penalty on the daily profile, small so the profile is free.
        drift_step: The weight of the drift's second-difference penalty, per training site-year.
        drift_ridge: A small ridge on the drift levels, which settles the level shared with the
            profile.
        step_penalty: The weight of the L1 penalty on the steps, per training site-year.
    """

    drift: DriftType = "none"
    ridge: float = 1.0
    profile_ridge: float = 1e-3
    drift_step: float = 3e4
    drift_ridge: float = 1e-2
    step_penalty: float = 300.0


@dataclass(frozen=True)
class TrainingWindow:
    """Which rows a fit may train on, relative to its cut.

    Attributes:
        start: The first usable instant (the window's lower bound), or `None` for the substation's
            first usable instant.
        months: If set, the window is the last `months` months before the cut and `start` is
            ignored.
    """

    start: datetime | None = None
    months: int | None = None


@dataclass(frozen=True)
class GamLayout:
    """The columns of the design for one fit.

    Attributes:
        spec: The penalties and drift form.
        season_years: The season-years that have a capacity increment.
        knot_days: The drift knots as days since 1970-01-01, ascending; the last is the cut.
        slices: Each term's columns.
        n_columns: The number of columns.
        constrained: True for the columns that must be non-negative.
    """

    spec: GamSpec
    season_years: tuple[int, ...]
    knot_days: np.ndarray
    slices: dict[str, slice]
    n_columns: int
    constrained: np.ndarray


def make_layout(
    *, spec: GamSpec, season_years: Sequence[int], cut: datetime, earliest: np.datetime64
) -> GamLayout:
    """Lay out the design's columns.

    Args:
        spec: The penalties and drift form.
        season_years: The season-years that get a capacity increment.
        cut: The end of the training window; the last drift knot is here.
        earliest: The earliest instant any training row can have, as UTC microseconds, which
            sets how many monthly knots are needed.

    Returns:
        The layout.
    """
    n_bands = N_BANDS
    sizes: dict[str, int] = {
        "profile": len(DAY_TYPES) * HALF_HOURS_PER_DAY,
        "holiday": len(HOLIDAY_PERIODS) * n_bands,
        "tent": 2 * n_bands,
        "harmonics": 4 * n_bands,
        "heating": len(SMOOTHING_HOURS) * n_bands,
        "lighting": 4,
        "capacity": len(season_years),
    }
    cut_day = naive_utc(moment=cut).astype("datetime64[s]").astype(np.int64) / 86400.0
    first_day = earliest.astype("datetime64[s]").astype(np.int64) / 86400.0
    n_knots = int(np.ceil((cut_day - first_day) / KNOT_SPACING_DAYS)) + 1
    knot_days = cut_day - KNOT_SPACING_DAYS * np.arange(n_knots)[::-1]
    if spec.drift != "none":
        sizes["drift"] = n_bands * n_knots
    if spec.drift == "d1_steps":
        sizes["steps"] = 2 * n_bands * (n_knots - 1)
    slices: dict[str, slice] = {}
    start = 0
    for name, size in sizes.items():
        slices[name] = slice(start, start + size)
        start += size
    constrained = np.zeros(start, dtype=bool)
    for name in ("heating", "capacity", "steps"):
        if name in slices:
            constrained[slices[name]] = True
    return GamLayout(
        spec=spec,
        season_years=tuple(int(s) for s in season_years),
        knot_days=knot_days,
        slices=slices,
        n_columns=start,
        constrained=constrained,
    )


def derated_unit_shape(*, features: SiteFeatures, index: np.ndarray) -> np.ndarray:
    """Return the unit PV shape times its temperature derating at the rows `index`.

    Args:
        features: The substation's arrays.
        index: Row positions on the grid.

    Returns:
        `u * (1 - 0.004 * (T_cell - 25))` with `T_cell = T + 25 * u`, never negative.
    """
    unit = features.pv_unit_shape[index]
    cell = features.temperature_c[index] + CELL_TEMPERATURE_RISE_K * unit
    return unit * np.clip(
        1.0 - DERATING_PER_KELVIN * (cell - REFERENCE_CELL_TEMPERATURE_C), 0, None
    )


def _band_one_hot(*, band: np.ndarray) -> np.ndarray:
    """Return the (n, 4) one-hot of the hour band."""
    return (band[:, None] == np.arange(N_BANDS)[None, :]).astype(np.float64)


def _hat_functions(*, days: np.ndarray, knot_days: np.ndarray) -> np.ndarray:
    """Return the (n, K) linear-interpolation weights of `days` on the knots, held beyond them."""
    clipped = np.clip(days, knot_days[0], knot_days[-1])
    upper = np.clip(np.searchsorted(knot_days, clipped, side="right"), 1, len(knot_days) - 1)
    lower = upper - 1
    fraction = (clipped - knot_days[lower]) / (knot_days[upper] - knot_days[lower])
    weights = np.zeros((len(days), len(knot_days)))
    rows = np.arange(len(days))
    weights[rows, lower] = 1.0 - fraction
    weights[rows, upper] = fraction
    return weights


def design_matrix(
    *, cache: FeatureCache, features: SiteFeatures, layout: GamLayout, index: np.ndarray
) -> np.ndarray:
    """Build the design matrix of the grid rows `index`.

    Args:
        cache: The cache, for the calendar columns and the grid.
        features: The substation's arrays.
        layout: The columns.
        index: Row positions on the grid.

    Returns:
        A float64 array of shape (len(index), layout.n_columns). The capacity columns carry a
        minus sign, so a non-negative coefficient lowers net power.
    """
    calendar = cache.calendar
    n = len(index)
    x = np.zeros((n, layout.n_columns))
    rows = np.arange(n)
    band = calendar.band[index].astype(np.int64)
    band_hot = _band_one_hot(band=band)
    x[
        rows,
        layout.slices["profile"].start
        + calendar.day_type[index].astype(np.int64) * HALF_HOURS_PER_DAY
        + calendar.half_hour[index],
    ] = 1.0
    period = calendar.holiday_period[index].astype(np.int64)
    in_period = period >= 0
    holiday_start = layout.slices["holiday"].start
    x[rows[in_period], holiday_start + period[in_period] * N_BANDS + band[in_period]] = 1.0
    tent = np.stack([calendar.tent[index], calendar.signed_tent[index]], axis=1)
    x[:, layout.slices["tent"]] = (tent[:, :, None] * band_hot[:, None, :]).reshape(n, -1)
    harmonics = calendar.harmonics[index]
    x[:, layout.slices["harmonics"]] = (harmonics[:, :, None] * band_hot[:, None, :]).reshape(n, -1)
    hinge = (
        np.clip(HEATING_THRESHOLD_C - features.smoothed_temperature_c[index], 0.0, None)
        / HEATING_SCALE_C
    )
    x[:, layout.slices["heating"]] = (hinge[:, :, None] * band_hot[:, None, :]).reshape(n, -1)
    x[:, layout.slices["lighting"]] = lighting_ramps(elevation_deg=features.elevation_deg[index])
    shape = derated_unit_shape(features=features, index=index)
    season = calendar.season_year[index]
    years = np.array(layout.season_years)
    x[:, layout.slices["capacity"]] = -shape[:, None] * (season[:, None] >= years[None, :])
    if "drift" in layout.slices:
        days = _days_since_epoch(cache=cache, index=index)
        hats = _hat_functions(days=days, knot_days=layout.knot_days)
        x[:, layout.slices["drift"]] = (band_hot[:, :, None] * hats[:, None, :]).reshape(n, -1)
    if "steps" in layout.slices:
        days = _days_since_epoch(cache=cache, index=index)
        after = (days[:, None] >= layout.knot_days[None, 1:]).astype(np.float64)
        steps = band_hot[:, :, None] * after[:, None, :]
        x[:, layout.slices["steps"]] = np.concatenate(
            [steps.reshape(n, -1), -steps.reshape(n, -1)], axis=1
        )
    return x


def _days_since_epoch(*, cache: FeatureCache, index: np.ndarray) -> np.ndarray:
    """Return the days since 1970-01-01 of the rows' half-hour ends."""
    return cache.end_times[index].astype("datetime64[s]").astype(np.int64) / 86400.0


def penalty_matrix(*, layout: GamLayout, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the quadratic penalty matrix and the linear cost vector of the objective.

    The objective is `sum w r^2 + theta^T P theta + c^T theta`.

    Args:
        layout: The columns.
        n_rows: The number of training rows, which scales the drift and step penalties.

    Returns:
        `(P, c)` with shapes (p, p) and (p,).
    """
    spec = layout.spec
    p = layout.n_columns
    matrix = np.zeros((p, p))
    cost = np.zeros(p)
    scale = n_rows / SITE_YEAR_ROWS
    matrix[layout.slices["profile"], layout.slices["profile"]] = spec.profile_ridge * np.eye(
        layout.slices["profile"].stop - layout.slices["profile"].start
    )
    for name in ("holiday", "tent", "harmonics", "heating", "lighting"):
        block = layout.slices[name]
        matrix[block, block] = spec.ridge * np.eye(block.stop - block.start)
    capacity = layout.slices["capacity"]
    n_years = capacity.stop - capacity.start
    cumulative = np.tril(np.ones((n_years, n_years)))
    matrix[capacity, capacity] = spec.ridge * cumulative.T @ cumulative
    if "drift" in layout.slices:
        n_knots = len(layout.knot_days)
        second = np.diff(np.eye(n_knots), n=2, axis=0)
        smooth = spec.drift_step * scale * second.T @ second + spec.drift_ridge * np.eye(n_knots)
        start = layout.slices["drift"].start
        for band in range(N_BANDS):
            block = slice(start + band * n_knots, start + (band + 1) * n_knots)
            matrix[block, block] = smooth
    if "steps" in layout.slices:
        block = layout.slices["steps"]
        matrix[block, block] = 1e-6 * np.eye(block.stop - block.start)
        cost[block] = spec.step_penalty * scale
    return matrix, cost


def training_rows(
    *,
    features: SiteFeatures,
    cache: FeatureCache,
    cut: datetime,
    window: TrainingWindow,
) -> np.ndarray:
    """Return the grid rows a fit may train on.

    A row trains when it is in `default_rows` (after the early-calibration cut, outside the
    lockdown window and the switching masks), has a reading and a finite solar shape, ends no later
    than `cut`, and lies inside the window.

    Args:
        features: The substation's arrays.
        cache: The cache.
        cut: The fit's cut; no row ends after it.
        window: The window rule.

    Returns:
        Sorted grid positions.
    """
    end = cache.end_times
    cut64 = naive_utc(moment=cut)
    if window.months is not None:
        start64 = naive_utc(moment=_months_before(moment=cut, months=window.months))
    elif window.start is not None:
        start64 = naive_utc(moment=window.start)
    else:
        start64 = naive_utc(moment=features.table.first_usable)
    keep = (
        features.default_rows
        & features.present
        & np.isfinite(features.y)
        & np.isfinite(features.pv_unit_shape)
        & np.isfinite(features.smoothed_temperature_c).all(axis=1)
        & (end <= cut64)
        & (end > start64)
    )
    return np.flatnonzero(keep)


def _months_before(*, moment: datetime, months: int) -> datetime:
    """Return the instant `months` calendar months before `moment`, clamping the day of month."""
    total = moment.year * 12 + (moment.month - 1) - months
    year, month = total // 12, total % 12 + 1
    day = min(moment.day, calendar.monthrange(year, month)[1])
    return moment.replace(year=year, month=month, day=day)


def assert_no_lookahead(*, cache: FeatureCache, index: np.ndarray, cut: datetime) -> None:
    """Raise if any training row ends after the cut.

    Args:
        cache: The cache.
        index: The training rows.
        cut: The fit's cut.

    Raises:
        ValueError: If a row ends after `cut`.
    """
    cut64 = naive_utc(moment=cut)
    if len(index) and cache.end_times[index].max() > cut64:
        msg = f"a training row ends after the cut {cut}: lookahead"
        raise ValueError(msg)


def assert_before_holdout(*, cache: FeatureCache, index: np.ndarray) -> None:
    """Raise if any row to score ends on or after the held-out period's start.

    Args:
        cache: The cache.
        index: The rows to score.

    Raises:
        ValueError: If a row ends on or after 2026-07-01 00:00 UTC.
    """
    holdout = naive_utc(moment=HOLDOUT_START)
    if len(index) and cache.end_times[index].max() >= holdout:
        msg = "a row to score is on or after 2026-07-01, which is held out"
        raise ValueError(msg)


def _chunks(*, index: np.ndarray) -> Iterator[slice]:
    """Yield consecutive slices of `index` of at most `CHUNK_ROWS` rows."""
    for start in range(0, len(index), CHUNK_ROWS):
        yield slice(start, min(start + CHUNK_ROWS, len(index)))


def accumulate_gram(
    *,
    cache: FeatureCache,
    features: SiteFeatures,
    layout: GamLayout,
    index: np.ndarray,
    weights: np.ndarray | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum `X^T W X` and `X^T W y` over the rows `index`, one Gram matrix per weighting.

    Args:
        cache: The cache.
        features: The substation's arrays.
        layout: The columns.
        index: The training rows.
        weights: Row weights of shape (batch, len(index)), or `None` for one weighting of ones.
        device: Where the sums are accumulated.
        dtype: The floating-point type of the sums.

    Returns:
        `(gram, rhs)` of shapes (batch, p, p) and (batch, p).
    """
    n = len(index)
    if weights is None:
        weights = np.ones((1, n))
    batch = weights.shape[0]
    p = layout.n_columns
    gram = torch.zeros((batch, p, p), dtype=dtype, device=device)
    rhs = torch.zeros((batch, p), dtype=dtype, device=device)
    weight_tensor = torch.as_tensor(weights, dtype=dtype, device=device)
    for chunk in _chunks(index=index):
        rows = index[chunk]
        x = torch.as_tensor(
            design_matrix(cache=cache, features=features, layout=layout, index=rows),
            dtype=dtype,
            device=device,
        )
        y = torch.as_tensor(features.y[rows], dtype=dtype, device=device)
        w = weight_tensor[:, chunk]
        weighted = w[:, :, None] * x[None, :, :]
        gram += torch.einsum("bnp,nq->bpq", weighted, x)
        rhs += torch.einsum("bnp,n->bp", weighted, y)
    return gram, rhs


PIVOT_PATIENCE: Final[int] = 3
"""Iterations without fewer broken conditions before block pivoting falls back to a single move."""


def default_tolerance(*, dtype: torch.dtype) -> float:
    """Return the solver's default optimality tolerance: 1e-7, or 10,000 machine epsilons if larger.

    Args:
        dtype: The floating-point type of the problem.

    Returns:
        1e-7 in float64 and about 1.2e-3 in float32.
    """
    return max(1e-7, 1e4 * float(torch.finfo(dtype).eps))


def solve_bounded_quadratic(
    *,
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    constrained: torch.Tensor,
    max_iterations: int = 500,
    tolerance: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimise `theta^T A theta - 2 b^T theta` with `theta_j >= 0` where `constrained[j]`.

    Block principal pivoting (Judice and Pires; Kim and Park). The optimality conditions of this
    problem are a linear complementarity problem whose matrix `A` is positive definite, a
    P-matrix, so principal pivoting terminates at the one optimum. Each iteration solves for the
    free parameters with the others at zero, then moves every parameter that broke a condition
    between the free set and the zero set. When the number of broken conditions stops falling for
    three iterations, only the broken parameter with the largest index moves, which guarantees
    termination.

    Args:
        matrix: `A`, shape (batch, p, p), positive definite.
        rhs: `b`, shape (batch, p).
        constrained: Boolean, shape (p,), True for the non-negative parameters.
        max_iterations: The most pivoting iterations.
        tolerance: The optimality tolerance, relative to the largest entry of `b`. The default is
            1e-7 in float64 and 10,000 machine epsilons (about 1e-3) in float32. The matrix of a
            real fit is badly conditioned, so float32 is only safe on well-conditioned problems and
            the fits of this series use float64.

    Returns:
        `(theta, violation)`: the solution of shape (batch, p), and for each problem the largest
        violation of the optimality conditions, relative to the largest entry of `b`.

    Raises:
        ValueError: If a problem has not converged after `max_iterations`.
    """
    constrained = constrained.to(matrix.device)
    epsilon = float(torch.finfo(matrix.dtype).eps)
    if tolerance is None:
        tolerance = default_tolerance(dtype=matrix.dtype)
    batch, p = rhs.shape
    scale = rhs.abs().amax(dim=1).clamp(min=1.0)
    unconstrained = torch.linalg.solve(matrix, rhs[:, :, None])[:, :, 0]
    free = ~constrained[None, :] | (unconstrained > 0.0)
    best_broken = torch.full((batch,), p + 1, dtype=torch.long, device=matrix.device)
    patience = torch.zeros(batch, dtype=torch.long, device=matrix.device)
    indices = torch.arange(p, device=matrix.device)
    theta = torch.zeros_like(rhs)
    for _ in range(max_iterations):
        theta = _solve_on_free_set(matrix=matrix, rhs=rhs, free=free)
        gradient = torch.einsum("bpq,bq->bp", matrix, theta) - rhs
        theta_tolerance = 100.0 * epsilon * theta.abs().amax(dim=1, keepdim=True).clamp(min=1.0)
        broken = (free & constrained[None, :] & (theta < -theta_tolerance)) | (
            ~free & (gradient < -tolerance * scale[:, None])
        )
        count = broken.sum(dim=1)
        if bool((count == 0).all()):
            break
        improved = count < best_broken
        best_broken = torch.where(improved, count, best_broken)
        patience = torch.where(improved, torch.zeros_like(patience), patience + 1)
        backup = (patience >= PIVOT_PATIENCE) & (count > 0)
        largest = torch.where(broken, indices[None, :], -torch.ones_like(indices)[None, :]).amax(
            dim=1
        )
        single = indices[None, :] == largest[:, None]
        exchange = torch.where(backup[:, None], broken & single, broken)
        free = free ^ exchange
    else:
        theta = _solve_on_free_set(matrix=matrix, rhs=rhs, free=free)
    theta = torch.where(constrained[None, :], theta.clamp(min=0.0), theta)
    gradient = torch.einsum("bpq,bq->bp", matrix, theta) - rhs
    violation = _kkt_violation(theta=theta, gradient=gradient, constrained=constrained) / scale
    if not bool((violation <= 10.0 * tolerance).all()):
        msg = f"the bounded solver did not converge: worst violation {float(violation.max()):.3g}"
        raise ValueError(msg)
    return theta, violation


def _solve_on_free_set(
    *, matrix: torch.Tensor, rhs: torch.Tensor, free: torch.Tensor
) -> torch.Tensor:
    """Solve `A_FF x_F = b_F` for each problem, with `x = 0` outside the free set."""
    outside = ~free
    mask = outside[:, :, None] | outside[:, None, :]
    reduced = torch.where(mask, torch.zeros_like(matrix), matrix)
    diagonal = torch.arange(matrix.shape[1], device=matrix.device)
    reduced[:, diagonal, diagonal] = torch.where(
        outside, torch.ones_like(rhs), reduced[:, diagonal, diagonal]
    )
    return torch.linalg.solve(
        reduced, torch.where(outside, torch.zeros_like(rhs), rhs)[:, :, None]
    )[:, :, 0]


def _kkt_violation(
    *, theta: torch.Tensor, gradient: torch.Tensor, constrained: torch.Tensor
) -> torch.Tensor:
    """Return the largest violation of the optimality conditions of each problem."""
    free = ~constrained[None, :] | (theta > 0.0)
    at_bound = ~free
    free_violation = torch.where(free, gradient.abs(), torch.zeros_like(gradient))
    bound_violation = torch.where(at_bound, (-gradient).clamp(min=0.0), torch.zeros_like(gradient))
    return torch.maximum(free_violation.amax(dim=1), bound_violation.amax(dim=1))


@dataclass(frozen=True)
class GamFit:
    """A fitted model, or a batch of them that share a layout.

    Attributes:
        layout: The columns.
        theta: The coefficients, shape (batch, p), float64.
        cut: The fit's cut; every training row ends no later.
        n_rows: The number of training rows.
        site: The substation's label.
        last_season_year: The latest season-year with a training row that carries solar signal.
            A later season-year takes this one's capacity.
        violation: Each fit's optimality violation.
    """

    layout: GamLayout
    theta: np.ndarray
    cut: datetime
    n_rows: int
    site: str
    last_season_year: int
    violation: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def capacity_points(self, *, member: int = 0) -> dict[int, float]:
        """Return the fitted capacity of each season-year, in points of effective capacity.

        Args:
            member: Which fit of the batch.

        Returns:
            A map from season-year to the sum of increments up to it.
        """
        increments = self.theta[member, self.layout.slices["capacity"]]
        return dict(zip(self.layout.season_years, np.cumsum(increments).tolist(), strict=True))


def prepare_fit(
    *,
    cache: FeatureCache,
    site: str,
    spec: GamSpec,
    cut: datetime,
    window: TrainingWindow,
    season_years: Sequence[int],
) -> tuple[GamLayout, np.ndarray]:
    """Choose a fit's training rows and lay out its columns, checking the rows for lookahead.

    Args:
        cache: The cache.
        site: The substation's label.
        spec: The penalties and drift form.
        cut: The cut; training rows end no later.
        window: The window rule.
        season_years: The season-years with a capacity increment.

    Returns:
        The layout and the sorted grid positions of the training rows.

    Raises:
        ValueError: If no row can train, or a training row ends after the cut.
    """
    index = training_rows(features=cache.sites[site], cache=cache, cut=cut, window=window)
    if len(index) == 0:
        msg = f"{site} has no training rows before {cut}"
        raise ValueError(msg)
    assert_no_lookahead(cache=cache, index=index, cut=cut)
    earliest = cache.end_times[index[0]]
    layout = make_layout(spec=spec, season_years=season_years, cut=cut, earliest=earliest)
    return layout, index


def fit_gam(
    *,
    cache: FeatureCache,
    site: str,
    specs: Sequence[GamSpec],
    cut: datetime,
    window: TrainingWindow,
    season_years: Sequence[int],
    weights: np.ndarray | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> GamFit:
    """Fit the convex model of one substation, for one or more penalty sets or weightings.

    All `specs` must share a drift form. The Gram matrix is built once and each batch member adds
    its own penalty, so a grid of penalties costs one pass over the rows. Give either several
    `specs` and no `weights`, or one spec and `weights`.

    Args:
        cache: The cache.
        site: The substation's label.
        specs: The penalty sets, one per batch member (or one, to share across `weights`).
        cut: The cut; training rows end no later.
        window: The window rule.
        season_years: The season-years with a capacity increment.
        weights: Optional row weights of shape (batch, n_training_rows).
        device: Where the arithmetic runs.
        dtype: The floating-point type.

    Returns:
        The fitted batch.

    Raises:
        ValueError: If the specs differ in drift form, the batch sizes do not match, or no row
            can train.
    """
    features = cache.sites[site]
    if len({spec.drift for spec in specs}) != 1:
        msg = "all specs of one call must share a drift form"
        raise ValueError(msg)
    layout, index = prepare_fit(
        cache=cache,
        site=site,
        spec=specs[0],
        cut=cut,
        window=window,
        season_years=season_years,
    )
    batch = max(len(specs), 1 if weights is None else weights.shape[0])
    if len(specs) not in (1, batch) or (weights is not None and weights.shape[0] != batch):
        msg = "specs and weights must give the same batch size, or one of them a single member"
        raise ValueError(msg)
    gram, rhs = accumulate_gram(
        cache=cache,
        features=features,
        layout=layout,
        index=index,
        weights=weights,
        device=device,
        dtype=dtype,
    )
    penalties = []
    costs = []
    for spec in specs:
        matrix, cost = penalty_matrix(layout=replace(layout, spec=spec), n_rows=len(index))
        penalties.append(matrix)
        costs.append(cost)
    penalty = torch.as_tensor(np.stack(penalties), dtype=dtype, device=device)
    linear = torch.as_tensor(np.stack(costs), dtype=dtype, device=device)
    if gram.shape[0] == 1 and batch > 1:
        gram = gram.expand(batch, -1, -1)
        rhs = rhs.expand(batch, -1)
    if penalty.shape[0] == 1 and batch > 1:
        penalty = penalty.expand(batch, -1, -1)
        linear = linear.expand(batch, -1)
    theta, violation = solve_bounded_quadratic(
        matrix=gram + penalty,
        rhs=rhs - 0.5 * linear,
        constrained=torch.as_tensor(layout.constrained),
    )
    signal = derated_unit_shape(features=features, index=index) > LAST_TRAINED_MIN_SHAPE
    if not signal.any():
        msg = f"{site} has no training row that carries solar signal before {cut}"
        raise ValueError(msg)
    return GamFit(
        layout=layout,
        theta=theta.cpu().numpy(),
        cut=cut,
        n_rows=len(index),
        site=site,
        last_season_year=int(cache.calendar.season_year[index[signal]].max()),
        violation=violation.cpu().numpy(),
    )


def predict_gam(
    *, cache: FeatureCache, fit: GamFit, index: np.ndarray, member: int = 0
) -> np.ndarray:
    """Predict net power in points at the grid rows `index`.

    The drift is held at its last knot, because the layout's knots end at the cut and the hat
    functions are clipped there. The capacity of a season-year after `fit.last_season_year` is set
    explicitly to that season-year's capacity.

    Args:
        cache: The cache.
        fit: The fitted batch.
        index: The grid rows to predict.
        member: Which fit of the batch.

    Returns:
        The predictions, float64, one per row.

    Raises:
        ValueError: If a row to predict has no finite input.
    """
    assert_before_holdout(cache=cache, index=index)
    features = cache.sites[fit.site]
    layout = fit.layout
    out = np.empty(len(index))
    theta = fit.theta[member]
    for chunk in _chunks(index=index):
        rows = index[chunk]
        x = design_matrix(cache=cache, features=features, layout=layout, index=rows)
        capacity = layout.slices["capacity"]
        held_year = np.minimum(cache.calendar.season_year[rows], fit.last_season_year)
        held = held_year[:, None] >= np.array(layout.season_years)[None, :]
        x[:, capacity] = -derated_unit_shape(features=features, index=rows)[:, None] * held
        if not np.isfinite(x).all():
            msg = f"{fit.site} has a non-finite input among the rows to predict"
            raise ValueError(msg)
        out[chunk] = x @ theta
    return out


@dataclass(frozen=True)
class ResidualSpread:
    """Empirical quantiles of residuals, per hour band and calendar quarter.

    Attributes:
        table: Shape (4 bands, 4 quarters, 11 levels).
    """

    table: np.ndarray

    @staticmethod
    def fit(*, residuals: np.ndarray, band: np.ndarray, quarter: np.ndarray) -> ResidualSpread:
        """Fit the quantiles of `residuals` (actual minus predicted mean).

        A (band, quarter) group with fewer than `MIN_SPREAD_GROUP_ROWS` residuals uses its band's
        residuals, and a band with fewer uses all residuals.

        Args:
            residuals: Replica residuals in points.
            band: The hour band of each residual.
            quarter: The calendar quarter of each residual.

        Returns:
            The fitted spread.

        Raises:
            ValueError: If there are no residuals.
        """
        if len(residuals) == 0:
            msg = "no residuals to fit the spread on"
            raise ValueError(msg)
        levels = np.array(QUANTILE_LEVELS)
        everything = np.quantile(residuals, levels)
        table = np.empty((N_BANDS, 4, len(levels)))
        for b in range(N_BANDS):
            in_band = band == b
            band_values = residuals[in_band]
            band_quantiles = (
                np.quantile(band_values, levels)
                if len(band_values) >= MIN_SPREAD_GROUP_ROWS
                else everything
            )
            for q in range(4):
                group = residuals[in_band & (quarter == q)]
                table[b, q] = (
                    np.quantile(group, levels)
                    if len(group) >= MIN_SPREAD_GROUP_ROWS
                    else band_quantiles
                )
        return ResidualSpread(table=table)

    def quantiles(self, *, mean: np.ndarray, band: np.ndarray, quarter: np.ndarray) -> np.ndarray:
        """Return the 11 quantiles around each predicted mean.

        Args:
            mean: Predicted mean in points.
            band: The hour band of each row.
            quarter: The calendar quarter of each row.

        Returns:
            An array of shape (len(mean), 11), non-decreasing along the last axis.
        """
        return mean[:, None] + self.table[band, quarter]


@dataclass(frozen=True)
class ReplicaResiduals:
    """Residuals of a replica of the scored forecast.

    Attributes:
        residuals: Actual minus predicted mean, in points.
        band: The hour band of each residual.
        quarter: The calendar quarter of each residual.
        index: The grid rows of the residuals.
        fit_cut: The replica fit's cut, 12 months before the final cut; every residual row ends
            after it, and none ends after the final cut.
    """

    residuals: np.ndarray
    band: np.ndarray
    quarter: np.ndarray
    index: np.ndarray
    fit_cut: datetime


def replica_cut(*, cut: datetime) -> datetime:
    """Return the cut of the replica fit: `REPLICA_MONTHS` months before the final cut."""
    return _months_before(moment=cut, months=REPLICA_MONTHS)


def replica_residuals(
    *,
    cache: FeatureCache,
    site: str,
    spec: GamSpec,
    cut: datetime,
    window: TrainingWindow,
    season_years: Sequence[int],
) -> ReplicaResiduals:
    """Return the residuals of a replica of the scored forecast.

    The scored forecast is one fit frozen for the 12 months after the cut. Its replica is the same
    arm fitted at the cut 12 months earlier, with the arm's window rule applied at that earlier
    cut and the drift held, and used to predict the 12 months up to the final cut. The residuals
    are therefore as stale as the scored forecast: 0 to 12 months, not the 0 to 3 months of a
    quarter-by-quarter fit. The replica year is the year before the cut, so the spread is
    fitted on a year that tuning may also have used.

    Args:
        cache: The cache.
        site: The substation's label.
        spec: The arm's penalties and drift form.
        cut: The cut of the final fit.
        window: The arm's window rule.
        season_years: The season-years with a capacity increment.

    Returns:
        The residuals over the rows that have a reading, sit outside the switching masks, and end
        after the substation's first usable instant.
    """
    features = cache.sites[site]
    fit_cut = replica_cut(cut=cut)
    fit = fit_gam(
        cache=cache,
        site=site,
        specs=[spec],
        cut=fit_cut,
        window=window,
        season_years=season_years,
    )
    keep = (
        features.present
        & ~features.switching_mask
        & np.isfinite(features.y)
        & np.isfinite(features.pv_unit_shape)
        & np.isfinite(features.smoothed_temperature_c).all(axis=1)
        & (cache.end_times > naive_utc(moment=fit_cut))
        & (cache.end_times > naive_utc(moment=features.table.first_usable))
        & (cache.end_times <= naive_utc(moment=cut))
    )
    index = np.flatnonzero(keep)
    predicted = predict_gam(cache=cache, fit=fit, index=index)
    return ReplicaResiduals(
        residuals=features.y[index] - predicted,
        band=cache.calendar.band[index],
        quarter=cache.calendar.quarter[index],
        index=index,
        fit_cut=fit_cut,
    )


def month_block_weights(
    *,
    cache: FeatureCache,
    index: np.ndarray,
    n_resamples: int,
    seed: int,
) -> np.ndarray:
    """Return month-block bootstrap row weights.

    Each resample draws whole calendar months with replacement, as many as the rows span, and a
    row's weight is how many times its month was drawn.

    Args:
        cache: The cache.
        index: The training rows.
        n_resamples: The number of resamples (the batch size).
        seed: Seeds the draws.

    Returns:
        Weights of shape (n_resamples, len(index)).
    """
    month = cache.end_times[index].astype("datetime64[M]")
    months, inverse = np.unique(month, return_inverse=True)
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(months), size=(n_resamples, len(months)))
    counts = np.stack([np.bincount(row, minlength=len(months)) for row in draws])
    return counts[:, inverse].astype(np.float64)
