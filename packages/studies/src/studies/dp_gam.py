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
- **Pinned capacity:** with `capacity="pinned"` the capacity is not fitted. The weekly capacity of
  `studies.dp_pinned_capacity` times the derated unit PV shape moves to the target, so the fit
  has no capacity columns and a forecast subtracts the held capacity times the shape.
- **COVID flag:** with `covid_flag=True` the fit keeps the lockdown rows and gives each hour band
  one indicator over the fixed lockdown window.
- **The final drift knot:** `final_knot` is `free`, `shrunk` (a forecast holds the last knot
  pulled towards the mean of the last 12 knots), or `pinned` (the fit ties the last knot to the one
  before it).

**Part 5 options (experiment 110), all off by default so that a default `GamSpec` reproduces Parts 2
to 4 exactly.**

- **Routes by which demand terms imitate solar.** `orthogonal_penalty` adds a quadratic penalty on
  the covariance of the harmonics block and of the heating block with the unit PV shape, within
  each calendar quarter and half-hour cell of the daylight rows. `shared_harmonics` gives the
  day-of-year harmonics one coefficient for all hour bands, and `slow_heating_only` keeps only the
  72 and 168 hour temperatures. `solar="anomaly"` writes the solar term as capacity times the
  anomaly of the unit PV shape from its month-by-half-hour mean, plus a free non-negative
  climatology per month and half-hour.
- **The export clip.** `clip` applies a smooth clip to the pinned solar output, and `profile_clip`
  profiles the clip's capacity multiplier and ceiling on a grid, one convex fit each.
- **The unmetered generator node.** `node` adds a non-negative level path `g_d` with one value per
  block of `step_days` days (a day by default), and a positive and a negative step part per block.
  A stiff quadratic tie makes the step parts carry the change in level, and a linear (L1) cost
  charges them, so the fit stays a bounded quadratic program and the level never goes below zero.

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
from datetime import datetime
from typing import Final, Literal

import numpy as np
import polars as pl
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
from studies.dp_features import FeatureCache, SiteFeatures, naive_utc
from studies.dp_pinned_capacity import PinnedCapacity, pinned_capacity
from studies.experiment_store import HOLDOUT_START

DriftType = Literal["none", "d1", "d1_steps"]
"""The drift form: no drift, the smooth level path, or the smooth path plus sparse steps."""

CapacityType = Literal["fitted", "pinned"]
"""Whether the PV capacity is fitted, or taken from the pinned real-time vintage."""

FinalKnotType = Literal["free", "shrunk", "pinned"]
"""How the drift's last knot is treated: free, shrunk at forecast time, or tied in the fit."""

SolarFormType = Literal["capacity", "anomaly"]
"""The solar term's form: capacity times the unit shape, or capacity times its anomaly plus a
climatology."""

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
FINAL_KNOT_TIE_WEIGHT: Final[float] = 1e6
"""The weight of the penalty that ties the last drift knot to the one before it.

The weight is absolute, not scaled by the number of rows, so a fit with `final_knot="pinned"` needs
float64 (the default).
"""
SHRINK_KNOTS: Final[int] = 12
"""A shrunk final knot is pulled towards the mean of this many last knots."""
SLOW_SMOOTHING_COLUMNS: Final[tuple[int, ...]] = (2, 3)
"""The columns of `SMOOTHING_HOURS` (72 and 168 hours) that `slow_heating_only` keeps."""
N_MONTHS: Final[int] = 12
"""Months of the year, for the solar climatology of the anomaly form."""
NODE_TIE_RATIO: Final[float] = 10000.0
"""The weight of the quadratic tie between a node level's change and its two step parts.

The weight is this multiple of the linear cost of a point of step.

With cost `c` per point of step and tie weight `w`, the steps carry a block-to-block change in level
less `c / (2 w)` points. At a ratio of 10,000 a change of 0.00005 points goes uncharged, so a slow
ramp of the level costs about as much as one step of the same total size. A larger ratio makes the
system worse conditioned.
"""
NODE_RIDGE: Final[float] = 1e-3
"""A tiny ridge on the node's columns, which keeps the step columns (with no design) well posed."""
DEFAULT_MAX_ITERATIONS: Final[int] = 500
"""The most pivoting iterations `solve_bounded_quadratic` makes by default."""
NODE_MAX_ITERATIONS: Final[int] = 20000
"""The most pivoting iterations of a fit with a node, whose many step columns sit at their bound."""
NODE_NEGATIVE_TOLERANCE: Final[float] = 1e-6
"""A node level this far below zero, in points, is a bug and raises."""
CLIP_BETA_PER_POINT: Final[float] = 2.0
"""The sharpness of the smooth clip, per point of effective capacity."""
CLIP_MULTIPLIERS: Final[tuple[float, ...]] = tuple(round(0.7 + 0.05 * i, 2) for i in range(13))
"""The grid of the clip's capacity multiplier m, from 0.7 to 1.3."""
CLIP_SHARES: Final[tuple[float, ...]] = tuple(round(0.4 + 0.05 * i, 2) for i in range(13))
"""The grid of the clip's ceiling as a share of m times the held capacity, from 40 to 100%."""
ORTHOGONAL_BLOCKS: Final[tuple[str, ...]] = ("harmonics", "heating")
"""The demand terms whose covariance with the unit PV shape the orthogonal penalty constrains."""
MIN_ORTHOGONAL_CELL_ROWS: Final[int] = 20
"""A quarter-by-half-hour cell with fewer daylight rows is left out of the orthogonal penalty."""


@dataclass(frozen=True)
class ClipSpec:
    """A smooth export clip on the pinned solar output.

    The solar term becomes `P_max - softplus(beta * (P_max - m * C * u)) / beta`, where `C` is the
    pinned capacity, `u` the derated unit PV shape, and `P_max` is `p_max_share * m * H` with `H`
    the pinned capacity held after the cut.

    Attributes:
        multiplier: The capacity multiplier `m`.
        p_max_share: The ceiling as a share of `m * H`; 1 is the top of the profile grid.
        beta: The sharpness, per point.
    """

    multiplier: float = 1.0
    p_max_share: float = 1.0
    beta: float = CLIP_BETA_PER_POINT


@dataclass(frozen=True)
class NodeSpec:
    """The penalties of the unmetered generator node.

    The costs do not depend on the length of the training window, so the same penalties mean the
    same thing at every cut. An outage's evidence (the rows it covers times its depth squared) is
    fixed, and so are the costs of the steps that make it.

    Attributes:
        step_penalty: The linear cost of each point of step, in the units of the squared error.
        level_penalty: The linear cost of each point of level for each day it lasts. It settles
            the level that the profile also explains, so it can be small.
        step_days: The length of one level block in days. Each block is one more unknown with a
            tie and two step parts, so a longer block makes the fit smaller and faster, and cannot
            resolve an outage shorter than the block.
    """

    step_penalty: float = 300.0
    level_penalty: float = 0.03
    step_days: int = 1

    def __post_init__(self) -> None:
        """Refuse a block shorter than one day.

        Raises:
            ValueError: If `step_days` is below 1.
        """
        if self.step_days < 1:
            msg = f"step_days must be at least 1, not {self.step_days}"
            raise ValueError(msg)


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
        capacity: Whether the PV capacity is fitted or pinned.
        covid_flag: True to keep the lockdown rows and fit one indicator per hour band over them.
        final_knot: How the drift's last knot is treated.
        final_knot_shrink: For `final_knot="shrunk"`, the share of the way a forecast moves the
            last knot towards the mean of the last 12 knots, between 0 and 1.
        solar: The solar term's form; `anomaly` needs a fitted capacity.
        anomaly_ridge: The ridge on the anomaly form's climatology, per training site-year.
        shared_harmonics: True to give the day-of-year harmonics one coefficient for all bands.
        slow_heating_only: True to keep only the 72 and 168 hour temperatures in the heating hinge.
        orthogonal_penalty: The weight of the covariance penalty, in units of the Gram matrix's
            mean diagonal over the penalty matrix's mean diagonal; 0 switches the penalty off.
        clip: The export clip on the pinned solar output, or `None` for none; needs a pinned
            capacity.
        node: The unmetered generator node, or `None` for none.
    """

    drift: DriftType = "none"
    ridge: float = 1.0
    profile_ridge: float = 1e-3
    drift_step: float = 3e4
    drift_ridge: float = 1e-2
    step_penalty: float = 300.0
    capacity: CapacityType = "fitted"
    covid_flag: bool = False
    final_knot: FinalKnotType = "free"
    final_knot_shrink: float = 0.5
    solar: SolarFormType = "capacity"
    anomaly_ridge: float = 1.0
    shared_harmonics: bool = False
    slow_heating_only: bool = False
    orthogonal_penalty: float = 0.0
    clip: ClipSpec | None = None
    node: NodeSpec | None = None

    def __post_init__(self) -> None:
        """Refuse inconsistent options.

        Raises:
            ValueError: If `final_knot_shrink` is outside 0 to 1, `final_knot` is not `free` while
                `drift` is `none`, the anomaly form is asked for with a pinned capacity, a clip
                is asked for without a pinned capacity, the orthogonal penalty is negative, or a
                node is asked for without drift.
        """
        if not 0.0 <= self.final_knot_shrink <= 1.0:
            msg = f"final_knot_shrink must be between 0 and 1, not {self.final_knot_shrink}"
            raise ValueError(msg)
        if self.final_knot != "free" and self.drift == "none":
            msg = f"final_knot={self.final_knot!r} needs a drift term, but drift is 'none'"
            raise ValueError(msg)
        if self.solar == "anomaly" and self.capacity != "fitted":
            msg = (
                "the anomaly form of the solar term fits the capacity, so capacity must be 'fitted'"
            )
            raise ValueError(msg)
        if self.clip is not None and self.capacity != "pinned":
            msg = "the export clip acts on the pinned solar output, so capacity must be 'pinned'"
            raise ValueError(msg)
        if self.orthogonal_penalty < 0.0:
            msg = f"orthogonal_penalty must not be negative, not {self.orthogonal_penalty}"
            raise ValueError(msg)
        if self.node is not None and self.drift == "none":
            msg = (
                "the node needs a drift term beside it: a non-negative step path on a base with no "
                "drift absorbs any fall in demand"
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class TrainingWindow:
    """Which rows a fit may train on, relative to its cut.

    Attributes:
        start: The first usable instant (the window's lower bound), or `None` for the substation's
            first usable instant.
        months: If set, the window is the last `months` months before the cut and `start` is
            ignored.
        dark_only: True to train only on the rows whose CAMS irradiance is below 1 W/m², where an
            MVA substation's reading holds no solar signal.
    """

    start: datetime | None = None
    months: int | None = None
    dark_only: bool = False


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
        anomaly_mean: For the anomaly form, the training mean of the derated unit shape by month
            (0 to 11) and local half-hour, shape (12, 48); `None` otherwise.
        node_first_day: The first day of the node's level path, as days since 1970-01-01.
        n_node_steps: The number of level blocks of the node's path, 0 without a node.
    """

    spec: GamSpec
    season_years: tuple[int, ...]
    knot_days: np.ndarray
    slices: dict[str, slice]
    n_columns: int
    constrained: np.ndarray
    anomaly_mean: np.ndarray | None = None
    node_first_day: float = 0.0
    n_node_steps: int = 0


def make_layout(
    *,
    spec: GamSpec,
    season_years: Sequence[int],
    cut: datetime,
    earliest: np.datetime64,
    anomaly_mean: np.ndarray | None = None,
) -> GamLayout:
    """Lay out the design's columns.

    Args:
        spec: The penalties and drift form.
        season_years: The season-years that get a capacity increment.
        cut: The end of the training window; the last drift knot is here.
        earliest: The earliest instant any training row can have, as UTC microseconds, which
            sets how many monthly knots are needed.
        anomaly_mean: The training mean of the derated unit shape by month and half-hour, shape
            (12, 48); required exactly when `spec.solar` is `anomaly`.

    Returns:
        The layout.

    Raises:
        ValueError: If `anomaly_mean` is missing for the anomaly form, or given for another form.
    """
    if (anomaly_mean is not None) != (spec.solar == "anomaly"):
        msg = "an anomaly mean is required exactly when the solar term has the anomaly form"
        raise ValueError(msg)
    n_bands = N_BANDS
    n_heating = len(SLOW_SMOOTHING_COLUMNS) if spec.slow_heating_only else len(SMOOTHING_HOURS)
    sizes: dict[str, int] = {
        "profile": len(DAY_TYPES) * HALF_HOURS_PER_DAY,
        "holiday": len(HOLIDAY_PERIODS) * n_bands,
        "tent": 2 * n_bands,
        "harmonics": 4 if spec.shared_harmonics else 4 * n_bands,
        "heating": n_heating * n_bands,
        "lighting": 4,
        "capacity": len(season_years) if spec.capacity == "fitted" else 0,
    }
    if spec.solar == "anomaly":
        sizes["climatology"] = N_MONTHS * HALF_HOURS_PER_DAY
    if spec.covid_flag:
        sizes["covid"] = n_bands
    cut_day = naive_utc(moment=cut).astype("datetime64[s]").astype(np.int64) / 86400.0
    first_day = earliest.astype("datetime64[s]").astype(np.int64) / 86400.0
    n_knots = int(np.ceil((cut_day - first_day) / KNOT_SPACING_DAYS)) + 1
    knot_days = cut_day - KNOT_SPACING_DAYS * np.arange(n_knots)[::-1]
    if spec.drift != "none":
        sizes["drift"] = n_bands * n_knots
    if spec.drift == "d1_steps":
        sizes["steps"] = 2 * n_bands * (n_knots - 1)
    n_node_steps = 0
    node_first_day = 0.0
    if spec.node is not None:
        node_first_day = float(np.floor(first_day))
        n_days = max(int(np.floor(cut_day - node_first_day - 1.0 / HALF_HOURS_PER_DAY)) + 1, 1)
        n_node_steps = -(-n_days // spec.node.step_days)
        sizes["node_level"] = n_node_steps
        sizes["node_up"] = n_node_steps
        sizes["node_down"] = n_node_steps
    slices: dict[str, slice] = {}
    start = 0
    for name, size in sizes.items():
        slices[name] = slice(start, start + size)
        start += size
    constrained = np.zeros(start, dtype=bool)
    for name in (
        "heating",
        "capacity",
        "steps",
        "climatology",
        "node_level",
        "node_up",
        "node_down",
    ):
        if name in slices:
            constrained[slices[name]] = True
    return GamLayout(
        spec=spec,
        season_years=tuple(int(s) for s in season_years) if spec.capacity == "fitted" else (),
        knot_days=knot_days,
        slices=slices,
        n_columns=start,
        constrained=constrained,
        anomaly_mean=anomaly_mean,
        node_first_day=node_first_day,
        n_node_steps=n_node_steps,
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


def calendar_cell(
    *, cache: FeatureCache, index: np.ndarray, kind: Literal["month", "quarter"]
) -> np.ndarray:
    """Return the calendar cell of each row: its local month or quarter times 48 plus its half-hour.

    Args:
        cache: The cache.
        index: Row positions on the grid.
        kind: `month` for the 12 months of the anomaly form's climatology, `quarter` for the 4
            quarters of the covariance penalty.

    Returns:
        The cell: 48 times the month or quarter plus the half-hour, so 576 cells for months and 192
        for quarters.
    """
    if kind == "month":
        period = _month_index(cache=cache, index=index)
    else:
        period = cache.calendar.quarter[index].astype(np.int64)
    return period * HALF_HOURS_PER_DAY + cache.calendar.half_hour[index].astype(np.int64)


def _month_index(*, cache: FeatureCache, index: np.ndarray) -> np.ndarray:
    """Return the local month of the rows, 0 for January to 11 for December."""
    months = cache.calendar.local_date[index].astype("datetime64[M]").astype(np.int64)
    return (months % N_MONTHS).astype(np.int64)


def solar_shape(
    *, cache: FeatureCache, features: SiteFeatures, layout: GamLayout, index: np.ndarray
) -> np.ndarray:
    """Return the shape the capacity columns multiply: the derated unit shape, or its anomaly.

    Args:
        cache: The cache, for the rows' local month and half-hour.
        features: The substation's arrays.
        layout: The columns; the anomaly form subtracts its training mean.
        index: Row positions on the grid.

    Returns:
        `derated_unit_shape`, minus the layout's month-by-half-hour mean for the anomaly form.
    """
    shape = derated_unit_shape(features=features, index=index)
    if layout.anomaly_mean is None:
        return shape
    month = _month_index(cache=cache, index=index)
    return shape - layout.anomaly_mean[month, cache.calendar.half_hour[index].astype(np.int64)]


def month_half_hour_mean(
    *, cache: FeatureCache, features: SiteFeatures, index: np.ndarray
) -> np.ndarray:
    """Return the mean derated unit shape by local month and half-hour over the rows `index`.

    Args:
        cache: The cache.
        features: The substation's arrays.
        index: The training rows; no other row contributes.

    Returns:
        An array of shape (12, 48); a cell with no row is 0.
    """
    shape = derated_unit_shape(features=features, index=index)
    cell = calendar_cell(cache=cache, index=index, kind="month")
    n_cells = N_MONTHS * HALF_HOURS_PER_DAY
    totals = np.bincount(cell, weights=shape, minlength=n_cells)
    counts = np.bincount(cell, minlength=n_cells)
    return (totals / np.maximum(counts, 1)).reshape(N_MONTHS, HALF_HOURS_PER_DAY)


def node_step_index(*, cache: FeatureCache, layout: GamLayout, index: np.ndarray) -> np.ndarray:
    """Return the node-path block of each row, held at the first and last block beyond them.

    Args:
        cache: The cache.
        layout: The columns, with a node.
        index: Row positions on the grid.

    Returns:
        The block of the half-hour's start, counted from the node path's first day.

    Raises:
        ValueError: If the layout has no node.
    """
    node = layout.spec.node
    if node is None:
        msg = "the layout's spec has no node"
        raise ValueError(msg)
    start = _days_since_epoch(cache=cache, index=index) - 1.0 / HALF_HOURS_PER_DAY
    block = np.floor((start - layout.node_first_day) / node.step_days).astype(np.int64)
    return np.clip(block, 0, layout.n_node_steps - 1)


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
        minus sign, so a non-negative coefficient lowers net power. A pinned-capacity layout has
        no capacity columns.
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
    if layout.spec.shared_harmonics:
        x[:, layout.slices["harmonics"]] = harmonics
    else:
        x[:, layout.slices["harmonics"]] = (harmonics[:, :, None] * band_hot[:, None, :]).reshape(
            n, -1
        )
    smoothed = features.smoothed_temperature_c[index]
    if layout.spec.slow_heating_only:
        smoothed = smoothed[:, SLOW_SMOOTHING_COLUMNS]
    hinge = np.clip(HEATING_THRESHOLD_C - smoothed, 0.0, None) / HEATING_SCALE_C
    x[:, layout.slices["heating"]] = (hinge[:, :, None] * band_hot[:, None, :]).reshape(n, -1)
    x[:, layout.slices["lighting"]] = lighting_ramps(elevation_deg=features.elevation_deg[index])
    shape = solar_shape(cache=cache, features=features, layout=layout, index=index)
    season = calendar.season_year[index]
    years = np.array(layout.season_years)
    x[:, layout.slices["capacity"]] = -shape[:, None] * (season[:, None] >= years[None, :])
    if "climatology" in layout.slices:
        cell = calendar_cell(cache=cache, index=index, kind="month")
        x[rows, layout.slices["climatology"].start + cell] = -1.0
    if "node_level" in layout.slices:
        block = node_step_index(cache=cache, layout=layout, index=index)
        x[rows, layout.slices["node_level"].start + block] = -1.0
    if "covid" in layout.slices:
        x[:, layout.slices["covid"]] = features.lockdown[index][:, None] * band_hot
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
    for name in ("holiday", "tent", "harmonics", "heating", "lighting", "covid"):
        if name not in layout.slices:
            continue
        block = layout.slices[name]
        matrix[block, block] = spec.ridge * np.eye(block.stop - block.start)
    capacity = layout.slices["capacity"]
    n_years = capacity.stop - capacity.start
    cumulative = np.tril(np.ones((n_years, n_years)))
    matrix[capacity, capacity] = spec.ridge * cumulative.T @ cumulative
    if "climatology" in layout.slices:
        block = layout.slices["climatology"]
        matrix[block, block] = spec.anomaly_ridge * scale * np.eye(block.stop - block.start)
    if "drift" in layout.slices:
        n_knots = len(layout.knot_days)
        second = np.diff(np.eye(n_knots), n=2, axis=0)
        smooth = spec.drift_step * scale * second.T @ second + spec.drift_ridge * np.eye(n_knots)
        start = layout.slices["drift"].start
        for band in range(N_BANDS):
            block = slice(start + band * n_knots, start + (band + 1) * n_knots)
            matrix[block, block] = smooth
            if spec.final_knot == "pinned":
                tie = np.zeros(n_knots)
                tie[-2:] = (-1.0, 1.0)
                matrix[block, block] += FINAL_KNOT_TIE_WEIGHT * np.outer(tie, tie)
    if "steps" in layout.slices:
        block = layout.slices["steps"]
        matrix[block, block] = 1e-6 * np.eye(block.stop - block.start)
        cost[block] = spec.step_penalty * scale
    if spec.node is not None:
        _add_node_penalty(matrix=matrix, cost=cost, layout=layout)
    return matrix, cost


def _add_node_penalty(*, matrix: np.ndarray, cost: np.ndarray, layout: GamLayout) -> None:
    """Add the node's tie, ridge, and linear costs to the penalty `matrix` and `cost` in place.

    The tie `NODE_TIE_RATIO * c * (g_d - g_(d-1) - up_d + down_d)^2` makes the two step parts carry
    the level's change from one block to the next, and the linear cost on the steps then charges its
    absolute value. The costs are not scaled by the number of training rows (see `NodeSpec`).
    """
    node = layout.spec.node
    if node is None:
        msg = "the layout's spec has no node"
        raise ValueError(msg)
    level = layout.slices["node_level"]
    up = layout.slices["node_up"]
    down = layout.slices["node_down"]
    for block in (level, up, down):
        matrix[block, block] += NODE_RIDGE * np.eye(block.stop - block.start)
    cost[level] += node.level_penalty * node.step_days
    cost[up] += node.step_penalty
    cost[down] += node.step_penalty
    blocks = np.arange(1, layout.n_node_steps)
    columns = np.stack(
        [
            level.start + blocks,
            level.start + blocks - 1,
            up.start + blocks,
            down.start + blocks,
        ]
    )
    signs = np.array([1.0, -1.0, -1.0, 1.0])
    weight = NODE_TIE_RATIO * node.step_penalty
    for first in range(4):
        for second in range(4):
            np.add.at(
                matrix,
                (columns[first], columns[second]),
                weight * signs[first] * signs[second],
            )


def training_rows(
    *,
    features: SiteFeatures,
    cache: FeatureCache,
    cut: datetime,
    window: TrainingWindow,
    keep_lockdown: bool = False,
) -> np.ndarray:
    """Return the grid rows a fit may train on.

    A row trains when it is in `default_rows` (after the early-calibration cut, outside the
    lockdown window and the switching masks), has a reading and a finite solar shape, ends no later
    than `cut`, and lies inside the window. With `keep_lockdown`, a lockdown row that is in the
    pool and outside the switching masks trains too.

    Args:
        features: The substation's arrays.
        cache: The cache.
        cut: The fit's cut; no row ends after it.
        window: The window rule.
        keep_lockdown: True to keep the lockdown rows, for a fit with the COVID flag.

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
    allowed = features.default_rows
    if window.dark_only:
        allowed = allowed & features.dark
    if keep_lockdown:
        allowed = allowed | (features.lockdown & features.in_pool & ~features.switching_mask)
    keep = (
        allowed
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


def row_chunks(*, index: np.ndarray) -> Iterator[slice]:
    """Yield consecutive slices of `index` of at most `CHUNK_ROWS` rows."""
    for start in range(0, len(index), CHUNK_ROWS):
        yield slice(start, min(start + CHUNK_ROWS, len(index)))


def clipped_pv(*, unclipped: np.ndarray, p_max: float, beta: float) -> np.ndarray:
    """Return `P_max - softplus(beta * (P_max - x)) / beta`, the smooth clip of `x` at `P_max`.

    Args:
        unclipped: The unclipped solar output `x`, in points.
        p_max: The ceiling `P_max`, in points; infinity means no clip.
        beta: The sharpness, per point.

    Returns:
        The clipped output, never below zero: `x` far below the ceiling, and `P_max` far above it.
    """
    if np.isinf(p_max):
        return unclipped
    return np.maximum(p_max - np.logaddexp(0.0, beta * (p_max - unclipped)) / beta, 0.0)


def pinned_solar_points(
    *,
    cache: FeatureCache,
    features: SiteFeatures,
    pinned: PinnedCapacity,
    index: np.ndarray,
    clip: ClipSpec | None = None,
) -> np.ndarray:
    """Return the pinned capacity times the derated unit PV shape at the rows `index`, in points.

    Args:
        cache: The cache, for the rows' times.
        features: The substation's arrays.
        pinned: The pinned capacity.
        index: Grid rows.
        clip: The export clip, or `None` for none.

    Returns:
        The solar output the model subtracts from demand, never negative. With a clip, the output
        of `m * C * u` clipped at `p_max_share * m * H`, where `H` is the capacity held after the
        cut.
    """
    solar = pinned.at_rows(cache=cache, index=index) * derated_unit_shape(
        features=features, index=index
    )
    if clip is None:
        return solar
    return clipped_pv(
        unclipped=clip.multiplier * solar,
        p_max=clip.p_max_share * clip.multiplier * pinned.held_points,
        beta=clip.beta,
    )


def accumulate_gram(
    *,
    cache: FeatureCache,
    features: SiteFeatures,
    layout: GamLayout,
    index: np.ndarray,
    weights: np.ndarray | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    pinned: PinnedCapacity | None = None,
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
        pinned: The pinned capacity, whose solar output is added to the target; required exactly
            when the layout's capacity is pinned.

    Returns:
        `(gram, rhs)` of shapes (batch, p, p) and (batch, p).

    Raises:
        ValueError: If `pinned` is missing for a pinned layout, or given for a fitted one.
    """
    if (pinned is not None) != (layout.spec.capacity == "pinned"):
        msg = "a pinned capacity is required exactly when the layout's capacity is pinned"
        raise ValueError(msg)
    n = len(index)
    if weights is None:
        weights = np.ones((1, n))
    batch = weights.shape[0]
    p = layout.n_columns
    gram = torch.zeros((batch, p, p), dtype=dtype, device=device)
    rhs = torch.zeros((batch, p), dtype=dtype, device=device)
    weight_tensor = torch.as_tensor(weights, dtype=dtype, device=device)
    for chunk in row_chunks(index=index):
        rows = index[chunk]
        x = torch.as_tensor(
            design_matrix(cache=cache, features=features, layout=layout, index=rows),
            dtype=dtype,
            device=device,
        )
        target = features.y[rows]
        if pinned is not None:
            target = target + pinned_solar_points(
                cache=cache, features=features, pinned=pinned, index=rows, clip=layout.spec.clip
            )
        y = torch.as_tensor(target, dtype=dtype, device=device)
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
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
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
    unconstrained = _solve(matrix=matrix, rhs=rhs)
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


def _solve(*, matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve `A x = b` for each problem of the batch.

    On the CPU a batch of more than one system is solved one system at a time. PyTorch's batched
    LU solve with the MKL backend fails ("Pivots given to lu_solve must all be greater or equal to
    1") from about 200 by 200 upwards when it runs on four or more threads, and it can return wrong
    answers without raising; a single system is unaffected. The GPU solves the batch in one call.

    Args:
        matrix: Shape (batch, p, p).
        rhs: Shape (batch, p).

    Returns:
        The solutions, shape (batch, p).
    """
    if matrix.device.type == "cpu" and matrix.shape[0] > 1:
        return torch.stack(
            [torch.linalg.solve(m, r[:, None])[:, 0] for m, r in zip(matrix, rhs, strict=True)]
        )
    return torch.linalg.solve(matrix, rhs[:, :, None])[:, :, 0]


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
    return _solve(matrix=reduced, rhs=torch.where(outside, torch.zeros_like(rhs), rhs))


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
        pinned: The pinned capacity of a fit with `capacity="pinned"`, else `None`.
    """

    layout: GamLayout
    theta: np.ndarray
    cut: datetime
    n_rows: int
    site: str
    last_season_year: int
    violation: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pinned: PinnedCapacity | None = None

    def capacity_points(self, *, member: int = 0) -> dict[int, float]:
        """Return the fitted capacity of each season-year, in points of effective capacity.

        Args:
            member: Which fit of the batch.

        Returns:
            A map from season-year to the sum of increments up to it; empty for a pinned fit.
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
    index = training_rows(
        features=cache.sites[site],
        cache=cache,
        cut=cut,
        window=window,
        keep_lockdown=spec.covid_flag,
    )
    if len(index) == 0:
        msg = f"{site} has no training rows before {cut}"
        raise ValueError(msg)
    assert_no_lookahead(cache=cache, index=index, cut=cut)
    earliest = cache.end_times[index[0]]
    anomaly_mean = (
        month_half_hour_mean(cache=cache, features=cache.sites[site], index=index)
        if spec.solar == "anomaly"
        else None
    )
    layout = make_layout(
        spec=spec,
        season_years=season_years,
        cut=cut,
        earliest=earliest,
        anomaly_mean=anomaly_mean,
    )
    return layout, index


def check_pinned_capacity(
    *,
    capacity: CapacityType,
    pinned: PinnedCapacity | None,
    site: str,
    features: SiteFeatures,
    cut: datetime,
) -> None:
    """Raise unless a pinned capacity is given exactly when the capacity is pinned, for this fit.

    Args:
        capacity: The spec's capacity mode.
        pinned: The pinned capacity given to the fit, or `None`.
        site: The substation's label.
        features: The substation's arrays.
        cut: The fit's cut.

    Raises:
        ValueError: If the pinned capacity is missing, unexpected, or built for another
            substation or cut.
    """
    if (pinned is not None) != (capacity == "pinned"):
        msg = "a pinned capacity is required exactly when the specs pin the capacity"
        raise ValueError(msg)
    if pinned is not None and (pinned.cut != cut or pinned.site != features.table.site):
        msg = (
            f"the pinned capacity was built for {pinned.site} at {pinned.cut}, not {site} at {cut}"
        )
        raise ValueError(msg)


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
    pinned: PinnedCapacity | None = None,
) -> GamFit:
    """Fit the convex model of one substation, for one or more penalty sets or weightings.

    All `specs` must share a drift form, a capacity mode, a COVID flag, a final-knot mode, a solar
    form, the harmonics and heating options, a clip, and a node block length. The
    Gram matrix is built once and each batch member adds its own penalty, so a grid of penalties
    costs one pass over the rows. Give either several `specs` and no `weights`, or one spec and
    `weights`.

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
        pinned: The pinned capacity built for `cut`; required exactly when the specs pin the
            capacity.

    Returns:
        The fitted batch.

    Raises:
        ValueError: If the specs differ in any of the structure fields above, the batch sizes do
            not match, the pinned capacity is missing, unexpected or built for another cut, or no
            row can train.
    """
    features = cache.sites[site]
    structure = {
        (
            spec.drift,
            spec.capacity,
            spec.covid_flag,
            spec.final_knot,
            spec.solar,
            spec.shared_harmonics,
            spec.slow_heating_only,
            spec.clip,
            None if spec.node is None else spec.node.step_days,
        )
        for spec in specs
    }
    if len(structure) != 1:
        msg = (
            "all specs of one call must share a drift form, capacity mode, COVID flag, final knot,"
            " solar form, harmonics and heating options, clip, and node presence"
        )
        raise ValueError(msg)
    check_pinned_capacity(
        capacity=specs[0].capacity, pinned=pinned, site=site, features=features, cut=cut
    )
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
        pinned=pinned,
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
    if any(spec.orthogonal_penalty > 0.0 for spec in specs):
        penalty = penalty + _orthogonal_term(
            specs=specs,
            gram=gram,
            matrix=orthogonal_penalty_matrix(
                cache=cache, features=features, layout=layout, index=index, weights=weights
            ),
            dtype=dtype,
            device=device,
        )
    theta, violation = solve_bounded_quadratic(
        matrix=gram + penalty,
        rhs=rhs - 0.5 * linear,
        constrained=torch.as_tensor(layout.constrained),
        max_iterations=(
            NODE_MAX_ITERATIONS if specs[0].node is not None else DEFAULT_MAX_ITERATIONS
        ),
    )
    return GamFit(
        layout=layout,
        theta=theta.cpu().numpy(),
        cut=cut,
        n_rows=len(index),
        site=site,
        last_season_year=last_signal_season_year(
            cache=cache, site=site, index=index, cut=cut, pinned=pinned
        ),
        violation=violation.cpu().numpy(),
        pinned=pinned,
    )


def last_signal_season_year(
    *,
    cache: FeatureCache,
    site: str,
    index: np.ndarray,
    cut: datetime,
    pinned: PinnedCapacity | None,
) -> int:
    """Return the latest season-year with a training row that carries solar signal.

    Args:
        cache: The cache.
        site: The substation's label.
        index: The training rows.
        cut: The fit's cut, for the error message.
        pinned: The pinned capacity of the fit, or `None` for a fitted capacity.

    Returns:
        The season-year, or -1 for a pinned fit with no signal row.

    Raises:
        ValueError: If a fitted-capacity fit has no training row that carries solar signal.
    """
    signal = derated_unit_shape(features=cache.sites[site], index=index) > LAST_TRAINED_MIN_SHAPE
    if not signal.any() and pinned is None:
        msg = f"{site} has no training row that carries solar signal before {cut}"
        raise ValueError(msg)
    return int(cache.calendar.season_year[index[signal]].max()) if signal.any() else -1


def predict_gam(
    *,
    cache: FeatureCache,
    fit: GamFit,
    index: np.ndarray,
    member: int = 0,
    columns: np.ndarray | None = None,
) -> np.ndarray:
    """Predict net power in points at the grid rows `index`.

    The drift is held at its last knot, because the layout's knots end at the cut and the hat
    functions are clipped there. With `final_knot="shrunk"` the held value is the last knot pulled
    towards the mean of the last 12 knots, for rows that end after the fit's cut only; a row at or
    before the cut uses the fitted knot. The capacity of a season-year after
    `fit.last_season_year` is set explicitly to that season-year's capacity, and a pinned fit
    subtracts the pinned capacity times the derated unit shape.

    Args:
        cache: The cache.
        fit: The fitted batch.
        index: The grid rows to predict.
        member: Which fit of the batch.
        columns: A boolean mask over the layout's columns. If given, only those columns
            contribute, and a pinned capacity's solar output is not subtracted, so the prediction
            is the sum of the chosen terms (for example the demand terms alone).

    Returns:
        The predictions, float64, one per row.

    Raises:
        ValueError: If a row to predict has no finite input.
    """
    assert_before_holdout(cache=cache, index=index)
    features = cache.sites[fit.site]
    layout = fit.layout
    out = np.empty(len(index))
    theta = _forecast_theta(fit=fit, member=member)
    free_theta = fit.theta[member]
    if columns is not None:
        theta = np.where(columns, theta, 0.0)
        free_theta = np.where(columns, free_theta, 0.0)
    after_cut = cache.end_times > naive_utc(moment=fit.cut)
    for chunk in row_chunks(index=index):
        rows = index[chunk]
        x = design_matrix(cache=cache, features=features, layout=layout, index=rows)
        capacity = layout.slices["capacity"]
        held_year = np.minimum(cache.calendar.season_year[rows], fit.last_season_year)
        held = held_year[:, None] >= np.array(layout.season_years)[None, :]
        x[:, capacity] = (
            -solar_shape(cache=cache, features=features, layout=layout, index=rows)[:, None] * held
        )
        if not np.isfinite(x).all():
            msg = f"{fit.site} has a non-finite input among the rows to predict"
            raise ValueError(msg)
        out[chunk] = np.where(after_cut[rows], x @ theta, x @ free_theta)
        if fit.pinned is not None and columns is None:
            out[chunk] -= pinned_solar_points(
                cache=cache, features=features, pinned=fit.pinned, index=rows, clip=layout.spec.clip
            )
    return out


def _forecast_theta(*, fit: GamFit, member: int) -> np.ndarray:
    """Return the coefficients a forecast uses: the fit's, with a shrunk final drift knot.

    Args:
        fit: The fitted batch.
        member: Which fit of the batch.

    Returns:
        A copy of the member's coefficients in which, for `final_knot="shrunk"`, each band's last
        knot has moved `final_knot_shrink` of the way towards the mean of that band's last
        `SHRINK_KNOTS` knots.
    """
    theta = fit.theta[member].copy()
    spec = fit.layout.spec
    if "drift" not in fit.layout.slices or spec.final_knot != "shrunk":
        return theta
    n_knots = len(fit.layout.knot_days)
    start = fit.layout.slices["drift"].start
    for band in range(N_BANDS):
        block = theta[start + band * n_knots : start + (band + 1) * n_knots]
        block[-1] += spec.final_knot_shrink * (block[-SHRINK_KNOTS:].mean() - block[-1])
    return theta


def orthogonal_penalty_matrix(
    *,
    cache: FeatureCache,
    features: SiteFeatures,
    layout: GamLayout,
    index: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return `sum cov(term, u)^2` as a quadratic form in the coefficients.

    For each of `ORTHOGONAL_BLOCKS` and each calendar quarter by local half-hour cell of the
    daylight training rows (derated unit shape above `LAST_TRAINED_MIN_SHAPE`), the covariance of
    the block's contribution `x_block theta` with the derated unit shape `u` is linear in theta:
    `c theta`, with `c = mean(u x_block) - mean(u) mean(x_block)` over the cell's rows. The matrix
    is the sum of `c^T c` over blocks and cells, so the penalty keeps the fit convex. A cell with
    fewer than `MIN_ORTHOGONAL_CELL_ROWS` rows (a total weight under that, for weighted rows) is
    left out. Only the training rows `index` contribute.

    Args:
        cache: The cache.
        features: The substation's arrays.
        layout: The columns.
        index: The training rows.
        weights: Row weights of shape (batch, len(index)), such as month-block bootstrap counts, so
            that each weighting gets the covariance of its own resample; `None` for unit weights.

    Returns:
        A positive semi-definite matrix of shape (p, p), unscaled, or of shape (batch, p, p) when
        `weights` is given.
    """
    p = layout.n_columns
    shape = derated_unit_shape(features=features, index=index)
    daylight = shape > LAST_TRAINED_MIN_SHAPE
    rows = index[daylight]
    unit = shape[daylight]
    row_weights = np.ones((1, len(rows))) if weights is None else np.asarray(weights)[:, daylight]
    batch = row_weights.shape[0]
    n_cells = 4 * HALF_HOURS_PER_DAY
    cell = calendar_cell(cache=cache, index=rows, kind="quarter")
    counts = np.stack([np.bincount(cell, weights=w, minlength=n_cells) for w in row_weights])
    unit_sums = np.stack(
        [np.bincount(cell, weights=w * unit, minlength=n_cells) for w in row_weights]
    )
    unit_times_x = np.zeros((batch, n_cells, p))
    x_sums = np.zeros((batch, n_cells, p))
    for chunk in row_chunks(index=rows):
        chunk_rows = rows[chunk]
        x = design_matrix(cache=cache, features=features, layout=layout, index=chunk_rows)
        one_hot = (cell[chunk, None] == np.arange(n_cells)[None, :]).astype(np.float64)
        for member in range(batch):
            weighted = one_hot * row_weights[member, chunk, None]
            unit_times_x[member] += weighted.T @ (unit[chunk, None] * x)
            x_sums[member] += weighted.T @ x
    safe = np.maximum(counts, 1e-12)[:, :, None]
    covariance = unit_times_x / safe - (unit_sums[:, :, None] / safe) * (x_sums / safe)
    covariance[counts < MIN_ORTHOGONAL_CELL_ROWS] = 0.0
    matrix = np.zeros((batch, p, p))
    for name in ORTHOGONAL_BLOCKS:
        block = layout.slices[name]
        c = covariance[:, :, block]
        matrix[:, block, block] += np.einsum("bcp,bcq->bpq", c, c)
    return matrix[0] if weights is None else matrix


def _orthogonal_term(
    *,
    specs: Sequence[GamSpec],
    gram: torch.Tensor,
    matrix: np.ndarray,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Return each batch member's scaled orthogonal penalty.

    The weight is `lambda * mean(diag(gram)) / mean(diag(matrix))`, so lambda is in units of the
    Gram matrix's mean diagonal over the penalty matrix's mean diagonal.

    Args:
        specs: The specs, whose `orthogonal_penalty` values are the lambdas; one spec is shared
            across the whole batch.
        gram: The Gram matrices, shape (batch, p, p).
        matrix: The unscaled penalty matrix of `orthogonal_penalty_matrix`, shape (p, p) shared by
            the batch or (batch, p, p) with one per member.
        dtype: The floating-point type.
        device: Where the arithmetic runs.

    Returns:
        The penalty of each batch member, shape (batch, p, p).

    Raises:
        ValueError: If a matrix is all zero.
    """
    batch = gram.shape[0]
    matrices = torch.as_tensor(
        np.array(np.broadcast_to(matrix, (batch, *matrix.shape[-2:]))), dtype=dtype, device=device
    )
    mean_diagonal = torch.diagonal(matrices, dim1=-2, dim2=-1).mean(dim=-1)
    if bool((mean_diagonal <= 0.0).any()):
        msg = "the orthogonal penalty matrix is zero: no daylight cell has enough rows"
        raise ValueError(msg)
    weights = torch.as_tensor(
        [spec.orthogonal_penalty for spec in specs], dtype=dtype, device=device
    ).expand(batch)
    gram_diagonal = torch.diagonal(gram, dim1=-2, dim2=-1).mean(dim=-1)
    scale = weights * gram_diagonal / mean_diagonal
    return scale[:, None, None] * matrices


def generator_evidence(*, path: np.ndarray) -> float:
    """Return the depth of the deepest outage in a node's level path, in points.

    A generator is evidenced only by an outage, a dip that is higher on both sides: its level is
    aliased with the profile everywhere else. For each block this is the lower of the highest level
    before it and the highest level after it, minus its own level, and the result is the largest
    such depth. A single unpaired step, such as a fall in demand read as a generator starting, has
    no higher level on one side and gives zero, however large the step is.

    Args:
        path: The node's level path, in points.

    Returns:
        The deepest dip, zero or more.
    """
    before = np.maximum.accumulate(path)
    after = np.maximum.accumulate(path[::-1])[::-1]
    return float(np.max(np.minimum(before, after) - path))


def generator_path(*, fit: GamFit, member: int = 0) -> np.ndarray:
    """Return the node's level path, one value per block of `step_days` days, in points.

    Args:
        fit: A fit whose spec has a node.
        member: Which fit of the batch.

    Returns:
        The level of each block from the node path's first day, as the fit holds it.

    Raises:
        ValueError: If the fit has no node, or a level is below zero.
    """
    if "node_level" not in fit.layout.slices:
        msg = "the fit has no unmetered generator node"
        raise ValueError(msg)
    path = fit.theta[member, fit.layout.slices["node_level"]]
    if path.min() < -NODE_NEGATIVE_TOLERANCE:
        msg = f"the generator level of {fit.site} is {path.min():.3g} points: below zero"
        raise ValueError(msg)
    return path


@dataclass(frozen=True)
class ClipProfile:
    """The profile of the export clip's capacity multiplier and ceiling.

    Attributes:
        multipliers: The grid of the capacity multiplier m, shape (M,).
        shares: The grid of the ceiling as a share of m times the held capacity, shape (S,).
        objective: The penalised training objective of each (m, share), shape (M, S).
        open_objective: The same objective with no clip.
        best: The (m, share) with the lowest objective; a tie goes to the larger share.
    """

    multipliers: np.ndarray
    shares: np.ndarray
    objective: np.ndarray
    open_objective: float
    best: ClipSpec


def profile_clip(
    *,
    cache: FeatureCache,
    site: str,
    spec: GamSpec,
    cut: datetime,
    window: TrainingWindow,
    season_years: Sequence[int],
    pinned: PinnedCapacity,
    multipliers: Sequence[float] = CLIP_MULTIPLIERS,
    shares: Sequence[float] = CLIP_SHARES,
    beta: float = CLIP_BETA_PER_POINT,
) -> ClipProfile:
    """Fit the pinned-capacity model once for each (multiplier, ceiling) of a grid.

    For given `(m, P_max)` the clipped solar output is a fixed column that moves to the target, so
    the design and Gram matrix are shared and only the right-hand side differs: one pass over the
    training rows builds `X^T y` and `X^T pv` for every grid point, and one batched solve fits them
    all. The penalised objective `sum (y + pv - X theta)^2 + theta^T P theta + c^T theta` is
    compared across grid points (and against the open model with no clip), which is a
    likelihood-style comparison because the target of each fit is `y + pv`.

    Args:
        cache: The cache.
        site: The substation's label.
        spec: A spec with a pinned capacity and no clip.
        cut: The cut; training rows end no later.
        window: The window rule.
        season_years: The season-years (unused columns for a pinned capacity).
        pinned: The pinned capacity built for `cut`.
        multipliers: The grid of m.
        shares: The grid of the ceiling share.
        beta: The clip's sharpness, per point.

    Returns:
        The profile.

    Raises:
        ValueError: If the spec has a clip or does not pin the capacity, or the pinned capacity
            was built for another site or cut.
    """
    features = cache.sites[site]
    if spec.clip is not None or spec.capacity != "pinned":
        msg = "profile_clip needs a spec with a pinned capacity and no clip"
        raise ValueError(msg)
    if pinned.cut != cut or pinned.site != features.table.site:
        msg = (
            f"the pinned capacity was built for {pinned.site} at {pinned.cut}, not {site} at {cut}"
        )
        raise ValueError(msg)
    layout, index = prepare_fit(
        cache=cache, site=site, spec=spec, cut=cut, window=window, season_years=season_years
    )
    grid = [
        ClipSpec(multiplier=m, p_max_share=share, beta=beta)
        for m in multipliers
        for share in shares
    ]
    n_models = len(grid) + 1
    p = layout.n_columns
    gram = np.zeros((p, p))
    x_y = np.zeros(p)
    x_pv = np.zeros((p, n_models))
    target_squares = np.zeros(n_models)
    for chunk in row_chunks(index=index):
        rows = index[chunk]
        x = design_matrix(cache=cache, features=features, layout=layout, index=rows)
        y = features.y[rows]
        pv = np.stack(
            [
                pinned_solar_points(
                    cache=cache, features=features, pinned=pinned, index=rows, clip=clip
                )
                for clip in [*grid, None]
            ],
            axis=1,
        )
        gram += x.T @ x
        x_y += x.T @ y
        x_pv += x.T @ pv
        target_squares += np.square(y[:, None] + pv).sum(axis=0)
    matrix, cost = penalty_matrix(layout=layout, n_rows=len(index))
    system = np.broadcast_to(gram + matrix, (n_models, p, p))
    right = x_y[None, :] + x_pv.T
    theta, _ = solve_bounded_quadratic(
        matrix=torch.as_tensor(np.ascontiguousarray(system), dtype=torch.float64),
        rhs=torch.as_tensor(right - 0.5 * cost[None, :], dtype=torch.float64),
        constrained=torch.as_tensor(layout.constrained),
    )
    theta = theta.numpy()
    objective = (
        target_squares
        - 2.0 * np.einsum("bp,bp->b", theta, right)
        + np.einsum("bp,pq,bq->b", theta, gram + matrix, theta)
        + theta @ cost
    )
    surface = objective[:-1].reshape(len(multipliers), len(shares))
    best_m, best_share = _best_cell(surface=surface)
    return ClipProfile(
        multipliers=np.array(multipliers),
        shares=np.array(shares),
        objective=surface,
        open_objective=float(objective[-1]),
        best=ClipSpec(
            multiplier=float(multipliers[best_m]),
            p_max_share=float(shares[best_share]),
            beta=beta,
        ),
    )


def _best_cell(*, surface: np.ndarray) -> tuple[int, int]:
    """Return the (row, column) of the lowest objective, preferring the larger column on a tie."""
    lowest = surface.min()
    close = np.flatnonzero(surface.ravel() <= lowest + 1e-9 * max(abs(lowest), 1.0))
    rows, columns = np.unravel_index(close, surface.shape)
    best = np.lexsort((rows, columns))[-1]
    return int(rows[best]), int(columns[best])


@dataclass(frozen=True)
class ResidualSpread:
    """Empirical quantiles of residuals, per hour band and calendar quarter.

    **Centred or not.** The residuals of a replica fitted 12 months earlier carry that replica's
    mean bias. Added raw to the scored year's mean forecast, they shift every quantile by last
    year's bias. A centred spread subtracts each (band, quarter) group's median from its quantiles,
    so the median of the spread is zero and only the shape of the error around the forecast is
    kept. Centring on the median keeps the forecast's mean-to-median skew out of the spread, whereas
    the onboarding spread (`studies.dp_onboarding`) centres its residuals on the mean before calling
    `fit` with `centre=False`. The uncentred spread stays available (`centre=False`) so that the
    bias can be reported as its own result.

    Attributes:
        table: Shape (4 bands, 4 quarters, 11 levels).
    """

    table: np.ndarray

    @staticmethod
    def fit(
        *, residuals: np.ndarray, band: np.ndarray, quarter: np.ndarray, centre: bool
    ) -> ResidualSpread:
        """Fit the quantiles of `residuals` (actual minus predicted mean).

        A (band, quarter) group with fewer than `MIN_SPREAD_GROUP_ROWS` residuals uses its band's
        residuals, and a band with fewer uses all residuals. The fallback is applied before the
        centring, so every group of a centred spread has median zero.

        Args:
            residuals: Replica residuals in points.
            band: The hour band of each residual.
            quarter: The calendar quarter of each residual.
            centre: True to subtract each group's median from its quantiles; False to keep the raw
                quantiles, which carry the replica's mean bias.

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
        if centre:
            table -= table[:, :, [QUANTILE_LEVELS.index(0.5)]]
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
        switching: True where a residual row sits inside the hand-filtered switching masks; all
            false unless the replica kept those rows.
    """

    residuals: np.ndarray
    band: np.ndarray
    quarter: np.ndarray
    index: np.ndarray
    fit_cut: datetime
    switching: np.ndarray


def replica_cut(*, cut: datetime) -> datetime:
    """Return the cut of the replica fit: `REPLICA_MONTHS` months before the final cut."""
    return _months_before(moment=cut, months=REPLICA_MONTHS)


def replica_rows(
    *, cache: FeatureCache, site: str, fit_cut: datetime, cut: datetime, drop_switching: bool = True
) -> np.ndarray:
    """Return the rows a replica fit's residuals are taken over.

    Args:
        cache: The cache.
        site: The substation's label.
        fit_cut: The replica fit's cut.
        cut: The final fit's cut.
        drop_switching: True to leave out the rows inside the switching masks.

    Returns:
        The rows that have a reading and finite inputs, end after the replica's cut and after the
        substation's first usable instant, and end no later than the final cut.
    """
    features = cache.sites[site]
    keep = (
        features.present
        & (~features.switching_mask | (not drop_switching))
        & np.isfinite(features.y)
        & np.isfinite(features.pv_unit_shape)
        & np.isfinite(features.smoothed_temperature_c).all(axis=1)
        & (cache.end_times > naive_utc(moment=fit_cut))
        & (cache.end_times > naive_utc(moment=features.table.first_usable))
        & (cache.end_times <= naive_utc(moment=cut))
    )
    return np.flatnonzero(keep)


def replica_residuals(
    *,
    cache: FeatureCache,
    site: str,
    spec: GamSpec,
    cut: datetime,
    window: TrainingWindow,
    season_years: Sequence[int],
    capacity_vintages: pl.DataFrame | None = None,
    drop_switching: bool = True,
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
        capacity_vintages: The real-time capacity vintages, from which the replica's pinned
            capacity at its own cut is built; required when the spec pins the capacity.
        drop_switching: True to leave out the rows inside the switching masks, as a spread does;
            False to keep them and flag them, as the recent-level filter needs, since the filter
            has no mask.

    Returns:
        The residuals over the rows that have a reading and end after the substation's first usable
        instant, outside the switching masks unless `drop_switching` is False.

    Raises:
        ValueError: If the spec pins the capacity and no vintages are given.
    """
    features = cache.sites[site]
    fit_cut = replica_cut(cut=cut)
    pinned = None
    if spec.capacity == "pinned":
        if capacity_vintages is None:
            msg = "a replica of a pinned-capacity arm needs the capacity vintages"
            raise ValueError(msg)
        pinned = pinned_capacity(vintages=capacity_vintages, table=features.table, cut=fit_cut)
    fit = fit_gam(
        cache=cache,
        site=site,
        specs=[spec],
        cut=fit_cut,
        window=window,
        season_years=season_years,
        pinned=pinned,
    )
    index = replica_rows(
        cache=cache, site=site, fit_cut=fit_cut, cut=cut, drop_switching=drop_switching
    )
    predicted = predict_gam(cache=cache, fit=fit, index=index)
    return ReplicaResiduals(
        residuals=features.y[index] - predicted,
        band=cache.calendar.band[index],
        quarter=cache.calendar.quarter[index],
        index=index,
        fit_cut=fit_cut,
        switching=features.switching_mask[index],
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
