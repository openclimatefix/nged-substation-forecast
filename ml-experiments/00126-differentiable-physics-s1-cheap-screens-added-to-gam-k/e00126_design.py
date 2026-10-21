"""The designs of experiment 126: T-all's columns plus each screen's extra columns.

**T-all is experiment 114's interaction model at its tuned penalty weight**: B0 (`studies.dp_gam`
with GAM-K's settings) with every heating column split by day type and half of its hour band, and
every lighting column split by day type. Each arm of this experiment is T-all with one change:

- **A-mono, A-free** add a temperature curve on the 24-hour smoothed temperature in each of 4 hour
  bands: monotone ramps with non-negative coefficients, or a natural cubic spline with a
  difference penalty.
- **S-ewma, S-mm** add heating hinges (15.5 C) on exponentially weighted temperatures, or on the
  previous 24 hours' minimum and maximum temperature and their 7-day means.
- **C-wind, C-hw** compute T-all's heating hinge on a wind-chilled or an apparent temperature.
- **D-sch** adds 48 half-hour deviations on school-holiday weekdays.
- **E-sol** adds B-spline functions of the sun's elevation in the evening, weekday and weekend
  apart.
- **W-basis** replaces the day-of-year harmonics by a basis learned from the training years.
- **W-mix** multiplies the heating block by w(T) and adds a second copy multiplied by 1 - w(T).
- **W-stn** adds the previous day's gap between a weather station and ERA5.

Every arm is a penalised least-squares problem with sign bounds, solved on the Gram matrix by
`studies.dp_gam.solve_bounded_quadratic`.
"""

from dataclasses import dataclass
from datetime import datetime
from itertools import product
from typing import Final, Literal

import e00126_aux as aux_module
import e00126_common as common
import numpy as np
import polars as pl
import torch
from scipy.interpolate import BSpline
from studies.dp_calendar import SMOOTHING_HOURS
from studies.dp_features import FeatureCache
from studies.dp_gam import (
    HEATING_SCALE_C,
    N_BANDS,
    SITE_YEAR_ROWS,
    GamLayout,
    design_matrix,
    penalty_matrix,
    pinned_solar_points,
    prepare_fit,
    solve_bounded_quadratic,
)
from studies.dp_pinned_capacity import PinnedCapacity, pinned_capacity

FactorType = Literal["daytype", "subband", "parity_halfhour", "parity_day"]
"""How rows are split into two groups: weekday against other days, the first against the second
half of the hour band, or (as placebos with no time-of-day or weekly structure) even against odd
local half-hours of the day, or even against odd days of the year."""
ExtraType = Literal[
    "a_mono",
    "a_free",
    "s_ewma",
    "s_mm",
    "c_wind",
    "c_hw",
    "d_sch",
    "e_sol",
    "w_basis",
    "w_mix",
    "w_stn",
]
"""The change an arm makes to T-all."""
EXTRA_TYPES: Final[tuple[ExtraType, ...]] = (
    "a_mono",
    "a_free",
    "s_ewma",
    "s_mm",
    "c_wind",
    "c_hw",
    "d_sch",
    "e_sol",
    "w_basis",
    "w_mix",
    "w_stn",
)
SUBBAND_SECOND_HALF_START: Final[tuple[int, ...]] = (6, 17, 27, 40)
"""The local half-hour of day (0 to 47) at which each hour band's second half starts."""
N_HEATING_COLUMNS: Final[int] = len(SMOOTHING_HOURS) * N_BANDS
CURVE_KNOTS_C: Final[tuple[float, ...]] = (2.0, 6.0, 10.0, 14.0, 18.0, 22.0)
"""The knots of arm A's temperature curve."""
CURVE_FUNCTIONS: Final[int] = len(CURVE_KNOTS_C) - 1
"""Five ramps (A-mono) or five spline functions (A-free) per hour band."""
CURVE_SMOOTHING_COLUMN: Final[int] = 1
"""The 24-hour smoothed temperature, the second of `SMOOTHING_HOURS`."""
CURVE_DIFFERENCE_WEIGHT: Final[float] = 1.0
"""A-free's difference penalty weight, in units of `n_rows / SITE_YEAR_ROWS`."""
ELEVATION_RANGE_DEG: Final[tuple[float, float]] = (-12.0, 10.0)
ELEVATION_KNOTS: Final[tuple[float, ...]] = (-12.0,) * 4 + (-1.0,) + (10.0,) * 4
"""Clamped cubic B-spline knots: 5 functions over the elevation range."""
EVENING_HALF_HOURS: Final[tuple[int, int]] = (30, 41)
"""The local half-hours that start between 15:00 and 21:00 (both ends included)."""
MIX_CENTRE_C: Final[float] = 8.0
MIX_SCALE_C: Final[float] = 3.0
"""W-mix's logistic weight of the cold regime: 1 / (1 + exp((T24 - 8) / 3))."""
BASIS_COLUMNS: Final[int] = 4
"""W-basis keeps T-all's four day-of-year harmonic columns."""
BASIS_SMOOTHING_DAYS: Final[int] = 15
MIN_PROFILE_DAYS: Final[int] = 250


@dataclass(frozen=True)
class ArmSpec:
    """One arm: T-all's splits at a penalty weight, plus one extra change.

    Attributes:
        name: The arm's name.
        heating_factors: The factors every heating column splits by.
        lighting_factors: The factors every lighting column splits by.
        penalty_weight: The difference penalty's weight `lambda` of the splits.
        extra: The screen's change, or `None`.
    """

    name: str
    heating_factors: tuple[FactorType, ...] = ()
    lighting_factors: tuple[FactorType, ...] = ()
    penalty_weight: float = 0.0
    extra: ExtraType | None = None


B0: Final[ArmSpec] = ArmSpec(name="b0")
"""GAM-K's convex model on past weather."""


def interaction_arm(*, name: str, penalty_weight: float) -> ArmSpec:
    """Return one of the interaction arms `t_all` or the parity placebos, at a penalty weight.

    Args:
        name: `t_all`, `p_halfhour`, or `p_day`.
        penalty_weight: The difference penalty's weight.

    Returns:
        The arm.

    Raises:
        ValueError: If the name is not one of the three.
    """
    heating: dict[str, tuple[FactorType, ...]] = {
        "t_all": ("daytype", "subband"),
        "p_halfhour": ("parity_halfhour",),
        "p_day": ("parity_day",),
    }
    if name not in heating:
        msg = f"unknown interaction arm {name!r}"
        raise ValueError(msg)
    lighting: tuple[FactorType, ...] = ("daytype",) if name == "t_all" else ()
    return ArmSpec(
        name=name,
        heating_factors=heating[name],
        lighting_factors=lighting,
        penalty_weight=penalty_weight,
    )


def screen_arm(*, extra: ExtraType, penalty_weight: float) -> ArmSpec:
    """Return T-all at a penalty weight with one screen's change.

    Args:
        extra: The screen.
        penalty_weight: T-all's tuned penalty weight.

    Returns:
        The arm, named after the screen.
    """
    base = interaction_arm(name="t_all", penalty_weight=penalty_weight)
    return ArmSpec(
        name=extra,
        heating_factors=base.heating_factors,
        lighting_factors=base.lighting_factors,
        penalty_weight=penalty_weight,
        extra=extra,
    )


@dataclass(frozen=True)
class BaseFit:
    """B0's inputs at one cut: the layout, the training rows, the design, and the target.

    Attributes:
        site: The substation's label.
        cut: The cut; no training row ends after it.
        layout: B0's columns.
        index: The training rows (grid positions).
        x0: B0's design matrix, shape (n, p0).
        target: Net power plus the pinned solar output, in points.
        pinned: The pinned capacity.
        months: The calendar months the rows span, ascending.
        month_of_row: Each row's position in `months`.
        quarter_of_month: The calendar quarter (0 to 3) of each month.
    """

    site: str
    cut: datetime
    layout: GamLayout
    index: np.ndarray
    x0: np.ndarray
    target: np.ndarray
    pinned: PinnedCapacity | None
    months: np.ndarray
    month_of_row: np.ndarray
    quarter_of_month: np.ndarray

    @property
    def n(self) -> int:
        """Return the number of training rows."""
        return len(self.index)


def build_base(
    *,
    cache: FeatureCache,
    vintages: pl.DataFrame,
    site: str,
    cut: datetime,
    not_before: datetime | None = None,
) -> BaseFit:
    """Build B0's design at one cut.

    Args:
        cache: The feature cache.
        vintages: Experiment 45's real-time capacity vintages.
        site: The substation's label.
        cut: The cut; no training row ends after it.
        not_before: If given, training rows must end after this instant.

    Returns:
        The base fit's inputs.
    """
    spec = common.base_spec()
    layout, index = prepare_fit(
        cache=cache,
        site=site,
        spec=spec,
        cut=cut,
        window=common.base_window(),
        season_years=common.SEASON_YEARS,
    )
    if not_before is not None:
        index = index[cache.end_times[index] > common.to_64(moment=not_before)]
    features = cache.sites[site]
    x0 = design_matrix(cache=cache, features=features, layout=layout, index=index)
    pinned = pinned_capacity(vintages=vintages, table=features.table, cut=cut)
    target = features.y[index] + pinned_solar_points(
        cache=cache, features=features, pinned=pinned, index=index
    )
    month_keys = cache.end_times[index].astype("datetime64[M]")
    months, month_of_row = np.unique(month_keys, return_inverse=True)
    quarter = ((months.astype(np.int64) % 12) // 3).astype(np.int64)
    return BaseFit(
        site=site,
        cut=cut,
        layout=layout,
        index=index,
        x0=x0,
        target=target,
        pinned=pinned,
        months=months,
        month_of_row=month_of_row,
        quarter_of_month=quarter,
    )


@dataclass(frozen=True)
class ArmContext:
    """What an arm needs besides the feature cache to build its extra columns.

    Attributes:
        aux: The substation's extra arrays.
        basis: W-basis's learned basis by local day of year (366 by 4), or `None`.
    """

    aux: aux_module.SiteAux
    basis: np.ndarray | None = None


_AUX: dict[str, aux_module.SiteAux] = {}


def site_aux(*, cache: FeatureCache, site: str) -> aux_module.SiteAux:
    """Return a substation's extra arrays, built once per process."""
    if site not in _AUX:
        _AUX[site] = aux_module.build_site_aux(cache=cache, site=site)
    return _AUX[site]


def factor_group(*, cache: FeatureCache, index: np.ndarray, factor: FactorType) -> np.ndarray:
    """Return each row's group (0 or 1) under a factor.

    Args:
        cache: The feature cache.
        index: Grid rows.
        factor: `daytype` (0 weekday, 1 Saturday, Sunday, or bank holiday), `subband` (0 first
            half of the row's hour band, 1 second half), `parity_halfhour` (the local half-hour of
            day modulo 2), or `parity_day` (the local day of the year modulo 2).

    Returns:
        An int array of 0 and 1.
    """
    calendar = cache.calendar
    if factor == "daytype":
        return (calendar.day_type[index] != 0).astype(np.int64)
    if factor == "parity_halfhour":
        return (calendar.half_hour[index] % 2).astype(np.int64)
    if factor == "parity_day":
        return (calendar.day_of_year[index] % 2).astype(np.int64)
    split = np.array(SUBBAND_SECOND_HALF_START)[calendar.band[index].astype(np.int64)]
    return (calendar.half_hour[index] >= split).astype(np.int64)


def piece_masks(
    *, cache: FeatureCache, index: np.ndarray, factors: tuple[FactorType, ...]
) -> np.ndarray:
    """Return the 0/1 masks of the pieces a column splits into, shape (n, 2^len(factors)).

    Pieces are ordered by `itertools.product` over the factors' groups, so the first factor
    varies slowest.

    Args:
        cache: The feature cache.
        index: Grid rows.
        factors: The factors the column splits by.

    Returns:
        The masks as floats; each row has exactly one 1.
    """
    groups = [factor_group(cache=cache, index=index, factor=f) for f in factors]
    combos = list(product((0, 1), repeat=len(factors)))
    masks = np.ones((len(index), len(combos)))
    for position, combo in enumerate(combos):
        for group, wanted in zip(groups, combo, strict=True):
            masks[:, position] *= group == wanted
    return masks


def _band_hot(*, cache: FeatureCache, index: np.ndarray) -> np.ndarray:
    band = cache.calendar.band[index].astype(np.int64)
    return (band[:, None] == np.arange(N_BANDS)[None, :]).astype(np.float64)


def _by_band(*, functions: np.ndarray, band_hot: np.ndarray) -> np.ndarray:
    """Return each function times each band's indicator, function-major and band-minor."""
    return (functions[:, :, None] * band_hot[:, None, :]).reshape(len(functions), -1)


def heating_hinges(*, smoothed: np.ndarray, band_hot: np.ndarray) -> np.ndarray:
    """Return the 16 heating columns for smoothed temperatures, smoothing-major and band-minor.

    Args:
        smoothed: Four smoothed temperatures per row, shape (n, 4).
        band_hot: The hour bands' indicators, shape (n, 4).

    Returns:
        An array of shape (n, 16).
    """
    hinge = np.clip(common.BASE_THRESHOLD_C - smoothed, 0.0, None) / HEATING_SCALE_C
    return _by_band(functions=hinge, band_hot=band_hot)


def monotone_ramps(*, temperature_c: np.ndarray) -> np.ndarray:
    """Return the five decreasing ramps of arm A-mono, each 1 below its lower knot, shape (n, 5)."""
    knots = np.array(CURVE_KNOTS_C)
    lower, upper = knots[:-1], knots[1:]
    return np.clip((upper[None, :] - temperature_c[:, None]) / (upper - lower)[None, :], 0.0, 1.0)


def natural_cubic_basis(*, temperature_c: np.ndarray) -> np.ndarray:
    """Return the natural cubic spline functions of arm A-free without the constant, shape (n, 5).

    The functions are Hastie, Tibshirani, and Friedman's `N_2` to `N_K` at the six knots, with the
    temperature divided by 10 so the columns are of order one.
    """
    knots = np.array(CURVE_KNOTS_C)
    last = knots[-1]
    t = temperature_c[:, None]

    def cube(value: np.ndarray) -> np.ndarray:
        return np.clip(value, 0.0, None) ** 3

    d = (cube(t - knots[None, :-1]) - cube(t - last)) / (last - knots[None, :-1])
    functions = np.concatenate([t, d[:, :-1] - d[:, -1:]], axis=1)
    return functions / np.array([10.0, 1000.0, 1000.0, 1000.0, 1000.0])[None, :]


def elevation_basis(*, elevation_deg: np.ndarray) -> np.ndarray:
    """Return E-sol's five cubic B-spline functions of elevation, shape (n, 5)."""
    clipped = np.clip(elevation_deg, ELEVATION_RANGE_DEG[0], ELEVATION_RANGE_DEG[1] - 1e-9)
    return BSpline.design_matrix(clipped, np.array(ELEVATION_KNOTS), 3).toarray()


def cold_weight(*, temperature_c: np.ndarray) -> np.ndarray:
    """Return W-mix's weight of the cold regime, 1 below the centre and 0 above it."""
    return 1.0 / (1.0 + np.exp((temperature_c - MIX_CENTRE_C) / MIX_SCALE_C))


def extra_block(  # noqa: PLR0911 (one branch per arm)
    *, cache: FeatureCache, site: str, index: np.ndarray, arm: ArmSpec, context: ArmContext
) -> np.ndarray | None:
    """Return the columns an arm appends after T-all's, or `None` for an arm that appends none."""
    features = cache.sites[site]
    aux = context.aux
    calendar = cache.calendar
    band_hot = _band_hot(cache=cache, index=index)
    if arm.extra == "a_mono":
        ramps = monotone_ramps(temperature_c=features.smoothed_temperature_c[index, 1])
        return _by_band(functions=ramps, band_hot=band_hot)
    if arm.extra == "a_free":
        spline = natural_cubic_basis(temperature_c=features.smoothed_temperature_c[index, 1])
        return _by_band(functions=spline, band_hot=band_hot)
    if arm.extra == "s_ewma":
        hinge = np.clip(common.BASE_THRESHOLD_C - aux.ewma[index], 0.0, None) / HEATING_SCALE_C
        return _by_band(functions=hinge, band_hot=band_hot)
    if arm.extra == "s_mm":
        hinge = np.clip(common.BASE_THRESHOLD_C - aux.extremes[index], 0.0, None) / HEATING_SCALE_C
        return _by_band(functions=hinge, band_hot=band_hot)
    if arm.extra == "d_sch":
        holiday = aux.school_holiday[index].astype(np.float64)
        half_hour = np.eye(48)[calendar.half_hour[index].astype(np.int64)]
        return half_hour * holiday[:, None]
    if arm.extra == "e_sol":
        spline = elevation_basis(elevation_deg=features.elevation_deg[index])
        half_hour = calendar.half_hour[index]
        evening = (half_hour >= EVENING_HALF_HOURS[0]) & (half_hour <= EVENING_HALF_HOURS[1])
        weekday = calendar.day_type[index] == 0
        indicator = np.stack([evening & weekday, evening & ~weekday], axis=1).astype(np.float64)
        return (spline[:, :, None] * indicator[:, None, :]).reshape(len(index), -1)
    if arm.extra == "w_stn":
        gap = np.nan_to_num(aux.station_gap[index], nan=0.0)
        return gap[:, None] * band_hot
    return None


def extra_penalty(*, arm: ArmSpec, ridge: float, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the penalty matrix and the sign bounds of an arm's appended columns.

    Args:
        arm: The arm.
        ridge: The ridge of the extra columns, GAM-K's 0.1.
        n_rows: The number of training rows, which scales A-free's difference penalty.

    Returns:
        The matrix, shape (k, k), and the bounds (True for non-negative), shape (k,); both empty
        for an arm that appends no columns.
    """
    sizes = {
        "a_mono": CURVE_FUNCTIONS * N_BANDS,
        "a_free": CURVE_FUNCTIONS * N_BANDS,
        "s_ewma": 2 * N_BANDS,
        "s_mm": 4 * N_BANDS,
        "d_sch": 48,
        "e_sol": 10,
        "w_stn": N_BANDS,
    }
    k = sizes.get(arm.extra or "", 0)
    matrix = ridge * np.eye(k)
    if arm.extra == "a_free":
        scale = CURVE_DIFFERENCE_WEIGHT * n_rows / SITE_YEAR_ROWS
        for band in range(N_BANDS):
            for f in range(CURVE_FUNCTIONS - 1):
                a, b = f * N_BANDS + band, (f + 1) * N_BANDS + band
                matrix[a, a] += scale
                matrix[b, b] += scale
                matrix[a, b] -= scale
                matrix[b, a] -= scale
    constrained = np.full(k, arm.extra in ("a_mono", "s_ewma", "s_mm"), dtype=bool)
    return matrix, constrained


def replaced_columns(
    *, layout: GamLayout, arm: ArmSpec, mix_columns: tuple[int, ...] = ()
) -> list[tuple[int, tuple[FactorType, ...]]]:
    """Return the base columns an arm splits, in ascending order, each with its factors.

    W-mix's appended second heating block (`mix_columns`) splits like the heating block.
    """
    replaced: list[tuple[int, tuple[FactorType, ...]]] = []
    heating = layout.slices["heating"]
    lighting = layout.slices["lighting"]
    if arm.heating_factors:
        replaced += [(j, arm.heating_factors) for j in range(heating.start, heating.stop)]
        replaced += [(j, arm.heating_factors) for j in mix_columns]
    if arm.lighting_factors:
        replaced += [(j, arm.lighting_factors) for j in range(lighting.start, lighting.stop)]
    return sorted(replaced)


def _modified_base(
    *,
    cache: FeatureCache,
    layout: GamLayout,
    site: str,
    index: np.ndarray,
    x0: np.ndarray,
    arm: ArmSpec,
    context: ArmContext,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Apply the arm's change to B0's columns and return the matrix and W-mix's appended columns."""
    x = x0.copy()
    band_hot = _band_hot(cache=cache, index=index)
    heating = layout.slices["heating"]
    aux = context.aux
    if arm.extra == "c_wind":
        x[:, heating] = heating_hinges(smoothed=aux.wind_chill[index], band_hot=band_hot)
    elif arm.extra == "c_hw":
        x[:, heating] = heating_hinges(smoothed=aux.apparent[index], band_hot=band_hot)
    elif arm.extra == "w_basis":
        if context.basis is None:
            msg = "W-basis needs the learned basis"
            raise ValueError(msg)
        values = context.basis[cache.calendar.day_of_year[index].astype(np.int64) - 1]
        x[:, layout.slices["harmonics"]] = _by_band(functions=values, band_hot=band_hot)
    elif arm.extra == "w_mix":
        weight = cold_weight(temperature_c=cache.sites[site].smoothed_temperature_c[index, 1])
        original = x[:, heating].copy()
        x[:, heating] = original * weight[:, None]
        x = np.hstack([x, original * (1.0 - weight)[:, None]])
        return x, tuple(range(x0.shape[1], x.shape[1]))
    return x, ()


def arm_matrix(
    *,
    cache: FeatureCache,
    layout: GamLayout,
    site: str,
    index: np.ndarray,
    x0: np.ndarray,
    arm: ArmSpec,
    context: ArmContext,
) -> np.ndarray:
    """Return an arm's design matrix at the rows `index`.

    The columns are B0's columns that the arm keeps, in B0's order, then the pieces of each split
    column in ascending column order, then the arm's appended columns.

    Args:
        cache: The feature cache.
        layout: B0's columns.
        site: The substation's label.
        index: Grid rows.
        x0: B0's design matrix at `index`.
        arm: The arm.
        context: The arm's extra arrays.

    Returns:
        The arm's design matrix.
    """
    x, mix_columns = _modified_base(
        cache=cache, layout=layout, site=site, index=index, x0=x0, arm=arm, context=context
    )
    replaced = replaced_columns(layout=layout, arm=arm, mix_columns=mix_columns)
    blocks: list[np.ndarray] = []
    if replaced:
        drop = {column for column, _ in replaced}
        keep = [j for j in range(x.shape[1]) if j not in drop]
        blocks.append(x[:, keep])
        for column, factors in replaced:
            masks = piece_masks(cache=cache, index=index, factors=factors)
            blocks.append(x[:, [column]] * masks)
    else:
        blocks.append(x)
    appended = extra_block(cache=cache, site=site, index=index, arm=arm, context=context)
    if appended is not None:
        blocks.append(appended)
    return np.hstack(blocks)


@dataclass(frozen=True)
class ArmPenalty:
    """An arm's penalty matrix and sign bounds.

    Attributes:
        matrix: The quadratic penalty, shape (p, p).
        constrained: True for the columns that must be non-negative.
        n_new_columns: The number of columns beyond B0's column count.
    """

    matrix: np.ndarray
    constrained: np.ndarray
    n_new_columns: int


def arm_penalty(*, layout: GamLayout, arm: ArmSpec, n_rows: int) -> ArmPenalty:
    """Return an arm's penalty matrix, in the column order of `arm_matrix`.

    Args:
        layout: B0's columns.
        arm: The arm.
        n_rows: The number of training rows, which scales the difference penalties.

    Returns:
        The penalty and the sign bounds.

    Raises:
        ValueError: If a split column carries a penalty that couples it to another column.
    """
    base, _ = penalty_matrix(layout=layout, n_rows=n_rows)
    constrained0 = layout.constrained.copy()
    mix_columns: tuple[int, ...] = ()
    if arm.extra == "w_mix":
        heating = layout.slices["heating"]
        width = heating.stop - heating.start
        p0 = base.shape[0]
        extended = np.zeros((p0 + width, p0 + width))
        extended[:p0, :p0] = base
        extended[p0:, p0:] = layout.spec.ridge * np.eye(width)
        base = extended
        constrained0 = np.concatenate([constrained0, constrained0[heating]])
        mix_columns = tuple(range(p0, p0 + width))
    replaced = replaced_columns(layout=layout, arm=arm, mix_columns=mix_columns)
    drop = {column for column, _ in replaced}
    keep = [j for j in range(base.shape[0]) if j not in drop]
    n_pieces = [2 ** len(factors) for _, factors in replaced]
    size = len(keep) + sum(n_pieces)
    matrix = np.zeros((size, size))
    constrained = np.zeros(size, dtype=bool)
    matrix[: len(keep), : len(keep)] = base[np.ix_(keep, keep)]
    constrained[: len(keep)] = constrained0[keep]
    scale = arm.penalty_weight * n_rows / SITE_YEAR_ROWS
    start = len(keep)
    for (column, factors), pieces in zip(replaced, n_pieces, strict=True):
        if np.count_nonzero(base[column]) != 1 or base[column, column] == 0.0:
            msg = f"column {column} carries a penalty other than a ridge, so it cannot be split"
            raise ValueError(msg)
        combos = list(product((0, 1), repeat=len(factors)))
        for offset in range(pieces):
            matrix[start + offset, start + offset] = base[column, column] / pieces
            constrained[start + offset] = constrained0[column]
        for factor_position in range(len(factors)):
            for combo_a, combo_b in _pairs(combos=combos, position=factor_position):
                a = start + combos.index(combo_a)
                b = start + combos.index(combo_b)
                matrix[a, a] += scale
                matrix[b, b] += scale
                matrix[a, b] -= scale
                matrix[b, a] -= scale
        start += pieces
    appended, appended_bounds = extra_penalty(arm=arm, ridge=layout.spec.ridge, n_rows=n_rows)
    k = len(appended_bounds)
    full = np.zeros((size + k, size + k))
    full[:size, :size] = matrix
    full[size:, size:] = appended
    return ArmPenalty(
        matrix=full,
        constrained=np.concatenate([constrained, appended_bounds]),
        n_new_columns=size + k - layout.n_columns,
    )


def _pairs(
    *, combos: list[tuple[int, ...]], position: int
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Return the pairs of piece combinations that differ only in the factor at `position`."""
    return [
        (combo, (*combo[:position], 1, *combo[position + 1 :]))
        for combo in combos
        if combo[position] == 0
    ]


def solve(*, gram: torch.Tensor, rhs: torch.Tensor, penalty: ArmPenalty) -> np.ndarray:
    """Solve an arm for every weighting of a batch.

    Args:
        gram: `X^T W X`, shape (B, p, p).
        rhs: `X^T W y`, shape (B, p).
        penalty: The arm's penalty and sign bounds.

    Returns:
        The coefficients, shape (B, p).
    """
    matrix = gram + torch.as_tensor(penalty.matrix)
    theta, _ = solve_bounded_quadratic(
        matrix=matrix, rhs=rhs, constrained=torch.as_tensor(penalty.constrained)
    )
    return theta.numpy()


@dataclass(frozen=True)
class ArmFit:
    """A fitted arm at one site and cut.

    Attributes:
        arm: The arm.
        theta: The coefficients in the column order of `arm_matrix`.
        context: The extra arrays the arm was fitted with (and predicts with).
        penalised_loss: The squared error plus the penalty at the solution.
        unpenalised_loss: The squared error alone.
        n_rows: The number of rows the fit used.
    """

    arm: ArmSpec
    theta: np.ndarray
    context: ArmContext
    penalised_loss: float
    unpenalised_loss: float
    n_rows: int


def fit_arm(
    *,
    cache: FeatureCache,
    base: BaseFit,
    arm: ArmSpec,
    context: ArmContext,
    target: np.ndarray | None = None,
) -> ArmFit:
    """Fit an arm on the full training sample of a base fit.

    Rows with a non-finite value in the arm's extra inputs (the first days of the grid, for the
    arms that smooth over a week) are left out of the fit.

    Args:
        cache: The feature cache.
        base: B0's inputs at the cut.
        arm: The arm.
        context: The arm's extra arrays.
        target: A replacement target (a synthetic one), or `None` for the real target.

    Returns:
        The fit.
    """
    x = arm_matrix(
        cache=cache,
        layout=base.layout,
        site=base.site,
        index=base.index,
        x0=base.x0,
        arm=arm,
        context=context,
    )
    y = base.target if target is None else target
    ok = np.isfinite(x).all(axis=1)
    x, y = x[ok], y[ok]
    penalty = arm_penalty(layout=base.layout, arm=arm, n_rows=len(y))
    xt = torch.as_tensor(x)
    gram = (xt.T @ xt)[None]
    rhs = torch.as_tensor(x.T @ y)[None]
    theta = solve(gram=gram, rhs=rhs, penalty=penalty)[0]
    residual = y - x @ theta
    unpenalised = float(residual @ residual)
    return ArmFit(
        arm=arm,
        theta=theta,
        context=context,
        penalised_loss=unpenalised + float(theta @ penalty.matrix @ theta),
        unpenalised_loss=unpenalised,
        n_rows=len(y),
    )


def predict_arm(
    *, cache: FeatureCache, base: BaseFit, fit: ArmFit, index: np.ndarray
) -> np.ndarray:
    """Predict net power in points at the grid rows `index`.

    The prediction is the arm's design times its coefficients, minus the pinned solar output.

    Args:
        cache: The feature cache.
        base: B0's inputs at the cut the fit was made at.
        fit: The fitted arm.
        index: Grid rows to predict.

    Returns:
        The predictions, one per row.

    Raises:
        ValueError: If a row has a non-finite input.
    """
    features = cache.sites[base.site]
    x0 = design_matrix(cache=cache, features=features, layout=base.layout, index=index)
    x = arm_matrix(
        cache=cache,
        layout=base.layout,
        site=base.site,
        index=index,
        x0=x0,
        arm=fit.arm,
        context=fit.context,
    )
    if not np.isfinite(x).all():
        msg = f"{base.site} has a non-finite input among the rows to predict"
        raise ValueError(msg)
    out = x @ fit.theta
    if base.pinned is not None:
        out = out - pinned_solar_points(
            cache=cache, features=features, pinned=base.pinned, index=index
        )
    return out


_BASES: dict[datetime, np.ndarray] = {}


def _circular_smooth(*, values: np.ndarray) -> np.ndarray:
    """Return a circular moving average over `BASIS_SMOOTHING_DAYS` days."""
    half = BASIS_SMOOTHING_DAYS // 2
    padded = np.concatenate([values[-half:], values, values[:half]])
    kernel = np.ones(BASIS_SMOOTHING_DAYS) / BASIS_SMOOTHING_DAYS
    return np.convolve(padded, kernel, mode="valid")


def learned_basis(*, cache: FeatureCache, vintages: pl.DataFrame, cut: datetime) -> np.ndarray:
    """Return W-basis's four columns by local day of year, learned from the rows before `cut`.

    For each decision substation, B0 is fitted with its day-of-year harmonic columns set to zero,
    the residual is averaged by day of year within each season-year, interpolated across missing
    days, smoothed over 15 days around the year, and centred. The leading four left singular vectors
    of the matrix of all substations' season-year profiles, scaled to the standard deviation of a
    sine, are the basis. No row ending after `cut` is read.

    Args:
        cache: The feature cache.
        vintages: Experiment 45's real-time capacity vintages.
        cut: The fit's cut.

    Returns:
        An array of shape (366, 4), indexed by local day of year minus one.

    Raises:
        ValueError: If the training rows hold fewer than four season-year profiles.
    """
    if cut in _BASES:
        return _BASES[cut]
    profiles: list[np.ndarray] = []
    for site in common.DECISION_SITES:
        try:
            base = build_base(cache=cache, vintages=vintages, site=site, cut=cut)
        except ValueError:
            continue
        x = base.x0.copy()
        x[:, base.layout.slices["harmonics"]] = 0.0
        matrix, _ = penalty_matrix(layout=base.layout, n_rows=base.n)
        penalty = ArmPenalty(matrix=matrix, constrained=base.layout.constrained, n_new_columns=0)
        xt = torch.as_tensor(x)
        theta = solve(
            gram=(xt.T @ xt)[None], rhs=torch.as_tensor(x.T @ base.target)[None], penalty=penalty
        )[0]
        residual = base.target - x @ theta
        day = cache.calendar.day_of_year[base.index].astype(np.int64) - 1
        season = cache.calendar.season_year[base.index]
        for season_year in np.unique(season):
            rows = season == season_year
            counts = np.bincount(day[rows], minlength=366)
            sums = np.bincount(day[rows], weights=residual[rows], minlength=366)
            present = counts > 0
            if present.sum() < MIN_PROFILE_DAYS:
                continue
            mean = np.where(present, sums / np.maximum(counts, 1), np.nan)
            filled = np.interp(np.arange(366), np.flatnonzero(present), mean[present], period=366)
            smooth = _circular_smooth(values=filled)
            profiles.append(smooth - smooth.mean())
    if len(profiles) < BASIS_COLUMNS:
        msg = f"only {len(profiles)} season-year profiles before {cut}; need {BASIS_COLUMNS}"
        raise ValueError(msg)
    left, _, _ = np.linalg.svd(np.stack(profiles, axis=1), full_matrices=False)
    basis = left[:, :BASIS_COLUMNS]
    basis = basis / basis.std(axis=0, keepdims=True) * np.sqrt(0.5)
    signs = np.sign(basis[np.abs(basis).argmax(axis=0), np.arange(BASIS_COLUMNS)])
    _BASES[cut] = basis * signs[None, :]
    return _BASES[cut]
