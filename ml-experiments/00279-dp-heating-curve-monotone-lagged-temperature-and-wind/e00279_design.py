"""The designs of experiment 279: T-all's columns plus each heating-curve arm's changes.

**T-all is experiment 114's interaction model at its tuned penalty weight**: B0 (`studies.dp_gam`
with GAM-K's settings) with every heating column split by day type and half of its hour band, and
every lighting column split by day type. Each arm of this experiment is T-all plus:

- **A1** a smooth monotone curve (decreasing smooth steps with non-negative coefficients) on each
  of the four trailing-mean temperatures, per hour band.
- **A2** A1 plus the same curve on exponentially weighted temperatures and on the previous two
  days' mean temperatures, per hour band.
- **A2w** A2 plus the trailing 24-hour and 72-hour mean wind speed times the 15.5 C hinge of the
  24-hour and 72-hour mean temperature, per hour band.
- **A3, A3 without wind** T-all with its heating columns replaced by a monotone neural network
  (`e00279_nn`) fitted to the residual of T-all's other terms.

Every linear arm is a penalised least-squares problem with sign bounds, solved on the Gram matrix
by `studies.dp_gam.solve_bounded_quadratic`.
"""

from dataclasses import dataclass
from datetime import datetime
from itertools import product
from typing import Any, Final, Literal

import e00279_aux as aux_module
import e00279_common as common
import e00279_nn as nn_module
import numpy as np
import polars as pl
import torch
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
ExtraType = Literal["a1", "a2", "a2w", "a3", "a3nw", "a3b", "a3bnw", "a3c"]
"""The change an arm makes to T-all."""
EXTRA_TYPES: Final[tuple[ExtraType, ...]] = ("a1", "a2", "a2w", "a3", "a3nw", "a3b", "a3bnw", "a3c")
LINEAR_EXTRAS: Final[tuple[str, ...]] = ("a1", "a2", "a2w")
NETWORK_EXTRAS: Final[tuple[str, ...]] = ("a3", "a3nw", "a3b", "a3bnw", "a3c")
WIND_NETWORKS: Final[tuple[str, ...]] = ("a3", "a3b", "a3c")
FULL_INPUT_NETWORKS: Final[tuple[str, ...]] = ("a3b", "a3bnw", "a3c")
BLOCK_VALIDATION_NETWORKS: Final[tuple[str, ...]] = ("a3c",)
"""A3c validates on every fifth calendar month of the training window, not the last 15% of rows."""
"""The post hoc networks, which also read A0's four trailing-mean temperatures and train longer."""
POST_HOC_EPOCHS: Final[int] = 200
POST_HOC_LEARNING_RATE: Final[float] = 1e-2
SUBBAND_SECOND_HALF_START: Final[tuple[int, ...]] = (6, 17, 27, 40)
"""The local half-hour of day (0 to 47) at which each hour band's second half starts."""
N_HEATING_COLUMNS: Final[int] = len(SMOOTHING_HOURS) * N_BANDS
CURVE_KNOTS_C: Final[tuple[float, ...]] = (0.0, 4.0, 8.0, 12.0, 16.0, 20.0)
"""The knots of the smooth monotone curve: each function falls from 1 to 0 between two knots."""
CURVE_FUNCTIONS: Final[int] = len(CURVE_KNOTS_C) - 1
"""Five smooth steps per temperature and hour band."""
N_DAYTYPE_GROUPS: Final[int] = 2
N_SUBBAND_GROUPS: Final[int] = 2


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
    """

    aux: aux_module.SiteAux


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


def smooth_steps(*, temperature_c: np.ndarray) -> np.ndarray:
    """Return the five decreasing smooth steps of the curve, shape (n, 5).

    Step `i` is 1 below knot `i`, 0 above knot `i + 1`, and the cubic smoothstep
    `1 - (3u^2 - 2u^3)` in between, so it is monotone and has no kink.
    """
    knots = np.array(CURVE_KNOTS_C)
    lower, upper = knots[:-1], knots[1:]
    u = np.clip((temperature_c[:, None] - lower[None, :]) / (upper - lower)[None, :], 0.0, 1.0)
    return 1.0 - (3.0 * u**2 - 2.0 * u**3)


def curve_columns(*, temperatures_c: np.ndarray, band_hot: np.ndarray) -> np.ndarray:
    """Return the smooth curve on each temperature column, per hour band."""
    blocks = [
        _by_band(functions=smooth_steps(temperature_c=temperatures_c[:, k]), band_hot=band_hot)
        for k in range(temperatures_c.shape[1])
    ]
    return np.hstack(blocks)


def lag_temperatures(*, features: object, aux: aux_module.SiteAux, index: np.ndarray) -> np.ndarray:
    """Return A2's four lag temperatures: two exponentially weighted, two previous-day means."""
    return np.hstack([aux.ewma[index], aux.previous_days[index]])


def extra_block(
    *, cache: FeatureCache, site: str, index: np.ndarray, arm: ArmSpec, context: ArmContext
) -> np.ndarray | None:
    """Return the columns a linear arm appends after T-all's, or `None` for any other arm."""
    if arm.extra not in LINEAR_EXTRAS:
        return None
    features = cache.sites[site]
    aux = context.aux
    band_hot = _band_hot(cache=cache, index=index)
    smoothed = features.smoothed_temperature_c[index]
    blocks = [curve_columns(temperatures_c=smoothed, band_hot=band_hot)]
    if arm.extra in ("a2", "a2w"):
        lags = lag_temperatures(features=features, aux=aux, index=index)
        blocks.append(curve_columns(temperatures_c=lags, band_hot=band_hot))
    if arm.extra == "a2w":
        wind = aux.wind[index]
        for column in (1, 2):
            hinge = np.clip(common.BASE_THRESHOLD_C - smoothed[:, [column]], 0.0, None)
            blocks.append(
                _by_band(functions=hinge * wind[:, [column - 1]] / 50.0, band_hot=band_hot)
            )
    return np.hstack(blocks)


def extra_penalty(*, arm: ArmSpec, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the ridge penalty and the non-negativity bounds of a linear arm's appended columns."""
    sizes = {
        "a1": N_BANDS * CURVE_FUNCTIONS * len(SMOOTHING_HOURS),
        "a2": N_BANDS * CURVE_FUNCTIONS * (len(SMOOTHING_HOURS) + 4),
        "a2w": N_BANDS * CURVE_FUNCTIONS * (len(SMOOTHING_HOURS) + 4) + 2 * N_BANDS,
    }
    k = sizes.get(arm.extra or "", 0)
    return ridge * np.eye(k), np.ones(k, dtype=bool)


def replaced_columns(
    *, layout: GamLayout, arm: ArmSpec
) -> list[tuple[int, tuple[FactorType, ...]]]:
    """Return the base columns an arm splits, in ascending order, each with its factors."""
    replaced: list[tuple[int, tuple[FactorType, ...]]] = []
    heating = layout.slices["heating"]
    lighting = layout.slices["lighting"]
    if arm.heating_factors:
        replaced += [(j, arm.heating_factors) for j in range(heating.start, heating.stop)]
    if arm.lighting_factors:
        replaced += [(j, arm.lighting_factors) for j in range(lighting.start, lighting.stop)]
    return sorted(replaced)


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
    x = x0
    replaced = replaced_columns(layout=layout, arm=arm)
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
    replaced = replaced_columns(layout=layout, arm=arm)
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
    appended, appended_bounds = extra_penalty(arm=arm, ridge=layout.spec.ridge)
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


def effective_degrees_of_freedom(*, gram: torch.Tensor, penalty: ArmPenalty) -> float:
    """Return `trace((G + P)^-1 G)`, the unconstrained effective degrees of freedom of a fit.

    Args:
        gram: The unweighted Gram matrix, shape (p, p).
        penalty: The arm's penalty.

    Returns:
        The trace, ignoring the sign bounds.
    """
    matrix = gram + torch.as_tensor(penalty.matrix)
    return float(torch.trace(torch.linalg.solve(matrix, gram)))


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
        network: The trained heating networks, for the network arms only.
    """

    arm: ArmSpec
    theta: np.ndarray
    context: ArmContext
    penalised_loss: float
    unpenalised_loss: float
    n_rows: int
    network: nn_module.Fitted | None = None


def heating_piece_columns(*, layout: GamLayout, arm: ArmSpec) -> np.ndarray:
    """Return the positions, in `arm_matrix`'s columns, of T-all's split heating columns."""
    replaced = replaced_columns(layout=layout, arm=arm)
    start = layout.n_columns - len(replaced)
    heating = layout.slices["heating"]
    positions: list[int] = []
    for column, factors in replaced:
        width = 2 ** len(factors)
        if heating.start <= column < heating.stop:
            positions += list(range(start, start + width))
        start += width
    return np.array(positions, dtype=np.int64)


def network_inputs(
    *, cache: FeatureCache, site: str, index: np.ndarray, context: ArmContext, extra: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return the network's input matrix and group of each row at the grid rows `index`.

    The three temperatures are the mean of the last 24 hours and of the two days before that; the
    wind is the 24-hour mean wind speed. The group is the hour band, the day type, and the half of
    the hour band, as T-all splits its heating columns.
    """
    features = cache.sites[site]
    if extra in FULL_INPUT_NETWORKS:
        temperatures = np.column_stack(
            [features.smoothed_temperature_c[index], context.aux.previous_days[index]]
        )
    else:
        temperatures = np.column_stack(
            [features.smoothed_temperature_c[index, 1], context.aux.previous_days[index]]
        )
    wind = context.aux.wind[index, 0] if extra in WIND_NETWORKS else None
    group = (
        cache.calendar.band[index].astype(np.int64) * 4
        + 2 * factor_group(cache=cache, index=index, factor="daytype")
        + factor_group(cache=cache, index=index, factor="subband")
    )
    return nn_module.inputs(temperatures_c=temperatures, wind_m_s=wind), group


def network_training(*, extra: str | None) -> dict[str, Any]:
    """Return a network arm's training settings: the pre-registered or the post hoc ones."""
    if extra in FULL_INPUT_NETWORKS:
        return {
            "distil_epochs": POST_HOC_EPOCHS,
            "epochs": POST_HOC_EPOCHS,
            "learning_rate": POST_HOC_LEARNING_RATE,
        }
    return {}


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
    network = None
    if arm.extra in NETWORK_EXTRAS:
        heating = heating_piece_columns(layout=base.layout, arm=arm)
        own = x[:, heating] @ theta[heating]
        theta[heating] = 0.0
        rows = base.index[ok]
        inputs, group = network_inputs(
            cache=cache, site=base.site, index=rows, context=context, extra=arm.extra
        )
        finite = np.isfinite(inputs).all(axis=1)
        network = nn_module.fit_network(
            x=inputs[finite],
            group=group[finite],
            own_heating=own[finite],
            target=(y - x @ theta)[finite],
            use_wind=arm.extra in WIND_NETWORKS,
            blocks=base.month_of_row[ok][finite]
            if arm.extra in BLOCK_VALIDATION_NETWORKS
            else None,
            **network_training(extra=arm.extra),
        )
    residual = y - x @ theta
    if network is not None:
        assert arm.extra is not None
        inputs, group = network_inputs(
            cache=cache,
            site=base.site,
            index=base.index[ok],
            context=context,
            extra=arm.extra,
        )
        residual = residual - network.predict(x=inputs, group=group)
    unpenalised = float(residual @ residual)
    return ArmFit(
        arm=arm,
        theta=theta,
        context=context,
        penalised_loss=unpenalised + float(theta @ penalty.matrix @ theta),
        unpenalised_loss=unpenalised,
        n_rows=len(y),
        network=network,
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
    if fit.network is not None:
        inputs, group = network_inputs(
            cache=cache,
            site=base.site,
            index=index,
            context=fit.context,
            extra=str(fit.arm.extra),
        )
        if not np.isfinite(inputs).all():
            msg = f"{base.site} has a non-finite network input among the rows to predict"
            raise ValueError(msg)
        out = out + fit.network.predict(x=inputs, group=group)
    if base.pinned is not None:
        out = out - pinned_solar_points(
            cache=cache, features=features, pinned=base.pinned, index=index
        )
    return out
