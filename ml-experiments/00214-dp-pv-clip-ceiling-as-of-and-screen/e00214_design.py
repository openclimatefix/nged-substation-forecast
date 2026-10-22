"""The designs of experiment 114: B0's columns, split interaction columns, and hinge thresholds.

**B0 is `studies.dp_gam`'s model with GAM-K's settings.** Every other arm is B0 with some of the
heating or lighting columns changed:

- **Interaction arms (N3)** split a column into pieces, one per group of rows. A heating column
  splits by day type (weekday against Saturday, Sunday, and bank holiday, factor `daytype`) and by
  half of its hour band (factor `subband`). A lighting column splits by day type. The pieces of one
  column sum to the column, each piece carries the column's ridge divided by the number of pieces,
  and a difference penalty `lambda * (n_rows / SITE_YEAR_ROWS) * (beta_a - beta_b)^2` ties the two
  pieces that differ in one factor. As lambda grows, the arm becomes B0 exactly.
- **Threshold arms (N4)** replace the 16 heating columns by the same hinge at another threshold.

Every arm is a penalised least-squares problem with sign bounds, solved on the Gram matrix by
`studies.dp_gam.solve_bounded_quadratic`.
"""

from dataclasses import dataclass
from datetime import datetime
from itertools import product
from typing import Final, Literal

import e00214_common as common
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

SUBBAND_SECOND_HALF_START: Final[tuple[int, ...]] = (6, 17, 27, 40)
"""The local half-hour of day (0 to 47) at which each hour band's second half starts: 03:00 for
the night, 08:30 for the morning, 13:30 for midday, and 20:00 for the evening."""
N_HEATING_COLUMNS: Final[int] = len(SMOOTHING_HOURS) * N_BANDS


@dataclass(frozen=True)
class ArmSpec:
    """One arm: which columns it splits, how strongly it ties the pieces, and its threshold.

    Attributes:
        name: The arm's name.
        heating_factors: The factors every heating column splits by.
        lighting_factors: The factors every lighting column splits by.
        penalty_weight: The difference penalty's weight `lambda`.
        hinge_threshold_c: The heating threshold in degrees C, or `None` for B0's 15.5.
    """

    name: str
    heating_factors: tuple[FactorType, ...] = ()
    lighting_factors: tuple[FactorType, ...] = ()
    penalty_weight: float = 0.0
    hinge_threshold_c: float | None = None


B0: Final[ArmSpec] = ArmSpec(name="b0")
"""GAM-K's convex model on past weather."""


def interaction_arm(*, name: str, penalty_weight: float) -> ArmSpec:
    """Return one of the interaction arms `t1`, `t2`, `t3`, or `t_all`, at a penalty weight.

    The placebo arms `p_halfhour` and `p_day` split each heating column into two pieces by an
    arbitrary parity, so they carry T2's number of columns and penalty without its time-of-day
    structure.

    Args:
        name: `t1` (heating by day type), `t2` (heating by half-band), `t3` (lighting by day
            type), or `t_all` (all three together).
        penalty_weight: The difference penalty's weight.

    Returns:
        The arm.

    Raises:
        ValueError: If the name is not one of the four.
    """
    heating: dict[str, tuple[FactorType, ...]] = {
        "t1": ("daytype",),
        "t2": ("subband",),
        "t3": (),
        "t_all": ("daytype", "subband"),
        "p_halfhour": ("parity_halfhour",),
        "p_day": ("parity_day",),
    }
    lighting: tuple[FactorType, ...] = ("daytype",) if name in ("t3", "t_all") else ()
    if name not in heating:
        msg = f"unknown interaction arm {name!r}"
        raise ValueError(msg)
    return ArmSpec(
        name=name,
        heating_factors=heating[name],
        lighting_factors=lighting,
        penalty_weight=penalty_weight,
    )


def threshold_arm(*, threshold_c: float) -> ArmSpec:
    """Return B0 with the heating hinge at another threshold.

    Args:
        threshold_c: The heating threshold in degrees C.

    Returns:
        The arm.
    """
    return ArmSpec(name=f"h_{threshold_c:g}", hinge_threshold_c=threshold_c)


@dataclass(frozen=True)
class BaseFit:
    """B0's inputs at one cut: the layout, the training rows, the design, and the target.

    Attributes:
        site: The substation's label.
        cut: The cut; no training row ends after it.
        layout: B0's columns.
        index: The training rows (grid positions).
        x0: B0's design matrix, shape (n, p0).
        target: Net power plus the pinned solar output for a pinned layout, in points.
        pinned: The pinned capacity, or `None` for a fitted-capacity layout.
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
    *, cache: FeatureCache, vintages: pl.DataFrame, site: str, cut: datetime, fitted: bool = False
) -> BaseFit:
    """Build B0's design at one cut.

    Args:
        cache: The feature cache.
        vintages: Experiment 45's real-time capacity vintages.
        site: The substation's label.
        cut: The cut; no training row ends after it.
        fitted: True to fit the PV capacity instead of pinning it.

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
    features = cache.sites[site]
    x0 = design_matrix(cache=cache, features=features, layout=layout, index=index)
    target = features.y[index]
    pinned = None
    if not fitted:
        pinned = pinned_capacity(vintages=vintages, table=features.table, cut=cut)
        target = target + pinned_solar_points(
            cache=cache,
            features=features,
            pinned=pinned,
            index=index,
            clip=common.clip_of(site=site, pinned=pinned),
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


def hinge_block(
    *, cache: FeatureCache, site: str, index: np.ndarray, threshold_c: float
) -> np.ndarray:
    """Return the 16 heating columns for a hinge at `threshold_c`.

    The columns are laid out as `studies.dp_gam.design_matrix` lays out the heating block:
    smoothing-major, band-minor, so that column `k * 4 + b` is the hinge of the k-th smoothed
    temperature in band b.

    Args:
        cache: The feature cache.
        site: The substation's label.
        index: Grid rows.
        threshold_c: The threshold in degrees C.

    Returns:
        An array of shape (len(index), 16).
    """
    smoothed = cache.sites[site].smoothed_temperature_c[index]
    hinge = np.clip(threshold_c - smoothed, 0.0, None) / HEATING_SCALE_C
    band_hot = (cache.calendar.band[index][:, None] == np.arange(N_BANDS)[None, :]).astype(
        np.float64
    )
    return (hinge[:, :, None] * band_hot[:, None, :]).reshape(len(index), -1)


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


def arm_matrix(
    *,
    cache: FeatureCache,
    layout: GamLayout,
    site: str,
    index: np.ndarray,
    x0: np.ndarray,
    arm: ArmSpec,
) -> np.ndarray:
    """Return an arm's design matrix at the rows `index`.

    The columns are B0's columns that the arm keeps, in B0's order, followed by the pieces of each
    split column in ascending column order.

    Args:
        cache: The feature cache.
        layout: B0's columns.
        site: The substation's label.
        index: Grid rows.
        x0: B0's design matrix at `index`.
        arm: The arm.

    Returns:
        The arm's design matrix.
    """
    x = x0
    if arm.hinge_threshold_c is not None:
        x = x0.copy()
        x[:, layout.slices["heating"]] = hinge_block(
            cache=cache, site=site, index=index, threshold_c=arm.hinge_threshold_c
        )
    replaced = replaced_columns(layout=layout, arm=arm)
    if not replaced:
        return x
    drop = {column for column, _ in replaced}
    keep = [j for j in range(x.shape[1]) if j not in drop]
    blocks = [x[:, keep]]
    for column, factors in replaced:
        masks = piece_masks(cache=cache, index=index, factors=factors)
        blocks.append(x[:, [column]] * masks)
    return np.hstack(blocks)


@dataclass(frozen=True)
class ArmPenalty:
    """An arm's penalty matrix and sign bounds.

    Attributes:
        matrix: The quadratic penalty, shape (p, p).
        constrained: True for the columns that must be non-negative.
        n_new_columns: The number of pieces beyond B0's column count.
    """

    matrix: np.ndarray
    constrained: np.ndarray
    n_new_columns: int


def arm_penalty(*, layout: GamLayout, arm: ArmSpec, n_rows: int) -> ArmPenalty:
    """Return an arm's penalty matrix, in the column order of `arm_matrix`.

    Args:
        layout: B0's columns.
        arm: The arm.
        n_rows: The number of training rows, which scales the difference penalty.

    Returns:
        The penalty and the sign bounds.

    Raises:
        ValueError: If a split column carries a penalty that couples it to another column.
    """
    base, _ = penalty_matrix(layout=layout, n_rows=n_rows)
    constrained0 = layout.constrained
    replaced = replaced_columns(layout=layout, arm=arm)
    if not replaced:
        return ArmPenalty(matrix=base, constrained=constrained0.copy(), n_new_columns=0)
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
    return ArmPenalty(matrix=matrix, constrained=constrained, n_new_columns=size - base.shape[0])


def _pairs(
    *, combos: list[tuple[int, ...]], position: int
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Return the pairs of piece combinations that differ only in the factor at `position`."""
    return [
        (combo, (*combo[:position], 1, *combo[position + 1 :]))
        for combo in combos
        if combo[position] == 0
    ]


@dataclass(frozen=True)
class MonthGrams:
    """The per-month Gram matrices of a design, which turn a bootstrap resample into a sum.

    Attributes:
        gram: `X_m^T X_m` for each month, shape (M, p, p), float64.
        rows: For each month, the positions of its rows.
        n_months: The number of months.
    """

    gram: torch.Tensor
    rows: tuple[np.ndarray, ...]
    n_months: int

    @staticmethod
    def build(*, x: np.ndarray, month_of_row: np.ndarray, n_months: int) -> MonthGrams:
        """Build the per-month Gram matrices of `x`.

        Args:
            x: The design, shape (n, p).
            month_of_row: Each row's month position.
            n_months: The number of months.

        Returns:
            The Gram matrices.
        """
        rows = tuple(np.flatnonzero(month_of_row == m) for m in range(n_months))
        grams = torch.zeros((n_months, x.shape[1], x.shape[1]), dtype=torch.float64)
        for m, members in enumerate(rows):
            block = torch.as_tensor(x[members])
            grams[m] = block.T @ block
        return MonthGrams(gram=grams, rows=rows, n_months=n_months)

    def cross_products(self, *, x: np.ndarray, target: np.ndarray) -> torch.Tensor:
        """Return `X_m^T y_m` for each month, shape (M, p).

        Args:
            x: The design the Gram matrices were built from.
            target: The target for every row.

        Returns:
            The cross products.
        """
        out = torch.zeros((self.n_months, x.shape[1]), dtype=torch.float64)
        for m, members in enumerate(self.rows):
            out[m] = torch.as_tensor(x[members].T @ target[members])
        return out

    def squares(self, *, target: np.ndarray) -> np.ndarray:
        """Return `y_m^T y_m` for each month."""
        return np.array([float(target[members] @ target[members]) for members in self.rows])

    def weighted(self, *, counts: np.ndarray) -> torch.Tensor:
        """Return `X^T W X` for each row of month counts, shape (B, p, p).

        Args:
            counts: Month counts, shape (B, M).

        Returns:
            The weighted Gram matrices.
        """
        return torch.einsum("bm,mpq->bpq", torch.as_tensor(counts), self.gram)


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
        threshold_c: The heating threshold the fit used.
        penalised_loss: The squared error plus the penalty at the solution.
        unpenalised_loss: The squared error alone.
    """

    arm: ArmSpec
    theta: np.ndarray
    threshold_c: float
    penalised_loss: float
    unpenalised_loss: float


def fit_arm(*, cache: FeatureCache, base: BaseFit, arm: ArmSpec) -> ArmFit:
    """Fit an arm on the full training sample of a base fit.

    Args:
        cache: The feature cache.
        base: B0's inputs at the cut.
        arm: The arm.

    Returns:
        The fit.
    """
    x = arm_matrix(
        cache=cache, layout=base.layout, site=base.site, index=base.index, x0=base.x0, arm=arm
    )
    penalty = arm_penalty(layout=base.layout, arm=arm, n_rows=base.n)
    xt = torch.as_tensor(x)
    gram = (xt.T @ xt)[None]
    rhs = torch.as_tensor(x.T @ base.target)[None]
    theta = solve(gram=gram, rhs=rhs, penalty=penalty)[0]
    residual = base.target - x @ theta
    unpenalised = float(residual @ residual)
    penalised = unpenalised + float(theta @ penalty.matrix @ theta)
    threshold = common.BASE_THRESHOLD_C if arm.hinge_threshold_c is None else arm.hinge_threshold_c
    return ArmFit(
        arm=arm,
        theta=theta,
        threshold_c=threshold,
        penalised_loss=penalised,
        unpenalised_loss=unpenalised,
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
    x = arm_matrix(cache=cache, layout=base.layout, site=base.site, index=index, x0=x0, arm=fit.arm)
    if not np.isfinite(x).all():
        msg = f"{base.site} has a non-finite input among the rows to predict"
        raise ValueError(msg)
    out = x @ fit.theta
    if base.pinned is not None:
        out = out - pinned_solar_points(
            cache=cache,
            features=features,
            pinned=base.pinned,
            index=index,
            clip=common.clip_of(site=base.site, pinned=base.pinned),
        )
    return out
