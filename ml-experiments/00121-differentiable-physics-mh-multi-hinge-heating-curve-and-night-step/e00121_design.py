"""The designs of experiment 121: B0's columns, a five-hinge heating curve, and night-step terms.

**B0 is `studies.dp_gam`'s model with GAM-K's settings.** Every other arm is B0 with columns added
(or, for MH-5, with the heating columns replaced):

- **MH** adds the heating hinge at 10, 12, 14, 16, and 18 degrees C (80 columns, each with a
  non-negative slope) to B0's hinge at 15.5. **MH-5** replaces B0's hinge by the five. **MH-sh** is
  MH with the added columns' ridge multiplied by a shrinkage factor.
- **NS-L and NS-U** add one free level for each of the 20 half-hours from 22:00 to 08:00, indexed
  on the local clock (NS-L) or the UTC clock (NS-U). **NS-H** adds B0's heating hinge at 15.5 times
  an indicator for 00:00 to 07:00 UTC, with non-negative slopes. **NS-HW** (post hoc, exploratory)
  gives that hinge a separate non-negative slope for each of the 20 UTC night half-hours.

Every added column carries B0's ridge (times the shrinkage factor), so each arm nests B0. Every arm
is a penalised least-squares problem with sign bounds, solved on the Gram matrix by
`studies.dp_gam.solve_bounded_quadratic`.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Final, Literal

import e00121_common as common
import numpy as np
import polars as pl
import torch
from studies.dp_calendar import SMOOTHING_HOURS
from studies.dp_features import FeatureCache
from studies.dp_gam import (
    HEATING_SCALE_C,
    N_BANDS,
    GamLayout,
    design_matrix,
    penalty_matrix,
    pinned_solar_points,
    prepare_fit,
    solve_bounded_quadratic,
)
from studies.dp_pinned_capacity import PinnedCapacity, pinned_capacity

Kind = Literal["b0", "hinges", "night_l", "night_u", "night_h", "night_hw", "night_ut"]
"""What an arm adds to B0: nothing, heating hinges, free night levels (local or UTC clock), or
the night-hinge interaction."""
N_HEATING_COLUMNS: Final[int] = len(SMOOTHING_HOURS) * N_BANDS
HALF_HOURS_PER_DAY: Final[int] = 48
LOCAL_ZONE: Final[str] = "Europe/London"


@dataclass(frozen=True)
class ArmSpec:
    """One arm: what it adds to B0, and how strongly its added columns are shrunk.

    Attributes:
        name: The arm's name.
        kind: What the arm adds.
        thresholds_c: The heating hinge thresholds a `hinges` arm adds, in degrees C.
        replace_heating: True to drop B0's own heating columns (MH-5).
        shrink: The multiplier on the added columns' ridge.
        smooth: The weight of a second-difference penalty on the added columns (NS-UT).
    """

    name: str
    kind: Kind = "b0"
    thresholds_c: tuple[float, ...] = ()
    replace_heating: bool = False
    shrink: float = 1.0
    smooth: float = 0.0


B0: Final[ArmSpec] = ArmSpec(name="b0")
"""GAM-K's convex model on past weather."""
MH: Final[ArmSpec] = ArmSpec(name="mh", kind="hinges", thresholds_c=common.MH_THRESHOLDS_C)
MH5: Final[ArmSpec] = ArmSpec(
    name="mh5", kind="hinges", thresholds_c=common.MH_THRESHOLDS_C, replace_heating=True
)
NS_L: Final[ArmSpec] = ArmSpec(name="ns_l", kind="night_l")
NS_U: Final[ArmSpec] = ArmSpec(name="ns_u", kind="night_u")
NS_H: Final[ArmSpec] = ArmSpec(name="ns_h", kind="night_h")
NS_UT: Final[ArmSpec] = ArmSpec(name="ns_ut", kind="night_ut", smooth=common.NS_UT_SMOOTH)
"""Added before any replicate or scored fit: a UTC night step (00:30 to 07:30) whose level evolves
slowly, with one non-negative level per month of the training window, a second-difference penalty,
and the last level held through the forecast year."""
NS_HW: Final[ArmSpec] = ArmSpec(name="ns_hw", kind="night_hw")
"""Post hoc, exploratory: the 15.5 degree hinge of each smoothing times each UTC night half-hour
(80 columns, non-negative), so the cold-weather night step can differ by half-hour."""


def shrunk_arm(*, shrink: float) -> ArmSpec:
    """Return MH with the five added blocks' ridge multiplied by `shrink`."""
    return ArmSpec(
        name=f"mh_sh{shrink:g}",
        kind="hinges",
        thresholds_c=common.MH_THRESHOLDS_C,
        shrink=shrink,
    )


def all_arms() -> dict[str, ArmSpec]:
    """Return every arm that is fitted, by name."""
    arms = [
        B0,
        MH,
        MH5,
        *(shrunk_arm(shrink=m) for m in common.SHRINK_GRID),
        NS_L,
        NS_U,
        NS_H,
        NS_HW,
        NS_UT,
    ]
    return {arm.name: arm for arm in arms}


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
    spec = common.fitted_capacity_spec() if fitted else common.base_spec()
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


def start_half_hour(*, cache: FeatureCache, index: np.ndarray, local: bool) -> np.ndarray:
    """Return the half-hour of day (0 to 47) in which each row starts, on the UTC or local clock.

    A row's grid time is the end of its half-hour, so the start is 30 minutes earlier.

    Args:
        cache: The feature cache.
        index: Grid rows.
        local: True for the Europe/London clock, False for UTC.

    Returns:
        An int array.
    """
    start = pl.Series(cache.end_times[index] - np.timedelta64(30, "m")).dt.replace_time_zone("UTC")
    if local:
        start = start.dt.convert_time_zone(LOCAL_ZONE)
    minute = start.dt.hour().cast(pl.Int64) * 60 + start.dt.minute().cast(pl.Int64)
    return (minute // 30).to_numpy().astype(np.int64)


def night_columns(*, cache: FeatureCache, index: np.ndarray, local: bool) -> np.ndarray:
    """Return the 20 night indicators, column `k` for the half-hour starting `22:00 + 30 k` min."""
    half = start_half_hour(cache=cache, index=index, local=local)
    wanted = (
        common.NIGHT_START_HALF_HOUR + np.arange(common.NIGHT_HALF_HOURS)
    ) % HALF_HOURS_PER_DAY
    return (half[:, None] == wanted[None, :]).astype(np.float64)


def cache_hinge(*, cache: FeatureCache, site: str, index: np.ndarray) -> np.ndarray:
    """Return the 15.5 degree hinge of each smoothed temperature, shape (n, 4)."""
    smoothed = cache.sites[site].smoothed_temperature_c[index]
    return np.clip(common.BASE_THRESHOLD_C - smoothed, 0.0, None) / HEATING_SCALE_C


def night_hinge_columns(*, cache: FeatureCache, site: str, index: np.ndarray) -> np.ndarray:
    """Return NS-H's four columns: the 15.5 degree hinge of each smoothing times the UTC night."""
    half = start_half_hour(cache=cache, index=index, local=False)
    indicator = (half < common.NS_H_END_HALF_HOUR_UTC).astype(np.float64)
    return cache_hinge(cache=cache, site=site, index=index) * indicator[:, None]


def night_ut_columns(*, cache: FeatureCache, index: np.ndarray, cut: datetime) -> np.ndarray:
    """Return NS-UT's monthly hat-function columns times the UTC night-window indicator.

    Knots sit at 30.4375-day spacing from 24 months before the cut to the cut (25 knots). A row
    after the cut takes the last knot's level.
    """
    start = np.datetime64(cut.replace(tzinfo=None), "us") - np.timedelta64(
        round(common.WINDOW_MONTHS * 30.4375 * 86400), "s"
    )
    days = (cache.end_times[index] - start) / np.timedelta64(1, "D")
    n_knots = common.WINDOW_MONTHS + 1
    position = np.clip(days / 30.4375, 0.0, n_knots - 1.0)
    lower = np.minimum(np.floor(position).astype(np.int64), n_knots - 2)
    weight = position - lower
    basis = np.zeros((len(index), n_knots))
    rows = np.arange(len(index))
    basis[rows, lower] = 1.0 - weight
    basis[rows, lower + 1] = weight
    half = start_half_hour(cache=cache, index=index, local=False)
    window = np.isin(half, list(common.NS_UT_WINDOW)).astype(np.float64)
    return basis * window[:, None]


def added_block(
    *,
    cache: FeatureCache,
    site: str,
    index: np.ndarray,
    arm: ArmSpec,
    cut: datetime | None = None,
) -> tuple[np.ndarray, bool]:
    """Return the columns an arm adds and whether their slopes must be non-negative.

    Args:
        cache: The feature cache.
        site: The substation's label.
        index: Grid rows.
        arm: The arm.
        cut: The cut of the fit (NS-UT's knots depend on it).

    Returns:
        The added columns (shape (n, 0) for B0) and the sign constraint.
    """
    if arm.kind == "hinges":
        blocks = [
            hinge_block(cache=cache, site=site, index=index, threshold_c=t)
            for t in arm.thresholds_c
        ]
        return np.hstack(blocks), True
    if arm.kind == "night_ut":
        if cut is None:
            msg = "NS-UT needs the cut"
            raise ValueError(msg)
        return night_ut_columns(cache=cache, index=index, cut=cut), True
    if arm.kind == "night_h":
        return night_hinge_columns(cache=cache, site=site, index=index), True
    if arm.kind == "night_hw":
        hinge = cache_hinge(cache=cache, site=site, index=index)
        night = night_columns(cache=cache, index=index, local=False)
        return (hinge[:, None, :] * night[:, :, None]).reshape(len(index), -1), True
    if arm.kind in ("night_l", "night_u"):
        return night_columns(cache=cache, index=index, local=arm.kind == "night_l"), False
    return np.zeros((len(index), 0)), True


def dropped_columns(*, layout: GamLayout, arm: ArmSpec) -> list[int]:
    """Return the base columns an arm drops (B0's heating columns for MH-5)."""
    if not arm.replace_heating:
        return []
    heating = layout.slices["heating"]
    return list(range(heating.start, heating.stop))


def arm_matrix(
    *,
    cache: FeatureCache,
    layout: GamLayout,
    site: str,
    index: np.ndarray,
    x0: np.ndarray,
    arm: ArmSpec,
    cut: datetime | None = None,
) -> np.ndarray:
    """Return an arm's design matrix: B0's kept columns, then the added columns.

    Args:
        cache: The feature cache.
        layout: B0's columns.
        site: The substation's label.
        index: Grid rows.
        x0: B0's design matrix at `index`.
        arm: The arm.
        cut: The cut of the fit (NS-UT's knots depend on it).

    Returns:
        The arm's design matrix.
    """
    drop = set(dropped_columns(layout=layout, arm=arm))
    keep = [j for j in range(x0.shape[1]) if j not in drop]
    block, _ = added_block(cache=cache, site=site, index=index, arm=arm, cut=cut)
    return np.hstack([x0[:, keep], block])


@dataclass(frozen=True)
class ArmPenalty:
    """An arm's penalty matrix and sign bounds.

    Attributes:
        matrix: The quadratic penalty, shape (p, p).
        constrained: True for the columns that must be non-negative.
        n_new_columns: The number of columns beyond B0's count (negative for MH-5's drops).
    """

    matrix: np.ndarray
    constrained: np.ndarray
    n_new_columns: int


def arm_penalty(
    *, layout: GamLayout, arm: ArmSpec, n_rows: int, n_added: int, added_constrained: bool
) -> ArmPenalty:
    """Return an arm's penalty matrix, in the column order of `arm_matrix`.

    Args:
        layout: B0's columns.
        arm: The arm.
        n_rows: The number of training rows.
        n_added: The number of columns the arm adds.
        added_constrained: Whether the added columns' slopes must be non-negative.

    Returns:
        The penalty and the sign bounds. Each added column has B0's ridge times `arm.shrink`.
    """
    base, _ = penalty_matrix(layout=layout, n_rows=n_rows)
    drop = set(dropped_columns(layout=layout, arm=arm))
    keep = [j for j in range(base.shape[0]) if j not in drop]
    size = len(keep) + n_added
    matrix = np.zeros((size, size))
    matrix[: len(keep), : len(keep)] = base[np.ix_(keep, keep)]
    matrix[len(keep) :, len(keep) :] = common.RIDGE * arm.shrink * np.eye(n_added)
    if arm.smooth > 0.0:
        second = np.diff(np.eye(n_added), n=2, axis=0)
        matrix[len(keep) :, len(keep) :] += arm.smooth * second.T @ second
    constrained = np.full(size, added_constrained)
    constrained[: len(keep)] = layout.constrained[keep]
    return ArmPenalty(matrix=matrix, constrained=constrained, n_new_columns=size - base.shape[0])


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
        unpenalised_loss: The squared error alone.
    """

    arm: ArmSpec
    theta: np.ndarray
    unpenalised_loss: float


def penalty_for(*, cache: FeatureCache, base: BaseFit, arm: ArmSpec, x: np.ndarray) -> ArmPenalty:
    """Return an arm's penalty given its design matrix `x` at the base fit."""
    _, constrained = added_block(
        cache=cache, site=base.site, index=base.index[:1], arm=arm, cut=base.cut
    )
    n_added = x.shape[1] - (base.x0.shape[1] - len(dropped_columns(layout=base.layout, arm=arm)))
    return arm_penalty(
        layout=base.layout,
        arm=arm,
        n_rows=base.n,
        n_added=n_added,
        added_constrained=constrained,
    )


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
        cache=cache,
        layout=base.layout,
        site=base.site,
        index=base.index,
        x0=base.x0,
        arm=arm,
        cut=base.cut,
    )
    penalty = penalty_for(cache=cache, base=base, arm=arm, x=x)
    xt = torch.as_tensor(x)
    gram = (xt.T @ xt)[None]
    rhs = torch.as_tensor(x.T @ base.target)[None]
    theta = solve(gram=gram, rhs=rhs, penalty=penalty)[0]
    residual = base.target - x @ theta
    return ArmFit(arm=arm, theta=theta, unpenalised_loss=float(residual @ residual))


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
        cut=base.cut,
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
