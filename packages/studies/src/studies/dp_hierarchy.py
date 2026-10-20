"""The joint fit of Primary substations and the bulk supply point (BSP) above them (experiment 124).

**A BSP reading is a soft hint about the Primaries beneath it, never an exact sum.** The relation is

    BSP(t) = sum over Primaries i of scale_i * d_i(t) - generators(t) + r(t)

where `d_i` is the convex model of `studies.dp_gam` for Primary `i` (net power in points of
effective capacity), `scale_i` converts points to MW once, `generators(t)` is the metered generation
connected at the BSP (export-positive, so it is subtracted from an import-positive reading), and
`r(t)` is a penalised remainder that absorbs whatever the Primaries and generators do not explain.

**The remainder has six terms.** A level, piecewise-constant monthly steps with an L1 penalty (the
drift), a non-negative unmetered PV capacity on a given shape (with a minus sign, like the
Primaries' capacity), one heating slope on the BSP's area-mean temperature, one daily shape of 48
half-hour effects, and an optional non-negative loss term on a given series.

**The joint objective is a bounded least-squares problem in the stacked coefficients.**

    sum_i (1/sigma_i^2) * (|y_i - X_i theta_i|^2 + penalties_i)
      + (w / sigma_b^2) * sum_b (|z_b - sum_i scale_i X_i theta_i - Z_b rho_b|^2 + penalties_b)

with `z_b = BSP_b + generators_b + (the pinned solar output of each member that has one)`. Each
Primary block carries the factor `1/sigma_i^2` on its loss *and* its penalties, so at `w = 0` the
joint solution equals the separate fits exactly (the minimiser of a scaled objective is the same).
**The remainder's penalties scale with the BSP term, `w / sigma_b^2`,** so changing `w` changes how
much the BSP reading matters and not how freely the remainder can absorb an offset; a tiny ridge
that does not scale with `w` keeps the matrix definite at `w = 0`. The Gram matrix of the Primary
blocks is block diagonal and does not depend on `w`, so one pass over the rows serves a whole grid
of weights, and `solve_bounded_quadratic` solves them as one batch.

**The remainder's level can absorb a constant offset,** so a BSP reading pulls a corrupted
Primary back where the error is local in time and not where it is constant. How much damage the
hint removes on a clean year is a property of the data and the penalties, and experiment 124
measures it.

**A member is `mw` or `mva_dark`.** An `mw` member contributes its net power, so its pinned solar
output (if any) moves to the BSP target as for a single-site fit. An `mva_dark` member contributes
its demand terms alone (the capacity columns are not used on the BSP rows), because a reading in
apparent power hides the sign in daylight; its embedded PV joins the remainder's PV term.

**Nothing here reads a row after the cut.** The Primary rows come from `dp_gam.prepare_fit`; the
BSP rows are cut by the same rule and checked with `dp_gam.assert_no_lookahead`.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Literal

import numpy as np
import torch

from studies.dp_calendar import HALF_HOURS_PER_DAY, HEATING_THRESHOLD_C
from studies.dp_features import FeatureCache, naive_utc
from studies.dp_gam import (
    HEATING_SCALE_C,
    KNOT_SPACING_DAYS,
    SITE_YEAR_ROWS,
    GamFit,
    GamLayout,
    GamSpec,
    TrainingWindow,
    assert_before_holdout,
    assert_no_lookahead,
    design_matrix,
    last_signal_season_year,
    penalty_matrix,
    pinned_solar_points,
    predict_gam,
    prepare_fit,
    row_chunks,
    solve_bounded_quadratic,
)
from studies.dp_pinned_capacity import PinnedCapacity

MemberKind = Literal["mw", "mva_dark"]
"""Whether a Primary's meter reads real power (`mw`) or apparent power fitted on dark rows."""

REMAINDER_RIDGE: Final[float] = 1e-6
"""The ridge on the level, the steps, and the PV and loss terms, which makes the matrix definite."""

DEFAULT_SHAPE_RIDGE: Final[float] = 1.0
"""The default ridge on the 48 half-hour effects of the remainder's daily shape, in MW squared."""

DEFAULT_STEP_PENALTY: Final[float] = 30.0
"""The default L1 cost of one step of the remainder's drift, per site-year of rows, in MW."""


@dataclass(frozen=True)
class RemainderSpec:
    """The penalties of a BSP's remainder.

    Attributes:
        step_penalty: The L1 cost of the absolute value of each monthly step, scaled by the number
            of BSP rows over `SITE_YEAR_ROWS`.
        shape_ridge: The ridge on the daily shape's 48 effects.
        with_loss: True to add the non-negative loss term.
        with_heating: True to add the heating slope.
    """

    step_penalty: float = DEFAULT_STEP_PENALTY
    shape_ridge: float = DEFAULT_SHAPE_RIDGE
    with_loss: bool = False
    with_heating: bool = True


@dataclass(frozen=True)
class RemainderLayout:
    """The columns of one BSP's remainder.

    Attributes:
        spec: The penalties and options.
        knot_days: The step knots as days since 1970-01-01, ascending; the last is the cut.
        slices: Each term's columns.
        n_columns: The number of columns.
        constrained: True for the columns that must be non-negative.
    """

    spec: RemainderSpec
    knot_days: np.ndarray
    slices: dict[str, slice]
    n_columns: int
    constrained: np.ndarray


@dataclass(frozen=True)
class BspMember:
    """A Primary that sits under a BSP.

    Attributes:
        site: The Primary's label in the feature cache.
        kind: `mw` or `mva_dark`.
        coefficient: The factor on the Primary's contribution to the BSP row, 1 unless the caller
            has fitted it (a closure audit before the cut). It is a constant of the fit, not a
            parameter.
    """

    site: str
    kind: MemberKind = "mw"
    coefficient: float = 1.0


@dataclass(frozen=True)
class BspSeries:
    """One BSP's reading and covariates on the cache's grid, all in MW and degrees.

    Attributes:
        name: A label for error messages; never published with a Primary.
        members: The Primaries beneath the BSP whose demand terms enter its row.
        reading_mw: The import-positive BSP power, shape (T,), `NaN` where there is none.
        generators_mw: The sum of the metered generation at the BSP (export-positive), shape (T,),
            `NaN` where a reading is missing, which drops the row.
        temperature_c: The BSP area's mean temperature, shape (T,).
        pv_shape: The unmetered PV's per-unit output shape, shape (T,), `NaN` where unknown.
        loss: A loss series for the optional loss term, shape (T,), or `None`.
        generator_coefficient: The factor on the generators in the BSP row, 1 unless the caller has
            fitted it before the cut; a constant of the fit.
        sigma_mw: The robust half-hourly residual scale of the BSP row, in MW. The caller must
            compute it from rows before the fit's cut; nothing here can check that.
    """

    name: str
    members: tuple[BspMember, ...]
    reading_mw: np.ndarray
    generators_mw: np.ndarray
    temperature_c: np.ndarray
    pv_shape: np.ndarray
    sigma_mw: float
    loss: np.ndarray | None = None
    generator_coefficient: float = 1.0


@dataclass(frozen=True)
class JointFit:
    """A batch of joint fits that differ only in the BSP weight.

    Attributes:
        weights: The BSP weight `w` of each batch member.
        sites: The Primaries in the order of their columns.
        site_layouts: Each Primary's layout.
        site_slices: Each Primary's columns of `theta`.
        remainder_layouts: Each BSP's remainder layout, by name.
        remainder_slices: Each BSP's remainder columns of `theta`.
        theta: The coefficients, shape (batch, p), float64.
        cut: The fit's cut; every training row ends no later.
        n_site_rows: The number of training rows of each Primary.
        n_bsp_rows: The number of training rows of each BSP.
        last_season_years: Each Primary's latest season-year with a solar signal.
        pinned: Each Primary's pinned capacity, or `None`.
        violation: Each fit's optimality violation.
        primary_blocks: Each Primary's block of the normal equations before the BSP rows are added:
            the Gram matrix plus the penalty matrix, divided by the Primary's sigma squared.
    """

    weights: tuple[float, ...]
    sites: tuple[str, ...]
    site_layouts: dict[str, GamLayout]
    site_slices: dict[str, slice]
    remainder_layouts: dict[str, RemainderLayout]
    remainder_slices: dict[str, slice]
    theta: np.ndarray
    cut: datetime
    n_site_rows: dict[str, int]
    n_bsp_rows: dict[str, int]
    last_season_years: dict[str, int]
    pinned: dict[str, PinnedCapacity | None]
    violation: np.ndarray
    primary_blocks: dict[str, np.ndarray]

    def site_fit(self, *, site: str) -> GamFit:
        """Return one Primary's coefficients as a `dp_gam.GamFit` batch over the weights.

        Args:
            site: The Primary's label.

        Returns:
            A `GamFit` whose batch members correspond to `weights`.
        """
        return GamFit(
            layout=self.site_layouts[site],
            theta=self.theta[:, self.site_slices[site]],
            cut=self.cut,
            n_rows=self.n_site_rows[site],
            site=site,
            last_season_year=self.last_season_years[site],
            violation=self.violation,
            pinned=self.pinned[site],
        )


def make_remainder_layout(
    *, spec: RemainderSpec, cut: datetime, earliest: np.datetime64
) -> RemainderLayout:
    """Lay out the columns of a BSP's remainder.

    Args:
        spec: The penalties and options.
        cut: The end of the training window; the last knot is here.
        earliest: The earliest instant any BSP training row can have.

    Returns:
        The layout, with a non-negative PV term, step parts, and loss term.
    """
    cut_day = naive_utc(moment=cut).astype("datetime64[s]").astype(np.int64) / 86400.0
    first_day = earliest.astype("datetime64[s]").astype(np.int64) / 86400.0
    n_knots = int(np.ceil((cut_day - first_day) / KNOT_SPACING_DAYS)) + 1
    knot_days = cut_day - KNOT_SPACING_DAYS * np.arange(n_knots)[::-1]
    sizes: dict[str, int] = {
        "level": 1,
        "steps": 2 * max(n_knots - 2, 0),
        "pv": 1,
        "shape": HALF_HOURS_PER_DAY,
    }
    if spec.with_heating:
        sizes["heating"] = 1
    if spec.with_loss:
        sizes["loss"] = 1
    slices: dict[str, slice] = {}
    start = 0
    for name, size in sizes.items():
        slices[name] = slice(start, start + size)
        start += size
    constrained = np.zeros(start, dtype=bool)
    for name in ("steps", "pv", "loss"):
        if name in slices:
            constrained[slices[name]] = True
    return RemainderLayout(
        spec=spec, knot_days=knot_days, slices=slices, n_columns=start, constrained=constrained
    )


def remainder_design(
    *, cache: FeatureCache, layout: RemainderLayout, bsp: BspSeries, index: np.ndarray
) -> np.ndarray:
    """Build the remainder's design matrix at the grid rows `index`.

    Args:
        cache: The cache, for the grid and the calendar.
        layout: The remainder's columns.
        bsp: The BSP's covariates.
        index: Grid rows.

    Returns:
        An array of shape (len(index), layout.n_columns). The PV columns carry a minus sign, so a
        non-negative coefficient lowers net power. The step columns are a positive and a negative
        part, each an indicator of the rows at or after a knot.
    """
    n = len(index)
    x = np.zeros((n, layout.n_columns))
    x[:, layout.slices["level"]] = 1.0
    days = cache.end_times[index].astype("datetime64[s]").astype(np.int64) / 86400.0
    after = (days[:, None] >= layout.knot_days[None, 1:-1]).astype(np.float64)
    x[:, layout.slices["steps"]] = np.concatenate([after, -after], axis=1)
    x[:, layout.slices["pv"]] = -bsp.pv_shape[index][:, None]
    x[np.arange(n), layout.slices["shape"].start + cache.calendar.half_hour[index]] = 1.0
    if "heating" in layout.slices:
        hinge = np.clip(HEATING_THRESHOLD_C - bsp.temperature_c[index], 0.0, None) / HEATING_SCALE_C
        x[:, layout.slices["heating"]] = hinge[:, None]
    if "loss" in layout.slices:
        if bsp.loss is None:
            msg = f"{bsp.name} has a loss term in its layout but no loss series"
            raise ValueError(msg)
        x[:, layout.slices["loss"]] = bsp.loss[index][:, None]
    return x


def remainder_penalty(*, layout: RemainderLayout, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the remainder's quadratic penalty matrix and linear cost vector.

    Args:
        layout: The remainder's columns.
        n_rows: The number of BSP training rows, which scales the step cost.

    Returns:
        `(P, c)` of shapes (p, p) and (p,): a ridge on every column (the daily shape's is
        `spec.shape_ridge`) and the L1 cost `c` on the step parts, both in MW units. `fit_joint`
        scales them by `w / sigma_b^2`, with the BSP rows.
    """
    matrix = REMAINDER_RIDGE * np.eye(layout.n_columns)
    cost = np.zeros(layout.n_columns)
    shape = layout.slices["shape"]
    matrix[shape, shape] = layout.spec.shape_ridge * np.eye(shape.stop - shape.start)
    cost[layout.slices["steps"]] = layout.spec.step_penalty * n_rows / SITE_YEAR_ROWS
    return matrix, cost


def bsp_rows(
    *,
    cache: FeatureCache,
    bsp: BspSeries,
    start: np.datetime64,
    cut: datetime,
) -> np.ndarray:
    """Return the grid rows a BSP's row may train on.

    A row trains when it ends no later than `cut` and after `start`, the BSP reading, the
    generators, the BSP covariates, and every member's covariates are finite.

    Args:
        cache: The cache.
        bsp: The BSP's series.
        start: The earliest end time a row may have.
        cut: The fit's cut.

    Returns:
        Sorted grid positions.

    Raises:
        ValueError: If a row ends after the cut (lookahead).
    """
    keep = (
        np.isfinite(bsp.reading_mw)
        & np.isfinite(bsp.generators_mw)
        & np.isfinite(bsp.temperature_c)
        & np.isfinite(bsp.pv_shape)
        & (cache.end_times <= naive_utc(moment=cut))
        & (cache.end_times > start)
    )
    if bsp.loss is not None:
        keep &= np.isfinite(bsp.loss)
    for member in bsp.members:
        features = cache.sites[member.site]
        keep &= np.isfinite(features.pv_unit_shape)
        keep &= np.isfinite(features.smoothed_temperature_c).all(axis=1)
    index = np.flatnonzero(keep)
    assert_no_lookahead(cache=cache, index=index, cut=cut)
    return index


def bsp_target(
    *,
    cache: FeatureCache,
    bsp: BspSeries,
    pinned: Mapping[str, PinnedCapacity | None],
    layouts: Mapping[str, GamLayout],
    index: np.ndarray,
) -> np.ndarray:
    """Return `z`, the BSP reading plus the generators plus each `mw` member's pinned solar, in MW.

    Args:
        cache: The cache.
        bsp: The BSP's series.
        pinned: Each Primary's pinned capacity, or `None`.
        layouts: Each Primary's layout, for its export clip.
        index: Grid rows.

    Returns:
        The target at the rows, in MW.
    """
    target = bsp.reading_mw[index] + bsp.generator_coefficient * bsp.generators_mw[index]
    for member in bsp.members:
        capacity = pinned.get(member.site)
        if member.kind != "mw" or capacity is None:
            continue
        features = cache.sites[member.site]
        target = target + (
            member.coefficient * features.table.effective_capacity_mw
        ) * pinned_solar_points(
            cache=cache,
            features=features,
            pinned=capacity,
            index=index,
            clip=layouts[member.site].spec.clip,
        )
    return target


def _member_design(
    *,
    cache: FeatureCache,
    member: BspMember,
    layout: GamLayout,
    index: np.ndarray,
) -> np.ndarray:
    """Return a member's design in MW at the BSP rows (demand-only for an `mva_dark` member)."""
    features = cache.sites[member.site]
    x = design_matrix(cache=cache, features=features, layout=layout, index=index)
    if member.kind == "mva_dark":
        x[:, layout.slices["capacity"]] = 0.0
    return member.coefficient * features.table.effective_capacity_mw * x


@dataclass
class JointAssembly:
    """The designs and Gram matrices of one cut, built once and solved for any targets and weights.

    A synthetic system or an injected error changes the readings and leaves the designs alone, so
    the expensive part of a fit (the designs and their Gram matrices) is shared by every system,
    injection, and weight, and `solve_joint` only recomputes the right-hand sides.

    Attributes:
        cache: The feature cache the designs were built from.
        sites: The Primaries in column order.
        bsps: The BSPs.
        cut: The cut; every row ends no later.
        layouts: Each Primary's layout.
        indices: Each Primary's training rows.
        offsets: Each Primary's columns.
        sigma_points: Each Primary's sigma in points.
        pinned: Each Primary's pinned capacity, or `None`.
        remainder_layouts: Each BSP's remainder layout.
        remainder_offsets: Each BSP's remainder columns.
        bsp_indices: Each BSP's training rows.
        bsp_columns: The columns each BSP's rows touch.
        site_designs: Each Primary's design at its training rows, shape (n_i, p_i).
        site_target_offsets: Each Primary's pinned solar output at its rows, added to its target.
        bsp_designs: Each BSP's stacked design at its rows (member columns, then remainder).
        bsp_target_offsets: Each BSP's generators plus its members' pinned solar at its rows.
        base: The Primary blocks of the matrix, scaled by 1 / sigma_i^2, plus the remainder ridge.
        base_cost: The Primaries' linear costs.
        hint: The BSP term of the matrix per unit weight (Gram matrices and remainder penalties).
        hint_cost: The remainder's linear costs per unit weight.
        constrained: True for the columns that must be non-negative.
        device: Where the tensors live.
        dtype: Their floating-point type.
    """

    cache: FeatureCache
    sites: tuple[str, ...]
    bsps: tuple[BspSeries, ...]
    cut: datetime
    layouts: dict[str, GamLayout]
    indices: dict[str, np.ndarray]
    offsets: dict[str, slice]
    sigma_points: dict[str, float]
    pinned: dict[str, PinnedCapacity | None]
    remainder_layouts: dict[str, RemainderLayout]
    remainder_offsets: dict[str, slice]
    bsp_indices: dict[str, np.ndarray]
    bsp_columns: dict[str, torch.Tensor]
    site_designs: dict[str, torch.Tensor]
    site_target_offsets: dict[str, np.ndarray]
    bsp_designs: dict[str, torch.Tensor]
    bsp_target_offsets: dict[str, np.ndarray]
    base: torch.Tensor
    base_cost: torch.Tensor
    hint: torch.Tensor
    hint_cost: torch.Tensor
    constrained: np.ndarray
    device: torch.device | str
    dtype: torch.dtype


def _check_members(
    *,
    bsps: Sequence[BspSeries],
    specs: Mapping[str, GamSpec],
    sigma_points: Mapping[str, float],
) -> tuple[str, ...]:
    """Return the Primaries in column order, checking each has a spec, a sigma, and one BSP."""
    owners: dict[str, str] = {}
    for bsp in bsps:
        if bsp.sigma_mw <= 0.0:
            msg = f"{bsp.name} has a sigma that is not positive"
            raise ValueError(msg)
        for member in bsp.members:
            if member.site in owners:
                msg = f"{member.site} sits under both {owners[member.site]} and {bsp.name}"
                raise ValueError(msg)
            if member.site not in specs or sigma_points.get(member.site, 0.0) <= 0.0:
                msg = f"{member.site} has no spec or no positive sigma"
                raise ValueError(msg)
            owners[member.site] = bsp.name
    return tuple(owners)


def _stack_designs(
    *,
    rows: np.ndarray,
    build: Callable[[np.ndarray], np.ndarray],
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a design chunk by chunk and return it as one tensor on `device`."""
    return torch.cat(
        [
            torch.as_tensor(build(rows[chunk]), dtype=dtype, device=device)
            for chunk in row_chunks(index=rows)
        ]
    )


def assemble_joint(
    *,
    cache: FeatureCache,
    specs: Mapping[str, GamSpec],
    bsps: Sequence[BspSeries],
    remainder_spec: RemainderSpec,
    sigma_points: Mapping[str, float],
    cut: datetime,
    window: TrainingWindow | Mapping[str, TrainingWindow],
    season_years: Sequence[int],
    pinned: Mapping[str, PinnedCapacity | None] | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> JointAssembly:
    """Build the designs and Gram matrices of the Primaries and BSPs for one cut.

    Args:
        cache: The cache.
        specs: The convex model's spec of each Primary that appears under any BSP, by label.
        bsps: The BSPs. Every member's site must be in `specs`, and no Primary may sit under two
            BSPs.
        remainder_spec: The penalties of every BSP's remainder.
        sigma_points: Each Primary's robust half-hourly residual scale in points, from pre-cut rows.
        cut: The cut; every training row ends no later.
        window: The window rule for the Primaries' rows, or one rule per Primary (an `mva_dark`
            Primary trains on dark rows only).
        season_years: The season-years with a capacity increment.
        pinned: Each Primary's pinned capacity built for `cut`, or `None`; a missing key means
            `None`.
        device: Where the arithmetic runs.
        dtype: The floating-point type.

    Returns:
        The assembly, ready for `solve_joint`.

    Raises:
        ValueError: If a member has no spec or no sigma, a Primary sits under two BSPs, a sigma is
            not positive, or a BSP has no training row.
    """
    pinned = dict(pinned or {})
    sites = _check_members(bsps=bsps, specs=specs, sigma_points=sigma_points)
    layouts: dict[str, GamLayout] = {}
    indices: dict[str, np.ndarray] = {}
    offsets: dict[str, slice] = {}
    start = 0
    for site in sites:
        layouts[site], indices[site] = prepare_fit(
            cache=cache,
            site=site,
            spec=specs[site],
            cut=cut,
            window=window[site] if isinstance(window, Mapping) else window,
            season_years=season_years,
        )
        offsets[site] = slice(start, start + layouts[site].n_columns)
        start += layouts[site].n_columns
    bsp_indices: dict[str, np.ndarray] = {}
    remainder_layouts: dict[str, RemainderLayout] = {}
    remainder_offsets: dict[str, slice] = {}
    for bsp in bsps:
        # The BSP rows start at the earliest training row of any member, so the hint covers the
        # years the Primaries train on and no earlier ones.
        first_member_row = min(cache.end_times[indices[m.site][0]] for m in bsp.members)
        index = bsp_rows(
            cache=cache, bsp=bsp, start=first_member_row - np.timedelta64(1, "us"), cut=cut
        )
        if len(index) == 0:
            msg = f"{bsp.name} has no training row before {cut}"
            raise ValueError(msg)
        bsp_indices[bsp.name] = index
        remainder_layouts[bsp.name] = make_remainder_layout(
            spec=remainder_spec, cut=cut, earliest=cache.end_times[index[0]]
        )
        remainder_offsets[bsp.name] = slice(start, start + remainder_layouts[bsp.name].n_columns)
        start += remainder_layouts[bsp.name].n_columns
    assembly = JointAssembly(
        cache=cache,
        sites=sites,
        bsps=tuple(bsps),
        cut=cut,
        layouts=layouts,
        indices=indices,
        offsets=offsets,
        sigma_points=dict(sigma_points),
        pinned=pinned,
        remainder_layouts=remainder_layouts,
        remainder_offsets=remainder_offsets,
        bsp_indices=bsp_indices,
        bsp_columns={},
        site_designs={},
        site_target_offsets={},
        bsp_designs={},
        bsp_target_offsets={},
        base=torch.zeros((start, start), dtype=dtype, device=device),
        base_cost=torch.zeros(start, dtype=dtype, device=device),
        hint=torch.zeros((start, start), dtype=dtype, device=device),
        hint_cost=torch.zeros(start, dtype=dtype, device=device),
        constrained=np.zeros(start, dtype=bool),
        device=device,
        dtype=dtype,
    )
    _add_primary_blocks(assembly=assembly)
    for bsp in bsps:
        _add_bsp_block(assembly=assembly, bsp=bsp)
    return assembly


def _add_primary_blocks(*, assembly: JointAssembly) -> None:
    """Add each Primary's scaled Gram matrix, penalties, and constraints to the assembly."""
    cache, device, dtype = assembly.cache, assembly.device, assembly.dtype
    for site, layout in assembly.layouts.items():
        features = cache.sites[site]
        index = assembly.indices[site]
        design = _stack_designs(
            rows=index,
            build=lambda rows, f=features, lay=layout: design_matrix(
                cache=cache, features=f, layout=lay, index=rows
            ),
            device=device,
            dtype=dtype,
        )
        capacity = assembly.pinned.get(site)
        assembly.site_designs[site] = design
        assembly.site_target_offsets[site] = (
            np.zeros(len(index))
            if capacity is None
            else pinned_solar_points(
                cache=cache, features=features, pinned=capacity, index=index, clip=layout.spec.clip
            )
        )
        penalty, cost = penalty_matrix(layout=layout, n_rows=len(index))
        scale = 1.0 / assembly.sigma_points[site] ** 2
        block = assembly.offsets[site]
        assembly.base[block, block] = scale * (
            design.T @ design + torch.as_tensor(penalty, dtype=dtype, device=device)
        )
        assembly.base_cost[block] = scale * torch.as_tensor(cost, dtype=dtype, device=device)
        assembly.constrained[block] = layout.constrained


def _add_bsp_block(*, assembly: JointAssembly, bsp: BspSeries) -> None:
    """Add one BSP's remainder penalties and its unweighted row Gram matrix to the assembly."""
    cache, device, dtype = assembly.cache, assembly.device, assembly.dtype
    index = assembly.bsp_indices[bsp.name]
    rem = assembly.remainder_layouts[bsp.name]
    rem_block = assembly.remainder_offsets[bsp.name]
    penalty, cost = remainder_penalty(layout=rem, n_rows=len(index))
    inverse_variance = 1.0 / bsp.sigma_mw**2
    # The remainder's penalties scale with the BSP term (w / sigma_b^2), so changing w changes how
    # much the hint matters and not how freely the remainder can absorb an offset. A tiny ridge
    # that does not scale with w keeps the matrix definite at w = 0.
    assembly.base[rem_block, rem_block] = REMAINDER_RIDGE * torch.eye(
        rem.n_columns, dtype=dtype, device=device
    )
    assembly.hint[rem_block, rem_block] += inverse_variance * torch.as_tensor(
        penalty, dtype=dtype, device=device
    )
    assembly.hint_cost[rem_block] = inverse_variance * torch.as_tensor(
        cost, dtype=dtype, device=device
    )
    assembly.constrained[rem_block] = rem.constrained
    columns = np.concatenate(
        [
            np.arange(assembly.offsets[m.site].start, assembly.offsets[m.site].stop)
            for m in bsp.members
        ]
        + [np.arange(rem_block.start, rem_block.stop)]
    )
    column_index = torch.as_tensor(columns, device=device)

    def build(rows: np.ndarray) -> np.ndarray:
        parts = [
            _member_design(cache=cache, member=m, layout=assembly.layouts[m.site], index=rows)
            for m in bsp.members
        ]
        parts.append(remainder_design(cache=cache, layout=rem, bsp=bsp, index=rows))
        return np.concatenate(parts, axis=1)

    design = _stack_designs(rows=index, build=build, device=device, dtype=dtype)
    assembly.bsp_designs[bsp.name] = design
    assembly.bsp_columns[bsp.name] = column_index
    assembly.bsp_target_offsets[bsp.name] = (
        bsp_target(
            cache=cache, bsp=bsp, pinned=assembly.pinned, layouts=assembly.layouts, index=index
        )
        - bsp.reading_mw[index]
    )
    assembly.hint[column_index[:, None], column_index[None, :]] += inverse_variance * (
        design.T @ design
    )


def solve_joint(
    *,
    assembly: JointAssembly,
    weights: Sequence[float],
    primary_power: Mapping[str, np.ndarray] | None = None,
    bsp_readings: Mapping[str, np.ndarray] | None = None,
    excluded_rows: Mapping[str, np.ndarray] | None = None,
) -> JointFit:
    """Solve the joint problem of an assembly for each BSP weight, for given readings.

    Only the right-hand sides depend on the readings, so a synthetic system or an injected error
    costs one matrix-vector product per block and one batched solve.

    Args:
        assembly: The designs and Gram matrices from `assemble_joint`.
        weights: The BSP weights `w`, one batch member each; 0 reproduces the separate fits.
        primary_power: Replacement readings of Primaries in points, each on the cache grid; a
            Primary not named keeps the cache's.
        bsp_readings: Replacement readings of BSPs in MW, each on the cache grid; a BSP not named
            keeps its own.
        excluded_rows: Grid rows to leave out of a Primary's own loss (not out of the BSP rows),
            for example the rows of a known metering error. The Primary's penalties keep the
            scale of its full training set.

    Returns:
        The fitted batch.
    """
    cache, device, dtype = assembly.cache, assembly.device, assembly.dtype
    primary_power = primary_power or {}
    bsp_readings = bsp_readings or {}
    excluded_rows = excluded_rows or {}
    base = assembly.base
    rhs0 = torch.zeros(assembly.base.shape[0], dtype=dtype, device=device)
    hint_rhs = torch.zeros_like(rhs0)
    for site in assembly.sites:
        y = primary_power.get(site, cache.sites[site].y)[assembly.indices[site]]
        target = torch.as_tensor(y + assembly.site_target_offsets[site], dtype=dtype, device=device)
        design = assembly.site_designs[site]
        scale = 1.0 / assembly.sigma_points[site] ** 2
        block = assembly.offsets[site]
        if site in excluded_rows:
            dropped = torch.as_tensor(
                np.isin(assembly.indices[site], excluded_rows[site]), device=device
            )
            design, target = design[~dropped], target[~dropped]
            removed = assembly.site_designs[site][dropped]
            if base is assembly.base:
                base = assembly.base.clone()
            base[block, block] -= scale * (removed.T @ removed)
        rhs0[block] = scale * (design.T @ target)
    for bsp in assembly.bsps:
        index = assembly.bsp_indices[bsp.name]
        reading = bsp_readings.get(bsp.name, bsp.reading_mw)[index]
        z = torch.as_tensor(
            reading + assembly.bsp_target_offsets[bsp.name], dtype=dtype, device=device
        )
        hint_rhs[assembly.bsp_columns[bsp.name]] += (
            assembly.bsp_designs[bsp.name].T @ z
        ) / bsp.sigma_mw**2
    w = torch.as_tensor(list(weights), dtype=dtype, device=device)
    theta, violation = solve_bounded_quadratic(
        matrix=base[None] + w[:, None, None] * assembly.hint[None],
        rhs=rhs0[None]
        + w[:, None] * hint_rhs[None]
        - 0.5 * (assembly.base_cost[None] + w[:, None] * assembly.hint_cost[None]),
        constrained=torch.as_tensor(assembly.constrained, device=device),
    )
    return JointFit(
        weights=tuple(float(x) for x in weights),
        sites=assembly.sites,
        site_layouts=assembly.layouts,
        site_slices=assembly.offsets,
        remainder_layouts=assembly.remainder_layouts,
        remainder_slices=assembly.remainder_offsets,
        theta=theta.cpu().numpy(),
        cut=assembly.cut,
        n_site_rows={s: len(assembly.indices[s]) for s in assembly.sites},
        n_bsp_rows={b: len(i) for b, i in assembly.bsp_indices.items()},
        last_season_years={
            s: last_signal_season_year(
                cache=cache,
                site=s,
                index=assembly.indices[s],
                cut=assembly.cut,
                pinned=assembly.pinned.get(s),
            )
            for s in assembly.sites
        },
        pinned={s: assembly.pinned.get(s) for s in assembly.sites},
        violation=violation.cpu().numpy(),
        primary_blocks={
            s: assembly.base[assembly.offsets[s], assembly.offsets[s]].cpu().numpy().copy()
            for s in assembly.sites
        },
    )


def fit_joint(
    *,
    cache: FeatureCache,
    specs: Mapping[str, GamSpec],
    bsps: Sequence[BspSeries],
    remainder_spec: RemainderSpec,
    weights: Sequence[float],
    sigma_points: Mapping[str, float],
    cut: datetime,
    window: TrainingWindow | Mapping[str, TrainingWindow],
    season_years: Sequence[int],
    pinned: Mapping[str, PinnedCapacity | None] | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> JointFit:
    """Fit the Primaries and the BSPs above them jointly, for each BSP weight in `weights`.

    This is `assemble_joint` followed by `solve_joint` with the cache's own readings; use those two
    directly to solve for many readings against one assembly.

    Args:
        cache: The cache.
        specs: The convex model's spec of each Primary that appears under any BSP, by label.
        bsps: The BSPs. Every member's site must be in `specs`, and no Primary may sit under two
            BSPs.
        remainder_spec: The penalties of every BSP's remainder.
        weights: The BSP weights `w`, one batch member each; 0 reproduces the separate fits.
        sigma_points: Each Primary's robust half-hourly residual scale in points, from pre-cut rows.
        cut: The cut; every training row ends no later.
        window: The window rule for the Primaries' rows, or one rule per Primary.
        season_years: The season-years with a capacity increment.
        pinned: Each Primary's pinned capacity built for `cut`, or `None`; a missing key means
            `None`.
        device: Where the arithmetic runs.
        dtype: The floating-point type.

    Returns:
        The fitted batch.
    """
    assembly = assemble_joint(
        cache=cache,
        specs=specs,
        bsps=bsps,
        remainder_spec=remainder_spec,
        sigma_points=sigma_points,
        cut=cut,
        window=window,
        season_years=season_years,
        pinned=pinned,
        device=device,
        dtype=dtype,
    )
    return solve_joint(assembly=assembly, weights=weights)


def remainder_prediction(
    *,
    cache: FeatureCache,
    fit: JointFit,
    bsp: BspSeries,
    index: np.ndarray,
    weight_index: int = 0,
) -> np.ndarray:
    """Return a BSP's remainder `r(t)` in MW at the grid rows `index`.

    Args:
        cache: The cache.
        fit: The joint fit.
        bsp: The BSP's series, with covariates at the rows to predict.
        index: Grid rows.
        weight_index: Which weight of the fit's batch.

    Returns:
        The remainder, one value per row.

    Raises:
        ValueError: If a row has a non-finite remainder input.
    """
    assert_before_holdout(cache=cache, index=index)
    layout = fit.remainder_layouts[bsp.name]
    x = remainder_design(cache=cache, layout=layout, bsp=bsp, index=index)
    if not np.isfinite(x).all():
        msg = f"{bsp.name} has a non-finite input among the rows to predict"
        raise ValueError(msg)
    return x @ fit.theta[weight_index, fit.remainder_slices[bsp.name]]


def predict_bsp(
    *,
    cache: FeatureCache,
    fit: JointFit,
    bsp: BspSeries,
    index: np.ndarray,
    generators_mw: np.ndarray,
    weight_index: int = 0,
) -> np.ndarray:
    """Predict a BSP's import-positive power in MW at the grid rows `index`.

    The prediction is the sum of the Primaries' predictions (net power for an `mw` member, demand
    terms alone for an `mva_dark` member) times each one's MW scale, minus the generators, plus the
    remainder.

    Args:
        cache: The cache.
        fit: The joint fit.
        bsp: The BSP's series.
        index: Grid rows.
        generators_mw: The generators' sum (observed in fitting, forecast in forecasting) at the
            rows, export-positive.
        weight_index: Which weight of the fit's batch.

    Returns:
        The BSP's predicted power in MW.
    """
    total = remainder_prediction(
        cache=cache, fit=fit, bsp=bsp, index=index, weight_index=weight_index
    )
    total = total - bsp.generator_coefficient * generators_mw
    for m in bsp.members:
        site_fit = fit.site_fit(site=m.site)
        columns = None
        if m.kind == "mva_dark":
            columns = np.ones(site_fit.layout.n_columns, dtype=bool)
            columns[site_fit.layout.slices["capacity"]] = False
        predicted = predict_gam(
            cache=cache, fit=site_fit, index=index, member=weight_index, columns=columns
        )
        total = total + m.coefficient * cache.sites[m.site].table.effective_capacity_mw * predicted
    return total


def separate_fits_equal_joint_at_zero_weight(
    *, joint: JointFit, separate: Mapping[str, GamFit], weight_index: int = 0, atol: float = 1e-6
) -> bool:
    """Return True if the joint solution at weight 0 equals the separate fits (control P1).

    Args:
        joint: A joint fit whose weight at `weight_index` is 0.
        separate: The Primaries' separate `dp_gam` fits, in the same specs, window, and cut.
        weight_index: Which weight of the joint fit's batch is 0.
        atol: The tolerance on each coefficient.

    Returns:
        Whether every Primary's coefficients agree.

    Raises:
        ValueError: If the batch member's weight is not 0.
    """
    if joint.weights[weight_index] != 0.0:
        msg = f"weight {weight_index} is {joint.weights[weight_index]}, not 0"
        raise ValueError(msg)
    return all(
        np.allclose(joint.theta[weight_index, joint.site_slices[s]], fit.theta[0], atol=atol)
        for s, fit in separate.items()
    )
