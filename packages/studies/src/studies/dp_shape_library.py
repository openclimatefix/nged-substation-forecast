"""A convex library of shared daily shapes, fitted jointly across substations.

**Each substation's daily profile is a non-negative mix of K shared daily shapes, plus a penalised
deviation.** The profile of substation `s` (4 day types times 48 half-hours, 192 columns in
`dp_gam`) is

    profile_s = m_s + sum_k w[s, k] * S_k + delta_s

where `m_s` is a free mean for each hour band (so a substation's size never pools), `S_k` are K
shared shapes, `w[s, k] >= 0` are the mixing weights, and `delta_s` is the deviation. Every other
family of `dp_gam_pooled` (holidays, Christmas tent, harmonics, heating, lighting) is pooled to a
fleet mean exactly as there, so a comparison with `dp_gam_pooled.fit_pooled` isolates the profile.

- **Shapes** are signed, centred per hour band (the centring `C` of `dp_gam_pooled`, so a shape
  never carries a band mean), and stored with unit norm over all 192 entries. The scale lives in
  the weights.
- **Weights** are shared across the four day types by default, which makes `w` a (substations, K)
  array. With `weights_per_day_type` each day type has its own weight per shape, a (substations,
  4 K) array whose index is `day_type * K + k`.
- **Deviation** `delta_s = Q d_s` is held in an orthonormal basis `Q` of the centred columns, and
  penalised by `deviation_weight * g_profile * ||d_s||^2`, so `deviation_weight` reads as rows of
  evidence exactly as `PoolingStrength` does in `dp_gam_pooled`.

**The fit alternates exact convex solves.** Each solve is a bounded quadratic of
`dp_gam.solve_bounded_quadratic`, and the objective is the penalised sum of squares of the
substations' joint model (`theta^T A theta - 2 b^T theta`, in the units of `dp_gam_pooled`).

1. *Weights step.* The shapes are fixed. One solve finds every weight (non-negative), band mean,
   deviation, and other-family coefficient at once.
2. *Shape step, one shape at a time (Gauss-Seidel).* The weights and the other shapes are fixed.
   One solve finds shape `k` and, with it, every other coefficient.
3. *Renormalise.* Each shape is scaled to unit norm and its scale moved into the weights, which
   leaves every prediction unchanged.

A sweep is the K shape steps, a renormalisation, and a weights step. The fit stops when a sweep
lowers the objective by less than `tolerance` per training row, or after `max_sweeps`. Each
convex step lowers (or leaves unchanged) the objective that the step minimises. The renormalisation
does not touch the data term, and the weights step that follows it restores a joint optimum, so
the fit's final state is always the result of a weights step. The fit is convex in each step but
not jointly, so it ends at a local optimum that depends on the starting shapes.

**The orthogonality penalty keeps two shapes from collapsing into one.** Shape `k` is penalised by
`orthogonality * kappa * ||U_{-k}^T S_k||^2`, where `U_{-k}` is an orthonormal basis of the other
`K - 1` shapes. The penalty is a convex quadratic in `S_k` while the other shapes are fixed, and it
equals the sum of squared pairwise cosines when the other shapes are orthogonal (otherwise it is a
surrogate). `kappa` is the data term's curvature along a unit shape at the first weights step, in
the units of the objective: `kappa = g_profile * sum_s n_s * mean(w[s]^2)`, with `n_s` the
substation's training rows. So `orthogonality` reads as a multiple of the shape's own data
curvature, and it is the same number for every substation count and window. `kappa` is fixed for
the rest of the fit, which keeps every step's objective a fixed function. The shape ridge
`SHAPE_RIDGE` acts on the same shape.

**The starting shapes decide which local optimum the fit finds.** `initial_shapes` builds them
from the profiles of the per-site fit (`lambda = 0`): the top K right singular vectors of the
centred profiles (`svd`), or the least-squares shapes for random Dirichlet weights (`dirichlet`).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final, Literal

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from studies.dp_calendar import DAY_TYPES, HALF_HOURS_PER_DAY
from studies.dp_features import FeatureCache
from studies.dp_gam import (
    N_BANDS,
    GamFit,
    ReplicaResiduals,
    last_signal_season_year,
    predict_gam,
    replica_cut,
    replica_rows,
    solve_bounded_quadratic,
)
from studies.dp_gam_pooled import (
    FLEET_MEAN_RIDGE,
    POOLED_FAMILIES,
    PooledFamilyType,
    PooledReplica,
    PooledSite,
    PoolingStrength,
    _band_indicator,
    _build_block,
    _check_site,
    _family_scales,
    _pooled_offsets,
    _pooled_slices,
    _SiteBlock,
    centring_matrix,
    fit_pooled,
)

InitialShapesType = Literal["svd", "dirichlet"]
"""How the starting shapes are built from the per-site profiles."""

OtherFamilyType = Literal["holiday", "tent", "harmonics", "heating", "lighting"]
"""The pooled families other than the profile, which the library replaces."""

SHAPE_RIDGE: Final[float] = 1e-3
"""The ridge on each unit-norm shape, fixed and never tuned; it matches `FLEET_MEAN_RIDGE`."""

PROFILE_COLUMNS: Final[int] = len(DAY_TYPES) * HALF_HOURS_PER_DAY
"""The columns of the daily profile: 4 day types times 48 half-hours."""

DEFAULT_OTHER_STRENGTH: Final[float] = 3000.0
"""The pooling weight of every non-profile family: experiment 109's tuned `lambda`, in rows."""

MIN_SHAPE_NORM: Final[float] = 1e-12
"""A shape whose norm is below this is left as it is by the renormalisation (a dead shape)."""

RANK_TOLERANCE: Final[float] = 1e-6
"""Singular values below this share of the largest drop out of the other-shapes basis."""


@dataclass(frozen=True)
class LibraryConfig:
    """The settings of one library fit.

    Attributes:
        n_shapes: The number of shared shapes, K.
        deviation_weight: The weight on the deviation `delta_s`, in rows of evidence (the scale of
            `PoolingStrength`); a very large value gives a pure library.
        orthogonality: The orthogonality penalty as a multiple of the data curvature; 0 for none.
        other_strength: The pooling weight of the non-profile families (holidays, Christmas tent,
            harmonics, heating, lighting), in rows of evidence.
        weights_per_day_type: True for one weight per shape and day type.
        max_sweeps: The most sweeps.
        tolerance: The drop in the objective per training row, in squared points, below which the
            fit stops. A residual mean square is of order 1 to 10 points squared, so 1e-6 is a
            relative drop of 1e-6 to 1e-7.
    """

    n_shapes: int
    deviation_weight: float
    orthogonality: float = 0.0
    other_strength: float = DEFAULT_OTHER_STRENGTH
    weights_per_day_type: bool = False
    max_sweeps: int = 50
    tolerance: float = 1e-6

    def __post_init__(self) -> None:
        """Raise if a setting is outside its range."""
        if self.n_shapes < 1:
            msg = "a library needs at least one shape"
            raise ValueError(msg)
        if min(self.deviation_weight, self.orthogonality, self.other_strength) < 0.0:
            msg = "the deviation, orthogonality, and other-family weights must be non-negative"
            raise ValueError(msg)
        if self.max_sweeps < 1:
            msg = "a fit needs at least one sweep"
            raise ValueError(msg)

    @property
    def n_weights(self) -> int:
        """The weights per substation: K, or 4 K with one per day type."""
        return self.n_shapes * (len(DAY_TYPES) if self.weights_per_day_type else 1)


@dataclass(frozen=True)
class ShapeLibraryFit:
    """A fitted library and the substations' mixes.

    Attributes:
        fits: Each substation's `GamFit` (one member), so `dp_gam.predict_gam` predicts with it.
        shapes: The shared shapes, shape (K, 192), centred, unit norm.
        weights: Each substation's non-negative weights, shape (K,) or (4 K,).
        deviations: Each substation's deviation, shape (192,), centred.
        band_means: Each substation's free hour-band means, shape (N_BANDS,).
        fleet_mean: The fleet mean of the non-profile pooled families, in the order of
            `POOLED_FAMILIES` without the profile.
        objective_trace: The objective after the first weights step and after each sweep.
        n_sweeps: The sweeps run.
        converged: True if the last sweep's drop per training row was under the tolerance.
        kappa: The data curvature that scales the orthogonality penalty.
        config: The settings.
        cut: The cut.
    """

    fits: dict[str, GamFit]
    shapes: np.ndarray
    weights: dict[str, np.ndarray]
    deviations: dict[str, np.ndarray]
    band_means: dict[str, np.ndarray]
    fleet_mean: np.ndarray
    objective_trace: tuple[float, ...]
    n_sweeps: int
    converged: bool
    kappa: float
    config: LibraryConfig
    cut: datetime

    @property
    def n_free_parameters(self) -> int:
        """Shape entries (rank of the centred space), weights, and deviations that the fit holds."""
        rank = PROFILE_COLUMNS - N_BANDS
        n_sites = len(self.weights)
        return self.config.n_shapes * rank + n_sites * self.config.n_weights + n_sites * rank


@dataclass(frozen=True)
class _Problem:
    """The pieces of the joint objective that do not change between steps."""

    labels: tuple[str, ...]
    blocks: tuple[_SiteBlock, ...]
    a_matrices: tuple[np.ndarray, ...]
    b_vectors: tuple[np.ndarray, ...]
    constrained: tuple[np.ndarray, ...]
    scales: Mapping[PooledFamilyType, float]
    mu_slices: Mapping[OtherFamilyType, slice]
    other_slices: Mapping[OtherFamilyType, slice]
    n_mu: int
    basis: np.ndarray
    indicator: np.ndarray
    n_rows: tuple[int, ...]

    @property
    def n_basis(self) -> int:
        """The dimension of the centred profile space."""
        return self.basis.shape[1]


@dataclass
class _State:
    """The unknowns of a fit: the shapes, the weights, and what the last solve found."""

    shapes: np.ndarray
    weights: np.ndarray
    band_means: np.ndarray
    deviations: np.ndarray
    others: list[np.ndarray]
    fleet_mean: np.ndarray
    kappa: float = 0.0
    trace: list[float] = field(default_factory=list)


def _centred_basis() -> np.ndarray:
    """Return an orthonormal basis, shape (192, 192 - N_BANDS), of the centred profile space."""
    centring = centring_matrix(family="profile", size=PROFILE_COLUMNS)
    values, vectors = np.linalg.eigh(centring)
    return vectors[:, values > 0.5]


def _build_problem(*, cache: FeatureCache, sites: Sequence[PooledSite], cut: datetime) -> _Problem:
    """Build every substation's Gram matrix, penalty, and the fleet-wide scales."""
    labels = [site.site for site in sites]
    if not labels or len(set(labels)) != len(labels):
        msg = "a library fit needs at least one substation, and each label once"
        raise ValueError(msg)
    for site in sites:
        _check_site(site=site, cache=cache, cut=cut)
    blocks = [_build_block(cache=cache, site=site, cut=cut) for site in sites]
    slices = _pooled_slices(blocks=blocks)
    if slices["profile"] != slice(0, PROFILE_COLUMNS):
        msg = "the library needs the daily profile as the first columns of the layout"
        raise ValueError(msg)
    scales = _family_scales(blocks=blocks, slices=slices)
    offsets = _pooled_offsets(slices=slices)
    mu_slices = {
        family: slice(
            offsets[family].start - PROFILE_COLUMNS, offsets[family].stop - PROFILE_COLUMNS
        )
        for family in POOLED_FAMILIES
        if family != "profile"
    }
    other_slices = {
        family: slice(slices[family].start - PROFILE_COLUMNS, slices[family].stop - PROFILE_COLUMNS)
        for family in POOLED_FAMILIES
        if family != "profile"
    }
    a_matrices = tuple(block.gram.numpy() + block.penalty for block in blocks)
    b_vectors = tuple(block.rhs.numpy() - 0.5 * block.cost for block in blocks)
    constrained = tuple(block.layout.constrained[PROFILE_COLUMNS:] for block in blocks)
    return _Problem(
        labels=tuple(labels),
        blocks=tuple(blocks),
        a_matrices=a_matrices,
        b_vectors=b_vectors,
        constrained=constrained,
        scales=scales,
        mu_slices=mu_slices,
        other_slices=other_slices,
        n_mu=max(piece.stop for piece in mu_slices.values()),
        basis=_centred_basis(),
        indicator=_band_indicator(),
        n_rows=tuple(len(block.index) for block in blocks),
    )


def _day_type_of_column() -> np.ndarray:
    """Return the day type (0 to 3) of each of the 192 profile columns."""
    return np.arange(PROFILE_COLUMNS) // HALF_HOURS_PER_DAY


def _omega(*, weights: np.ndarray, config: LibraryConfig) -> np.ndarray:
    """Return one substation's weight on each shape at each profile column, shape (K, 192)."""
    if not config.weights_per_day_type:
        return np.repeat(weights[:, None], PROFILE_COLUMNS, axis=1)
    per_day = weights.reshape(len(DAY_TYPES), config.n_shapes)
    return per_day[_day_type_of_column()].T


def _library_profile(
    *, shapes: np.ndarray, weights: np.ndarray, config: LibraryConfig
) -> np.ndarray:
    """Return `sum_k w_k S_k` of one substation, shape (192,)."""
    return (_omega(weights=weights, config=config) * shapes).sum(axis=0)


def _weight_columns(*, shapes: np.ndarray, config: LibraryConfig) -> np.ndarray:
    """Return the matrix (192, n_weights) whose columns multiply the weights."""
    if not config.weights_per_day_type:
        return shapes.T
    day = _day_type_of_column()
    columns = np.zeros((PROFILE_COLUMNS, config.n_weights))
    for day_type in range(len(DAY_TYPES)):
        mask = (day == day_type).astype(np.float64)
        columns[:, day_type * config.n_shapes : (day_type + 1) * config.n_shapes] = (
            shapes.T * mask[:, None]
        )
    return columns


def _theta(*, problem: _Problem, state: _State, config: LibraryConfig, site: int) -> np.ndarray:
    """Return one substation's coefficient vector implied by the state."""
    profile = (
        problem.indicator @ state.band_means[site]
        + problem.basis @ state.deviations[site]
        + _library_profile(shapes=state.shapes, weights=state.weights[site], config=config)
    )
    return np.concatenate([profile, state.others[site]])


def _orthonormal_others(*, shapes: np.ndarray, k: int) -> np.ndarray:
    """Return an orthonormal basis (192, r) of the span of the shapes other than `k`."""
    others = np.delete(shapes, k, axis=0)
    if len(others) == 0:
        return np.zeros((PROFILE_COLUMNS, 0))
    left, singular, _ = np.linalg.svd(others.T, full_matrices=False)
    keep = singular > RANK_TOLERANCE * max(float(singular.max()), 1e-300)
    return left[:, keep]


def _shape_penalty(
    *, problem: _Problem, config: LibraryConfig, shapes: np.ndarray, k: int, kappa: float
) -> np.ndarray:
    """Return the quadratic form (basis dim, basis dim) that penalises shape `k`'s coefficients."""
    basis = problem.basis
    penalty = SHAPE_RIDGE * np.eye(problem.n_basis)
    others = _orthonormal_others(shapes=shapes, k=k)
    if config.orthogonality > 0.0 and others.shape[1] > 0:
        projected = basis.T @ others
        penalty = penalty + config.orthogonality * kappa * (projected @ projected.T)
    return penalty


@dataclass(frozen=True)
class _Unknowns:
    """Where each substation's unknowns, the fleet mean, and the shape sit in the step's vector.

    Attributes:
        sizes: Each substation's unknowns: band means, deviation, other families, and (in the
            weights step) weights, in that order.
        starts: The first index of each substation's unknowns, then the end of the last.
        n_other: Each substation's number of non-profile columns.
        mu_start: The first index of the fleet mean.
        shape_start: The first index of the shape coefficients (the end of the fleet mean).
        total: The length of the vector.
    """

    sizes: tuple[int, ...]
    starts: np.ndarray
    n_other: tuple[int, ...]
    mu_start: int
    shape_start: int
    total: int

    def site(self, *, index: int) -> slice:
        """Return the slice of one substation's unknowns."""
        return slice(int(self.starts[index]), int(self.starts[index + 1]))


def _layout_unknowns(
    *, problem: _Problem, config: LibraryConfig, state: _State, weights_step: bool
) -> _Unknowns:
    """Return the positions of the unknowns of a weights step or a shape step."""
    n_other = tuple(len(other) for other in state.others)
    n_weights = config.n_weights if weights_step else 0
    sizes = tuple(N_BANDS + problem.n_basis + count + n_weights for count in n_other)
    starts = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    mu_start = int(starts[-1])
    shape_start = mu_start + problem.n_mu
    total = shape_start + (0 if weights_step else problem.n_basis)
    return _Unknowns(
        sizes=sizes,
        starts=starts,
        n_other=n_other,
        mu_start=mu_start,
        shape_start=shape_start,
        total=total,
    )


def _own_map(
    *,
    problem: _Problem,
    config: LibraryConfig,
    state: _State,
    site: int,
    weights_step: bool,
    n_other: int,
) -> np.ndarray:
    """Return the map from one substation's own unknowns to its coefficient vector."""
    n_basis = problem.n_basis
    n_columns = len(problem.b_vectors[site])
    n_weights = config.n_weights if weights_step else 0
    own_map = np.zeros((n_columns, N_BANDS + n_basis + n_other + n_weights))
    own_map[:PROFILE_COLUMNS, :N_BANDS] = problem.indicator
    own_map[:PROFILE_COLUMNS, N_BANDS : N_BANDS + n_basis] = problem.basis
    other_start = N_BANDS + n_basis
    own_map[PROFILE_COLUMNS:, other_start : other_start + n_other] = np.eye(n_other)
    if weights_step:
        own_map[:PROFILE_COLUMNS, other_start + n_other :] = _weight_columns(
            shapes=state.shapes, config=config
        )
    return own_map


def _add_site_terms(
    *,
    problem: _Problem,
    config: LibraryConfig,
    state: _State,
    unknowns: _Unknowns,
    shape_index: int | None,
    site: int,
    system: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Add one substation's data term, deviation penalty, and pooling to the step's system."""
    matrix, rhs, constrained = system
    weights_step = shape_index is None
    n_basis = problem.n_basis
    a, b = problem.a_matrices[site], problem.b_vectors[site]
    n_other = unknowns.n_other[site]
    own_map = _own_map(
        problem=problem,
        config=config,
        state=state,
        site=site,
        weights_step=weights_step,
        n_other=n_other,
    )
    offset = np.zeros(len(b))
    if shape_index is not None:
        omega = _omega(weights=state.weights[site], config=config)
        offset[:PROFILE_COLUMNS] = np.delete(omega * state.shapes, shape_index, axis=0).sum(axis=0)
    residual_rhs = b - a @ offset
    free = unknowns.site(index=site)
    matrix[free, free] += own_map.T @ a @ own_map
    rhs[free] += own_map.T @ residual_rhs
    deviation = slice(free.start + N_BANDS, free.start + N_BANDS + n_basis)
    matrix[deviation, deviation] += (
        config.deviation_weight * problem.scales["profile"] * np.eye(n_basis)
    )
    other_start = free.start + N_BANDS + n_basis
    constrained[other_start : other_start + n_other] = problem.constrained[site]
    if weights_step:
        constrained[other_start + n_other : free.stop] = True
    else:
        shared = slice(unknowns.shape_start, unknowns.total)
        shape_map = np.zeros((len(b), n_basis))
        omega_k = _omega(weights=state.weights[site], config=config)[shape_index]
        shape_map[:PROFILE_COLUMNS] = omega_k[:, None] * problem.basis
        matrix[free, shared] += own_map.T @ a @ shape_map
        matrix[shared, free] += shape_map.T @ a @ own_map
        matrix[shared, shared] += shape_map.T @ a @ shape_map
        rhs[shared] += shape_map.T @ residual_rhs
    for family, other_slice in problem.other_slices.items():
        weight = config.other_strength * problem.scales[family]
        if weight == 0.0:
            continue
        o_cols = slice(other_start + other_slice.start, other_start + other_slice.stop)
        mu_slice = problem.mu_slices[family]
        mu_cols = slice(unknowns.mu_start + mu_slice.start, unknowns.mu_start + mu_slice.stop)
        coupling = weight * np.eye(other_slice.stop - other_slice.start)
        matrix[o_cols, o_cols] += coupling
        matrix[mu_cols, mu_cols] += coupling
        matrix[o_cols, mu_cols] -= coupling
        matrix[mu_cols, o_cols] -= coupling


def _solve_step(
    *,
    problem: _Problem,
    config: LibraryConfig,
    state: _State,
    shape_index: int | None,
) -> None:
    """Solve one convex step and write its solution into `state`.

    Args:
        problem: The fixed pieces of the objective.
        config: The settings.
        state: The current state, updated in place.
        shape_index: The shape to re-solve (weights and the other shapes stay fixed), or `None`
            for the weights step (shapes fixed, weights free and non-negative).
    """
    weights_step = shape_index is None
    n_basis = problem.n_basis
    unknowns = _layout_unknowns(
        problem=problem, config=config, state=state, weights_step=weights_step
    )
    matrix = np.zeros((unknowns.total, unknowns.total))
    rhs = np.zeros(unknowns.total)
    constrained = np.zeros(unknowns.total, dtype=bool)
    for site in range(len(problem.labels)):
        _add_site_terms(
            problem=problem,
            config=config,
            state=state,
            unknowns=unknowns,
            shape_index=shape_index,
            site=site,
            system=(matrix, rhs, constrained),
        )
    mu_all = slice(unknowns.mu_start, unknowns.shape_start)
    matrix[mu_all, mu_all] += FLEET_MEAN_RIDGE * np.eye(problem.n_mu)
    shared = slice(unknowns.shape_start, unknowns.total)
    if shape_index is not None:
        matrix[shared, shared] += _shape_penalty(
            problem=problem, config=config, shapes=state.shapes, k=shape_index, kappa=state.kappa
        )
    solution, _ = solve_bounded_quadratic(
        matrix=torch.as_tensor(matrix, dtype=torch.float64)[None],
        rhs=torch.as_tensor(rhs, dtype=torch.float64)[None],
        constrained=torch.as_tensor(constrained),
    )
    solved = solution[0].numpy()
    for site in range(len(problem.labels)):
        piece = solved[unknowns.site(index=site)]
        n_other = unknowns.n_other[site]
        state.band_means[site] = piece[:N_BANDS]
        state.deviations[site] = piece[N_BANDS : N_BANDS + n_basis]
        state.others[site] = piece[N_BANDS + n_basis : N_BANDS + n_basis + n_other]
        if weights_step:
            state.weights[site] = piece[N_BANDS + n_basis + n_other :]
    state.fleet_mean = solved[mu_all]
    if shape_index is not None:
        state.shapes[shape_index] = problem.basis @ solved[shared]


def _objective(
    *,
    problem: _Problem,
    config: LibraryConfig,
    state: _State,
    only_shape: int | None = None,
) -> float:
    """Return the penalised objective of the state.

    Args:
        problem: The fixed pieces.
        config: The settings.
        state: The state.
        only_shape: A shape index to count only that shape's ridge and orthogonality terms (the
            function a shape step minimises), or `None` to count every shape's.

    Returns:
        `sum_s theta^T A theta - 2 b^T theta`, plus the deviation, pooling, fleet-mean, shape
        ridge, and orthogonality penalties.
    """
    total = 0.0
    for site in range(len(problem.labels)):
        theta = _theta(problem=problem, state=state, config=config, site=site)
        total += float(
            theta @ problem.a_matrices[site] @ theta - 2.0 * problem.b_vectors[site] @ theta
        )
        total += (
            config.deviation_weight
            * problem.scales["profile"]
            * float(state.deviations[site] @ state.deviations[site])
        )
        for family, other_slice in problem.other_slices.items():
            weight = config.other_strength * problem.scales[family]
            gap = state.others[site][other_slice] - state.fleet_mean[problem.mu_slices[family]]
            total += weight * float(gap @ gap)
    total += FLEET_MEAN_RIDGE * float(state.fleet_mean @ state.fleet_mean)
    shape_indices = range(config.n_shapes) if only_shape is None else [only_shape]
    for k in shape_indices:
        coefficients = problem.basis.T @ state.shapes[k]
        penalty = _shape_penalty(
            problem=problem, config=config, shapes=state.shapes, k=k, kappa=state.kappa
        )
        total += float(coefficients @ penalty @ coefficients)
    return total


def _renormalise(*, state: _State, config: LibraryConfig) -> None:
    """Scale each shape to unit norm and move its scale into the weights (predictions unchanged)."""
    norms = np.linalg.norm(state.shapes, axis=1)
    norms = np.where(norms < MIN_SHAPE_NORM, 1.0, norms)
    state.shapes = state.shapes / norms[:, None]
    if config.weights_per_day_type:
        state.weights = state.weights * np.tile(norms, len(DAY_TYPES))[None, :]
    else:
        state.weights = state.weights * norms[None, :]


def _curvature(*, problem: _Problem, config: LibraryConfig, state: _State) -> float:
    """Return `kappa`: the profile family's scale times the weights' squared mass over the rows."""
    mass = 0.0
    for site, rows in enumerate(problem.n_rows):
        mass += rows * float(np.mean(_omega(weights=state.weights[site], config=config) ** 2))
    return problem.scales["profile"] * mass


def initial_shapes(
    *, profiles: np.ndarray, n_shapes: int, method: InitialShapesType = "svd", seed: int = 1
) -> np.ndarray:
    """Return starting shapes built from the substations' fitted profiles.

    Args:
        profiles: Shape (substations, 192): each substation's daily profile from a per-site fit.
        n_shapes: The number of shapes, K.
        method: `svd` for the top K right singular vectors of the centred profiles, with the sign
            chosen so the mean projection is positive; `dirichlet` for the least-squares shapes
            of random Dirichlet(1) weights scaled to each profile's size.
        seed: The seed of the random generator (used by `dirichlet`, and to fill shapes beyond
            the number of substations).

    Returns:
        Shape (K, 192), centred, unit norm.
    """
    rng = np.random.default_rng(seed)
    centring = centring_matrix(family="profile", size=PROFILE_COLUMNS)
    centred = profiles @ centring
    if method == "svd":
        _, _, right = np.linalg.svd(centred, full_matrices=False)
        shapes = right[:n_shapes].copy()
        for k in range(len(shapes)):
            if float((centred @ shapes[k]).mean()) < 0.0:
                shapes[k] = -shapes[k]
    else:
        sizes = np.linalg.norm(centred, axis=1)
        weights = rng.dirichlet(np.ones(n_shapes), size=len(profiles)) * sizes[:, None]
        shapes = np.linalg.lstsq(weights, centred, rcond=None)[0]
    if len(shapes) < n_shapes:
        extra = rng.normal(size=(n_shapes - len(shapes), PROFILE_COLUMNS))
        shapes = np.vstack([shapes, extra])
    shapes = shapes @ centring
    norms = np.linalg.norm(shapes, axis=1)
    fallback = rng.normal(size=(n_shapes, PROFILE_COLUMNS)) @ centring
    shapes = np.where((norms < MIN_SHAPE_NORM)[:, None], fallback, shapes)
    return shapes / np.linalg.norm(shapes, axis=1, keepdims=True)


def fit_library(
    *,
    cache: FeatureCache,
    sites: Sequence[PooledSite],
    cut: datetime,
    config: LibraryConfig,
    start_shapes: np.ndarray | None = None,
    init_method: InitialShapesType = "svd",
    seed: int = 1,
) -> ShapeLibraryFit:
    """Fit the shared-shape library jointly across substations.

    Args:
        cache: The cache.
        sites: The substations, each with its own spec, window, and pinned capacity (the same
            inputs `dp_gam_pooled.fit_pooled` takes).
        cut: The cut; no training row ends after it.
        config: The settings.
        start_shapes: Shape (K, 192) starting shapes, or `None` to build them with
            `initial_shapes` from the per-site fit of the same substations.
        init_method: How to build the starting shapes when `start_shapes` is `None`.
        seed: The seed for `init_method`.

    Returns:
        The fit.

    Raises:
        ValueError: If there are no substations, a label repeats, a substation's inputs cannot
            join a joint fit, or `start_shapes` has the wrong shape.
    """
    problem = _build_problem(cache=cache, sites=sites, cut=cut)
    if start_shapes is None:
        per_site = fit_pooled(
            cache=cache, sites=sites, cut=cut, strengths=[PoolingStrength.uniform(value=0.0)]
        )
        profiles = np.stack([per_site.fits[site.site].theta[0, :PROFILE_COLUMNS] for site in sites])
        start_shapes = initial_shapes(
            profiles=profiles, n_shapes=config.n_shapes, method=init_method, seed=seed
        )
    if start_shapes.shape != (config.n_shapes, PROFILE_COLUMNS):
        msg = f"expected starting shapes of shape {(config.n_shapes, PROFILE_COLUMNS)}"
        raise ValueError(msg)
    n_sites = len(sites)
    state = _State(
        shapes=start_shapes.copy(),
        weights=np.zeros((n_sites, config.n_weights)),
        band_means=np.zeros((n_sites, N_BANDS)),
        deviations=np.zeros((n_sites, problem.n_basis)),
        others=[np.zeros(len(block.rhs) - PROFILE_COLUMNS) for block in problem.blocks],
        fleet_mean=np.zeros(problem.n_mu),
    )
    _solve_step(problem=problem, config=config, state=state, shape_index=None)
    state.kappa = _curvature(problem=problem, config=config, state=state)
    previous = _objective(problem=problem, config=config, state=state)
    state.trace.append(previous)
    converged = False
    sweeps = 0
    total_rows = float(sum(problem.n_rows))
    for _ in range(config.max_sweeps):
        sweeps += 1
        for k in range(config.n_shapes):
            _solve_step(problem=problem, config=config, state=state, shape_index=k)
        _renormalise(state=state, config=config)
        _solve_step(problem=problem, config=config, state=state, shape_index=None)
        current = _objective(problem=problem, config=config, state=state)
        state.trace.append(current)
        drop = (previous - current) / total_rows
        previous = current
        if drop < config.tolerance:
            converged = True
            break
    return _assemble(
        cache=cache,
        sites=sites,
        cut=cut,
        config=config,
        problem=problem,
        state=state,
        converged=converged,
        sweeps=sweeps,
    )


def _assemble(
    *,
    cache: FeatureCache,
    sites: Sequence[PooledSite],
    cut: datetime,
    config: LibraryConfig,
    problem: _Problem,
    state: _State,
    converged: bool,
    sweeps: int,
) -> ShapeLibraryFit:
    """Turn the final state into per-substation `GamFit`s and the library's readouts."""
    fits: dict[str, GamFit] = {}
    for position, block in enumerate(problem.blocks):
        theta = _theta(problem=problem, state=state, config=config, site=position)
        fits[problem.labels[position]] = GamFit(
            layout=block.layout,
            theta=theta[None, :],
            cut=cut,
            n_rows=len(block.index),
            site=problem.labels[position],
            last_season_year=last_signal_season_year(
                cache=cache,
                site=problem.labels[position],
                index=block.index,
                cut=cut,
                pinned=block.inputs.pinned,
            ),
            pinned=block.inputs.pinned,
        )
    return ShapeLibraryFit(
        fits=fits,
        shapes=state.shapes.copy(),
        weights={label: state.weights[i].copy() for i, label in enumerate(problem.labels)},
        deviations={
            label: problem.basis @ state.deviations[i] for i, label in enumerate(problem.labels)
        },
        band_means={label: state.band_means[i].copy() for i, label in enumerate(problem.labels)},
        fleet_mean=state.fleet_mean.copy(),
        objective_trace=tuple(state.trace),
        n_sweeps=sweeps,
        converged=converged,
        kappa=state.kappa,
        config=config,
        cut=cut,
    )


def library_replica_residuals(
    *,
    cache: FeatureCache,
    sites: Sequence[PooledSite],
    cut: datetime,
    config: LibraryConfig,
    init_method: InitialShapesType = "svd",
    seed: int = 1,
) -> dict[str, PooledReplica]:
    """Return every substation's residuals from a library fit 12 months before the cut.

    The sites must carry the windows and pinned capacities built for the replica's own cut,
    `replica_cut(cut=cut)`, as `dp_gam_pooled.pooled_replica_residuals` requires.

    Args:
        cache: The cache.
        sites: The substations as they are at the replica's cut.
        cut: The cut of the final fit.
        config: The settings.
        init_method: How the replica fit builds its starting shapes.
        seed: The seed for `init_method`.

    Returns:
        Each substation's replica residuals over the rows after the replica's cut and up to `cut`,
        outside the switching masks, with the rows it trained on.
    """
    fit_cut = replica_cut(cut=cut)
    joint = fit_library(
        cache=cache, sites=sites, cut=fit_cut, config=config, init_method=init_method, seed=seed
    )
    out: dict[str, PooledReplica] = {}
    for site in sites:
        features = cache.sites[site.site]
        index = replica_rows(cache=cache, site=site.site, fit_cut=fit_cut, cut=cut)
        predicted = predict_gam(cache=cache, fit=joint.fits[site.site], index=index)
        out[site.site] = PooledReplica(
            residuals=ReplicaResiduals(
                residuals=features.y[index] - predicted,
                band=cache.calendar.band[index],
                quarter=cache.calendar.quarter[index],
                index=index,
                fit_cut=fit_cut,
                switching=features.switching_mask[index],
            ),
            n_training_rows=joint.fits[site.site].n_rows,
        )
    return out


def cosine_matrix(*, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the signed cosines between the rows of two shape arrays.

    Args:
        first: Shape (K1, 192).
        second: Shape (K2, 192).

    Returns:
        `first_i . second_j / (|first_i| |second_j|)`; a zero row gives zeros.
    """
    norms_first = np.linalg.norm(first, axis=1)
    norms_second = np.linalg.norm(second, axis=1)
    denominator = np.outer(norms_first, norms_second)
    return np.divide(
        first @ second.T, denominator, out=np.zeros(denominator.shape), where=denominator > 0.0
    )


def match_shapes(*, first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match the shapes of two fits one to one by the Hungarian algorithm on the signed cosine.

    A sign flip is not a match, because `S` and `-S` differ under non-negative weights.

    Args:
        first: Shape (K, 192).
        second: Shape (K, 192).

    Returns:
        `(order, cosines)`: `order[i]` is the row of `second` matched to row `i` of `first`, and
        `cosines[i]` is their signed cosine.

    Raises:
        ValueError: If the two arrays hold a different number of shapes.
    """
    if first.shape[0] != second.shape[0]:
        msg = "the two fits must hold the same number of shapes"
        raise ValueError(msg)
    cosines = cosine_matrix(first=first, second=second)
    rows, columns = linear_sum_assignment(-cosines)
    return columns[np.argsort(rows)], cosines[rows, columns][np.argsort(rows)]


def largest_pair_cosine(*, shapes: np.ndarray) -> float:
    """Return the largest absolute cosine between two different shapes (0 for a single shape)."""
    if len(shapes) < 2:
        return 0.0
    cosines = np.abs(cosine_matrix(first=shapes, second=shapes))
    np.fill_diagonal(cosines, 0.0)
    return float(cosines.max())


__all__ = [
    "DEFAULT_OTHER_STRENGTH",
    "PROFILE_COLUMNS",
    "SHAPE_RIDGE",
    "InitialShapesType",
    "LibraryConfig",
    "ShapeLibraryFit",
    "cosine_matrix",
    "fit_library",
    "initial_shapes",
    "largest_pair_cosine",
    "library_replica_residuals",
    "match_shapes",
]
