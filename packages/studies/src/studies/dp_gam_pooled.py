"""The convex model of `studies.dp_gam`, fitted jointly across substations with partial pooling.

**Each substation keeps its own coefficients, pulled towards a fleet mean.** For substation `s` and
pooled coefficient family `f`, the coefficient vector is `theta[s, f] = mu[f] + delta[s, f]`: a
fleet mean plus a deviation. The joint objective is the sum of the per-site objectives of `dp_gam`
plus a ridge penalty on the deviations,

    sum_s [ ||y_s - X_s theta_s||^2 + theta_s^T P_s theta_s ]
        + lambda_f * g_f * ||C_f (theta[s, f] - mu[f])||^2   (summed over s and f)
        + 1e-3 * ||mu||^2

where `P_s` is the base penalty of `dp_gam.penalty_matrix`, `g_f` is the family's scale, and
`C_f` is a centring projector. The objective is a positive-definite convex quadratic. Sign
constraints stay box constraints on `theta_s` (heating slopes and capacity increments are
non-negative), and `mu` is unconstrained, so the joint fit has one optimum for every `lambda`. At
`lambda = 0` the fit is `dp_gam.fit_gam` per substation, because the design is `dp_gam`'s and no
column is added.

- **Pooled families:** the profile, holiday periods, Christmas tent, harmonics, heating, and
  lighting. Capacity (and the COVID indicators) are never pooled.
- **Centring `C_f`:** the identity for every family except the profile. For the profile it removes
  the mean of each hour band over that band's columns, so the profile's shape pools and each
  substation's band means stay free. A substation's size therefore never pools.
- **Scale `g_f`:** the mean diagonal of the family's Gram matrix per training row, over the rows the
  fit itself trains on. The penalty weight `lambda` then reads as rows of evidence in prediction
  space: a substation with far more training rows than `lambda` keeps its own predictions, and one
  with far fewer is shrunk to the fleet mean. The reading is approximate. A family whose columns
  are unequally active (the holiday periods and the Christmas tent) is shrunk less than
  `n / (n + lambda)` where its columns are busier than the family's mean, and the four smoothing
  windows of each heating band are nearly collinear, so their contrasts are shrunk harder than the
  predictions are.
- **Complete pooling (`tied=True`)** is exact tying, built as its own design with one coefficient
  vector per pooled family and a free band mean per substation. It is never a huge `lambda`, which
  would push the condition number towards the limit of float64.
- **Mixed capacity modes:** each substation brings its own `GamSpec`, window, and pinned capacity,
  so one joint fit holds substations whose capacity is pinned beside substations whose capacity is
  fitted from their own rows. A pinned capacity shifts its own substation's target and so its
  coefficients, and at `lambda > 0` those move the fleet mean and every other substation's
  forecast. A pinned capacity read from a vintage fitted long after the cut is refused for that
  reason, unless the caller marks the arm as an oracle.

**The spread follows `dp_gam`.** `pooled_replica_residuals` fits the joint model 12 months before
the cut and predicts the 12 months that follow. A substation with a full history takes the centred
`dp_gam.ResidualSpread` of its own replica residuals. A substation with a short history takes
`short_history_spread`, which uses only the residuals of each other substation in the replica fit in
which that other substation was the short one, with the same weeks of history, so the spread
carries neither the short substation's own residuals nor those of a full-history arm.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Final, Literal

import numpy as np
import torch

from studies.dp_calendar import DAY_TYPES, HALF_HOURS_PER_DAY, band_of_hour
from studies.dp_features import FeatureCache
from studies.dp_gam import (
    N_BANDS,
    GamFit,
    GamLayout,
    GamSpec,
    ReplicaResiduals,
    ResidualSpread,
    TrainingWindow,
    accumulate_gram,
    check_pinned_capacity,
    last_signal_season_year,
    penalty_matrix,
    predict_gam,
    prepare_fit,
    replica_cut,
    replica_rows,
    solve_bounded_quadratic,
    training_rows,
)
from studies.dp_pinned_capacity import PinnedCapacity

PooledFamilyType = Literal["profile", "holiday", "tent", "harmonics", "heating", "lighting"]
"""The coefficient families that share a fleet mean."""

POOLED_FAMILIES: Final[tuple[PooledFamilyType, ...]] = (
    "profile",
    "holiday",
    "tent",
    "harmonics",
    "heating",
    "lighting",
)
"""The coefficient families that share a fleet mean; capacity and the COVID flag do not."""

HEATING_FAMILY: Final[PooledFamilyType] = "heating"
"""The family whose strength `PoolingStrength.heating` sets; every other family uses `other`."""

WEEK_ROWS: Final[int] = 7 * HALF_HOURS_PER_DAY
"""Half-hours in a week."""

FLEET_MEAN_RIDGE: Final[float] = 1e-3
"""The ridge on every fleet-mean coordinate.

It matters only where the deviation penalty does not reach: the profile band means, which no
substation's coefficients read. Elsewhere it is negligible, because the penalty's own curvature,
`lambda * g_f * S`, is at least about 0.6 for `lambda >= 100`; below `lambda` of about 1 it would
shrink the fleet mean visibly. The ridge matches `GamSpec.profile_ridge`. A smaller one leaves a
condition number near 1e11 and limits a comparison with a dense solve to about 1e-6.
"""


@dataclass(frozen=True)
class PooledSite:
    """One substation's inputs to a joint fit.

    Attributes:
        site: The substation's label in the feature cache.
        spec: The substation's base penalties and capacity mode. The drift must be `none`, the COVID
            flag off, and the final knot free, because the fleet mean is defined on the columns that
            every substation shares.
        window: Which rows the substation trains on. A short-history substation passes a window that
            starts at its first retained week.
        season_years: The season-years with a capacity increment (empty effect for pinned capacity).
        pinned: The pinned capacity built for the fit's cut, required exactly when
            `spec.capacity == "pinned"`. Its source must not be `growth_prior`, a vintage fitted
            more than six weeks after the cut: in a joint fit that hindsight level would move the
            fleet mean and so every other substation's forecast. Give such a substation a fitted
            capacity instead.
        allow_late_vintage: True to accept a `growth_prior` pinned capacity, for an arm that is
            disclosed as an oracle.
    """

    site: str
    spec: GamSpec
    window: TrainingWindow
    season_years: tuple[int, ...]
    pinned: PinnedCapacity | None = None
    allow_late_vintage: bool = False


@dataclass(frozen=True)
class PoolingStrength:
    """The deviation penalty's weight, in rows of evidence, for one member of a batch.

    Attributes:
        heating: The weight on the heating slopes' deviations.
        other: The weight on every other pooled family's deviations.
    """

    heating: float
    other: float

    @staticmethod
    def uniform(*, value: float) -> PoolingStrength:
        """Return a strength that is the same for every family.

        Args:
            value: The weight, in rows of evidence.

        Returns:
            The strength.
        """
        return PoolingStrength(heating=value, other=value)

    def of(self, *, family: PooledFamilyType) -> float:
        """Return the weight of one family.

        Args:
            family: One of `POOLED_FAMILIES`.

        Returns:
            `heating` for the heating family, else `other`.
        """
        return self.heating if family == HEATING_FAMILY else self.other


@dataclass(frozen=True)
class PooledFit:
    """A joint fit: one `GamFit` per substation, each holding one member per strength.

    Attributes:
        fits: Each substation's fit; member `b` of every fit belongs to `strengths[b]`.
        fleet_mean: The fleet mean of the pooled families, shape (batch, m), in the order of
            `POOLED_FAMILIES`; for a tied fit, the shared coefficient vector. Its profile block is
            the centred shape: it carries no band means, which are about 0 and set only by the
            fleet-mean ridge.
        strengths: The deviation penalty's weight of each batch member.
        scales: The scale `g_f` of each pooled family.
        tied: True for complete pooling.
        cut: The cut, shared by every substation.
    """

    fits: dict[str, GamFit]
    fleet_mean: np.ndarray
    strengths: tuple[PoolingStrength, ...]
    scales: dict[PooledFamilyType, float]
    tied: bool
    cut: datetime


@dataclass(frozen=True)
class _SiteBlock:
    """The pieces of one substation's objective: its layout, rows, Gram matrix, and penalty."""

    inputs: PooledSite
    layout: GamLayout
    index: np.ndarray
    gram: torch.Tensor
    rhs: torch.Tensor
    penalty: np.ndarray
    cost: np.ndarray


def _check_site(*, site: PooledSite, cache: FeatureCache, cut: datetime) -> None:
    """Raise if a substation's inputs cannot join a joint fit at `cut`."""
    spec = site.spec
    if spec.drift != "none" or spec.covid_flag or spec.final_knot != "free":
        msg = f"{site.site}: a pooled fit needs drift 'none', no COVID flag, and a free final knot"
        raise ValueError(msg)
    check_pinned_capacity(
        capacity=spec.capacity,
        pinned=site.pinned,
        site=site.site,
        features=cache.sites[site.site],
        cut=cut,
    )
    if (
        site.pinned is not None
        and site.pinned.source == "growth_prior"
        and not site.allow_late_vintage
    ):
        msg = (
            f"{site.site}: the pinned capacity is a vintage fitted long after {cut}; give the "
            "substation a fitted capacity, or set allow_late_vintage for an oracle arm"
        )
        raise ValueError(msg)


def _build_block(*, cache: FeatureCache, site: PooledSite, cut: datetime) -> _SiteBlock:
    """Return a substation's layout, training rows, Gram matrix, and base penalty."""
    layout, index = prepare_fit(
        cache=cache,
        site=site.site,
        spec=site.spec,
        cut=cut,
        window=site.window,
        season_years=site.season_years,
    )
    gram, rhs = accumulate_gram(
        cache=cache,
        features=cache.sites[site.site],
        layout=layout,
        index=index,
        pinned=site.pinned,
    )
    penalty, cost = penalty_matrix(layout=layout, n_rows=len(index))
    return _SiteBlock(
        inputs=site,
        layout=layout,
        index=index,
        gram=gram[0],
        rhs=rhs[0],
        penalty=penalty,
        cost=cost,
    )


def _band_of_profile_column() -> np.ndarray:
    """Return the hour band of each profile column (day type major, half-hour minor)."""
    band_of_half_hour = band_of_hour(hour=np.arange(HALF_HOURS_PER_DAY) // 2)
    return np.tile(band_of_half_hour, len(DAY_TYPES))


def _band_indicator() -> np.ndarray:
    """Return the 0/1 matrix of shape (profile columns, bands) that marks each column's band."""
    band = _band_of_profile_column()
    return (band[:, None] == np.arange(N_BANDS)[None, :]).astype(np.float64)


def centring_matrix(*, family: PooledFamilyType, size: int) -> np.ndarray:
    """Return the centring projector `C_f` of a pooled family.

    Args:
        family: One of `POOLED_FAMILIES`.
        size: The family's number of columns.

    Returns:
        The identity, except for the profile, where `I - Pi` and `Pi` averages each column's hour
        band over the band's columns.
    """
    if family != "profile":
        return np.eye(size)
    indicator = _band_indicator()
    band_sizes = indicator.sum(axis=0)
    average = indicator @ (indicator / band_sizes[None, :]).T
    return np.eye(size) - average


def _pooled_slices(*, blocks: Sequence[_SiteBlock]) -> dict[PooledFamilyType, slice]:
    """Return the pooled families' column slices.

    `make_layout` places the pooled families before the capacity and COVID columns, so the slices
    are the same for every substation whatever its capacity mode.
    """
    reference = blocks[0].layout.slices
    return {family: reference[family] for family in POOLED_FAMILIES}


def _pooled_offsets(*, slices: Mapping[PooledFamilyType, slice]) -> dict[PooledFamilyType, slice]:
    """Return each pooled family's slice within the concatenated fleet-mean vector."""
    offsets: dict[PooledFamilyType, slice] = {}
    start = 0
    for family in POOLED_FAMILIES:
        size = slices[family].stop - slices[family].start
        offsets[family] = slice(start, start + size)
        start += size
    return offsets


def _family_scales(
    *, blocks: Sequence[_SiteBlock], slices: Mapping[PooledFamilyType, slice]
) -> dict[PooledFamilyType, float]:
    """Return `g_f`, the mean diagonal of each pooled family's Gram matrix per training row.

    Args:
        blocks: The substations' blocks, whose Gram matrices come from their training rows only.
        slices: The pooled families' column slices.

    Returns:
        The scale of each family, pooled over every substation's rows and columns of the family.
    """
    total_rows = sum(len(block.index) for block in blocks)
    scales: dict[PooledFamilyType, float] = {}
    for family in POOLED_FAMILIES:
        columns = slices[family]
        diagonal_sum = sum(float(torch.diagonal(block.gram)[columns].sum()) for block in blocks)
        scales[family] = diagonal_sum / ((columns.stop - columns.start) * total_rows)
    return scales


def fit_pooled(
    *,
    cache: FeatureCache,
    sites: Sequence[PooledSite],
    cut: datetime,
    strengths: Sequence[PoolingStrength] = (),
    tied: bool = False,
    device: torch.device | str = "cpu",
) -> PooledFit:
    """Fit the partially pooled convex model of several substations jointly.

    Args:
        cache: The cache.
        sites: The substations, each with its own spec, window, and pinned capacity.
        cut: The cut; no training row ends after it.
        strengths: One batch member per entry; at least one unless `tied` is True, when none.
        tied: True for complete pooling by exact tying, which has no deviation penalty and so
            takes no strengths.
        device: Where the solve runs.

    Returns:
        The joint fit.

    Raises:
        ValueError: If there are no substations, a label repeats, a substation's inputs cannot
            join a joint fit, a partial fit has no strength, or a tied fit has one.
    """
    labels = [site.site for site in sites]
    if not labels or len(set(labels)) != len(labels):
        msg = "a joint fit needs at least one substation, and each label once"
        raise ValueError(msg)
    if tied == bool(strengths):
        msg = "a partial fit needs at least one strength, and a tied fit takes none"
        raise ValueError(msg)
    for site in sites:
        _check_site(site=site, cache=cache, cut=cut)
    blocks = [_build_block(cache=cache, site=site, cut=cut) for site in sites]
    slices = _pooled_slices(blocks=blocks)
    scales = _family_scales(blocks=blocks, slices=slices)
    if tied:
        thetas, fleet_mean, violation = _solve_tied(blocks=blocks, slices=slices, device=device)
    else:
        thetas, fleet_mean, violation = _solve_partial(
            blocks=blocks, slices=slices, scales=scales, strengths=strengths, device=device
        )
    fits = {
        block.inputs.site: GamFit(
            layout=block.layout,
            theta=thetas[block.inputs.site],
            cut=cut,
            n_rows=len(block.index),
            site=block.inputs.site,
            last_season_year=last_signal_season_year(
                cache=cache,
                site=block.inputs.site,
                index=block.index,
                cut=cut,
                pinned=block.inputs.pinned,
            ),
            violation=violation,
            pinned=block.inputs.pinned,
        )
        for block in blocks
    }
    return PooledFit(
        fits=fits,
        fleet_mean=fleet_mean,
        strengths=tuple(strengths),
        scales=scales,
        tied=tied,
        cut=cut,
    )


def _solve_partial(
    *,
    blocks: Sequence[_SiteBlock],
    slices: Mapping[PooledFamilyType, slice],
    scales: Mapping[PooledFamilyType, float],
    strengths: Sequence[PoolingStrength],
    device: torch.device | str,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Solve the joint problem in the coefficients `[theta_1, ..., theta_S, mu]`."""
    offsets = _pooled_offsets(slices=slices)
    sizes = [block.layout.n_columns for block in blocks]
    starts = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    n_mu = offsets[POOLED_FAMILIES[-1]].stop
    mu_start = int(starts[-1])
    total = mu_start + n_mu
    batch = len(strengths)
    base = torch.zeros((total, total), dtype=torch.float64, device=device)
    rhs = torch.zeros((total,), dtype=torch.float64, device=device)
    constrained = np.zeros(total, dtype=bool)
    for block, start in zip(blocks, starts[:-1], strict=True):
        stop = int(start) + block.layout.n_columns
        base[start:stop, start:stop] = block.gram.to(device) + torch.as_tensor(
            block.penalty, dtype=torch.float64, device=device
        )
        rhs[start:stop] = block.rhs.to(device) - 0.5 * torch.as_tensor(
            block.cost, dtype=torch.float64, device=device
        )
        constrained[start:stop] = block.layout.constrained
    base[mu_start:total, mu_start:total] += FLEET_MEAN_RIDGE * torch.eye(
        n_mu, dtype=torch.float64, device=device
    )
    matrix = base[None].repeat(batch, 1, 1)
    for family in POOLED_FAMILIES:
        size = slices[family].stop - slices[family].start
        centring = torch.as_tensor(
            centring_matrix(family=family, size=size), dtype=torch.float64, device=device
        )
        mu_columns = slice(mu_start + offsets[family].start, mu_start + offsets[family].stop)
        for member, strength in enumerate(strengths):
            weight = strength.of(family=family) * scales[family]
            if weight == 0.0:
                continue
            coupling = weight * centring
            for start in starts[:-1]:
                theta_columns = slice(
                    int(start) + slices[family].start, int(start) + slices[family].stop
                )
                matrix[member, theta_columns, theta_columns] += coupling
                matrix[member, mu_columns, mu_columns] += coupling
                matrix[member, theta_columns, mu_columns] -= coupling
                matrix[member, mu_columns, theta_columns] -= coupling
    solution, violation = solve_bounded_quadratic(
        matrix=matrix,
        rhs=rhs[None].repeat(batch, 1),
        constrained=torch.as_tensor(constrained),
    )
    solved = solution.cpu().numpy()
    thetas = {
        block.inputs.site: solved[:, int(start) : int(start) + block.layout.n_columns]
        for block, start in zip(blocks, starts[:-1], strict=True)
    }
    return thetas, solved[:, mu_start:total], violation.cpu().numpy()


def _tying_map(
    *, block: _SiteBlock, slices: Mapping[PooledFamilyType, slice], n_shared: int
) -> np.ndarray:
    """Return the matrix mapping a substation's own free columns to its `theta`.

    Its columns are the shared fleet vector (`n_shared` entries), then the substation's band means
    (`N_BANDS`), then its own non-pooled columns.
    """
    layout = block.layout
    offsets = _pooled_offsets(slices=slices)
    own = np.flatnonzero(~_pooled_mask(block=block, slices=slices))
    mapping = np.zeros((layout.n_columns, n_shared + N_BANDS + len(own)))
    for family in POOLED_FAMILIES:
        rows = slices[family]
        size = rows.stop - rows.start
        mapping[rows, offsets[family]] = centring_matrix(family=family, size=size)
    mapping[slices["profile"], n_shared : n_shared + N_BANDS] = _band_indicator()
    mapping[own, n_shared + N_BANDS + np.arange(len(own))] = 1.0
    return mapping


def _solve_tied(
    *,
    blocks: Sequence[_SiteBlock],
    slices: Mapping[PooledFamilyType, slice],
    device: torch.device | str,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Solve complete pooling: one coefficient vector per pooled family, band means free."""
    offsets = _pooled_offsets(slices=slices)
    n_shared = offsets[POOLED_FAMILIES[-1]].stop
    maps = [_tying_map(block=block, slices=slices, n_shared=n_shared) for block in blocks]
    widths = [mapping.shape[1] - n_shared for mapping in maps]
    starts = np.concatenate([[0], np.cumsum(widths)]).astype(int) + n_shared
    total = int(starts[-1])
    matrix = np.zeros((total, total))
    rhs = np.zeros(total)
    constrained = np.zeros(total, dtype=bool)
    constrained[offsets[HEATING_FAMILY]] = True
    for block, mapping, start in zip(blocks, maps, starts[:-1], strict=True):
        own_columns = np.concatenate(
            [np.arange(n_shared), int(start) + np.arange(mapping.shape[1] - n_shared)]
        )
        gram = block.gram.numpy() + block.penalty
        local = mapping.T @ gram @ mapping
        matrix[np.ix_(own_columns, own_columns)] += local
        rhs[own_columns] += mapping.T @ (block.rhs.numpy() - 0.5 * block.cost)
        non_pooled = np.flatnonzero(~_pooled_mask(block=block, slices=slices))
        first_own = int(start) + N_BANDS
        constrained[first_own : first_own + len(non_pooled)] = block.layout.constrained[non_pooled]
    matrix[np.arange(n_shared), np.arange(n_shared)] += FLEET_MEAN_RIDGE
    solution, violation = solve_bounded_quadratic(
        matrix=torch.as_tensor(matrix, dtype=torch.float64, device=device)[None],
        rhs=torch.as_tensor(rhs, dtype=torch.float64, device=device)[None],
        constrained=torch.as_tensor(constrained),
    )
    solved = solution.cpu().numpy()
    thetas = {}
    for block, mapping, start in zip(blocks, maps, starts[:-1], strict=True):
        free = np.concatenate(
            [
                solved[:, :n_shared],
                solved[:, int(start) : int(start) + mapping.shape[1] - n_shared],
            ],
            axis=1,
        )
        thetas[block.inputs.site] = free @ mapping.T
    return thetas, solved[:, :n_shared], violation.cpu().numpy()


def _pooled_mask(*, block: _SiteBlock, slices: Mapping[PooledFamilyType, slice]) -> np.ndarray:
    """Return True for the columns of a substation that belong to a pooled family."""
    mask = np.zeros(block.layout.n_columns, dtype=bool)
    for family in POOLED_FAMILIES:
        mask[slices[family]] = True
    return mask


@dataclass(frozen=True)
class PooledReplica:
    """One substation's residuals from a replica fit, and the rows that fit trained on.

    Attributes:
        residuals: The substation's replica residuals.
        n_training_rows: The substation's training rows in the replica fit.
    """

    residuals: ReplicaResiduals
    n_training_rows: int


def pooled_replica_residuals(
    *,
    cache: FeatureCache,
    sites: Sequence[PooledSite],
    cut: datetime,
    strength: PoolingStrength,
    tied: bool = False,
) -> dict[str, PooledReplica]:
    """Return every substation's residuals from a joint fit 12 months before the cut.

    The sites must carry the windows and pinned capacities built for the replica's own cut,
    `replica_cut(cut=cut)`, as `dp_gam.replica_residuals` builds them for one substation.

    Args:
        cache: The cache.
        sites: The substations as they are at the replica's cut.
        cut: The cut of the final fit.
        strength: The deviation penalty's weight.
        tied: True for complete pooling.

    Returns:
        Each substation's replica residuals over the rows after the replica's cut and up to `cut`,
        outside the switching masks, with the rows it trained on in the replica fit.
    """
    fit_cut = replica_cut(cut=cut)
    joint = fit_pooled(cache=cache, sites=sites, cut=fit_cut, strengths=[strength], tied=tied)
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


@dataclass(frozen=True)
class ShortHistoryReplica:
    """The replica residuals of one substation that was the short one in its replica fit.

    Build one with `short_history_replica`, which takes the training rows from the replica fit
    itself.

    Attributes:
        short_site: The substation that held only `weeks_of_history` weeks in the replica fit.
        weeks_of_history: How many weeks of history the short substation held.
        residuals: The short substation's own replica residuals.
        n_training_rows: The short substation's training rows in the replica fit.
    """

    short_site: str
    weeks_of_history: int
    residuals: ReplicaResiduals
    n_training_rows: int


def short_history_replica(
    *, replicas: Mapping[str, PooledReplica], short_site: str, weeks_of_history: int
) -> ShortHistoryReplica:
    """Return one substation's entry from a replica fit in which it held a short history.

    Args:
        replicas: The output of `pooled_replica_residuals` for a fit with `short_site` short.
        short_site: The substation that held the short history.
        weeks_of_history: The weeks of history it held (a month counts as 4.33 weeks, rounded).

    Returns:
        The entry, carrying the substation's training rows in that fit.
    """
    replica = replicas[short_site]
    return ShortHistoryReplica(
        short_site=short_site,
        weeks_of_history=weeks_of_history,
        residuals=replica.residuals,
        n_training_rows=replica.n_training_rows,
    )


def short_history_spread(
    *, replicas: Sequence[ShortHistoryReplica], held_out: str, weeks_of_history: int
) -> ResidualSpread:
    """Fit the spread for a short-history substation from the other substations' short replicas.

    Each entry must be a replica fit in which that other substation was the short one, with the
    same weeks of history. The held-out substation's own replica (and any entry that is a
    full-history arm) is refused, so the spread is as wide as a short history makes it and carries
    nothing of the substation it is for.

    Args:
        replicas: One entry per other substation.
        held_out: The substation whose forecast the spread is for.
        weeks_of_history: The weeks of history of the arm the spread is for.

    Returns:
        The centred spread.

    Raises:
        ValueError: If a substation appears twice, the entries were fitted at different cuts, an
            entry is the held-out substation's, has other weeks of history, trained
            on more rows than that many weeks hold (so it was not a short-history fit), or no
            entry remains.
    """
    sites = [replica.short_site for replica in replicas]
    if len(set(sites)) != len(sites):
        msg = "each substation may appear once among the short-history replicas"
        raise ValueError(msg)
    if len({replica.residuals.fit_cut for replica in replicas}) > 1:
        msg = "the short-history replicas were fitted at different cuts"
        raise ValueError(msg)
    for replica in replicas:
        if replica.short_site == held_out:
            msg = f"the replica of {held_out} cannot make {held_out}'s own spread"
            raise ValueError(msg)
        if replica.weeks_of_history != weeks_of_history:
            msg = (
                f"{replica.short_site}'s replica holds {replica.weeks_of_history} weeks of "
                f"history, not {weeks_of_history}"
            )
            raise ValueError(msg)
        if replica.n_training_rows > weeks_of_history * WEEK_ROWS:
            msg = (
                f"{replica.short_site}'s replica trained on {replica.n_training_rows} rows, more "
                f"than {weeks_of_history} weeks hold: it was not a short-history fit"
            )
            raise ValueError(msg)
    if not replicas:
        msg = f"no other substation has a short-history replica to make {held_out}'s spread"
        raise ValueError(msg)
    return ResidualSpread.fit(
        residuals=np.concatenate([r.residuals.residuals for r in replicas]),
        band=np.concatenate([r.residuals.band for r in replicas]),
        quarter=np.concatenate([r.residuals.quarter for r in replicas]),
        centre=True,
    )


EFFECTIVE_CAPACITY_QUANTILE: Final[float] = 0.99
"""A series' effective capacity is this quantile of its absolute power."""


def rebuilt_effective_capacity_mw(
    *, cache: FeatureCache, site: str, cut: datetime, window: TrainingWindow
) -> float:
    """Return the effective capacity rebuilt from the rows a short window trains on.

    The effective capacity of a series is the 99th percentile of its absolute power over its
    history. A new substation has only its window, so its scale is rebuilt from the same rows the
    fit sees, and never from rows it does not.

    Args:
        cache: The cache, in the whole-history effective capacity.
        site: The substation's label.
        cut: The cut.
        window: The substation's window.

    Returns:
        The rebuilt effective capacity in MW.

    Raises:
        ValueError: If the window holds no training row.
    """
    features = cache.sites[site]
    rows = training_rows(features=features, cache=cache, cut=cut, window=window)
    if len(rows) == 0:
        msg = f"{site} has no training row in its window before {cut}"
        raise ValueError(msg)
    power_mw = np.abs(features.y[rows]) * features.table.effective_capacity_mw / 100.0
    return float(np.quantile(power_mw, EFFECTIVE_CAPACITY_QUANTILE))


def with_effective_capacity(*, cache: FeatureCache, site: str, capacity_mw: float) -> FeatureCache:
    """Return the cache with one substation's points rebuilt on another effective capacity.

    The target and the table's capacity move together, so the power in MW is unchanged and a
    pinned capacity in MW converts to the new points.

    Args:
        cache: The cache.
        site: The substation's label.
        capacity_mw: The new effective capacity in MW.

    Returns:
        A cache whose `site` has `y` multiplied by the old capacity over `capacity_mw`, and
        `table.effective_capacity_mw` set to `capacity_mw`.
    """
    features = cache.sites[site]
    factor = features.table.effective_capacity_mw / capacity_mw
    changed = replace(
        features,
        table=replace(features.table, effective_capacity_mw=capacity_mw),
        y=features.y * factor,
    )
    return replace(cache, sites={**cache.sites, site: changed})


__all__ = [
    "FLEET_MEAN_RIDGE",
    "HEATING_FAMILY",
    "POOLED_FAMILIES",
    "PooledFamilyType",
    "PooledFit",
    "PooledReplica",
    "PooledSite",
    "PoolingStrength",
    "ShortHistoryReplica",
    "centring_matrix",
    "fit_pooled",
    "pooled_replica_residuals",
    "rebuilt_effective_capacity_mw",
    "short_history_replica",
    "short_history_spread",
    "with_effective_capacity",
]
