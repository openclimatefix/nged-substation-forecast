"""The mixture-CRPS objective of experiment 99 and the constant-Hessian pinball it is compared with.

A training group is one (series, run, valid time): `K` rows, one per drawn ensemble member, all with
the same observed value. Each row is predicted as 11 atoms, one per quantile level. The pooled
predictive distribution of the group is the mixture of the `K` rows' atoms. The atom of level `j`
carries probability mass `CRPS_MASSES[j] / K`.

The masses are the ones that make the continuous ranked probability score (CRPS) of a single row
smallest when atom `j` sits at the true quantile of level `j`. The CRPS of a distribution made of
atoms `a_i` with probabilities `p_i` is `sum_i p_i |a_i - y| - 1/2 sum_i sum_k p_i p_k |a_i - a_k|`,
and the derivative with respect to `a_i` is `p_i (sgn(a_i - y) - sum_k p_k sgn(a_i - a_k))`. The
second term is the probability below the atom minus the probability above it, which a sort and a
cumulative sum give for every atom of a group at once.

The sample of `K` members under-represents the spread between the `N_MEMBERS` members that are
pooled at prediction time. `fair` replaces the plug-in mixture by the population one: a pair of
atoms from two different members gets `(N-1)/N * K/(K-1)` times its plug-in weight, and a pair from
the same member gets `K/N` times it.
"""

from typing import Final

import numpy as np

LEVELS: Final[np.ndarray] = np.array(
    [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
)
N_LEVELS: Final[int] = len(LEVELS)
N_MEMBERS: Final[int] = 51


def crps_masses(*, levels: np.ndarray = LEVELS) -> np.ndarray:
    """Return the atom masses for which the CRPS-optimal atom of each level is its quantile.

    The cumulative mass reaches the level at the middle of the atom's mass, so atom `j` has mass
    `2 * (level_j - mass_below_j)`.

    Args:
        levels: Ascending quantile levels, symmetric about 0.5 so the masses add to 1.

    Returns:
        The masses, summing to 1.

    Raises:
        ValueError: If a mass is not positive or the masses do not add to 1.
    """
    masses = np.empty(len(levels))
    below = 0.0
    for j, level in enumerate(levels):
        masses[j] = 2.0 * (level - below)
        below += masses[j]
    if (masses <= 0).any() or abs(masses.sum() - 1.0) > 1e-9:
        msg = f"levels {levels} give no valid CRPS masses: {masses}"
        raise ValueError(msg)
    return masses


CRPS_MASSES: Final[np.ndarray] = crps_masses()
"""0.02, 0.06, 0.04, 0.26, 0.04, 0.16, 0.04, 0.26, 0.04, 0.06, 0.02 for the 11 levels."""


def _below_minus_above(*, atoms: np.ndarray, probability: np.ndarray) -> np.ndarray:
    """Return, for every atom, the probability of strictly smaller atoms minus strictly larger.

    Atoms with equal values count for neither side. This matters because rows that fall in the same
    leaves of every tree have identical atoms. The arrays are float32 and int32 because the sort
    and the cumulative sums over every group dominate the cost of a boosting round.

    Args:
        atoms: Shape (groups, atoms per group), float32.
        probability: Shape (atoms per group,), each atom's probability, float32.

    Returns:
        Shape (groups, atoms per group), float32.
    """
    groups, width = atoms.shape
    order = np.argsort(atoms, axis=1)
    sorted_atoms = np.take_along_axis(atoms, order, axis=1)
    mass = probability[order]
    inclusive = np.cumsum(mass, axis=1, dtype=np.float32)
    exclusive = inclusive - mass
    position = np.arange(width, dtype=np.int32)
    first_of_run = np.concatenate(
        [np.ones((groups, 1), dtype=bool), sorted_atoms[:, 1:] != sorted_atoms[:, :-1]], axis=1
    )
    last_of_run = np.concatenate([first_of_run[:, 1:], np.ones((groups, 1), dtype=bool)], axis=1)
    run_start = np.maximum.accumulate(np.where(first_of_run, position, np.int32(0)), axis=1)
    run_end = np.minimum.accumulate(
        np.where(last_of_run, position, np.int32(width))[:, ::-1], axis=1
    )[:, ::-1]
    below = np.take_along_axis(exclusive, run_start, axis=1)
    above = np.float32(1.0) - np.take_along_axis(inclusive, run_end, axis=1)
    difference = np.empty_like(below)
    np.put_along_axis(difference, order, below - above, axis=1)
    return difference


def crps_gradient(*, atoms: np.ndarray, actual: np.ndarray, members: int, fair: bool) -> np.ndarray:
    """Return the gradient of the mixture CRPS with respect to every row's 11 atoms.

    Each group's gradient is multiplied by `members`, so an atom's gradient is `CRPS_MASSES[j]`
    times a number between -2 and 2 whatever `members` is.

    Args:
        atoms: Shape (groups, members, 11): the predicted atoms of each member's row.
        actual: Shape (groups,), the observed value of each group.
        members: `K`, the number of rows per group.
        fair: Use the population-matched pair weights instead of the plug-in mixture.

    Returns:
        Shape (groups, members, 11), float32.
    """
    groups = atoms.shape[0]
    flat = atoms.reshape(groups, members * N_LEVELS)
    probability = (np.tile(CRPS_MASSES, members) / members).astype(np.float32)
    difference = _below_minus_above(atoms=flat, probability=probability).reshape(atoms.shape)
    if fair:
        distinct = (N_MEMBERS - 1) / N_MEMBERS * members / (members - 1)
        own = members / N_MEMBERS
        own_difference = _below_minus_above(
            atoms=atoms.reshape(groups * members, N_LEVELS),
            probability=CRPS_MASSES.astype(np.float32),
        ).reshape(atoms.shape) / np.float32(members)
        difference = distinct * difference + (own - distinct) * own_difference
    sign = np.sign(atoms - actual[:, None, None])
    return (CRPS_MASSES * (sign - difference)).astype(np.float32)


def crps_loss(*, atoms: np.ndarray, actual: np.ndarray, members: int) -> np.ndarray:
    """Return the plug-in mixture CRPS of every group, for monitoring and the self-test.

    Args:
        atoms: Shape (groups, members, 11).
        actual: Shape (groups,).
        members: `K`.

    Returns:
        Shape (groups,), float64.
    """
    groups = atoms.shape[0]
    flat = atoms.reshape(groups, members * N_LEVELS).astype(np.float64)
    order = np.argsort(flat, axis=1)
    values = np.take_along_axis(flat, order, axis=1)
    probability = (np.tile(CRPS_MASSES, members) / members)[order]
    absolute_error = (probability * np.abs(values - actual[:, None])).sum(axis=1)
    below = np.cumsum(probability, axis=1) - probability
    # For sorted atoms, sum_i sum_k p_i p_k |a_i - a_k| = 2 * sum_i p_i a_i (P_below_i - P_above_i),
    # and P_below_i - P_above_i = 2 * P_below_i + p_i - 1 when no two atoms are equal.
    spread = 2.0 * (probability * values * (2.0 * below + probability - 1.0)).sum(axis=1)
    return absolute_error - 0.5 * spread


def pinball_gradient(*, atoms: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return the gradient of the summed pinball loss of the 11 levels, row by row.

    Args:
        atoms: Shape (groups, members, 11).
        actual: Shape (groups,).

    Returns:
        Shape (groups, members, 11), float32: 1 - level where the atom is above the observed value,
        and -level where it is not.
    """
    above = atoms > actual[:, None, None]
    return (above - LEVELS).astype(np.float32)


def fair_crps_loss(*, atoms: np.ndarray, actual: np.ndarray, members: int) -> np.ndarray:
    """Return the population-matched mixture CRPS of every group by brute force over atom pairs.

    Args:
        atoms: Shape (groups, members, 11), float64.
        actual: Shape (groups,).
        members: `K`.

    Returns:
        Shape (groups,).
    """
    groups = atoms.shape[0]
    distinct = (N_MEMBERS - 1) / N_MEMBERS * members / (members - 1)
    own = members / N_MEMBERS
    flat = atoms.reshape(groups, members * N_LEVELS)
    probability = np.tile(CRPS_MASSES, members) / members
    owner = np.repeat(np.arange(members), N_LEVELS)
    factor = np.where(owner[:, None] == owner[None, :], own, distinct)
    first = (probability * np.abs(flat - actual[:, None])).sum(axis=1)
    pairs = np.abs(flat[:, :, None] - flat[:, None, :])
    second = (probability[:, None] * probability[None, :] * factor * pairs).sum(axis=(1, 2))
    return first - 0.5 * second


def self_test() -> None:
    """Check `crps_gradient` against a finite difference of the brute-force loss.

    The atoms of each row are deliberately not in ascending order, as the 11 outputs of a row come
    from separate trees.

    Raises:
        ValueError: If the largest difference exceeds 1e-4.
    """
    rng = np.random.default_rng(0)
    members = 6
    atoms = rng.normal(size=(7, members, N_LEVELS)) * 0.3 + rng.normal(size=(7, members, 1)) * 0.2
    actual = rng.normal(size=7)
    step = 1e-6
    for fair, loss in (
        (False, lambda a: crps_loss(atoms=a, actual=actual, members=members)),
        (True, lambda a: fair_crps_loss(atoms=a, actual=actual, members=members)),
    ):
        analytic = crps_gradient(
            atoms=atoms.astype(np.float32),
            actual=actual.astype(np.float32),
            members=members,
            fair=fair,
        )
        numeric = np.zeros_like(atoms)
        for m in range(members):
            for j in range(N_LEVELS):
                up, down = atoms.copy(), atoms.copy()
                up[:, m, j] += step
                down[:, m, j] -= step
                numeric[:, m, j] = (loss(up) - loss(down)) / (2 * step) * members
        error = float(np.abs(analytic - numeric).max())
        if error > 1e-4:
            msg = f"crps_gradient (fair={fair}) differs from a finite difference by {error:.2e}"
            raise ValueError(msg)
        print(f"gradient check fair={fair}: max difference {error:.1e}")


if __name__ == "__main__":
    self_test()
