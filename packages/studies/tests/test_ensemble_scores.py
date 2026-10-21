import numpy as np
import pytest
from studies.ensemble_scores import fair_crps, rank_histogram, tail_weighted_crps, weighted_crps


def _brute_crps(*, values: np.ndarray, weights: np.ndarray, y: float) -> float:
    first = float((weights * np.abs(values - y)).sum())
    second = 0.5 * float(
        sum(
            weights[i] * weights[j] * abs(values[i] - values[j])
            for i in range(len(values))
            for j in range(len(values))
        )
    )
    return first - second


def test_weighted_crps_matches_brute_force_with_uneven_weights() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=(5, 7))
    weights = rng.random(7)
    weights /= weights.sum()
    actual = rng.normal(size=5)
    got = weighted_crps(values=values, weights=weights, actual=actual)
    want = [_brute_crps(values=values[i], weights=weights, y=actual[i]) for i in range(5)]
    np.testing.assert_allclose(got, want)


def test_weighted_crps_of_a_point_mass_is_the_absolute_error() -> None:
    got = weighted_crps(values=np.array([[2.0]]), weights=np.array([1.0]), actual=np.array([5.0]))
    assert got[0] == pytest.approx(3.0)


def test_weighted_crps_rejects_weights_not_summing_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        weighted_crps(values=np.zeros((1, 2)), weights=np.array([0.5, 0.6]), actual=np.zeros(1))


def test_fair_crps_two_members_known_value() -> None:
    # members 0 and 2, y = 1: mean|x-y| = 1; fair spread term = |0-2| * 2 / (2*2*1) = 1.
    got = fair_crps(members=np.array([[0.0, 2.0]]), actual=np.array([1.0]))
    assert got[0] == pytest.approx(0.0)


def test_fair_crps_is_unbiased_in_member_count() -> None:
    rng = np.random.default_rng(1)
    scores = {}
    for m in (2, 10):
        members = rng.normal(size=(40000, m))
        actual = rng.normal(size=40000)
        scores[m] = fair_crps(members=members, actual=actual).mean()
    assert scores[2] == pytest.approx(scores[10], abs=0.02)


def test_fair_crps_needs_two_members() -> None:
    with pytest.raises(ValueError, match="at least two"):
        fair_crps(members=np.zeros((1, 1)), actual=np.zeros(1))


def test_tail_weighted_crps_ignores_values_on_the_other_side() -> None:
    values = np.array([[-10.0, -5.0, 3.0, 4.0]])
    weights = np.full(4, 0.25)
    kwargs = {"weights": weights, "actual": np.array([-7.0]), "threshold": 0.0, "tail": "upper"}
    base = tail_weighted_crps(values=values, **kwargs)
    moved = tail_weighted_crps(values=np.array([[-99.0, -1.0, 3.0, 4.0]]), **kwargs)
    assert base[0] == pytest.approx(moved[0])


def test_tail_weighted_crps_lower_mirrors_upper() -> None:
    rng = np.random.default_rng(2)
    values = rng.normal(size=(6, 5))
    actual = rng.normal(size=6)
    weights = np.full(5, 0.2)
    upper = tail_weighted_crps(
        values=values, weights=weights, actual=actual, threshold=0.3, tail="upper"
    )
    lower = tail_weighted_crps(
        values=-values, weights=weights, actual=-actual, threshold=-0.3, tail="lower"
    )
    np.testing.assert_allclose(upper, lower)


def test_tail_weighted_crps_accepts_one_threshold_per_case() -> None:
    values = np.array([[0.0, 1.0], [0.0, 1.0]])
    got = tail_weighted_crps(
        values=values,
        weights=np.array([0.5, 0.5]),
        actual=np.array([2.0, 2.0]),
        threshold=np.array([0.0, 10.0]),
        tail="upper",
    )
    assert got[0] > 0.0
    assert got[1] == pytest.approx(0.0)


def test_rank_histogram_is_flat_for_exchangeable_members() -> None:
    rng = np.random.default_rng(3)
    draws = rng.normal(size=(30000, 6))
    counts = rank_histogram(members=draws[:, :5], actual=draws[:, 5], rng=rng)
    assert counts.sum() == 30000
    assert len(counts) == 6
    assert counts.min() > 0.9 * 5000
    assert counts.max() < 1.1 * 5000


def test_rank_histogram_is_u_shaped_when_members_are_too_narrow() -> None:
    rng = np.random.default_rng(4)
    counts = rank_histogram(
        members=0.3 * rng.normal(size=(20000, 5)), actual=rng.normal(size=20000), rng=rng
    )
    assert counts[0] > counts[2]
    assert counts[-1] > counts[2]


def test_rank_histogram_splits_ties_across_ranks() -> None:
    rng = np.random.default_rng(5)
    counts = rank_histogram(members=np.zeros((20000, 3)), actual=np.zeros(20000), rng=rng)
    assert counts.min() > 0.9 * 5000
