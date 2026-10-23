import numpy as np
import pytest
from scipy.stats import norm
from studies.quantile_sums import (
    copula_sum_quantiles,
    independent_sum_quantiles,
    inverse_cdf,
    member_rank_levels,
    member_wise_sum_quantiles,
    pit_normal_scores,
    sum_of_quantiles,
)

LEVELS = (0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99)
OUTPUT_LEVELS = (0.05, 0.50, 0.95)
N_ROWS = 3


def _normal_parts(*, n_parts: int) -> list[np.ndarray]:
    """Quantiles of a standard normal forecast centred on 100, for each part and row."""
    row = 100.0 + norm.ppf(LEVELS)
    return [np.tile(row, (N_ROWS, 1)) for _ in range(n_parts)]


def test_sum_of_quantiles_adds_each_level() -> None:
    parts = _normal_parts(n_parts=3)

    total = sum_of_quantiles(parts=parts, part_levels=LEVELS, levels=OUTPUT_LEVELS)

    assert total[0, 1] == pytest.approx(300.0)
    assert total[0, 2] == pytest.approx(300.0 + 3 * norm.ppf(0.95))


def test_independent_sum_matches_the_normal_sum() -> None:
    parts = _normal_parts(n_parts=3)

    total = independent_sum_quantiles(
        parts=parts, part_levels=LEVELS, levels=OUTPUT_LEVELS, n_draws=20_000, seed=0
    )

    assert total[0, 2] == pytest.approx(300.0 + np.sqrt(3) * norm.ppf(0.95), abs=0.1)


def test_independent_sum_is_narrower_than_the_sum_of_quantiles() -> None:
    parts = _normal_parts(n_parts=3)

    independent = independent_sum_quantiles(
        parts=parts, part_levels=LEVELS, levels=OUTPUT_LEVELS, n_draws=5_000, seed=0
    )
    perfect = sum_of_quantiles(parts=parts, part_levels=LEVELS, levels=OUTPUT_LEVELS)

    assert np.all(independent[:, 2] < perfect[:, 2])


def test_member_wise_sum_with_identical_members_is_close_to_the_sum_of_quantiles() -> None:
    parts = _normal_parts(n_parts=3)
    shared = np.tile(np.arange(51.0), (N_ROWS, 1))

    member_wise = member_wise_sum_quantiles(
        parts=parts,
        part_levels=LEVELS,
        members=[shared, shared, shared],
        levels=OUTPUT_LEVELS,
        seed=0,
        n_draws_per_member=1,
    )
    perfect = sum_of_quantiles(parts=parts, part_levels=LEVELS, levels=OUTPUT_LEVELS)

    assert member_wise[0, 1] == pytest.approx(perfect[0, 1], abs=0.2)
    assert member_wise[0, 2] == pytest.approx(perfect[0, 2], abs=0.3)


def test_member_wise_sum_with_unrelated_members_is_close_to_the_independent_sum() -> None:
    rng = np.random.default_rng(1)
    parts = _normal_parts(n_parts=3)
    members = [rng.normal(size=(N_ROWS, 51)) for _ in range(3)]

    member_wise = member_wise_sum_quantiles(
        parts=parts, part_levels=LEVELS, members=members, levels=OUTPUT_LEVELS, seed=0
    )

    assert member_wise[0, 2] < 300.0 + 3 * norm.ppf(0.95) - 1.0


def test_copula_sum_widens_with_correlation() -> None:
    parts = _normal_parts(n_parts=2)
    low = copula_sum_quantiles(
        parts=parts,
        part_levels=LEVELS,
        correlation=np.array([[1.0, 0.0], [0.0, 1.0]]),
        levels=OUTPUT_LEVELS,
        n_draws=5_000,
        seed=0,
    )
    high = copula_sum_quantiles(
        parts=parts,
        part_levels=LEVELS,
        correlation=np.array([[1.0, 0.9], [0.9, 1.0]]),
        levels=OUTPUT_LEVELS,
        n_draws=5_000,
        seed=0,
    )

    assert high[0, 2] > low[0, 2] + 0.3


def test_inverse_cdf_clips_to_the_lower_bound() -> None:
    quantiles = np.array([[0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]])

    values = inverse_cdf(quantiles=quantiles, levels=LEVELS, uniforms=np.array([[0.001]]))

    assert values[0, 0] == 0.0


def test_pit_normal_scores_are_standard_normal_for_a_calibrated_forecast() -> None:
    rng = np.random.default_rng(0)
    quantiles = np.tile(norm.ppf(LEVELS), (20_000, 1))
    actual = rng.normal(size=20_000)

    scores = pit_normal_scores(quantiles=quantiles, levels=LEVELS, actual=actual, seed=0)

    assert scores.mean() == pytest.approx(0.0, abs=0.05)
    assert scores.std() == pytest.approx(1.0, abs=0.15)


def test_pit_normal_scores_spread_over_a_plateau() -> None:
    quantiles = np.tile(np.zeros(len(LEVELS)), (1_000, 1))

    scores = pit_normal_scores(quantiles=quantiles, levels=LEVELS, actual=np.zeros(1_000), seed=0)

    assert scores.min() < scores.max()


def test_parts_with_different_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="shape"):
        sum_of_quantiles(
            parts=[np.zeros((2, 11)), np.zeros((3, 11))], part_levels=LEVELS, levels=OUTPUT_LEVELS
        )


def test_member_rank_levels_cover_the_unit_interval_evenly() -> None:
    rng = np.random.default_rng(0)
    members = rng.normal(size=(200, 51))

    levels = member_rank_levels(member_values=members, rng=rng, n_draws=40)

    assert levels.mean() == pytest.approx(0.5, abs=0.002)
    assert levels.min() < 0.005
    assert levels.max() > 0.995


def test_member_rank_levels_of_tied_members_still_cover_the_unit_interval() -> None:
    rng = np.random.default_rng(0)

    levels = member_rank_levels(member_values=np.zeros((10, 51)), rng=rng, n_draws=40)

    assert levels.mean() == pytest.approx(0.5, abs=0.01)
    assert levels.std() == pytest.approx(np.sqrt(1 / 12), abs=0.01)


def test_pit_normal_scores_of_a_top_plateau_reach_the_upper_tail() -> None:
    quantiles = np.tile(np.concatenate([np.arange(5.0), np.full(6, 10.0)]), (2_000, 1))

    scores = pit_normal_scores(
        quantiles=quantiles, levels=LEVELS, actual=np.full(2_000, 10.0), seed=0
    )

    assert scores.max() > norm.ppf(0.995)
