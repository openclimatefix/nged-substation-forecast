import numpy as np
import pytest
from studies.bootstrap import BOOTSTRAP_SEED, N_BOOTSTRAP_RESAMPLES
from studies.correlation import pooled_correlation_interval

MONTHS = np.repeat(np.array([f"2025-{month:02d}" for month in range(1, 13)]), 20)
"""Twelve months of twenty rows each."""


def _noisy_pair(*, seeds: int = 3) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(7)
    actual = generator.normal(size=MONTHS.shape[0])
    prediction = actual[None, :] + generator.normal(scale=0.7, size=(seeds, MONTHS.shape[0]))
    return actual, prediction


def test_the_point_estimate_is_the_mean_over_seeds_of_each_seeds_correlation():
    actual, prediction = _noisy_pair()

    result = pooled_correlation_interval(actual=actual, prediction=prediction, months=MONTHS)

    expected = np.mean(
        [np.corrcoef(actual, seed_prediction)[0, 1] for seed_prediction in prediction]
    )
    assert result["correlation"] == pytest.approx(expected)


def test_a_perfect_prediction_has_correlation_one_and_a_degenerate_interval():
    actual, _ = _noisy_pair()

    result = pooled_correlation_interval(
        actual=actual, prediction=np.stack([2.0 * actual + 1.0] * 2), months=MONTHS
    )

    assert result["correlation"] == pytest.approx(1.0)
    assert result["lower"] == pytest.approx(1.0)
    assert result["upper"] == pytest.approx(1.0)


def test_the_interval_brackets_the_point_estimate_and_counts_rows_and_months():
    actual, prediction = _noisy_pair()

    result = pooled_correlation_interval(actual=actual, prediction=prediction, months=MONTHS)

    assert result["lower"] < result["correlation"] < result["upper"]
    assert result["n_rows"] == MONTHS.shape[0]
    assert result["n_months"] == 12


def test_whole_months_are_resampled_so_a_month_that_disagrees_widens_the_interval():
    actual, prediction = _noisy_pair(seeds=1)
    first_month = MONTHS == "2025-01"
    anti = prediction.copy()
    anti[:, first_month] = -actual[first_month]

    plain = pooled_correlation_interval(actual=actual, prediction=prediction, months=MONTHS)
    with_anti_month = pooled_correlation_interval(actual=actual, prediction=anti, months=MONTHS)

    width_plain = plain["upper"] - plain["lower"]
    width_anti = with_anti_month["upper"] - with_anti_month["lower"]
    assert width_anti > width_plain


def test_a_higher_level_gives_a_wider_interval():
    actual, prediction = _noisy_pair()

    narrow = pooled_correlation_interval(
        actual=actual, prediction=prediction, months=MONTHS, level=80.0
    )
    wide = pooled_correlation_interval(
        actual=actual, prediction=prediction, months=MONTHS, level=99.0
    )

    assert wide["upper"] - wide["lower"] > narrow["upper"] - narrow["lower"]


def test_a_fraction_is_rejected_as_a_level():
    actual, prediction = _noisy_pair()

    with pytest.raises(ValueError, match="percent"):
        pooled_correlation_interval(actual=actual, prediction=prediction, months=MONTHS, level=0.95)


def test_mismatched_shapes_are_rejected():
    actual, prediction = _noisy_pair()

    with pytest.raises(ValueError, match="n_seeds, n_rows"):
        pooled_correlation_interval(actual=actual, prediction=prediction[:, :-1], months=MONTHS)


def test_the_same_inputs_give_the_same_interval():
    actual, prediction = _noisy_pair()

    first = pooled_correlation_interval(actual=actual, prediction=prediction, months=MONTHS)
    second = pooled_correlation_interval(actual=actual, prediction=prediction, months=MONTHS)

    assert first == second


def test_each_resample_draws_a_seed_so_a_noisy_seed_widens_the_interval():
    actual, _ = _noisy_pair()
    perfect = 2.0 * actual + 1.0
    noisy = actual + np.random.default_rng(3).normal(scale=2.0, size=actual.shape[0])

    result = pooled_correlation_interval(
        actual=actual, prediction=np.stack([perfect, noisy]), months=MONTHS
    )

    assert result["upper"] == pytest.approx(1.0)
    assert result["lower"] < 0.9


def test_the_interval_matches_a_direct_resample_of_a_seed_then_whole_months():
    actual, prediction = _noisy_pair()

    result = pooled_correlation_interval(
        actual=actual, prediction=prediction, months=MONTHS, level=90.0
    )

    unique_months, month_index = np.unique(MONTHS, return_inverse=True)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    resampled = []
    for _ in range(N_BOOTSTRAP_RESAMPLES):
        seed = generator.integers(0, prediction.shape[0])
        drawn = generator.integers(0, len(unique_months), size=len(unique_months))
        rows = np.concatenate([np.flatnonzero(month_index == month) for month in drawn])
        resampled.append(np.corrcoef(actual[rows], prediction[seed, rows])[0, 1])
    assert result["lower"] == pytest.approx(np.percentile(resampled, 5.0))
    assert result["upper"] == pytest.approx(np.percentile(resampled, 95.0))


def test_every_seed_is_drawn_in_the_resamples():
    actual, _ = _noisy_pair()
    prediction = np.stack([actual, -actual])

    result = pooled_correlation_interval(actual=actual, prediction=prediction, months=MONTHS)

    assert result["lower"] == pytest.approx(-1.0)
    assert result["upper"] == pytest.approx(1.0)
