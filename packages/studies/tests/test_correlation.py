import numpy as np
import pytest
from studies.bootstrap import N_BOOTSTRAP_RESAMPLES
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
    assert N_BOOTSTRAP_RESAMPLES == 2000
