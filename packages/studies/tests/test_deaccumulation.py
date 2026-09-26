import numpy as np
import pytest
from studies.deaccumulation import deaccumulate_to_rates

# Steps at 0, 3, 6 and 12 hours: the last interval is 6 hours, as after 144 hours in ENS.
SECONDS = np.array([0, 10_800, 21_600, 43_200])


def test_a_three_to_six_hour_change_divides_by_the_actual_elapsed_seconds():
    # Rates of 2, 1 and 3 per second over the three steps.
    accumulations = np.array([0.0, 21_600.0, 32_400.0, 32_400.0 + 21_600 * 3])

    result = deaccumulate_to_rates(
        accumulations=accumulations, elapsed_seconds=SECONDS, invalid_below_rate=-1.0
    )

    assert np.isnan(result.rates[0])
    assert result.rates[1:].tolist() == [2.0, 1.0, 3.0]
    assert not result.clamped.any()
    assert not result.invalid.any()


def test_the_first_rate_takes_the_accumulation_before_step_zero_as_zero():
    accumulations = np.array([5.0, 10_805.0, 10_805.0, 10_805.0])

    result = deaccumulate_to_rates(
        accumulations=accumulations, elapsed_seconds=SECONDS, invalid_below_rate=-1.0
    )

    assert result.rates[1] == (10_805.0 / 10_800)


def test_a_small_negative_rate_from_quantisation_is_set_to_zero():
    accumulations = np.array([0.0, 100.0, 99.0, 99.0])

    result = deaccumulate_to_rates(
        accumulations=accumulations, elapsed_seconds=SECONDS, invalid_below_rate=-1e-3
    )

    assert result.rates[2] == 0.0
    assert result.clamped.tolist() == [False, False, True, False]
    assert not result.invalid.any()


def test_a_large_negative_rate_is_set_to_nan():
    accumulations = np.array([0.0, 100.0, 40_000.0, 0.0])

    result = deaccumulate_to_rates(
        accumulations=accumulations, elapsed_seconds=SECONDS, invalid_below_rate=-1e-3
    )

    assert np.isnan(result.rates[3])
    assert result.invalid.tolist() == [False, False, False, True]
    assert not result.clamped.any()


def test_a_nan_accumulation_gives_nan_rates_at_and_after_it_without_counting_as_invalid():
    accumulations = np.array([0.0, np.nan, 10.0, 20.0])

    result = deaccumulate_to_rates(
        accumulations=accumulations, elapsed_seconds=SECONDS, invalid_below_rate=-1.0
    )

    assert np.isnan(result.rates[1:3]).all()
    assert result.rates[3] == pytest.approx(10 / 21_600)
    assert not result.invalid.any()


def test_the_step_axis_is_the_first_axis_of_a_larger_array():
    accumulations = np.stack(
        [np.array([0.0, 10_800.0, 21_600.0, 43_200.0]), np.array([0.0, 0.0, 10_800.0, 10_800.0])],
        axis=1,
    )

    result = deaccumulate_to_rates(
        accumulations=accumulations, elapsed_seconds=SECONDS, invalid_below_rate=-1.0
    )

    assert result.rates[1:].tolist() == [[1.0, 0.0], [1.0, 1.0], [1.0, 0.0]]


def test_elapsed_seconds_that_do_not_increase_are_refused():
    with pytest.raises(ValueError, match="increase"):
        deaccumulate_to_rates(
            accumulations=np.zeros(4),
            elapsed_seconds=np.array([0, 10, 10, 20]),
            invalid_below_rate=-1.0,
        )


def test_elapsed_seconds_of_the_wrong_length_are_refused():
    with pytest.raises(ValueError, match="one value per step"):
        deaccumulate_to_rates(
            accumulations=np.zeros(4),
            elapsed_seconds=np.array([0, 10, 20]),
            invalid_below_rate=-1.0,
        )
