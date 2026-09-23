import polars as pl
import pytest
from studies.raw_comparison import raw_column_comparison


def test_bias_is_treatment_minus_reference():
    # Treatment is always 3 above reference, so a bias of -3 would mean the subtraction is
    # backwards.
    frame = pl.DataFrame({"treatment": [10.0, 20.0, 30.0], "reference": [7.0, 17.0, 27.0]})

    result = raw_column_comparison(frame=frame, treatment="treatment", reference="reference")

    assert result["bias"] == pytest.approx(3.0)


def test_mad_is_the_mean_of_the_absolute_differences_not_the_absolute_mean():
    # The signed differences are +10 and -10, which average to 0. A bug that took the absolute
    # value of the mean, rather than the mean of the absolute values, would report 0 here too.
    frame = pl.DataFrame({"treatment": [10.0, 0.0], "reference": [0.0, 10.0]})

    result = raw_column_comparison(frame=frame, treatment="treatment", reference="reference")

    assert result["bias"] == pytest.approx(0.0)
    assert result["mad"] == pytest.approx(10.0)


def test_correlation_of_a_perfect_linear_relationship_is_one():
    frame = pl.DataFrame({"treatment": [1.0, 2.0, 3.0, 4.0], "reference": [2.0, 4.0, 6.0, 8.0]})

    result = raw_column_comparison(frame=frame, treatment="treatment", reference="reference")

    assert result["correlation"] == pytest.approx(1.0)


def test_correlation_of_unrelated_columns_is_near_zero():
    frame = pl.DataFrame({"treatment": [1.0, 2.0, 3.0, 4.0], "reference": [3.0, 1.0, 4.0, 2.0]})

    result = raw_column_comparison(frame=frame, treatment="treatment", reference="reference")

    assert abs(result["correlation"]) < 0.5
