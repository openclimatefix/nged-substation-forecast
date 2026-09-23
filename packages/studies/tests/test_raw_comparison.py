import polars as pl
import pytest
from studies.raw_comparison import (
    mean_per_site_correlation,
    raw_column_comparison,
    raw_mad_difference,
)


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


def test_mean_per_site_correlation_averages_each_sites_own_correlation():
    # Site A is a perfect positive relationship, site B a perfect negative one. Pooling both
    # sites' rows before correlating would not average to 0 in general (it depends on each site's
    # scale and offset); correlating within each site first and then averaging always does, for
    # two sites whose own correlations are +1 and -1.
    frame = pl.DataFrame(
        {
            "site": ["A", "A", "A", "B", "B", "B"],
            "column": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "reference": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
        }
    )

    result = mean_per_site_correlation(frame=frame, column="column", reference="reference")

    assert result == pytest.approx(0.0)


def test_mean_per_site_correlation_of_one_site_is_that_sites_own_correlation():
    frame = pl.DataFrame(
        {"site": ["A", "A", "A"], "column": [1.0, 2.0, 3.0], "reference": [2.0, 4.0, 6.0]}
    )

    result = mean_per_site_correlation(frame=frame, column="column", reference="reference")

    assert result == pytest.approx(1.0)


def test_raw_mad_difference_is_treatment_mad_minus_reference_mad():
    # Treatment is always 4 away from baseline; reference is always 1 away. A bug that computed
    # reference minus treatment, or that dropped the absolute value before differencing, would not
    # give +3 here.
    frame = pl.DataFrame(
        {
            "treatment": [4.0, -4.0, 4.0, -4.0],
            "reference": [1.0, -1.0, 1.0, -1.0],
            "baseline": [0.0, 0.0, 0.0, 0.0],
            "month": ["2024-01", "2024-01", "2024-02", "2024-02"],
        }
    )

    result = raw_mad_difference(
        frame=frame, treatment="treatment", reference="reference", baseline="baseline"
    )

    assert result["difference"] == pytest.approx(3.0)
    assert result["n_months"] == 2
    assert result["n_rows"] == 4
