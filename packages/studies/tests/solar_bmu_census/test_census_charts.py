"""Tests for the pure helpers of `studies/solar_bmu_census/census_charts.py`.

Each test fails on the bug it exists for. Among those bugs are line labels that overprint, and an
axis that hides the zero line of a storage BMU that charges.
"""

import census_charts
import polars as pl
import pytest


def test_labels_closer_than_the_gap_are_pushed_up_in_value_order() -> None:
    heights = census_charts.label_positions(values=[50.0, 10.0, 12.0, 100.0], gap=5.0)
    assert heights == [50.0, 10.0, 15.0, 100.0]


def test_a_cluster_of_equal_values_is_spread_by_the_gap() -> None:
    heights = census_charts.label_positions(values=[20.0, 20.0, 20.0], gap=3.0)
    assert sorted(heights) == [20.0, 23.0, 26.0]


def test_the_week_axis_reaches_below_zero_for_a_charging_battery() -> None:
    low, high = census_charts.week_domain(capacity_mw=50.0, megawatts=pl.Series([-40.0, 0.0, 30.0]))
    assert low == pytest.approx(-44.0)
    assert high == pytest.approx(55.0)


def test_the_week_axis_leaves_room_below_zero_and_above_a_large_output() -> None:
    low, high = census_charts.week_domain(capacity_mw=50.0, megawatts=pl.Series([0.0, 80.0]))
    assert high == pytest.approx(88.0)
    assert low == pytest.approx(-4.4)
