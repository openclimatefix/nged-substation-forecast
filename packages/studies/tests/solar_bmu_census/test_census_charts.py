"""Tests for the pure helpers of `studies/solar_bmu_census/census_charts.py`.

Each test fails on the bug it exists for. Among those bugs are line labels that overprint, and an
axis that hides the zero line of a storage BMU that charges.
"""

from itertools import pairwise

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


def test_the_six_figures_of_cleve_hill_solar_1_get_labels_that_never_overlap() -> None:
    values = [112.0, 112.0, 350.0, 112.0, 373.0, 88.1]
    gap = 373.0 * 1.1 * census_charts.LABEL_GAP_SHARE
    heights = sorted(census_charts.label_positions(values=values, gap=gap))
    assert all(upper - lower >= gap - 1e-9 for lower, upper in pairwise(heights))
    assert heights[0] == pytest.approx(88.1)
    assert all(height >= value for height, value in zip(heights, sorted(values), strict=True))


def _census_row(
    bmu_id: str,
    *,
    technology: str,
    correlation: float,
    storage: str = "",
    scope: str = "single-site",
) -> dict[str, object]:
    return {
        "elexon_bmu_id": bmu_id,
        "scope": scope,
        "basis": "behaviour",
        "technology": technology,
        "correlation": correlation,
        "storage_bmu_ids": storage,
    }


def test_the_examples_are_the_pure_pv_bmu_and_the_highest_median_and_lowest_hybrids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _census_row("PURE", technology="pure PV", correlation=0.85),
        _census_row("H1", technology="hybrid", correlation=0.90, storage="S1"),
        _census_row("H2", technology="hybrid", correlation=0.88),
        _census_row("H3", technology="hybrid", correlation=0.86, storage="S3"),
        _census_row("H4", technology="hybrid", correlation=0.84),
        _census_row("H5", technology="hybrid", correlation=0.82, storage="S5"),
        _census_row("AGG", technology="hybrid", correlation=0.99, scope="aggregate"),
    ]
    census = pl.DataFrame(rows)
    monkeypatch.setattr(
        census_charts,
        "storage_bmu_at",
        lambda *, site_bmu_id, census: {"H1": "S1", "H3": None, "H5": "S5"}.get(site_bmu_id),
    )
    examples = census_charts.choose_examples(census=census)
    assert [(e.solar_bmu_id, e.kind, e.storage_bmu_id) for e in examples] == [
        ("PURE", "pure PV", None),
        ("H1", "hybrid", "S1"),
        ("H3", "hybrid", None),
        ("H5", "hybrid", "S5"),
    ]
