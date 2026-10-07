"""Tests for the pure helpers of `studies/solar_bmu_census/census_charts.py`.

Each test fails on the bug it exists for. Among those bugs are line labels that overprint, and an
axis that hides the zero line of a storage BMU that charges.
"""

from datetime import UTC, datetime, timedelta
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


def test_an_arrow_ends_at_the_pixel_height_of_its_value() -> None:
    geometry = census_charts.label_geometry(values=[25.0, 75.0], domain_high=100.0, height_px=200.0)
    assert geometry.domain_high == pytest.approx(100.0)
    assert geometry.line_y_px == pytest.approx([150.0, 50.0])
    assert geometry.label_y_px == pytest.approx([150.0, 50.0])


def test_labels_in_the_same_order_as_values_keep_arrows_from_crossing() -> None:
    values = [112.0, 112.0, 350.0, 112.0, 373.0, 88.1, 90.0]
    geometry = census_charts.label_geometry(values=values, domain_high=410.0, height_px=230.0)
    order = sorted(range(len(values)), key=lambda i: values[i])
    labels = [geometry.label_y_px[i] for i in order]
    lines = [geometry.line_y_px[i] for i in order]
    assert labels == sorted(labels, reverse=True)
    assert lines == sorted(lines, reverse=True)
    gap_px = 230.0 * census_charts.LABEL_GAP_SHARE * 410.0 / geometry.domain_high
    assert all(a - b >= gap_px - 1e-9 for a, b in pairwise(labels))


def test_stacked_labels_raise_the_axis_so_the_top_label_stays_inside_the_plot() -> None:
    geometry = census_charts.label_geometry(
        values=[90.0, 91.0, 92.0], domain_high=100.0, height_px=100.0
    )
    assert geometry.domain_high > 100.0
    assert min(geometry.label_y_px) > 0
    assert geometry.label_y_px[0] > geometry.label_y_px[2]


def test_cleve_hill_s_three_equal_values_form_one_group_and_tec_never_joins() -> None:
    values = {
        "Generation Capacity": 112.0,
        "IGCPU installed": 112.0,
        "TEC": 112.0,
        "Largest MEL": 112.0,
        "REPD installed": 373.0,
        "P99 of output": 112.0,
    }
    assert census_charts.coinciding_groups(values=values) == [
        ["Largest MEL", "IGCPU installed", "Generation Capacity"]
    ]


def test_larks_green_groups_two_values_and_leaves_generation_capacity_alone() -> None:
    values = {"Generation Capacity": 49.9, "IGCPU installed": 50.0, "Largest MEL": 49.96}
    assert census_charts.coinciding_groups(values=values) == [["Largest MEL", "IGCPU installed"]]


def test_values_that_all_differ_form_no_group() -> None:
    values = {"Generation Capacity": 49.9, "IGCPU installed": 50.0, "Largest MEL": 50.2}
    assert census_charts.coinciding_groups(values=values) == []


def test_the_on_line_text_is_built_from_the_names_and_the_value() -> None:
    assert (
        census_charts.online_text(
            names=["Largest MEL", "IGCPU installed", "Generation Capacity"], value=112.0
        )
        == "Largest MEL = IGCPU installed = Generation Capacity = 112.0 MW"
    )
    assert (
        census_charts.online_text(names=["Largest MEL", "IGCPU installed"], value=49.96)
        == "Largest MEL = IGCPU installed = 50.0 MW"
    )


def test_the_on_line_text_goes_where_the_fewest_readings_rise_above_its_line() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    times = [start + timedelta(days=day, hours=12) for day in range(10)]
    # Readings above the 50 MW line on days 0, 1, and 5 only.
    megawatts = [60.0, 60.0, 10.0, 10.0, 10.0, 60.0, 10.0, 10.0, 10.0, 10.0]
    series = pl.DataFrame({"time": times, "megawatts": megawatts})
    left = census_charts.online_text_start(
        series=series, value=50.0, start=start, end=start + timedelta(days=10), width_days=2.5
    )
    assert left == start + timedelta(days=2)


def test_the_on_line_text_takes_the_earliest_place_on_a_tie() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    series = pl.DataFrame({"time": [start], "megawatts": [10.0]})
    left = census_charts.online_text_start(
        series=series, value=50.0, start=start, end=start + timedelta(days=10), width_days=3.0
    )
    assert left == start


def test_the_max_of_output_label_sits_between_its_neighbours_at_larks_green() -> None:
    # Margin values at Larks Green: Generation Capacity, TEC, REPD, P99, and the max of output.
    values = [49.9, 99.4, 70.0, 49.9, 50.1]
    geometry = census_charts.label_geometry(values=values, domain_high=99.4 * 1.1, height_px=230.0)
    assert geometry.domain_high == pytest.approx(99.4 * 1.1)
    assert geometry.line_y_px[4] == pytest.approx((1 - 50.1 / (99.4 * 1.1)) * 230.0)
    order = sorted(range(len(values)), key=lambda i: values[i])
    labels = [geometry.label_y_px[i] for i in order]
    assert order[2] == 4
    gap_px = 230.0 * census_charts.LABEL_GAP_SHARE
    assert all(a - b >= gap_px - 1e-9 for a, b in pairwise(labels))


def test_the_max_of_output_label_is_pushed_clear_of_the_p99_label_at_cleve_hill() -> None:
    # Margin values at Cleve Hill Solar 1: TEC, REPD, P99, and the max of output.
    values = [350.0, 373.0, 88.1, 116.2]
    geometry = census_charts.label_geometry(values=values, domain_high=373.0 * 1.1, height_px=230.0)
    gap = 373.0 * 1.1 * census_charts.LABEL_GAP_SHARE
    assert geometry.label_mw[2] == pytest.approx(88.1)
    assert geometry.label_mw[3] == pytest.approx(88.1 + gap)
    assert geometry.line_y_px[3] < geometry.line_y_px[2]


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
