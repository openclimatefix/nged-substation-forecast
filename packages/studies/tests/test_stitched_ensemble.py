from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from studies.stitched_ensemble import (
    hold_backward_mean_hourly,
    interpolate_clearness_hourly,
    interpolate_instants_hourly,
    newest_run_member_means,
)

DAY = datetime(2026, 7, 1, tzinfo=UTC)


def _member_rows(
    *, site: str, init_time: datetime, valid_time: datetime, values: list[float]
) -> pl.DataFrame:
    """One row per member of one run at one valid time, member `m` reading `values[m]`."""
    return pl.DataFrame(
        {
            "site": site,
            "init_time": init_time,
            "valid_time": valid_time,
            "lead_hours": int((valid_time - init_time).total_seconds() // 3600),
            "ensemble_member": list(range(len(values))),
            "value": values,
        }
    )


def test_the_newest_run_supplies_a_valid_time_two_runs_cover():
    valid = DAY + timedelta(days=1)
    older = _member_rows(site="A", init_time=DAY, valid_time=valid, values=[0.0, 10.0])
    newer = _member_rows(site="A", init_time=valid, valid_time=valid, values=[20.0, 40.0])

    means = newest_run_member_means(
        members=pl.concat([newer, older]),
        value_columns=["value"],
        expected_members=2,
        max_lead_hours=69,
    )

    assert means["init_time"].to_list() == [valid]
    assert means["value"].to_list() == [30.0]


def test_a_lead_beyond_the_maximum_is_never_used():
    near = _member_rows(
        site="A", init_time=DAY, valid_time=DAY + timedelta(hours=24), values=[1.0, 3.0]
    )
    far = _member_rows(
        site="A", init_time=DAY, valid_time=DAY + timedelta(hours=72), values=[5.0, 7.0]
    )

    means = newest_run_member_means(
        members=pl.concat([near, far]),
        value_columns=["value"],
        expected_members=2,
        max_lead_hours=69,
    )

    assert means["valid_time"].to_list() == [DAY + timedelta(hours=24)]


def test_a_lead_of_exactly_the_maximum_is_used():
    rows = _member_rows(
        site="A", init_time=DAY, valid_time=DAY + timedelta(hours=69), values=[1.0, 3.0]
    )

    means = newest_run_member_means(
        members=rows, value_columns=["value"], expected_members=2, max_lead_hours=69
    )

    assert means["value"].to_list() == [2.0]


def test_a_run_missing_a_member_raises():
    rows = _member_rows(
        site="A", init_time=DAY, valid_time=DAY + timedelta(hours=3), values=[1.0, 3.0, 5.0]
    ).filter(pl.col("ensemble_member") != 2)

    with pytest.raises(ValueError, match="other than 3 members"):
        newest_run_member_means(
            members=rows, value_columns=["value"], expected_members=3, max_lead_hours=69
        )


def test_each_site_keeps_its_own_newest_run():
    valid = DAY + timedelta(hours=3)
    site_a = _member_rows(site="A", init_time=DAY, valid_time=valid, values=[1.0])
    site_b_old = _member_rows(site="B", init_time=DAY, valid_time=valid, values=[2.0])
    site_b_new = _member_rows(
        site="B", init_time=DAY + timedelta(hours=1), valid_time=valid, values=[4.0]
    )

    means = newest_run_member_means(
        members=pl.concat([site_a, site_b_old, site_b_new]),
        value_columns=["value"],
        expected_members=1,
        max_lead_hours=69,
    )

    assert means.sort("site")["value"].to_list() == [1.0, 4.0]


def _steps(*, values: list[float], site: str = "A") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "site": site,
            "valid_time": [DAY + timedelta(hours=3 * index) for index in range(len(values))],
            "value": values,
        }
    )


def test_a_three_hour_mean_is_held_for_the_three_hours_ending_at_its_label():
    hourly = hold_backward_mean_hourly(steps=_steps(values=[10.0, 20.0]), value_columns=["value"])

    # The step labelled 00:00 covers the hours labelled 22:00, 23:00 and 00:00.
    assert hourly["time"].to_list() == [
        DAY - timedelta(hours=2),
        DAY - timedelta(hours=1),
        DAY,
        DAY + timedelta(hours=1),
        DAY + timedelta(hours=2),
        DAY + timedelta(hours=3),
    ]
    assert hourly["value"].to_list() == [10.0, 10.0, 10.0, 20.0, 20.0, 20.0]


def test_instants_are_interpolated_linearly_between_steps():
    hourly = interpolate_instants_hourly(
        steps=_steps(values=[0.0, 3.0, 9.0]), value_columns=["value"]
    )

    assert hourly["time"].to_list() == [DAY + timedelta(hours=hour) for hour in range(7)]
    assert hourly["value"].to_list() == pytest.approx([0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 9.0])


def test_interpolation_never_crosses_a_site_boundary():
    steps = pl.concat([_steps(values=[0.0, 3.0]), _steps(values=[100.0, 130.0], site="B")])

    hourly = interpolate_instants_hourly(steps=steps, value_columns=["value"])

    assert hourly.filter(pl.col("site") == "B")["value"].to_list() == pytest.approx(
        [100.0, 110.0, 120.0, 130.0]
    )
    assert hourly.filter(pl.col("site") == "A")["value"].to_list() == pytest.approx(
        [0.0, 1.0, 2.0, 3.0]
    )


def test_interpolation_across_a_missing_step_raises():
    steps = _steps(values=[0.0, 3.0, 6.0]).filter(pl.col("value") != 3.0)

    with pytest.raises(ValueError, match="3 hours"):
        interpolate_instants_hourly(steps=steps, value_columns=["value"])


def test_interpolation_across_a_gap_shorter_than_a_step_raises():
    steps = _steps(values=[0.0, 3.0, 6.0])
    steps = steps.with_columns(
        valid_time=pl.Series([DAY, DAY + timedelta(hours=3), DAY + timedelta(hours=5)])
    )

    with pytest.raises(ValueError, match="3 hours"):
        interpolate_instants_hourly(steps=steps, value_columns=["value"])


def _hourly_extraterrestrial(*, first: datetime, values: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "site": "A",
            "time": [first + timedelta(hours=hour) for hour in range(len(values))],
            "extraterrestrial_horizontal_w_m2": values,
        }
    )


def test_clearness_interpolation_ramps_between_two_steps_and_keeps_the_sun_shape():
    extraterrestrial = _hourly_extraterrestrial(
        first=DAY + timedelta(hours=1), values=[100.0, 100.0, 100.0, 200.0, 200.0, 200.0]
    )
    steps = pl.DataFrame(
        {
            "site": "A",
            "valid_time": [DAY + timedelta(hours=3), DAY + timedelta(hours=6)],
            "ghi": [50.0, 200.0],
        }
    )

    hourly = interpolate_clearness_hourly(
        steps=steps, extraterrestrial_hourly=extraterrestrial, value_column="ghi"
    )

    assert hourly["ghi"].to_list() == pytest.approx(
        [50.0, 50.0, 100.0 * (0.5 + 0.5 / 3), 200.0 * (1.0 - 0.5 / 3), 200.0, 200.0]
    )


def test_a_step_with_no_daylight_gives_zero_and_leaves_its_neighbour_unblended():
    extraterrestrial = _hourly_extraterrestrial(
        first=DAY + timedelta(hours=1), values=[0.0, 0.0, 0.0, 100.0, 100.0, 100.0]
    )
    steps = pl.DataFrame(
        {
            "site": "A",
            "valid_time": [DAY + timedelta(hours=3), DAY + timedelta(hours=6)],
            "ghi": [0.0, 80.0],
        }
    )

    hourly = interpolate_clearness_hourly(
        steps=steps, extraterrestrial_hourly=extraterrestrial, value_column="ghi"
    )

    assert hourly["ghi"].to_list() == pytest.approx([0.0, 0.0, 0.0, 80.0, 80.0, 80.0])


def test_an_hour_with_no_extraterrestrial_value_raises():
    extraterrestrial = _hourly_extraterrestrial(first=DAY + timedelta(hours=1), values=[1.0, 1.0])
    steps = pl.DataFrame({"site": "A", "valid_time": [DAY + timedelta(hours=3)], "ghi": [1.0]})

    with pytest.raises(ValueError, match="no extraterrestrial"):
        interpolate_clearness_hourly(
            steps=steps, extraterrestrial_hourly=extraterrestrial, value_column="ghi"
        )


def test_a_partly_lit_step_keeps_its_clearness_and_zeroes_the_dark_hours():
    extraterrestrial = _hourly_extraterrestrial(
        first=DAY + timedelta(hours=1), values=[0.0, 0.0, 100.0]
    )
    steps = pl.DataFrame({"site": "A", "valid_time": [DAY + timedelta(hours=3)], "ghi": [30.0]})

    hourly = interpolate_clearness_hourly(
        steps=steps, extraterrestrial_hourly=extraterrestrial, value_column="ghi"
    )

    assert hourly["ghi"].to_list() == pytest.approx([0.0, 0.0, 90.0])


def test_a_clearness_index_above_the_maximum_is_clipped():
    extraterrestrial = _hourly_extraterrestrial(
        first=DAY + timedelta(hours=1), values=[100.0, 100.0, 100.0]
    )
    steps = pl.DataFrame({"site": "A", "valid_time": [DAY + timedelta(hours=3)], "ghi": [1000.0]})

    hourly = interpolate_clearness_hourly(
        steps=steps, extraterrestrial_hourly=extraterrestrial, value_column="ghi"
    )

    assert hourly["ghi"].to_list() == pytest.approx([150.0, 150.0, 150.0])


def test_two_sites_never_borrow_each_others_neighbouring_step():
    hours = [100.0] * 6
    extraterrestrial = pl.concat(
        [
            _hourly_extraterrestrial(first=DAY + timedelta(hours=1), values=hours),
            _hourly_extraterrestrial(first=DAY + timedelta(hours=1), values=hours).with_columns(
                site=pl.lit("B")
            ),
        ]
    )
    steps = pl.DataFrame(
        {
            "site": ["A", "A", "B"],
            "valid_time": [
                DAY + timedelta(hours=3),
                DAY + timedelta(hours=6),
                DAY + timedelta(hours=3),
            ],
            "ghi": [50.0, 100.0, 20.0],
        }
    )

    hourly = interpolate_clearness_hourly(
        steps=steps, extraterrestrial_hourly=extraterrestrial, value_column="ghi"
    )

    site_b = hourly.filter(pl.col("site") == "B").filter(pl.col("time") <= DAY + timedelta(hours=3))
    assert site_b["ghi"].to_list() == pytest.approx([20.0, 20.0, 20.0])
