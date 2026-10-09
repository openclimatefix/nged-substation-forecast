from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from studies.battery_forecast import (
    SYMMETRIC_BANDS,
    asof_at_issue_time,
    band_coverage_and_width,
    climatology_quantiles,
    conformal_quantiles,
    day_shuffle_map,
    day_type,
    issue_time_for,
    leading_idle_end,
    level_weights,
    neighbour_ids,
    neighbour_statistics,
    next_uk_day_hour,
    output_bounds,
    persistence_source_times,
    pinball_losses,
    price_published_at,
    repair_quantiles,
    residual_quantile_table,
    weighted_crps,
    with_issue_time,
)
from studies.cross_validation import QUANTILE_LEVELS, crps

NINE_LEVELS = list(QUANTILE_LEVELS)
THIRTEEN_LEVELS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.98, 0.99]
NO_HOLIDAYS: frozenset[date] = frozenset()


def _utc(day: int, hour: int = 0, minute: int = 0, month: int = 10) -> datetime:
    return datetime(2025, month, day, hour, minute, tzinfo=UTC)


def _history(*, days: int, value_of_day) -> pl.DataFrame:  # noqa: ANN001
    """Half-hourly output from 2025-09-01, whose value on day i is value_of_day(i)."""
    start = datetime(2025, 9, 1, tzinfo=UTC)
    times = [start + timedelta(minutes=30 * i) for i in range(days * 48)]
    return pl.DataFrame(
        {"time": times, "output_mw": [float(value_of_day(i // 48)) for i in range(days * 48)]}
    ).with_columns(pl.col("time").dt.cast_time_unit("us"))


# ---- issue times


def test_issue_times_are_six_and_eighteen_the_day_before_and_gate_closure() -> None:
    target = _utc(10, 13, 30)

    assert issue_time_for(target_start=target, issue="DA-early") == _utc(9, 6)
    assert issue_time_for(target_start=target, issue="DA-late") == _utc(9, 18)
    assert issue_time_for(target_start=target, issue="ID-1h") == _utc(10, 12, 30)


def test_a_first_half_hour_of_the_day_is_issued_on_the_day_before_even_at_one_hour_ahead() -> None:
    assert issue_time_for(target_start=_utc(10, 0, 0), issue="ID-1h") == _utc(9, 23)


def test_the_frame_version_agrees_with_the_scalar_version() -> None:
    times = [_utc(10, 0, 0), _utc(10, 13, 30), _utc(11, 23, 30)]
    frame = pl.DataFrame({"time": times}).with_columns(pl.col("time").dt.cast_time_unit("us"))

    for issue in ("DA-early", "DA-late", "ID-1h"):
        expected = [issue_time_for(target_start=t, issue=issue) for t in times]
        assert with_issue_time(frame=frame, issue=issue)["issue_time"].to_list() == expected


# ---- the as-of join


def _vintages() -> pl.DataFrame:
    valid = _utc(10, 12)
    return pl.DataFrame(
        {
            "time": [valid, valid, valid],
            "publish_time": [_utc(8, 6), _utc(9, 6), _utc(9, 7)],
            "wind_mw": [1.0, 2.0, 3.0],
        }
    )


def _target(issue_time: datetime) -> pl.DataFrame:
    return pl.DataFrame({"time": [_utc(10, 12)], "issue_time": [issue_time]})


def test_asof_uses_the_newest_vintage_published_by_the_issue_time() -> None:
    result = asof_at_issue_time(
        targets=_target(_utc(9, 6, 30)), vintages=_vintages(), value_columns=["wind_mw"]
    )

    assert result["wind_mw"].to_list() == [2.0]


def test_asof_counts_a_vintage_published_exactly_at_the_issue_time() -> None:
    result = asof_at_issue_time(
        targets=_target(_utc(9, 6)), vintages=_vintages(), value_columns=["wind_mw"]
    )

    assert result["wind_mw"].to_list() == [2.0]


def test_asof_never_uses_a_vintage_published_after_the_issue_time() -> None:
    result = asof_at_issue_time(
        targets=_target(_utc(9, 5, 59)), vintages=_vintages(), value_columns=["wind_mw"]
    )

    assert result["wind_mw"].to_list() == [1.0]


def test_asof_gives_null_when_nothing_was_published_yet_and_keeps_the_target_order() -> None:
    targets = pl.DataFrame(
        {"time": [_utc(10, 12), _utc(10, 12)], "issue_time": [_utc(9, 7), _utc(7, 0)]}
    )

    result = asof_at_issue_time(targets=targets, vintages=_vintages(), value_columns=["wind_mw"])

    assert result["issue_time"].to_list() == [_utc(9, 7), _utc(7, 0)]
    assert result["wind_mw"].to_list() == [3.0, None]


# ---- persistence


def test_persistence_at_da_early_is_two_days_back_for_half_hours_ending_after_the_issue() -> None:
    sources = persistence_source_times(
        target_times=[_utc(10, 5, 0), _utc(10, 5, 30), _utc(10, 12, 0)], issue="DA-early"
    )

    # The half-hour 05:00 on the 9th ends at 05:30, before the 06:00 issue time.
    assert sources[0] == _utc(9, 5, 0)
    # The half-hour 05:30 on the 9th ends at 06:00, which is the issue time itself.
    assert sources[1] == _utc(9, 5, 30)
    assert sources[2] == _utc(8, 12, 0)


def test_persistence_at_da_late_uses_the_day_before_only_for_half_hours_ended_by_1800() -> None:
    sources = persistence_source_times(
        target_times=[_utc(10, 17, 30), _utc(10, 18, 0)], issue="DA-late"
    )

    assert sources == [_utc(9, 17, 30), _utc(8, 18, 0)]


def test_persistence_at_one_hour_ahead_is_the_half_hour_that_ended_at_the_issue_time() -> None:
    target = _utc(10, 12, 0)
    (source,) = persistence_source_times(target_times=[target], issue="ID-1h")

    assert source == target - timedelta(hours=1, minutes=30)
    assert source + timedelta(minutes=30) == issue_time_for(target_start=target, issue="ID-1h")


def test_persistence_never_ends_after_the_issue_time_at_any_half_hour_of_the_day() -> None:
    targets = [_utc(10) + timedelta(minutes=30 * i) for i in range(48)]

    for issue in ("DA-early", "DA-late", "ID-1h"):
        sources = persistence_source_times(target_times=targets, issue=issue)
        for target, source in zip(targets, sources, strict=True):
            issued = issue_time_for(target_start=target, issue=issue)
            assert source + timedelta(minutes=30) <= issued
            assert source < target


# ---- climatology


def _climatology_targets(*, times: list[datetime], issue: str) -> pl.DataFrame:
    frame = pl.DataFrame({"time": times}).with_columns(pl.col("time").dt.cast_time_unit("us"))
    return with_issue_time(frame=frame, issue=issue)  # ty: ignore[invalid-argument-type]


def test_climatology_window_stops_at_the_last_complete_day() -> None:
    # Value equals the day index, so the window's days can be read off the quantiles.
    history = _history(days=60, value_of_day=lambda i: i)
    target = datetime(2025, 10, 22, 12, 0, tzinfo=UTC)  # day index 51, a Wednesday
    targets = _climatology_targets(times=[target], issue="DA-early")

    result = climatology_quantiles(
        history=history,
        targets=targets,
        levels=[0.0, 1.0],
        non_working_dates=NO_HOLIDAYS,
        window_days=10,
    )

    # DA-early issue is 06:00 on day 50, so the last complete day is 49, a Monday and a working
    # day. The newest working day in the ten-day window 40 to 49 is therefore day 49 itself; an
    # off-by-one that admitted the issue day (day 50, a Tuesday) would return 50.
    assert result["q1.0"].to_list() == [49.0]
    assert result["q0.0"].to_list()[0] >= 40.0


def test_climatology_at_da_late_ignores_the_afternoon_of_the_day_before() -> None:
    # Day 50 is the working day before the target (day 51). Make its values huge: they must not
    # enter, because part of it ends after the 18:00 issue time.
    history = _history(days=60, value_of_day=lambda i: 1000.0 if i == 50 else 1.0)
    target = datetime(2025, 10, 22, 12, 0, tzinfo=UTC)
    targets = _climatology_targets(times=[target], issue="DA-late")

    result = climatology_quantiles(
        history=history, targets=targets, levels=[1.0], non_working_dates=NO_HOLIDAYS
    )

    assert result["q1.0"].to_list() == [1.0]


def test_climatology_at_one_hour_ahead_includes_the_day_before_but_not_the_target_day() -> None:
    history = _history(days=60, value_of_day=lambda i: 1000.0 if i == 51 else float(i))
    # A Wednesday, day index 51, is the target day itself; day 50, a Tuesday, is the day before.
    target = datetime(2025, 10, 22, 12, 0, tzinfo=UTC)
    targets = _climatology_targets(times=[target], issue="ID-1h")

    result = climatology_quantiles(
        history=history, targets=targets, levels=[1.0], non_working_dates=NO_HOLIDAYS
    )

    assert result["q1.0"].to_list() == [50.0]


def test_climatology_with_a_target_before_any_history_has_an_empty_sample() -> None:
    # The target is on the history's first day, so the last complete day is two days before it.
    # An unclamped negative window stop would wrap round and take almost the whole history.
    history = _history(days=60, value_of_day=float)
    target = datetime(2025, 9, 1, 12, 0, tzinfo=UTC)
    targets = _climatology_targets(times=[target], issue="DA-early")

    result = climatology_quantiles(
        history=history, targets=targets, levels=[0.5], non_working_dates=NO_HOLIDAYS
    )

    assert result["climatology_n"].to_list() == [0]
    assert result["q0.5"].to_list() == [None]


def test_climatology_matches_day_type() -> None:
    history = _history(days=60, value_of_day=lambda i: 100.0 if (i + 0) % 7 in (5, 6) else 1.0)
    # 2025-09-01 is a Monday, so index % 7 in (5, 6) is Saturday or Sunday.
    weekday_target = datetime(2025, 10, 22, 12, 0, tzinfo=UTC)
    weekend_target = datetime(2025, 10, 25, 12, 0, tzinfo=UTC)
    targets = _climatology_targets(times=[weekday_target, weekend_target], issue="DA-early")

    result = climatology_quantiles(
        history=history, targets=targets, levels=[0.5], non_working_dates=NO_HOLIDAYS
    )

    assert result["q0.5"].to_list() == [1.0, 100.0]


def test_climatology_treats_a_bank_holiday_as_non_working() -> None:
    history = _history(days=60, value_of_day=lambda i: 100.0 if i == 35 else 1.0)
    holiday = date(2025, 10, 6)  # day index 35, a Monday
    target = datetime(2025, 10, 20, 12, 0, tzinfo=UTC)  # a Monday, a holiday too in this test
    targets = _climatology_targets(times=[target], issue="DA-early")

    result = climatology_quantiles(
        history=history,
        targets=targets,
        levels=[1.0],
        non_working_dates=frozenset({holiday, target.date()}),
    )

    # The window is days 0 to 47: 13 weekend days plus the holiday, 14 days, each read at the
    # target half-hour and its two neighbours. The holiday's value 100 is the sample's maximum, so
    # a day type that ignored bank holidays would return 1.0 and a sample of 39.
    assert result["climatology_n"].to_list() == [14 * 3]
    assert result["q1.0"].to_list() == [100.0]


def test_climatology_uses_the_neighbouring_half_hours_and_reports_the_sample_size() -> None:
    history = _history(days=60, value_of_day=lambda i: 0.0)
    middle = datetime(2025, 10, 20, 12, 0, tzinfo=UTC)
    first = datetime(2025, 10, 20, 0, 0, tzinfo=UTC)
    targets = _climatology_targets(times=[middle, first], issue="DA-early")

    result = climatology_quantiles(
        history=history, targets=targets, levels=[0.5], non_working_dates=NO_HOLIDAYS
    )

    n_middle, n_first = result["climatology_n"].to_list()
    assert n_middle * 2 == n_first * 3  # three half-hours against two at the edge of the day


def test_climatology_with_no_history_before_the_issue_time_is_null() -> None:
    history = _history(days=5, value_of_day=lambda i: 1.0)
    targets = _climatology_targets(times=[datetime(2025, 9, 2, 12, tzinfo=UTC)], issue="DA-early")

    result = climatology_quantiles(
        history=history, targets=targets, levels=[0.5], non_working_dates=NO_HOLIDAYS
    )

    assert result["q0.5"].to_list() == [None]
    assert result["climatology_n"].to_list() == [0]


# ---- conformal quantiles


def test_residual_quantiles_are_computed_within_each_group_and_skip_nans() -> None:
    residuals = np.array([1.0, 2.0, 3.0, 10.0, 20.0, np.nan])
    groups = np.array([0, 0, 0, 1, 1, 1])

    table = residual_quantile_table(residuals=residuals, groups=groups, levels=[0.0, 1.0])

    assert table[0].tolist() == [1.0, 3.0]
    assert table[1].tolist() == [10.0, 20.0]


def test_conformal_quantiles_add_the_group_residuals_to_each_centre() -> None:
    table = {0: np.array([-1.0, 1.0]), 1: np.array([-10.0, 10.0])}

    result = conformal_quantiles(
        centre=np.array([5.0, 50.0, 7.0]), groups=np.array([0, 1, 2]), table=table
    )

    assert result[0].tolist() == [4.0, 6.0]
    assert result[1].tolist() == [40.0, 60.0]
    assert np.isnan(result[2]).all()


# ---- repair and bounds


def test_repair_sorts_crossing_quantiles_then_clips_to_the_bounds() -> None:
    quantiles = np.array([[3.0, -9.0, 1.0, 20.0]])

    repaired = repair_quantiles(quantiles=quantiles, lower=-5.0, upper=10.0)

    assert repaired.tolist() == [[-5.0, 1.0, 3.0, 10.0]]


def test_output_bounds_are_the_tenth_and_ninety_ninth_ninth_percentiles() -> None:
    output = np.linspace(-1000.0, 1000.0, 20001)

    lower, upper = output_bounds(training_output=np.append(output, np.nan))

    assert lower == pytest.approx(-998.0)
    assert upper == pytest.approx(998.0)


# ---- scores


def test_level_weights_of_nine_equal_levels_are_the_riemann_spacing() -> None:
    assert level_weights(levels=NINE_LEVELS) == pytest.approx(np.full(9, 0.1))


def test_level_weights_give_a_wider_gap_more_weight() -> None:
    weights = level_weights(levels=THIRTEEN_LEVELS)

    assert weights[6] == pytest.approx((0.65 - 0.35) / 2)
    assert weights[0] == pytest.approx(0.02 / 2)
    assert weights[-1] == pytest.approx((1.0 - 0.98) / 2)


def test_weighted_crps_equals_the_cross_validation_crps_on_its_nine_levels() -> None:
    rng = np.random.default_rng(0)
    truth = rng.normal(size=200)
    quantiles = np.sort(rng.normal(size=(200, 9)), axis=1)

    expected = crps(actual=truth, quantiles=quantiles)
    actual = weighted_crps(truth=truth, quantiles=quantiles, levels=NINE_LEVELS)

    assert actual == pytest.approx(expected)


def test_a_forecast_whose_every_quantile_equals_the_truth_scores_zero() -> None:
    truth = np.array([3.0, -2.0, 0.0])
    quantiles = np.repeat(truth[:, None], 13, axis=1)

    assert weighted_crps(truth=truth, quantiles=quantiles, levels=THIRTEEN_LEVELS) == pytest.approx(
        0.0
    )


def test_crps_of_a_point_mass_is_the_absolute_error_scaled_by_the_covered_probability() -> None:
    quantiles = np.zeros((1, 13))

    score = weighted_crps(truth=np.array([4.0]), quantiles=quantiles, levels=THIRTEEN_LEVELS)

    # The levels cover 0.99 - 0.01 + ... the weights sum to (1 + 0.99 - 0.01) / 2 = 0.99.
    assert score[0] == pytest.approx(4.0 * 0.99)


def test_coverage_counts_a_value_on_the_band_edge_as_inside() -> None:
    quantiles = np.tile(np.arange(13, dtype=float), (4, 1))  # level i has value i
    truth = np.array([3.0, 9.0, 2.9, 9.1])  # p10 is index 3, p90 is index 9

    table = band_coverage_and_width(truth=truth, quantiles=quantiles, levels=THIRTEEN_LEVELS)
    row = table.filter(pl.col("lower_level") == 0.10).row(0, named=True)

    assert row["coverage"] == 0.5
    assert row["mean_width"] == 6.0
    assert row["nominal_coverage"] == pytest.approx(0.8)
    assert table.height == len(SYMMETRIC_BANDS)


def test_a_wider_band_never_covers_less_than_a_narrower_one() -> None:
    rng = np.random.default_rng(1)
    truth = rng.normal(size=500)
    quantiles = np.tile(np.quantile(rng.normal(size=5000), THIRTEEN_LEVELS), (500, 1))

    table = band_coverage_and_width(truth=truth, quantiles=quantiles, levels=THIRTEEN_LEVELS)

    assert table["coverage"].to_list() == sorted(table["coverage"].to_list())
    assert table["coverage"].to_list()[2] == pytest.approx(0.8, abs=0.08)


# ---- the day shuffle


def test_shuffle_never_draws_the_day_itself_and_keeps_month_and_day_type() -> None:
    days = [date(2025, 10, 1) + timedelta(days=i) for i in range(61)]

    drawn = day_shuffle_map(days=days, non_working_dates=NO_HOLIDAYS, seed=3)

    assert drawn.height == 61
    for row in drawn.iter_rows(named=True):
        assert row["source_date"] != row["date"]
        assert (row["source_date"].year, row["source_date"].month) == (
            row["date"].year,
            row["date"].month,
        )
        assert day_type(day=row["source_date"], non_working_dates=NO_HOLIDAYS) == day_type(
            day=row["date"], non_working_dates=NO_HOLIDAYS
        )


def test_shuffle_is_fixed_by_the_seed_and_changes_with_it() -> None:
    days = [date(2025, 10, 1) + timedelta(days=i) for i in range(31)]

    first = day_shuffle_map(days=days, non_working_dates=NO_HOLIDAYS, seed=1)
    again = day_shuffle_map(days=days, non_working_dates=NO_HOLIDAYS, seed=1)
    other = day_shuffle_map(days=days, non_working_dates=NO_HOLIDAYS, seed=2)

    assert first.equals(again)
    assert not first.equals(other)


def test_shuffle_treats_a_bank_holiday_as_a_non_working_day() -> None:
    days = [date(2025, 12, 1) + timedelta(days=i) for i in range(31)]
    holiday = date(2025, 12, 25)  # a Thursday

    drawn = day_shuffle_map(days=days, non_working_dates=frozenset({holiday}), seed=0)

    source = drawn.filter(pl.col("date") == holiday)["source_date"].to_list()[0]
    assert source.weekday() >= 5 or source == holiday
    assert source != holiday


def test_shuffle_raises_when_a_month_has_a_lone_day_of_one_type() -> None:
    days = [date(2025, 10, 1), date(2025, 10, 2), date(2025, 10, 4)]

    with pytest.raises(ValueError, match="only one"):
        day_shuffle_map(days=days, non_working_dates=NO_HOLIDAYS, seed=0)


# ---- neighbours


def test_neighbours_exclude_the_target_and_its_whole_lead_party() -> None:
    parties = {"a1": "A", "a2": "A", "b1": "B", "c1": "C"}

    assert neighbour_ids(target="a1", lead_party=parties) == ["b1", "c1"]
    assert neighbour_ids(target="a1", lead_party=parties, rule="same_party") == ["a2"]


def _fpn(rows: list[tuple[int, str, float | None]]) -> pl.DataFrame:
    base = datetime(2025, 10, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            "time": [base + timedelta(minutes=30 * minutes) for minutes, _, _ in rows],
            "bmu_id": [b for _, b, _ in rows],
            "fpn_mw": [v for _, _, v in rows],
        },
        schema_overrides={"time": pl.Datetime("us", "UTC")},
    )


def test_neighbour_statistics_scale_each_fpn_by_its_own_battery_and_average() -> None:
    fpn = _fpn([(0, "x", 50.0), (0, "y", -10.0), (0, "target", 999.0)])

    result = neighbour_statistics(fpn=fpn, neighbours=["x", "y"], p99_mw={"x": 100.0, "y": 20.0})

    assert result["neighbour_mean_fraction"].to_list() == [pytest.approx((0.5 - 0.5) / 2)]
    assert result["neighbour_discharging_share"].to_list() == [0.5]


def test_a_missing_neighbour_drops_out_instead_of_counting_as_zero() -> None:
    fpn = _fpn([(0, "x", 50.0), (0, "y", None), (1, "x", 50.0), (1, "y", 100.0)])

    result = neighbour_statistics(fpn=fpn, neighbours=["x", "y"], p99_mw={"x": 100.0, "y": 100.0})

    assert result["neighbour_mean_fraction"].to_list() == [0.5, 0.75]
    assert result["neighbour_discharging_share"].to_list() == [1.0, 1.0]


def test_the_previous_slot_holds_the_half_hour_before_and_is_null_at_the_start() -> None:
    fpn = _fpn([(0, "x", 10.0), (1, "x", 20.0), (2, "x", 40.0)])

    result = neighbour_statistics(fpn=fpn, neighbours=["x"], p99_mw={"x": 100.0})

    assert result["neighbour_mean_fraction_previous"].to_list() == [None, 0.1, 0.2]


def test_a_notification_for_a_later_half_hour_never_enters_an_earlier_row() -> None:
    base = _fpn([(0, "x", 10.0), (1, "x", 20.0), (2, "x", 40.0)])
    changed = _fpn([(0, "x", 10.0), (1, "x", 20.0), (2, "x", -90.0)])

    first = neighbour_statistics(fpn=base, neighbours=["x"], p99_mw={"x": 100.0})
    second = neighbour_statistics(fpn=changed, neighbours=["x"], p99_mw={"x": 100.0})

    assert first.head(2).equals(second.head(2))
    assert not first.equals(second)


def test_no_neighbours_give_an_empty_frame_with_the_same_columns() -> None:
    fpn = _fpn([(0, "x", 10.0)])

    result = neighbour_statistics(fpn=fpn, neighbours=[], p99_mw={})

    assert result.height == 0
    assert result.columns == [
        "time",
        "neighbour_mean_fraction",
        "neighbour_mean_fraction_previous",
        "neighbour_discharging_share",
    ]


# ---- the UK delivery day of the day-ahead auction


def _published_at(hour_start: datetime) -> datetime:
    frame = pl.DataFrame({"time": [hour_start]}).with_columns(
        pl.col("time").dt.cast_time_unit("us")
    )
    return frame.select(published=price_published_at(time=pl.col("time")))["published"][0]


def test_the_last_utc_hour_of_a_summer_day_belongs_to_the_next_uk_day() -> None:
    frame = pl.DataFrame(
        {
            "time": [
                datetime(2025, 7, 1, 22, tzinfo=UTC),
                datetime(2025, 7, 1, 23, tzinfo=UTC),
                datetime(2025, 12, 1, 23, tzinfo=UTC),
            ]
        }
    ).with_columns(pl.col("time").dt.cast_time_unit("us"))

    flags = frame.select(flag=next_uk_day_hour(time=pl.col("time")))["flag"].to_list()

    assert flags == [False, True, False]


def test_a_summer_hours_price_is_public_before_the_uk_midnight_that_starts_its_day() -> None:
    # 23:00 UTC on 1 July is midnight UK time on 2 July, so the day-ahead auction for 2 July
    # (public by 10:00 UTC on 1 July) prices it. 22:00 UTC is 23:00 UK time on 1 July.
    assert _published_at(datetime(2025, 7, 1, 23, tzinfo=UTC)) == datetime(
        2025, 7, 1, 10, tzinfo=UTC
    )
    assert _published_at(datetime(2025, 7, 1, 22, tzinfo=UTC)) == datetime(
        2025, 6, 30, 10, tzinfo=UTC
    )


def test_a_winter_hours_price_is_public_by_the_morning_before_its_day() -> None:
    assert _published_at(datetime(2025, 12, 1, 23, tzinfo=UTC)) == datetime(
        2025, 11, 30, 11, tzinfo=UTC
    )


def test_the_spring_clock_change_day_still_has_its_last_utc_hour_in_the_next_uk_day() -> None:
    # 29 March 2026: clocks go forward at 01:00 UTC, so 23:00 UTC is midnight on 30 March.
    assert _published_at(datetime(2026, 3, 29, 23, tzinfo=UTC)) == datetime(
        2026, 3, 29, 10, tzinfo=UTC
    )


# ---- the idle lead-in


def _monthly_output(*, zero_share_by_month: list[float]) -> pl.DataFrame:
    """Output from 1 September 2025 with a given share of exact zeros in each calendar month."""
    times, values = [], []
    for block, share in enumerate(zero_share_by_month):
        start = datetime(2025, 9 + block, 1, tzinfo=UTC)
        for i in range(100):
            times.append(start + timedelta(hours=i))
            values.append(0.0 if i < round(share * 100) else 5.0)
    return pl.DataFrame({"time": times, "output_mw": values}).with_columns(
        pl.col("time").dt.cast_time_unit("us")
    )


def test_a_battery_that_is_idle_for_its_first_months_has_an_idle_lead_in() -> None:
    output = _monthly_output(zero_share_by_month=[1.0, 1.0, 0.2, 0.0])

    end = leading_idle_end(output=output)

    assert end is not None
    assert end.date() == date(2025, 11, 1)  # the first calendar month that is not idle


def test_a_working_battery_has_no_idle_lead_in() -> None:
    assert leading_idle_end(output=_monthly_output(zero_share_by_month=[0.3, 0.0])) is None


def test_an_outage_after_the_first_working_month_is_not_a_lead_in() -> None:
    assert leading_idle_end(output=_monthly_output(zero_share_by_month=[0.0, 1.0, 1.0])) is None


def test_a_month_with_ninety_nine_per_cent_zeros_is_idle_and_with_ninety_eight_is_not() -> None:
    idle = leading_idle_end(output=_monthly_output(zero_share_by_month=[0.99, 0.0]))
    working = leading_idle_end(output=_monthly_output(zero_share_by_month=[0.98, 0.0]))

    assert idle is not None
    assert working is None


def test_a_battery_idle_throughout_has_a_lead_in_that_covers_the_whole_series() -> None:
    output = _monthly_output(zero_share_by_month=[1.0, 1.0])

    end = leading_idle_end(output=output)

    assert end is not None
    assert end > output.select(pl.col("time").max()).item()


# ---- pinball loss


def test_pinball_loss_charges_an_underforecast_at_the_level_and_an_overforecast_at_the_rest() -> (
    None
):
    quantiles = np.array([[2.0, 2.0]])
    levels = [0.2, 0.9]

    under = pinball_losses(truth=np.array([5.0]), quantiles=quantiles, levels=levels)
    over = pinball_losses(truth=np.array([1.0]), quantiles=quantiles, levels=levels)

    assert under[0].tolist() == pytest.approx([0.2 * 3.0, 0.9 * 3.0])
    assert over[0].tolist() == pytest.approx([0.8 * 1.0, 0.1 * 1.0])
