from datetime import UTC, datetime, timedelta
from pathlib import Path

import forecast_inputs as fi
import polars as pl
import pytest

START = datetime(2025, 9, 1, tzinfo=UTC)


def _utc(month: int, day: int, hour: int = 0, minute: int = 0, year: int = 2025) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def _output(*, value_of_time) -> pl.DataFrame:  # noqa: ANN001
    grid = fi.half_hour_grid()
    return grid.select(
        "time", output_mw=pl.Series([float(value_of_time(t)) for t in grid["time"].to_list()])
    )


def _battery(*, battery_id: str, party: str | None, value_of_time, has_fpn: bool = True):  # noqa: ANN001, ANN202
    return fi.Battery(
        battery_id=battery_id,
        output=_output(value_of_time=value_of_time),
        p99_mw=10.0,
        lead_party=party,
        has_fpn=has_fpn,
    )


def _fpn(*, battery_id: str, value_of_time) -> pl.DataFrame:  # noqa: ANN001
    grid = fi.half_hour_grid()
    return grid.select(
        bmu_id=pl.lit(battery_id),
        time=pl.col("time"),
        fpn_mw=pl.Series([float(value_of_time(t)) for t in grid["time"].to_list()]),
    )


def _day_index(t: datetime) -> int:
    return (t - START).days


def test_the_grid_is_one_unbroken_year_of_half_hours() -> None:
    grid = fi.half_hour_grid()

    assert grid.height == fi.WINDOW_HALF_HOURS
    assert grid["time"].diff().drop_nulls().unique().to_list() == [timedelta(minutes=30)]
    assert grid["time"][0] == START


def test_price_columns_rank_within_the_day_and_summarise_it() -> None:
    times = [START + timedelta(minutes=30 * i) for i in range(96)]
    prices = [float(i % 48) for i in range(48)] + [100.0 + (i % 48) * 2 for i in range(48)]
    frame = pl.DataFrame({"time": times, "price_actual": prices}).with_columns(
        pl.col("time").dt.cast_time_unit("us")
    )

    result = fi.price_columns(frame=frame, source="actual")

    assert result["price_rank_actual"][0] == pytest.approx(0.5 / 48)
    assert result["price_rank_actual"][47] == pytest.approx(47.5 / 48)
    assert result["price_rank_actual"][48] == pytest.approx(0.5 / 48)
    assert result["price_day_mean_actual"][:2].to_list() == [23.5, 23.5]
    assert result["price_day_range_actual"][50] == 94.0
    assert result["rank_rule_actual"].len() == 96


def test_a_day_with_a_missing_price_gets_a_zero_rank_rule_schedule() -> None:
    times = [START + timedelta(minutes=30 * i) for i in range(96)]
    prices = [float(i % 48) for i in range(48)] + [float(100 - (i % 48)) for i in range(48)]
    prices[60] = None  # ty: ignore[invalid-assignment]
    frame = pl.DataFrame({"time": times, "price_actual": prices}).with_columns(
        pl.col("time").dt.cast_time_unit("us")
    )

    result = fi.price_columns(frame=frame, source="actual")

    assert result["rank_rule_actual"][48:].sum() == 0.0


def _synthetic_market(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every hour's price equal its day index times 1000 plus its hour of day."""
    grid = fi.half_hour_grid()
    market = grid.select(
        "time",
        day_ahead_gbp_per_mwh=pl.col("time").map_elements(
            lambda t: _day_index(t) * 1000.0 + t.hour, return_dtype=pl.Float64
        ),
    )
    monkeypatch.setattr(fi, "market_frame", lambda: market)


def test_the_naive_price_is_the_price_seven_days_earlier(monkeypatch: pytest.MonkeyPatch) -> None:
    _synthetic_market(monkeypatch)

    prices = fi.price_sources(grid=fi.half_hour_grid())

    row = prices.filter(pl.col("time") == _utc(9, 20, 5, 30)).row(0, named=True)
    assert row["price_naive"] == row["price_actual"] - 7000.0
    assert prices["price_naive"][: 7 * 48].null_count() == 7 * 48


def test_the_shuffled_price_comes_from_another_day_of_the_same_month_and_half_hour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _synthetic_market(monkeypatch)

    prices = fi.price_sources(grid=fi.half_hour_grid())

    day = (prices["price_shuffled"] // 1000).cast(pl.Int64)
    own_day = prices["price_actual"] // 1000
    source_dates = [START.date() + timedelta(days=int(d)) for d in day.to_list()]
    own_dates = prices["time"].dt.date().to_list()
    assert (day != own_day.cast(pl.Int64)).all()
    assert all(s.month == o.month for s, o in zip(source_dates, own_dates, strict=True))
    assert (prices["price_shuffled"] % 1000 == prices["price_actual"] % 1000).all()


def test_the_wind_forecast_at_issue_is_the_newest_published_by_then(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "elexon_windfor"
    folder.mkdir()
    valid = _utc(9, 3, 12)
    pl.DataFrame(
        {
            "publish_time": [_utc(9, 1, 5, 30), _utc(9, 2, 6, 0), _utc(9, 2, 6, 30)],
            "time": [valid, valid, valid],
            "forecast_mw": [1.0, 2.0, 3.0],
        }
    ).write_parquet(folder / "elexon_windfor.parquet")
    monkeypatch.setattr(fi, "MARKET_DOWNLOADS_DIR", tmp_path)
    grid = pl.DataFrame({"time": [valid, valid + timedelta(minutes=30)]})

    result = fi.wind_forecast_at_issue(issue="DA-early", grid=grid)

    # The DA-early issue time is 06:00 on the 2nd: the 06:30 publication is not known yet.
    assert result["wind_forecast_mw"].to_list() == [2.0, 2.0]


def test_the_testbed_keeps_embedded_batteries_with_enough_output_and_notification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    listed = pl.DataFrame(
        {
            "elexon_bmu_id": ["E_FULL-1", "E_NOPN-1", "E_SHORT-1", "T_BIG-1"],
            "bmu_name": ["a", "b", "c", "d"],
            "lead_party_name": ["P1", "P2", "P3", "P4"],
            "gsp_group_name": ["Eastern"] * 4,
            "generation_capacity_mw": [50.0] * 4,
            "bmu_type": ["E", "E", "E", "T"],
        }
    )
    csv = tmp_path / "bmu_list.csv"
    listed.write_csv(csv)
    for bmu_id, rows in [
        ("E_FULL-1", fi.WINDOW_HALF_HOURS),
        ("E_NOPN-1", fi.WINDOW_HALF_HOURS),
        ("E_SHORT-1", fi.WINDOW_HALF_HOURS // 2),
        ("T_BIG-1", fi.WINDOW_HALF_HOURS),
    ]:
        pl.DataFrame({"x": range(rows)}).write_parquet(tmp_path / f"{bmu_id}{fi.B1610_SUFFIX}")
    grid = fi.half_hour_grid()
    pn = pl.concat(
        [
            grid.select(bmu_id=pl.lit(b), time=pl.col("time"), fpn_mw=pl.lit(1.0))
            for b in ("E_FULL-1", "E_SHORT-1", "T_BIG-1")
        ]
    )
    monkeypatch.setattr(fi, "BMU_LIST_PATH", csv)
    monkeypatch.setattr(fi, "B1610_DIR", tmp_path)

    members = fi.testbed(pn=pn)

    assert members["elexon_bmu_id"].to_list() == ["E_FULL-1"]


def test_every_arm_is_given_the_same_fixed_column_tuple() -> None:
    columns = [
        "tod", "day_of_week", "price_actual", "price_rank_actual", "price_day_mean_actual",
        "price_day_range_actual", "rank_rule_actual", "persistence_mw",
        "climatology_median_mw", "climatology_filler_mw", "own_fpn_mw", "own_fpn_filler_mw",
        "different_party__mean", "different_party__mean_filler",
    ]  # fmt: skip
    for arms in (fi.DAY_AHEAD_ARMS, fi.GATE_CLOSURE_ARMS):
        for arm in arms.values():
            expressions = fi.arm_expressions(columns=columns, arm=arm)
            assert tuple(expressions) == fi.FEATURE_COLUMNS
    assert len(fi.FEATURE_COLUMNS) == 13


def _issue_frames(monkeypatch: pytest.MonkeyPatch, issue: str):  # noqa: ANN202
    _synthetic_market(monkeypatch)
    grid = fi.half_hour_grid()
    prices = fi.price_sources(grid=grid)
    shuffle = fi.shuffled_time(grid=grid)

    def wave(t: datetime) -> float:
        return (t - START).total_seconds() / 1800.0 % 48 + 100 * ((t - START).days % 3)

    batteries = {
        "E_A-1": _battery(battery_id="E_A-1", party="P1", value_of_time=wave),
        "E_A-2": _battery(battery_id="E_A-2", party="P1", value_of_time=lambda t: 5.0),
        "E_B-1": _battery(battery_id="E_B-1", party="P2", value_of_time=lambda t: -5.0),
    }
    pn = pl.concat(
        [
            _fpn(battery_id="E_A-1", value_of_time=lambda t: 1.0 + _day_index(t)),
            _fpn(battery_id="E_A-2", value_of_time=lambda t: 2.0),
            _fpn(battery_id="E_B-1", value_of_time=lambda t: -3.0 - _day_index(t)),
        ]
    )
    base = fi.build_issue_frame(
        battery=batteries["E_A-1"],
        issue=issue,  # ty: ignore[invalid-argument-type]
        grid=grid,
        prices=prices,
        pn=pn,
        batteries=batteries,
        shuffle=shuffle,
    )
    return base, batteries, pn, grid, prices, shuffle


def test_the_issue_frame_has_one_row_per_half_hour_and_the_persistence_the_issue_allows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, *_ = _issue_frames(monkeypatch, "ID-1h")

    assert base.height == fi.WINDOW_HALF_HOURS
    target = base.filter(pl.col("time") == _utc(10, 5, 12, 0)).row(0, named=True)
    source = base.filter(pl.col("time") == _utc(10, 5, 10, 30)).row(0, named=True)
    assert target["persistence_mw"] == source["output_mw"]


def test_the_day_ahead_persistence_is_the_day_before_only_for_half_hours_ended_by_the_issue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, *_ = _issue_frames(monkeypatch, "DA-late")

    early = base.filter(pl.col("time") == _utc(10, 5, 17, 30)).row(0, named=True)
    late = base.filter(pl.col("time") == _utc(10, 5, 18, 0)).row(0, named=True)
    day_before = base.filter(pl.col("time") == _utc(10, 4, 17, 30)).row(0, named=True)
    two_before = base.filter(pl.col("time") == _utc(10, 3, 18, 0)).row(0, named=True)
    assert early["persistence_mw"] == day_before["output_mw"]
    assert late["persistence_mw"] == two_before["output_mw"]


def test_the_neighbours_of_a_battery_exclude_its_own_lead_party(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, *_ = _issue_frames(monkeypatch, "ID-1h")

    row = base.filter(pl.col("time") == _utc(10, 5, 12, 0)).row(0, named=True)
    # October 5 is day 34. E_A-1's different-party neighbour is E_B-1 alone: (-3 - 34) MW over a
    # p99 of 10 MW.
    assert row["different_party__mean"] == pytest.approx(-3.7)
    # Its same-party neighbour is E_A-2 alone: 2 MW over 10 MW.
    assert row["same_party__mean"] == pytest.approx(0.2)
    # Every other battery: both, averaged.
    assert row["all_testbed__mean"] == pytest.approx((0.2 - 3.7) / 2)


def test_the_filler_is_the_slots_own_value_a_week_earlier(monkeypatch: pytest.MonkeyPatch) -> None:
    base, *_ = _issue_frames(monkeypatch, "ID-1h")

    row = base.filter(pl.col("time") == _utc(10, 5, 12, 0)).row(0, named=True)
    assert row["own_fpn_mw"] == 35.0
    assert row["own_fpn_filler_mw"] == 28.0
    assert row["different_party__mean_filler"] == pytest.approx(-3.0)
    assert base.filter(pl.col("time") < _utc(9, 8))["own_fpn_filler_mw"].null_count() == 7 * 48


def test_arms_put_their_own_source_in_each_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    base, *_ = _issue_frames(monkeypatch, "ID-1h")
    time = _utc(10, 5, 12, 0)

    def row_of(name: str) -> dict[str, float]:
        frame = fi.arm_frame(base=base, arm=fi.GATE_CLOSURE_ARMS[name])
        return frame.filter(pl.col("time") == time).row(0, named=True)

    assert row_of("no_neighbour")["neighbour_mean_slot"] == pytest.approx(-3.0)
    assert row_of("no_neighbour")["own_fpn_slot"] == 28.0
    assert row_of("neighbour_fpn")["neighbour_mean_slot"] == pytest.approx(-3.7)
    assert row_of("own_fpn")["own_fpn_slot"] == 35.0
    assert row_of("own_fpn")["neighbour_mean_slot"] == pytest.approx(-3.0)
    assert row_of("neighbour_fpn_same_party")["neighbour_mean_slot"] == pytest.approx(0.2)
    assert row_of("fleet_fpn")["neighbour_mean_slot"] == pytest.approx((0.2 - 3.7) / 2)


def test_the_shuffled_neighbour_arm_reads_another_days_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, _, _, _, _, shuffle = _issue_frames(monkeypatch, "ID-1h")
    time = _utc(10, 5, 12, 0)
    source = shuffle.filter(pl.col("time") == time)["source_time"][0]

    row = fi.arm_frame(base=base, arm=fi.GATE_CLOSURE_ARMS["neighbour_fpn_shuffled"]).filter(
        pl.col("time") == time
    )

    assert source.date() != time.date()
    assert row["neighbour_mean_slot"][0] == pytest.approx((-3.0 - _day_index(source)) / 10.0)


def test_a_battery_with_no_notification_and_no_lead_party_gets_the_climatology_filler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, batteries, pn, grid, prices, shuffle = _issue_frames(monkeypatch, "ID-1h")
    battery_a = _battery(
        battery_id=fi.NGED_BATTERY_A, party=None, value_of_time=lambda t: 3.0, has_fpn=False
    )

    base = fi.build_issue_frame(
        battery=battery_a,
        issue="ID-1h",
        grid=grid,
        prices=prices,
        pn=pn,
        batteries=batteries,
        shuffle=shuffle,
    )

    assert "own_fpn_mw" not in base.columns
    assert "different_party__mean" not in base.columns
    assert "all_testbed__mean" in base.columns
    arm = fi.arm_frame(base=base, arm=fi.GATE_CLOSURE_ARMS["no_neighbour"])
    later = arm.filter(pl.col("time") >= _utc(10, 1))
    assert later["own_fpn_slot"].to_list() == [3.0] * later.height  # median of the constant series
    assert later["neighbour_mean_slot"].to_list() == [3.0] * later.height


def test_scoring_starts_on_the_first_of_october_and_needs_every_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, *_ = _issue_frames(monkeypatch, "DA-late")
    arms = {k: v for k, v in fi.DAY_AHEAD_ARMS.items() if v.price_source != "model"}

    scored = fi.is_scored(base=base, arms=arms)

    assert not scored.filter(base["time"] < fi.SCORING_START).any()
    assert scored.filter(base["time"] >= _utc(10, 2)).all()
    with_gap = base.with_columns(
        output_mw=pl.when(pl.col("time") == _utc(10, 5, 12))
        .then(None)
        .otherwise(pl.col("output_mw"))
    )
    assert (
        not fi.is_scored(base=with_gap, arms=arms).filter(with_gap["time"] == _utc(10, 5, 12)).any()
    )


def test_one_arms_missing_input_drops_the_row_for_every_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, *_ = _issue_frames(monkeypatch, "DA-late")
    arms = {k: v for k, v in fi.DAY_AHEAD_ARMS.items() if v.price_source != "model"}
    target_time = _utc(10, 6, 12)
    broken = base.with_columns(
        price_shuffled=pl.when(pl.col("time") == target_time)
        .then(None)
        .otherwise(pl.col("price_shuffled"))
    )

    scored = fi.is_scored(base=broken, arms=arms)

    assert not scored.filter(base["time"] == target_time).any()
    assert scored.sum() == scored.len() - base.filter(pl.col("time") < fi.SCORING_START).height - 1


def test_a_testbed_battery_is_scored_on_the_four_core_gate_closure_arms() -> None:
    battery = _battery(battery_id="E_A-1", party="P1", value_of_time=lambda t: 0.0)

    arms = fi.scored_arms(battery=battery, issue="ID-1h", has_model_price=False)

    assert list(arms) == [
        "no_neighbour",
        "own_fpn",
        "neighbour_fpn",
        "neighbour_fpn_shuffled",
    ]


def test_nged_battery_a_is_scored_on_the_fleet_arms_only() -> None:
    battery = _battery(battery_id="A", party=None, value_of_time=lambda t: 0.0, has_fpn=False)

    arms = fi.scored_arms(battery=battery, issue="ID-1h", has_model_price=False)

    assert list(arms) == ["no_neighbour", "fleet_fpn", "fleet_fpn_shuffled"]


def test_the_model_price_arm_joins_the_day_ahead_arms_once_the_price_model_has_run() -> None:
    battery = _battery(battery_id="E_A-1", party="P1", value_of_time=lambda t: 0.0)

    without = fi.scored_arms(battery=battery, issue="DA-early", has_model_price=False)
    with_model = fi.scored_arms(battery=battery, issue="DA-early", has_model_price=True)
    day_late = fi.scored_arms(battery=battery, issue="DA-late", has_model_price=True)

    assert "price_model" not in without
    assert "price_model" in with_model
    assert "price_model" not in day_late
    assert len(without) == 3
