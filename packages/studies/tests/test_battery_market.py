from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from studies.battery_market import FOLD_MONTHS, battery_output, p99_output_mw

from studies import battery_market


def test_battery_output_moves_the_stamp_from_the_period_end_to_the_period_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    end = datetime(2025, 10, 1, 0, 30, tzinfo=UTC)
    pl.DataFrame({"half_hour_end_time": [end], "output_mwh": [1.5]}).write_parquet(
        tmp_path / f"BMU-1{battery_market.B1610_SUFFIX}"
    )
    monkeypatch.setattr(battery_market, "B1610_DIR", tmp_path)

    result = battery_output(bmu_id="BMU-1")

    assert result["time"].to_list() == [datetime(2025, 10, 1, 0, 0, tzinfo=UTC)]
    assert result["output_mw"].to_list() == [3.0]  # half-hour energy doubled to power


def test_p99_is_of_the_absolute_output_so_a_big_import_counts() -> None:
    frame = pl.DataFrame({"output_mw": [-100.0] + [1.0] * 99 + [2.0] * 900})

    assert p99_output_mw(frame=frame) == pytest.approx(2.0)
    assert p99_output_mw(frame=frame.with_columns(output_mw=pl.col("output_mw") * -1)) == (
        pytest.approx(2.0)
    )
    assert p99_output_mw(frame=pl.DataFrame({"output_mw": [-5.0] * 50 + [1.0] * 50})) == 5.0


def test_every_calendar_month_is_in_exactly_one_of_four_three_month_folds() -> None:
    assert sorted(FOLD_MONTHS) == list(range(1, 13))
    for fold in range(4):
        assert sum(1 for f in FOLD_MONTHS.values() if f == fold) == 3


def test_p99_is_the_99th_percentile_not_the_95th() -> None:
    frame = pl.DataFrame({"output_mw": [float(v) for v in range(101)]})

    assert p99_output_mw(frame=frame) == pytest.approx(99.0)


def test_the_study_week_is_the_monday_of_the_week_holding_the_winter_solstice() -> None:
    solstice = datetime(2025, 12, 21, tzinfo=UTC)

    assert battery_market.WEEK_START.weekday() == 0
    assert 0 <= (solstice - battery_market.WEEK_START).days < 7


def test_market_frame_joins_the_three_prices_on_period_starts_within_the_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start = battery_market.WINDOW_START
    n_periods = 365 * 48
    periods = [start + timedelta(minutes=30 * (i - 4)) for i in range(n_periods + 8)]
    hours = sorted({p.replace(minute=0) for p in periods})
    missing_apx = periods[10]
    missing_hour = hours[20]
    for name, frame in {
        "neso_n2ex_day_ahead": pl.DataFrame(
            {"time": hours, "price_gbp_per_mwh": [float(h.timestamp() / 3600) for h in hours]}
        ).filter(pl.col("time") != missing_hour),
        "elexon_system_prices": pl.DataFrame(
            {
                "time": list(reversed(periods)),
                "system_sell_price_gbp_per_mwh": [1.0] * len(periods),
                "system_buy_price_gbp_per_mwh": [2.0] * len(periods),
            }
        ),
        "elexon_mid_apx": pl.DataFrame(
            {"time": periods, "price_gbp_per_mwh": [3.0] * len(periods)}
        ).filter(pl.col("time") != missing_apx),
    }.items():
        (tmp_path / name).mkdir()
        frame.write_parquet(tmp_path / name / f"{name}.parquet")
    monkeypatch.setattr(battery_market, "MARKET_DOWNLOADS_DIR", tmp_path)

    result = battery_market.market_frame()

    assert result["time"].to_list() == periods[4 : 4 + n_periods]  # sorted, [start, start + 365 d)
    assert set(result["system_price_gbp_per_mwh"]) == {1.0}  # the sell price
    assert set(result["system_buy_price_gbp_per_mwh"]) == {2.0}
    for row in result.iter_rows(named=True):
        hour = row["time"].replace(minute=0)
        if hour == missing_hour:
            assert row["day_ahead_gbp_per_mwh"] is None  # the half-hours stay, without a price
        else:
            assert row["day_ahead_gbp_per_mwh"] == hour.timestamp() / 3600  # both half-hours
        assert row["apx_index_gbp_per_mwh"] == (None if row["time"] == missing_apx else 3.0)
