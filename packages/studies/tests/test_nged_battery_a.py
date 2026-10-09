"""Tests of the loaders for NGED battery A, on synthetic stand-ins for the private data."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from studies.battery_market import WINDOW_START
from studies.nged_battery_a import (
    battery_a_frame,
    battery_a_raw,
    battery_a_units,
    market_full,
    nearest_cams_hourly,
    window_filter,
)

from studies import nged_battery_a

DAY = datetime(2025, 9, 1, tzinfo=UTC)


def _write_metadata(*, directory: Path, names: list[str]) -> None:
    (directory / "NGED").mkdir()
    pl.DataFrame(
        {
            "time_series_id": list(range(len(names))),
            "time_series_name": names,
            "latitude": [50.0] * len(names),
            "longitude": [0.0] * len(names),
            "units": ["MVA"] * len(names),
        }
    ).write_parquet(directory / "NGED" / "metadata.parquet")


def test_the_series_is_found_by_a_name_marking_it_as_a_battery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_metadata(directory=tmp_path, names=["Solar Farm", "Some BESS Site", "Wind"])
    monkeypatch.setattr(nged_battery_a, "REPO_DATA_DIR", tmp_path)

    assert battery_a_units() == "MVA"


@pytest.mark.parametrize("names", [["Solar", "Wind"], ["Battery One", "Storage Two", "Wind"]])
def test_a_search_that_matches_no_series_or_two_series_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, names: list[str]
) -> None:
    _write_metadata(directory=tmp_path, names=names)
    monkeypatch.setattr(nged_battery_a, "REPO_DATA_DIR", tmp_path)

    with pytest.raises(ValueError, match="not 1"):
        battery_a_units()


def test_the_raw_series_is_sorted_and_moved_from_period_ends_to_period_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_metadata(directory=tmp_path, names=["Solar", "My battery"])  # the battery has id 1
    monkeypatch.setattr(nged_battery_a, "REPO_DATA_DIR", tmp_path)
    ends = [DAY + timedelta(minutes=30 * i) for i in (3, 1, 2)]
    power = pl.LazyFrame(
        {
            "time_series_id": [1, 1, 1, 0],
            "time": [*ends, DAY],
            "power": [3.0, 1.0, 2.0, 99.0],
        },
        schema_overrides={"time": pl.Datetime("ns", "UTC")},
    )
    monkeypatch.setattr(nged_battery_a, "scan_power", lambda: power)

    result = battery_a_raw()

    assert result["time"].to_list() == [DAY + timedelta(minutes=30 * i) for i in (0, 1, 2)]
    assert result["power"].to_list() == [1.0, 2.0, 3.0]
    assert result["power"].dtype == pl.Float64


def test_the_window_is_the_365_days_from_the_window_start_with_the_end_excluded() -> None:
    end = WINDOW_START + timedelta(days=365)
    half_hour = timedelta(minutes=30)
    times = [WINDOW_START - half_hour, WINDOW_START, end - half_hour, end]

    kept = pl.DataFrame({"time": times}).filter(window_filter())["time"].to_list()

    assert kept == times[1:3]


def _write_market(*, directory: Path, hours: list[datetime], periods: list[datetime]) -> None:
    for name, frame in {
        "neso_n2ex_day_ahead": pl.DataFrame(
            {"time": hours, "price_gbp_per_mwh": [float(i) for i in range(len(hours))]}
        ),
        "elexon_system_prices": pl.DataFrame(
            {
                "time": list(reversed(periods)),
                "system_sell_price_gbp_per_mwh": [10.0] * len(periods),
                "system_buy_price_gbp_per_mwh": [20.0] * len(periods),
            }
        ),
    }.items():
        (directory / name).mkdir()
        frame.write_parquet(directory / name / f"{name}.parquet")


def test_the_full_market_has_the_sell_price_and_gives_both_half_hours_their_hours_price(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    periods = [DAY + timedelta(minutes=30 * i) for i in range(6)]
    _write_market(directory=tmp_path, hours=[DAY, DAY + timedelta(hours=2)], periods=periods)
    monkeypatch.setattr(nged_battery_a, "MARKET_DOWNLOADS_DIR", tmp_path)

    result = market_full()

    assert result["time"].to_list() == periods  # sorted
    assert result["system_price_gbp_per_mwh"].to_list() == [10.0] * 6
    # The hour from 01:00 has no auction price, and its half-hours stay with a null.
    assert result["day_ahead_gbp_per_mwh"].to_list() == [0.0, 0.0, None, None, 1.0, 1.0]
    assert result.columns == ["time", "system_price_gbp_per_mwh", "day_ahead_gbp_per_mwh"]


def _two_days(*, power_of) -> tuple[pl.DataFrame, pl.DataFrame]:  # noqa: ANN001
    """Raw power and the market over 2025-09-01 and 2025-09-02, plus one half-hour before them."""
    times = [DAY - timedelta(minutes=30) + timedelta(minutes=30 * i) for i in range(97)]
    raw = pl.DataFrame(
        {"time": times, "power": [power_of(t) for t in times]},
        schema_overrides={"time": pl.Datetime("us", "UTC")},
    )
    market = pl.DataFrame(
        {
            "time": times,
            "system_price_gbp_per_mwh": [1.0] * len(times),
            "day_ahead_gbp_per_mwh": [
                float((7 * (t.hour * 2 + t.minute // 30)) % 48) for t in times
            ],
        },
        schema_overrides={"time": pl.Datetime("us", "UTC")},
    )
    return raw, market


def test_the_battery_frame_scales_by_the_p99_and_ranks_prices_within_each_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def power_of(time: datetime) -> float:
        return 1000.0 if time < DAY else float(time.hour * 2 + time.minute // 30 - 24)

    raw, market = _two_days(power_of=power_of)
    # The 03:00 half-hour has no price (a null), and the 04:00 half-hour is not in the market.
    market = market.with_columns(
        day_ahead_gbp_per_mwh=pl.when(pl.col("time") == DAY + timedelta(hours=3))
        .then(None)
        .otherwise(pl.col("day_ahead_gbp_per_mwh"))
    ).filter(pl.col("time") != DAY + timedelta(hours=4))
    monkeypatch.setattr(nged_battery_a, "battery_a_raw", lambda: raw)
    monkeypatch.setattr(nged_battery_a, "market_full", lambda: market)

    frame, scale = battery_a_frame(sign=-1.0)

    window = raw.filter(window_filter())
    assert scale == pytest.approx(np.quantile(window["power"].abs().to_numpy(), 0.99))
    assert scale < 25.0  # the 1000 MW row before the window is outside it
    assert frame.columns == [
        "time",
        "output_mw",
        "output_mwh",
        "system_price_gbp_per_mwh",
        "day_ahead_gbp_per_mwh",
        "date",
        "tod",
        "month",
        "rank_pct",
        "fold",
        "tercile",
    ]
    assert frame.height == 96 - 2
    assert frame["time"].to_list() == sorted(frame["time"].to_list())
    expected_power = -window.filter(
        ~pl.col("time").is_in([DAY + timedelta(hours=3), DAY + timedelta(hours=4)])
    )["power"]
    assert frame["output_mw"].to_list() == pytest.approx((expected_power / scale).to_list())
    assert frame["output_mwh"].to_list() == pytest.approx((expected_power / scale / 2).to_list())
    first_day = frame.filter(pl.col("date") == DAY.date()).sort("time")
    assert first_day["tod"].to_list() == [i for i in range(48) if i not in (6, 8)]
    assert set(frame["month"]) == {"2025-09"}
    assert set(frame["fold"]) == {0}
    # The 46 priced half-hours of a date take ranks 1 to 46 with ties averaged, so no rank is a
    # rank across both dates; the price of tod 0 is the lowest of its date.
    prices = first_day["day_ahead_gbp_per_mwh"]
    ranks = prices.rank("average")
    assert first_day["rank_pct"].to_list() == pytest.approx(((ranks - 0.5) / 48).to_list())
    assert first_day["tercile"].to_list() == [int(r * 3) for r in first_day["rank_pct"].to_list()]
    assert first_day["tercile"].max() == 2


def test_nearest_point_weights_longitude_by_0_6_and_ignores_non_public_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_metadata(directory=tmp_path, names=["Some battery"])  # at 50.0 N, 0.0 E
    monkeypatch.setattr(nged_battery_a, "REPO_DATA_DIR", tmp_path)
    points = {
        "gb_north": (50.5, 0.0),  # squared distance 0.25
        "gb_east": (50.0, 0.8),  # squared distance 0.2304 once longitude is weighted by 0.6
        "other_here": (50.0, 0.0),  # exactly at the series, but not a public point
    }
    hours = [DAY + timedelta(hours=h) for h in (2, 0, 1)]
    rows = [
        {
            "point_id": point,
            "latitude": lat,
            "longitude": lon,
            "time": hour,
            "ghi_w_m2": 100.0 * i + (5.0 if point == "gb_east" else 0.0),
            "clear_sky_ghi_w_m2": 200.0 + i,
        }
        for point, (lat, lon) in points.items()
        for i, hour in enumerate(hours)
    ]
    path = tmp_path / "cams.parquet"
    pl.DataFrame(rows).write_parquet(path)
    monkeypatch.setattr(nged_battery_a, "CAMS_PUBLIC_POINTS_PATH", path)

    result = nearest_cams_hourly()

    assert result.columns == ["hour_start", "ghi_w_m2", "clear_sky_ghi_w_m2"]
    assert result["hour_start"].to_list() == [
        DAY - timedelta(hours=1),
        DAY,
        DAY + timedelta(hours=1),
    ]  # sorted, and moved from the hour's end to its start
    assert result["ghi_w_m2"].to_list() == [105.0, 205.0, 5.0]
