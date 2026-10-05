from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from studies import pv_dataset, wind_product_frames


def _serve_half_hours(*, monkeypatch: pytest.MonkeyPatch) -> pl.DataFrame:
    """Serve one generator's four half-hours from 00:30, and return its roster."""
    start = datetime(2025, 5, 1, tzinfo=UTC)
    power = pl.DataFrame(
        {
            "time_series_id": [1] * 4,
            "time": [start + timedelta(minutes=30 * step) for step in range(1, 5)],
            "power": [2.0, 4.0, 6.0, 8.0],
        }
    )
    monkeypatch.setattr(pl, "scan_delta", lambda uri, **_: power.lazy())
    return pl.DataFrame({"time_series_id": [1], "site": ["W1"]})


def test_wind_hours_are_centred_where_solar_hours_end_at_their_label(
    monkeypatch: pytest.MonkeyPatch,
):
    # Swapping the two hourly-power functions is a silent error: both return the same columns, and
    # every stamp differs by 30 minutes.
    roster = _serve_half_hours(monkeypatch=monkeypatch)
    start = datetime(2025, 5, 1, tzinfo=UTC)

    wind = wind_product_frames.wind_hourly_power(sites=roster)
    solar = pv_dataset.solar_hourly_power(sites=roster)

    assert wind.select("time", "power_mw").rows() == [(start + timedelta(hours=1), 5.0)]
    assert solar.select("time", "power_mw").rows() == [
        (start + timedelta(hours=1), 3.0),
        (start + timedelta(hours=2), 7.0),
    ]


def test_hour_ending_wind_power_matches_the_solar_convention(monkeypatch: pytest.MonkeyPatch):
    roster = _serve_half_hours(monkeypatch=monkeypatch)

    wind = wind_product_frames.wind_hourly_power(sites=roster, centred=False)
    solar = pv_dataset.solar_hourly_power(sites=roster)

    assert wind.equals(solar)


def test_wind_columns_hold_speed_direction_and_ten_metre_speed():
    assert wind_product_frames.wind_columns(product="ukv") == (
        "speed_hub_ukv",
        "direction_sin_ukv",
        "direction_cos_ukv",
        "speed_10m_ukv",
    )


@pytest.mark.parametrize(
    ("product", "height_m"),
    [("icon_d2", 80), ("icon_eu", 80), ("icon_global", 80), ("ukv", 100), ("era5", 100)],
)
def test_icon_products_are_read_at_80_m_and_the_others_at_100_m(product: str, height_m: int):
    assert wind_product_frames.hub_height_m(product=product) == height_m


def test_common_rows_drops_zero_hours_and_the_upgrade_tail_and_adds_the_fit_columns():
    upgrade = wind_product_frames.UPGRADE_DAY
    times = [
        datetime(2025, 5, 1, 12, tzinfo=UTC),
        datetime(2025, 5, 1, 13, tzinfo=UTC),
        upgrade,
        upgrade + timedelta(days=2),
        datetime(2026, 2, 1, tzinfo=UTC),
    ]
    frame = pl.DataFrame(
        {
            "time": times,
            "has_zero_half_hour": [False, True, False, False, False],
            "site": ["W1"] * 5,
        }
    )

    dropped = wind_product_frames.common_rows(frame=frame)
    kept = wind_product_frames.common_rows(frame=frame, drop_zero_hours=False)

    assert dropped["time"].to_list() == [times[0], times[4]]
    assert kept["time"].to_list() == [times[0], times[1], times[4]]
    assert dropped["constrained"].to_list() == [False, False]
    assert dropped["cap_mw"].null_count() == 2
