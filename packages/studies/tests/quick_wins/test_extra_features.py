"""Tests for the extra features: each is checked against a value worked out by hand."""

from datetime import UTC, datetime

import patito as pt
import polars as pl
import pytest
from contracts.power_schemas import LIST_OF_TIME_SERIES_TYPES, TimeSeriesMetadata
from extra_features import (
    EXTRA_FEATURE_NAMES,
    add_extra_features,
    haurwitz_clear_sky_ghi,
    wind_power_proxy,
)


def _metadata(*, latitude: float, longitude: float) -> pt.DataFrame[TimeSeriesMetadata]:
    return TimeSeriesMetadata.validate(
        pl.DataFrame(
            {
                "time_series_id": pl.Series([1], dtype=pl.Int32),
                "time_series_name": ["S"],
                "time_series_type": pl.Series(["PV"]).cast(pl.Enum(LIST_OF_TIME_SERIES_TYPES)),
                "units": pl.Series(["MW"]).cast(pl.Enum(["MW", "MVA"])),
                "licence_area": pl.Series(["EMids"]).cast(pl.Enum(["EMids"])),
                "substation_number": pl.Series([1], dtype=pl.Int32),
                "substation_type": pl.Series(["Primary"]).cast(
                    pl.Enum(["BSP", "EHV Customer", "GSP", "HV Customer", "Primary"])
                ),
                "latitude": pl.Series([latitude], dtype=pl.Float32),
                "longitude": pl.Series([longitude], dtype=pl.Float32),
                "h3_res_5": pl.Series([617700169958293503], dtype=pl.UInt64),
            }
        )
    )


def _frame(*, times: list[datetime], ghi: float, temperature: float, speed: float) -> pl.DataFrame:
    n = len(times)
    return pl.DataFrame(
        {
            "time_series_id": pl.Series([1] * n, dtype=pl.Int32),
            "valid_time": pl.Series(times, dtype=pl.Datetime("us", "UTC")),
            "downward_short_wave_radiation_flux_surface": [ghi] * n,
            "temperature_2m": [temperature] * n,
            "wind_speed_10m": [speed] * n,
            "wind_direction_10m": [270.0] * n,
            "wind_speed_100m": [speed] * n,
            "wind_direction_100m": [270.0] * n,
        }
    )


def test_haurwitz_matches_its_formula_at_the_zenith_and_is_zero_below_the_horizon():
    """At cos z = 1 the Haurwitz value is 1098 exp(-0.057); at and below the horizon it is 0."""
    values = pl.DataFrame({"c": [1.0, 0.0, -0.3]}).select(
        out=haurwitz_clear_sky_ghi(cos_zenith_values=pl.col("c"))
    )["out"]
    assert values[0] == pytest.approx(1098.0 * 0.944594, rel=1e-4)
    assert values[1] == 0.0
    assert values[2] == 0.0


def test_wind_power_proxy_has_dead_zones_and_a_flat_top():
    """Zero below cut-in, a cube up to rated, 1 to cut-out, zero in a storm."""
    speeds = pl.DataFrame({"v": [2.0, 3.0, 12.5, 20.0, 25.0, 26.0]}).select(
        out=wind_power_proxy(speed=pl.col("v"))
    )["out"]
    assert speeds.to_list() == pytest.approx([0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    mid = pl.DataFrame({"v": [8.0]}).select(out=wind_power_proxy(speed=pl.col("v")))["out"][0]
    assert mid == pytest.approx((8.0**3 - 27.0) / (12.5**3 - 27.0))


def test_columns_are_added_in_order_and_wind_components_follow_the_met_convention():
    """A westerly (270 degrees) of 10 m/s blows towards the east: u = +10, v ~ 0."""
    frame = _frame(
        times=[datetime(2025, 6, 21, 12, 0, tzinfo=UTC)], ghi=800.0, temperature=20.0, speed=10.0
    )
    out = add_extra_features(frame=frame, metadata=_metadata(latitude=53.0, longitude=0.0))
    assert out.columns == [*frame.columns, *EXTRA_FEATURE_NAMES]
    assert out["wind_u_100m"][0] == pytest.approx(10.0, abs=1e-6)
    assert out["wind_v_100m"][0] == pytest.approx(0.0, abs=1e-6)


def test_clear_sky_index_is_null_at_night_and_a_ratio_by_day():
    """Midnight has no index. Noon at midsummer gives measured over Haurwitz clear-sky."""
    frame = _frame(
        times=[datetime(2025, 6, 21, 0, 0, tzinfo=UTC), datetime(2025, 6, 21, 12, 0, tzinfo=UTC)],
        ghi=400.0,
        temperature=15.0,
        speed=5.0,
    )
    out = add_extra_features(frame=frame, metadata=_metadata(latitude=53.0, longitude=0.0))
    assert out["clear_sky_index"][0] is None
    assert out["cos_zenith"][1] > 0.85
    assert out["clear_sky_ghi_window_mean"][1] > 700.0
    assert out["clear_sky_index"][1] == pytest.approx(
        400.0 / out["clear_sky_ghi_window_mean"][1], rel=1e-6
    )


def test_heating_degrees_c_floor_at_zero_and_count_below_the_base():
    """5 C is 10.5 degrees below the 15.5 C base; 20 C is above it."""
    cold = _frame(times=[datetime(2025, 1, 10, 12, tzinfo=UTC)], ghi=0.0, temperature=5.0, speed=1)
    warm = _frame(times=[datetime(2025, 7, 10, 12, tzinfo=UTC)], ghi=0.0, temperature=20.0, speed=1)
    metadata = _metadata(latitude=53.0, longitude=0.0)
    assert add_extra_features(frame=cold, metadata=metadata)["heating_degrees_c"][0] == (
        pytest.approx(10.5)
    )
    assert add_extra_features(frame=warm, metadata=metadata)["heating_degrees_c"][0] == 0.0


def test_holiday_columns_use_the_local_date_and_the_nearest_holiday_gap():
    """Easter Monday 2025 is a holiday; the Saturday before it is 2 days from Good Friday's gap."""
    times = [
        datetime(2025, 4, 21, 12, tzinfo=UTC),  # Easter Monday
        datetime(2025, 4, 19, 12, tzinfo=UTC),  # the Saturday: 1 day after Good Friday
        datetime(2025, 6, 10, 12, tzinfo=UTC),  # a Tuesday in June
    ]
    out = add_extra_features(
        frame=_frame(times=times, ghi=0.0, temperature=10.0, speed=1),
        metadata=_metadata(latitude=53.0, longitude=0.0),
    )
    assert out["is_bank_holiday"].to_list() == [1, 0, 0]
    assert out["days_to_nearest_bank_holiday"].to_list() == [0, 1, 15]


def test_late_evening_utc_in_summer_belongs_to_the_next_local_day_for_calendar_columns():
    """23:30 UTC on 20 April 2025 is 00:30 on Easter Monday, 21 April, in London."""
    out = add_extra_features(
        frame=_frame(
            times=[datetime(2025, 4, 20, 23, 30, tzinfo=UTC)], ghi=0, temperature=10, speed=1
        ),
        metadata=_metadata(latitude=53.0, longitude=0.0),
    )
    assert out["is_bank_holiday"][0] == 1  # 00:30 on Easter Monday, 21 April, in London
    assert out["half_hour_of_day"][0] == 1
    assert out["day_of_year"][0] == 111


def test_a_null_wind_speed_gives_a_null_power_proxy_not_zero():
    """A missing speed must reach the model as missing, not as 'no power'."""
    out = pl.DataFrame({"v": [None, 8.0]}, schema={"v": pl.Float64}).select(
        out=wind_power_proxy(speed=pl.col("v"))
    )["out"]
    assert out[0] is None
    assert out[1] is not None


def test_a_northerly_blows_towards_the_south_and_a_null_radiation_gives_a_null_index():
    """From 0 degrees: u = 0, v = -speed. A null irradiance by day gives a null index."""
    frame = _frame(
        times=[datetime(2025, 6, 21, 12, 0, tzinfo=UTC)], ghi=300.0, temperature=20.0, speed=10.0
    ).with_columns(
        wind_direction_100m=pl.lit(0.0),
        downward_short_wave_radiation_flux_surface=pl.lit(None, dtype=pl.Float64),
    )
    out = add_extra_features(frame=frame, metadata=_metadata(latitude=53.0, longitude=0.0))
    assert out["wind_u_100m"][0] == pytest.approx(0.0, abs=1e-6)
    assert out["wind_v_100m"][0] == pytest.approx(-10.0, abs=1e-6)
    assert out["clear_sky_index"][0] is None


def test_repeated_valid_times_keep_their_rows_and_order():
    """Five ensemble members at one valid time stay five rows, in order, with equal sun columns."""
    time = datetime(2025, 6, 21, 12, 0, tzinfo=UTC)
    frame = _frame(times=[time] * 5, ghi=500.0, temperature=10.0, speed=5.0).with_columns(
        ensemble_member=pl.Series(range(5), dtype=pl.Int8)
    )
    out = add_extra_features(frame=frame, metadata=_metadata(latitude=53.0, longitude=0.0))
    assert out["ensemble_member"].to_list() == [0, 1, 2, 3, 4]
    assert out["cos_zenith"].n_unique() == 1
