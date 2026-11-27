"""Unit tests of the input columns on synthetic data; the golden tests cover real fixtures."""

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from pvlib import spa
from substation_demand_forecaster.bundle import PUBLICATION_DELAY, Bundle
from substation_demand_forecaster.ens_summary import SUMMARY_COLUMNS
from substation_demand_forecaster.features import (
    B1_COLUMNS,
    CALENDAR_COLUMNS,
    STEP5_COLUMNS,
    STEPS_PER_DAY,
    SUMMARY_STEPS,
    ExpectedRows,
    b1_columns,
    build_expected_rows,
    build_input_matrix,
    calendar_columns,
    grid_index,
    lag_columns,
    lag_indices,
    standardise_input_matrix,
    sun_columns,
)

_LOOKBACK = 576
_WINDOW = _LOOKBACK + 1 + SUMMARY_STEPS
_AT = np.arange(_LOOKBACK + 1, _WINDOW)
_COLD_SNAP = B1_COLUMNS.index("t_cold_snap_days")


def _us(*stamps: datetime) -> np.ndarray:
    return np.array([int(stamp.timestamp() * 1_000_000) for stamp in stamps], dtype=np.int64)


def _b1(*, bundle: Bundle, temperature: np.ndarray, at: np.ndarray = _AT) -> np.ndarray:
    return b1_columns(
        temperature=temperature,
        shortwave=np.full(len(temperature), 100.0),
        clear_sky=np.full(len(temperature), 200.0),
        local_date_codes=np.arange(len(temperature)) // STEPS_PER_DAY,
        at=at,
        bundle=bundle,
    )


def _cold_snap_reference(*, temperature: np.ndarray, threshold: float, cap: float) -> np.ndarray:
    """Days since the trailing 24-hour mean last exceeded the threshold, by a plain loop."""
    out = np.empty(len(temperature))
    last_warm = None
    for p in range(len(temperature)):
        window = temperature[max(0, p - STEPS_PER_DAY + 1) : p + 1]
        finite = window[np.isfinite(window)]
        if (finite.mean() if len(finite) else 0.0) > threshold:
            last_warm = p
        out[p] = cap if last_warm is None else min((p - last_warm) / STEPS_PER_DAY, cap)
    return out


@pytest.mark.parametrize(("days_back", "expected"), [(11.0, 10.0), (9.5, 9.5)])
def test_cold_snap_days_caps_at_ten(bundle: Bundle, days_back: float, expected: float) -> None:
    # One value of -1000 C pulls any 24-hour window holding it below 5 C, so the last warm
    # 24-hour mean ends exactly at the last warm half-hour.
    at = _AT[-1]
    last_warm = at - int(days_back * STEPS_PER_DAY)
    temperature = np.where(np.arange(_WINDOW) <= last_warm, 10.0, -1000.0)
    out = _b1(bundle=bundle, temperature=temperature, at=np.array([at]))
    assert out[0, _COLD_SNAP] == expected


def test_b1_with_a_missing_history_day_is_finite_except_the_daily_extremes(bundle: Bundle) -> None:
    rng = np.random.default_rng(1)
    temperature = 5.0 + 3.0 * rng.standard_normal(_WINDOW)
    # The last whole history day before the origin, on the test's local dates.
    first = _LOOKBACK - STEPS_PER_DAY
    temperature[first:_LOOKBACK] = np.nan
    at = np.arange(first + 12, _LOOKBACK)
    out = _b1(bundle=bundle, temperature=temperature, at=at)
    extremes = [B1_COLUMNS.index("t_min_day"), B1_COLUMNS.index("t_max_day")]
    others = [k for k in range(len(B1_COLUMNS)) if k not in extremes]
    assert np.isfinite(out[:, others]).all()
    assert np.isnan(out[:, extremes]).all()
    # The trailing 3-hour mean over a stretch of no finite value is 0.
    assert (out[:, B1_COLUMNS.index("t_mean_3h")] == 0.0).all()
    # The standardiser maps an all-NaN daily extreme to 0.
    raw = np.zeros((len(at), 60), dtype=np.float32)
    raw[:, 46:] = out
    standardised = standardise_input_matrix(raw=raw, target_index=at, bundle=bundle)
    assert (standardised[:, [46 + k for k in extremes]] == 0.0).all()


@pytest.mark.parametrize("missing_day", [0, 3, 6, 9, 11])
def test_cold_snap_days_with_a_missing_day_matches_a_plain_loop(
    bundle: Bundle, missing_day: int
) -> None:
    hours = np.arange(_WINDOW) / 2.0
    temperature = 4.0 + 3.0 * np.sin(2 * np.pi * hours / 24.0) + np.sin(2 * np.pi * hours / 170.0)
    temperature[missing_day * STEPS_PER_DAY : (missing_day + 1) * STEPS_PER_DAY] = np.nan
    out = _b1(bundle=bundle, temperature=temperature)
    reference = _cold_snap_reference(
        temperature=temperature,
        threshold=bundle.b1.cold_snap_threshold_c,
        cap=bundle.b1.cold_snap_cap_days,
    )
    np.testing.assert_allclose(out[:, _COLD_SNAP], reference[_AT], rtol=0, atol=1e-12)


def test_lag_indices_step_back_whole_days_and_weeks_beyond_the_newest_reading(
    bundle: Bundle,
) -> None:
    origin = 100_000
    target = origin + np.array([40, 60])  # 20 hours and 30 hours ahead
    day, week = lag_indices(target_index=target, origin_index=np.full(2, origin), lags=bundle.lags)
    assert (target - day).tolist() == [96, 144]
    assert (target - week).tolist() == [336, 336]


def test_a_missing_lag_reads_value_zero_flag_one_and_the_largest_age(bundle: Bundle) -> None:
    target = np.array([1000, 1000, 1000])
    lag_index = np.array([904, 904, -5])
    power = np.array([0.75, np.nan, 0.5])
    out = lag_columns(power=power, lag_index=lag_index, target_index=target, which=0, bundle=bundle)
    # (0.75 - 0.5) / 0.25 = 1; 96 half-hours is 48 hours, over the 336-hour scale.
    np.testing.assert_allclose(out[0], [1.0, 0.0, 48.0 / 336.0], rtol=1e-6)
    expected_missing = [0.0, 1.0, bundle.lags.missing_age_code_day]
    np.testing.assert_allclose(out[1], expected_missing, rtol=1e-6)
    np.testing.assert_allclose(out[2], expected_missing, rtol=1e-6)


def _local_hour(columns: np.ndarray) -> np.ndarray:
    sin = columns[:, CALENDAR_COLUMNS.index("sin_hour")]
    cos = columns[:, CALENDAR_COLUMNS.index("cos_hour")]
    return np.round(np.mod(np.arctan2(sin, cos) * 24.0 / (2 * np.pi), 24.0), 6)


def test_calendar_local_hour_follows_the_clock_changes(bundle: Bundle) -> None:
    spring = datetime(2026, 3, 29, tzinfo=UTC)  # clocks go forward at 01:00 UTC
    autumn = datetime(2026, 10, 25, tzinfo=UTC)  # clocks go back at 01:00 UTC
    stamps = _us(
        spring + timedelta(minutes=30),
        spring + timedelta(hours=1),
        autumn + timedelta(minutes=30),
        autumn + timedelta(hours=1),
    )
    hour = _local_hour(calendar_columns(valid_us=stamps, calendar=bundle.calendar))
    assert hour.tolist() == [0.5, 2.0, 1.5, 1.0]


def test_calendar_marks_good_friday_by_the_local_date_of_the_stamp(bundle: Bundle) -> None:
    # 23:00 UTC on 2 April is midnight BST on Good Friday, 3 April 2026.
    stamps = _us(
        datetime(2026, 4, 2, 22, 30, tzinfo=UTC),
        datetime(2026, 4, 2, 23, 0, tzinfo=UTC),
        datetime(2026, 4, 3, 12, 0, tzinfo=UTC),
    )
    columns = calendar_columns(valid_us=stamps, calendar=bundle.calendar)
    assert columns[:, CALENDAR_COLUMNS.index("bank_holiday")].tolist() == [0.0, 1.0, 1.0]
    nearest = columns[:, CALENDAR_COLUMNS.index("nearest_days")] * 7
    np.testing.assert_allclose(nearest, [1.0, 0.0, 0.0], atol=1e-12)


def test_sun_columns_use_pvlib_spa_at_the_half_hour_midpoint() -> None:
    stamps = _us(
        *(datetime(2026, 6, 21, tzinfo=UTC) + timedelta(minutes=30 * k) for k in range(1, 49))
    )
    midpoints = (stamps - 15 * 60 * 1_000_000) / 1e6
    # pvlib's `spa_python` defaults: sea level, 1013.25 mbar, 12 C, delta T of 67 s.
    position = spa.solar_position(midpoints, 52.5, -1.9, 0.0, 1013.25, 12.0, 67.0, 0.5667)
    zenith, azimuth = np.radians(position[0]), np.radians(position[4])
    reference = np.stack([np.cos(zenith), np.sin(azimuth), np.cos(azimuth)], axis=1)
    out = sun_columns(valid_us=stamps, latitude=52.5, longitude=-1.9)
    np.testing.assert_allclose(out, reference, rtol=0, atol=1e-6)
    # At the June solstice the noon sun stands 90 - 52.5 + 23.44 degrees up.
    assert math.isclose(out[:, 0].max(), math.cos(math.radians(52.5 - 23.44)), abs_tol=2e-3)


# --- expected-weather rows ------------------------------------------------------------------

_DAY = datetime(2026, 4, 17, tzinfo=UTC)


def _summary(bundle: Bundle, *, run_days_back: list[int]) -> pl.DataFrame:
    """A summary whose mean temperature is the number of days the run is before `_DAY`."""
    runs = pl.DataFrame({"back": run_days_back}).with_columns(
        nwp_init_time=pl.lit(_DAY) - pl.duration(days=pl.col("back")),
        step=pl.lit(list(range(1, SUMMARY_STEPS + 1))),
    )
    frame = (
        runs.explode("step")
        .join(bundle.primaries.select("time_series_id"), how="cross")
        .with_columns(
            valid_time=pl.col("nwp_init_time")
            + PUBLICATION_DELAY
            + pl.duration(minutes=30 * pl.col("step")),
            **dict.fromkeys(SUMMARY_COLUMNS, pl.lit(0.0)),
        )
        .with_columns(mean_temperature_2m=pl.col("back").cast(pl.Float64))
        .drop("back", "step")
    )
    return frame.with_columns(
        pl.col("nwp_init_time", "valid_time").cast(pl.Datetime("us", "UTC")),
        pl.col("time_series_id").cast(pl.Int32),
        pl.col(SUMMARY_COLUMNS).cast(pl.Float32),
    )


def _row_at(rows: ExpectedRows, *, primary: int, stamp: datetime) -> int:
    index = grid_index(_us(stamp))[0]
    return int(np.flatnonzero((rows.primary_index == primary) & (rows.grid_index == index))[0])


def test_expected_rows_come_from_the_freshest_run_strictly_before(bundle: Bundle) -> None:
    rows = build_expected_rows(
        bundle=bundle, summary=_summary(bundle, run_days_back=list(range(14))), nwp_init_time=_DAY
    )
    origin = _DAY + PUBLICATION_DELAY
    assert len(rows.grid_index) == bundle.primaries.height * 7 * STEPS_PER_DAY
    temperature = 2  # mean_temperature_2m's position in the expected-weather row
    # The run one day back has its native origin at O - 1 day, not strictly before that half-hour.
    for stamp, back in [(origin - timedelta(days=1), 2), (origin - timedelta(days=7.5), 8)]:
        row = _row_at(rows, primary=1, stamp=stamp)
        assert rows.raw[row, temperature] == back
        assert rows.run_init_time[row] == np.datetime64(
            (_DAY - timedelta(days=back)).replace(tzinfo=None), "us"
        )


def test_expected_rows_fall_back_to_an_older_run_then_to_zero(bundle: Bundle) -> None:
    summary = _summary(bundle, run_days_back=[3, 4]).filter(
        ~((pl.col("time_series_id") == 101) & (pl.col("nwp_init_time") == _DAY - timedelta(days=3)))
    )
    rows = build_expected_rows(bundle=bundle, summary=summary, nwp_init_time=_DAY)
    origin = _DAY + PUBLICATION_DELAY
    covered = _row_at(rows, primary=0, stamp=origin - timedelta(days=1))
    assert rows.raw[covered, 2] == 4.0
    uncovered = _row_at(rows, primary=0, stamp=origin - timedelta(days=4))
    assert np.isnan(rows.raw[uncovered]).all()
    assert (rows.standardised[uncovered] == 0).all()
    assert np.isnat(rows.run_init_time[uncovered])
    assert math.isclose(
        float(rows.raw[_row_at(rows, primary=1, stamp=origin - timedelta(days=1)), 2]), 3.0
    )


def test_a_primary_with_no_readings_keeps_its_lag_grid_indices(bundle: Bundle) -> None:
    empty_power = pl.DataFrame(
        schema={"time_series_id": pl.Int32, "time": pl.Datetime("us", "UTC"), "power": pl.Float32}
    )
    matrix = build_input_matrix(
        bundle=bundle,
        summary=_summary(bundle, run_days_back=list(range(14))),
        cleaned_power=empty_power,
        nwp_init_time=_DAY,
    )
    target = matrix.keys["target_grid_index"].to_numpy()
    origin = grid_index(_us(_DAY + PUBLICATION_DELAY))[0]
    day, week = lag_indices(
        target_index=target, origin_index=np.full(len(target), origin), lags=bundle.lags
    )
    columns = list(STEP5_COLUMNS)
    x = matrix.standardised
    assert (x[:, columns.index("day_lag_missing")] == 1).all()
    assert (x[:, columns.index("week_lag_missing")] == 1).all()
    assert (x[:, columns.index("day_lag_grid_index")] == day).all()
    assert (x[:, columns.index("week_lag_grid_index")] == week).all()
