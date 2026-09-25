import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from studies.gfs_native import gfs_leads, window_hours

_STUDY_DIR = Path(__file__).resolve().parents[3] / "studies" / "nwp_forecast_comparison"
sys.path.insert(0, str(_STUDY_DIR))
sys.path.insert(0, str(_STUDY_DIR.parent / "beam_diffuse_split"))

from build_forecast_inputs import (  # noqa: E402
    EXTRA_LEAD_BUILDS,
    GFS_NATIVE_DAYS,
    _gfs_native_direct_arm,
    _gfs_native_step_radiation,
    _gfs_native_weather,
    gfs_native_arm,
)

_INIT = datetime(2025, 3, 8, tzinfo=UTC)


def _extract() -> pl.DataFrame:
    """One site's 8 March 2025 run at hourly leads 0 to 120, each value a known function of lead."""
    leads = list(range(121))
    return pl.DataFrame(
        {
            "site": ["A"] * 121,
            "init_time": [_INIT] * 121,
            "lead_hours": leads,
            "ghi_w_m2": [None] + [10.0 * lead for lead in leads[1:]],
            "temp_c": [float(lead) for lead in leads],
            "speed_100m": [2.0 * lead for lead in leads],
            "direction_100m": [90.0] * 121,
            "speed_10m": [float(lead) for lead in leads],
        },
        schema_overrides={"init_time": pl.Datetime("us", "UTC"), "lead_hours": pl.Int32},
    )


def _keys(*, times: list[datetime]) -> pl.DataFrame:
    return pl.DataFrame({"site": ["A"] * len(times), "time": times}).with_columns(
        pl.col("time").dt.replace_time_zone("UTC").cast(pl.Datetime("us", "UTC"))
    )


def test_the_third_batch_builds_only_the_native_gfs_arms():
    build = EXTRA_LEAD_BUILDS["third"]

    assert build.gfs_native_days == GFS_NATIVE_DAYS == (0, 1, 2, 3, 5, 7, 10, 14)
    assert not build.product_day_offsets
    assert not build.ens_mean_days
    assert not build.ens_control_days
    assert not build.gefs_days


def test_a_solar_hour_reads_the_radiation_of_its_lead_and_the_temperature_at_its_midpoint():
    keys = _keys(times=[datetime(2025, 3, 10, 13), datetime(2025, 3, 11, 0)])

    arm = _gfs_native_direct_arm(keys=keys, extract=_extract(), domain="solar", day=2)

    ghi, temp = "gfs_native_day2_ghi", "gfs_native_day2_temp"
    # 13:00 on 10 March reads lead 61 of the 8 March run. The hour ending 00:00 on 11 March is the
    # last hour of 10 March, so it reads lead 72 of the same run.
    assert arm[ghi].to_list() == [610.0, 720.0]
    assert arm[temp].to_list() == [60.5, 71.5]


def test_a_solar_hour_with_no_run_carries_nulls_and_keeps_its_row():
    keys = _keys(times=[datetime(2025, 3, 20, 13)])

    arm = _gfs_native_direct_arm(keys=keys, extract=_extract(), domain="solar", day=2)

    assert arm.height == 1
    assert arm["gfs_native_day2_ghi"].null_count() == 1


def test_a_wind_hour_reads_the_instantaneous_values_at_its_label():
    keys = _keys(times=[datetime(2025, 3, 10, 13)])

    arm = _gfs_native_direct_arm(keys=keys, extract=_extract(), domain="wind", day=2)

    row = arm.row(0, named=True)
    assert row["gfs_native_day2_speed_100m"] == 2.0 * 61
    assert row["gfs_native_day2_speed_10m"] == 61.0
    assert math.isclose(row["gfs_native_day2_sin_100m"], 1.0)
    assert math.isclose(row["gfs_native_day2_cos_100m"], 0.0, abs_tol=1e-12)


def test_wind_components_become_the_direction_the_wind_blows_from():
    raw = pl.DataFrame(
        {
            "cell": [1, 1],
            "init_time": [_INIT, _INIT],
            "lead_hours": [1, 2],
            "temperature_2m": [5.0, 6.0],
            "wind_u_100m": [3.0, 0.0],  # blowing towards the east: from the west
            "wind_v_100m": [0.0, -4.0],  # blowing towards the south: from the north
            "wind_u_10m": [0.0, 0.0],
            "wind_v_10m": [0.0, 0.0],
        }
    )

    weather = _gfs_native_weather(raw=raw)

    assert weather["direction_100m"].to_list() == [270.0, 0.0]
    assert weather["speed_100m"].to_list() == [3.0, 4.0]


def _served_windows(*, hourly: np.ndarray) -> pl.DataFrame:
    """The since-reset window means GFS would serve for known hourly means, one long frame."""
    leads = gfs_leads()
    window = window_hours(leads=leads)
    values = [hourly[lead - w : lead].mean() for lead, w in zip(leads, window, strict=True)]
    return pl.DataFrame(
        {
            "cell": [1] * len(leads),
            "init_time": [_INIT] * len(leads),
            "lead_hours": leads.astype(np.int32),
            "ghi_raw": values,
        },
        schema_overrides={"init_time": pl.Datetime("us", "UTC")},
    )


def test_the_store_windows_convert_to_hourly_and_three_hourly_means():
    hourly = np.random.default_rng(1).uniform(10.0, 900.0, size=384)

    hourly_long, three_long = _gfs_native_step_radiation(radiation=_served_windows(hourly=hourly))

    by_lead = dict(zip(hourly_long["lead_hours"], hourly_long["ghi_w_m2"], strict=True))
    three_by_lead = dict(zip(three_long["lead_hours"], three_long["ghi_w_m2"], strict=True))
    assert math.isclose(by_lead[5], hourly[4])
    assert math.isclose(by_lead[120], hourly[119])
    assert math.isclose(by_lead[126], hourly[123:126].mean())
    assert math.isclose(three_by_lead[6], hourly[3:6].mean())
    assert math.isclose(three_by_lead[126], hourly[123:126].mean())
    assert 5 not in three_by_lead


def test_a_missing_lead_leaves_nulls_not_nans():
    served = _served_windows(hourly=np.full(384, 100.0)).filter(pl.col("lead_hours") != 8)

    hourly_long, _ = _gfs_native_step_radiation(radiation=served)

    missing = hourly_long.filter(pl.col("ghi_w_m2").is_null())
    assert missing["lead_hours"].to_list() == [8, 9]
    assert hourly_long["ghi_w_m2"].is_nan().sum() == 0


def test_the_arm_is_named_by_prefix_and_day():
    assert gfs_native_arm(day=14) == "gfs_native_day14"
