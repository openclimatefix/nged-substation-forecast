import math
import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

_STUDY_DIR = Path(__file__).resolve().parents[3] / "studies" / "nwp_forecast_comparison"
sys.path.insert(0, str(_STUDY_DIR))
sys.path.insert(0, str(_STUDY_DIR.parent / "beam_diffuse_split"))

from build_forecast_inputs import (  # noqa: E402
    EXTRA_LEAD_BUILDS,
    IFS_SINGLE_DAYS,
    KMH_TO_MS,
    _ifs_single_arm_columns,
    _ifs_single_extract,
)
from nwp_forecast_comparison import DomainType, arm_columns  # noqa: E402

_TIME_DTYPE = pl.Datetime("us", "UTC")
_RUN = datetime(2025, 3, 8)


def _archive() -> pl.DataFrame:
    """One site's 8 March 2025 run at hourly leads 0 to 240, each value a known function of lead."""
    leads = list(range(241))
    return pl.DataFrame(
        {
            "site": ["A"] * 241,
            "init_time": [_RUN] * 241,
            "lead_hours": leads,
            "shortwave_radiation": [None] + [10.0 * lead - 500.0 for lead in leads[1:]],
            "temperature_2m": [float(lead) for lead in leads],
            "wind_speed_100m": [3.6 * lead for lead in leads],
            "wind_direction_100m": [90.0] * 241,
            "wind_speed_10m": [1.8 * lead for lead in leads],
        },
        schema_overrides={"lead_hours": pl.Int32},
    )


def _keys(*, times: list[datetime]) -> pl.DataFrame:
    return pl.DataFrame({"site": ["A"] * len(times), "time": times}).with_columns(
        pl.col("time").dt.replace_time_zone("UTC").cast(_TIME_DTYPE)
    )


def test_the_fourth_build_reads_only_the_ifs_single_runs_arms_and_not_day_ten():
    build = EXTRA_LEAD_BUILDS["fourth"]

    assert build.ifs_single_days == IFS_SINGLE_DAYS == (0, 1, 2, 3, 5, 7)
    assert not build.product_day_offsets
    assert not build.ens_mean_days
    assert not build.ens_control_days
    assert not build.gefs_days
    assert not build.gfs_native_days


@pytest.mark.parametrize("domain", ["solar", "wind"])
def test_the_built_columns_are_the_columns_the_fit_reads(domain: DomainType):
    extract = _ifs_single_extract(raw=_archive(), domain=domain, time_dtype=_TIME_DTYPE)
    keys = _keys(times=[datetime(2025, 3, 10, 13)])

    arm = _ifs_single_arm_columns(keys=keys, extract=extract, domain=domain, day=2)

    weather = [
        name
        for name in arm_columns(domain=domain, prefixes=("ifs_single_day2",))
        if name.startswith("ifs_single_day2_")
    ]
    assert arm.columns == ["site", "time", *weather]


def test_a_solar_hour_reads_the_radiation_and_temperature_at_its_lead():
    keys = _keys(times=[datetime(2025, 3, 10, 13), datetime(2025, 3, 11, 0)])
    extract = _ifs_single_extract(raw=_archive(), domain="solar", time_dtype=_TIME_DTYPE)

    arm = _ifs_single_arm_columns(keys=keys, extract=extract, domain="solar", day=2)

    # 13:00 on 10 March is lead 61 of the 8 March run; the hour ending 00:00 on 11 March is the
    # last hour of 10 March, lead 72 of the same run.
    assert arm["ifs_single_day2_ghi"].to_list() == [10.0 * 61 - 500.0, 10.0 * 72 - 500.0]
    assert arm["ifs_single_day2_temp"].to_list() == [61.0, 72.0]


def test_radiation_below_zero_is_clipped_to_zero():
    keys = _keys(times=[datetime(2025, 3, 8, 3)])
    extract = _ifs_single_extract(raw=_archive(), domain="solar", time_dtype=_TIME_DTYPE)

    arm = _ifs_single_arm_columns(keys=keys, extract=extract, domain="solar", day=0)

    # Lead 3 holds -470 W/m2 in the fixture.
    assert arm["ifs_single_day0_ghi"].to_list() == [0.0]


def test_a_wind_hour_reads_the_speeds_in_metres_per_second_and_the_direction_as_sine_and_cosine():
    keys = _keys(times=[datetime(2025, 3, 10, 13)])
    extract = _ifs_single_extract(raw=_archive(), domain="wind", time_dtype=_TIME_DTYPE)

    row = _ifs_single_arm_columns(keys=keys, extract=extract, domain="wind", day=2).row(
        0, named=True
    )

    assert math.isclose(row["ifs_single_day2_speed_100m"], 3.6 * 61 * KMH_TO_MS)
    assert math.isclose(row["ifs_single_day2_speed_10m"], 1.8 * 61 * KMH_TO_MS)
    assert math.isclose(row["ifs_single_day2_sin_100m"], 1.0)
    assert math.isclose(row["ifs_single_day2_cos_100m"], 0.0, abs_tol=1e-12)


def test_a_gap_day_carries_nulls_and_is_not_filled_from_another_run():
    keys = _keys(times=[datetime(2025, 3, 10, 13), datetime(2025, 3, 11, 13)])
    # The archive holds the 8 March run only. Day 2 for 11 March needs the 9 March run.
    extract = _ifs_single_extract(raw=_archive(), domain="solar", time_dtype=_TIME_DTYPE)

    arm = _ifs_single_arm_columns(keys=keys, extract=extract, domain="solar", day=2)

    assert arm.height == 2
    assert arm["ifs_single_day2_ghi"].to_list()[1] is None
    assert arm["ifs_single_day2_temp"].to_list()[1] is None
    assert arm["ifs_single_day2_ghi"].to_list()[0] is not None


def test_day_ten_raises_because_the_runs_end_at_lead_240():
    keys = _keys(times=[datetime(2025, 3, 18, 13)])
    extract = _ifs_single_extract(raw=_archive(), domain="wind", time_dtype=_TIME_DTYPE)

    with pytest.raises(ValueError, match="day 10"):
        _ifs_single_arm_columns(keys=keys, extract=extract, domain="wind", day=10)
