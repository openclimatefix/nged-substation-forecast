from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from studies.arm_runner import SHARED_FEATURES
from studies.product_frames import SOLAR, WIND

from studies import ens_members

RUN = datetime(2025, 5, 1, tzinfo=UTC)


def test_each_domain_names_its_weather_fields_and_its_arms_prefix_them():
    assert ens_members.fields(domain="solar") == ("ghi", "temp")
    assert ens_members.ens_columns(arm="ens_mean", domain="solar") == (
        "ens_mean_ghi",
        "ens_mean_temp",
    )
    assert len(ens_members.fields(domain="wind")) == 4


def test_prefixed_renames_the_fields_to_the_arms_columns_and_keeps_site_and_time():
    frame = pl.DataFrame({"site": ["A"], "time": [RUN], "ghi": [1.0], "temp": [2.0]})

    result = ens_members.prefixed(frame=frame, arm="ens_mean", domain="solar")

    assert result.columns == ["site", "time", "ens_mean_ghi", "ens_mean_temp"]
    assert result.row(0)[2:] == (1.0, 2.0)


def _members(*, rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        [{"site": "A", "time": RUN + timedelta(hours=1), "init_time": RUN, **row} for row in rows]
    )


def test_the_control_way_keeps_only_the_control_members_values():
    hourly = _members(
        rows=[
            {"member": ens_members.CONTROL_MEMBER, "ghi": 10.0, "temp": 5.0},
            {"member": 1, "ghi": 30.0, "temp": 7.0},
        ]
    )

    result = ens_members.reduce_members(
        hourly=hourly, domain="solar", way="control", ensemble_size=2
    )

    assert result.to_dicts() == [
        {"site": "A", "time": RUN + timedelta(hours=1), "ghi": 10.0, "temp": 5.0}
    ]


def test_the_solar_mean_averages_the_members():
    hourly = _members(
        rows=[
            {"member": 0, "ghi": 10.0, "temp": 5.0},
            {"member": 1, "ghi": 10.0, "temp": 5.0},
            {"member": 2, "ghi": 70.0, "temp": 11.0},
        ]
    )

    result = ens_members.reduce_members(hourly=hourly, domain="solar", way="mean", ensemble_size=3)

    assert result.select("ghi", "temp").row(0) == (30.0, 7.0)


def test_the_wind_mean_direction_is_the_mean_wind_vectors_not_the_mean_of_directions():
    hourly = _members(
        rows=[
            {"member": 0, "speed_100m": 3.0, "sin_100m": 1.0, "cos_100m": 0.0, "speed_10m": 1.0},
            {"member": 1, "speed_100m": 4.0, "sin_100m": 0.0, "cos_100m": 1.0, "speed_10m": 2.0},
        ]
    )

    result = ens_members.reduce_members(hourly=hourly, domain="wind", way="mean", ensemble_size=2)

    row = result.row(0, named=True)
    assert row["speed_100m"] == pytest.approx(3.5)
    assert row["speed_10m"] == pytest.approx(1.5)
    assert (row["sin_100m"], row["cos_100m"]) == pytest.approx((0.6, 0.8))


def test_a_run_missing_a_member_is_refused():
    hourly = _members(rows=[{"member": 0, "ghi": 10.0, "temp": 5.0}])

    with pytest.raises(ValueError, match="member"):
        ens_members.reduce_members(hourly=hourly, domain="solar", way="mean", ensemble_size=2)


def test_the_solar_arms_share_the_past_weather_columns_without_era5_temperature_plus_the_era():
    features = ens_members.shared_features(domain=SOLAR)

    assert "temp_c" not in features
    assert features == (*(c for c in SHARED_FEATURES if c != "temp_c"), "era_code")
    assert ens_members.shared_features(domain=WIND) == WIND.shared_features


# `long_frame`'s `explode` raises Polars' `empty_as_null` deprecation warning, which the suite turns
# into an error. The moved code is unchanged, so the test tolerates the warning.
@pytest.mark.filterwarnings("ignore:In Polars 2.0:DeprecationWarning")
def test_long_frame_keys_each_value_by_site_run_time_and_member():
    keys = pl.DataFrame(
        {"site": ["A", "A"], "init_time": [RUN.replace(tzinfo=None)] * 2, "ensemble_member": [0, 1]}
    )
    steps = ens_members.Steps(
        keys=keys,
        leads=np.array([1, 2]),
        widths=np.array([1, 1]),
        values={},
        ensemble_size=2,
    )

    result = ens_members.long_frame(
        steps=steps,
        targets=np.array([1, 2]),
        values={"ghi": np.array([[10.0, 11.0], [20.0, 21.0]])},
    )

    assert result.select("member", "time", "ghi").rows() == [
        (0, RUN + timedelta(hours=1), 10.0),
        (0, RUN + timedelta(hours=2), 11.0),
        (1, RUN + timedelta(hours=1), 20.0),
        (1, RUN + timedelta(hours=2), 21.0),
    ]


def test_clear_sky_arrays_average_the_table_over_each_step_and_read_it_at_each_target_hour():
    run = datetime(2025, 5, 1)
    keys = pl.DataFrame({"site": ["A", "A"], "init_time": [run, run], "ensemble_member": [0, 1]})
    steps = ens_members.Steps(
        keys=keys,
        leads=np.array([3, 6]),
        widths=np.array([3, 3]),
        values={},
        ensemble_size=2,
    )
    # Each hour's clear-sky value is its lead, so a three-hour step's mean is its middle hour.
    clear_sky = pl.DataFrame(
        {
            "site": ["A"] * 6,
            "time": [run + timedelta(hours=hour) for hour in range(1, 7)],
            "clear_sky_w_m2": [float(hour) for hour in range(1, 7)],
        }
    )

    step_clear_sky, target_clear_sky = ens_members.clear_sky_arrays(
        steps=steps, targets=np.array([2, 5]), clear_sky=clear_sky
    )

    np.testing.assert_allclose(step_clear_sky, [[2.0, 5.0], [2.0, 5.0]])
    np.testing.assert_allclose(target_clear_sky, [[2.0, 5.0], [2.0, 5.0]])


def test_clear_sky_arrays_refuse_a_run_whose_hours_are_missing_from_the_table():
    run = datetime(2025, 5, 1)
    steps = ens_members.Steps(
        keys=pl.DataFrame({"site": ["A"], "init_time": [run], "ensemble_member": [0]}),
        leads=np.array([3]),
        widths=np.array([3]),
        values={},
        ensemble_size=1,
    )
    clear_sky = pl.DataFrame(
        {
            "site": ["A"],
            "time": [run + timedelta(hours=1)],
            "clear_sky_w_m2": [1.0],
        }
    )

    with pytest.raises(ValueError, match="clear-sky"):
        ens_members.clear_sky_arrays(steps=steps, targets=np.array([2]), clear_sky=clear_sky)
