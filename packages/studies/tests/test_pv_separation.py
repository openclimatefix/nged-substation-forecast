from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from studies.pv_physics import SunAndSky, half_hour_sun_and_sky
from studies.pv_separation import (
    FLEET_ORIENTATIONS,
    baseline_design,
    separate,
    separate_by_differences,
    solar_basis,
)


def _times(*, days: int) -> pl.Series:
    start = datetime(2025, 3, 1, 0, 30, tzinfo=UTC)
    return pl.Series("t", [start + timedelta(minutes=30 * k) for k in range(48 * days)])


def _sky(*, days: int) -> SunAndSky:
    hour_ends = [datetime(2025, 3, 1, 1, tzinfo=UTC) + timedelta(hours=h) for h in range(24 * days)]
    rng = np.random.default_rng(2)
    clearness = np.repeat(rng.uniform(0.1, 1.0, size=days), 24)
    hour = np.array([t.hour for t in hour_ends])
    ghi = 800.0 * clearness * np.clip(np.sin(np.pi * (hour - 5.0) / 14.0), 0.0, None)
    hourly = pl.DataFrame(
        {"time": pl.Series(hour_ends).dt.replace_time_zone("UTC"), "ghi_w_m2": ghi}
    )
    return half_hour_sun_and_sky(hourly=hourly, latitude=52.0, longitude=-1.0)


def test_the_daily_design_has_one_indicator_per_row_in_the_local_half_hour() -> None:
    # 00:30 UTC on 1 June 2025 ends the half-hour 00:00 to 00:30 UTC, which is 01:00 to 01:30 BST:
    # slot 2.
    times = pl.Series("t", [datetime(2025, 6, 2, 0, 30, tzinfo=UTC)])

    design = baseline_design(half_hour_end_time=times, flexibility="daily")

    assert design.shape == (1, 144)
    assert design.sum() == 1.0
    assert int(np.argmax(design[0])) == 2  # 2 June 2025 is a Monday, so the weekday block


def test_weekends_use_their_own_blocks() -> None:
    saturday = pl.Series("t", [datetime(2025, 3, 8, 12, 0, tzinfo=UTC)])
    sunday = pl.Series("t", [datetime(2025, 3, 9, 12, 0, tzinfo=UTC)])

    blocks = [
        int(np.argmax(baseline_design(half_hour_end_time=t, flexibility="daily")[0])) // 48
        for t in (saturday, sunday)
    ]

    assert blocks == [1, 2]


def test_flexibilities_have_the_stated_numbers_of_columns() -> None:
    times = _times(days=3)

    widths = {
        f: baseline_design(half_hour_end_time=times, flexibility=f).shape[1]
        for f in ("daily", "seasonal", "monthly")
    }

    assert widths == {"daily": 144, "seasonal": 144 + 4 * 48, "monthly": 144 + 12 * 48}


def test_the_solar_basis_clips_at_one_megawatt_per_megawatt_of_capacity() -> None:
    sky = _sky(days=4)

    basis = solar_basis(sky=sky, dc_ac_ratio=3.0)

    assert basis.shape == (len(sky.ghi_w_m2), len(FLEET_ORIENTATIONS))
    assert basis.max() == 1.0
    assert basis.min() >= 0.0


def test_separation_recovers_a_known_solar_capacity_and_calendar_baseline() -> None:
    # 25 days from 1 March end before the clocks change, so a demand that repeats every 48
    # half-hours of UTC is a fixed pattern of local half-hours.
    days = 25
    sky = _sky(days=days)
    times = pl.Series(
        "t", pl.Series(sky.half_hour_end_time).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    )
    basis = solar_basis(sky=sky, dc_ac_ratio=1.3)
    design = baseline_design(half_hour_end_time=times, flexibility="daily")
    demand = -(10.0 + 5.0 * np.sin(2 * np.pi * np.arange(len(times)) / 48.0))
    output = 12.0 * basis[:, 1] + demand

    result = separate(output_mw=output, basis=basis, design=design)

    assert result is not None
    assert result.total_capacity_mw == pytest.approx(12.0, rel=0.05)
    np.testing.assert_allclose(result.solar_mw, 12.0 * basis[:, 1], atol=0.6)


def test_a_flat_aggregate_gets_no_solar() -> None:
    sky = _sky(days=40)
    times = pl.Series(
        "t", pl.Series(sky.half_hour_end_time).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    )

    result = separate(
        output_mw=np.full(len(times), -5.0),
        basis=solar_basis(sky=sky, dc_ac_ratio=1.3),
        design=baseline_design(half_hour_end_time=times, flexibility="daily"),
    )

    assert result is not None
    assert result.total_capacity_mw == pytest.approx(0.0, abs=0.05)


def test_solar_weights_are_never_negative_when_output_falls_with_the_sun() -> None:
    sky = _sky(days=40)
    times = pl.Series(
        "t", pl.Series(sky.half_hour_end_time).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    )
    basis = solar_basis(sky=sky, dc_ac_ratio=1.3)

    result = separate(
        output_mw=-8.0 * basis[:, 1],
        basis=basis,
        design=baseline_design(half_hour_end_time=times, flexibility="daily"),
    )

    assert result is not None
    assert (result.capacities_mw >= 0).all()


def test_separation_leaves_out_half_hours_with_missing_output_or_irradiance() -> None:
    sky = _sky(days=30)
    times = pl.Series(
        "t", pl.Series(sky.half_hour_end_time).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    )
    basis = solar_basis(sky=sky, dc_ac_ratio=1.3)
    output = 10.0 * basis[:, 1]
    output[:100] = np.nan
    basis[100:200] = np.nan

    result = separate(
        output_mw=output,
        basis=basis,
        design=baseline_design(half_hour_end_time=times, flexibility="daily"),
    )

    assert result is not None
    assert result.total_capacity_mw == pytest.approx(10.0, rel=0.05)


def test_separation_needs_at_least_a_thousand_usable_half_hours() -> None:
    sky = _sky(days=10)
    times = pl.Series(
        "t", pl.Series(sky.half_hour_end_time).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    )

    result = separate(
        output_mw=np.ones(len(times)),
        basis=solar_basis(sky=sky, dc_ac_ratio=1.3),
        design=baseline_design(half_hour_end_time=times, flexibility="daily"),
    )

    assert result is None


def test_a_larger_ridge_lets_the_solar_curve_absorb_a_daytime_pattern_the_baseline_could_fit() -> (
    None
):
    sky = _sky(days=25)
    times = pl.Series(
        "t", pl.Series(sky.half_hour_end_time).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    )
    basis = solar_basis(sky=sky, dc_ac_ratio=1.3)
    design = baseline_design(half_hour_end_time=times, flexibility="daily")
    clear_day_shape = np.tile(basis[:48, 1], 25)
    output = 10.0 * clear_day_shape  # identical every day, so the daily baseline fits it exactly

    loose = separate(output_mw=output, basis=basis, design=design, ridge=1e-6)
    tight = separate(output_mw=output, basis=basis, design=design, ridge=100.0)

    assert loose is not None
    assert tight is not None
    assert tight.total_capacity_mw > loose.total_capacity_mw + 1.0


def test_difference_separation_recovers_solar_beside_a_slow_drift_the_calendar_cannot_follow() -> (
    None
):
    sky = _sky(days=40)
    times = pl.Series(
        "t", pl.Series(sky.half_hour_end_time).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    )
    basis = solar_basis(sky=sky, dc_ac_ratio=1.3)
    drift = 30.0 * np.sin(2 * np.pi * np.arange(len(times)) / (48.0 * 9.0))  # a nine-day swing
    output = 12.0 * basis[:, 1] + drift

    result = separate_by_differences(
        output_mw=output,
        basis=basis,
        design=baseline_design(half_hour_end_time=times, flexibility="daily"),
    )

    assert result is not None
    assert result.total_capacity_mw == pytest.approx(12.0, rel=0.1)


def test_difference_separation_finds_no_solar_in_a_slow_drift_alone() -> None:
    sky = _sky(days=40)
    times = pl.Series(
        "t", pl.Series(sky.half_hour_end_time).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    )
    drift = 30.0 * np.sin(2 * np.pi * np.arange(len(times)) / (48.0 * 9.0))

    result = separate_by_differences(
        output_mw=drift,
        basis=solar_basis(sky=sky, dc_ac_ratio=1.3),
        design=baseline_design(half_hour_end_time=times, flexibility="daily"),
    )

    assert result is not None
    assert result.total_capacity_mw < 1.0


def test_difference_separation_parts_sum_to_the_output_and_skip_missing_half_hours() -> None:
    sky = _sky(days=30)
    times = pl.Series(
        "t", pl.Series(sky.half_hour_end_time).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    )
    basis = solar_basis(sky=sky, dc_ac_ratio=1.3)
    output = 10.0 * basis[:, 1] - 5.0
    output[:100] = np.nan

    result = separate_by_differences(
        output_mw=output,
        basis=basis,
        design=baseline_design(half_hour_end_time=times, flexibility="daily"),
    )

    assert result is not None
    seen = np.isfinite(output)
    np.testing.assert_allclose(
        (result.solar_mw + result.baseline_mw + result.residual_mw)[seen], output[seen], atol=1e-9
    )
    assert result.total_capacity_mw == pytest.approx(10.0, rel=0.1)


def test_difference_separation_needs_a_thousand_usable_pairs() -> None:
    sky = _sky(days=10)
    times = pl.Series(
        "t", pl.Series(sky.half_hour_end_time).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    )

    assert (
        separate_by_differences(
            output_mw=np.ones(len(times)),
            basis=solar_basis(sky=sky, dc_ac_ratio=1.3),
            design=baseline_design(half_hour_end_time=times, flexibility="daily"),
        )
        is None
    )
