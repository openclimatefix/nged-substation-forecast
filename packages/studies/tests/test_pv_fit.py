from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from studies.pv_fit import (
    FIXED_AZIMUTH_DEG,
    FIXED_TILT_DEG,
    TIGHT_PRIORS,
    PlantPriors,
    fit_plant,
    fit_plant_to_changes,
    lagged_pairs,
)
from studies.pv_physics import PlantParameters, SunAndSky, half_hour_sun_and_sky, plant_power_mw


def _sky(*, days: int = 120) -> SunAndSky:
    start = datetime(2025, 3, 1, 1, tzinfo=UTC)
    times = [start + timedelta(hours=h) for h in range(24 * days)]
    rng = np.random.default_rng(1)
    clearness = np.repeat(rng.uniform(0.2, 1.0, size=days), 24)
    hour = np.array([t.hour for t in times])
    ghi = 850.0 * clearness * np.clip(np.sin(np.pi * (hour - 5.0) / 14.0), 0.0, None)
    hourly = pl.DataFrame({"time": pl.Series(times).dt.replace_time_zone("UTC"), "ghi_w_m2": ghi})
    return half_hour_sun_and_sky(hourly=hourly, latitude=52.0, longitude=-1.0)


@pytest.fixture(scope="module")
def sky() -> SunAndSky:
    return _sky()


def test_a_free_fit_recovers_the_parameters_of_a_noiseless_plant(sky: SunAndSky) -> None:
    truth = PlantParameters(
        tilt_deg=25.0, azimuth_deg=200.0, dc_capacity_mw=70.0, ac_capacity_mw=50.0
    )
    output = plant_power_mw(sky=sky, parameters=truth)

    fit = fit_plant(sky=sky, output_mw=output, orientation="free")

    assert fit is not None
    assert fit.parameters.ac_capacity_mw == pytest.approx(50.0, rel=0.03)
    assert fit.parameters.dc_ac_ratio == pytest.approx(1.4, abs=0.1)
    assert fit.parameters.azimuth_deg == pytest.approx(200.0, abs=6.0)


def test_a_fixed_fit_keeps_the_fixed_orientation(sky: SunAndSky) -> None:
    truth = PlantParameters(
        tilt_deg=FIXED_TILT_DEG,
        azimuth_deg=FIXED_AZIMUTH_DEG,
        dc_capacity_mw=70.0,
        ac_capacity_mw=50.0,
    )

    fit = fit_plant(
        sky=sky, output_mw=plant_power_mw(sky=sky, parameters=truth), orientation="fixed"
    )

    assert fit is not None
    assert (fit.parameters.tilt_deg, fit.parameters.azimuth_deg) == (
        FIXED_TILT_DEG,
        FIXED_AZIMUTH_DEG,
    )
    assert fit.parameters.ac_capacity_mw == pytest.approx(50.0, rel=0.03)


def test_a_tracker_fit_fits_a_tracker_better_than_a_fixed_fit_does(sky: SunAndSky) -> None:
    truth = PlantParameters(
        tilt_deg=0.0, azimuth_deg=180.0, dc_capacity_mw=70.0, ac_capacity_mw=50.0, tracker=True
    )
    output = plant_power_mw(sky=sky, parameters=truth)

    tracker = fit_plant(sky=sky, output_mw=output, orientation="tracker")
    fixed = fit_plant(sky=sky, output_mw=output, orientation="fixed")

    assert tracker is not None
    assert fixed is not None
    assert tracker.parameters.tracker
    assert tracker.loss < 0.1 * fixed.loss


def test_the_usable_mask_keeps_bad_half_hours_out_of_the_fit(sky: SunAndSky) -> None:
    truth = PlantParameters(
        tilt_deg=30.0, azimuth_deg=180.0, dc_capacity_mw=70.0, ac_capacity_mw=50.0
    )
    output = plant_power_mw(sky=sky, parameters=truth)
    corrupted = output.copy()
    corrupted[: len(output) // 2] = 0.0
    usable = np.arange(len(output)) >= len(output) // 2

    fit = fit_plant(sky=sky, output_mw=corrupted, orientation="fixed", usable=usable)

    assert fit is not None
    assert fit.parameters.ac_capacity_mw == pytest.approx(50.0, rel=0.03)


def test_a_fit_needs_output_above_zero(sky: SunAndSky) -> None:
    assert fit_plant(sky=sky, output_mw=np.zeros(len(sky.ghi_w_m2)), orientation="fixed") is None


def test_a_fit_needs_enough_daytime_half_hours(sky: SunAndSky) -> None:
    output = np.full(len(sky.ghi_w_m2), np.nan)
    output[:50] = 1.0

    assert fit_plant(sky=sky, output_mw=output, orientation="fixed") is None


def _aggregate_with_a_plant(sky: SunAndSky) -> tuple[np.ndarray, PlantParameters]:
    truth = PlantParameters(
        tilt_deg=25.0, azimuth_deg=200.0, dc_capacity_mw=70.0, ac_capacity_mw=50.0
    )
    drift = 30.0 * np.sin(2 * np.pi * np.arange(len(sky.ghi_w_m2)) / (48.0 * 9.0))
    return plant_power_mw(sky=sky, parameters=truth) + drift, truth


def test_a_fit_to_changes_recovers_a_plant_beside_a_slow_drift(sky: SunAndSky) -> None:
    aggregate, truth = _aggregate_with_a_plant(sky)

    fit = fit_plant_to_changes(
        sky=sky, output_mw=aggregate, orientation="free", ac_guess_mw=40.0, lag_half_hours=8
    )

    assert fit is not None
    assert fit.parameters.ac_capacity_mw == pytest.approx(truth.ac_capacity_mw, rel=0.1)
    assert fit.parameters.azimuth_deg == pytest.approx(truth.azimuth_deg, abs=12.0)


def test_a_tight_prior_pulls_the_fitted_azimuth_towards_south(sky: SunAndSky) -> None:
    aggregate, _ = _aggregate_with_a_plant(sky)

    free = fit_plant_to_changes(
        sky=sky, output_mw=aggregate, orientation="free", ac_guess_mw=40.0, lag_half_hours=8
    )
    pulled = fit_plant_to_changes(
        sky=sky,
        output_mw=aggregate,
        orientation="free",
        ac_guess_mw=40.0,
        lag_half_hours=8,
        priors=PlantPriors(
            tilt_mean_deg=25.0,
            tilt_sd_deg=1.0,
            azimuth_mean_deg=150.0,
            azimuth_sd_deg=1.0,
            ratio_mean=1.4,
            ratio_sd=0.01,
        ),
    )

    assert free is not None
    assert pulled is not None
    assert abs(pulled.parameters.azimuth_deg - 150.0) < abs(free.parameters.azimuth_deg - 150.0)
    assert pulled.parameters.dc_ac_ratio == pytest.approx(1.4, abs=0.05)


def test_a_fit_to_changes_needs_enough_pairs(sky: SunAndSky) -> None:
    output = np.full(len(sky.ghi_w_m2), np.nan)
    output[:100] = 1.0

    assert (
        fit_plant_to_changes(
            sky=sky,
            output_mw=output,
            orientation="free",
            ac_guess_mw=1.0,
            lag_half_hours=8,
            priors=TIGHT_PRIORS,
        )
        is None
    )


def test_lagged_pairs_skip_pairs_that_span_a_gap_in_the_sky(sky: SunAndSky) -> None:
    lag = 8
    output = np.ones(len(sky.half_hour_end_time))
    everywhere = lagged_pairs(sky=sky, output_mw=output, lag_half_hours=lag)
    keep = np.ones(len(output), dtype=bool)
    keep[2000:2010] = False
    gappy = SunAndSky(**{name: value[keep] for name, value in vars(sky).items()})

    pairs = lagged_pairs(sky=gappy, output_mw=output[keep], lag_half_hours=lag)

    stamps = gappy.half_hour_end_time
    assert (stamps[pairs] - stamps[pairs - lag] == np.timedelta64(30 * lag, "m")).all()
    assert 0 < len(pairs) < len(everywhere)


def test_lagged_pairs_reject_a_lag_below_one(sky: SunAndSky) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        lagged_pairs(sky=sky, output_mw=np.ones(len(sky.half_hour_end_time)), lag_half_hours=0)
