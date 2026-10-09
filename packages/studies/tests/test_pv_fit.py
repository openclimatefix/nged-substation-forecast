from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from studies.pv_fit import (
    FIXED_AZIMUTH_DEG,
    FIXED_TILT_DEG,
    TIGHT_PRIORS,
    PlantPriors,
    _best_start,
    _bounds,
    _prior_residuals,
    _smooth_min,
    _starts,
    change_variance_explained,
    fit_plant,
    fit_plant_to_changes,
    fit_plant_to_changes_with_regressors,
    lagged_pairs,
)
from studies.pv_physics import (
    PlantParameters,
    SunAndSky,
    daylight,
    half_hour_sun_and_sky,
    plant_power_mw,
)


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


def _daily_pattern(sky: SunAndSky, *, seed: int) -> np.ndarray:
    """A regressor that swings within each day, with a random level on each day."""
    rng = np.random.default_rng(seed)
    n = len(sky.ghi_w_m2)
    day = np.arange(n) // 48
    phase = rng.uniform(0.0, 2 * np.pi, size=day.max() + 1)[day]
    return np.sin(2 * np.pi * (np.arange(n) % 48) / 48.0 + phase) + rng.normal(0.0, 0.2, size=n)


@pytest.mark.parametrize("coefficient", [12.0, -12.0])
def test_a_fit_with_a_regressor_recovers_the_plant_and_the_regressors_coefficient(
    sky: SunAndSky, coefficient: float
) -> None:
    aggregate, truth = _aggregate_with_a_plant(sky)
    regressor = _daily_pattern(sky, seed=3)
    aggregate = aggregate + coefficient * regressor

    with_regressor = fit_plant_to_changes_with_regressors(
        sky=sky,
        output_mw=aggregate,
        regressors=regressor[:, None],
        orientation="free",
        ac_guess_mw=40.0,
        lag_half_hours=8,
    )
    without = fit_plant_to_changes_with_regressors(
        sky=sky,
        output_mw=aggregate,
        regressors=np.empty((len(aggregate), 0)),
        orientation="free",
        ac_guess_mw=40.0,
        lag_half_hours=8,
    )

    assert with_regressor is not None
    assert without is not None
    assert with_regressor.coefficients[0] == pytest.approx(coefficient, abs=0.3)
    assert with_regressor.fit.parameters.ac_capacity_mw == pytest.approx(
        truth.ac_capacity_mw, rel=0.05
    )
    assert with_regressor.fit.loss < 0.25 * without.fit.loss


def test_a_regressor_of_pure_noise_gets_a_coefficient_near_zero(sky: SunAndSky) -> None:
    aggregate, truth = _aggregate_with_a_plant(sky)
    noise = np.random.default_rng(7).normal(size=len(aggregate))

    fit = fit_plant_to_changes_with_regressors(
        sky=sky,
        output_mw=aggregate,
        regressors=noise[:, None],
        orientation="free",
        ac_guess_mw=40.0,
        lag_half_hours=8,
    )

    assert fit is not None
    assert abs(fit.coefficients[0]) < 0.3
    assert fit.fit.parameters.ac_capacity_mw == pytest.approx(truth.ac_capacity_mw, rel=0.1)


def test_a_fit_with_no_regressor_columns_matches_the_plain_fit_to_changes(
    sky: SunAndSky,
) -> None:
    aggregate, _ = _aggregate_with_a_plant(sky)

    plain = fit_plant_to_changes(
        sky=sky, output_mw=aggregate, orientation="free", ac_guess_mw=40.0, lag_half_hours=8
    )
    extended = fit_plant_to_changes_with_regressors(
        sky=sky,
        output_mw=aggregate,
        regressors=np.empty((len(aggregate), 0)),
        orientation="free",
        ac_guess_mw=40.0,
        lag_half_hours=8,
    )

    assert plain is not None
    assert extended is not None
    assert extended.fit.loss == pytest.approx(plain.loss, rel=1e-6)
    assert extended.fit.parameters == plain.parameters


def test_a_fit_with_regressors_rejects_the_wrong_shape_and_missing_values(
    sky: SunAndSky,
) -> None:
    aggregate, _ = _aggregate_with_a_plant(sky)
    arguments = {
        "sky": sky,
        "output_mw": aggregate,
        "orientation": "free",
        "ac_guess_mw": 40.0,
        "lag_half_hours": 8,
    }
    with pytest.raises(ValueError, match="shape"):
        fit_plant_to_changes_with_regressors(regressors=np.ones((10, 1)), **arguments)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="shape"):
        fit_plant_to_changes_with_regressors(regressors=np.ones(len(aggregate)), **arguments)  # ty: ignore[invalid-argument-type]
    holes = np.ones((len(aggregate), 1))
    holes[1000, 0] = np.nan
    with pytest.raises(ValueError, match="regressors are not finite"):
        fit_plant_to_changes_with_regressors(regressors=holes, **arguments)  # ty: ignore[invalid-argument-type]


def test_a_fit_with_a_regressor_works_for_an_orientation_with_no_tilt_or_azimuth(
    sky: SunAndSky,
) -> None:
    truth = PlantParameters(
        tilt_deg=FIXED_TILT_DEG,
        azimuth_deg=FIXED_AZIMUTH_DEG,
        dc_capacity_mw=70.0,
        ac_capacity_mw=50.0,
    )
    regressor = _daily_pattern(sky, seed=3)
    aggregate = plant_power_mw(sky=sky, parameters=truth) + 12.0 * regressor

    fit = fit_plant_to_changes_with_regressors(
        sky=sky,
        output_mw=aggregate,
        regressors=regressor[:, None],
        orientation="fixed",
        ac_guess_mw=40.0,
        lag_half_hours=8,
    )

    assert fit is not None
    assert fit.coefficients[0] == pytest.approx(12.0, abs=0.3)
    assert fit.fit.parameters.ac_capacity_mw == pytest.approx(50.0, rel=0.05)


def test_a_fit_to_changes_reports_the_number_of_pairs_it_used(sky: SunAndSky) -> None:
    aggregate, _ = _aggregate_with_a_plant(sky)
    pairs = len(lagged_pairs(sky=sky, output_mw=aggregate, lag_half_hours=8))

    plain = fit_plant_to_changes(
        sky=sky, output_mw=aggregate, orientation="fixed", ac_guess_mw=40.0, lag_half_hours=8
    )
    with_regressors = fit_plant_to_changes_with_regressors(
        sky=sky,
        output_mw=aggregate,
        regressors=np.empty((len(aggregate), 0)),
        orientation="fixed",
        ac_guess_mw=40.0,
        lag_half_hours=8,
    )

    assert plain is not None
    assert with_regressors is not None
    assert plain.half_hours_fitted == pairs
    assert with_regressors.fit.half_hours_fitted == pairs


@pytest.mark.parametrize("ac_guess_mw", [0.0, -5.0])
def test_a_fit_to_changes_needs_a_positive_starting_capacity(
    sky: SunAndSky, ac_guess_mw: float
) -> None:
    aggregate, _ = _aggregate_with_a_plant(sky)
    arguments = {
        "sky": sky,
        "output_mw": aggregate,
        "orientation": "fixed",
        "ac_guess_mw": ac_guess_mw,
        "lag_half_hours": 8,
    }

    assert fit_plant_to_changes(**arguments) is None  # ty: ignore[invalid-argument-type]
    assert (
        fit_plant_to_changes_with_regressors(
            regressors=np.empty((len(aggregate), 0)),
            **arguments,  # ty: ignore[invalid-argument-type]
        )
        is None
    )


def test_a_fit_needs_more_than_a_hundred_daytime_half_hours(sky: SunAndSky) -> None:
    truth = PlantParameters(
        tilt_deg=FIXED_TILT_DEG,
        azimuth_deg=FIXED_AZIMUTH_DEG,
        dc_capacity_mw=70.0,
        ac_capacity_mw=50.0,
    )
    power = plant_power_mw(sky=sky, parameters=truth)
    daytime = np.flatnonzero(daylight(sky=sky) & (power > 1.0))

    def fit_on(count: int) -> object:
        output = np.full(len(power), np.nan)
        output[daytime[:count]] = power[daytime[:count]]
        return fit_plant(sky=sky, output_mw=output, orientation="fixed")

    assert fit_on(99) is None
    assert fit_on(100) is not None


def test_a_fit_reports_its_loss_per_half_hour_and_the_half_hours_it_used(sky: SunAndSky) -> None:
    truth = PlantParameters(
        tilt_deg=FIXED_TILT_DEG,
        azimuth_deg=FIXED_AZIMUTH_DEG,
        dc_capacity_mw=70.0,
        ac_capacity_mw=50.0,
    )
    noisy = plant_power_mw(sky=sky, parameters=truth) + np.random.default_rng(5).normal(
        0.0, 3.0, size=len(sky.ghi_w_m2)
    )
    half = np.arange(len(noisy)) < len(noisy) // 2

    whole = fit_plant(sky=sky, output_mw=noisy, orientation="fixed")
    first_half = fit_plant(sky=sky, output_mw=noisy, orientation="fixed", usable=half)

    assert whole is not None
    assert first_half is not None
    assert whole.half_hours_fitted == int(daylight(sky=sky).sum())
    assert first_half.half_hours_fitted == int((daylight(sky=sky) & half).sum())
    assert first_half.loss == pytest.approx(whole.loss, rel=0.3)


def _reference_pairs(*, sky: SunAndSky, output: np.ndarray, lag: int) -> list[int]:
    stamps = sky.half_hour_end_time
    day = daylight(sky=sky)
    return [
        i
        for i in range(lag, len(output))
        if np.isfinite(output[i])
        and np.isfinite(output[i - lag])
        and stamps[i] - stamps[i - lag] == np.timedelta64(30 * lag, "m")
        and (day[i] or day[i - lag])
    ]


def test_lagged_pairs_keep_exactly_the_finite_daylight_pairs_a_lag_apart(sky: SunAndSky) -> None:
    lag = 3
    output = np.ones(len(sky.half_hour_end_time))
    output[[200, 201, 5000]] = np.nan

    pairs = lagged_pairs(sky=sky, output_mw=output, lag_half_hours=lag)

    assert pairs.tolist() == _reference_pairs(sky=sky, output=output, lag=lag)
    assert len(pairs) < len(output) - lag


def test_the_change_variance_explained_is_one_for_the_same_output_and_zero_for_none(
    sky: SunAndSky,
) -> None:
    output = _daily_pattern(sky, seed=11) + 10.0 * np.arange(len(sky.ghi_w_m2)) % 7
    arguments = {"sky": sky, "output_mw": output, "lag_half_hours": 4}

    assert change_variance_explained(power_mw=output, **arguments) == pytest.approx(1.0)
    assert change_variance_explained(power_mw=np.zeros_like(output), **arguments) == pytest.approx(
        0.0
    )
    # Half the output's changes leave a quarter of their variance unexplained.
    assert change_variance_explained(power_mw=0.5 * output, **arguments) == pytest.approx(0.75)


def test_the_change_variance_explained_uses_changes_not_levels(sky: SunAndSky) -> None:
    output = _daily_pattern(sky, seed=12)

    shifted = change_variance_explained(
        sky=sky, output_mw=output, power_mw=output + 100.0, lag_half_hours=4
    )

    assert shifted == pytest.approx(1.0)


def test_the_smooth_minimum_follows_the_power_below_the_limit_and_the_limit_above_it() -> None:
    limit = 10.0
    values = _smooth_min(power=np.array([1.0, 5.0, 10.0, 20.0, 40.0]), limit=limit)

    np.testing.assert_allclose(values[:2], [1.0, 5.0], atol=1e-6)
    # At the limit the rounding is ln(2) / beta below it, where beta = 50 / limit.
    assert values[2] == pytest.approx(limit - np.log(2.0) * limit / 50.0)
    np.testing.assert_allclose(values[3:], [limit, limit], atol=1e-6)


def test_the_starting_vectors_and_bounds_cover_nine_orientations_when_free_and_one_otherwise() -> (
    None
):
    free = _starts(orientation="free", ac_guess_mw=10.0)
    fixed = _starts(orientation="fixed", ac_guess_mw=10.0)
    low, high = _bounds(orientation="free", ac_guess_mw=10.0, ratio_bounds=(1.0, 2.2))
    fixed_low, fixed_high = _bounds(orientation="fixed", ac_guess_mw=10.0, ratio_bounds=(1.0, 2.2))

    assert len(free) == 9
    assert {(float(v[2]), float(v[3])) for v in free} == {
        (t, a) for t in (15.0, 30.0, 45.0) for a in (150.0, 180.0, 210.0)
    }
    assert len(fixed) == 1
    np.testing.assert_allclose(low, [np.log(2.0), 1.0, 0.0, 90.0])
    np.testing.assert_allclose(high, [np.log(50.0), 2.2, 70.0, 270.0])
    np.testing.assert_allclose(fixed_low, low[:2])
    np.testing.assert_allclose(fixed_high, high[:2])


def test_each_prior_residual_is_the_noise_times_the_standard_scores_of_its_parameter() -> None:
    priors = PlantPriors(
        tilt_mean_deg=30.0,
        tilt_sd_deg=10.0,
        azimuth_mean_deg=180.0,
        azimuth_sd_deg=20.0,
        ratio_mean=1.3,
        ratio_sd=0.1,
    )
    theta = np.array([0.0, 1.5, 50.0, 140.0])

    free = _prior_residuals(theta=theta, orientation="free", priors=priors, noise=2.0)
    fixed = _prior_residuals(theta=theta, orientation="fixed", priors=priors, noise=2.0)
    tracker = _prior_residuals(theta=theta, orientation="tracker", priors=priors, noise=2.0)

    np.testing.assert_allclose(free, [2.0 * 2.0, 2.0 * 2.0, 2.0 * -2.0])
    np.testing.assert_allclose(fixed, free[:1])
    np.testing.assert_allclose(tracker, free[:1])


def test_the_best_start_is_the_lowest_cost_local_minimum_and_uses_a_robust_loss() -> None:
    def two_wells(theta: np.ndarray) -> np.ndarray:
        # Roots of the first term at -1 and 1; the second term adds a cost only at the left well.
        return np.array([(theta[0] ** 2 - 1.0) ** 2, 0.3 * (theta[0] - 1.0)])

    bounds = (np.array([-3.0]), np.array([3.0]))
    forwards = _best_start(
        residuals=two_wells,
        starts=[np.array([-1.1]), np.array([0.9])],
        bounds=bounds,
        f_scale=1.0,
    )
    backwards = _best_start(
        residuals=two_wells,
        starts=[np.array([0.9]), np.array([-1.1])],
        bounds=bounds,
        f_scale=1.0,
    )

    assert forwards is not None
    assert backwards is not None
    assert forwards[1][0] == pytest.approx(1.0, abs=1e-3)
    assert backwards[1][0] == pytest.approx(1.0, abs=1e-3)

    data = np.array([0.0, 0.0, 0.0, 0.0, 100.0])
    robust = _best_start(
        residuals=lambda theta: theta[0] - data,
        starts=[np.array([10.0])],
        bounds=(np.array([-200.0]), np.array([200.0])),
        f_scale=1.0,
    )
    assert robust is not None
    assert abs(robust[1][0]) < 5.0
    assert _best_start(residuals=two_wells, starts=[], bounds=bounds, f_scale=1.0) is None
