"""Tests of the Kalman filter on the convex model's effects (`studies.dp_effects_filter`).

The filter is checked against a dense NumPy Kalman filter on random small problems, against the
per-band filter of `studies.dp_assimilation` when only the level weights move, by changing every
residual after the cut, and by recovering the random-walk variances of a simulated stream. The
decomposition of a fit into effects is checked on the synthetic substation of `conftest.py`.
"""

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from typing import Final

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")

from studies.dp_assimilation import FilterHyperparameters, run_kalman  # noqa: E402
from studies.dp_effects_filter import (  # noqa: E402
    FLOORED_WEIGHTS,
    HEATING,
    HOLIDAY_INDEX,
    LEVEL,
    N_STATES,
    N_WEIGHTS,
    NON_LEVEL,
    PROFILE,
    PV_INDEX,
    FilterNoise,
    FilterState,
    FilterVariant,
    SiteStream,
    active_counts,
    correction_from_state,
    effect_regressors,
    fit_effects_noise,
    fit_effects_noise_sites,
    initial_sd_for,
    regressor_scale,
    relative_step_per_year,
    run_effects_filter,
    state_at,
    stitched_effect_regressors,
)
from studies.dp_ens import proxy_residual_stream  # noqa: E402
from studies.dp_features import FeatureCache  # noqa: E402
from studies.dp_gam import GamSpec, TrainingWindow, fit_gam, predict_gam  # noqa: E402
from studies.dp_pinned_capacity import PinnedCapacity, pinned_capacity  # noqa: E402

from studies import dp_gam  # noqa: E402

CUT: Final[datetime] = datetime(2024, 7, 1, tzinfo=UTC)
SEASON_YEARS: Final[range] = range(2019, 2026)
N_BANDS: Final[int] = 4
HYPER: Final[FilterHyperparameters] = FilterHyperparameters(
    log_q=np.log(0.4), logit_phi=3.5, log_sigma_z=np.log(0.8), log_sigma_eps=np.log(0.7)
)
ALL_MOVING: Final[np.ndarray] = np.ones(N_WEIGHTS, dtype=bool)
LEVEL_ONLY: Final[np.ndarray] = np.arange(N_WEIGHTS) < N_BANDS


def _bands(*, n: int) -> np.ndarray:
    """Hour bands that run for 12 half-hours at a time, like the real bands."""
    return (np.arange(n) // 12) % 4


def _noise(*, q: np.ndarray, moving: np.ndarray, use_anomaly: bool = True) -> FilterNoise:
    return FilterNoise(
        q=np.where(moving, q, 0.0),
        phi=HYPER.phi,
        sigma_z=HYPER.sigma_z,
        sigma_eps=HYPER.sigma_eps,
        initial_sd=initial_sd_for(scale=np.ones(N_WEIGHTS), moving=moving),
        use_anomaly=use_anomaly,
    )


def _dense_reference(
    *, residual: np.ndarray, regressors: np.ndarray, band: np.ndarray, noise: FilterNoise
) -> tuple[np.ndarray, float]:
    """Run a dense Kalman filter that steps every half-hour; return the state after each reading.

    A level weight's random walk starts at its band's first reading, as in
    `studies.dp_assimilation`.
    """
    stationary = noise.sigma_z**2 / (1.0 - noise.phi**2)
    transition = np.diag(np.r_[np.ones(N_WEIGHTS), np.full(N_BANDS, noise.phi)])
    state = np.zeros(N_STATES)
    covariance = np.diag(np.r_[noise.initial_sd**2, np.full(N_BANDS, stationary)])
    started = np.ones(N_WEIGHTS)
    started[:N_BANDS] = 0.0
    states = []
    log_likelihood = 0.0
    first = np.flatnonzero(np.isfinite(residual))[0]
    for position in range(first, len(residual)):
        if position > first:
            process = np.diag(np.r_[noise.q / 48.0 * started, np.full(N_BANDS, 0.0)])
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process
            covariance[N_WEIGHTS:, N_WEIGHTS:] += np.eye(N_BANDS) * noise.sigma_z**2
        if not np.isfinite(residual[position]):
            continue
        started[band[position]] = 1.0
        h = np.zeros(N_STATES)
        h[:N_WEIGHTS] = regressors[position]
        if noise.use_anomaly:
            h[N_WEIGHTS + band[position]] = 1.0
        variance = h @ covariance @ h + noise.sigma_eps**2
        innovation = residual[position] - h @ state
        log_likelihood -= 0.5 * (np.log(2 * np.pi * variance) + innovation**2 / variance)
        gain = covariance @ h / variance
        state = state + gain * innovation
        covariance = covariance - np.outer(gain, gain) * variance
        states.append(state.copy())
    return np.array(states), log_likelihood


def _random_problem(*, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, FilterNoise]:
    generator = np.random.default_rng(seed)
    n = 60
    band = _bands(n=n)
    regressors = generator.normal(size=(n, N_WEIGHTS))
    regressors[:, :N_BANDS] = np.eye(N_BANDS)[band]
    regressors[generator.random(size=(n, N_WEIGHTS)) < 0.4] = 0.0
    regressors[:, :N_BANDS] = np.eye(N_BANDS)[band]
    residual = generator.normal(size=n)
    residual[generator.random(n) < 0.25] = np.nan
    residual[:4] = generator.normal(size=4)
    noise = _noise(
        q=np.exp(generator.uniform(-6.0, 0.0, size=N_WEIGHTS)),
        moving=generator.random(N_WEIGHTS) < 0.8,
        use_anomaly=bool(seed % 2),
    )
    return residual, regressors, band, noise


@pytest.mark.parametrize("seed", range(200))
def test_the_rank_one_filter_matches_a_dense_kalman_filter(seed: int):
    residual, regressors, band, noise = _random_problem(seed=seed)
    trace = run_effects_filter(residual=residual, regressors=regressors, band=band, noise=noise)
    states, log_likelihood = _dense_reference(
        residual=residual, regressors=regressors, band=band, noise=noise
    )
    assert np.allclose(trace.state, states, atol=1e-8, rtol=1e-8)
    assert trace.log_likelihood == pytest.approx(log_likelihood, abs=1e-8)


def test_a_batch_of_noise_settings_gives_the_likelihoods_of_separate_runs():
    from studies.dp_effects_filter import _filter_core

    residual, regressors, band, noise = _random_problem(seed=1)
    settings = [
        _noise(q=np.exp(np.full(N_WEIGHTS, log_q)), moving=ALL_MOVING) for log_q in (-8, -4, -1)
    ]
    batch, _, _, _ = _filter_core(
        residual=residual,
        regressors=regressors,
        band=band,
        q=np.stack([setting.q for setting in settings]),
        sigma_eps=np.array([setting.sigma_eps for setting in settings]),
        phi=noise.phi,
        sigma_z=noise.sigma_z,
        initial_sd=settings[0].initial_sd,
        use_anomaly=True,
        likelihood_mask=None,
        keep_trace=False,
    )
    for setting, value in zip(settings, batch, strict=True):
        single = run_effects_filter(
            residual=residual, regressors=regressors, band=band, noise=setting
        )
        assert single.log_likelihood == pytest.approx(value, abs=1e-9)


def test_with_only_the_level_weights_moving_the_filter_equals_the_per_band_filter():
    n = 48 * 40
    band = _bands(n=n)
    generator = np.random.default_rng(3)
    residual = generator.normal(size=n) * 1.5 + np.sin(np.arange(n) / 50.0)
    residual[generator.random(n) < 0.1] = np.nan
    regressors = np.zeros((n, N_WEIGHTS))
    regressors[np.arange(n), band] = 1.0
    noise = _noise(q=np.full(N_WEIGHTS, HYPER.q), moving=LEVEL_ONLY)
    trace = run_effects_filter(residual=residual, regressors=regressors, band=band, noise=noise)
    reference = run_kalman(residual=residual, band=band, hyper=HYPER)
    row_of_position = {int(position): row for row, position in enumerate(trace.positions)}
    for b in range(N_BANDS):
        rows = [row_of_position[int(position)] for position in reference.positions[b]]
        assert np.allclose(trace.state[rows, b], reference.level[b], atol=1e-9, rtol=0.0)
        assert np.allclose(
            trace.state[rows, N_WEIGHTS + b], reference.anomaly[b], atol=1e-9, rtol=0.0
        )
    assert trace.log_likelihood == pytest.approx(reference.log_likelihood, abs=1e-8)
    assert np.all(trace.state[:, N_BANDS:N_WEIGHTS] == 0.0)


def test_changing_any_residual_after_the_cut_leaves_the_state_at_the_cut_unchanged():
    residual, regressors, band, noise = _random_problem(seed=4)
    cutoff = 40
    before = state_at(
        trace=run_effects_filter(residual=residual, regressors=regressors, band=band, noise=noise),
        cutoff=cutoff,
    )
    altered = residual.copy()
    altered[cutoff + 1 :] = 50.0
    changed_regressors = regressors.copy()
    changed_regressors[cutoff + 1 :, N_BANDS:] *= 3.0
    after = state_at(
        trace=run_effects_filter(
            residual=altered, regressors=changed_regressors, band=band, noise=noise
        ),
        cutoff=cutoff,
    )
    assert np.array_equal(before.delta, after.delta)
    assert np.array_equal(before.anomaly, after.anomaly)
    assert before.position == after.position <= cutoff


def test_a_cutoff_before_the_first_reading_gives_a_zero_state():
    residual, regressors, band, noise = _random_problem(seed=5)
    trace = run_effects_filter(residual=residual, regressors=regressors, band=band, noise=noise)
    state = state_at(trace=trace, cutoff=-1)
    assert state.position == -1
    assert not state.delta.any()
    assert not state.anomaly.any()


def _correction_inputs() -> tuple[FilterState, np.ndarray, np.ndarray, np.ndarray]:
    delta = np.zeros(N_WEIGHTS)
    delta[LEVEL] = (1.0, 2.0, 3.0, 4.0)
    delta[HEATING.start] = -5.0
    delta[PV_INDEX] = 0.5
    delta[HOLIDAY_INDEX] = -9.0
    state = FilterState(
        delta=delta, anomaly=np.array([1.0, 2.0, 3.0, 4.0]), sd=np.zeros(N_WEIGHTS), position=100
    )
    band = np.array([0, 1, 2, 3])
    target = np.array([101, 102, 110, 150])
    contributions = np.zeros((2, 4, N_WEIGHTS))
    contributions[:, np.arange(4), band] = 1.0
    contributions[:, 0, HEATING.start] = (1.0, 2.0)
    contributions[:, 1, PV_INDEX] = (-1.0, -3.0)
    contributions[:, 2, HOLIDAY_INDEX] = (1.0, 1.0)
    return state, contributions, band, target


def test_the_correction_holds_the_weights_floors_three_groups_and_decays_the_anomaly():
    state, contributions, band, target = _correction_inputs()
    scale = np.full(N_WEIGHTS, 2.0)
    result = correction_from_state(
        state=state,
        standardised_contributions=contributions,
        scale=scale,
        band=band,
        target=target,
        phi=0.9,
    )
    anomaly = 0.9 ** (target - 100) * state.anomaly[band]
    assert np.allclose(result.anomaly, anomaly)
    assert np.allclose(result.level, (1.0, 2.0, 3.0, 4.0))
    # The heating weight is floored at -scale = -2, so member 0 gets level 1 + (-2)(1) + anomaly.
    assert result.shift[0, 0] == pytest.approx(1.0 - 2.0 + anomaly[0])
    assert result.shift[1, 0] == pytest.approx(1.0 - 4.0 + anomaly[0])
    # PV is not floored here (0.5 > -2): level 2 + 0.5 * (-1) + anomaly.
    assert result.shift[0, 1] == pytest.approx(2.0 - 0.5 + anomaly[1])
    # The holiday weight is not floored: level 3 + (-9)(1) + anomaly.
    assert result.shift[0, 2] == pytest.approx(3.0 - 9.0 + anomaly[2])
    assert not FLOORED_WEIGHTS[:N_BANDS].any()
    assert FLOORED_WEIGHTS[PROFILE.start : PV_INDEX + 1].all()
    assert not FLOORED_WEIGHTS[HOLIDAY_INDEX]


def test_a_state_with_no_reading_adds_no_anomaly():
    state = FilterState(
        delta=np.zeros(N_WEIGHTS), anomaly=np.zeros(N_BANDS), sd=np.zeros(N_WEIGHTS), position=-1
    )
    result = correction_from_state(
        state=state,
        standardised_contributions=np.ones((1, 3, N_WEIGHTS)),
        scale=np.ones(N_WEIGHTS),
        band=np.array([0, 1, 2]),
        target=np.array([5, 6, 7]),
        phi=0.9,
    )
    assert not result.shift.any()


def _simulate_stream(
    *, true_q: np.ndarray, seed: int, n: int = 48 * 330, noise_sd: float = 0.6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate weights that follow random walks of known variance on smooth regressors."""
    generator = np.random.default_rng(seed)
    band = _bands(n=n)
    day = np.arange(n) / 48.0
    regressors = np.zeros((n, N_WEIGHTS))
    regressors[np.arange(n), band] = 1.0
    regressors[:, HEATING.start] = 1.0 + 0.8 * np.sin(2 * np.pi * day / 6.0)
    regressors[:, PV_INDEX] = 1.0 + 0.8 * np.cos(2 * np.pi * day / 5.0)
    regressors[:, PROFILE.start] = 1.0 + 0.8 * np.sin(2 * np.pi * day / 3.0 + 1.0)
    weights = np.cumsum(generator.normal(size=(n, N_WEIGHTS)) * np.sqrt(true_q / 48.0), axis=0)
    residual = (weights * regressors).sum(axis=1) + noise_sd * generator.normal(size=n)
    return residual, regressors, band


def test_a_simulated_stream_recovers_the_random_walk_variances_it_was_built_with():
    true_q = np.zeros(N_WEIGHTS)
    true_q[HEATING.start] = 1e-2
    true_q[PV_INDEX] = 1e-1
    residual, regressors, band = _simulate_stream(true_q=true_q, seed=7)
    moving = np.zeros(N_WEIGHTS, dtype=bool)
    moving[[HEATING.start, PV_INDEX, PROFILE.start]] = True
    fitted = fit_effects_noise(
        residual=residual,
        regressors=regressors,
        band=band,
        scale=np.ones(N_WEIGHTS),
        variant=FilterVariant(moving=moving, use_anomaly=False),
        phi=0.98,
        sigma_z=0.3,
        sigma_eps_start=0.6,
        likelihood_mask=np.ones(len(residual), dtype=bool),
        prior_mean=np.full(N_WEIGHTS, np.log(0.05)),
        prior_spread=np.full(N_WEIGHTS, 20.0),
    )
    assert fitted.noise.q[HEATING.start] == pytest.approx(true_q[HEATING.start], rel=2.0)
    assert fitted.noise.q[PV_INDEX] == pytest.approx(true_q[PV_INDEX], rel=2.0)
    assert fitted.noise.q[PROFILE.start] < 1e-6
    assert fitted.noise.sigma_eps == pytest.approx(0.6, rel=0.15)
    assert not fitted.set_by_prior.any()
    assert fitted.noise.q[~moving].max() == 0.0


def test_a_weight_with_too_few_active_half_hours_has_its_q_set_to_the_prior_mean():
    true_q = np.zeros(N_WEIGHTS)
    residual, regressors, band = _simulate_stream(true_q=true_q, seed=8)
    regressors[:, HOLIDAY_INDEX] = 0.0
    regressors[:30, HOLIDAY_INDEX] = 1.0
    moving = np.zeros(N_WEIGHTS, dtype=bool)
    moving[[HEATING.start, HOLIDAY_INDEX]] = True
    prior_mean = np.full(N_WEIGHTS, -4.0)
    prior_mean[HOLIDAY_INDEX] = -2.5
    fitted = fit_effects_noise(
        residual=residual,
        regressors=regressors,
        band=band,
        scale=np.ones(N_WEIGHTS),
        variant=FilterVariant(moving=moving, use_anomaly=False),
        phi=0.98,
        sigma_z=0.3,
        sigma_eps_start=0.6,
        likelihood_mask=np.ones(len(residual), dtype=bool),
        prior_mean=prior_mean,
        prior_spread=np.full(N_WEIGHTS, 2.0),
    )
    assert fitted.set_by_prior[HOLIDAY_INDEX]
    assert not fitted.set_by_prior[HEATING.start]
    assert fitted.noise.q[HOLIDAY_INDEX] == pytest.approx(np.exp(-2.5))
    assert fitted.n_active[HOLIDAY_INDEX] == 30


def test_the_identity_variant_fits_one_q_for_every_moving_weight():
    true_q = np.zeros(N_WEIGHTS)
    true_q[[HEATING.start, PV_INDEX]] = 5e-2
    residual, regressors, band = _simulate_stream(true_q=true_q, seed=9)
    moving = np.zeros(N_WEIGHTS, dtype=bool)
    moving[[HEATING.start, PV_INDEX]] = True
    fitted = fit_effects_noise(
        residual=residual,
        regressors=regressors,
        band=band,
        scale=np.ones(N_WEIGHTS),
        variant=FilterVariant(moving=moving, use_anomaly=False, shared_q=True),
        phi=0.98,
        sigma_z=0.3,
        sigma_eps_start=0.6,
        likelihood_mask=np.ones(len(residual), dtype=bool),
        prior_mean=np.full(N_WEIGHTS, -3.0),
        prior_spread=np.full(N_WEIGHTS, 20.0),
    )
    assert fitted.noise.q[HEATING.start] == fitted.noise.q[PV_INDEX]
    assert fitted.noise.q[HEATING.start] == pytest.approx(5e-2, rel=2.0)


def test_the_site_prior_is_refitted_from_the_sites_that_fitted_each_weight():
    streams = {}
    for site, seed in (("A", 10), ("B", 11)):
        true_q = np.zeros(N_WEIGHTS)
        true_q[HEATING.start] = 2e-2
        residual, regressors, band = _simulate_stream(true_q=true_q, seed=seed, n=48 * 200)
        streams[site] = SiteStream(
            residual=residual,
            regressors=regressors,
            scale=np.ones(N_WEIGHTS),
            likelihood_mask=np.ones(len(residual), dtype=bool),
            phi=0.98,
            sigma_z=0.3,
            sigma_eps_start=0.6,
        )
    moving = np.zeros(N_WEIGHTS, dtype=bool)
    moving[HEATING.start] = True
    fits, prior_mean, prior_spread = fit_effects_noise_sites(
        streams=streams, band=band, variant=FilterVariant(moving=moving, use_anomaly=False)
    )
    fitted = [fit.log_q[HEATING.start] for fit in fits.values()]
    assert prior_mean[HEATING.start] == pytest.approx(np.mean(fitted))
    assert prior_spread[HEATING.start] >= 0.5
    with pytest.raises(ValueError, match="no sites"):
        fit_effects_noise_sites(
            streams={}, band=band, variant=FilterVariant(moving=moving, use_anomaly=False)
        )


def test_the_scale_is_the_rms_over_active_half_hours_and_a_level_has_scale_one():
    contributions = np.zeros((10, N_WEIGHTS))
    contributions[:, 0] = 1.0
    contributions[:4, PV_INDEX] = (3.0, 4.0, 3.0, 4.0)
    mask = np.ones(10, dtype=bool)
    scale = regressor_scale(contributions=contributions, mask=mask)
    assert scale[0] == 1.0
    assert scale[PV_INDEX] == pytest.approx(np.sqrt(12.5))
    assert scale[HOLIDAY_INDEX] == 1.0
    assert active_counts(contributions=contributions, mask=mask)[PV_INDEX] == 4
    steps = relative_step_per_year(
        noise=_noise(q=np.full(N_WEIGHTS, 1.0 / 365.0), moving=ALL_MOVING), scale=scale
    )
    assert steps[0] == pytest.approx(1.0)
    assert steps[PV_INDEX] == pytest.approx(1.0 / scale[PV_INDEX])


def test_a_held_weight_has_no_starting_spread_and_a_level_starts_at_ten_points():
    scale = np.full(N_WEIGHTS, 2.0)
    moving = np.zeros(N_WEIGHTS, dtype=bool)
    moving[[0, PV_INDEX]] = True
    sd = initial_sd_for(scale=scale, moving=moving)
    assert sd[0] == pytest.approx(10.0)
    assert sd[PV_INDEX] == pytest.approx(0.2)
    assert sd[HEATING.start] == 0.0
    assert NON_LEVEL.sum() == 10


def _pinned_capacity(*, cache: FeatureCache, cut: datetime) -> PinnedCapacity:
    weeks = []
    week = date(2019, 9, 2)
    while week <= cut.date():
        weeks.append(week)
        week += timedelta(days=7)
    vintages = pl.DataFrame(
        {
            "time_series_id": cache.sites["S1"].table.time_series_id,
            "variant": "frozen",
            "origin_monday": cut.date(),
            "week_start": weeks,
            "capacity_mw": [1.0 + 0.01 * i for i in range(len(weeks))],
            "unobserved": False,
        }
    )
    return pinned_capacity(vintages=vintages, table=cache.sites["S1"].table, cut=cut)


def _fitted(*, cache: FeatureCache) -> tuple[FeatureCache, dp_gam.GamFit]:
    """Fit the pinned-capacity model on power made from a plausible truth."""
    spec = GamSpec(capacity="pinned", ridge=0.01)
    pinned = _pinned_capacity(cache=cache, cut=CUT)
    layout, _ = dp_gam.prepare_fit(
        cache=cache,
        site="S1",
        spec=spec,
        cut=CUT,
        window=TrainingWindow(),
        season_years=SEASON_YEARS,
    )
    theta = np.zeros(layout.n_columns)
    profile = layout.slices["profile"]
    shape = 40.0 + 6.0 * np.sin(2 * np.pi * np.arange(48) / 48 - 1.0)
    for day_type, offset in enumerate((0.0, -4.0, -6.0, -6.0)):
        theta[profile.start + day_type * 48 : profile.start + (day_type + 1) * 48] = shape + offset
    heating = layout.slices["heating"]
    for band, slope in enumerate((0.8, 1.5, 1.0, 1.8)):
        theta[heating.start + N_BANDS + band] = slope
    theta[layout.slices["holiday"]] = 0.5
    features = cache.sites["S1"]
    usable = np.flatnonzero(
        np.isfinite(features.pv_unit_shape)
        & np.isfinite(features.smoothed_temperature_c).all(axis=1)
    )
    y = np.full(len(cache.end_times), np.nan)
    for start in range(0, len(usable), 20000):
        rows = usable[start : start + 20000]
        y[rows] = (
            dp_gam.design_matrix(cache=cache, features=features, layout=layout, index=rows) @ theta
        )
        y[rows] -= dp_gam.pinned_solar_points(
            cache=cache, features=features, pinned=pinned, index=rows
        )
    y = y + 0.15 * np.random.default_rng(12).normal(size=len(y))
    truth_cache = replace(cache, sites={"S1": replace(features, y=y)})
    fit = fit_gam(
        cache=truth_cache,
        site="S1",
        specs=[spec],
        cut=CUT,
        window=TrainingWindow(),
        season_years=SEASON_YEARS,
        pinned=pinned,
    )
    return truth_cache, fit


def _rows_after_cut(*, cache: FeatureCache, days: int = 60) -> np.ndarray:
    start = np.datetime64("2024-07-01T00:00", "us")
    return np.flatnonzero(
        (cache.end_times > start)
        & (cache.end_times <= start + np.timedelta64(days, "D"))
        & np.isfinite(cache.sites["S1"].pv_unit_shape)
        & np.isfinite(cache.sites["S1"].smoothed_temperature_c).all(axis=1)
    )


def test_the_effects_and_the_remainder_sum_to_the_prediction_and_split_the_terms_correctly(
    dp_synthetic_cache: FeatureCache,
):
    cache, fit = _fitted(cache=dp_synthetic_cache)
    index = _rows_after_cut(cache=cache)
    pieces = effect_regressors(cache=cache, fit=fit, index=index)
    total = predict_gam(cache=cache, fit=fit, index=index)
    summed = pieces.remainder + pieces.contributions[:, N_BANDS:].sum(axis=1)
    assert np.abs(summed - total).max() < 1e-6
    layout = fit.layout
    others = np.ones(layout.n_columns, dtype=bool)
    for name in ("profile", "heating", "holiday", "tent"):
        others[layout.slices[name]] = False
    day_type = cache.calendar.day_type[index].astype(np.int64)
    profile_mean = fit.theta[0][layout.slices["profile"]].reshape(4, 48).mean(axis=1)
    expected = (
        predict_gam(cache=cache, fit=fit, index=index, columns=others) + profile_mean[day_type]
    )
    assert np.abs(pieces.remainder - expected).max() < 1e-6


def test_each_regressor_is_zero_where_its_term_is_zero(dp_synthetic_cache: FeatureCache):
    cache, fit = _fitted(cache=dp_synthetic_cache)
    index = _rows_after_cut(cache=cache)
    contributions = effect_regressors(cache=cache, fit=fit, index=index).contributions
    band = cache.calendar.band[index].astype(np.int64)
    day_type = cache.calendar.day_type[index].astype(np.int64)
    shape = cache.sites["S1"].pv_unit_shape[index]
    assert np.array_equal(contributions[:, LEVEL], np.eye(N_BANDS)[band])
    for b in range(N_BANDS):
        assert not contributions[band != b, HEATING.start + b].any()
        assert (contributions[:, HEATING.start + b] >= 0.0).all()
    for d in range(4):
        assert not contributions[day_type != d, PROFILE.start + d].any()
    assert (contributions[shape == 0.0, PV_INDEX] == 0.0).all()
    assert (contributions[:, PV_INDEX] <= 0.0).all()
    assert contributions[:, PV_INDEX].min() < 0.0
    for d in range(4):
        rows = day_type == d
        if rows.any():
            half_hour = cache.calendar.half_hour[index][rows]
            centred = contributions[rows, PROFILE.start + d]
            assert (
                abs(np.mean([centred[half_hour == h].mean() for h in np.unique(half_hour)])) < 1.0
            )


def test_the_stitched_regressors_are_finite_exactly_where_the_residual_stream_is(
    dp_synthetic_cache: FeatureCache,
):
    cache, fit = _fitted(cache=dp_synthetic_cache)
    later = dp_gam.fit_gam(
        cache=cache,
        site="S1",
        specs=[GamSpec(capacity="pinned", ridge=0.01)],
        cut=datetime(2024, 8, 1, tzinfo=UTC),
        window=TrainingWindow(),
        season_years=SEASON_YEARS,
        pinned=_pinned_capacity(cache=cache, cut=datetime(2024, 8, 1, tzinfo=UTC)),
    )
    stream = proxy_residual_stream(cache=cache, fits=[fit, later], site="S1")
    pieces = stitched_effect_regressors(cache=cache, fits=[fit, later], site="S1")
    assert np.array_equal(np.isfinite(stream), np.isfinite(pieces.contributions).all(axis=1))
    rows = np.flatnonzero(np.isfinite(stream))[:500]
    assert rows.size
    with pytest.raises(ValueError, match="strictly ascending"):
        stitched_effect_regressors(cache=cache, fits=[later, fit], site="S1")
    with pytest.raises(ValueError, match="at least one fit"):
        stitched_effect_regressors(cache=cache, fits=[], site="S1")


def test_a_fit_without_a_pinned_capacity_is_refused(dp_synthetic_cache: FeatureCache):
    _, fit = _fitted(cache=dp_synthetic_cache)
    with pytest.raises(ValueError, match="pinned capacity"):
        effect_regressors(
            cache=dp_synthetic_cache, fit=replace(fit, pinned=None), index=np.array([0])
        )
