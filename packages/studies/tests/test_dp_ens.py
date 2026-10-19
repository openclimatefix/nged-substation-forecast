"""Tests of the convex model driven by ensemble weather (`studies.dp_ens`).

The synthetic feature cache of `conftest.py` supplies the grid. A forecast built from an ensemble
that equals the cache's own ERA5 and CAMS weather must reproduce the convex model's forecast on the
cache. The leakage tests give the ensemble weather that differs from the cache's, so a forecast that
read ERA5 where it should read the analysis proxy would change when the cache's ERA5 changes.
"""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Final

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from studies.dp_assimilation import (  # noqa: E402
    FilterHyperparameters,
    run_kalman,
    usable_cutoff_positions,
)
from studies.dp_features import FeatureCache, naive_utc  # noqa: E402
from studies.dp_gam import (  # noqa: E402
    GamFit,
    GamSpec,
    TrainingWindow,
    fit_gam,
    predict_gam,
)
from studies.ens_members import ENSEMBLE_SIZE  # noqa: E402

from studies import dp_ens  # noqa: E402
from studies import dp_ens_weather as weather  # noqa: E402

SITE: Final[str] = "S1"
CUTS: Final[tuple[datetime, ...]] = (
    datetime(2024, 7, 1, tzinfo=UTC),
    datetime(2024, 8, 1, tzinfo=UTC),
    datetime(2024, 9, 5, tzinfo=UTC),
)
ORIGIN: Final[datetime] = datetime(2024, 9, 10, 9, tzinfo=UTC)
N_RUN_TARGETS: Final[int] = 720
ENSEMBLE_OFFSET_K: Final[float] = -4.0


@pytest.fixture(scope="module")
def responsive_cache(dp_synthetic_cache: FeatureCache) -> FeatureCache:
    """Return the synthetic cache with power that responds to temperature and sunshine.

    The conftest power is flat, so a fit on it ignores the weather and no test of the weather
    path could fail. Here power rises as the trailing temperatures fall below the heating
    threshold and falls with the PV shape, so a forecast changes with the weather it is given.
    """
    features = dp_synthetic_cache.sites[SITE]
    cold_long = np.clip(15.5 - features.smoothed_temperature_c[:, 3], 0.0, None)
    cold_short = np.clip(15.5 - features.smoothed_temperature_c[:, 0], 0.0, None)
    power = 40.0 + 1.5 * cold_long + 1.0 * cold_short - 30.0 * features.pv_unit_shape
    return weather.replace_site(cache=dp_synthetic_cache, site=SITE, y=power)


@pytest.fixture(scope="module")
def fits(responsive_cache: FeatureCache) -> list[GamFit]:
    """Fit the convex model at three cuts."""
    return [
        fit_gam(
            cache=responsive_cache,
            site=SITE,
            specs=[GamSpec()],
            cut=cut,
            window=TrainingWindow(),
            season_years=range(2019, 2026),
        )
        for cut in CUTS
    ]


def _position(*, cache: FeatureCache, moment: datetime) -> int:
    return int(np.searchsorted(cache.end_times, naive_utc(moment=moment)))


def _run(
    *,
    cache: FeatureCache,
    origin_position: int,
    temperature_offset: float = 0.0,
    member_step: float = 0.0,
) -> weather.EnsSiteRun:
    """Build a run from the cache's ERA5 and CAMS values, shifted, with members spread apart."""
    features = cache.sites[SITE]
    span = slice(origin_position + 1, origin_position + 1 + N_RUN_TARGETS)
    temperature = np.tile(features.temperature_c[span] + temperature_offset, (ENSEMBLE_SIZE, 1))
    temperature += member_step * np.arange(ENSEMBLE_SIZE)[:, None]
    return weather.EnsSiteRun(
        origin_position=origin_position,
        temperature_c=temperature,
        shortwave_w_m2=np.tile(features.ghi_w_m2[span], (ENSEMBLE_SIZE, 1)),
    )


def _context(
    *, cache: FeatureCache, origin_position: int, days_back: int = 20, offset: float = 0.0
) -> weather.WeatherContext:
    runs = [
        _run(cache=cache, origin_position=origin_position - 48 * back, temperature_offset=offset)
        for back in range(days_back + 1)
    ]
    return weather.WeatherContext(
        site=SITE,
        calibration=weather.Calibration.identity(),
        climatology=weather.ghi_climatology(cache=cache, site=SITE, before=CUTS[0]),
        proxy=weather.AnalysisProxy.build(runs=runs, n_rows=len(cache.end_times)),
    )


def _spread() -> dp_ens.LeadHourSpread:
    generator = np.random.default_rng(5)
    n = 8_000
    lead_hours = generator.uniform(0.5, 168.0, n)
    return dp_ens.LeadHourSpread.fit(
        residuals=generator.normal(0.0, 1.0 + lead_hours / 40.0),
        band=generator.integers(0, 4, n),
        quarter=generator.integers(0, 4, n),
        lead_hours=lead_hours,
    )


def _trace(*, cache: FeatureCache, stream: np.ndarray):  # noqa: ANN202
    hyper = FilterHyperparameters(log_q=-3.0, logit_phi=4.0, log_sigma_z=-1.0, log_sigma_eps=0.0)
    return run_kalman(residual=stream, band=cache.calendar.band, hyper=hyper)


def _gam_k_forecast(
    *,
    cache: FeatureCache,
    fits: list[GamFit],
    run: weather.EnsSiteRun,
    context: weather.WeatherContext,
) -> dp_ens.RunForecast:
    """Run the whole GAM-K chain: observations, residual stream, filter, forecast."""
    stream = dp_ens.proxy_residual_stream(
        cache=weather.proxy_cache(cache=cache, context=context), fits=fits, site=SITE
    )
    return dp_ens.forecast_run(
        cache=cache,
        context=context,
        run=run,
        origin=ORIGIN,
        fits=fits,
        spread=_spread(),
        trace=_trace(cache=cache, stream=stream),
    )


def test_the_ensemble_path_with_an_identity_calibration_reproduces_the_cache_forecast(
    responsive_cache: FeatureCache, fits: list[GamFit]
):
    cache = responsive_cache
    position = _position(cache=cache, moment=ORIGIN)
    result = dp_ens.member_means(
        cache=cache,
        context=_context(cache=cache, origin_position=position),
        run=_run(cache=cache, origin_position=position),
        origin=ORIGIN,
        fits=fits,
        members=[0, 7],
    )
    index = np.arange(result.window.start, result.window.stop)
    expected = predict_gam(cache=cache, fit=fits[2], index=index)
    assert result.means.shape == (2, weather.HORIZON_HALF_HOURS)
    assert result.fit.cut == CUTS[2]
    assert np.abs(result.means - expected).max() < 1e-6


def test_members_that_all_equal_the_control_give_one_forecast_and_the_mixture_equals_it(
    responsive_cache: FeatureCache, fits: list[GamFit]
):
    cache = responsive_cache
    position = _position(cache=cache, moment=ORIGIN)
    result = dp_ens.member_means(
        cache=cache,
        context=_context(cache=cache, origin_position=position),
        run=_run(cache=cache, origin_position=position),
        origin=ORIGIN,
        fits=fits,
        members=range(0, 51, 10),
    )
    assert (result.means == result.means[0]).all()


def test_a_forecast_ignores_the_cache_weather_and_follows_its_own_proxy(
    responsive_cache: FeatureCache, fits: list[GamFit]
):
    cache = responsive_cache
    position = _position(cache=cache, moment=ORIGIN)
    context = _context(cache=cache, origin_position=position, offset=ENSEMBLE_OFFSET_K)
    run = _run(cache=cache, origin_position=position, temperature_offset=ENSEMBLE_OFFSET_K)
    before = dp_ens.member_means(
        cache=cache, context=context, run=run, origin=ORIGIN, fits=fits, members=[3]
    )
    features = cache.sites[SITE]
    changed = weather.replace_site(
        cache=cache,
        site=SITE,
        y=features.y + 40.0,
        temperature_c=features.temperature_c - 15.0,
        smoothed_temperature_c=features.smoothed_temperature_c - 15.0,
        pv_unit_shape=np.zeros_like(features.pv_unit_shape),
        ghi_w_m2=np.zeros_like(features.ghi_w_m2),
    )
    after = dp_ens.member_means(
        cache=changed, context=context, run=run, origin=ORIGIN, fits=fits, members=[3]
    )
    assert np.array_equal(before.means, after.means)
    warmer = _context(cache=cache, origin_position=position, offset=ENSEMBLE_OFFSET_K + 3.0)
    moved = dp_ens.member_means(
        cache=cache, context=warmer, run=run, origin=ORIGIN, fits=fits, members=[3]
    )
    assert np.abs(moved.means - before.means).max() > 0.01


def test_a_forecast_uses_the_latest_cut_at_or_before_its_origin_and_refuses_a_cut_in_the_delay(
    fits: list[GamFit],
):
    cuts = [fit.cut for fit in fits]
    assert dp_ens.latest_cut_index(cuts=cuts, moment=datetime(2024, 6, 30, tzinfo=UTC)) == -1
    assert dp_ens.latest_cut_index(cuts=cuts, moment=CUTS[1]) == 1
    assert dp_ens.latest_cut_index(cuts=cuts, moment=datetime(2024, 7, 31, 23, tzinfo=UTC)) == 0
    with pytest.raises(ValueError, match="lookahead"):
        dp_ens.assert_cut_not_after(fit=fits[1], origin=datetime(2024, 7, 31, tzinfo=UTC))
    with pytest.raises(ValueError, match=r"within 3\.5 hours"):
        dp_ens.choose_fit(fits=fits, origin=CUTS[1] + timedelta(hours=3))
    assert dp_ens.choose_fit(fits=fits, origin=CUTS[1] + timedelta(hours=4)) is fits[1]
    with pytest.raises(ValueError, match="no fit has a cut"):
        dp_ens.choose_fit(fits=fits, origin=datetime(2024, 6, 20, 9, tzinfo=UTC))


def test_a_forecast_between_two_cuts_uses_the_earlier_fit(
    responsive_cache: FeatureCache, fits: list[GamFit]
):
    cache = responsive_cache
    origin = datetime(2024, 7, 20, 9, tzinfo=UTC)
    position = _position(cache=cache, moment=origin)
    result = dp_ens.member_means(
        cache=cache,
        context=_context(cache=cache, origin_position=position),
        run=_run(cache=cache, origin_position=position),
        origin=origin,
        fits=fits,
        members=[0],
    )
    index = np.arange(result.window.start, result.window.stop)
    assert result.fit is fits[0]
    assert np.abs(result.means[0] - predict_gam(cache=cache, fit=fits[0], index=index)).max() < 1e-6
    assert np.abs(result.means[0] - predict_gam(cache=cache, fit=fits[1], index=index)).max() > 1e-3


def test_the_residual_stream_is_forward_chained_and_never_in_sample(
    responsive_cache: FeatureCache, fits: list[GamFit]
):
    cache = responsive_cache
    position = _position(cache=cache, moment=ORIGIN)
    proxied = weather.proxy_cache(
        cache=cache, context=_context(cache=cache, origin_position=position)
    )
    stream = dp_ens.proxy_residual_stream(cache=proxied, fits=fits, site=SITE)
    cut_1 = _position(cache=cache, moment=CUTS[1])
    cut_2 = _position(cache=cache, moment=CUTS[2])
    features = proxied.sites[SITE]
    early = np.flatnonzero(np.isfinite(stream[cut_1 + 1 : cut_2 + 1])) + cut_1 + 1
    late = np.flatnonzero(np.isfinite(stream[cut_2 + 1 :])) + cut_2 + 1
    assert len(early) > 100
    assert len(late) > 100
    assert np.isnan(stream[: cut_1 + 1]).all()
    for rows, fit in ((early, fits[1]), (late, fits[2])):
        expected = features.y[rows] - predict_gam(cache=proxied, fit=fit, index=rows)
        assert np.abs(stream[rows] - expected).max() < 1e-9
    wrong = features.y[early] - predict_gam(cache=proxied, fit=fits[2], index=early)
    assert np.abs(stream[early] - wrong).max() > 1e-3


def test_the_residual_stream_needs_fits_with_strictly_ascending_cuts(
    responsive_cache: FeatureCache, fits: list[GamFit]
):
    with pytest.raises(ValueError, match="strictly ascending"):
        dp_ens.proxy_residual_stream(cache=responsive_cache, fits=[fits[1], fits[0]], site=SITE)
    with pytest.raises(ValueError, match="at least one fit"):
        dp_ens.proxy_residual_stream(cache=responsive_cache, fits=[], site=SITE)


def test_a_gam_k_forecast_ignores_every_observation_after_the_telemetry_cutoff(
    responsive_cache: FeatureCache, fits: list[GamFit]
):
    cache = responsive_cache
    position = _position(cache=cache, moment=ORIGIN)
    context = _context(cache=cache, origin_position=position, offset=ENSEMBLE_OFFSET_K)
    run = _run(cache=cache, origin_position=position, temperature_offset=ENSEMBLE_OFFSET_K)
    cutoff = int(
        usable_cutoff_positions(
            end_times=cache.end_times,
            initialisation=np.array([naive_utc(moment=ORIGIN)]),
            delay_hours=3.5,
        )[0]
    )
    assert cutoff == position - 7
    base = _gam_k_forecast(cache=cache, fits=fits, run=run, context=context)
    assert np.abs(base.level).max() > 0.001
    features = cache.sites[SITE]
    for first in (cutoff + 1, cutoff + 7, position + 1):
        y = features.y.copy()
        y[first:] += 25.0
        moved_cache = weather.replace_site(cache=cache, site=SITE, y=y)
        after = _gam_k_forecast(cache=moved_cache, fits=fits, run=run, context=context)
        assert np.array_equal(base.quantiles, after.quantiles), first
    y = features.y.copy()
    y[cutoff - 3 : cutoff + 1] += 25.0
    seen = _gam_k_forecast(
        cache=weather.replace_site(cache=cache, site=SITE, y=y), fits=fits, run=run, context=context
    )
    assert np.abs(seen.quantiles - base.quantiles).max() > 1.0


def test_a_forecast_without_a_filter_has_no_correction_and_the_mixture_brackets_the_mean(
    responsive_cache: FeatureCache, fits: list[GamFit]
):
    cache = responsive_cache
    position = _position(cache=cache, moment=ORIGIN)
    forecast = dp_ens.forecast_run(
        cache=cache,
        context=_context(cache=cache, origin_position=position),
        run=_run(cache=cache, origin_position=position, member_step=0.05),
        origin=ORIGIN,
        fits=fits,
        spread=_spread(),
    )
    assert (forecast.level == 0.0).all()
    assert (forecast.anomaly == 0.0).all()
    assert forecast.quantiles.shape == (weather.HORIZON_HALF_HOURS, 11)
    assert (np.diff(forecast.quantiles, axis=1) >= 0).all()
    assert (forecast.quantiles[:, 0] < forecast.mean).all()
    assert (forecast.quantiles[:, -1] > forecast.mean).all()


def test_a_member_with_missing_weather_is_left_out_of_the_mixture(
    responsive_cache: FeatureCache, fits: list[GamFit]
):
    cache = responsive_cache
    position = _position(cache=cache, moment=ORIGIN)
    run = _run(cache=cache, origin_position=position, member_step=0.05)
    holed = run.shortwave_w_m2.copy()
    holed[36, 100] = np.nan
    forecast = dp_ens.forecast_run(
        cache=cache,
        context=_context(cache=cache, origin_position=position),
        run=replace(run, shortwave_w_m2=holed),
        origin=ORIGIN,
        fits=fits,
        spread=_spread(),
    )
    assert 36 not in forecast.members
    assert len(forecast.members) == ENSEMBLE_SIZE - 1
    assert np.isfinite(forecast.quantiles).all()


def test_the_margin_frame_gives_the_label_unit_and_megawatts_and_names_the_fit_it_used(
    responsive_cache: FeatureCache, fits: list[GamFit]
):
    cache = responsive_cache
    capacity = cache.sites[SITE].table.effective_capacity_mw
    for origin, fit in (
        (ORIGIN, fits[2]),
        (datetime(2024, 8, 20, 9, tzinfo=UTC), fits[1]),
        (datetime(2024, 7, 20, 9, tzinfo=UTC), fits[0]),
    ):
        position = _position(cache=cache, moment=origin)
        frame = dp_ens.margin_frame(
            cache=cache,
            context=_context(cache=cache, origin_position=position),
            run=_run(cache=cache, origin_position=position),
            origin=origin,
            fits=fits,
        )
        index = np.arange(position + 1, position + 1 + weather.HORIZON_HALF_HOURS)
        points = predict_gam(cache=cache, fit=fit, index=index)
        assert frame.height == weather.HORIZON_HALF_HOURS
        assert np.abs(frame["margin_fraction"].to_numpy() - points / 100.0).max() < 1e-6
        assert np.abs(frame["margin_mw"].to_numpy() - points * capacity / 100.0).max() < 1e-6
        assert np.allclose(
            frame["margin_mw"].to_numpy() / capacity, frame["margin_fraction"].to_numpy()
        )
        assert set(frame["fit_cut"].to_list()) == {fit.cut}
        assert (frame["fit_cut"] <= frame["power_fcst_init_time"]).all()


def test_the_spread_scales_each_lead_band_by_its_own_factor_and_stays_ordered():
    generator = np.random.default_rng(3)
    n = 4_000
    lead_hours = generator.uniform(0.5, 168.0, n)
    residuals = generator.normal(0.0, 1.0 + lead_hours / 50.0)
    band = generator.integers(0, 4, n)
    quarter = generator.integers(0, 4, n)
    spread = dp_ens.LeadHourSpread.fit(
        residuals=residuals, band=band, quarter=quarter, lead_hours=lead_hours
    )
    mean = np.zeros(n)
    base = spread.quantiles(mean=mean, band=band, quarter=quarter, lead_hours=lead_hours)
    wider = spread.with_factors(factors=[1.0, 2.0, 1.0, 1.0]).quantiles(
        mean=mean, band=band, quarter=quarter, lead_hours=lead_hours
    )
    lead = weather.lead_band_of_hours(lead_hours=lead_hours)
    assert np.allclose(wider[lead == 1], 2.0 * base[lead == 1])
    assert np.array_equal(wider[lead != 1], base[lead != 1])
    assert (np.diff(base, axis=1) >= 0).all()
    assert np.abs(base[:, 5]).max() < 1e-12
    assert base[lead == 3, -1].mean() > base[lead == 0, -1].mean()
    with pytest.raises(ValueError, match="one factor per lead band"):
        spread.with_factors(factors=[1.0])


def test_the_spread_refuses_a_lead_band_with_no_residuals():
    with pytest.raises(ValueError, match="holds no residual"):
        dp_ens.LeadHourSpread.fit(
            residuals=np.ones(10),
            band=np.zeros(10, dtype=int),
            quarter=np.zeros(10, dtype=int),
            lead_hours=np.full(10, 3.0),
        )


def test_a_window_slice_predicts_what_the_full_cache_predicts(
    responsive_cache: FeatureCache, fits: list[GamFit]
):
    sliced = weather.slice_cache(cache=responsive_cache, start=70_000, stop=70_100, sites=[SITE])
    full = predict_gam(cache=responsive_cache, fit=fits[1], index=np.arange(70_000, 70_100))
    part = predict_gam(cache=sliced, fit=fits[1], index=np.arange(100))
    assert np.abs(full - part).max() < 1e-12
