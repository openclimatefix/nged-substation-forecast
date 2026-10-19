"""Tests of the ensemble weather layer of the differentiable-physics series (`studies.dp_ens`).

The tests use the synthetic feature cache of `conftest.py`. Each ensemble run is built from the
cache's own ERA5 and CAMS values, so the layer with an identity calibration must reproduce the
convex model's forecast on the cache exactly, and any change to a value the forecast could not have
seen must leave the forecast unchanged.
"""

from dataclasses import replace
from datetime import UTC, datetime
from typing import Final

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")

from studies.dp_assimilation import (  # noqa: E402
    FilterHyperparameters,
    correction_at,
    run_kalman,
    usable_cutoff_positions,
)
from studies.dp_features import FeatureCache  # noqa: E402
from studies.dp_gam import GamFit, GamSpec, TrainingWindow, fit_gam, naive_utc, predict_gam  # noqa: E402
from studies.ens_members import ENSEMBLE_SIZE  # noqa: E402

from studies import dp_ens  # noqa: E402

SITE: Final[str] = "S1"
CUTS: Final[tuple[datetime, ...]] = (
    datetime(2024, 7, 1, tzinfo=UTC),
    datetime(2024, 8, 1, tzinfo=UTC),
)
ORIGIN: Final[datetime] = datetime(2024, 9, 10, 9, tzinfo=UTC)
N_RUN_TARGETS: Final[int] = 720


@pytest.fixture(scope="module")
def fits(dp_synthetic_cache: FeatureCache) -> list[GamFit]:
    """Fit the convex model at two monthly cuts."""
    return [
        fit_gam(
            cache=dp_synthetic_cache,
            site=SITE,
            specs=[GamSpec()],
            cut=cut,
            window=TrainingWindow(),
            season_years=range(2019, 2026),
        )
        for cut in CUTS
    ]


def _position(*, cache: FeatureCache, moment: datetime) -> int:
    """Return the grid position of the half-hour that ends at `moment`."""
    return int(np.searchsorted(cache.end_times, naive_utc(moment=moment)))


def _run_from_cache(
    *, cache: FeatureCache, origin_position: int, members_offset: float = 0.0
) -> dp_ens.EnsSiteRun:
    """Build a run whose every member equals the cache's ERA5 and CAMS values after the origin."""
    features = cache.sites[SITE]
    span = slice(origin_position + 1, origin_position + 1 + N_RUN_TARGETS)
    temperature = np.tile(features.temperature_c[span], (ENSEMBLE_SIZE, 1))
    temperature += members_offset * np.arange(ENSEMBLE_SIZE)[:, None]
    return dp_ens.EnsSiteRun(
        origin=cache.end_times[origin_position],
        origin_position=origin_position,
        temperature_c=temperature,
        shortwave_w_m2=np.tile(features.ghi_w_m2[span], (ENSEMBLE_SIZE, 1)),
    )


def _context(
    *, cache: FeatureCache, origin_position: int, days_back: int = 9
) -> dp_ens.WeatherContext:
    """Build an identity-calibrated context whose proxy comes from the cache's own weather."""
    runs = [
        _run_from_cache(cache=cache, origin_position=origin_position - 48 * back)
        for back in range(days_back + 1)
    ]
    return dp_ens.WeatherContext(
        site=SITE,
        calibration=dp_ens.Calibration.identity(),
        climatology=dp_ens.ghi_climatology(cache=cache, site=SITE, before=CUTS[0]),
        proxy=dp_ens.AnalysisProxy.build(runs=runs, n_rows=len(cache.end_times)),
    )


def test_the_ensemble_path_with_an_identity_calibration_reproduces_the_cache_forecast(
    dp_synthetic_cache: FeatureCache, fits: list[GamFit]
):
    cache = dp_synthetic_cache
    position = _position(cache=cache, moment=ORIGIN)
    context = _context(cache=cache, origin_position=position)
    run = _run_from_cache(cache=cache, origin_position=position)
    means, window = dp_ens.member_means(
        cache=cache, context=context, run=run, origin=ORIGIN, fits=fits, members=[0, 7]
    )
    expected = predict_gam(cache=cache, fit=fits[1], index=np.arange(window.start, window.stop))
    assert means.shape == (2, dp_ens.HORIZON_HALF_HOURS)
    assert np.abs(means - expected).max() < 1e-6


def test_members_that_all_equal_the_control_give_one_forecast_and_the_mixture_equals_it(
    dp_synthetic_cache: FeatureCache, fits: list[GamFit]
):
    cache = dp_synthetic_cache
    position = _position(cache=cache, moment=ORIGIN)
    context = _context(cache=cache, origin_position=position)
    run = _run_from_cache(cache=cache, origin_position=position)
    means, _ = dp_ens.member_means(
        cache=cache, context=context, run=run, origin=ORIGIN, fits=fits, members=range(0, 51, 10)
    )
    assert (means == means[0]).all()
    spread = np.linspace(-3.0, 3.0, 11)
    per_member = means[:, :3, None] + spread
    pooled = dp_ens.mixture_quantiles(member_quantiles=per_member.transpose(1, 0, 2))
    assert np.abs(pooled - per_member[0]).max() < 1e-12


def test_the_mixture_of_two_shifted_members_lies_between_them():
    low = np.linspace(0.0, 10.0, 11)
    pooled = dp_ens.mixture_quantiles(member_quantiles=np.stack([low, low + 100.0])[None])
    assert pooled[0, 0] < 10.0
    assert pooled[0, -1] > 100.0
    assert (np.diff(pooled[0]) >= 0).all()


def test_changing_a_value_a_forecast_could_not_have_seen_leaves_it_unchanged(
    dp_synthetic_cache: FeatureCache, fits: list[GamFit]
):
    cache = dp_synthetic_cache
    position = _position(cache=cache, moment=ORIGIN)
    context = _context(cache=cache, origin_position=position)
    run = _run_from_cache(cache=cache, origin_position=position)
    before, _ = dp_ens.member_means(
        cache=cache, context=context, run=run, origin=ORIGIN, fits=fits, members=[3]
    )
    features = cache.sites[SITE]
    after_origin = np.arange(len(cache.end_times)) > position
    changed = dp_ens.replace_site(
        cache=cache,
        site=SITE,
        y=np.where(after_origin, features.y + 40.0, features.y),
        temperature_c=np.where(after_origin, features.temperature_c - 15.0, features.temperature_c),
        smoothed_temperature_c=np.where(
            after_origin[:, None],
            features.smoothed_temperature_c - 15.0,
            features.smoothed_temperature_c,
        ),
        pv_unit_shape=np.where(after_origin, 0.0, features.pv_unit_shape),
        ghi_w_m2=np.where(after_origin, 0.0, features.ghi_w_m2),
    )
    later_run = _run_from_cache(cache=cache, origin_position=position + 48, members_offset=5.0)
    runs = [
        _run_from_cache(cache=cache, origin_position=position - 48 * back) for back in range(10)
    ]
    leaky_proxy = dp_ens.AnalysisProxy.build(runs=[*runs, later_run], n_rows=len(cache.end_times))
    unchanged_context = replace(context, proxy=leaky_proxy)
    after, _ = dp_ens.member_means(
        cache=changed, context=unchanged_context, run=run, origin=ORIGIN, fits=fits, members=[3]
    )
    assert (before == after).all()


def test_the_proxy_takes_each_half_hour_from_the_latest_run_published_before_it(
    dp_synthetic_cache: FeatureCache,
):
    cache = dp_synthetic_cache
    position = _position(cache=cache, moment=ORIGIN)
    runs = [
        _run_from_cache(cache=cache, origin_position=position - 48 * back, members_offset=1.0)
        for back in range(3)
    ]
    proxy = dp_ens.AnalysisProxy.build(runs=runs, n_rows=len(cache.end_times))
    older = position - 48
    assert proxy.run_origin_position[older] == position - 96
    assert proxy.run_origin_position[position] == older
    assert proxy.run_origin_position[position + 1] == position
    assert proxy.run_origin_position[older + 1] == older
    assert proxy.run_origin_position[position + 48] == position
    assert proxy.run_origin_position[position + 49] == -1
    proxy.assert_unseen_by(origin_position=position, through=position)
    with pytest.raises(ValueError, match="only up to its own origin"):
        proxy.assert_unseen_by(origin_position=position, through=position + 1)
    tampered = replace(
        proxy,
        run_origin_position=np.where(
            proxy.run_origin_position == older, position + 5, proxy.run_origin_position
        ),
    )
    with pytest.raises(ValueError, match="published after"):
        tampered.assert_unseen_by(origin_position=position, through=position)


def test_the_proxy_refuses_two_runs_with_one_origin(dp_synthetic_cache: FeatureCache):
    run = _run_from_cache(cache=dp_synthetic_cache, origin_position=50_000)
    with pytest.raises(ValueError, match="share an origin"):
        dp_ens.AnalysisProxy.build(runs=[run, run], n_rows=len(dp_synthetic_cache.end_times))


def test_a_forecast_uses_the_latest_cut_at_or_before_its_origin_and_refuses_a_later_fit(
    fits: list[GamFit], dp_synthetic_cache: FeatureCache
):
    cuts = [fit.cut for fit in fits]
    assert dp_ens.latest_cut_index(cuts=cuts, moment=datetime(2024, 6, 30, tzinfo=UTC)) == -1
    assert dp_ens.latest_cut_index(cuts=cuts, moment=CUTS[1]) == 1
    assert dp_ens.latest_cut_index(cuts=cuts, moment=datetime(2024, 7, 31, 23, tzinfo=UTC)) == 0
    with pytest.raises(ValueError, match="lookahead"):
        dp_ens.assert_cut_not_after(fit=fits[1], origin=datetime(2024, 7, 31, tzinfo=UTC))
    cache = dp_synthetic_cache
    early = datetime(2024, 6, 20, 9, tzinfo=UTC)
    position = _position(cache=cache, moment=early)
    run = _run_from_cache(cache=cache, origin_position=position)
    with pytest.raises(ValueError, match="no fit has a cut"):
        dp_ens.member_means(
            cache=cache,
            context=_context(cache=cache, origin_position=position),
            run=run,
            origin=early,
            fits=fits,
            members=[0],
        )


def test_a_forecast_in_the_month_before_the_next_cut_uses_the_earlier_fit(
    dp_synthetic_cache: FeatureCache, fits: list[GamFit]
):
    cache = dp_synthetic_cache
    origin = datetime(2024, 7, 20, 9, tzinfo=UTC)
    position = _position(cache=cache, moment=origin)
    run = _run_from_cache(cache=cache, origin_position=position)
    means, window = dp_ens.member_means(
        cache=cache,
        context=_context(cache=cache, origin_position=position),
        run=run,
        origin=origin,
        fits=fits,
        members=[0],
    )
    index = np.arange(window.start, window.stop)
    assert np.abs(means[0] - predict_gam(cache=cache, fit=fits[0], index=index)).max() < 1e-6
    assert np.abs(means[0] - predict_gam(cache=cache, fit=fits[1], index=index)).max() > 1e-3


def test_the_residual_stream_ignores_a_proxy_value_the_filter_could_not_have_seen(
    dp_synthetic_cache: FeatureCache, fits: list[GamFit]
):
    cache = dp_synthetic_cache
    position = _position(cache=cache, moment=ORIGIN)
    context = _context(cache=cache, origin_position=position)
    base = dp_ens.proxy_residual_stream(
        cache=dp_ens.proxy_cache(cache=cache, context=context), fits=fits, site=SITE
    )
    cutoff = int(
        usable_cutoff_positions(
            end_times=cache.end_times,
            initialisation=np.array([naive_utc(moment=ORIGIN)]),
            delay_hours=3.5,
        )[0]
    )
    runs = [
        _run_from_cache(cache=cache, origin_position=position - 48 * back) for back in range(10)
    ]
    # The newest run's control member differs, and it supplies only half-hours after the origin.
    last = runs[0]
    shifted = replace(last, temperature_c=last.temperature_c + 9.0)
    shifted_proxy = dp_ens.AnalysisProxy.build(
        runs=[shifted, *runs[1:]], n_rows=len(cache.end_times)
    )
    shifted_stream = dp_ens.proxy_residual_stream(
        cache=dp_ens.proxy_cache(cache=cache, context=replace(context, proxy=shifted_proxy)),
        fits=fits,
        site=SITE,
    )
    hyper = FilterHyperparameters(log_q=-3.0, logit_phi=4.0, log_sigma_z=-1.0, log_sigma_eps=0.0)
    band = cache.calendar.band
    target = np.arange(position + 1, position + 49)
    kinds = []
    for stream in (base, shifted_stream):
        trace = run_kalman(residual=stream, band=band, hyper=hyper)
        kinds.append(
            correction_at(
                trace=trace,
                cutoff=np.full(len(target), cutoff),
                target=target,
                band=band[target],
            ).total
        )
    assert np.isfinite(base[cutoff - 5 : cutoff]).any()
    assert np.array_equal(base[: position - 47], shifted_stream[: position - 47], equal_nan=True)
    assert np.array_equal(kinds[0], kinds[1])
    assert not np.array_equal(base, shifted_stream, equal_nan=True)


def test_a_known_bias_is_recovered_by_the_calibration_within_fifteen_percent():
    generator = np.random.default_rng(0)
    n = 40_000
    season = generator.integers(0, 4, n)
    lead = generator.integers(0, 4, n)
    band = generator.integers(0, 4, n)
    truth_t = generator.normal(10.0, 5.0, n)
    bias = 1.5 + 1.5 * lead / 3.0
    ens_t = truth_t - bias + generator.normal(0.0, 0.3, n)
    truth_g = generator.uniform(50.0, 600.0, n)
    ens_g = truth_g / 0.8 * generator.normal(1.0, 0.02, n)
    calibration = dp_ens.Calibration.fit(
        season=season,
        lead_band=lead,
        hour_band=band,
        ens_mean_temperature_c=ens_t,
        target_temperature_c=truth_t,
        ens_mean_ghi=ens_g,
        target_ghi=truth_g,
    )
    for lead_index in range(4):
        expected_shift = 1.5 + 1.5 * lead_index / 3.0
        assert np.abs(calibration.temperature_shift_c[:, lead_index] - expected_shift).max() < (
            0.15 * expected_shift
        )
    assert np.abs(calibration.irradiance_ratio - 0.8).max() < 0.15 * 0.8


def test_a_calibration_cell_with_few_rows_keeps_the_identity():
    calibration = dp_ens.Calibration.fit(
        season=np.zeros(5, dtype=int),
        lead_band=np.zeros(5, dtype=int),
        hour_band=np.zeros(5, dtype=int),
        ens_mean_temperature_c=np.zeros(5),
        target_temperature_c=np.full(5, 9.0),
        ens_mean_ghi=np.full(5, 100.0),
        target_ghi=np.full(5, 300.0),
    )
    assert (calibration.temperature_shift_c == 0.0).all()
    assert (calibration.irradiance_ratio == 1.0).all()


def test_the_ratio_cannot_lift_irradiance_above_clear_sky_and_a_zero_shrink_gives_the_climatology():
    ratio = np.full((4, 4, 4), 2.0)
    calibration = dp_ens.Calibration(
        temperature_shift_c=np.zeros((4, 4, 4)),
        irradiance_ratio=ratio,
        solar_shrink=np.array([1.0, 1.0, 1.0, 0.0]),
    )
    ghi = np.array([300.0, 300.0])
    clear = np.array([400.0, 400.0])
    climatology = np.array([120.0, 120.0])
    zero = np.zeros(2, dtype=np.int64)
    lifted = calibration.apply_irradiance(
        ghi=ghi,
        clear_sky_ghi=clear,
        climatology_ghi=climatology,
        season=zero,
        lead_band=zero,
        hour_band=zero,
    )
    assert lifted.tolist() == [400.0, 400.0]
    last = np.full(2, 3, dtype=np.int64)
    shrunk = calibration.apply_irradiance(
        ghi=ghi,
        clear_sky_ghi=clear,
        climatology_ghi=climatology,
        season=zero,
        lead_band=last,
        hour_band=zero,
    )
    assert shrunk.tolist() == [120.0, 120.0]


def test_trailing_means_match_polars_and_are_nan_when_the_window_is_incomplete():
    values = np.arange(30, dtype=float)
    values[12] = np.nan
    ours = dp_ens.trailing_means(values=values, window=6)
    reference = (
        pl.Series(values).fill_nan(None).rolling_mean(window_size=6, min_samples=6).to_numpy()
    )
    assert np.allclose(ours, reference, equal_nan=True)
    assert np.isnan(ours[12:18]).all()
    assert np.isfinite(ours[18])


def test_the_lead_bands_follow_the_calibration_edges():
    leads = np.array([0.5, 24.0, 24.5, 48.0, 48.5, 144.0, 144.5, 168.0, 170.0])
    bands = dp_ens.lead_band_of_hours(lead_hours=leads)
    assert bands.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 3]


def test_seasons_group_december_to_february_together():
    assert dp_ens.season_of_months(months=np.arange(1, 13)).tolist() == [
        0,
        0,
        1,
        1,
        1,
        2,
        2,
        2,
        3,
        3,
        3,
        0,
    ]


def test_a_block_of_hours_must_start_half_past(dp_synthetic_cache: FeatureCache):
    ends = dp_synthetic_cache.end_times[100:102]
    odd_start = dp_synthetic_cache.end_times[101:103]
    for block in (ends, odd_start):
        minute = dp_ens._minute(block[0])
        if minute != 30:
            with pytest.raises(ValueError, match="30 minutes past"):
                dp_ens.unit_pv_shape_and_ghi(
                    end_times=block, ghi_w_m2=np.ones(2), latitude=53.0, longitude=0.0
                )


def test_a_window_slice_predicts_what_the_full_cache_predicts(
    dp_synthetic_cache: FeatureCache, fits: list[GamFit]
):
    rows = np.arange(70_000, 70_100)
    sliced = dp_ens.slice_cache(cache=dp_synthetic_cache, start=70_000, stop=70_100, sites=[SITE])
    full = predict_gam(cache=dp_synthetic_cache, fit=fits[1], index=rows)
    part = predict_gam(cache=sliced, fit=fits[1], index=np.arange(100))
    assert np.abs(full - part).max() < 1e-12


def test_read_site_run_checks_members_and_times(dp_synthetic_cache: FeatureCache):
    cache = dp_synthetic_cache
    origin = ORIGIN
    stamps = pl.datetime_range(
        datetime(2024, 9, 10, 9, 30, tzinfo=UTC),
        datetime(2024, 9, 10, 12, 0, tzinfo=UTC),
        interval="30m",
        eager=True,
    )
    frame = pl.concat(
        [
            pl.DataFrame(
                {
                    "ensemble_member": pl.Series([member] * len(stamps), dtype=pl.Int8),
                    "valid_time": stamps,
                    "temperature_2m": pl.Series(
                        np.full(len(stamps), 10.0 + member), dtype=pl.Float32
                    ),
                    "shortwave": pl.Series(np.full(len(stamps), 50.0), dtype=pl.Float32),
                }
            )
            for member in range(ENSEMBLE_SIZE)
        ]
    )
    run = dp_ens.read_site_run(frame=frame, origin=origin, end_times=cache.end_times)
    assert run.temperature_c.shape == (ENSEMBLE_SIZE, len(stamps))
    assert run.temperature_c[5, 0] == 15.0
    with pytest.raises(ValueError, match="not 0 to 50"):
        dp_ens.read_site_run(
            frame=frame.filter(pl.col("ensemble_member") < 50),
            origin=origin,
            end_times=cache.end_times,
        )
    with pytest.raises(ValueError, match="starting 30 minutes after the origin"):
        dp_ens.read_site_run(
            frame=frame, origin=datetime(2024, 9, 10, 8, tzinfo=UTC), end_times=cache.end_times
        )


def test_the_margin_frame_scales_points_to_megawatts_and_names_the_fit_it_used(
    dp_synthetic_cache: FeatureCache, fits: list[GamFit]
):
    cache = dp_synthetic_cache
    position = _position(cache=cache, moment=ORIGIN)
    context = _context(cache=cache, origin_position=position)
    run = _run_from_cache(cache=cache, origin_position=position)
    frame = dp_ens.margin_frame(cache=cache, context=context, run=run, origin=ORIGIN, fits=fits)
    index = np.arange(position + 1, position + 1 + dp_ens.HORIZON_HALF_HOURS)
    expected = predict_gam(cache=cache, fit=fits[1], index=index)
    capacity = cache.sites[SITE].table.effective_capacity_mw
    assert frame.height == dp_ens.HORIZON_HALF_HOURS
    assert np.abs(frame["margin_mw"].to_numpy() - expected * capacity / 100.0).max() < 1e-6
    assert set(frame["fit_cut"].to_list()) == {CUTS[1]}
    assert frame["valid_time"][0] == datetime(2024, 9, 10, 9, 30, tzinfo=UTC)
    assert frame["power_fcst_init_time"].n_unique() == 1
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
    lead = dp_ens.lead_band_of_hours(lead_hours=lead_hours)
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


def test_the_frame_and_cache_observed_power_must_agree(dp_synthetic_cache: FeatureCache):
    cache = dp_synthetic_cache
    times = cache.end_times[60_000:60_010]
    points = cache.sites[SITE].y[60_000:60_010]
    assert (
        dp_ens.assert_actual_matches_cache(
            valid_time=times, actual_points=points, cache=cache, site=SITE
        )
        == 10
    )
    with pytest.raises(ValueError, match="differs"):
        dp_ens.assert_actual_matches_cache(
            valid_time=times, actual_points=points * 1.001, cache=cache, site=SITE
        )
    with pytest.raises(ValueError, match="no row"):
        dp_ens.assert_actual_matches_cache(
            valid_time=times + np.timedelta64(7, "m"), actual_points=points, cache=cache, site=SITE
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


def test_a_forecast_reads_the_filter_only_up_to_the_telemetry_cutoff(
    dp_synthetic_cache: FeatureCache, fits: list[GamFit]
):
    cache = dp_synthetic_cache
    position = _position(cache=cache, moment=ORIGIN)
    context = _context(cache=cache, origin_position=position)
    run = _run_from_cache(cache=cache, origin_position=position)
    cutoff = int(
        usable_cutoff_positions(
            end_times=cache.end_times,
            initialisation=np.array([naive_utc(moment=ORIGIN)]),
            delay_hours=3.5,
        )[0]
    )
    assert cutoff == position - 7
    stream = np.full(len(cache.end_times), np.nan)
    stream[position - 600 : position + 400] = np.sin(np.arange(1000) / 30.0) * 3.0 + 1.0
    spread = _spread()
    base = dp_ens.forecast_run(
        cache=cache,
        context=context,
        run=run,
        origin=ORIGIN,
        fits=fits,
        spread=spread,
        trace=_trace(cache=cache, stream=stream),
    )
    after = stream.copy()
    after[cutoff + 1 :] += 25.0
    unchanged = dp_ens.forecast_run(
        cache=cache,
        context=context,
        run=run,
        origin=ORIGIN,
        fits=fits,
        spread=spread,
        trace=_trace(cache=cache, stream=after),
    )
    assert np.array_equal(base.quantiles, unchanged.quantiles)
    at_cutoff = stream.copy()
    at_cutoff[cutoff - 3 : cutoff + 1] += 25.0
    moved = dp_ens.forecast_run(
        cache=cache,
        context=context,
        run=run,
        origin=ORIGIN,
        fits=fits,
        spread=spread,
        trace=_trace(cache=cache, stream=at_cutoff),
    )
    assert np.abs(moved.quantiles - base.quantiles).max() > 1.0
    assert np.abs(base.level).max() > 0.1


def test_a_forecast_without_a_filter_has_no_correction_and_the_mixture_brackets_the_mean(
    dp_synthetic_cache: FeatureCache, fits: list[GamFit]
):
    cache = dp_synthetic_cache
    position = _position(cache=cache, moment=ORIGIN)
    context = _context(cache=cache, origin_position=position)
    run = _run_from_cache(cache=cache, origin_position=position, members_offset=0.05)
    forecast = dp_ens.forecast_run(
        cache=cache, context=context, run=run, origin=ORIGIN, fits=fits, spread=_spread()
    )
    assert (forecast.level == 0.0).all()
    assert (forecast.anomaly == 0.0).all()
    assert forecast.quantiles.shape == (dp_ens.HORIZON_HALF_HOURS, 11)
    assert (np.diff(forecast.quantiles, axis=1) >= 0).all()
    assert (forecast.quantiles[:, 0] < forecast.mean).all()
    assert (forecast.quantiles[:, -1] > forecast.mean).all()
