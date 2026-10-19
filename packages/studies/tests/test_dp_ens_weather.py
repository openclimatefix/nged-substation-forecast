"""Tests of the ensemble weather layer of the DP series (`studies.dp_ens_weather`).

These tests need no torch, so they run in CI. The synthetic feature cache of `conftest.py` supplies
the grid. Each ensemble run is built from the cache's own temperature and irradiance, shifted or
scaled where a test needs the ensemble to differ from the training weather.
"""

from dataclasses import replace
from datetime import UTC, datetime
from typing import Final

import numpy as np
import polars as pl
import pytest
from studies.dp_features import FeatureCache, naive_utc
from studies.ens_members import ENSEMBLE_SIZE, mixture_quantiles
from studies.pv_physics import half_hour_sun_and_sky, plant_power_mw

from studies import dp_ens_weather as weather

SITE: Final[str] = "S1"
ORIGIN: Final[datetime] = datetime(2024, 9, 10, 9, tzinfo=UTC)
N_RUN_TARGETS: Final[int] = 720
MIDDAY_W_M2: Final[float] = 300.0


def _position(*, cache: FeatureCache, moment: datetime) -> int:
    return int(np.searchsorted(cache.end_times, naive_utc(moment=moment)))


def _run(
    *,
    cache: FeatureCache,
    origin_position: int,
    temperature_offset: float = 0.0,
    shortwave_scale: float = 1.0,
) -> weather.EnsSiteRun:
    """Build a run from the cache's ERA5 and CAMS values after the origin, shifted or scaled."""
    features = cache.sites[SITE]
    span = slice(origin_position + 1, origin_position + 1 + N_RUN_TARGETS)
    return weather.EnsSiteRun(
        origin_position=origin_position,
        temperature_c=np.tile(
            features.temperature_c[span] + temperature_offset, (ENSEMBLE_SIZE, 1)
        ),
        shortwave_w_m2=np.tile(features.ghi_w_m2[span] * shortwave_scale, (ENSEMBLE_SIZE, 1)),
    )


def _context(
    *,
    cache: FeatureCache,
    origin_position: int,
    days_back: int = 20,
    calibration: weather.Calibration | None = None,
    temperature_offset: float = 0.0,
) -> weather.WeatherContext:
    runs = [
        _run(
            cache=cache,
            origin_position=origin_position - 48 * back,
            temperature_offset=temperature_offset,
        )
        for back in range(days_back + 1)
    ]
    return weather.WeatherContext(
        site=SITE,
        calibration=weather.Calibration.identity() if calibration is None else calibration,
        climatology=weather.ghi_climatology(
            cache=cache, site=SITE, before=datetime(2024, 7, 1, tzinfo=UTC)
        ),
        proxy=weather.AnalysisProxy.build(runs=runs, n_rows=len(cache.end_times)),
    )


def test_the_mixture_of_identical_members_equals_one_member_and_of_shifted_members_spans_them():
    levels = np.linspace(-3.0, 3.0, 11)
    same = np.tile(levels, (4, 51, 1))
    assert np.abs(mixture_quantiles(member_quantiles=same) - levels).max() < 1e-12
    low = np.linspace(0.0, 10.0, 11)
    pooled = mixture_quantiles(member_quantiles=np.stack([low, low + 100.0])[None])
    assert pooled[0, 0] < 10.0
    assert pooled[0, -1] > 100.0
    assert (np.diff(pooled[0]) >= 0).all()


def test_the_proxy_takes_each_half_hour_from_the_latest_run_published_before_it(
    dp_synthetic_cache: FeatureCache,
):
    cache = dp_synthetic_cache
    position = _position(cache=cache, moment=ORIGIN)
    runs = [_run(cache=cache, origin_position=position - 48 * back) for back in range(3)]
    proxy = weather.AnalysisProxy.build(runs=runs, n_rows=len(cache.end_times))
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


def test_the_proxy_refuses_two_runs_with_one_origin_and_reports_an_uncovered_gap(
    dp_synthetic_cache: FeatureCache,
):
    cache = dp_synthetic_cache
    run = _run(cache=cache, origin_position=50_000)
    with pytest.raises(ValueError, match="share an origin"):
        weather.AnalysisProxy.build(runs=[run, run], n_rows=len(cache.end_times))
    runs = [_run(cache=cache, origin_position=50_000 + 48 * k) for k in (0, 1, 3)]
    proxy = weather.AnalysisProxy.build(runs=runs, n_rows=len(cache.end_times))
    proxy.assert_covers(first=50_001, last=50_000 + 96)
    with pytest.raises(ValueError, match="uncovered"):
        proxy.assert_covers(first=50_001, last=50_000 + 150)


def test_a_known_bias_is_recovered_by_the_calibration_within_fifteen_percent():
    generator = np.random.default_rng(0)
    n = 40_000
    season = generator.integers(0, 4, n)
    lead = generator.integers(0, 4, n)
    band = generator.integers(0, 4, n)
    truth_t = generator.normal(10.0, 5.0, n)
    ens_t = truth_t - (1.5 + 1.5 * lead / 3.0) + generator.normal(0.0, 0.3, n)
    truth_g = generator.uniform(50.0, 600.0, n)
    ens_g = truth_g / 0.8 * generator.normal(1.0, 0.02, n)
    calibration = weather.Calibration.fit(
        origin_time=np.full(n, np.datetime64("2024-07-15T12:00", "us")),
        allowed_months={(2024, 7)},
        season=season,
        lead_band=lead,
        hour_band=band,
        ens_mean_temperature_c=ens_t,
        target_temperature_c=truth_t,
        ens_mean_ghi=ens_g,
        target_ghi=truth_g,
    )
    for lead_index in range(4):
        expected = 1.5 + 1.5 * lead_index / 3.0
        assert np.abs(calibration.temperature_shift_c[:, lead_index] - expected).max() < (
            0.15 * expected
        )
    assert np.abs(calibration.irradiance_ratio - 0.8).max() < 0.15 * 0.8


def test_a_calibration_cell_with_few_rows_keeps_the_identity():
    calibration = weather.Calibration.fit(
        origin_time=np.full(5, np.datetime64("2024-07-15T12:00", "us")),
        allowed_months={(2024, 7)},
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


def test_the_calibration_refuses_rows_from_a_month_that_is_not_a_tuning_month():
    times = np.array(["2024-07-15T12:00", "2024-08-15T12:00"], dtype="datetime64[us]")
    with pytest.raises(ValueError, match="not allowed"):
        weather.assert_months_allowed(valid_time=times, allowed={(2024, 7)})
    weather.assert_months_allowed(valid_time=times, allowed={(2024, 7), (2024, 8)})


def test_each_calibration_cell_is_applied_to_its_own_season_lead_band_and_hour_band():
    shift = np.arange(64, dtype=float).reshape(4, 4, 4)
    calibration = replace(weather.Calibration.identity(), temperature_shift_c=shift)
    season = np.array([0, 1, 2, 3, 3])
    lead = np.array([0, 1, 2, 3, 0])
    band = np.array([3, 2, 1, 0, 2])
    out = calibration.apply_temperature(
        temperature_c=np.full(5, 10.0), season=season, lead_band=lead, hour_band=band
    )
    assert out.tolist() == [
        10.0 + 16 * s + 4 * lead_i + b for s, lead_i, b in zip(season, lead, band, strict=True)
    ]


def test_the_ratio_cannot_lift_irradiance_above_clear_sky_and_a_zero_shrink_gives_the_climatology():
    calibration = replace(
        weather.Calibration.identity(),
        irradiance_ratio=np.full((4, 4, 4), 2.0),
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
    shrunk = calibration.apply_irradiance(
        ghi=ghi,
        clear_sky_ghi=clear,
        climatology_ghi=climatology,
        season=zero,
        lead_band=np.full(2, 3, dtype=np.int64),
        hour_band=zero,
    )
    assert shrunk.tolist() == [120.0, 120.0]
    unshrunk = calibration.apply_irradiance(
        ghi=ghi,
        clear_sky_ghi=clear,
        climatology_ghi=climatology,
        season=zero,
        lead_band=np.full(2, 3, dtype=np.int64),
        hour_band=zero,
        shrink=False,
    )
    assert unshrunk.tolist() == [400.0, 400.0]


def test_the_solar_shrink_needs_one_value_per_lead_band():
    with pytest.raises(ValueError, match="one value per lead band"):
        weather.Calibration.identity().with_solar_shrink(solar_shrink=[1.0])


def test_the_climatology_is_filed_under_the_month_and_the_half_hour_of_its_end(
    dp_synthetic_cache: FeatureCache,
):
    cache = dp_synthetic_cache
    stamps = pl.Series(cache.end_times).dt.replace_time_zone("UTC")
    month = stamps.dt.month().to_numpy().astype(np.int64)
    hour = stamps.dt.hour().to_numpy().astype(np.int64)
    minute = stamps.dt.minute().to_numpy().astype(np.int64)
    slot = (hour * 2 + minute // 30 - 1) % 48
    coded = (month * 100 + slot).astype(float)
    late = cache.end_times > np.datetime64("2024-07-01", "us")
    poisoned = np.where(late, 99_999.0, coded)
    marked = weather.replace_site(cache=cache, site=SITE, ghi_w_m2=poisoned)
    table = weather.ghi_climatology(
        cache=marked, site=SITE, before=datetime(2024, 7, 1, tzinfo=UTC)
    )
    expected = np.array([[(m + 1) * 100 + s for s in range(48)] for m in range(12)], dtype=float)
    assert np.array_equal(table, expected)
    rows = np.arange(60_000, 60_100)
    assert np.array_equal(
        weather.climatology_at(table=table, end_times=cache.end_times[rows]), coded[rows]
    )


def test_trailing_means_match_polars_and_are_nan_when_the_window_is_incomplete():
    values = np.arange(30, dtype=float)
    values[12] = np.nan
    ours = weather.trailing_means(values=values, window=6)
    reference = pl.Series(values).fill_nan(None).rolling_mean(window_size=6, min_samples=6)
    assert np.allclose(ours, reference.to_numpy(), equal_nan=True)
    assert np.isnan(ours[12:18]).all()
    assert np.isfinite(ours[18])


def test_the_lead_bands_follow_the_calibration_edges_and_the_seasons_group_the_winter_months():
    leads = np.array([0.5, 24.0, 24.5, 48.0, 48.5, 144.0, 144.5, 168.0, 170.0])
    assert weather.lead_band_of_hours(lead_hours=leads).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 3]
    seasons = weather.season_of_months(months=np.arange(1, 13)).tolist()
    assert seasons == [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0]


def test_the_sun_geometry_needs_whole_hours_that_start_half_past(dp_synthetic_cache: FeatureCache):
    ends = dp_synthetic_cache.end_times
    at_half_past = next(i for i in range(60_000, 60_010) if weather._minute(ends[i]) == 30)
    with pytest.raises(ValueError, match="30 minutes past"):
        weather.SunGeometry.build(
            end_times=ends[at_half_past + 1 : at_half_past + 3], latitude=53.0, longitude=0.0
        )
    with pytest.raises(ValueError, match="whole hours"):
        weather.SunGeometry.build(
            end_times=ends[at_half_past : at_half_past + 3], latitude=53.0, longitude=0.0
        )
    sun = weather.SunGeometry.build(
        end_times=ends[at_half_past : at_half_past + 4], latitude=53.0, longitude=0.0
    )
    assert np.isfinite(sun.zenith_deg).all()


def test_the_batched_pv_shape_equals_the_pipeline_of_the_cache_and_blanks_a_missing_hour():
    hours = pl.datetime_range(
        datetime(2024, 6, 21, 1, tzinfo=UTC),
        datetime(2024, 6, 21, 23, tzinfo=UTC),
        interval="1h",
        eager=True,
    )
    ghi_hourly = np.linspace(100.0, 700.0, hours.len())
    sky = half_hour_sun_and_sky(
        hourly=pl.DataFrame({"time": hours, "ghi_w_m2": ghi_hourly}), latitude=53.0, longitude=0.0
    )
    from studies.dp_features import FLEET

    reference = plant_power_mw(sky=sky, parameters=FLEET)
    ends = sky.half_hour_end_time
    sun = weather.SunGeometry.build(end_times=ends, latitude=53.0, longitude=0.0)
    half_hourly = np.repeat(ghi_hourly, 2)
    shape, flux = sun.pv_shape_and_ghi(ghi_w_m2=np.stack([half_hourly, half_hourly * 0.5]))
    assert np.abs(shape[0] - reference).max() < 1e-9
    assert np.abs(flux[0] - sky.ghi_w_m2).max() < 1e-9
    assert np.abs(flux[1] - 0.5 * sky.ghi_w_m2).max() < 1e-9
    broken = half_hourly.copy()
    broken[10] = np.nan
    blanked, _ = sun.pv_shape_and_ghi(ghi_w_m2=broken[None, :])
    assert np.isnan(blanked[0, 10])
    assert np.isnan(blanked[0, 11])
    assert np.isfinite(blanked[0, 12])


def test_the_clear_sky_peaks_at_solar_noon_and_is_zero_at_night():
    ends = (
        pl.datetime_range(
            datetime(2024, 6, 21, 0, 30, tzinfo=UTC),
            datetime(2024, 6, 21, 23, 0, tzinfo=UTC),
            interval="30m",
            eager=True,
        )
        .dt.replace_time_zone(None)
        .to_numpy()
    )
    sun = weather.SunGeometry.build(end_times=ends, latitude=53.0, longitude=0.0)
    peak = ends[int(np.argmax(sun.clear_sky_w_m2))]
    assert abs((peak - np.datetime64("2024-06-21T12:00", "us")) / np.timedelta64(1, "m")) <= 45
    assert 800.0 < sun.clear_sky_w_m2.max() < 1100.0
    assert sun.clear_sky_w_m2[0] == 0.0


def test_a_run_window_refuses_a_short_window_a_missing_past_and_a_proxy_from_a_later_run(
    dp_synthetic_cache: FeatureCache,
):
    cache = dp_synthetic_cache
    position = _position(cache=cache, moment=ORIGIN)
    context = _context(cache=cache, origin_position=position)
    short = replace(
        _run(cache=cache, origin_position=position),
        temperature_c=np.zeros((ENSEMBLE_SIZE, 20)),
        shortwave_w_m2=np.zeros((ENSEMBLE_SIZE, 20)),
    )
    with pytest.raises(ValueError, match="shorter than a day"):
        weather.RunWindow.build(cache=cache, context=context, run=short)
    sparse = _context(cache=cache, origin_position=position, days_back=2)
    with pytest.raises(ValueError, match="lacks the past week"):
        weather.RunWindow.build(
            cache=cache, context=sparse, run=_run(cache=cache, origin_position=position)
        )
    tampered = replace(
        context,
        proxy=replace(
            context.proxy,
            run_origin_position=np.where(
                context.proxy.run_origin_position == position - 48,
                position + 1,
                context.proxy.run_origin_position,
            ),
        ),
    )
    with pytest.raises(ValueError, match="published after"):
        weather.RunWindow.build(
            cache=cache, context=tampered, run=_run(cache=cache, origin_position=position)
        )


def test_the_driven_window_indexes_the_calibration_by_season_lead_band_and_hour_band(
    dp_synthetic_cache: FeatureCache,
):
    cache = dp_synthetic_cache
    position = _position(cache=cache, moment=ORIGIN)
    shift = np.arange(64, dtype=float).reshape(4, 4, 4)
    ratio = np.broadcast_to(np.array([1.0, 1.5, 2.0, 0.5])[None, :, None], (4, 4, 4)).copy()
    calibration = replace(
        weather.Calibration.identity(), temperature_shift_c=shift, irradiance_ratio=ratio
    )
    run = replace(
        _run(cache=cache, origin_position=position),
        temperature_c=np.full((ENSEMBLE_SIZE, N_RUN_TARGETS), 10.0),
        shortwave_w_m2=np.full((ENSEMBLE_SIZE, N_RUN_TARGETS), 50.0),
    )
    context = _context(cache=cache, origin_position=position, calibration=calibration)
    window = weather.RunWindow.build(cache=cache, context=context, run=run)
    driven = window.driven_caches(context=context, run=run, members=[3])[0]
    features = driven.sites[SITE]
    local = (
        pl.Series(driven.end_times)
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone("Europe/London")
    )
    start_local = local.dt.offset_by("-30m")
    season = ((start_local.dt.month().to_numpy() % 12) // 3).astype(int)
    lead = np.minimum((np.arange(1, len(season) + 1) / 2.0 - 1e-9) // 1, 0).astype(int)
    hours = np.arange(1, len(season) + 1) / 2.0
    lead = np.searchsorted([24.0, 48.0, 144.0, 168.0], hours, side="left").clip(max=3)
    band = driven.calendar.band.astype(int)
    assert np.allclose(features.temperature_c, 10.0 + shift[season, lead, band])
    pair_mean = features.ghi_w_m2.reshape(-1, 2).mean(axis=1)
    midday = window.sun.clear_sky_w_m2.reshape(-1, 2).min(axis=1) > MIDDAY_W_M2
    assert midday.any()
    expected = (50.0 * ratio[0, lead, 0])[::2]
    assert np.allclose(pair_mean[midday], expected[midday], rtol=1e-6)


def test_a_zero_shrink_gives_the_climatology_hour_by_hour_and_a_unit_shrink_keeps_the_ensemble(
    dp_synthetic_cache: FeatureCache,
):
    cache = dp_synthetic_cache
    position = _position(cache=cache, moment=ORIGIN)
    run = replace(
        _run(cache=cache, origin_position=position),
        shortwave_w_m2=np.full((ENSEMBLE_SIZE, N_RUN_TARGETS), 50.0),
    )
    kept = {}
    for shrink in (0.0, 1.0):
        calibration = weather.Calibration.identity().with_solar_shrink(solar_shrink=[shrink] * 4)
        context = _context(cache=cache, origin_position=position, calibration=calibration)
        window = weather.RunWindow.build(cache=cache, context=context, run=run)
        driven = window.driven_caches(context=context, run=run, members=[0])[0]
        kept[shrink] = (
            driven.sites[SITE].ghi_w_m2.reshape(-1, 2).mean(axis=1),
            window.climatology_w_m2.reshape(-1, 2).mean(axis=1),
            window.sun.clear_sky_w_m2.reshape(-1, 2).min(axis=1),
        )
    pair_mean, climatology, clear = kept[0.0]
    daylight = clear > MIDDAY_W_M2
    ceiling = np.maximum(clear, 50.0)
    assert np.allclose(pair_mean[daylight], np.minimum(climatology, ceiling)[daylight], rtol=1e-6)
    assert np.allclose(kept[1.0][0][daylight], 50.0, rtol=1e-6)


def test_the_proxy_cache_uses_the_first_lead_band_and_does_not_shrink(
    dp_synthetic_cache: FeatureCache,
):
    cache = dp_synthetic_cache
    position = _position(cache=cache, moment=ORIGIN)
    shift = np.zeros((4, 4, 4))
    shift[:, 0, :] = 1.0
    shift[:, 3, :] = 9.0
    calibration = replace(
        weather.Calibration.identity(),
        temperature_shift_c=shift,
        solar_shrink=np.zeros(4),
    )
    context = _context(cache=cache, origin_position=position, calibration=calibration)
    proxied = weather.proxy_cache(cache=cache, context=context).sites[SITE]
    covered = np.isfinite(context.proxy.temperature_c)
    assert np.allclose(proxied.temperature_c[covered], context.proxy.temperature_c[covered] + 1.0)
    covered_rows = np.flatnonzero(covered)
    first = next(r for r in covered_rows if weather._minute(cache.end_times[r]) == 30)
    count = (covered_rows[-1] - first + 1) // 2 * 2
    span = slice(first, first + count)
    pairs = proxied.ghi_w_m2[span].reshape(-1, 2).mean(axis=1)
    raw = cache.sites[SITE].ghi_w_m2[span].reshape(-1, 2).mean(axis=1)
    midday = raw > MIDDAY_W_M2
    assert midday.any()
    assert np.allclose(pairs[midday], raw[midday], rtol=1e-6)
    assert np.isnan(proxied.temperature_c[~covered]).all()


def test_a_member_with_a_missing_value_is_dropped_and_a_missing_control_refuses_the_run(
    dp_synthetic_cache: FeatureCache,
):
    cache = dp_synthetic_cache
    run = _run(cache=cache, origin_position=60_000)
    assert run.usable_members() == list(range(ENSEMBLE_SIZE))
    holed = run.shortwave_w_m2.copy()
    holed[36, 5] = np.nan
    assert replace(run, shortwave_w_m2=holed).usable_members() == [
        m for m in range(ENSEMBLE_SIZE) if m != 36
    ]
    holed[0, 5] = np.nan
    with pytest.raises(ValueError, match="control member"):
        replace(run, shortwave_w_m2=holed).usable_members()


def _frame(*, members: range, stamps: pl.Series) -> pl.DataFrame:
    return pl.concat(
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
            for member in members
        ]
    )


def test_read_site_run_checks_members_valid_times_and_the_origin(dp_synthetic_cache: FeatureCache):
    cache = dp_synthetic_cache
    stamps = pl.datetime_range(
        datetime(2024, 9, 10, 9, 30, tzinfo=UTC),
        datetime(2024, 9, 10, 12, 0, tzinfo=UTC),
        interval="30m",
        eager=True,
    )
    frame = _frame(members=range(ENSEMBLE_SIZE), stamps=stamps)
    run = weather.read_site_run(frame=frame, origin=ORIGIN, end_times=cache.end_times)
    assert run.temperature_c.shape == (ENSEMBLE_SIZE, len(stamps))
    assert run.temperature_c[5, 0] == 15.0
    assert run.origin_position == _position(cache=cache, moment=ORIGIN)
    with pytest.raises(ValueError, match="not 0 to 50"):
        weather.read_site_run(
            frame=frame.filter(pl.col("ensemble_member") < 50),
            origin=ORIGIN,
            end_times=cache.end_times,
        )
    with pytest.raises(ValueError, match="starting 30 minutes after the origin"):
        weather.read_site_run(
            frame=frame, origin=datetime(2024, 9, 10, 8, tzinfo=UTC), end_times=cache.end_times
        )
    shifted = frame.with_columns(
        valid_time=pl.when(pl.col("ensemble_member") == 7)
        .then(pl.col("valid_time") + pl.duration(minutes=30))
        .otherwise(pl.col("valid_time"))
    )
    with pytest.raises(ValueError, match="do not share"):
        weather.read_site_run(frame=shifted, origin=ORIGIN, end_times=cache.end_times)
    off_grid = datetime(2024, 9, 10, 9, 15, tzinfo=UTC)
    off_stamps = pl.datetime_range(
        datetime(2024, 9, 10, 9, 45, tzinfo=UTC),
        datetime(2024, 9, 10, 12, 15, tzinfo=UTC),
        interval="30m",
        eager=True,
    )
    with pytest.raises(ValueError, match="not a half-hour end"):
        weather.read_site_run(
            frame=_frame(members=range(ENSEMBLE_SIZE), stamps=off_stamps),
            origin=off_grid,
            end_times=cache.end_times,
        )


def test_the_frame_and_cache_observed_power_must_agree(dp_synthetic_cache: FeatureCache):
    cache = dp_synthetic_cache
    times = cache.end_times[60_000:60_010]
    points = cache.sites[SITE].y[60_000:60_010]
    shared = weather.assert_actual_matches_cache(
        valid_time=times, actual_points=points, cache=cache, site=SITE
    )
    assert shared == 10
    with pytest.raises(ValueError, match="differs"):
        weather.assert_actual_matches_cache(
            valid_time=times, actual_points=points * 1.001, cache=cache, site=SITE
        )
    with pytest.raises(ValueError, match="no row"):
        weather.assert_actual_matches_cache(
            valid_time=times + np.timedelta64(7, "m"),
            actual_points=points,
            cache=cache,
            site=SITE,
        )


def test_a_cache_slice_holds_views_of_the_same_rows(dp_synthetic_cache: FeatureCache):
    sliced = weather.slice_cache(cache=dp_synthetic_cache, start=70_000, stop=70_100, sites=[SITE])
    original = dp_synthetic_cache.sites[SITE]
    assert np.array_equal(sliced.end_times, dp_synthetic_cache.end_times[70_000:70_100])
    assert np.array_equal(sliced.sites[SITE].temperature_c, original.temperature_c[70_000:70_100])
    assert np.array_equal(
        sliced.sites[SITE].smoothed_temperature_c, original.smoothed_temperature_c[70_000:70_100]
    )
    assert np.array_equal(sliced.calendar.band, dp_synthetic_cache.calendar.band[70_000:70_100])
