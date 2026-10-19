"""Tests of the generation-side options of the convex model (`studies.dp_gam`, experiment 110).

The options are the covariance penalty and the other routes by which demand terms can imitate solar,
the anomaly form of the solar term, the smooth export clip with its profile, and the unmetered
generator node. Each test replaces the power with a truth made from the model's own design, or from
a known clip or generator, so a fit that recovers the truth shows that the design, the penalties,
the constraints, and the readouts agree.
"""

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from typing import Final

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")

from studies.dp_features import FeatureCache  # noqa: E402
from studies.dp_gam import (  # noqa: E402
    ClipSpec,
    GamSpec,
    NodeSpec,
    TrainingWindow,
    clipped_pv,
    design_matrix,
    fit_gam,
    generator_path,
    month_half_hour_mean,
    orthogonal_penalty_matrix,
    pinned_solar_points,
    predict_gam,
    prepare_fit,
    profile_clip,
)
from studies.dp_pinned_capacity import PinnedCapacity, pinned_capacity  # noqa: E402

from studies import dp_gam  # noqa: E402

CUT: Final[datetime] = datetime(2024, 7, 1, tzinfo=UTC)
SEASON_YEARS: Final[range] = range(2019, 2026)
NOISE_POINTS: Final[float] = 0.15
WINDOW: Final[TrainingWindow] = TrainingWindow(months=18)
N_BANDS: Final[int] = 4


def _pinned(*, cache: FeatureCache, cut: datetime = CUT) -> PinnedCapacity:
    """Pin S1's capacity to a weekly vintage that rises from 20 points at 0.05 points a week."""
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
            "capacity_mw": [2.0 + 0.005 * i for i in range(len(weeks))],
            "unobserved": False,
        }
    )
    return pinned_capacity(vintages=vintages, table=cache.sites["S1"].table, cut=cut)


def _truth(*, layout: dp_gam.GamLayout) -> np.ndarray:
    """Return plausible coefficients in the layout's columns."""
    theta = np.zeros(layout.n_columns)
    profile = layout.slices["profile"]
    shape = 40.0 + 6.0 * np.sin(2 * np.pi * np.arange(48) / 48 - 1.0)
    for day_type, offset in enumerate((0.0, -4.0, -6.0, -6.0)):
        theta[profile.start + day_type * 48 : profile.start + (day_type + 1) * 48] = shape + offset
    heating = layout.slices["heating"]
    n_smooth = (heating.stop - heating.start) // N_BANDS
    for band, slope in enumerate((0.8, 1.5, 1.0, 1.8)):
        theta[heating.start + (n_smooth - 1) * N_BANDS + band] = slope
    theta[layout.slices["lighting"]] = (0.5, 0.3, 0.2, 0.1)
    return theta


def _with_power(*, cache: FeatureCache, power: np.ndarray) -> FeatureCache:
    """Return the cache with S1's power replaced, plus noise."""
    features = cache.sites["S1"]
    noise = NOISE_POINTS * np.random.default_rng(5).normal(size=len(power))
    return replace(cache, sites={"S1": replace(features, y=power + noise)})


def _usable_rows(*, cache: FeatureCache) -> np.ndarray:
    features = cache.sites["S1"]
    return np.flatnonzero(
        np.isfinite(features.pv_unit_shape)
        & np.isfinite(features.smoothed_temperature_c).all(axis=1)
    )


def _power_from_design(
    *, cache: FeatureCache, layout: dp_gam.GamLayout, theta: np.ndarray
) -> np.ndarray:
    features = cache.sites["S1"]
    power = np.full(len(cache.end_times), np.nan)
    usable = _usable_rows(cache=cache)
    for start in range(0, len(usable), 20000):
        rows = usable[start : start + 20000]
        power[rows] = (
            design_matrix(cache=cache, features=features, layout=layout, index=rows) @ theta
        )
    return power


def _fit(
    *,
    cache: FeatureCache,
    spec: GamSpec,
    pinned: PinnedCapacity | None = None,
    window: TrainingWindow = WINDOW,
) -> dp_gam.GamFit:
    return fit_gam(
        cache=cache,
        site="S1",
        specs=[spec],
        cut=CUT,
        window=window,
        season_years=SEASON_YEARS,
        pinned=pinned,
    )


def _profile(
    *,
    cache: FeatureCache,
    pinned: PinnedCapacity,
    spec: GamSpec,
    multipliers: tuple[float, ...],
    shares: tuple[float, ...],
) -> dp_gam.ClipProfile:
    return profile_clip(
        cache=cache,
        site="S1",
        spec=spec,
        cut=CUT,
        window=WINDOW,
        season_years=SEASON_YEARS,
        pinned=pinned,
        multipliers=multipliers,
        shares=shares,
    )


def _pinned_truth_cache(
    *, cache: FeatureCache, spec: GamSpec, pinned: PinnedCapacity, clip: ClipSpec | None = None
) -> FeatureCache:
    """Return the cache with power made from a pinned-capacity truth with an optional clip."""
    layout, _ = prepare_fit(
        cache=cache,
        site="S1",
        spec=spec,
        cut=CUT,
        window=WINDOW,
        season_years=SEASON_YEARS,
    )
    power = _power_from_design(cache=cache, layout=layout, theta=_truth(layout=layout))
    usable = _usable_rows(cache=cache)
    for start in range(0, len(usable), 20000):
        rows = usable[start : start + 20000]
        power[rows] -= pinned_solar_points(
            cache=cache, features=cache.sites["S1"], pinned=pinned, index=rows, clip=clip
        )
    return _with_power(cache=cache, power=power)


def test_an_open_clip_and_every_option_switched_off_reproduce_the_base_fit_exactly(
    dp_synthetic_cache: FeatureCache,
):
    pinned = _pinned(cache=dp_synthetic_cache)
    base = GamSpec(capacity="pinned", ridge=0.01)
    cache = _pinned_truth_cache(cache=dp_synthetic_cache, spec=base, pinned=pinned)
    reference = _fit(cache=cache, spec=base, pinned=pinned)
    off = replace(
        base,
        orthogonal_penalty=0.0,
        shared_harmonics=False,
        slow_heating_only=False,
        node=None,
        clip=ClipSpec(multiplier=1.0, p_max_share=float("inf")),
    )
    open_fit = _fit(cache=cache, spec=off, pinned=pinned)
    assert np.array_equal(open_fit.theta, reference.theta)


def test_the_smooth_clip_follows_its_input_far_below_the_ceiling_and_the_ceiling_far_above():
    x = np.array([0.0, 5.0, 10.0, 30.0, 60.0])
    clipped = clipped_pv(unclipped=x, p_max=20.0, beta=2.0)
    assert clipped[:3] == pytest.approx(x[:3], abs=1e-3)
    assert clipped[4] == pytest.approx(20.0, abs=1e-6)
    assert np.all(np.diff(clipped) >= 0.0)
    assert np.all(clipped <= 20.0)
    assert np.array_equal(clipped_pv(unclipped=x, p_max=float("inf"), beta=2.0), x)


def test_the_orthogonal_penalty_matrix_equals_a_direct_sum_of_squared_covariances(
    dp_synthetic_cache: FeatureCache,
):
    spec = GamSpec(capacity="pinned")
    cache = dp_synthetic_cache
    layout, index = prepare_fit(
        cache=cache, site="S1", spec=spec, cut=CUT, window=WINDOW, season_years=SEASON_YEARS
    )
    features = cache.sites["S1"]
    matrix = orthogonal_penalty_matrix(cache=cache, features=features, layout=layout, index=index)
    theta = np.random.default_rng(3).normal(size=layout.n_columns)
    shape = dp_gam.derated_unit_shape(features=features, index=index)
    rows = index[shape > dp_gam.LAST_TRAINED_MIN_SHAPE]
    unit = shape[shape > dp_gam.LAST_TRAINED_MIN_SHAPE]
    design = design_matrix(cache=cache, features=features, layout=layout, index=rows)
    cell = cache.calendar.quarter[rows].astype(int) * 48 + cache.calendar.half_hour[rows]
    expected = 0.0
    for name in dp_gam.ORTHOGONAL_BLOCKS:
        term = design[:, layout.slices[name]] @ theta[layout.slices[name]]
        for value in np.unique(cell):
            in_cell = cell == value
            if in_cell.sum() < dp_gam.MIN_ORTHOGONAL_CELL_ROWS:
                continue
            covariance = np.mean(unit[in_cell] * term[in_cell]) - np.mean(unit[in_cell]) * np.mean(
                term[in_cell]
            )
            expected += covariance**2
    assert theta @ matrix @ theta == pytest.approx(expected, rel=1e-9)


def test_the_orthogonal_penalty_shrinks_the_covariance_of_the_demand_terms_with_solar(
    dp_synthetic_cache: FeatureCache,
):
    pinned = _pinned(cache=dp_synthetic_cache)
    base = GamSpec(capacity="pinned", ridge=0.01)
    cache = _pinned_truth_cache(cache=dp_synthetic_cache, spec=base, pinned=pinned)
    free = _fit(cache=cache, spec=base, pinned=pinned)
    penalised = _fit(cache=cache, spec=replace(base, orthogonal_penalty=100.0), pinned=pinned)
    layout, index = prepare_fit(
        cache=cache, site="S1", spec=base, cut=CUT, window=WINDOW, season_years=SEASON_YEARS
    )
    matrix = orthogonal_penalty_matrix(
        cache=cache, features=cache.sites["S1"], layout=layout, index=index
    )
    free_value = free.theta[0] @ matrix @ free.theta[0]
    penalised_value = penalised.theta[0] @ matrix @ penalised.theta[0]
    assert penalised_value < 0.5 * free_value
    assert (penalised.violation <= 1e-5).all()


def test_the_shared_harmonics_and_slow_heating_options_shrink_the_layout_and_recover_the_truth(
    dp_synthetic_cache: FeatureCache,
):
    spec = GamSpec(capacity="pinned", ridge=0.01, shared_harmonics=True, slow_heating_only=True)
    layout, _ = prepare_fit(
        cache=dp_synthetic_cache,
        site="S1",
        spec=spec,
        cut=CUT,
        window=WINDOW,
        season_years=SEASON_YEARS,
    )
    assert layout.slices["harmonics"].stop - layout.slices["harmonics"].start == 4
    assert layout.slices["heating"].stop - layout.slices["heating"].start == 2 * N_BANDS
    pinned = _pinned(cache=dp_synthetic_cache)
    cache = _pinned_truth_cache(cache=dp_synthetic_cache, spec=spec, pinned=pinned)
    fit = _fit(cache=cache, spec=spec, pinned=pinned)
    heating = layout.slices["heating"]
    fitted = fit.theta[0][heating][N_BANDS : 2 * N_BANDS]
    assert fitted == pytest.approx((0.8, 1.5, 1.0, 1.8), rel=0.15)


def test_the_slow_heating_option_uses_the_72_and_168_hour_temperatures(
    dp_synthetic_cache: FeatureCache,
):
    spec = GamSpec(capacity="pinned", slow_heating_only=True)
    layout, index = prepare_fit(
        cache=dp_synthetic_cache,
        site="S1",
        spec=spec,
        cut=CUT,
        window=WINDOW,
        season_years=SEASON_YEARS,
    )
    features = dp_synthetic_cache.sites["S1"]
    rows = index[:50]
    design = design_matrix(cache=dp_synthetic_cache, features=features, layout=layout, index=rows)
    band = dp_synthetic_cache.calendar.band[rows].astype(int)
    hinge = (
        np.clip(15.5 - features.smoothed_temperature_c[rows][:, 3], 0.0, None)
        / dp_gam.HEATING_SCALE_C
    )
    heating = layout.slices["heating"]
    # The second column of the slow block is the 168-hour temperature, at the row's band.
    assert design[np.arange(50), heating.start + N_BANDS + band] == pytest.approx(hinge)


def test_the_anomaly_form_uses_training_rows_only_for_its_mean_and_recovers_capacity(
    dp_synthetic_cache: FeatureCache,
):
    spec = GamSpec(solar="anomaly", ridge=0.01, anomaly_ridge=1.0)
    layout, index = prepare_fit(
        cache=dp_synthetic_cache,
        site="S1",
        spec=spec,
        cut=CUT,
        window=WINDOW,
        season_years=SEASON_YEARS,
    )
    assert layout.anomaly_mean is not None
    theta = _truth(layout=layout)
    theta[layout.slices["capacity"].start + layout.season_years.index(2022)] = 12.0
    climatology = layout.slices["climatology"]
    theta[climatology] = 0.0
    cache = _with_power(
        cache=dp_synthetic_cache,
        power=_power_from_design(cache=dp_synthetic_cache, layout=layout, theta=theta),
    )
    fit = _fit(cache=cache, spec=spec)
    assert fit.capacity_points()[2023] == pytest.approx(12.0, rel=0.15)
    after_cut = np.flatnonzero(cache.end_times > np.datetime64("2024-07-01T00:00", "us"))
    features = cache.sites["S1"]
    moved = replace(
        cache,
        sites={
            "S1": replace(
                features,
                pv_unit_shape=np.where(
                    np.isin(np.arange(len(cache.end_times)), after_cut),
                    0.0,
                    features.pv_unit_shape,
                ),
            )
        },
    )
    refit = _fit(cache=moved, spec=spec)
    assert refit.layout.anomaly_mean is not None
    assert fit.layout.anomaly_mean is not None
    assert np.array_equal(refit.layout.anomaly_mean, fit.layout.anomaly_mean)
    assert np.allclose(refit.theta, fit.theta)
    mean = month_half_hour_mean(cache=cache, features=features, index=index)
    assert mean.shape == (12, 48)
    assert mean[:, :6].max() == 0.0


def test_a_clip_profile_recovers_an_injected_clip_and_stays_open_without_one(
    dp_synthetic_cache: FeatureCache,
):
    pinned = _pinned(cache=dp_synthetic_cache)
    base = GamSpec(capacity="pinned", ridge=0.01)
    truth_clip = ClipSpec(multiplier=1.0, p_max_share=0.6)
    clipped_cache = _pinned_truth_cache(
        cache=dp_synthetic_cache, spec=base, pinned=pinned, clip=truth_clip
    )
    open_cache = _pinned_truth_cache(cache=dp_synthetic_cache, spec=base, pinned=pinned)
    shares = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    multipliers = (0.9, 1.0, 1.1)
    found = _profile(
        cache=clipped_cache, pinned=pinned, spec=base, multipliers=multipliers, shares=shares
    )
    assert found.best.p_max_share == pytest.approx(0.6, abs=0.1)
    assert found.best.multiplier == pytest.approx(1.0, abs=0.1)
    assert found.objective.min() < found.open_objective
    unclipped = _profile(
        cache=open_cache, pinned=pinned, spec=base, multipliers=multipliers, shares=shares
    )
    assert unclipped.best.p_max_share == 1.0
    assert unclipped.objective.min() == pytest.approx(unclipped.open_objective, rel=0.05)


def test_the_profile_grid_matches_a_separate_fit_at_one_grid_point(
    dp_synthetic_cache: FeatureCache,
):
    pinned = _pinned(cache=dp_synthetic_cache)
    base = GamSpec(capacity="pinned", ridge=0.01)
    clip = ClipSpec(multiplier=1.1, p_max_share=0.7)
    cache = _pinned_truth_cache(cache=dp_synthetic_cache, spec=base, pinned=pinned, clip=clip)
    profile = profile_clip(
        cache=cache,
        site="S1",
        spec=base,
        cut=CUT,
        window=WINDOW,
        season_years=SEASON_YEARS,
        pinned=pinned,
        multipliers=(1.0, 1.1),
        shares=(0.7, 1.0),
    )
    fit = _fit(cache=cache, spec=replace(base, clip=clip), pinned=pinned)
    layout, index = prepare_fit(
        cache=cache,
        site="S1",
        spec=replace(base, clip=clip),
        cut=CUT,
        window=WINDOW,
        season_years=SEASON_YEARS,
    )
    features = cache.sites["S1"]
    target = features.y[index] + pinned_solar_points(
        cache=cache, features=features, pinned=pinned, index=index, clip=clip
    )
    design = design_matrix(cache=cache, features=features, layout=layout, index=index)
    matrix, cost = dp_gam.penalty_matrix(layout=layout, n_rows=len(index))
    theta = fit.theta[0]
    direct = float(np.square(target - design @ theta).sum()) + theta @ matrix @ theta + cost @ theta
    assert profile.objective[1, 0] == pytest.approx(direct, rel=1e-6)


def test_the_profile_refuses_a_spec_that_already_has_a_clip_or_a_fitted_capacity(
    dp_synthetic_cache: FeatureCache,
):
    pinned = _pinned(cache=dp_synthetic_cache)
    grid = {"multipliers": (1.0,), "shares": (1.0,)}
    with pytest.raises(ValueError, match="pinned capacity and no clip"):
        _profile(
            cache=dp_synthetic_cache,
            pinned=pinned,
            spec=GamSpec(capacity="pinned", clip=ClipSpec()),
            **grid,
        )
    with pytest.raises(ValueError, match="pinned capacity and no clip"):
        _profile(cache=dp_synthetic_cache, pinned=pinned, spec=GamSpec(capacity="fitted"), **grid)


def test_a_clip_or_an_anomaly_form_with_the_wrong_capacity_mode_is_refused():
    with pytest.raises(ValueError, match="pinned solar output"):
        GamSpec(capacity="fitted", clip=ClipSpec())
    with pytest.raises(ValueError, match="capacity must be 'fitted'"):
        GamSpec(capacity="pinned", solar="anomaly")
    with pytest.raises(ValueError, match="must not be negative"):
        GamSpec(orthogonal_penalty=-1.0)


NODE_WINDOW: Final[TrainingWindow] = TrainingWindow(months=12)


def _node_spec(*, step_penalty: float = 300.0, level_penalty: float = 1.0) -> GamSpec:
    return GamSpec(
        capacity="pinned",
        ridge=0.01,
        node=NodeSpec(step_penalty=step_penalty, level_penalty=level_penalty),
    )


def _epoch_day(*, year: int, month: int, day: int) -> int:
    """Return the days from 1970-01-01 to a date."""
    return (datetime(year, month, day, tzinfo=UTC) - datetime(1970, 1, 1, tzinfo=UTC)).days


def _generator_truth_cache(
    *,
    cache: FeatureCache,
    pinned: PinnedCapacity,
    generator: Callable[[np.ndarray], np.ndarray] | None,
) -> FeatureCache:
    """Return the cache whose power also subtracts a generator that is a function of the day.

    The generator is a function of each half-hour's day, counted in days since 1970-01-01.
    """
    base = _pinned_truth_cache(
        cache=cache, spec=GamSpec(capacity="pinned", ridge=0.01), pinned=pinned
    )
    if generator is None:
        return base
    features = base.sites["S1"]
    days = np.floor(
        base.end_times.astype("datetime64[s]").astype(np.int64) / 86400.0 - 1.0 / 48.0
    ).astype(int)
    return replace(base, sites={"S1": replace(features, y=features.y - generator(days))})


def test_a_node_on_a_series_with_no_generator_stays_under_half_a_point(
    dp_synthetic_cache: FeatureCache,
):
    pinned = _pinned(cache=dp_synthetic_cache)
    cache = _generator_truth_cache(cache=dp_synthetic_cache, pinned=pinned, generator=None)
    fit = _fit(cache=cache, spec=_node_spec(), pinned=pinned, window=NODE_WINDOW)
    path = generator_path(fit=fit)
    assert path.max() - path.min() < 0.5
    assert (fit.violation <= 1e-5).all()


def test_a_node_finds_an_injected_step_and_a_week_long_outage_and_never_goes_negative(
    dp_synthetic_cache: FeatureCache,
):
    pinned = _pinned(cache=dp_synthetic_cache)
    step_day = _epoch_day(year=2023, month=11, day=1)
    outage_start = _epoch_day(year=2024, month=2, day=1)

    def generator(day: np.ndarray) -> np.ndarray:
        level = np.where(day >= step_day, 10.0, 5.0)
        return np.where((day >= outage_start) & (day < outage_start + 7), 0.0, level)

    cache = _generator_truth_cache(cache=dp_synthetic_cache, pinned=pinned, generator=generator)
    fit = _fit(cache=cache, spec=_node_spec(), pinned=pinned, window=NODE_WINDOW)
    path = generator_path(fit=fit)
    assert path.min() >= -1e-6
    first_day = int(fit.layout.node_first_day)
    levels = {first_day + i: value for i, value in enumerate(path)}
    during = np.mean([levels[outage_start + k] for k in range(1, 6)])
    before = np.mean([levels[outage_start - k] for k in range(2, 8)])
    after = np.mean([levels[outage_start + 7 + k] for k in range(2, 8)])
    assert before - during > 0.8 * 10.0
    assert after == pytest.approx(before, abs=1.5)
    assert levels[step_day + 10] - levels[step_day - 10] == pytest.approx(5.0, abs=1.5)


def test_a_rise_in_demand_is_read_as_a_generator_switching_off_never_as_negative_generation(
    dp_synthetic_cache: FeatureCache,
):
    pinned = _pinned(cache=dp_synthetic_cache)
    rise_day = _epoch_day(year=2023, month=12, day=15)
    cache = _generator_truth_cache(
        cache=dp_synthetic_cache,
        pinned=pinned,
        generator=lambda day: np.where(day >= rise_day, -4.0, 0.0),
    )
    fit = _fit(cache=cache, spec=_node_spec(), pinned=pinned, window=NODE_WINDOW)
    path = generator_path(fit=fit)
    assert path.min() >= -1e-6
    levels = {int(fit.layout.node_first_day) + i: value for i, value in enumerate(path)}
    assert levels[rise_day - 20] - levels[rise_day + 20] == pytest.approx(4.0, abs=1.0)


def test_the_node_level_is_held_after_the_cut(dp_synthetic_cache: FeatureCache):
    pinned = _pinned(cache=dp_synthetic_cache)
    cache = _generator_truth_cache(
        cache=dp_synthetic_cache, pinned=pinned, generator=lambda day: np.full(len(day), 7.0)
    )
    fit = _fit(cache=cache, spec=_node_spec(), pinned=pinned, window=NODE_WINDOW)
    path = generator_path(fit=fit)
    plain = _fit(
        cache=cache, spec=GamSpec(capacity="pinned", ridge=0.01), pinned=pinned, window=NODE_WINDOW
    )
    rows = np.flatnonzero(
        (cache.end_times > np.datetime64("2024-07-01T00:00", "us"))
        & (cache.end_times <= np.datetime64("2024-08-01T00:00", "us"))
    )
    with_node = predict_gam(cache=cache, fit=fit, index=rows)
    without = predict_gam(cache=cache, fit=plain, index=rows)
    # A constant generator is aliased with the profile, so both fits forecast the same net power
    # within the noise, and the node holds its last day's level.
    assert path[-1] >= 0.0
    assert np.abs(with_node - without).mean() < 1.0


def test_a_node_with_weekly_blocks_is_smaller_and_still_places_a_step_within_a_week(
    dp_synthetic_cache: FeatureCache,
):
    pinned = _pinned(cache=dp_synthetic_cache)
    step_day = _epoch_day(year=2023, month=11, day=1)
    cache = _generator_truth_cache(
        cache=dp_synthetic_cache,
        pinned=pinned,
        generator=lambda day: np.where(day >= step_day, 10.0, 5.0),
    )
    weekly = replace(_node_spec(), node=NodeSpec(step_days=7))
    daily = _node_spec()
    layout_weekly, _ = prepare_fit(
        cache=cache,
        site="S1",
        spec=weekly,
        cut=CUT,
        window=NODE_WINDOW,
        season_years=SEASON_YEARS,
    )
    layout_daily, _ = prepare_fit(
        cache=cache, site="S1", spec=daily, cut=CUT, window=NODE_WINDOW, season_years=SEASON_YEARS
    )
    assert layout_weekly.n_node_steps == -(-layout_daily.n_node_steps // 7)
    fit = _fit(cache=cache, spec=weekly, pinned=pinned, window=NODE_WINDOW)
    path = generator_path(fit=fit)
    first_day = int(fit.layout.node_first_day)
    step_block = (step_day - first_day) // 7
    assert path[step_block + 2] - path[step_block - 2] == pytest.approx(5.0, abs=1.5)


def test_a_node_block_shorter_than_a_day_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        NodeSpec(step_days=0)


def test_a_generator_path_refuses_a_fit_with_no_node(dp_synthetic_cache: FeatureCache):
    pinned = _pinned(cache=dp_synthetic_cache)
    fit = _fit(
        cache=dp_synthetic_cache,
        spec=GamSpec(capacity="pinned", ridge=0.01),
        pinned=pinned,
    )
    with pytest.raises(ValueError, match="no unmetered generator node"):
        generator_path(fit=fit)


def test_a_batch_of_orthogonal_weights_gives_the_same_fits_as_separate_calls(
    dp_synthetic_cache: FeatureCache,
):
    pinned = _pinned(cache=dp_synthetic_cache)
    base = GamSpec(capacity="pinned", ridge=0.01)
    cache = _pinned_truth_cache(cache=dp_synthetic_cache, spec=base, pinned=pinned)
    weights = (0.0, 1.0, 100.0)
    batch = fit_gam(
        cache=cache,
        site="S1",
        specs=[replace(base, orthogonal_penalty=value) for value in weights],
        cut=CUT,
        window=WINDOW,
        season_years=SEASON_YEARS,
        pinned=pinned,
    )
    for member, value in enumerate(weights):
        single = _fit(cache=cache, spec=replace(base, orthogonal_penalty=value), pinned=pinned)
        assert np.allclose(batch.theta[member], single.theta[0], atol=1e-6)


def _garbled_after_cut(*, cache: FeatureCache) -> FeatureCache:
    """Return the cache with S1's power and solar shape replaced by garbage after the cut."""
    features = cache.sites["S1"]
    after = cache.end_times > np.datetime64("2024-07-01T00:00", "us")
    return replace(
        cache,
        sites={
            "S1": replace(
                features,
                y=np.where(after, 1e3, features.y),
                pv_unit_shape=np.where(after, 0.0, features.pv_unit_shape),
            )
        },
    )


def test_the_clip_profile_and_the_orthogonal_penalty_use_training_rows_only(
    dp_synthetic_cache: FeatureCache,
):
    pinned = _pinned(cache=dp_synthetic_cache)
    base = GamSpec(capacity="pinned", ridge=0.01)
    cache = _pinned_truth_cache(cache=dp_synthetic_cache, spec=base, pinned=pinned)
    garbled = _garbled_after_cut(cache=cache)
    first = _profile(
        cache=cache, pinned=pinned, spec=base, multipliers=(0.9, 1.0), shares=(0.6, 1.0)
    )
    second = _profile(
        cache=garbled, pinned=pinned, spec=base, multipliers=(0.9, 1.0), shares=(0.6, 1.0)
    )
    assert np.array_equal(first.objective, second.objective)
    layout, index = prepare_fit(
        cache=cache, site="S1", spec=base, cut=CUT, window=WINDOW, season_years=SEASON_YEARS
    )
    assert np.array_equal(
        orthogonal_penalty_matrix(
            cache=cache, features=cache.sites["S1"], layout=layout, index=index
        ),
        orthogonal_penalty_matrix(
            cache=garbled, features=garbled.sites["S1"], layout=layout, index=index
        ),
    )


def test_the_clip_scales_the_output_by_the_multiplier_and_the_ceiling_by_the_held_capacity(
    dp_synthetic_cache: FeatureCache,
):
    pinned = _pinned(cache=dp_synthetic_cache)
    features = dp_synthetic_cache.sites["S1"]
    rows = np.flatnonzero(features.pv_unit_shape > 0.5)[:200]
    plain = pinned_solar_points(
        cache=dp_synthetic_cache, features=features, pinned=pinned, index=rows
    )
    open_clip = pinned_solar_points(
        cache=dp_synthetic_cache,
        features=features,
        pinned=pinned,
        index=rows,
        clip=ClipSpec(multiplier=1.2, p_max_share=float("inf")),
    )
    assert open_clip == pytest.approx(1.2 * plain)
    ceiling = 0.5 * 1.2 * pinned.held_points
    clipped = pinned_solar_points(
        cache=dp_synthetic_cache,
        features=features,
        pinned=pinned,
        index=rows,
        clip=ClipSpec(multiplier=1.2, p_max_share=0.5),
    )
    assert clipped.max() <= ceiling
    assert clipped.max() > 0.9 * ceiling
