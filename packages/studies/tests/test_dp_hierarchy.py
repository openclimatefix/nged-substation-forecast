"""Tests of the joint fit of Primaries and their bulk supply point (`studies.dp_hierarchy`).

The fixtures build a feature cache from synthetic weather with the real builder, copy the one
synthetic substation into two with different capacities, and replace their power with a truth made
from the model's own design. A bulk supply point (BSP) is then built exactly from the relation the
module fits, so a fit that recovers the truth shows the stacked design, the weights, and the
remainder agree.
"""

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from functools import partial
from typing import Final

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")

from studies.dp_features import FeatureCache  # noqa: E402
from studies.dp_gam import (  # noqa: E402
    GamLayout,
    GamSpec,
    TrainingWindow,
    accumulate_gram,
    design_matrix,
    fit_gam,
    penalty_matrix,
    pinned_solar_points,
    predict_gam,
    prepare_fit,
)
from studies.dp_hierarchy import (  # noqa: E402
    BspMember,
    BspSeries,
    JointFit,
    RemainderSpec,
    assemble_joint,
    bsp_rows,
    fit_joint,
    make_remainder_layout,
    predict_bsp,
    remainder_design,
    remainder_penalty,
    remainder_prediction,
    separate_fits_equal_joint_at_zero_weight,
    solve_joint,
)
from studies.dp_pinned_capacity import PinnedCapacity, pinned_capacity  # noqa: E402

CUT: Final[datetime] = datetime(2024, 7, 1, tzinfo=UTC)
WINDOW: Final[TrainingWindow] = TrainingWindow(months=12)
SEASON_YEARS: Final[range] = range(2019, 2026)
SPEC: Final[GamSpec] = GamSpec(drift="none", ridge=1e-3)
CAPACITY_MW: Final[dict[str, float]] = {"S1": 10.0, "S2": 25.0}
SIGMA_POINTS: Final[dict[str, float]] = {"S1": 0.5, "S2": 2.0}
TRUE_LEVEL_MW: Final[float] = 3.0
TRUE_PV_MW: Final[float] = 2.0
TRUE_HEATING_MW: Final[float] = 0.5
GENERATORS_MEAN_MW: Final[float] = 4.0


def _truth_theta(*, layout: GamLayout, scale: float) -> np.ndarray:
    """Return a plausible substation's coefficients: a daily profile and one heating slope."""
    theta = np.zeros(layout.n_columns)
    profile = layout.slices["profile"]
    half_hours = np.arange(48)
    for day_type, offset in enumerate((0.0, -4.0, -6.0, -6.0)):
        shape = scale * (40.0 + 6.0 * np.sin(2 * np.pi * half_hours / 48 - 1.0) + offset)
        theta[profile.start + day_type * 48 : profile.start + (day_type + 1) * 48] = shape
    heating = layout.slices["heating"]
    theta[heating.start + 4 : heating.start + 8] = scale * 1.2
    return theta


def _two_site_cache(*, cache: FeatureCache) -> tuple[FeatureCache, dict[str, np.ndarray]]:
    """Copy S1 into S1 and S2 with truth power; return the cache and each site's truth theta."""
    base = cache.sites["S1"]
    sites = {}
    thetas = {}
    for label, scale in (("S1", 1.0), ("S2", 0.7)):
        table = replace(base.table, site=label, effective_capacity_mw=CAPACITY_MW[label])
        features = replace(base, table=table)
        layout, _ = prepare_fit(
            cache=cache, site="S1", spec=SPEC, cut=CUT, window=WINDOW, season_years=SEASON_YEARS
        )
        theta = _truth_theta(layout=layout, scale=scale)
        usable = np.flatnonzero(
            np.isfinite(features.pv_unit_shape)
            & np.isfinite(features.smoothed_temperature_c).all(axis=1)
        )
        y = np.full(len(cache.end_times), np.nan)
        for start in range(0, len(usable), 20000):
            rows = usable[start : start + 20000]
            x = design_matrix(cache=cache, features=features, layout=layout, index=rows)
            y[rows] = x @ theta
        sites[label] = replace(features, y=y)
        thetas[label] = theta
    return replace(cache, sites=sites, node_ids={s: (s,) for s in sites}), thetas


def _generators(*, n: int) -> np.ndarray:
    """Return a metered-generation series that varies in time and repeats nowhere.

    A constant would be absorbed by the remainder's level, and a daily cycle by its daily shape,
    so a wrong sign on the generator term would go unseen.
    """
    noise = np.random.default_rng(5).normal(size=n)
    return GENERATORS_MEAN_MW + 2.0 * np.convolve(noise, np.ones(6) / 6.0, mode="same")


def _bsp(
    *, cache: FeatureCache, members: tuple[BspMember, ...], truth_members: tuple[str, ...]
) -> BspSeries:
    """Build a BSP exactly from the module's relation, with the true remainder."""
    base = cache.sites["S1"]
    n = len(cache.end_times)
    hinge = np.clip(15.5 - base.temperature_c, 0.0, None) / 10.0
    remainder = TRUE_LEVEL_MW - TRUE_PV_MW * base.pv_unit_shape + TRUE_HEATING_MW * hinge
    generators = _generators(n=n)
    reading = remainder - generators
    for site in truth_members:
        reading = reading + CAPACITY_MW[site] * cache.sites[site].y
    return BspSeries(
        name="B",
        members=members,
        reading_mw=reading,
        generators_mw=generators,
        temperature_c=base.temperature_c,
        pv_shape=base.pv_unit_shape,
        sigma_mw=0.5,
    )


@pytest.fixture(scope="module")
def two_sites(dp_synthetic_cache: FeatureCache) -> FeatureCache:
    """The synthetic cache with two Primaries of different capacities."""
    cache, _thetas = _two_site_cache(cache=dp_synthetic_cache)
    return cache


def _members() -> tuple[BspMember, ...]:
    return (BspMember(site="S1"), BspMember(site="S2"))


def _fit(
    *, cache: FeatureCache, bsp: BspSeries, weights: list[float], step_penalty: float = 1.0
) -> JointFit:
    return fit_joint(
        cache=cache,
        specs={"S1": SPEC, "S2": SPEC},
        bsps=[bsp],
        remainder_spec=RemainderSpec(step_penalty=step_penalty, shape_ridge=1e-3),
        weights=weights,
        sigma_points=SIGMA_POINTS,
        cut=CUT,
        window=WINDOW,
        season_years=SEASON_YEARS,
    )


def test_at_weight_zero_the_joint_fit_equals_the_separate_fits_whatever_the_sigmas(
    two_sites: FeatureCache,
):
    # Control P1 of the experiment's plan: at weight 0 the joint fit is the separate fits.
    # The sigmas differ by a factor of 4, so a joint block that scaled the loss but not the
    # penalties (or the reverse) would move the minimiser.
    noisy = {
        s: replace(f, y=f.y + 0.3 * np.sin(np.arange(len(f.y)))) for s, f in two_sites.sites.items()
    }
    cache = replace(two_sites, sites=noisy)
    bsp = _bsp(cache=cache, members=_members(), truth_members=("S1", "S2"))
    joint = _fit(cache=cache, bsp=bsp, weights=[0.0, 1.0])
    separate = {
        s: fit_gam(
            cache=cache, site=s, specs=[SPEC], cut=CUT, window=WINDOW, season_years=SEASON_YEARS
        )
        for s in ("S1", "S2")
    }
    assert separate_fits_equal_joint_at_zero_weight(joint=joint, separate=separate, weight_index=0)
    # The check can fail: with the BSP switched on, the Primaries' coefficients move.
    assert not np.allclose(
        joint.theta[1, joint.site_slices["S1"]], separate["S1"].theta[0], atol=1e-6
    )


def test_the_weight_zero_check_refuses_a_member_with_another_weight(two_sites: FeatureCache):
    bsp = _bsp(cache=two_sites, members=_members(), truth_members=("S1", "S2"))
    joint = _fit(cache=two_sites, bsp=bsp, weights=[0.0, 1.0])
    with pytest.raises(ValueError, match="not 0"):
        separate_fits_equal_joint_at_zero_weight(joint=joint, separate={}, weight_index=1)


def test_a_noise_free_bsp_built_from_the_relation_is_recovered_at_weight_one(
    two_sites: FeatureCache,
):
    # Control P2: a noise-free BSP built exactly from the relation is recovered at weight 1.
    bsp = _bsp(cache=two_sites, members=_members(), truth_members=("S1", "S2"))
    joint = _fit(cache=two_sites, bsp=bsp, weights=[1.0])
    index = bsp_rows(
        cache=two_sites, bsp=bsp, start=np.datetime64("2023-07-01T00:00", "us"), cut=CUT
    )
    predicted = predict_bsp(
        cache=two_sites,
        fit=joint,
        bsp=bsp,
        index=index,
        generators_mw=bsp.generators_mw[index],
    )
    assert (
        np.abs(predicted - bsp.reading_mw[index]).max()
        < 1e-3 * np.abs(bsp.reading_mw[index]).mean()
    )
    remainder = remainder_prediction(cache=two_sites, fit=joint, bsp=bsp, index=index)
    base = two_sites.sites["S1"]
    truth = (
        TRUE_LEVEL_MW
        - TRUE_PV_MW * base.pv_unit_shape[index]
        + TRUE_HEATING_MW * np.clip(15.5 - base.temperature_c[index], 0.0, None) / 10.0
    )
    assert np.sqrt(np.mean((remainder - truth) ** 2)) < 0.1


def test_dropping_a_member_from_the_bsp_row_moves_the_remainder_by_that_members_demand(
    two_sites: FeatureCache,
):
    # The known-answer gate's sabotage reading: the audit has to see a missing Primary. The BSP
    # is built with both Primaries, but only S1 is declared under it, so S2's demand must land in
    # the remainder.
    full = _bsp(cache=two_sites, members=_members(), truth_members=("S1", "S2"))
    partial = replace(full, members=(BspMember(site="S1"),))
    fit_full = _fit(cache=two_sites, bsp=full, weights=[1.0])
    fit_partial = fit_joint(
        cache=two_sites,
        specs={"S1": SPEC},
        bsps=[partial],
        remainder_spec=RemainderSpec(step_penalty=1.0, shape_ridge=1e-3),
        weights=[1.0],
        sigma_points=SIGMA_POINTS,
        cut=CUT,
        window=WINDOW,
        season_years=SEASON_YEARS,
    )
    index = bsp_rows(
        cache=two_sites, bsp=full, start=np.datetime64("2023-07-01T00:00", "us"), cut=CUT
    )
    shift = remainder_prediction(
        cache=two_sites, fit=fit_partial, bsp=partial, index=index
    ) - remainder_prediction(cache=two_sites, fit=fit_full, bsp=full, index=index)
    s2_demand = CAPACITY_MW["S2"] * two_sites.sites["S2"].y[index]
    assert abs(shift.mean() - s2_demand.mean()) < 0.1 * s2_demand.mean()


@pytest.mark.parametrize("step_penalty", [1e9, 30.0])
def test_the_bsp_hint_pulls_a_corrupted_primary_back_inside_the_corrupted_window(
    two_sites: FeatureCache, step_penalty: float
):
    # A 30% scale error on 90 days of S1's training readings, with S1 alone under the BSP and a
    # drift term free to follow the error. The BSP is built from the clean truth. The remainder's
    # level can absorb a constant offset, so the control reads the error inside the window, where
    # the damage is, and the BSP residual, which the weight is there to shrink. The experiment's
    # gate measures the whole damage reduction.
    spec = GamSpec(drift="d1", ridge=1e-3, drift_step=10.0, drift_ridge=1e-3)
    s1 = two_sites.sites["S1"]
    bsp = _bsp(cache=two_sites, members=(BspMember(site="S1"),), truth_members=("S1",))
    stamps = two_sites.end_times
    window = (stamps > np.datetime64("2024-01-01T00:00", "us")) & (
        stamps <= np.datetime64("2024-03-31T00:00", "us")
    )
    corrupted = replace(
        two_sites,
        sites={**two_sites.sites, "S1": replace(s1, y=np.where(window, 1.3 * s1.y, s1.y))},
    )
    joint = fit_joint(
        cache=corrupted,
        specs={"S1": spec},
        bsps=[bsp],
        remainder_spec=RemainderSpec(step_penalty=step_penalty, shape_ridge=1e-3),
        weights=[0.0, 30.0],
        sigma_points={"S1": 0.5},
        cut=CUT,
        window=WINDOW,
        season_years=SEASON_YEARS,
    )
    index = bsp_rows(
        cache=two_sites, bsp=bsp, start=np.datetime64("2023-07-01T00:00", "us"), cut=CUT
    )
    inside = window[index]
    damage = []
    residual = []
    for member in (0, 1):
        predicted = predict_gam(
            cache=two_sites, fit=joint.site_fit(site="S1"), index=index, member=member
        )
        damage.append(np.abs(predicted - s1.y[index])[inside].mean())
        bsp_hat = predict_bsp(
            cache=two_sites,
            fit=joint,
            bsp=bsp,
            index=index,
            generators_mw=bsp.generators_mw[index],
            weight_index=member,
        )
        residual.append(np.abs(bsp_hat - bsp.reading_mw[index]).mean())
    assert damage[1] < 0.5 * damage[0]
    assert residual[1] < 0.01 * residual[0]


def test_no_fit_reads_a_row_after_the_cut(two_sites: FeatureCache):
    bsp = _bsp(cache=two_sites, members=_members(), truth_members=("S1", "S2"))
    after = two_sites.end_times > np.datetime64("2024-07-01T00:00", "us")
    changed = replace(
        bsp,
        reading_mw=np.where(after, 1e6, bsp.reading_mw),
        generators_mw=np.where(after, 1e6, bsp.generators_mw),
        temperature_c=np.where(after, -1e3, bsp.temperature_c),
        pv_shape=np.where(after, 1e3, bsp.pv_shape),
    )
    index = bsp_rows(
        cache=two_sites, bsp=changed, start=np.datetime64("2023-07-01T00:00", "us"), cut=CUT
    )
    assert two_sites.end_times[index].max() <= np.datetime64("2024-07-01T00:00", "us")
    first = _fit(cache=two_sites, bsp=bsp, weights=[1.0])
    second = _fit(cache=two_sites, bsp=changed, weights=[1.0])
    assert np.array_equal(first.theta, second.theta)


def test_the_unmetered_pv_term_stays_non_negative_when_the_truth_is_negative(
    two_sites: FeatureCache,
):
    bsp = _bsp(cache=two_sites, members=_members(), truth_members=("S1", "S2"))
    base = two_sites.sites["S1"]
    flipped = replace(bsp, reading_mw=bsp.reading_mw - 2 * TRUE_PV_MW * base.pv_unit_shape)
    joint = _fit(cache=two_sites, bsp=flipped, weights=[1.0])
    layout = joint.remainder_layouts["B"]
    pv = joint.theta[0, joint.remainder_slices["B"]][layout.slices["pv"]]
    assert (pv >= 0.0).all()
    assert (joint.theta[0, joint.remainder_slices["B"]][layout.slices["steps"]] >= 0.0).all()


def test_each_primary_block_is_the_separate_gram_plus_penalty_divided_by_its_sigma_squared(
    two_sites: FeatureCache,
):
    # Control P1, block by block: the block that `fit_joint` stacks for a Primary equals that
    # Primary's own Gram matrix plus penalty matrix, divided by the Primary's sigma squared.
    bsp = _bsp(cache=two_sites, members=_members(), truth_members=("S1", "S2"))
    joint = _fit(cache=two_sites, bsp=bsp, weights=[0.0])
    for site in ("S1", "S2"):
        layout, index = prepare_fit(
            cache=two_sites,
            site=site,
            spec=SPEC,
            cut=CUT,
            window=WINDOW,
            season_years=SEASON_YEARS,
        )
        gram, _rhs = accumulate_gram(
            cache=two_sites, features=two_sites.sites[site], layout=layout, index=index
        )
        penalty, _cost = penalty_matrix(layout=layout, n_rows=len(index))
        expected = (gram[0].numpy() + penalty) / SIGMA_POINTS[site] ** 2
        assert np.abs(joint.primary_blocks[site] - expected).max() < 1e-9 * np.abs(expected).max()


def test_a_very_large_weight_ties_the_primaries_to_the_bsp(two_sites: FeatureCache):
    # With w = 1e6 and a remainder that can only shift the level and the unmetered PV, the BSP
    # reading is matched to a relative error under 1e-3, and a noise-free truth is still the
    # solution because it satisfies the Primaries' rows and the BSP rows together.
    bsp = _bsp(cache=two_sites, members=_members(), truth_members=("S1", "S2"))
    joint = fit_joint(
        cache=two_sites,
        specs={"S1": SPEC, "S2": SPEC},
        bsps=[bsp],
        remainder_spec=RemainderSpec(shape_ridge=1e9, with_heating=False),
        weights=[1e6],
        sigma_points=SIGMA_POINTS,
        cut=CUT,
        window=WINDOW,
        season_years=SEASON_YEARS,
    )
    index = bsp_rows(
        cache=two_sites, bsp=bsp, start=np.datetime64("2023-07-01T00:00", "us"), cut=CUT
    )
    predicted = predict_bsp(
        cache=two_sites, fit=joint, bsp=bsp, index=index, generators_mw=bsp.generators_mw[index]
    )
    assert (
        np.abs(predicted - bsp.reading_mw[index]).mean()
        < 1e-3 * np.abs(bsp.reading_mw[index]).mean()
    )
    for site in ("S1", "S2"):
        fitted = predict_gam(cache=two_sites, fit=joint.site_fit(site=site), index=index)
        assert np.abs(fitted - two_sites.sites[site].y[index]).mean() < 0.05


def test_the_remainder_design_has_the_documented_columns_and_signs(two_sites: FeatureCache):
    bsp = _bsp(cache=two_sites, members=_members(), truth_members=("S1", "S2"))
    layout = make_remainder_layout(
        spec=RemainderSpec(with_loss=True),
        cut=CUT,
        earliest=np.datetime64("2023-07-01T00:00", "us"),
    )
    bsp = replace(bsp, loss=np.full(len(two_sites.end_times), 2.0))
    index = bsp_rows(
        cache=two_sites, bsp=bsp, start=np.datetime64("2023-07-01T00:00", "us"), cut=CUT
    )[:200]
    x = remainder_design(cache=two_sites, layout=layout, bsp=bsp, index=index)
    assert x.shape == (200, layout.n_columns)
    assert (x[:, layout.slices["level"]] == 1.0).all()
    assert (x[:, layout.slices["pv"]] <= 0.0).all()
    assert (x[:, layout.slices["shape"]].sum(axis=1) == 1.0).all()
    assert (x[:, layout.slices["loss"]] == 2.0).all()
    steps = x[:, layout.slices["steps"]]
    half = steps.shape[1] // 2
    assert np.array_equal(steps[:, :half], -steps[:, half:])
    penalty, cost = remainder_penalty(layout=layout, n_rows=17532)
    assert np.allclose(cost[layout.slices["steps"]], layout.spec.step_penalty)
    assert (cost[layout.slices["level"]] == 0.0).all()
    assert penalty[layout.slices["shape"], layout.slices["shape"]][0, 0] == layout.spec.shape_ridge


def test_a_primary_under_two_bsps_and_a_non_positive_sigma_are_refused(two_sites: FeatureCache):
    first = _bsp(cache=two_sites, members=_members(), truth_members=("S1", "S2"))
    second = replace(first, name="C", members=(BspMember(site="S2"),))
    fit = partial(
        fit_joint,
        cache=two_sites,
        specs={"S1": SPEC, "S2": SPEC},
        remainder_spec=RemainderSpec(),
        weights=[1.0],
        cut=CUT,
        window=WINDOW,
        season_years=SEASON_YEARS,
    )
    with pytest.raises(ValueError, match="sits under both"):
        fit(bsps=[first, second], sigma_points=SIGMA_POINTS)
    with pytest.raises(ValueError, match="no positive sigma"):
        fit(bsps=[first], sigma_points={"S1": 0.5, "S2": 0.0})
    with pytest.raises(ValueError, match="sigma that is not positive"):
        fit(bsps=[replace(first, sigma_mw=0.0)], sigma_points=SIGMA_POINTS)


def test_the_bsp_sigma_and_the_weight_enter_only_as_their_ratio(two_sites: FeatureCache):
    # Halving sigma at fixed weight is the same as quadrupling the weight at the original sigma.
    bsp = _bsp(cache=two_sites, members=_members(), truth_members=("S1", "S2"))
    noisy = {
        s: replace(f, y=f.y + 0.3 * np.sin(np.arange(len(f.y)))) for s, f in two_sites.sites.items()
    }
    cache = replace(two_sites, sites=noisy)
    narrow = _fit(cache=cache, bsp=replace(bsp, sigma_mw=0.5), weights=[1.0])
    wide = _fit(cache=cache, bsp=replace(bsp, sigma_mw=1.0), weights=[4.0])
    assert np.allclose(narrow.theta, wide.theta, rtol=1e-6, atol=1e-6)


def test_the_primary_step_costs_scale_with_the_sigma_like_the_loss(two_sites: FeatureCache):
    # P1 with a spec that has a linear (L1) step cost: a joint fit that scaled the loss and the
    # quadratic penalties by 1/sigma^2 but not the linear cost would move the minimiser.
    steps = GamSpec(drift="d1_steps", ridge=1e-3, drift_step=10.0, step_penalty=30.0)
    noisy = {
        s: replace(f, y=f.y + 0.5 * np.sin(np.arange(len(f.y)) / 7.0))
        for s, f in two_sites.sites.items()
    }
    cache = replace(two_sites, sites=noisy)
    bsp = _bsp(cache=cache, members=_members(), truth_members=("S1", "S2"))
    joint = fit_joint(
        cache=cache,
        specs={"S1": steps, "S2": steps},
        bsps=[bsp],
        remainder_spec=RemainderSpec(),
        weights=[0.0],
        sigma_points=SIGMA_POINTS,
        cut=CUT,
        window=WINDOW,
        season_years=SEASON_YEARS,
    )
    separate = {
        s: fit_gam(
            cache=cache, site=s, specs=[steps], cut=CUT, window=WINDOW, season_years=SEASON_YEARS
        )
        for s in ("S1", "S2")
    }
    assert separate_fits_equal_joint_at_zero_weight(joint=joint, separate=separate, atol=1e-5)


def test_the_bsp_rows_start_where_the_primaries_training_window_starts(two_sites: FeatureCache):
    # The 12-month window starts at 2023-07-01, so BSP readings before it must not enter the fit.
    bsp = _bsp(cache=two_sites, members=_members(), truth_members=("S1", "S2"))
    before = two_sites.end_times <= np.datetime64("2023-07-01T00:00", "us")
    changed = replace(bsp, reading_mw=np.where(before, 1e6, bsp.reading_mw))
    first = _fit(cache=two_sites, bsp=bsp, weights=[1.0])
    second = _fit(cache=two_sites, bsp=changed, weights=[1.0])
    assert np.array_equal(first.theta, second.theta)
    assert first.n_bsp_rows["B"] == second.n_bsp_rows["B"]


def test_one_assembly_solves_for_new_readings_exactly_as_a_fresh_fit_does(two_sites: FeatureCache):
    # The cached path: designs and Gram matrices built once, then solved for replaced readings,
    # must equal a fresh fit on a cache that holds those readings.
    bsp = _bsp(cache=two_sites, members=_members(), truth_members=("S1", "S2"))
    rng = np.random.default_rng(3)
    new_s1 = two_sites.sites["S1"].y + 0.4 * rng.normal(size=len(two_sites.end_times))
    new_reading = bsp.reading_mw + 1.5 * rng.normal(size=len(two_sites.end_times))
    assembly = assemble_joint(
        cache=two_sites,
        specs={"S1": SPEC, "S2": SPEC},
        bsps=[bsp],
        remainder_spec=RemainderSpec(step_penalty=1.0, shape_ridge=1e-3),
        sigma_points=SIGMA_POINTS,
        cut=CUT,
        window=WINDOW,
        season_years=SEASON_YEARS,
    )
    cached = solve_joint(
        assembly=assembly,
        weights=[0.0, 3.0],
        primary_power={"S1": new_s1},
        bsp_readings={"B": new_reading},
    )
    changed = replace(
        two_sites, sites={**two_sites.sites, "S1": replace(two_sites.sites["S1"], y=new_s1)}
    )
    fresh = _fit(cache=changed, bsp=replace(bsp, reading_mw=new_reading), weights=[0.0, 3.0])
    assert np.allclose(cached.theta, fresh.theta, rtol=1e-6, atol=1e-6)
    default = solve_joint(assembly=assembly, weights=[3.0])
    assert not np.allclose(default.theta[0], cached.theta[1], atol=1e-6)


def _pinned_capacity(*, cache: FeatureCache, site: str) -> PinnedCapacity:
    """Pin a site's capacity to a weekly vintage that rises from 1 MW at 0.01 MW a week."""
    weeks = []
    week = date(2019, 9, 2)
    while week <= CUT.date():
        weeks.append(week)
        week += timedelta(days=7)
    vintages = pl.DataFrame(
        {
            "time_series_id": cache.sites[site].table.time_series_id,
            "variant": "frozen",
            "origin_monday": CUT.date(),
            "week_start": weeks,
            "capacity_mw": [1.0 + 0.01 * i for i in range(len(weeks))],
            "unobserved": False,
        }
    )
    return pinned_capacity(vintages=vintages, table=cache.sites[site].table, cut=CUT)


def test_a_pinned_capacity_moves_the_primarys_solar_output_into_both_targets(
    two_sites: FeatureCache,
):
    # With a pinned capacity the model fits y + pinned solar = X theta, so net power y falls by the
    # solar output. The Primary rows and the BSP target both need that output added back, which the
    # plain-capacity tests never exercise.
    spec = GamSpec(drift="none", ridge=1e-3, capacity="pinned")
    pinned = {s: _pinned_capacity(cache=two_sites, site=s) for s in ("S1", "S2")}
    sites = {}
    for label, scale in (("S1", 1.0), ("S2", 0.7)):
        features = two_sites.sites[label]
        layout, _ = prepare_fit(
            cache=two_sites,
            site=label,
            spec=spec,
            cut=CUT,
            window=WINDOW,
            season_years=SEASON_YEARS,
        )
        theta = _truth_theta(layout=layout, scale=scale)
        usable = np.flatnonzero(
            np.isfinite(features.pv_unit_shape)
            & np.isfinite(features.smoothed_temperature_c).all(axis=1)
        )
        net = np.full(len(two_sites.end_times), np.nan)
        for start in range(0, len(usable), 20000):
            rows = usable[start : start + 20000]
            demand = design_matrix(cache=two_sites, features=features, layout=layout, index=rows)
            solar = pinned_solar_points(
                cache=two_sites, features=features, pinned=pinned[label], index=rows
            )
            net[rows] = demand @ theta - solar
        sites[label] = replace(features, y=net)
    cache = replace(two_sites, sites=sites)
    bsp = _bsp(cache=cache, members=_members(), truth_members=("S1", "S2"))
    joint = fit_joint(
        cache=cache,
        specs={"S1": spec, "S2": spec},
        bsps=[bsp],
        remainder_spec=RemainderSpec(step_penalty=1.0, shape_ridge=1e-3),
        weights=[0.0, 1.0],
        sigma_points=SIGMA_POINTS,
        cut=CUT,
        window=WINDOW,
        season_years=SEASON_YEARS,
        pinned=pinned,
    )
    separate = {
        s: fit_gam(
            cache=cache,
            site=s,
            specs=[spec],
            cut=CUT,
            window=WINDOW,
            season_years=SEASON_YEARS,
            pinned=pinned[s],
        )
        for s in ("S1", "S2")
    }
    assert separate_fits_equal_joint_at_zero_weight(joint=joint, separate=separate, atol=1e-5)
    index = bsp_rows(cache=cache, bsp=bsp, start=np.datetime64("2023-07-01T00:00", "us"), cut=CUT)
    predicted = predict_bsp(
        cache=cache,
        fit=joint,
        bsp=bsp,
        index=index,
        generators_mw=bsp.generators_mw[index],
        weight_index=1,
    )
    assert (
        np.abs(predicted - bsp.reading_mw[index]).max()
        < 1e-3 * np.abs(bsp.reading_mw[index]).mean()
    )


def test_an_mva_dark_member_enters_the_bsp_row_with_its_demand_terms_only(two_sites: FeatureCache):
    # S2 has a fitted PV capacity. As an `mva_dark` member its capacity columns are zeroed on the
    # BSP rows, so a BSP built from S2's demand terms alone is matched, and one that carried S2's
    # solar term would leave a residual.
    base = two_sites.sites["S2"]
    layout, _ = prepare_fit(
        cache=two_sites, site="S2", spec=SPEC, cut=CUT, window=WINDOW, season_years=SEASON_YEARS
    )
    theta = _truth_theta(layout=layout, scale=0.7)
    capacity = layout.slices["capacity"]
    theta[capacity.start + layout.season_years.index(2021)] = 6.0
    usable = np.flatnonzero(
        np.isfinite(base.pv_unit_shape) & np.isfinite(base.smoothed_temperature_c).all(axis=1)
    )
    net = np.full(len(two_sites.end_times), np.nan)
    demand_only = np.full(len(two_sites.end_times), np.nan)
    for start in range(0, len(usable), 20000):
        rows = usable[start : start + 20000]
        x = design_matrix(cache=two_sites, features=base, layout=layout, index=rows)
        net[rows] = x @ theta
        x[:, capacity] = 0.0
        demand_only[rows] = x @ theta
    cache = replace(two_sites, sites={**two_sites.sites, "S2": replace(base, y=net)})
    truth_cache = replace(two_sites, sites={**two_sites.sites, "S2": replace(base, y=demand_only)})
    members = (BspMember(site="S1"), BspMember(site="S2", kind="mva_dark"))
    bsp = _bsp(cache=truth_cache, members=members, truth_members=("S1", "S2"))
    joint = _fit(cache=cache, bsp=bsp, weights=[1e3])
    index = bsp_rows(cache=cache, bsp=bsp, start=np.datetime64("2023-07-01T00:00", "us"), cut=CUT)
    predicted = predict_bsp(
        cache=cache, fit=joint, bsp=bsp, index=index, generators_mw=bsp.generators_mw[index]
    )
    assert (
        np.abs(predicted - bsp.reading_mw[index]).mean()
        < 0.02 * np.abs(bsp.reading_mw[index]).mean()
    )
    fitted_capacity = joint.site_fit(site="S2").capacity_points()[2023]
    assert abs(fitted_capacity - 6.0) < 0.6


def test_a_window_per_primary_gives_the_same_fit_as_one_shared_window(two_sites: FeatureCache):
    bsp = _bsp(cache=two_sites, members=_members(), truth_members=("S1", "S2"))
    shared = _fit(cache=two_sites, bsp=bsp, weights=[1.0])
    per_site = fit_joint(
        cache=two_sites,
        specs={"S1": SPEC, "S2": SPEC},
        bsps=[bsp],
        remainder_spec=RemainderSpec(step_penalty=1.0, shape_ridge=1e-3),
        weights=[1.0],
        sigma_points=SIGMA_POINTS,
        cut=CUT,
        window={"S1": WINDOW, "S2": WINDOW},
        season_years=SEASON_YEARS,
    )
    assert np.array_equal(shared.theta, per_site.theta)
    narrower = fit_joint(
        cache=two_sites,
        specs={"S1": SPEC, "S2": SPEC},
        bsps=[bsp],
        remainder_spec=RemainderSpec(step_penalty=1.0, shape_ridge=1e-3),
        weights=[1.0],
        sigma_points=SIGMA_POINTS,
        cut=CUT,
        window={"S1": WINDOW, "S2": TrainingWindow(months=6)},
        season_years=SEASON_YEARS,
    )
    assert narrower.n_site_rows["S2"] < shared.n_site_rows["S2"]
    assert narrower.n_site_rows["S1"] == shared.n_site_rows["S1"]


def test_excluding_rows_from_a_primary_equals_fitting_without_those_readings(
    two_sites: FeatureCache,
):
    # With a spec whose penalties do not scale with the number of rows, leaving rows out of the
    # Primary's own loss equals a fresh fit in which those readings are missing. The BSP rows
    # keep them either way.
    bsp = _bsp(cache=two_sites, members=_members(), truth_members=("S1", "S2"))
    window = (two_sites.end_times > np.datetime64("2024-01-01T00:00", "us")) & (
        two_sites.end_times <= np.datetime64("2024-02-15T00:00", "us")
    )
    rows = np.flatnonzero(window)
    assembly = assemble_joint(
        cache=two_sites,
        specs={"S1": SPEC, "S2": SPEC},
        bsps=[bsp],
        remainder_spec=RemainderSpec(step_penalty=1.0, shape_ridge=1e-3),
        sigma_points=SIGMA_POINTS,
        cut=CUT,
        window=WINDOW,
        season_years=SEASON_YEARS,
    )
    cached = solve_joint(assembly=assembly, weights=[0.0, 5.0], excluded_rows={"S1": rows})
    missing = np.where(window, np.nan, two_sites.sites["S1"].y)
    cache = replace(
        two_sites, sites={**two_sites.sites, "S1": replace(two_sites.sites["S1"], y=missing)}
    )
    fresh = _fit(cache=cache, bsp=bsp, weights=[0.0, 5.0])
    assert np.allclose(cached.theta, fresh.theta, rtol=1e-6, atol=1e-6)
    assert not np.allclose(cached.theta, solve_joint(assembly=assembly, weights=[0.0, 5.0]).theta)
