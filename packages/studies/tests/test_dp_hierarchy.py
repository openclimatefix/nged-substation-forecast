"""Tests of the joint fit of Primaries and their bulk supply point (`studies.dp_hierarchy`).

The fixtures build a feature cache from synthetic weather with the real builder, copy the one
synthetic substation into two with different capacities, and replace their power with a truth made
from the model's own design. A bulk supply point (BSP) is then built exactly from the relation the
module fits, so a fit that recovers the truth shows the stacked design, the weights, and the
remainder agree.
"""

from dataclasses import replace
from datetime import UTC, datetime
from typing import Final

import numpy as np
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
    predict_gam,
    prepare_fit,
)
from studies.dp_hierarchy import (  # noqa: E402
    BspMember,
    BspSeries,
    RemainderSpec,
    bsp_rows,
    fit_joint,
    make_remainder_layout,
    predict_bsp,
    remainder_design,
    remainder_penalty,
    remainder_prediction,
    separate_fits_equal_joint_at_zero_weight,
)

CUT: Final[datetime] = datetime(2024, 7, 1, tzinfo=UTC)
WINDOW: Final[TrainingWindow] = TrainingWindow(months=12)
SEASON_YEARS: Final[range] = range(2019, 2026)
SPEC: Final[GamSpec] = GamSpec(drift="none", ridge=1e-3)
CAPACITY_MW: Final[dict[str, float]] = {"S1": 10.0, "S2": 25.0}
SIGMA_POINTS: Final[dict[str, float]] = {"S1": 0.5, "S2": 2.0}
TRUE_LEVEL_MW: Final[float] = 3.0
TRUE_PV_MW: Final[float] = 2.0
TRUE_HEATING_MW: Final[float] = 0.5
GENERATORS_MW: Final[float] = 4.0


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
            y[rows] = design_matrix(cache=cache, features=features, layout=layout, index=rows)
            y[rows] = y[rows] @ theta
        sites[label] = replace(features, y=y)
        thetas[label] = theta
    return replace(cache, sites=sites, node_ids={s: (s,) for s in sites}), thetas


def _bsp(
    *, cache: FeatureCache, members: tuple[BspMember, ...], truth_members: tuple[str, ...]
) -> BspSeries:
    """Build a BSP exactly from the module's relation, with the true remainder."""
    base = cache.sites["S1"]
    n = len(cache.end_times)
    hinge = np.clip(15.5 - base.temperature_c, 0.0, None) / 10.0
    remainder = TRUE_LEVEL_MW - TRUE_PV_MW * base.pv_unit_shape + TRUE_HEATING_MW * hinge
    reading = remainder - GENERATORS_MW
    for site in truth_members:
        reading = reading + CAPACITY_MW[site] * cache.sites[site].y
    return BspSeries(
        name="B",
        members=members,
        reading_mw=reading,
        generators_mw=np.full(n, GENERATORS_MW),
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


def _fit(*, cache: FeatureCache, bsp: BspSeries, weights: list[float]):
    return fit_joint(
        cache=cache,
        specs={"S1": SPEC, "S2": SPEC},
        bsps=[bsp],
        remainder_spec=RemainderSpec(step_penalty=1.0, shape_ridge=1e-3),
        weights=weights,
        sigma_points=SIGMA_POINTS,
        cut=CUT,
        window=WINDOW,
        season_years=SEASON_YEARS,
    )


def test_at_weight_zero_the_joint_fit_equals_the_separate_fits_whatever_the_sigmas(
    two_sites: FeatureCache,
):
    # P1. The sigmas differ by a factor of 4, so a joint block that scaled the loss but not the
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
    assert separate_fits_equal_joint_at_zero_weight(joint=joint, separate=separate, member=0)
    # The check can fail: with the BSP switched on, the Primaries' coefficients move.
    assert not np.allclose(
        joint.theta[1, joint.site_slices["S1"]], separate["S1"].theta[0], atol=1e-6
    )


def test_the_weight_zero_check_refuses_a_member_with_another_weight(two_sites: FeatureCache):
    bsp = _bsp(cache=two_sites, members=_members(), truth_members=("S1", "S2"))
    joint = _fit(cache=two_sites, bsp=bsp, weights=[0.0, 1.0])
    with pytest.raises(ValueError, match="not 0"):
        separate_fits_equal_joint_at_zero_weight(joint=joint, separate={}, member=1)


def test_a_noise_free_bsp_built_from_the_relation_is_recovered_at_weight_one(
    two_sites: FeatureCache,
):
    # P2.
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
        < 0.05 * np.abs(bsp.reading_mw[index]).mean()
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
    # The sabotage reading: the audit has to be able to see a missing Primary. The BSP is built with
    # both Primaries, but only S1 is declared under it, so S2's demand must land in the remainder.
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


def test_a_scale_error_at_one_primary_does_less_damage_with_the_bsp_hint(two_sites: FeatureCache):
    # The positive control for the claim: a 30% scale error on 90 days of S1's training readings.
    bsp = _bsp(cache=two_sites, members=_members(), truth_members=("S1", "S2"))
    s1 = two_sites.sites["S1"]
    stamps = two_sites.end_times
    window = (stamps > np.datetime64("2024-01-01T00:00", "us")) & (
        stamps <= np.datetime64("2024-03-31T00:00", "us")
    )
    corrupted = replace(
        two_sites,
        sites={**two_sites.sites, "S1": replace(s1, y=np.where(window, 1.3 * s1.y, s1.y))},
    )
    joint = _fit(cache=corrupted, bsp=bsp, weights=[0.0, 30.0])
    index = bsp_rows(
        cache=two_sites, bsp=bsp, start=np.datetime64("2023-07-01T00:00", "us"), cut=CUT
    )[::50]
    clean = s1.y[index]
    damage = []
    for member in (0, 1):
        predicted = predict_gam(
            cache=two_sites, fit=joint.site_fit(site="S1"), index=index, member=member
        )
        damage.append(np.abs(predicted - clean).mean())
    assert damage[1] < 0.7 * damage[0]


def test_no_fit_reads_a_row_after_the_cut(two_sites: FeatureCache):
    bsp = _bsp(cache=two_sites, members=_members(), truth_members=("S1", "S2"))
    after = two_sites.end_times > np.datetime64("2024-07-01T00:00", "us")
    changed = replace(bsp, reading_mw=np.where(after, 1e6, bsp.reading_mw))
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


def test_the_stacked_primary_block_is_the_separate_gram_matrix_scaled_by_its_sigma(
    two_sites: FeatureCache,
):
    # P1's block-by-block half: at weight 0 the normal equations of a Primary's block are the
    # separate fit's, divided by sigma squared. The solution check above cannot see a Gram error
    # that the constraints hide, so compare the matrices through a one-column perturbation.
    layout, index = prepare_fit(
        cache=two_sites, site="S2", spec=SPEC, cut=CUT, window=WINDOW, season_years=SEASON_YEARS
    )
    gram, rhs = accumulate_gram(
        cache=two_sites, features=two_sites.sites["S2"], layout=layout, index=index
    )
    x = design_matrix(cache=two_sites, features=two_sites.sites["S2"], layout=layout, index=index)
    y = two_sites.sites["S2"].y[index]
    assert np.allclose(gram[0].numpy(), x.T @ x, rtol=1e-9)
    assert np.allclose(rhs[0].numpy(), x.T @ y, rtol=1e-9)


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
    kwargs = {
        "cache": two_sites,
        "specs": {"S1": SPEC, "S2": SPEC},
        "remainder_spec": RemainderSpec(),
        "weights": [1.0],
        "cut": CUT,
        "window": WINDOW,
        "season_years": SEASON_YEARS,
    }
    with pytest.raises(ValueError, match="sits under both"):
        fit_joint(bsps=[first, second], sigma_points=SIGMA_POINTS, **kwargs)
    with pytest.raises(ValueError, match="no positive sigma"):
        fit_joint(bsps=[first], sigma_points={"S1": 0.5, "S2": 0.0}, **kwargs)
    with pytest.raises(ValueError, match="sigma that is not positive"):
        fit_joint(bsps=[replace(first, sigma_mw=0.0)], sigma_points=SIGMA_POINTS, **kwargs)
