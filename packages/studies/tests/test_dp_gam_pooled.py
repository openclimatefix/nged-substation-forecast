"""Tests of the partially pooled convex model (`studies.dp_gam_pooled`).

The fixture clones the synthetic substation of `conftest.py` into seven substations whose power is
made from a known fleet mean plus known deviations, so the tests can compare the joint fit with
independent dense computations and with `dp_gam.fit_gam`.
"""

import time
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from typing import Final

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")

from studies.dp_calendar import BAND_FIRST_HOURS, DAY_TYPES, HALF_HOURS_PER_DAY  # noqa: E402
from studies.dp_features import FeatureCache  # noqa: E402
from studies.dp_gam import (  # noqa: E402
    N_BANDS,
    GamLayout,
    GamSpec,
    ReplicaResiduals,
    TrainingWindow,
    design_matrix,
    fit_gam,
    penalty_matrix,
    predict_gam,
    prepare_fit,
    replica_cut,
    training_rows,
)
from studies.dp_gam_pooled import (  # noqa: E402
    POOLED_FAMILIES,
    PooledFamilyType,
    PooledSite,
    PoolingStrength,
    ShortHistoryReplica,
    centring_matrix,
    fit_pooled,
    pooled_replica_residuals,
    rebuilt_effective_capacity_mw,
    short_history_replica,
    short_history_spread,
    with_effective_capacity,
)
from studies.dp_pinned_capacity import PinnedCapacity, pinned_capacity  # noqa: E402

CUT: Final[datetime] = datetime(2024, 7, 1, tzinfo=UTC)
SEASON_YEARS: Final[tuple[int, ...]] = tuple(range(2019, 2026))
LABELS: Final[tuple[str, ...]] = ("D1", "D2", "D3", "D4", "D5", "D6", "D7")
WINDOW: Final[TrainingWindow] = TrainingWindow(months=24)
NOISE_POINTS: Final[float] = 0.3
EXPECTED_FLEET_MEAN_RIDGE: Final[float] = 1e-3
"""Written out here, so a change of the module's constant is caught by the dense references."""


def _site_truth(*, n_columns: int, layout: GamLayout, generator: np.random.Generator) -> np.ndarray:
    """Return one substation's coefficients: a fleet part shared by every site plus deviations."""
    fleet = np.random.default_rng(100)
    theta = np.zeros(n_columns)
    half_hours = np.arange(HALF_HOURS_PER_DAY)
    shape = 40.0 + 6.0 * np.sin(2 * np.pi * half_hours / HALF_HOURS_PER_DAY - 1.0)
    profile = layout.slices["profile"]
    for day_type, offset in enumerate((0.0, -4.0, -6.0, -6.0)):
        start = profile.start + day_type * HALF_HOURS_PER_DAY
        theta[start : start + HALF_HOURS_PER_DAY] = shape + offset
    holiday = layout.slices["holiday"]
    theta[holiday] = 0.5 * fleet.normal(size=holiday.stop - holiday.start)
    heating = layout.slices["heating"]
    theta[heating] = 0.6 + 0.4 * fleet.uniform(size=heating.stop - heating.start)
    theta[layout.slices["lighting"]] = (0.5, 0.3, 0.2, 0.1)
    capacity = layout.slices["capacity"]
    for year, increment in {2020: 8.0, 2022: 10.0, 2023: 5.0}.items():
        theta[capacity.start + layout.season_years.index(year)] = increment
    for name, scale in (("heating", 0.15), ("holiday", 0.2), ("harmonics", 0.2)):
        block = layout.slices[name]
        theta[block] += scale * generator.normal(size=block.stop - block.start)
    theta[heating] = np.abs(theta[heating])
    theta[profile] += generator.normal(scale=3.0)  # a site's size: the free band means
    return theta


@pytest.fixture(scope="module")
def pooled_cache(dp_synthetic_cache: FeatureCache) -> FeatureCache:
    """Seven clones of the synthetic substation, each with its own known coefficients."""
    base = dp_synthetic_cache.sites["S1"]
    layout, _ = prepare_fit(
        cache=dp_synthetic_cache,
        site="S1",
        spec=GamSpec(),
        cut=CUT,
        window=TrainingWindow(),
        season_years=SEASON_YEARS,
    )
    usable = np.flatnonzero(
        np.isfinite(base.pv_unit_shape) & np.isfinite(base.smoothed_temperature_c).all(axis=1)
    )
    sites = {}
    for number, label in enumerate(LABELS):
        generator = np.random.default_rng(number)
        theta = _site_truth(n_columns=layout.n_columns, layout=layout, generator=generator)
        y = np.full(len(dp_synthetic_cache.end_times), np.nan)
        for start in range(0, len(usable), 20000):
            rows = usable[start : start + 20000]
            y[rows] = (
                design_matrix(cache=dp_synthetic_cache, features=base, layout=layout, index=rows)
                @ theta
            )
        y = y + NOISE_POINTS * generator.normal(size=len(y))
        table = replace(base.table, site=label, time_series_id=number + 1)
        sites[label] = replace(base, table=table, y=y)
    return replace(
        dp_synthetic_cache,
        sites=sites,
        node_ids={label: (label,) for label in LABELS},
    )


def _sites(*, labels: tuple[str, ...] = LABELS, ridge: float = 1.0) -> list[PooledSite]:
    return [
        PooledSite(
            site=label,
            spec=GamSpec(ridge=ridge),
            window=WINDOW,
            season_years=SEASON_YEARS,
        )
        for label in labels
    ]


def _pinned_for(
    *, cache: FeatureCache, site: str, cut: datetime, origin_after_cut_days: int = 0
) -> PinnedCapacity:
    weeks = []
    week = date(2019, 9, 2)
    while week <= cut.date():
        weeks.append(week)
        week += timedelta(days=7)
    vintages = pl.DataFrame(
        {
            "time_series_id": cache.sites[site].table.time_series_id,
            "variant": "frozen",
            "origin_monday": cut.date() + timedelta(days=origin_after_cut_days),
            "week_start": weeks,
            "capacity_mw": [8.0 + 0.002 * i for i in range(len(weeks))],
            "unobserved": False,
        }
    )
    return pinned_capacity(vintages=vintages, table=cache.sites[site].table, cut=cut)


def _fit_dense(
    *, cache: FeatureCache, sites: list[PooledSite], heating: float, other: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Return the joint solution from an independent dense computation, and its pieces.

    Builds the normal equations from the design matrices directly, with the deviation penalty
    written as an explicit difference operator per substation and family.
    """
    designs, targets, penalties, layouts = [], [], [], []
    for site in sites:
        layout, index = prepare_fit(
            cache=cache,
            site=site.site,
            spec=site.spec,
            cut=CUT,
            window=site.window,
            season_years=site.season_years,
        )
        x = design_matrix(cache=cache, features=cache.sites[site.site], layout=layout, index=index)
        designs.append(x)
        targets.append(cache.sites[site.site].y[index])
        penalties.append(penalty_matrix(layout=layout, n_rows=len(index))[0])
        layouts.append(layout)
    sizes = [layout.n_columns for layout in layouts]
    starts = np.concatenate([[0], np.cumsum(sizes)])
    family_sizes = {
        f: layouts[0].slices[f].stop - layouts[0].slices[f].start for f in POOLED_FAMILIES
    }
    n_mu = sum(family_sizes.values())
    total = int(starts[-1]) + n_mu
    total_rows = sum(len(x) for x in designs)
    hessian = np.zeros((total, total))
    rhs = np.zeros(total)
    constrained = np.zeros(total, dtype=bool)
    for x, y, penalty, layout, start in zip(
        designs, targets, penalties, layouts, starts, strict=False
    ):
        stop = int(start) + layout.n_columns
        hessian[start:stop, start:stop] += x.T @ x + penalty
        rhs[start:stop] += x.T @ y
        constrained[start:stop] = layout.constrained
    mu_at = int(starts[-1])
    cursor = mu_at
    scales = {}
    for family in POOLED_FAMILIES:
        size = family_sizes[family]
        columns = layouts[0].slices[family]
        g = sum(np.diag(x.T @ x)[columns].sum() for x in designs) / (size * total_rows)
        scales[family] = g
        weight = (heating if family == "heating" else other) * g
        centre = centring_matrix(family=family, size=size)
        for start in starts[:-1]:
            difference = np.zeros((size, total))
            difference[:, int(start) + columns.start : int(start) + columns.stop] = centre
            difference[:, cursor : cursor + size] = -centre
            hessian += weight * difference.T @ difference
        cursor += size
    hessian[mu_at:, mu_at:] += EXPECTED_FLEET_MEAN_RIDGE * np.eye(n_mu)
    return hessian, rhs, constrained, starts, scales


def _solve_on_active_set(
    *, hessian: np.ndarray, rhs: np.ndarray, constrained: np.ndarray, reference_theta: np.ndarray
) -> np.ndarray:
    """Solve the dense problem with the columns the reference holds at their bound fixed at zero."""
    fixed = constrained & (reference_theta <= 0.0)
    free = ~fixed
    solution = np.zeros(len(rhs))
    solution[free] = np.linalg.solve(hessian[np.ix_(free, free)], rhs[free])
    return solution


def test_lambda_zero_equals_fit_gam_to_1e_minus_8(pooled_cache: FeatureCache):
    labels = ("D1", "D2", "D3")
    pinned = _pinned_for(cache=pooled_cache, site="D3", cut=CUT)
    sites = _sites(labels=labels)
    sites[2] = replace(sites[2], spec=GamSpec(capacity="pinned"), pinned=pinned)
    joint = fit_pooled(
        cache=pooled_cache, sites=sites, cut=CUT, strengths=[PoolingStrength.uniform(value=0.0)]
    )
    for site in sites:
        alone = fit_gam(
            cache=pooled_cache,
            site=site.site,
            specs=[site.spec],
            cut=CUT,
            window=site.window,
            season_years=site.season_years,
            pinned=site.pinned,
        )
        assert np.abs(joint.fits[site.site].theta - alone.theta).max() < 1e-8
    assert np.abs(joint.fleet_mean).max() < 1e-12


def test_the_joint_solution_is_optimal_and_equals_a_dense_ridge_on_its_active_set(
    pooled_cache: FeatureCache,
):
    sites = _sites(labels=("D1", "D2", "D3", "D4"))
    strength = PoolingStrength(heating=3000.0, other=1000.0)
    joint = fit_pooled(cache=pooled_cache, sites=sites, cut=CUT, strengths=[strength])
    hessian, rhs, constrained, _, expected_scales = _fit_dense(
        cache=pooled_cache, sites=sites, heating=3000.0, other=1000.0
    )
    solved = np.concatenate(
        [joint.fits[site.site].theta[0] for site in sites] + [joint.fleet_mean[0]]
    )
    dense = _solve_on_active_set(
        hessian=hessian, rhs=rhs, constrained=constrained, reference_theta=solved
    )
    assert np.abs(dense - solved).max() < 1e-8
    # Independent optimality conditions of the dense problem: constrained entries are non-negative,
    # the gradient is zero on free entries, and non-negative at a bound.
    gradient = hessian @ dense - rhs
    at_bound = constrained & (dense <= 0.0)
    free = ~at_bound
    assert (dense[constrained] >= 0.0).all()
    assert np.abs(gradient[free]).max() < 1e-5
    assert (gradient[at_bound] > -1e-5).all()
    assert at_bound.any()  # the heating box binds in this fit, so the test exercises it
    for family, scale in joint.scales.items():
        assert scale == pytest.approx(expected_scales[family])
    assert joint.fits["D1"].violation.max() < 1e-7


def _active_set_solution(
    *, hessian: np.ndarray, rhs: np.ndarray, constrained: np.ndarray
) -> np.ndarray:
    """Minimise the quadratic, holding at 0 each constrained column that comes out negative."""
    fixed = np.zeros(len(rhs), dtype=bool)
    for _ in range(50):
        solution = np.zeros(len(rhs))
        solution[~fixed] = np.linalg.solve(hessian[np.ix_(~fixed, ~fixed)], rhs[~fixed])
        broken = constrained & ~fixed & (solution < 0.0)
        if not broken.any():
            return solution
        fixed |= broken
    msg = "the reference active set did not settle"
    raise AssertionError(msg)


def _tied_mapping(
    *,
    layout: GamLayout,
    sizes: Mapping[PooledFamilyType, int],
    shared_start: Mapping[PooledFamilyType, int],
    n_shared: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the matrix from a site's free columns to its theta, and its non-pooled columns.

    Written with explicit loops, apart from the module's vectorised construction.
    """
    band_of_half_hour = np.searchsorted(np.array(BAND_FIRST_HOURS), np.arange(48) // 2, "right") - 1
    band_of_column = np.tile(band_of_half_hour, len(DAY_TYPES))
    pooled = np.zeros(layout.n_columns, dtype=bool)
    for family in POOLED_FAMILIES:
        pooled[layout.slices[family]] = True
    own = np.flatnonzero(~pooled)
    mapping = np.zeros((layout.n_columns, n_shared + N_BANDS + len(own)))
    for family in POOLED_FAMILIES:
        rows = layout.slices[family]
        for k in range(sizes[family]):
            mapping[rows.start + k, shared_start[family] + k] = 1.0
    profile = layout.slices["profile"]
    for j in range(sizes["profile"]):
        for k in range(sizes["profile"]):  # theta = C tau + band offset
            same_band = band_of_column[j] == band_of_column[k]
            mapping[profile.start + j, shared_start["profile"] + k] -= same_band / np.sum(
                band_of_column == band_of_column[k]
            )
        mapping[profile.start + j, n_shared + band_of_column[j]] = 1.0
    mapping[own, n_shared + N_BANDS + np.arange(len(own))] = 1.0
    return mapping, own


def _tied_dense(
    *, cache: FeatureCache, sites: list[PooledSite]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return each substation's theta from a separately built complete-pooling fit."""
    layouts, designs, targets, penalties = {}, {}, {}, {}
    for site in sites:
        layout, index = prepare_fit(
            cache=cache,
            site=site.site,
            spec=site.spec,
            cut=CUT,
            window=site.window,
            season_years=site.season_years,
        )
        layouts[site.site] = layout
        designs[site.site] = design_matrix(
            cache=cache, features=cache.sites[site.site], layout=layout, index=index
        )
        targets[site.site] = cache.sites[site.site].y[index]
        penalties[site.site] = penalty_matrix(layout=layout, n_rows=len(index))[0]
    first = layouts[sites[0].site]
    sizes = {f: first.slices[f].stop - first.slices[f].start for f in POOLED_FAMILIES}
    shared_start = {}
    cursor = 0
    for family in POOLED_FAMILIES:
        shared_start[family] = cursor
        cursor += sizes[family]
    n_shared = cursor
    maps = {}
    non_pooled = {}
    for site in sites:
        maps[site.site], non_pooled[site.site] = _tied_mapping(
            layout=layouts[site.site], sizes=sizes, shared_start=shared_start, n_shared=n_shared
        )
    widths = {label: m.shape[1] - n_shared for label, m in maps.items()}
    total = n_shared + sum(widths.values())
    hessian = np.zeros((total, total))
    rhs = np.zeros(total)
    constrained = np.zeros(total, dtype=bool)
    constrained[shared_start["heating"] : shared_start["heating"] + sizes["heating"]] = True
    cursor = n_shared
    placement = {}
    for site in sites:
        label = site.site
        columns = np.concatenate([np.arange(n_shared), cursor + np.arange(widths[label])])
        placement[label] = columns
        x_mapped = designs[label] @ maps[label]
        hessian[np.ix_(columns, columns)] += (
            x_mapped.T @ x_mapped + maps[label].T @ penalties[label] @ maps[label]
        )
        rhs[columns] += x_mapped.T @ targets[label]
        constrained[cursor + N_BANDS : cursor + widths[label]] = layouts[label].constrained[
            non_pooled[label]
        ]
        cursor += widths[label]
    hessian[:n_shared, :n_shared] += EXPECTED_FLEET_MEAN_RIDGE * np.eye(n_shared)
    solution = _active_set_solution(hessian=hessian, rhs=rhs, constrained=constrained)
    return {label: maps[label] @ solution[placement[label]] for label in maps}, solution[:n_shared]


def test_tying_equals_a_separately_built_complete_pooling_fit_and_large_lambda_approaches_it(
    pooled_cache: FeatureCache,
):
    sites = _sites(labels=("D1", "D2", "D3", "D4"))
    tied = fit_pooled(
        cache=pooled_cache,
        sites=sites,
        cut=CUT,
        tied=True,
    )
    dense, shared = _tied_dense(cache=pooled_cache, sites=sites)
    for site in sites:
        assert np.abs(tied.fits[site.site].theta[0] - dense[site.site]).max() < 1e-6
    assert np.abs(tied.fleet_mean[0] - shared).max() < 1e-6

    # Lambda counts rows and a two-year site holds about 35,000, so 1e4 is not yet in the regime
    # where the gap to complete pooling falls as 1 / lambda: that starts once lambda times the
    # family scale exceeds the largest eigenvalue of each family's Gram matrix.
    mid, rate, large = 1e4, 1e7, 1e8
    joint = fit_pooled(
        cache=pooled_cache,
        sites=sites,
        cut=CUT,
        strengths=[PoolingStrength.uniform(value=v) for v in (mid, rate, large)],
    )
    pooled_columns = []
    layout = joint.fits["D1"].layout
    for family in POOLED_FAMILIES:
        pooled_columns.extend(range(layout.slices[family].start, layout.slices[family].stop))

    def gap(member: int) -> float:
        return max(
            np.abs(
                joint.fits[s.site].theta[member, pooled_columns]
                - tied.fits[s.site].theta[0, pooled_columns]
            ).max()
            for s in sites
        )

    assert gap(2) < 0.01 * gap(0)
    assert (
        0.08 < gap(2) / gap(1) < 0.12
    )  # the 1 / lambda rate: ten times the weight, a tenth the gap


def test_a_held_out_sites_readings_outside_its_window_change_nothing(pooled_cache: FeatureCache):
    window = TrainingWindow(start=datetime(2024, 6, 3, tzinfo=UTC))
    sites = _sites(labels=("D1", "D2", "D3"))
    sites[2] = replace(sites[2], window=window)
    strengths = [PoolingStrength.uniform(value=1000.0)]
    before = fit_pooled(cache=pooled_cache, sites=sites, cut=CUT, strengths=strengths)
    features = pooled_cache.sites["D3"]
    outside = pooled_cache.end_times <= np.datetime64("2024-06-03T00:00:00", "us")
    poisoned = replace(
        features,
        y=np.where(outside, 500.0, features.y),
        temperature_c=np.where(outside, -40.0, features.temperature_c),
        smoothed_temperature_c=np.where(outside[:, None], -40.0, features.smoothed_temperature_c),
    )
    other = replace(pooled_cache, sites={**pooled_cache.sites, "D3": poisoned})
    after = fit_pooled(cache=other, sites=sites, cut=CUT, strengths=strengths)
    for label in ("D1", "D2", "D3"):
        assert np.array_equal(before.fits[label].theta, after.fits[label].theta)
    assert np.array_equal(before.fleet_mean, after.fleet_mean)
    assert before.scales == after.scales
    assert before.fits["D3"].n_rows < before.fits["D1"].n_rows / 10


def test_nothing_after_the_cut_reaches_the_fit_and_a_pinned_capacity_for_another_cut_is_refused(
    pooled_cache: FeatureCache,
):
    sites = _sites(labels=("D1", "D2"))
    strengths = [PoolingStrength.uniform(value=1000.0)]
    before = fit_pooled(cache=pooled_cache, sites=sites, cut=CUT, strengths=strengths)
    late = pooled_cache.end_times > np.datetime64("2024-07-01T00:00:00", "us")
    changed = {}
    for label in ("D1", "D2"):
        features = pooled_cache.sites[label]
        changed[label] = replace(
            features,
            y=np.where(late, 900.0, features.y),
            smoothed_temperature_c=np.where(late[:, None], -30.0, features.smoothed_temperature_c),
        )
    after = fit_pooled(
        cache=replace(pooled_cache, sites={**pooled_cache.sites, **changed}),
        sites=sites,
        cut=CUT,
        strengths=strengths,
    )
    assert np.array_equal(before.fleet_mean, after.fleet_mean)
    assert before.scales == after.scales
    for label in ("D1", "D2"):
        assert np.array_equal(before.fits[label].theta, after.fits[label].theta)

    earlier = _pinned_for(cache=pooled_cache, site="D2", cut=datetime(2024, 6, 3, tzinfo=UTC))
    pinned_sites = [sites[0], replace(sites[1], spec=GamSpec(capacity="pinned"), pinned=earlier)]
    with pytest.raises(ValueError, match="pinned capacity was built for"):
        fit_pooled(cache=pooled_cache, sites=pinned_sites, cut=CUT, strengths=strengths)
    with pytest.raises(ValueError, match="pinned capacity is required"):
        fit_pooled(
            cache=pooled_cache,
            sites=[replace(sites[0], pinned=earlier), sites[1]],
            cut=CUT,
            strengths=strengths,
        )


def test_a_pinned_capacity_reaches_another_site_only_through_the_fleet_mean(
    pooled_cache: FeatureCache,
):
    sites = _sites(labels=("D1", "D2", "D3"))
    other_capacity = replace(
        _pinned_for(cache=pooled_cache, site="D3", cut=CUT),
        points=np.full(len(_pinned_for(cache=pooled_cache, site="D3", cut=CUT).points), 40.0),
        held_points=40.0,
    )

    def fit(*, pinned: PinnedCapacity, value: float) -> dict[str, np.ndarray]:
        mixed = [
            sites[0],
            sites[1],
            replace(sites[2], spec=GamSpec(capacity="pinned"), pinned=pinned),
        ]
        joint = fit_pooled(
            cache=pooled_cache,
            sites=mixed,
            cut=CUT,
            strengths=[PoolingStrength.uniform(value=value)],
        )
        assert joint.fits["D3"].layout.slices["capacity"].stop == (
            joint.fits["D3"].layout.slices["capacity"].start
        )
        return {label: joint.fits[label].theta[0] for label in ("D1", "D2", "D3")}

    base = _pinned_for(cache=pooled_cache, site="D3", cut=CUT)
    alone_before = fit(pinned=base, value=0.0)
    alone_after = fit(pinned=other_capacity, value=0.0)
    assert np.array_equal(alone_before["D1"], alone_after["D1"])
    assert np.array_equal(alone_before["D2"], alone_after["D2"])
    assert not np.allclose(alone_before["D3"], alone_after["D3"])
    pooled_before = fit(pinned=base, value=1000.0)
    pooled_after = fit(pinned=other_capacity, value=1000.0)
    assert np.abs(pooled_before["D1"] - pooled_after["D1"]).max() > 1e-6


def test_a_larger_lambda_pulls_the_sites_heating_slopes_together(pooled_cache: FeatureCache):
    strengths = [PoolingStrength.uniform(value=v) for v in (0.0, 1e3, 1e5)]
    joint = fit_pooled(
        cache=pooled_cache, sites=_sites(labels=LABELS[:5]), cut=CUT, strengths=strengths
    )
    heating = joint.fits["D1"].layout.slices["heating"]
    spread = [
        np.std([joint.fits[s].theta[member, heating] for s in LABELS[:5]], axis=0).mean()
        for member in range(3)
    ]
    assert spread[0] > spread[1] > spread[2]
    assert all(fit.violation.max() < 1e-7 for fit in joint.fits.values())


def test_the_profile_centring_keeps_each_hour_bands_mean_free():
    centre = centring_matrix(family="profile", size=len(DAY_TYPES) * HALF_HOURS_PER_DAY)
    assert np.allclose(centre, centre.T)
    assert np.allclose(centre @ centre, centre)
    assert np.linalg.matrix_rank(np.eye(len(centre)) - centre) == N_BANDS
    assert np.array_equal(centring_matrix(family="heating", size=16), np.eye(16))


def test_a_joint_fit_of_seven_sites_and_nine_lambdas_takes_under_two_minutes(
    pooled_cache: FeatureCache,
):
    strengths = [
        PoolingStrength.uniform(value=v) for v in (0.0, 100.0, 300.0, 1e3, 3e3, 1e4, 3e4, 1e5, 1e6)
    ]
    threads = torch.get_num_threads()
    torch.set_num_threads(6)
    try:
        started = time.monotonic()
        joint = fit_pooled(cache=pooled_cache, sites=_sites(), cut=CUT, strengths=strengths)
        elapsed = time.monotonic() - started
    finally:
        torch.set_num_threads(threads)
    assert elapsed < 120.0
    assert joint.fits["D1"].theta.shape[0] == 9


def test_a_joint_fit_refuses_a_drift_a_repeated_label_and_no_sites(pooled_cache: FeatureCache):
    strengths = [PoolingStrength.uniform(value=1.0)]
    drifting = [replace(_sites(labels=("D1",))[0], spec=GamSpec(drift="d1"))]
    with pytest.raises(ValueError, match="drift 'none'"):
        fit_pooled(cache=pooled_cache, sites=drifting, cut=CUT, strengths=strengths)
    with pytest.raises(ValueError, match="each label once"):
        fit_pooled(
            cache=pooled_cache, sites=_sites(labels=("D1", "D1")), cut=CUT, strengths=strengths
        )
    with pytest.raises(ValueError, match="at least one substation"):
        fit_pooled(cache=pooled_cache, sites=[], cut=CUT, strengths=strengths)


def test_the_replica_is_fitted_at_the_replica_cut_and_scored_on_the_year_after_it(
    pooled_cache: FeatureCache,
):
    replica_sites = _sites(labels=("D1", "D2"))
    strength = PoolingStrength.uniform(value=1000.0)
    replicas = pooled_replica_residuals(
        cache=pooled_cache, sites=replica_sites, cut=CUT, strength=strength
    )
    fit_cut = replica_cut(cut=CUT)
    independent = fit_pooled(
        cache=pooled_cache, sites=replica_sites, cut=fit_cut, strengths=[strength]
    )
    for label in ("D1", "D2"):
        replica = replicas[label]
        ends = pooled_cache.end_times[replica.residuals.index]
        assert replica.residuals.fit_cut == fit_cut
        assert ends.min() > np.datetime64("2023-07-01T00:00:00", "us")
        assert ends.max() <= np.datetime64("2024-07-01T00:00:00", "us")
        # The residuals are the actual minus the forecast of a fit at the replica cut, so a replica
        # fitted at the final cut (which would train on these very rows) gives other numbers.
        expected = pooled_cache.sites[label].y[replica.residuals.index] - predict_gam(
            cache=pooled_cache, fit=independent.fits[label], index=replica.residuals.index
        )
        assert np.abs(replica.residuals.residuals - expected).max() < 1e-10
        assert replica.n_training_rows == independent.fits[label].n_rows
        assert independent.fits[label].cut == fit_cut


def _short_replica(
    *, short_site: str, weeks: int, scale: float, rows: int = 1000
) -> ShortHistoryReplica:
    generator = np.random.default_rng(sum(ord(character) for character in short_site))
    n = 2000
    residuals = ReplicaResiduals(
        residuals=scale * generator.normal(size=n),
        band=generator.integers(0, N_BANDS, size=n).astype(np.int8),
        quarter=generator.integers(0, 4, size=n).astype(np.int8),
        index=np.arange(n),
        fit_cut=datetime(2023, 7, 1, tzinfo=UTC),
        switching=np.zeros(n, dtype=bool),
    )
    return ShortHistoryReplica(
        short_site=short_site,
        weeks_of_history=weeks,
        residuals=residuals,
        n_training_rows=rows,
    )


def test_the_short_history_spread_uses_only_other_sites_short_replicas():
    replicas = [
        _short_replica(short_site="D2", weeks=4, scale=2.0),
        _short_replica(short_site="D3", weeks=4, scale=2.0),
    ]
    spread = short_history_spread(replicas=replicas, held_out="D1", weeks_of_history=4)
    assert 1.5 < spread.table[0, 0, -1] < 6.0
    assert abs(spread.table[0, 0, 5]) < 1e-12  # centred: the median level is zero
    with pytest.raises(ValueError, match="cannot make D2's own spread"):
        short_history_spread(replicas=replicas, held_out="D2", weeks_of_history=4)
    with pytest.raises(ValueError, match="not 8"):
        short_history_spread(replicas=replicas, held_out="D1", weeks_of_history=8)
    with pytest.raises(ValueError, match="no other substation"):
        short_history_spread(replicas=[], held_out="D1", weeks_of_history=4)
    full_history = _short_replica(short_site="D4", weeks=4, scale=2.0, rows=30_000)
    with pytest.raises(ValueError, match="not a short-history fit"):
        short_history_spread(replicas=[full_history], held_out="D1", weeks_of_history=4)


def test_a_short_history_entry_takes_its_training_rows_from_the_replica_fit(
    pooled_cache: FeatureCache,
):
    window = TrainingWindow(start=datetime(2023, 6, 3, tzinfo=UTC))
    sites = _sites(labels=("D1", "D2"))
    sites[1] = replace(sites[1], window=window)
    replicas = pooled_replica_residuals(
        cache=pooled_cache, sites=sites, cut=CUT, strength=PoolingStrength.uniform(value=1000.0)
    )
    entry = short_history_replica(replicas=replicas, short_site="D2", weeks_of_history=4)
    assert entry.n_training_rows == replicas["D2"].n_training_rows
    assert entry.n_training_rows <= 4 * 336
    full = short_history_replica(replicas=replicas, short_site="D1", weeks_of_history=4)
    with pytest.raises(ValueError, match="not a short-history fit"):
        short_history_spread(replicas=[full], held_out="D3", weeks_of_history=4)


def test_a_late_vintage_capacity_is_refused_unless_the_arm_is_an_oracle(
    pooled_cache: FeatureCache,
):
    late = _pinned_for(cache=pooled_cache, site="D2", cut=CUT, origin_after_cut_days=120)
    assert late.source == "growth_prior"
    strengths = [PoolingStrength.uniform(value=1000.0)]
    sites = _sites(labels=("D1", "D2"))
    pinned_site = replace(sites[1], spec=GamSpec(capacity="pinned"), pinned=late)
    with pytest.raises(ValueError, match="fitted long after"):
        fit_pooled(cache=pooled_cache, sites=[sites[0], pinned_site], cut=CUT, strengths=strengths)
    oracle = replace(pinned_site, allow_late_vintage=True)
    joint = fit_pooled(cache=pooled_cache, sites=[sites[0], oracle], cut=CUT, strengths=strengths)
    assert joint.fits["D2"].pinned is late


def test_a_rebuilt_effective_capacity_moves_the_target_and_the_table_together(
    pooled_cache: FeatureCache,
):
    window = TrainingWindow(start=datetime(2024, 6, 3, tzinfo=UTC))
    before = pooled_cache.sites["D1"]
    rebuilt = rebuilt_effective_capacity_mw(cache=pooled_cache, site="D1", cut=CUT, window=window)
    rows = training_rows(features=before, cache=pooled_cache, cut=CUT, window=window)
    expected = np.quantile(np.abs(before.y[rows]) * before.table.effective_capacity_mw / 100, 0.99)
    assert rebuilt == pytest.approx(expected)

    # Readings outside the window must not move the rebuilt scale.
    outside = pooled_cache.end_times <= np.datetime64("2024-06-03T00:00:00", "us")
    poisoned = replace(before, y=np.where(outside, 900.0, before.y))
    other = replace(pooled_cache, sites={**pooled_cache.sites, "D1": poisoned})
    assert rebuilt_effective_capacity_mw(cache=other, site="D1", cut=CUT, window=window) == rebuilt

    changed = with_effective_capacity(cache=pooled_cache, site="D1", capacity_mw=rebuilt)
    after = changed.sites["D1"]
    assert after.table.effective_capacity_mw == rebuilt
    finite = np.isfinite(before.y)
    power_before = before.y[finite] * before.table.effective_capacity_mw
    power_after = after.y[finite] * after.table.effective_capacity_mw
    assert np.allclose(power_before, power_after)
    # A pinned capacity in megawatts converts to the new points.
    old = _pinned_for(cache=pooled_cache, site="D1", cut=CUT)
    new = _pinned_for(cache=changed, site="D1", cut=CUT)
    ratio = before.table.effective_capacity_mw / rebuilt
    assert new.held_points == pytest.approx(old.held_points * ratio)


def test_the_tied_fit_holds_a_negative_heating_slope_at_zero(pooled_cache: FeatureCache):
    layout, _ = prepare_fit(
        cache=pooled_cache,
        site="D1",
        spec=GamSpec(ridge=1.0),
        cut=CUT,
        window=WINDOW,
        season_years=SEASON_YEARS,
    )
    heating = layout.slices["heating"]
    unit = np.zeros(layout.n_columns)
    unit[heating.start + 1 * N_BANDS + 1] = 1.0
    sites = _sites(labels=("D1", "D2", "D3", "D4"))
    changed = {}
    for site in sites:
        features = pooled_cache.sites[site.site]
        usable = np.flatnonzero(np.isfinite(features.y))
        y = features.y.copy()
        for start in range(0, len(usable), 20000):
            rows = usable[start : start + 20000]
            y[rows] -= 4.0 * (
                design_matrix(cache=pooled_cache, features=features, layout=layout, index=rows)
                @ unit
            )
        changed[site.site] = replace(features, y=y)
    cache = replace(pooled_cache, sites={**pooled_cache.sites, **changed})
    tied = fit_pooled(cache=cache, sites=sites, cut=CUT, tied=True)
    dense, shared = _tied_dense(cache=cache, sites=sites)
    for site in sites:
        theta = tied.fits[site.site].theta[0]
        assert (theta[heating] >= 0.0).all()
        assert theta[heating.start + 1 * N_BANDS + 1] == 0.0
        assert np.abs(theta - dense[site.site]).max() < 1e-6
    assert np.abs(tied.fleet_mean[0] - shared).max() < 1e-6


def test_a_joint_fit_refuses_a_covid_flag_a_final_knot_and_a_missing_or_surplus_strength(
    pooled_cache: FeatureCache,
):
    strengths = [PoolingStrength.uniform(value=1.0)]
    site = _sites(labels=("D1",))[0]
    for spec in (
        GamSpec(covid_flag=True),
        GamSpec(drift="d1", final_knot="shrunk"),
    ):
        with pytest.raises(ValueError, match="drift 'none'"):
            fit_pooled(
                cache=pooled_cache, sites=[replace(site, spec=spec)], cut=CUT, strengths=strengths
            )
    with pytest.raises(ValueError, match="needs at least one strength"):
        fit_pooled(cache=pooled_cache, sites=[site], cut=CUT)
    with pytest.raises(ValueError, match="takes none"):
        fit_pooled(cache=pooled_cache, sites=[site], cut=CUT, strengths=strengths, tied=True)


def test_a_tied_fit_and_a_replica_hold_a_pinned_site_beside_a_fitted_one(
    pooled_cache: FeatureCache,
):
    sites = _sites(labels=("D1", "D2"))
    pinned = _pinned_for(cache=pooled_cache, site="D2", cut=CUT)
    sites[1] = replace(sites[1], spec=GamSpec(capacity="pinned"), pinned=pinned)
    tied = fit_pooled(cache=pooled_cache, sites=sites, cut=CUT, tied=True)
    assert tied.strengths == ()
    assert tied.fits["D2"].pinned is pinned
    assert tied.fits["D1"].capacity_points()
    index = np.flatnonzero(np.isfinite(pooled_cache.sites["D2"].y))[-300:]
    assert np.isfinite(predict_gam(cache=pooled_cache, fit=tied.fits["D2"], index=index)).all()

    fit_cut = replica_cut(cut=CUT)
    replica_sites = [
        sites[0],
        replace(
            sites[1],
            pinned=_pinned_for(cache=pooled_cache, site="D2", cut=fit_cut),
        ),
    ]
    replicas = pooled_replica_residuals(
        cache=pooled_cache,
        sites=replica_sites,
        cut=CUT,
        strength=PoolingStrength.uniform(value=1000.0),
    )
    assert replicas["D2"].residuals.fit_cut == fit_cut
    assert np.isfinite(replicas["D2"].residuals.residuals).all()
    tied_replicas = pooled_replica_residuals(
        cache=pooled_cache,
        sites=replica_sites,
        cut=CUT,
        strength=PoolingStrength.uniform(value=0.0),
        tied=True,
    )
    assert tied_replicas["D2"].n_training_rows == replicas["D2"].n_training_rows
    with pytest.raises(ValueError, match="was built for"):
        pooled_replica_residuals(
            cache=pooled_cache,
            sites=sites,
            cut=CUT,
            strength=PoolingStrength.uniform(value=1000.0),
        )


def test_the_short_history_spread_refuses_a_repeated_site_and_mixed_cuts():
    first = _short_replica(short_site="D2", weeks=4, scale=2.0)
    same_site = _short_replica(short_site="D2", weeks=4, scale=3.0)
    with pytest.raises(ValueError, match="appear once"):
        short_history_spread(replicas=[first, same_site], held_out="D1", weeks_of_history=4)
    other = _short_replica(short_site="D3", weeks=4, scale=2.0)
    shifted = replace(
        other, residuals=replace(other.residuals, fit_cut=datetime(2022, 7, 1, tzinfo=UTC))
    )
    with pytest.raises(ValueError, match="different cuts"):
        short_history_spread(replicas=[first, shifted], held_out="D1", weeks_of_history=4)
