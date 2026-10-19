"""Tests of the partially pooled convex model (`studies.dp_gam_pooled`).

The fixture clones the synthetic substation of `conftest.py` into seven substations whose power is
made from a known fleet mean plus known deviations, so the tests can compare the joint fit with
independent dense computations and with `dp_gam.fit_gam`.
"""

import time
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
)
from studies.dp_gam_pooled import (  # noqa: E402
    FLEET_MEAN_RIDGE,
    POOLED_FAMILIES,
    PooledSite,
    PoolingStrength,
    ShortHistoryReplica,
    centring_matrix,
    fit_pooled,
    pooled_replica_residuals,
    short_history_spread,
    with_window,
)
from studies.dp_pinned_capacity import PinnedCapacity, pinned_capacity  # noqa: E402

CUT: Final[datetime] = datetime(2024, 7, 1, tzinfo=UTC)
SEASON_YEARS: Final[tuple[int, ...]] = tuple(range(2019, 2026))
LABELS: Final[tuple[str, ...]] = ("D1", "D2", "D3", "D4", "D5", "D6", "D7")
WINDOW: Final[TrainingWindow] = TrainingWindow(months=24)
NOISE_POINTS: Final[float] = 0.3


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


def _pinned_for(*, cache: FeatureCache, site: str, cut: datetime) -> PinnedCapacity:
    weeks = []
    week = date(2019, 9, 2)
    while week <= cut.date():
        weeks.append(week)
        week += timedelta(days=7)
    vintages = pl.DataFrame(
        {
            "time_series_id": cache.sites[site].table.time_series_id,
            "variant": "frozen",
            "origin_monday": cut.date(),
            "week_start": weeks,
            "capacity_mw": [8.0 + 0.002 * i for i in range(len(weeks))],
            "unobserved": False,
        }
    )
    return pinned_capacity(vintages=vintages, table=cache.sites[site].table, cut=cut)


def _fit_dense(
    *, cache: FeatureCache, sites: list[PooledSite], strength: PoolingStrength
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    for family in POOLED_FAMILIES:
        size = family_sizes[family]
        columns = layouts[0].slices[family]
        g = sum(np.diag(x.T @ x)[columns].sum() for x in designs) / (size * total_rows)
        weight = strength.of(family=family) * g
        centre = centring_matrix(family=family, size=size)
        for start in starts[:-1]:
            difference = np.zeros((size, total))
            difference[:, int(start) + columns.start : int(start) + columns.stop] = centre
            difference[:, cursor : cursor + size] = -centre
            hessian += weight * difference.T @ difference
        cursor += size
    hessian[mu_at:, mu_at:] += FLEET_MEAN_RIDGE * np.eye(n_mu)
    return hessian, rhs, constrained, starts


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


def test_the_joint_solution_equals_a_dense_generalised_ridge_when_no_bound_binds(
    pooled_cache: FeatureCache,
):
    sites = _sites(labels=("D1", "D2", "D3", "D4"))
    strength = PoolingStrength(heating=3000.0, other=1000.0)
    joint = fit_pooled(cache=pooled_cache, sites=sites, cut=CUT, strengths=[strength])
    hessian, rhs, constrained, _ = _fit_dense(cache=pooled_cache, sites=sites, strength=strength)
    solved = np.concatenate(
        [joint.fits[site.site].theta[0] for site in sites] + [joint.fleet_mean[0]]
    )
    dense = _solve_on_active_set(
        hessian=hessian, rhs=rhs, constrained=constrained, reference_theta=solved
    )
    assert np.abs(dense - solved).max() < 1e-8
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
    *, layout: GamLayout, sizes: dict[str, int], shared_start: dict[str, int], n_shared: int
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
    hessian[:n_shared, :n_shared] += FLEET_MEAN_RIDGE * np.eye(n_shared)
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
        strengths=[PoolingStrength.uniform(value=0.0)],
        tied=True,
    )
    dense, shared = _tied_dense(cache=pooled_cache, sites=sites)
    for site in sites:
        assert np.abs(tied.fits[site.site].theta[0] - dense[site.site]).max() < 1e-6
    assert np.abs(tied.fleet_mean[0] - shared).max() < 1e-6

    mid, large = 1e4, 1e8  # lambda counts rows; a two-year site holds about 35,000
    joint = fit_pooled(
        cache=pooled_cache,
        sites=sites,
        cut=CUT,
        strengths=[PoolingStrength.uniform(value=mid), PoolingStrength.uniform(value=large)],
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

    assert gap(1) < 0.01 * gap(0)


def test_a_held_out_sites_readings_outside_its_window_change_nothing(pooled_cache: FeatureCache):
    window = TrainingWindow(start=datetime(2024, 6, 3, tzinfo=UTC))
    sites = _sites(labels=("D1", "D2", "D3"))
    sites[2] = with_window(site=sites[2], window=window)
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


def test_nothing_after_the_cut_and_no_late_vintage_reaches_the_fit(pooled_cache: FeatureCache):
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


def test_a_pinned_capacity_enters_only_its_own_sites_target(pooled_cache: FeatureCache):
    sites = _sites(labels=("D1", "D2", "D3"))
    strengths = [PoolingStrength.uniform(value=1000.0)]
    pinned = _pinned_for(cache=pooled_cache, site="D3", cut=CUT)
    mixed = [sites[0], sites[1], replace(sites[2], spec=GamSpec(capacity="pinned"), pinned=pinned)]
    joint = fit_pooled(cache=pooled_cache, sites=mixed, cut=CUT, strengths=strengths)
    assert (
        joint.fits["D3"].layout.slices["capacity"].stop
        == joint.fits["D3"].layout.slices["capacity"].start
    )
    assert (
        joint.fits["D1"].layout.slices["capacity"].stop
        > joint.fits["D1"].layout.slices["capacity"].start
    )
    index = np.flatnonzero(np.isfinite(pooled_cache.sites["D3"].y))[-500:]
    forecast = predict_gam(cache=pooled_cache, fit=joint.fits["D3"], index=index)
    assert np.isfinite(forecast).all()


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


def test_the_replica_residuals_cover_the_year_after_the_replica_cut(pooled_cache: FeatureCache):
    replica_sites = _sites(labels=("D1", "D2"))
    residuals = pooled_replica_residuals(
        cache=pooled_cache,
        sites=replica_sites,
        cut=CUT,
        strength=PoolingStrength.uniform(value=1000.0),
    )
    fit_cut = replica_cut(cut=CUT)
    for label in ("D1", "D2"):
        replica = residuals[label]
        ends = pooled_cache.end_times[replica.index]
        assert replica.fit_cut == fit_cut
        assert ends.min() > np.datetime64("2023-07-01T00:00:00", "us")
        assert ends.max() <= np.datetime64("2024-07-01T00:00:00", "us")
        assert abs(np.median(replica.residuals)) < 1.0


def _short_replica(*, short_site: str, weeks: int, scale: float) -> ShortHistoryReplica:
    generator = np.random.default_rng(len(short_site))
    n = 2000
    residuals = ReplicaResiduals(
        residuals=scale * generator.normal(size=n),
        band=generator.integers(0, N_BANDS, size=n).astype(np.int8),
        quarter=generator.integers(0, 4, size=n).astype(np.int8),
        index=np.arange(n),
        fit_cut=datetime(2023, 7, 1, tzinfo=UTC),
        switching=np.zeros(n, dtype=bool),
    )
    return ShortHistoryReplica(short_site=short_site, weeks_of_history=weeks, residuals=residuals)


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
