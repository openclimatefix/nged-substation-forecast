"""Tests of the graph-penalised convex model (`studies.dp_gam_graph`).

The fixture clones the synthetic substation of `conftest.py` into four substations with known
coefficients, so the tests can compare the joint fit with `dp_gam.fit_gam`, with
`dp_gam_pooled.fit_pooled`, and with independent dense computations.
"""

import time
from dataclasses import replace
from datetime import UTC, datetime
from typing import Final

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from studies.dp_calendar import HALF_HOURS_PER_DAY  # noqa: E402
from studies.dp_features import FeatureCache  # noqa: E402
from studies.dp_gam import (  # noqa: E402
    GamLayout,
    GamSpec,
    TrainingWindow,
    design_matrix,
    fit_gam,
    penalty_matrix,
    prepare_fit,
)
from studies.dp_gam_graph import (  # noqa: E402
    GraphPenalty,
    GraphSolution,
    InterconnectionGraph,
    MonthAxis,
    NodeDesign,
    build_node_design,
    degree_preserving_permutation,
    edge_similarity_ratio,
    family_scales,
    gam_fits_from_solution,
    heating_band_slopes,
    holiday_level,
    joint_month_counts,
    kappa_for_neighbour_correlation,
    latest_capacity,
    mean_neighbour_correlation,
    month_axis_of,
    month_cross_products,
    profile_scale,
    ring_with_chords,
    sample_smooth_fields,
    solve_graph_fit,
)
from studies.dp_gam_pooled import (  # noqa: E402
    PooledSite,
    PoolingStrength,
    centring_matrix,
    fit_pooled,
)

CUT: Final[datetime] = datetime(2024, 7, 1, tzinfo=UTC)
SEASON_YEARS: Final[tuple[int, ...]] = tuple(range(2019, 2026))
LABELS: Final[tuple[str, ...]] = ("D1", "D2", "D3", "D4")
WINDOW: Final[TrainingWindow] = TrainingWindow(months=24)
NOISE_POINTS: Final[float] = 0.3
TOLERANCE: Final[float] = 1e-6


def _truth(*, layout_columns: int, layout: GamLayout, number: int) -> np.ndarray:
    """Return one substation's coefficients: a shared shape plus its own deviations."""
    generator = np.random.default_rng(number)
    fleet = np.random.default_rng(100)
    theta = np.zeros(layout_columns)
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
    for name, scale in (("heating", 0.15), ("holiday", 0.2)):
        block = layout.slices[name]
        theta[block] += scale * generator.normal(size=block.stop - block.start)
    theta[heating] = np.abs(theta[heating])
    theta[profile] += generator.normal(scale=3.0)
    return theta


@pytest.fixture(scope="module")
def graph_cache(dp_synthetic_cache: FeatureCache) -> FeatureCache:
    """Four clones of the synthetic substation, each with its own known coefficients."""
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
        theta = _truth(layout_columns=layout.n_columns, layout=layout, number=number)
        y = np.full(len(dp_synthetic_cache.end_times), np.nan)
        for start in range(0, len(usable), 20000):
            rows = usable[start : start + 20000]
            y[rows] = (
                design_matrix(cache=dp_synthetic_cache, features=base, layout=layout, index=rows)
                @ theta
            )
        y = y + NOISE_POINTS * np.random.default_rng(number).normal(size=len(y))
        table = replace(base.table, site=label, time_series_id=number + 1)
        sites[label] = replace(base, table=table, y=y)
    return replace(dp_synthetic_cache, sites=sites, node_ids={label: (label,) for label in LABELS})


def _sites(*, labels: tuple[str, ...] = LABELS) -> list[PooledSite]:
    return [
        PooledSite(site=label, spec=GamSpec(ridge=1.0), window=WINDOW, season_years=SEASON_YEARS)
        for label in labels
    ]


def _designs(*, cache: FeatureCache, labels: tuple[str, ...] = LABELS) -> list[NodeDesign]:
    return [build_node_design(cache=cache, site=site, cut=CUT) for site in _sites(labels=labels)]


def _path(*, labels: tuple[str, ...]) -> InterconnectionGraph:
    pairs = [(labels[i], labels[i + 1]) for i in range(len(labels) - 1)]
    return InterconnectionGraph.from_pairs(nodes=labels, pairs=pairs)


def _solve(
    *,
    designs: list[NodeDesign],
    graph: InterconnectionGraph,
    penalty: GraphPenalty,
    axis: MonthAxis | None = None,
    counts: np.ndarray | None = None,
) -> GraphSolution:
    return solve_graph_fit(
        designs=designs,
        graph=graph,
        penalty=penalty,
        scales=family_scales(designs=designs),
        axis=axis,
        counts=counts,
    )


def test_gamma_zero_and_lambda_zero_equal_fit_gam_to_1e_minus_8(graph_cache: FeatureCache):
    designs = _designs(cache=graph_cache, labels=LABELS[:3])
    solution = _solve(designs=designs, graph=_path(labels=LABELS[:3]), penalty=GraphPenalty.none())
    for site in _sites(labels=LABELS[:3]):
        alone = fit_gam(
            cache=graph_cache,
            site=site.site,
            specs=[site.spec],
            cut=CUT,
            window=site.window,
            season_years=site.season_years,
        )
        assert np.abs(solution.thetas[site.site] - alone.theta).max() < 1e-8


def test_a_fleet_mean_penalty_alone_equals_fit_pooled(graph_cache: FeatureCache):
    designs = _designs(cache=graph_cache)
    solution = _solve(
        designs=designs,
        graph=_path(labels=LABELS),
        penalty=GraphPenalty.of(fleet_rows=3000.0),
    )
    pooled = fit_pooled(
        cache=graph_cache,
        sites=_sites(),
        cut=CUT,
        strengths=[PoolingStrength.uniform(value=3000.0)],
    )
    for label in LABELS:
        assert np.abs(solution.thetas[label] - pooled.fits[label].theta).max() < TOLERANCE
    assert solution.fleet_mean is not None
    assert np.abs(solution.fleet_mean - pooled.fleet_mean).max() < TOLERANCE


def test_a_graph_with_no_edges_equals_the_per_site_fit_and_the_fleet_mean_fit(
    graph_cache: FeatureCache,
):
    designs = _designs(cache=graph_cache)
    no_edges = InterconnectionGraph(nodes=LABELS, edges=())
    per_site = _solve(designs=designs, graph=no_edges, penalty=GraphPenalty.none())
    with_graph_weight = _solve(
        designs=designs, graph=no_edges, penalty=GraphPenalty.of(graph_rows=10_000.0)
    )
    fleet_only = _solve(designs=designs, graph=no_edges, penalty=GraphPenalty.of(fleet_rows=3000.0))
    fleet_and_graph = _solve(
        designs=designs,
        graph=no_edges,
        penalty=GraphPenalty.of(fleet_rows=3000.0, graph_rows=10_000.0),
    )
    for label in LABELS:
        assert np.abs(with_graph_weight.thetas[label] - per_site.thetas[label]).max() < 1e-8
        assert np.abs(fleet_and_graph.thetas[label] - fleet_only.thetas[label]).max() < 1e-8


def test_a_holiday_only_fleet_mean_pools_only_the_holiday_coefficients(graph_cache: FeatureCache):
    designs = _designs(cache=graph_cache)
    graph = InterconnectionGraph(nodes=LABELS, edges=())
    per_site = _solve(designs=designs, graph=graph, penalty=GraphPenalty.none())
    holiday = _solve(
        designs=designs, graph=graph, penalty=GraphPenalty.of(holiday_fleet_rows=1_000_000.0)
    )
    layout = designs[0].layout
    holiday_columns = layout.slices["holiday"]

    def across_sites(solution: GraphSolution) -> float:
        stacked = np.stack([solution.thetas[label][0, holiday_columns] for label in LABELS])
        return float(stacked.std(axis=0).mean())

    spread_before = across_sites(per_site)
    spread_after = across_sites(holiday)
    assert spread_after < 0.1 * spread_before
    heating = layout.slices["heating"]
    for label in LABELS:
        moved = np.abs(holiday.thetas[label][0, heating] - per_site.thetas[label][0, heating])
        assert moved.max() < 0.15


def _dense_graph_reference(
    *,
    cache: FeatureCache,
    labels: tuple[str, ...],
    edges: list[tuple[str, str]],
    gamma: float,
    fixed: np.ndarray,
) -> np.ndarray:
    """Return the joint solution of an independent dense computation, holding `fixed` at zero."""
    designs, targets, penalties, layouts = [], [], [], []
    for site in _sites(labels=labels):
        layout, index = prepare_fit(
            cache=cache,
            site=site.site,
            spec=site.spec,
            cut=CUT,
            window=site.window,
            season_years=site.season_years,
        )
        designs.append(
            design_matrix(cache=cache, features=cache.sites[site.site], layout=layout, index=index)
        )
        targets.append(cache.sites[site.site].y[index])
        penalties.append(penalty_matrix(layout=layout, n_rows=len(index))[0])
        layouts.append(layout)
    sizes = [layout.n_columns for layout in layouts]
    starts = np.concatenate([[0], np.cumsum(sizes)])
    total = int(starts[-1])
    total_rows = sum(len(x) for x in designs)
    hessian = np.zeros((total, total))
    rhs = np.zeros(total)
    for x, y, penalty, start in zip(designs, targets, penalties, starts, strict=False):
        stop = int(start) + x.shape[1]
        hessian[start:stop, start:stop] += x.T @ x + penalty
        rhs[start:stop] += x.T @ y
    for family in ("heating", "profile", "holiday"):
        columns = layouts[0].slices[family]
        size = columns.stop - columns.start
        scale = sum(np.diag(x.T @ x)[columns].sum() for x in designs) / (size * total_rows)
        centre = centring_matrix(family=family, size=size)
        for first, second in edges:
            i, j = labels.index(first), labels.index(second)
            difference = np.zeros((size, total))
            difference[:, int(starts[i]) + columns.start : int(starts[i]) + columns.stop] = centre
            difference[:, int(starts[j]) + columns.start : int(starts[j]) + columns.stop] = -centre
            hessian += gamma * scale * difference.T @ difference
    free = ~fixed
    solution = np.zeros(total)
    solution[free] = np.linalg.solve(hessian[np.ix_(free, free)], rhs[free])
    return solution


def test_the_solver_matches_a_dense_program_on_a_three_node_path_and_ignores_node_order(
    graph_cache: FeatureCache,
):
    labels = LABELS[:3]
    gamma = 2000.0
    designs = _designs(cache=graph_cache, labels=labels)
    penalty = GraphPenalty(fleet={}, graph={"heating": gamma, "profile": gamma, "holiday": gamma})
    graph = _path(labels=labels)
    solution = _solve(designs=designs, graph=graph, penalty=penalty)
    stacked = np.concatenate([solution.thetas[label][0] for label in labels])
    constrained = np.concatenate([design.layout.constrained for design in designs])
    reference = _dense_graph_reference(
        cache=graph_cache,
        labels=labels,
        edges=list(graph.edges),
        gamma=gamma,
        fixed=constrained & (stacked <= 0.0),
    )
    assert np.abs(stacked - reference).max() < TOLERANCE
    reversed_graph = InterconnectionGraph(nodes=tuple(reversed(labels)), edges=graph.edges)
    other = _solve(designs=designs, graph=reversed_graph, penalty=penalty)
    for label in labels:
        assert np.abs(other.thetas[label] - solution.thetas[label]).max() < 1e-8


def test_a_larger_gamma_pulls_two_connected_nodes_together(graph_cache: FeatureCache):
    labels = LABELS[:2]
    designs = _designs(cache=graph_cache, labels=labels)
    graph = _path(labels=labels)
    layout = designs[0].layout
    gaps = []
    for gamma in (0.0, 1e2, 1e4, 1e6):
        penalty = GraphPenalty(fleet={}, graph={"heating": gamma, "holiday": gamma})
        solution = _solve(designs=designs, graph=graph, penalty=penalty)
        gaps.append(
            np.abs(
                solution.thetas[labels[0]][0, layout.slices["heating"]]
                - solution.thetas[labels[1]][0, layout.slices["heating"]]
            ).max()
        )
    assert gaps == sorted(gaps, reverse=True)
    assert gaps[-1] < 0.05 * gaps[0]


def test_an_edge_does_not_penalise_a_family_one_end_does_not_identify(
    graph_cache: FeatureCache,
):
    designs = _designs(cache=graph_cache, labels=LABELS[:2])
    graph = _path(labels=LABELS[:2])
    layout = designs[0].layout
    penalty = GraphPenalty(fleet={}, graph={"heating": 1e6})
    per_site = _solve(designs=designs, graph=graph, penalty=GraphPenalty.none())
    tied = _solve(designs=designs, graph=graph, penalty=penalty)
    heating = layout.slices["heating"]
    tied_gap = np.abs(tied.thetas[LABELS[0]][0, heating] - tied.thetas[LABELS[1]][0, heating]).max()
    free_gap = np.abs(
        per_site.thetas[LABELS[0]][0, heating] - per_site.thetas[LABELS[1]][0, heating]
    ).max()
    assert tied_gap < 0.1 * free_gap
    unidentified = designs[1].identified.copy()
    unidentified[heating] = False
    designs[1] = replace(designs[1], identified=unidentified)
    released = _solve(designs=designs, graph=graph, penalty=penalty)
    for label in LABELS[:2]:
        assert np.abs(released.thetas[label] - per_site.thetas[label]).max() < 1e-8


def test_the_capacity_edge_ties_the_latest_capacity_of_two_fitted_nodes(
    graph_cache: FeatureCache,
):
    designs = _designs(cache=graph_cache, labels=LABELS[:2])
    graph = _path(labels=LABELS[:2])
    layout = designs[0].layout
    before = _solve(designs=designs, graph=graph, penalty=GraphPenalty.none())
    after = _solve(
        designs=designs, graph=graph, penalty=GraphPenalty(fleet={}, graph={"capacity": 1e7})
    )
    gap_before = abs(
        latest_capacity(layout=layout, theta=before.thetas[LABELS[0]])[0]
        - latest_capacity(layout=layout, theta=before.thetas[LABELS[1]])[0]
    )
    gap_after = abs(
        latest_capacity(layout=layout, theta=after.thetas[LABELS[0]])[0]
        - latest_capacity(layout=layout, theta=after.thetas[LABELS[1]])[0]
    )
    assert gap_after < 0.05 * max(gap_before, 1e-9) + 1e-3


def test_nothing_after_the_cut_at_a_neighbour_reaches_a_sites_fit(graph_cache: FeatureCache):
    graph = _path(labels=LABELS[:2])
    penalty = GraphPenalty.of(fleet_rows=3000.0, graph_rows=3000.0)
    baseline = _solve(
        designs=_designs(cache=graph_cache, labels=LABELS[:2]), graph=graph, penalty=penalty
    )
    neighbour = graph_cache.sites[LABELS[1]]
    after_cut = graph_cache.end_times > np.datetime64(CUT.replace(tzinfo=None), "us")
    y = neighbour.y.copy()
    y[after_cut] = np.nan_to_num(y[after_cut]) + 1000.0
    changed_cache = replace(
        graph_cache, sites={**graph_cache.sites, LABELS[1]: replace(neighbour, y=y)}
    )
    changed = _solve(
        designs=_designs(cache=changed_cache, labels=LABELS[:2]), graph=graph, penalty=penalty
    )
    for label in LABELS[:2]:
        assert np.abs(changed.thetas[label] - baseline.thetas[label]).max() < 1e-10


def test_the_full_sample_row_of_a_bootstrap_equals_the_unweighted_fit(graph_cache: FeatureCache):
    designs = _designs(cache=graph_cache, labels=LABELS[:2])
    graph = _path(labels=LABELS[:2])
    penalty = GraphPenalty.of(graph_rows=3000.0)
    axis = month_axis_of(designs=designs)
    counts = joint_month_counts(axis=axis, n_resamples=3, rng=np.random.default_rng(0))
    plain = _solve(designs=designs, graph=graph, penalty=penalty)
    batched = _solve(designs=designs, graph=graph, penalty=penalty, axis=axis, counts=counts)
    assert batched.thetas[LABELS[0]].shape[0] == 4
    for label in LABELS[:2]:
        assert np.abs(batched.thetas[label][0] - plain.thetas[label][0]).max() < 1e-8
    assert np.abs(batched.thetas[LABELS[0]][1] - plain.thetas[LABELS[0]][0]).max() > 1e-6


def test_joint_month_counts_keep_every_quarters_size_and_start_with_the_full_sample():
    months = np.arange("2022-07", "2024-07", dtype="datetime64[M]")
    axis = MonthAxis.of(months=np.concatenate([months, months]))
    counts = joint_month_counts(axis=axis, n_resamples=20, rng=np.random.default_rng(1))
    assert counts.shape == (21, len(months))
    assert (counts[0] == 1.0).all()
    for quarter in range(4):
        members = axis.quarter == quarter
        assert (counts[1:, members].sum(axis=1) == members.sum()).all()


def test_a_synthetic_target_gives_the_matching_right_hand_side(graph_cache: FeatureCache):
    design = _designs(cache=graph_cache, labels=LABELS[:1])[0]
    own_target = graph_cache.sites[LABELS[0]].y[design.index]
    own = month_cross_products(design=design, target=own_target)
    assert torch.allclose(own, design.month_rhs, atol=1e-6)


def test_the_readouts_sum_the_right_columns(graph_cache: FeatureCache):
    layout = _designs(cache=graph_cache, labels=LABELS[:1])[0].layout
    theta = np.zeros((2, layout.n_columns))
    theta[:, layout.slices["heating"]] = 1.0
    theta[0, layout.slices["capacity"]] = (1.0,) * (
        layout.slices["capacity"].stop - layout.slices["capacity"].start
    )
    theta[:, layout.slices["holiday"]] = (0.5, 1.5)[0]
    profile = layout.slices["profile"]
    theta[1, profile] = 7.0
    slopes = heating_band_slopes(layout=layout, theta=theta)
    assert slopes.shape == (2, 4)
    assert (slopes == 4.0).all()
    assert latest_capacity(layout=layout, theta=theta)[0] == pytest.approx(
        layout.slices["capacity"].stop - layout.slices["capacity"].start
    )
    assert holiday_level(layout=layout, theta=theta)[0] == pytest.approx(0.5)
    assert profile_scale(layout=layout, theta=theta)[1] == pytest.approx(0.0, abs=1e-9)
    shaped = theta.copy()
    shaped[0, profile.start] = 10.0
    assert profile_scale(layout=layout, theta=shaped)[0] > 0.0


def test_gam_fits_from_a_solution_carry_the_batch_and_the_layout(graph_cache: FeatureCache):
    designs = _designs(cache=graph_cache, labels=LABELS[:2])
    solution = _solve(designs=designs, graph=_path(labels=LABELS[:2]), penalty=GraphPenalty.none())
    fits = gam_fits_from_solution(cache=graph_cache, designs=designs, solution=solution, cut=CUT)
    assert set(fits) == set(LABELS[:2])
    assert fits[LABELS[0]].theta.shape == (1, designs[0].layout.n_columns)
    assert fits[LABELS[0]].cut == CUT


def test_a_fifteen_member_joint_fit_of_four_nodes_takes_under_a_minute(
    graph_cache: FeatureCache,
):
    designs = _designs(cache=graph_cache)
    axis = month_axis_of(designs=designs)
    counts = joint_month_counts(axis=axis, n_resamples=14, rng=np.random.default_rng(2))
    started = time.perf_counter()
    _solve(
        designs=designs,
        graph=_path(labels=LABELS),
        penalty=GraphPenalty.of(fleet_rows=3000.0, graph_rows=3000.0),
        axis=axis,
        counts=counts,
    )
    assert time.perf_counter() - started < 60.0


def test_the_solver_refuses_mismatched_nodes_and_counts_without_an_axis(graph_cache: FeatureCache):
    designs = _designs(cache=graph_cache, labels=LABELS[:2])
    with pytest.raises(ValueError, match="same"):
        _solve(designs=designs, graph=_path(labels=LABELS[:3]), penalty=GraphPenalty.none())
    with pytest.raises(ValueError, match="month axis"):
        _solve(
            designs=designs,
            graph=_path(labels=LABELS[:2]),
            penalty=GraphPenalty.none(),
            counts=np.ones((2, 3)),
        )


def test_a_graph_refuses_repeats_self_loops_unknown_nodes_and_unsorted_pairs():
    with pytest.raises(ValueError, match="once"):
        InterconnectionGraph(nodes=("A", "A"), edges=())
    with pytest.raises(ValueError, match="itself"):
        InterconnectionGraph(nodes=("A", "B"), edges=(("A", "A"),))
    with pytest.raises(ValueError, match="not in the graph"):
        InterconnectionGraph(nodes=("A", "B"), edges=(("A", "C"),))
    with pytest.raises(ValueError, match="smaller"):
        InterconnectionGraph(nodes=("A", "B"), edges=(("B", "A"),))
    with pytest.raises(ValueError, match="edge may appear once"):
        InterconnectionGraph(nodes=("A", "B"), edges=(("A", "B"), ("A", "B")))


def test_from_pairs_drops_duplicates_self_pairs_and_unknown_nodes():
    graph = InterconnectionGraph.from_pairs(
        nodes=("A", "B", "C"), pairs=[("B", "A"), ("A", "B"), ("C", "C"), ("A", "Z")]
    )
    assert graph.edges == (("A", "B"),)
    assert graph.degrees() == {"A": 1, "B": 1, "C": 0}


def test_a_ring_with_chords_has_the_requested_edges_and_a_fixed_seed_repeats():
    nodes = tuple(f"N{i}" for i in range(15))
    graph = ring_with_chords(nodes=nodes, n_chords=6, rng=np.random.default_rng(3))
    assert len(graph.edges) == 21
    assert graph == ring_with_chords(nodes=nodes, n_chords=6, rng=np.random.default_rng(3))
    with pytest.raises(ValueError, match="at least three"):
        ring_with_chords(nodes=("A", "B"), n_chords=0, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="free"):
        ring_with_chords(nodes=nodes, n_chords=1000, rng=np.random.default_rng(0))


def test_a_degree_preserving_permutation_keeps_every_degree_and_changes_the_edges():
    nodes = tuple(f"N{i}" for i in range(15))
    graph = ring_with_chords(nodes=nodes, n_chords=6, rng=np.random.default_rng(3))
    permuted = degree_preserving_permutation(graph=graph, rng=np.random.default_rng(4))
    assert permuted.degrees() == graph.degrees()
    assert len(permuted.edges) == len(graph.edges)
    assert permuted.edges != graph.edges


def test_the_coupling_for_a_target_neighbour_correlation_is_found_and_monotone():
    nodes = tuple(f"N{i}" for i in range(15))
    graph = ring_with_chords(nodes=nodes, n_chords=6, rng=np.random.default_rng(3))
    kappa = kappa_for_neighbour_correlation(graph=graph, target=0.8)
    assert mean_neighbour_correlation(graph=graph, kappa=kappa) == pytest.approx(0.8, abs=1e-6)
    assert mean_neighbour_correlation(graph=graph, kappa=0.0) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="must lie in"):
        kappa_for_neighbour_correlation(graph=graph, target=1.5)


def test_smooth_fields_are_alike_along_edges_and_the_similarity_ratio_shows_it():
    nodes = tuple(f"N{i}" for i in range(15))
    graph = ring_with_chords(nodes=nodes, n_chords=6, rng=np.random.default_rng(3))
    kappa = kappa_for_neighbour_correlation(graph=graph, target=0.8)
    fields = sample_smooth_fields(
        graph=graph, kappa=kappa, n_fields=400, rng=np.random.default_rng(5)
    )
    assert fields.shape == (15, 400)
    assert fields.std(axis=1).mean() == pytest.approx(1.0, abs=0.15)
    smooth = np.mean(
        [
            edge_similarity_ratio(
                graph=graph,
                values=dict(zip(nodes, fields[:, k].tolist(), strict=True)),
                rng=np.random.default_rng(k),
            )
            for k in range(60)
        ]
    )
    independent = np.mean(
        [
            edge_similarity_ratio(
                graph=graph,
                values=dict(
                    zip(nodes, np.random.default_rng(k).normal(size=15).tolist(), strict=True)
                ),
                rng=np.random.default_rng(k),
            )
            for k in range(60)
        ]
    )
    assert smooth < 0.8
    assert independent == pytest.approx(1.0, abs=0.25)
    with pytest.raises(ValueError, match="no value"):
        edge_similarity_ratio(graph=graph, values={}, rng=np.random.default_rng(0))
