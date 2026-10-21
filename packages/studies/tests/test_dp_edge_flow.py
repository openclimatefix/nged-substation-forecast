"""Tests of the edge-flow module (`studies.dp_edge_flow`).

The path solver is checked against a quadratic-programme reference solved by SLSQP, and the ADMM
solver against the same reference. The block-coordinate fit is checked against a joint program on
a three-node graph. The causal window is checked by changing every residual after the cutoff.
"""

from typing import Any

import numpy as np
import pytest
from scipy.optimize import minimize

torch = pytest.importorskip("torch", reason="studies.dp_gam needs torch: uv sync --group gpu")

from studies.dp_assimilation import FilterHyperparameters, run_kalman  # noqa: E402
from studies.dp_edge_flow import (  # noqa: E402
    BandFilterRun,
    FlowFit,
    LeadHoursSpread,
    PathProblem,
    PathSpec,
    SolverMethodType,
    _compress,
    _signs,
    block_mean_residuals,
    build_paths,
    carry_at,
    causal_window_fit,
    decision_edges,
    edge_name,
    expand_blocks,
    expand_held_flat,
    fit_flows,
    fit_survival,
    flow_mw_days,
    huber_loss,
    joint_objective,
    moved_mw_ratio,
    mw_weighted_recall,
    node_flow,
    noise_before,
    path_objective,
    release_delay,
    robust_residual_scale,
    run_band_filter,
    run_known_input_filter,
    run_lengths,
    runs_to_trace,
    seed_month_bootstrap,
    solve_path,
    undirected_edges,
)
from studies.dp_gam import N_BANDS  # noqa: E402

QP_TOLERANCE = 1e-6
"""How close an exact solver's objective must be to the reference's."""


def _reference_objective(*, problem: PathProblem) -> float:
    """Solve a path problem as a quadratic programme with SLSQP and return its objective.

    The Huber loss is written as its minimum over a slack `e` of `(x - e)**2 / 2 + delta * |e|`,
    and each absolute value as a variable bounded below by plus and minus its argument.
    """
    kept, ends = _compress(problem=problem)
    n, k_ends, signs = len(kept), len(ends), _signs(ends=ends)

    def parts(x: np.ndarray) -> tuple[np.ndarray, ...]:
        flow = x[:n]
        slack = x[n : n + k_ends * n].reshape(k_ends, n)
        slack_abs = x[n + k_ends * n : n + 2 * k_ends * n].reshape(k_ends, n)
        offset = n + 2 * k_ends * n
        return flow, slack, slack_abs, x[offset : offset + n - 1], x[offset + n - 1 :]

    def objective(x: np.ndarray) -> float:
        flow, slack, slack_abs, step_abs, level_abs = parts(x)
        total = problem.step_penalty * step_abs.sum() + problem.level_penalty * level_abs.sum()
        for k, (end, sign) in enumerate(zip(ends, signs, strict=True)):
            fit = end.value - sign * end.gain * flow - slack[k]
            total += np.sum(end.weight * (0.5 * fit**2 + end.threshold * slack_abs[k]))
        return float(total)

    def constraints(x: np.ndarray) -> np.ndarray:
        flow, slack, slack_abs, step_abs, level_abs = parts(x)
        step = np.diff(flow)
        return np.concatenate(
            [
                (slack_abs - slack).ravel(),
                (slack_abs + slack).ravel(),
                step_abs - step,
                step_abs + step,
                level_abs - flow,
                level_abs + flow,
            ]
        )

    n_variables = n + 2 * k_ends * n + (n - 1) + n
    result = minimize(
        objective,
        np.zeros(n_variables),
        constraints=[{"type": "ineq", "fun": constraints}],
        bounds=[
            (0.0, None) if problem.non_negative and i < n else (None, None)
            for i in range(n_variables)
        ],
        method="SLSQP",
        options={"ftol": 1e-15, "maxiter": 3000},
    )
    return float(result.fun)


def _random_problem(*, rng: np.random.Generator, non_negative: bool = False) -> PathProblem:
    """Draw a small path problem with a step, missing readings, and random penalties."""
    n = int(rng.integers(1, 8))
    two_ends = rng.random() < 0.7
    a = rng.normal(0.0, 1.0, n) + np.repeat(rng.normal(0.0, 3.0, 2), [n // 2, n - n // 2])
    b = -a + rng.normal(0.0, 1.0, n) if two_ends else None
    a[rng.random(n) < 0.2] = np.nan
    if b is not None:
        b[rng.random(n) < 0.2] = np.nan
    return PathProblem(
        residual_a=a,
        noise_a=float(rng.uniform(0.5, 2.0)),
        residual_b=b,
        noise_b=float(rng.uniform(0.5, 2.0)),
        step_penalty=float(rng.uniform(0.0, 3.0)),
        level_penalty=float(rng.choice([0.0, 0.3, 1.0])),
        non_negative=non_negative,
    )


def _has_a_reading(*, problem: PathProblem) -> bool:
    ends = [problem.residual_a] + ([] if problem.residual_b is None else [problem.residual_b])
    return any(np.any(np.isfinite(end)) for end in ends)


def test_the_dynamic_programme_matches_a_quadratic_programme_on_random_problems():
    rng = np.random.default_rng(0)
    checked = 0
    while checked < 200:
        problem = _random_problem(rng=rng)
        if not _has_a_reading(problem=problem):
            continue
        solution = solve_path(problem=problem)
        assert abs(solution.objective - _reference_objective(problem=problem)) < QP_TOLERANCE
        assert solution.objective == pytest.approx(
            path_objective(problem=problem, flow=solution.flow)
        )
        checked += 1


def test_admm_matches_a_quadratic_programme_with_and_without_a_non_negative_flow():
    rng = np.random.default_rng(1)
    for non_negative in (False, True):
        checked = 0
        while checked < 40:
            problem = _random_problem(rng=rng, non_negative=non_negative)
            if not _has_a_reading(problem=problem):
                continue
            solution = solve_path(problem=problem, method="admm")
            assert solution.converged
            assert abs(solution.objective - _reference_objective(problem=problem)) < QP_TOLERANCE
            if non_negative:
                assert np.all(solution.flow >= -1e-9)
            checked += 1


def test_a_path_with_no_reading_has_a_zero_flow():
    problem = PathProblem(residual_a=np.full(5, np.nan), noise_a=1.0, step_penalty=1.0)

    solution = solve_path(problem=problem)

    assert np.all(solution.flow == 0.0)
    assert solution.objective == 0.0


def test_a_flow_is_held_flat_across_a_gap_and_zero_before_the_first_reading():
    a = np.array([np.nan, 3.0, 3.0, np.nan, np.nan, 3.0, 3.0, np.nan])
    b = np.where(np.isfinite(a), -a, np.nan)

    solution = solve_path(
        problem=PathProblem(
            residual_a=a,
            noise_a=1.0,
            residual_b=b,
            noise_b=1.0,
            step_penalty=5.0,
            level_penalty=0.0,
        )
    )

    assert solution.flow[0] == 0.0
    assert solution.flow[1] > 2.0
    assert np.allclose(solution.flow[1:], solution.flow[1])


def test_a_large_step_penalty_gives_one_flat_flow_and_a_zero_penalty_follows_the_data():
    rng = np.random.default_rng(2)
    a = np.concatenate([np.full(6, 2.0), np.full(6, 0.0)]) + rng.normal(0, 0.1, 12)
    b = -a + rng.normal(0, 0.1, 12)

    stiff = solve_path(
        problem=PathProblem(residual_a=a, noise_a=1.0, residual_b=b, noise_b=1.0, step_penalty=1e4)
    )
    loose = solve_path(
        problem=PathProblem(residual_a=a, noise_a=1.0, residual_b=b, noise_b=1.0, step_penalty=0.0)
    )

    assert np.ptp(stiff.flow) < 1e-9
    assert np.max(np.abs(loose.flow - (a - b) / 2.0)) < 0.2


def test_a_single_spike_is_outweighed_by_the_huber_loss():
    a = np.zeros(21)
    a[10] = 100.0

    solution = solve_path(
        problem=PathProblem(residual_a=a, noise_a=1.0, step_penalty=3.0, level_penalty=0.0)
    )

    assert np.max(np.abs(solution.flow)) < 2.0


def test_huber_loss_is_quadratic_inside_the_threshold_and_linear_outside():
    x = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])

    loss = huber_loss(x=x, threshold=1.0)

    assert loss == pytest.approx([2.5, 0.125, 0.0, 0.125, 2.5])


def test_expand_held_flat_holds_each_value_until_the_next_kept_row():
    grid = expand_held_flat(kept=np.array([2, 5]), values=np.array([1.0, -1.0]), length=8)

    assert grid.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0]


def test_the_graph_builder_makes_an_edge_path_each_and_an_outside_path_per_node():
    paths = build_paths(
        edges=[("B", "A"), ("A", "B"), ("B", "C")],
        nodes=["A", "B", "C"],
        step_penalty_by_edge={"A-B": 2.0, "B-C": 4.0},
        level_penalty_by_edge={"A-B": 0.5, "B-C": 1.0},
        outside_factor=2.0,
    )

    assert [path.name for path in paths] == ["A-B", "B-C", "outside:A", "outside:B", "outside:C"]
    outside = {path.name: path for path in paths if path.node_b is None}
    assert outside["outside:A"].step_penalty == 4.0
    assert outside["outside:C"].level_penalty == 2.0


def test_the_graph_builder_rejects_a_self_edge_an_unknown_node_and_a_small_outside_factor():
    common = {
        "nodes": ["A", "B"],
        "step_penalty_by_edge": {"A-B": 1.0},
        "level_penalty_by_edge": {"A-B": 1.0},
    }
    with pytest.raises(ValueError, match="joins a node to itself"):
        build_paths(edges=[("A", "A")], outside_factor=1.0, **common)
    with pytest.raises(ValueError, match="unknown node"):
        build_paths(edges=[("A", "Z")], outside_factor=1.0, **common)
    with pytest.raises(ValueError, match="at least 1"):
        build_paths(edges=[("A", "B")], outside_factor=0.5, **common)


def test_edges_are_undirected_deduplicated_and_filtered_by_meter_and_exclusion():
    pairs = [("B", "A"), ("A", "B"), ("C", "A"), ("D", "A"), ("E", "E")]

    assert undirected_edges(pairs=pairs) == (("A", "B"), ("A", "C"), ("A", "D"))
    assert decision_edges(pairs=pairs, metered={"A", "B", "C"}, excluded={"C"}) == (("A", "B"),)
    assert edge_name(node_a="B", node_b="A") == "A-B"


def _three_node_residuals(*, rng: np.random.Generator) -> dict[str, np.ndarray]:
    n = 6
    transfer = np.array([0, 0, 2.0, 2.0, 0, 0])
    return {
        "A": transfer + rng.normal(0, 0.3, n),
        "B": -transfer + rng.normal(0, 0.3, n),
        "C": rng.normal(0, 0.3, n),
    }


def _joint_reference(
    *, paths: list[PathSpec], residuals: dict[str, np.ndarray], noise: dict[str, float]
) -> float:
    """Minimise the three-node joint objective with SLSQP, using the same Huber and L1 forms."""
    n = len(residuals["A"])
    n_paths = len(paths)

    def flows_of(x: np.ndarray) -> dict[str, np.ndarray]:
        return {path.name: x[k * n : (k + 1) * n] for k, path in enumerate(paths)}

    def total(x: np.ndarray) -> float:
        return joint_objective(paths=paths, flows=flows_of(x), residuals=residuals, noise=noise)

    best = np.inf
    for seed in range(3):
        start = np.random.default_rng(seed).normal(0, 0.5, n_paths * n)
        result = minimize(
            total, start, method="Powell", options={"xtol": 1e-10, "ftol": 1e-13, "maxiter": 200000}
        )
        best = min(best, float(result.fun))
    return best


def test_block_coordinate_descent_reaches_the_joint_optimum_of_a_three_node_graph():
    rng = np.random.default_rng(3)
    residuals = _three_node_residuals(rng=rng)
    noise = {"A": 0.5, "B": 0.5, "C": 0.5}
    paths = [
        PathSpec(name="A-B", node_a="A", node_b="B", step_penalty=1.0, level_penalty=0.2),
        PathSpec(name="B-C", node_a="B", node_b="C", step_penalty=1.0, level_penalty=0.2),
        PathSpec(name="outside:A", node_a="A", step_penalty=2.0, level_penalty=0.4),
    ]

    fit = fit_flows(paths=paths, residuals=residuals, noise=noise)

    assert fit.converged
    assert fit.objective <= _joint_reference(paths=paths, residuals=residuals, noise=noise) + 1e-4


def test_a_conserved_transfer_with_no_noise_is_recovered_on_its_edge_and_nowhere_else():
    n = 40
    transfer = np.zeros(n)
    transfer[10:25] = 3.0
    residuals = {"A": transfer.copy(), "B": -transfer, "C": np.zeros(n)}
    noise = {"A": 1.0, "B": 1.0, "C": 1.0}
    paths = build_paths(
        edges=[("A", "B"), ("B", "C"), ("A", "C")],
        nodes=["A", "B", "C"],
        step_penalty_by_edge={"A-B": 1e-6, "B-C": 1e-6, "A-C": 1e-6},
        level_penalty_by_edge={"A-B": 0.0, "B-C": 0.0, "A-C": 0.0},
        outside_factor=2.0,
    )

    fit = fit_flows(paths=paths, residuals=residuals, noise=noise)

    assert moved_mw_ratio(fitted=fit.flows["A-B"][10:25], true=transfer[10:25]) == pytest.approx(
        1.0, abs=1e-5
    )
    for name, flow in fit.flows.items():
        if name != "A-B":
            assert np.max(np.abs(flow)) < 1e-5, name


def test_fit_flows_rejects_a_path_on_a_node_with_no_residual():
    with pytest.raises(ValueError, match="has no residual"):
        fit_flows(
            paths=[PathSpec(name="A-Z", node_a="A", node_b="Z")],
            residuals={"A": np.zeros(4)},
            noise={"A": 1.0},
        )


def test_a_one_sided_step_is_left_to_the_outside_path_and_not_the_edge():
    n = 60
    step = np.zeros(n)
    step[20:] = 2.0
    residuals = {"A": step, "B": np.zeros(n)}
    paths = build_paths(
        edges=[("A", "B")],
        nodes=["A", "B"],
        step_penalty_by_edge={"A-B": 3.0},
        level_penalty_by_edge={"A-B": 0.1},
        outside_factor=1.0,
    )

    fit = fit_flows(paths=paths, residuals=residuals, noise={"A": 1.0, "B": 1.0})

    assert np.mean(fit.flows["outside:A"][25:]) > 1.5
    assert np.max(np.abs(fit.flows["A-B"])) < 0.5


def test_the_causal_window_fit_ignores_everything_after_the_cutoff():
    rng = np.random.default_rng(4)
    n = 200
    transfer = np.zeros(n)
    transfer[60:120] = 2.0
    residuals = {"A": transfer + rng.normal(0, 0.3, n), "B": -transfer + rng.normal(0, 0.3, n)}
    changed = {node: residual.copy() for node, residual in residuals.items()}
    for residual in changed.values():
        residual[101:] += rng.normal(5.0, 5.0, n - 101)
    paths = [PathSpec(name="A-B", node_a="A", node_b="B", step_penalty=5.0)]
    noise = {"A": 0.5, "B": 0.5}

    before, first_before = causal_window_fit(
        paths=paths, residuals=residuals, noise=noise, cutoff=100, window_length=80
    )
    after, first_after = causal_window_fit(
        paths=paths, residuals=changed, noise=noise, cutoff=100, window_length=80
    )

    assert first_before == first_after == 21
    assert np.array_equal(before.flows["A-B"], after.flows["A-B"])
    assert len(before.flows["A-B"]) == 80


def test_the_causal_window_fit_rejects_a_negative_cutoff():
    with pytest.raises(ValueError, match="cutoff"):
        causal_window_fit(paths=[], residuals={}, noise={}, cutoff=-1, window_length=10)


def test_node_flow_adds_at_one_end_and_subtracts_at_the_other():
    paths = [PathSpec(name="A-B", node_a="A", node_b="B"), PathSpec(name="outside:A", node_a="A")]
    flows = {"A-B": np.array([1.0, 2.0]), "outside:A": np.array([0.5, 0.5])}

    assert node_flow(paths=paths, flows=flows, node="A").tolist() == [1.5, 2.5]
    assert node_flow(paths=paths, flows=flows, node="B").tolist() == [-1.0, -2.0]


def test_block_means_average_the_readings_and_weight_a_block_by_its_share_of_readings():
    residual = np.array(
        [1.0, 3.0, np.nan, np.nan, 5.0, 7.0, np.nan, np.nan, np.nan, np.nan, np.nan]
    )

    means, reliability = block_mean_residuals(residuals={"A": residual}, block_size=4)

    assert means["A"][0] == pytest.approx(2.0)
    assert means["A"][1] == pytest.approx(6.0)
    assert np.isnan(means["A"][2])
    assert reliability["A"].tolist() == [0.5, 0.5, 0.0]
    assert expand_blocks(
        block_values=np.array([1.0, 2.0, 3.0]), block_size=4, length=11
    ).tolist() == ([1.0] * 4 + [2.0] * 4 + [3.0] * 3)


def test_the_robust_scale_ignores_outliers_and_nan_and_rejects_an_empty_stream():
    rng = np.random.default_rng(5)
    stream = rng.normal(0.0, 2.0, 5000)
    stream[:10] = 1000.0
    stream[10:20] = np.nan

    assert robust_residual_scale(residual=stream) == pytest.approx(2.0, rel=0.1)
    with pytest.raises(ValueError, match="no finite residual"):
        robust_residual_scale(residual=np.full(3, np.nan))


def test_run_lengths_mark_a_run_that_reaches_the_end_as_censored():
    flow = np.array([0, 2.0, 2.0, 0, 0, -1.5, 0, 3.0, 3.0, 3.0])

    lengths, censored = run_lengths(flow=flow, threshold=0.1)

    assert lengths.tolist() == [2, 1, 3]
    assert censored.tolist() == [False, False, True]


def test_the_kaplan_meier_curve_matches_a_hand_calculation_and_handles_censoring():
    lengths = np.array([2, 3, 3, 5, 8])
    censored = np.array([False, False, True, False, True])

    curve = fit_survival(lengths=lengths, censored=censored)

    assert curve.times.tolist() == [2, 3, 5]
    assert curve.survival == pytest.approx([4 / 5, 4 / 5 * 3 / 4, 4 / 5 * 3 / 4 * 1 / 2])
    assert curve.at(age=1) == 1.0
    assert curve.at(age=4) == pytest.approx(0.6)
    assert curve.held_share(age=2, ahead=3) == pytest.approx(0.3 / 0.8)
    assert curve.held_share(age=1, ahead=0) == pytest.approx(1.0)


def test_a_flow_older_than_every_observed_run_has_no_held_share():
    curve = fit_survival(lengths=np.array([2, 4]), censored=np.array([False, False]))

    assert float(curve.held_share(age=10, ahead=2)) == 0.0


def test_the_survival_fit_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        fit_survival(lengths=np.array([1, 2]), censored=np.array([False]))


def _filter_inputs(*, rng: np.random.Generator, n: int = 480) -> tuple[np.ndarray, np.ndarray]:
    band = (np.arange(n) // 12) % N_BANDS
    residual = np.cumsum(rng.normal(0, 0.05, n)) + rng.normal(0, 0.5, n)
    residual[rng.random(n) < 0.1] = np.nan
    return residual, band


def test_the_filter_run_from_a_cold_start_matches_the_assimilation_module():
    rng = np.random.default_rng(6)
    residual, band = _filter_inputs(rng=rng)
    hyper = FilterHyperparameters(log_q=-3.0, logit_phi=4.0, log_sigma_z=-1.0, log_sigma_eps=-0.5)

    reference = run_kalman(residual=residual, band=band, hyper=hyper)
    runs = run_known_input_filter(
        residual=residual,
        flow=np.zeros(len(residual)),
        band=band,
        first=0,
        last=len(residual) - 1,
        hyper=hyper,
        carry=None,
    )

    for band_index, run in enumerate(runs):
        assert np.array_equal(run.positions, reference.positions[band_index])
        assert np.allclose(run.level, reference.level[band_index])
        assert np.allclose(run.anomaly, reference.anomaly[band_index])
    trace = runs_to_trace(runs=runs, hyper=hyper)
    assert trace.phi == hyper.phi


def test_two_windows_joined_by_the_carried_state_match_one_run_over_the_whole_stream():
    rng = np.random.default_rng(7)
    residual, band = _filter_inputs(rng=rng)
    hyper = FilterHyperparameters(log_q=-3.0, logit_phi=4.0, log_sigma_z=-1.0, log_sigma_eps=-0.5)
    n, split = len(residual), 200
    whole = run_known_input_filter(
        residual=residual, flow=np.zeros(n), band=band, first=0, last=n - 1, hyper=hyper, carry=None
    )

    head = run_known_input_filter(
        residual=residual,
        flow=np.zeros(split),
        band=band,
        first=0,
        last=split - 1,
        hyper=hyper,
        carry=None,
    )
    carry = carry_at(runs=head, position=split - 1, fallback=None)
    tail = run_known_input_filter(
        residual=residual,
        flow=np.zeros(n - split),
        band=band,
        first=split,
        last=n - 1,
        hyper=hyper,
        carry=carry,
    )

    for band_index in range(N_BANDS):
        joined_level = np.concatenate([head[band_index].level, tail[band_index].level])
        assert np.allclose(joined_level, whole[band_index].level)


def test_the_known_input_filter_does_not_see_a_flow_it_is_told_about():
    n = 240
    band = np.zeros(n, dtype=int)
    transfer = np.zeros(n)
    transfer[100:] = 3.0
    hyper = FilterHyperparameters(log_q=-3.0, logit_phi=4.0, log_sigma_z=-1.0, log_sigma_eps=-0.5)

    blind = run_known_input_filter(
        residual=transfer, flow=np.zeros(n), band=band, first=0, last=n - 1, hyper=hyper, carry=None
    )
    informed = run_known_input_filter(
        residual=transfer, flow=transfer, band=band, first=0, last=n - 1, hyper=hyper, carry=None
    )

    assert blind[0].level[-1] + blind[0].anomaly[-1] > 2.5
    assert np.max(np.abs(informed[0].level)) < 1e-9


def test_the_known_input_filter_rejects_a_flow_of_the_wrong_length_and_carry_falls_back():
    hyper = FilterHyperparameters(log_q=-3.0, logit_phi=4.0, log_sigma_z=-1.0, log_sigma_eps=-0.5)
    with pytest.raises(ValueError, match="one per row"):
        run_known_input_filter(
            residual=np.zeros(10),
            flow=np.zeros(3),
            band=np.zeros(10, dtype=int),
            first=0,
            last=9,
            hyper=hyper,
            carry=None,
        )
    empty = BandFilterRun(*(np.zeros(0) for _ in range(6)))
    assert carry_at(runs=[empty] * N_BANDS, position=5, fallback=None) == (None,) * N_BANDS
    state = run_band_filter(
        positions=np.array([1, 2]), values=np.array([1.0, 1.0]), hyper=hyper, initial=None
    ).state_at(position=2)
    assert (
        carry_at(runs=[empty] * N_BANDS, position=5, fallback=(state,) * N_BANDS)
        == (state,) * N_BANDS
    )


def test_the_recall_ratio_and_release_measures_on_a_known_transfer():
    true = np.array([2.0, 2.0, 2.0, 2.0])
    fitted = np.array([2.0, 1.0, 3.0, -1.0])

    assert mw_weighted_recall(fitted=fitted, true=true) == pytest.approx((2 + 1 + 2 + 0) / 8)
    assert moved_mw_ratio(fitted=fitted, true=true) == pytest.approx(0.625)
    assert np.isnan(mw_weighted_recall(fitted=fitted, true=np.zeros(4)))
    assert np.isnan(moved_mw_ratio(fitted=fitted, true=np.zeros(4)))
    after = np.array([2.0, 2.0, 1.0, 0.3, 0.2])
    assert release_delay(fitted=after, moved_mw=2.0, end=2) == 1
    assert release_delay(fitted=np.array([2.0, 2.0, 2.0]), moved_mw=2.0, end=1) is None


def test_the_lead_hours_spread_is_wider_where_the_residuals_are_wider():
    rng = np.random.default_rng(8)
    n = 4000
    lead_hours = rng.uniform(0.5, 48.0, n)
    residuals = rng.normal(0.0, 0.2 + lead_hours / 24.0)
    band = rng.integers(0, N_BANDS, n)
    quarter = rng.integers(0, 4, n)
    spread = LeadHoursSpread.fit(
        residuals=residuals, band=band, quarter=quarter, lead_hours=lead_hours
    )
    probe_lead = np.array([1.0, 40.0])

    quantiles = spread.quantiles(
        mean=np.zeros(2),
        band=np.zeros(2, dtype=int),
        quarter=np.zeros(2, dtype=int),
        lead_hours=probe_lead,
    )

    assert np.all(np.diff(quantiles, axis=1) >= 0.0)
    assert quantiles[1, -1] - quantiles[1, 0] > 2.0 * (quantiles[0, -1] - quantiles[0, 0])
    assert LeadHoursSpread.bin_of(lead_hours=np.array([0.5, 3.0, 47.9, 90.0])).tolist() == [
        0,
        1,
        5,
        5,
    ]


def test_the_seed_month_bootstrap_covers_the_true_mean_and_widens_with_noise():
    rng = np.random.default_rng(9)
    seeds = np.repeat(np.arange(5), 240)
    months = np.tile(np.repeat(np.arange(12), 20), 5)
    quiet = rng.normal(1.0, 0.1, len(seeds))
    noisy = rng.normal(1.0, 3.0, len(seeds))

    estimate, lower, upper = seed_month_bootstrap(
        values=quiet, seeds=seeds, months=months, level=95.0
    )
    _, noisy_lower, noisy_upper = seed_month_bootstrap(
        values=noisy, seeds=seeds, months=months, level=95.0
    )

    assert lower < estimate < upper
    assert lower < 1.0 < upper
    assert noisy_upper - noisy_lower > 5 * (upper - lower)


def test_the_seed_month_bootstrap_is_zero_width_for_a_constant_and_rejects_bad_input():
    seeds = np.repeat([0, 1], 24)
    months = np.tile(np.repeat([1, 2, 3], 8), 2)

    estimate, lower, upper = seed_month_bootstrap(
        values=np.full(48, 2.5), seeds=seeds, months=months, level=95.0
    )

    assert (estimate, lower, upper) == pytest.approx((2.5, 2.5, 2.5))
    with pytest.raises(ValueError, match="one entry per row"):
        seed_month_bootstrap(values=np.zeros(3), seeds=seeds, months=months, level=95.0)
    with pytest.raises(ValueError, match="percent"):
        seed_month_bootstrap(values=np.zeros(48), seeds=seeds, months=months, level=0.95)


def test_a_month_effect_shared_by_every_seed_widens_the_seed_month_interval():
    rng = np.random.default_rng(10)
    seeds = np.repeat(np.arange(20), 120)
    months = np.tile(np.repeat(np.arange(12), 10), 20)
    month_effect = rng.normal(0.0, 1.0, 12)
    values = month_effect[months] + rng.normal(0.0, 0.1, len(seeds))

    _, lower, upper = seed_month_bootstrap(values=values, seeds=seeds, months=months, level=95.0)

    # The 20 seeds share the 12 month effects, so the mean has the spread of 12 draws, about 0.29.
    assert upper - lower > 0.5


def test_a_warm_start_of_the_wrong_length_and_a_reliability_weight_are_handled():
    rng = np.random.default_rng(11)
    residuals = {"A": rng.normal(0, 1, 200), "B": rng.normal(0, 1, 200)}
    paths = [PathSpec(name="A-B", node_a="A", node_b="B", step_penalty=5.0)]
    noise = {"A": 1.0, "B": 1.0}

    with pytest.raises(ValueError, match="one value per row"):
        causal_window_fit(
            paths=paths,
            residuals=residuals,
            noise=noise,
            cutoff=150,
            window_length=100,
            start={"A-B": np.zeros(99)},
        )
    fit, first = causal_window_fit(
        paths=paths,
        residuals=residuals,
        noise=noise,
        cutoff=150,
        window_length=100,
        start={"A-B": np.zeros(100)},
        reliability={"A": np.ones(200), "B": np.full(200, 0.5)},
    )

    assert first == 51
    assert len(fit.flows["A-B"]) == 100


def test_noise_before_uses_only_rows_up_to_the_last_known_row():
    stream = np.concatenate([np.random.default_rng(12).normal(0, 1, 500), np.full(500, 1000.0)])

    scale = noise_before(residuals={"A": stream}, last_row=499)

    assert scale["A"] == pytest.approx(1.0, rel=0.15)


def test_a_path_with_zero_gains_has_a_zero_flow_and_matches_admm_elsewhere():
    n = 40
    rng = np.random.default_rng(13)
    gain = np.where(np.arange(n) % 8 < 4, 0.0, 1.0)
    a = rng.normal(0, 1, n) + 2.0 * (np.arange(n) > 20)
    b = -a + rng.normal(0, 1, n)
    flat = PathProblem(
        residual_a=a, noise_a=1.0, residual_b=b, noise_b=1.0, gain_a=np.zeros(n), gain_b=np.zeros(n)
    )
    shaped = PathProblem(
        residual_a=a,
        noise_a=1.0,
        residual_b=b,
        noise_b=1.0,
        gain_a=gain,
        gain_b=gain,
        step_penalty=4.0,
        level_penalty=0.2,
    )

    exact, iterative = solve_path(problem=shaped), solve_path(problem=shaped, method="admm")

    assert np.all(np.isfinite(solve_path(problem=flat).flow))
    assert np.all(solve_path(problem=flat).flow == 0.0)
    assert exact.objective == pytest.approx(iterative.objective, abs=1e-5)


def test_the_graph_builder_rejects_a_node_with_no_edge_and_a_dash_in_a_name():
    common = {"step_penalty_by_edge": {"A-B": 1.0}, "level_penalty_by_edge": {"A-B": 1.0}}
    with pytest.raises(ValueError, match="no edge"):
        build_paths(edges=[("A", "B")], nodes=["A", "B", "C"], outside_factor=1.0, **common)
    with pytest.raises(ValueError, match="'-'"):
        build_paths(edges=[("A", "B-2")], nodes=["A", "B-2"], outside_factor=1.0, **common)


def test_the_sweep_limit_stops_a_coupled_fit_early_and_reports_it():
    rng = np.random.default_rng(14)
    n = 120
    transfer = np.where((np.arange(n) > 30) & (np.arange(n) < 90), 3.0, 0.0)
    residuals = {
        "A": transfer + rng.normal(0, 0.5, n),
        "B": -0.5 * transfer + rng.normal(0, 0.5, n),
        "C": -0.5 * transfer + rng.normal(0, 0.5, n),
    }
    paths = [
        PathSpec(name="A-B", node_a="A", node_b="B", step_penalty=2.0),
        PathSpec(name="A-C", node_a="A", node_b="C", step_penalty=2.0),
        PathSpec(name="B-C", node_a="B", node_b="C", step_penalty=2.0),
    ]
    noise = {"A": 0.5, "B": 0.5, "C": 0.5}

    stopped = fit_flows(paths=paths, residuals=residuals, noise=noise, max_sweeps=1)
    settled = fit_flows(paths=paths, residuals=residuals, noise=noise, sweep_tolerance=1e-3)

    assert stopped.sweeps == 1
    assert not stopped.converged
    assert settled.sweeps > 1
    assert settled.objective <= stopped.objective + 1e-9


def test_a_slow_opposite_drift_is_an_edge_flow_unless_the_level_penalty_is_large():
    n = 600
    drift = np.linspace(0.0, 2.0, n)
    residuals = {"A": drift, "B": -drift * 0.8}
    noise = {"A": 0.5, "B": 0.5}
    free = [PathSpec(name="A-B", node_a="A", node_b="B", step_penalty=5.0, level_penalty=0.0)]
    costly = [PathSpec(name="A-B", node_a="A", node_b="B", step_penalty=5.0, level_penalty=200.0)]

    assert (
        flow_mw_days(flow=fit_flows(paths=free, residuals=residuals, noise=noise).flows["A-B"])
        > 1.0
    )
    assert (
        flow_mw_days(flow=fit_flows(paths=costly, residuals=residuals, noise=noise).flows["A-B"])
        < 1e-6
    )


def test_a_noisy_transfer_on_one_edge_leaves_the_other_edges_nearly_empty():
    rng = np.random.default_rng(15)
    n = 400
    transfer = np.where((np.arange(n) > 100) & (np.arange(n) < 300), 4.0, 0.0)
    residuals = {
        "A": transfer + rng.normal(0, 0.3, n),
        "B": -transfer + rng.normal(0, 0.3, n),
        "C": rng.normal(0, 0.3, n),
    }
    paths = build_paths(
        edges=[("A", "B"), ("A", "C"), ("B", "C")],
        nodes=["A", "B", "C"],
        step_penalty_by_edge={"A-B": 20.0, "A-C": 20.0, "B-C": 20.0},
        level_penalty_by_edge={"A-B": 1.0, "A-C": 1.0, "B-C": 1.0},
        outside_factor=2.0,
    )

    fit = fit_flows(paths=paths, residuals=residuals, noise={"A": 0.3, "B": 0.3, "C": 0.3})

    assert fit.flows["A-B"][150:250].mean() > 3.0
    assert flow_mw_days(flow=fit.flows["A-C"]) < 0.5
    assert flow_mw_days(flow=fit.flows["B-C"]) < 0.5


def test_flow_mw_days_sums_magnitudes_over_forty_eight():
    assert flow_mw_days(flow=np.array([2.0, -2.0, 0.0, 4.0])) == pytest.approx(8.0 / 48.0)


def test_a_large_huber_threshold_turns_the_data_term_into_squared_error():
    spike = np.zeros(21)
    spike[10] = 100.0
    robust = PathProblem(residual_a=spike, noise_a=1.0, step_penalty=3.0)
    squared = PathProblem(residual_a=spike, noise_a=1.0, step_penalty=3.0, huber_sigmas=1e6)

    assert np.max(np.abs(solve_path(problem=robust).flow)) < 2.0
    assert np.max(np.abs(solve_path(problem=squared).flow)) > 10.0


def _one_row_flow(*, method: SolverMethodType = "dynamic_programme", **settings: Any) -> float:
    """Return the flow of a one-row, one-ended path with a squared-error data term."""
    problem = PathProblem(
        residual_a=np.array([5.0]), huber_sigmas=1e6, level_penalty=0.5, **settings
    )
    return float(solve_path(problem=problem, method=method).flow[0])


@pytest.mark.parametrize("method", ["dynamic_programme", "admm"])
def test_a_row_is_weighted_by_one_over_the_noise_variance(method: SolverMethodType):
    # Minimising 0.5 * w * (5 - f)**2 + 0.5 * |f| gives f = 5 - 0.5 / w, and w = 1 / noise**2.
    assert _one_row_flow(method=method, noise_a=2.0) == pytest.approx(3.0, abs=1e-3)
    assert _one_row_flow(method=method, noise_a=0.5) == pytest.approx(4.875, abs=1e-3)


@pytest.mark.parametrize("method", ["dynamic_programme", "admm"])
def test_a_reliability_weight_scales_the_row_weight(method: SolverMethodType):
    # w = 0.5 / 4 = 0.125, so f = 5 - 0.5 / 0.125 = 1.
    flow = _one_row_flow(method=method, noise_a=2.0, reliability_a=np.array([0.5]))

    assert flow == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("method", ["dynamic_programme", "admm"])
def test_a_gain_other_than_zero_or_one_scales_the_flow_into_the_residual(
    method: SolverMethodType,
):
    # Minimising 0.5 * (5 - 2 f)**2 + 0.5 * |f| gives f = (5 - 0.5 / 2) / 2 = 2.375.
    flow = _one_row_flow(method=method, noise_a=1.0, gain_a=np.array([2.0]))

    assert flow == pytest.approx(2.375, abs=1e-3)


def _constant_offset_fit(*, order: list[int], level: float) -> FlowFit:
    """Fit one edge and two outside paths to a constant offset that either could carry."""
    residuals = {"A": np.full(30, 3.0), "B": np.full(30, -3.0)}
    paths = [
        PathSpec(name="A-B", node_a="A", node_b="B", step_penalty=50.0, level_penalty=level),
        PathSpec(name="outside:A", node_a="A", step_penalty=50.0, level_penalty=2.0 * level),
        PathSpec(name="outside:B", node_a="B", step_penalty=50.0, level_penalty=2.0 * level),
    ]
    return fit_flows(
        paths=[paths[i] for i in order],
        residuals=residuals,
        noise={"A": 1.0, "B": 1.0},
        max_sweeps=50,
    )


def test_a_constant_offset_is_ambiguous_without_a_level_penalty_and_unique_with_one():
    free = [_constant_offset_fit(order=order, level=0.0) for order in ([0, 1, 2], [1, 2, 0])]
    penalised = [_constant_offset_fit(order=order, level=0.3) for order in ([0, 1, 2], [1, 2, 0])]

    assert free[0].objective == pytest.approx(free[1].objective, abs=1e-6)
    assert abs(free[0].flows["A-B"][0] - free[1].flows["A-B"][0]) > 1.0
    assert penalised[0].flows["A-B"] == pytest.approx(penalised[1].flows["A-B"], abs=1e-3)
    assert penalised[0].flows["A-B"][0] > 2.0
