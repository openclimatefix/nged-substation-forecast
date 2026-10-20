"""Tests of the convex shape library (`studies.dp_shape_library`).

The fixture clones the synthetic substation of `conftest.py` into five substations whose profile is
made from two known shared shapes with sparse non-negative weights, so the tests can check the fit
against independent computations and against a known answer.
"""

from dataclasses import replace
from datetime import UTC, datetime
from typing import Final

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scipy.optimize import nnls  # noqa: E402
from studies.dp_calendar import DAY_TYPES, HALF_HOURS_PER_DAY  # noqa: E402
from studies.dp_features import FeatureCache  # noqa: E402
from studies.dp_gam import (  # noqa: E402
    N_BANDS,
    GamLayout,
    GamSpec,
    TrainingWindow,
    design_matrix,
    predict_gam,
    prepare_fit,
)
from studies.dp_gam_pooled import (  # noqa: E402
    PooledSite,
    PoolingStrength,
    centring_matrix,
    fit_pooled,
)
from studies.dp_shape_library import (  # noqa: E402
    PROFILE_COLUMNS,
    LibraryConfig,
    _build_problem,
    _objective,
    _Problem,
    _renormalise,
    _shape_penalty,
    _solve_step,
    _State,
    _theta,
    cosine_matrix,
    fit_library,
    initial_shapes,
    largest_pair_cosine,
    library_replica_residuals,
    match_shapes,
)

CUT: Final[datetime] = datetime(2024, 7, 1, tzinfo=UTC)
SEASON_YEARS: Final[tuple[int, ...]] = tuple(range(2019, 2026))
LABELS: Final[tuple[str, ...]] = ("D1", "D2", "D3", "D4", "D5")
WINDOW: Final[TrainingWindow] = TrainingWindow(months=24)
NOISE_POINTS: Final[float] = 0.3
SPARSE_WEIGHTS: Final[tuple[tuple[float, float], ...]] = (
    (6.0, 0.0),
    (5.0, 0.5),
    (0.0, 6.0),
    (0.5, 5.0),
    (3.0, 3.0),
)
"""Two shapes, five substations: the first four sit near a vertex, the fifth in the middle."""


def _truth_shapes() -> np.ndarray:
    """Return two known centred unit shapes, shape (2, 192): a morning bump and an evening bump."""
    half_hours = np.arange(HALF_HOURS_PER_DAY)
    centring = centring_matrix(family="profile", size=PROFILE_COLUMNS)
    shapes = np.zeros((2, PROFILE_COLUMNS))
    for day_type in range(len(DAY_TYPES)):
        block = slice(day_type * HALF_HOURS_PER_DAY, (day_type + 1) * HALF_HOURS_PER_DAY)
        scale = 1.0 - 0.2 * day_type
        shapes[0, block] = scale * np.exp(-0.5 * ((half_hours - 16) / 4.0) ** 2)
        shapes[1, block] = scale * np.exp(-0.5 * ((half_hours - 38) / 4.0) ** 2)
    shapes = shapes @ centring
    return shapes / np.linalg.norm(shapes, axis=1, keepdims=True)


def _site_truth(
    *, layout: GamLayout, generator: np.random.Generator, weights: tuple[float, float]
) -> np.ndarray:
    """Return one substation's coefficients: a library profile plus fleet-shared other families."""
    fleet = np.random.default_rng(100)
    theta = np.zeros(layout.n_columns)
    profile = layout.slices["profile"]
    theta[profile] = 40.0 + generator.normal(scale=2.0)
    theta[profile] += weights[0] * _truth_shapes()[0] + weights[1] * _truth_shapes()[1]
    holiday = layout.slices["holiday"]
    theta[holiday] = 0.5 * fleet.normal(size=holiday.stop - holiday.start)
    heating = layout.slices["heating"]
    theta[heating] = 0.6 + 0.4 * fleet.uniform(size=heating.stop - heating.start)
    theta[layout.slices["lighting"]] = (0.5, 0.3, 0.2, 0.1)
    capacity = layout.slices["capacity"]
    for year, increment in {2020: 8.0, 2022: 10.0, 2023: 5.0}.items():
        theta[capacity.start + layout.season_years.index(year)] = increment
    theta[heating] = np.abs(
        theta[heating] + 0.1 * generator.normal(size=heating.stop - heating.start)
    )
    return theta


@pytest.fixture(scope="module")
def library_cache(dp_synthetic_cache: FeatureCache) -> FeatureCache:
    """Five clones of the synthetic substation with a two-shape library in their profiles."""
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
        theta = _site_truth(layout=layout, generator=generator, weights=SPARSE_WEIGHTS[number])
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
    return replace(dp_synthetic_cache, sites=sites, node_ids={label: (label,) for label in LABELS})


def _sites(*, labels: tuple[str, ...] = LABELS) -> list[PooledSite]:
    return [
        PooledSite(site=label, spec=GamSpec(ridge=1.0), window=WINDOW, season_years=SEASON_YEARS)
        for label in labels
    ]


def _fresh_state(*, problem: _Problem, config: LibraryConfig, shapes: np.ndarray) -> _State:
    n_sites = len(problem.labels)
    return _State(
        shapes=shapes.copy(),
        weights=np.ones((n_sites, config.n_weights)),
        band_means=np.zeros((n_sites, N_BANDS)),
        deviations=np.zeros((n_sites, problem.n_basis)),
        others=[np.zeros(len(block.rhs) - PROFILE_COLUMNS) for block in problem.blocks],
        fleet_mean=np.zeros(problem.n_mu),
    )


def test_one_shape_with_weights_fixed_at_one_reproduces_the_partially_pooled_fit(
    library_cache: FeatureCache,
):
    config = LibraryConfig(n_shapes=1, deviation_weight=3000.0)
    sites = _sites()
    problem = _build_problem(cache=library_cache, sites=sites, cut=CUT)
    state = _fresh_state(
        problem=problem, config=config, shapes=initial_shapes(profiles=np.eye(5, 192), n_shapes=1)
    )
    _solve_step(problem=problem, config=config, state=state, shape_index=0)
    reference = fit_pooled(
        cache=library_cache,
        sites=sites,
        cut=CUT,
        strengths=[PoolingStrength.uniform(value=3000.0)],
    )
    for position, label in enumerate(LABELS):
        theta = _theta(problem=problem, state=state, config=config, site=position)
        assert theta == pytest.approx(reference.fits[label].theta[0], rel=1e-6, abs=1e-6)


def test_each_convex_step_does_not_raise_the_objective_it_minimises(library_cache: FeatureCache):
    config = LibraryConfig(n_shapes=2, deviation_weight=1000.0, orthogonality=1.0)
    problem = _build_problem(cache=library_cache, sites=_sites(), cut=CUT)
    state = _fresh_state(problem=problem, config=config, shapes=_truth_shapes() + 0.05)
    _solve_step(problem=problem, config=config, state=state, shape_index=None)
    state.kappa = 1e3
    for _ in range(2):
        before = _objective(problem=problem, config=config, state=state)
        _solve_step(problem=problem, config=config, state=state, shape_index=None)
        assert _objective(problem=problem, config=config, state=state) <= before + 1e-6
        for k in range(2):
            before = _objective(problem=problem, config=config, state=state, only_shape=k)
            _solve_step(problem=problem, config=config, state=state, shape_index=k)
            after = _objective(problem=problem, config=config, state=state, only_shape=k)
            assert after <= before + 1e-6


def test_the_weights_step_equals_non_negative_least_squares_on_the_weights(
    library_cache: FeatureCache,
):
    config = LibraryConfig(n_shapes=2, deviation_weight=3000.0)
    problem = _build_problem(cache=library_cache, sites=_sites(), cut=CUT)
    state = _fresh_state(problem=problem, config=config, shapes=_truth_shapes())
    _solve_step(problem=problem, config=config, state=state, shape_index=None)
    solved = state.weights.copy()

    def objective_at(weights: np.ndarray) -> float:
        probe = replace(state, weights=weights.reshape(solved.shape))
        return _objective(problem=problem, config=config, state=probe)

    size = solved.size
    base = objective_at(np.zeros(size))
    unit = np.eye(size)
    linear = np.array([objective_at(unit[i]) for i in range(size)])
    hessian = np.zeros((size, size))
    for i in range(size):
        for j in range(i + 1, size):
            both = objective_at(unit[i] + unit[j])
            hessian[i, j] = hessian[j, i] = (both - linear[i] - linear[j] + base) / 2.0
        hessian[i, i] = (linear[i] + objective_at(-unit[i]) - 2.0 * base) / 2.0
    gradient = np.array(
        [(objective_at(unit[i]) - objective_at(-unit[i])) / 2.0 for i in range(size)]
    )
    factor = np.linalg.cholesky(hessian)
    target = np.linalg.solve(factor, -gradient / 2.0)
    expected, _ = nnls(factor.T, target)
    assert solved.reshape(-1) == pytest.approx(expected, rel=1e-5, abs=1e-5)
    assert (solved >= 0.0).all()


def test_the_orthogonality_penalty_is_zero_for_orthogonal_shapes_and_rises_for_parallel_ones(
    library_cache: FeatureCache,
):
    config = LibraryConfig(n_shapes=2, deviation_weight=1.0, orthogonality=1.0)
    problem = _build_problem(cache=library_cache, sites=_sites(), cut=CUT)
    basis = problem.basis
    first, second = basis[:, 0], basis[:, 1]
    orthogonal = np.stack([first, second])
    parallel = np.stack([first, first + 0.1 * second])

    def penalty_of(shapes: np.ndarray) -> float:
        quadratic = _shape_penalty(problem=problem, config=config, shapes=shapes, k=0, kappa=1.0)
        coefficients = basis.T @ shapes[0]
        return float(coefficients @ quadratic @ coefficients)

    ridge_only = float(basis.T @ first @ (1e-3 * np.eye(basis.shape[1])) @ (basis.T @ first))
    assert penalty_of(orthogonal) == pytest.approx(ridge_only, rel=1e-9)
    assert penalty_of(parallel) > 0.9 * ridge_only + 0.9


def test_a_fit_ignores_rows_after_the_cut(library_cache: FeatureCache):
    config = LibraryConfig(n_shapes=2, deviation_weight=1000.0, max_sweeps=2)
    sites = _sites()
    first = fit_library(cache=library_cache, sites=sites, cut=CUT, config=config)
    after = library_cache.end_times > np.datetime64(CUT.replace(tzinfo=None), "us")
    corrupted = {
        label: replace(features, y=np.where(after, 1e6, features.y))
        for label, features in library_cache.sites.items()
    }
    second = fit_library(
        cache=replace(library_cache, sites=corrupted), sites=sites, cut=CUT, config=config
    )
    assert second.shapes == pytest.approx(first.shapes, abs=1e-10)
    for label in LABELS:
        assert second.fits[label].theta == pytest.approx(first.fits[label].theta, abs=1e-8)


def test_renormalising_leaves_every_prediction_unchanged(library_cache: FeatureCache):
    config = LibraryConfig(n_shapes=2, deviation_weight=1000.0)
    problem = _build_problem(cache=library_cache, sites=_sites(), cut=CUT)
    state = _fresh_state(problem=problem, config=config, shapes=_truth_shapes())
    _solve_step(problem=problem, config=config, state=state, shape_index=None)
    state.shapes = state.shapes * np.array([[3.0], [0.25]])
    before = [_theta(problem=problem, state=state, config=config, site=i) for i in range(5)]
    _renormalise(state=state, config=config)
    assert np.linalg.norm(state.shapes, axis=1) == pytest.approx(1.0)
    for position, expected in enumerate(before):
        theta = _theta(problem=problem, state=state, config=config, site=position)
        assert theta == pytest.approx(expected, rel=1e-10, abs=1e-10)


def test_a_fit_is_deterministic_for_a_seed_and_differs_between_seeds(library_cache: FeatureCache):
    config = LibraryConfig(n_shapes=2, deviation_weight=1000.0, max_sweeps=2)
    sites = _sites()
    runs = [
        fit_library(
            cache=library_cache,
            sites=sites,
            cut=CUT,
            config=config,
            init_method="dirichlet",
            seed=seed,
        )
        for seed in (3, 3, 4)
    ]
    assert runs[0].shapes == pytest.approx(runs[1].shapes, abs=1e-12)
    assert not np.allclose(runs[0].shapes, runs[2].shapes, atol=1e-6)


def test_the_fit_recovers_a_known_library_and_predicts_with_the_gam_machinery(
    library_cache: FeatureCache,
):
    config = LibraryConfig(n_shapes=2, deviation_weight=3000.0, max_sweeps=30)
    fit = fit_library(
        cache=library_cache, sites=_sites(), cut=CUT, config=config, init_method="dirichlet"
    )
    order, cosines = match_shapes(first=_truth_shapes(), second=fit.shapes)
    assert cosines.min() > 0.9
    weights = np.stack([fit.weights[label] for label in LABELS])[:, order]
    assert (weights >= 0.0).all()
    truth = np.array(SPARSE_WEIGHTS)
    assert (
        np.median(np.abs(weights - truth) / np.maximum(truth.max(axis=1, keepdims=True), 1.0))
        < 0.15
    )
    assert fit.objective_trace[-1] <= fit.objective_trace[0] + 1e-6
    features = library_cache.sites["D1"]
    index = np.flatnonzero(np.isfinite(features.y))[-500:]
    predicted = predict_gam(cache=library_cache, fit=fit.fits["D1"], index=index)
    assert np.sqrt(np.mean((predicted - features.y[index]) ** 2)) < 1.0
    assert fit.n_free_parameters > 0


def test_a_per_day_type_fit_holds_four_weights_per_shape(library_cache: FeatureCache):
    config = LibraryConfig(
        n_shapes=2, deviation_weight=3000.0, weights_per_day_type=True, max_sweeps=2
    )
    fit = fit_library(cache=library_cache, sites=_sites(), cut=CUT, config=config)
    assert fit.weights["D1"].shape == (8,)
    assert (fit.weights["D1"] >= 0.0).all()


def test_the_library_replica_is_fitted_a_year_early_and_scored_on_the_year_after(
    library_cache: FeatureCache,
):
    config = LibraryConfig(n_shapes=2, deviation_weight=1000.0, max_sweeps=1)
    replica = library_replica_residuals(cache=library_cache, sites=_sites(), cut=CUT, config=config)
    for entry in replica.values():
        residuals = entry.residuals
        assert residuals.fit_cut == datetime(2023, 7, 1, tzinfo=UTC)
        assert len(residuals.residuals) > 0
        assert np.isfinite(residuals.residuals).all()


def test_shape_matching_uses_the_signed_cosine():
    rng = np.random.default_rng(0)
    first = rng.normal(size=(3, 192))
    second = np.stack([-first[1], first[2], first[0]])
    order, cosines = match_shapes(first=first, second=second)
    assert list(order) == [2, 0, 1] or cosines.min() > -1.0
    assert cosines[0] == pytest.approx(1.0)
    flipped = match_shapes(first=first[:1], second=-first[:1])[1]
    assert flipped[0] == pytest.approx(-1.0)
    assert largest_pair_cosine(shapes=np.stack([first[0], -first[0]])) == pytest.approx(1.0)
    assert largest_pair_cosine(shapes=first[:1]) == 0.0
    assert cosine_matrix(first=first, second=first).diagonal() == pytest.approx(1.0)


def test_starting_shapes_are_centred_and_unit_norm_for_both_methods():
    profiles = np.random.default_rng(1).normal(size=(5, 192)) + 40.0
    centring = centring_matrix(family="profile", size=192)
    for method in ("svd", "dirichlet"):
        shapes = initial_shapes(profiles=profiles, n_shapes=6, method=method, seed=2)
        assert shapes.shape == (6, 192)
        assert np.linalg.norm(shapes, axis=1) == pytest.approx(1.0)
        assert shapes @ centring == pytest.approx(shapes, abs=1e-10)


def test_a_config_refuses_out_of_range_settings():
    with pytest.raises(ValueError, match="at least one shape"):
        LibraryConfig(n_shapes=0, deviation_weight=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        LibraryConfig(n_shapes=2, deviation_weight=-1.0)
    with pytest.raises(ValueError, match="at least one sweep"):
        LibraryConfig(n_shapes=2, deviation_weight=1.0, max_sweeps=0)


def test_a_fit_refuses_no_substations_and_a_repeated_label(library_cache: FeatureCache):
    config = LibraryConfig(n_shapes=2, deviation_weight=1.0)
    with pytest.raises(ValueError, match="each label once"):
        fit_library(cache=library_cache, sites=[], cut=CUT, config=config)
    with pytest.raises(ValueError, match="each label once"):
        fit_library(cache=library_cache, sites=_sites(labels=("D1", "D1")), cut=CUT, config=config)
    with pytest.raises(ValueError, match="starting shapes"):
        fit_library(
            cache=library_cache,
            sites=_sites(),
            cut=CUT,
            config=config,
            start_shapes=np.zeros((3, 192)),
        )
