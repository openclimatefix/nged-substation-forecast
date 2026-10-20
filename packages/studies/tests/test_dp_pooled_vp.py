"""Tests of the variable-projection fit of the pooled differentiable-physics model."""

from collections.abc import Callable
from dataclasses import replace
from typing import Final

import numpy as np
import pytest

torch = pytest.importorskip("torch")
optimize = pytest.importorskip("scipy.optimize")

from studies.dp_features import FeatureCache  # noqa: E402
from studies.dp_onboarding import onboard_site, select_rows, site_mask  # noqa: E402
from studies.dp_pooled import (  # noqa: E402
    DpConfig,
    PooledBatch,
    PooledDemandModel,
    PopulationScales,
    build_pooled_batch,
)
from studies.dp_pooled_vp import (  # noqa: E402
    VpLayout,
    VpProblem,
    VpSettings,
    _check_mva_demand,
    _evaluate,
    _last_layer,
    assemble_system,
    fit_pooled_adam_on_vp_objective,
    fit_pooled_vp,
    freeze_operator_for_vp,
    learned_increment_mask,
    linear_design,
    load_theta,
    make_vp_layout,
    make_vp_problem,
    phi_parameters,
    site_bounds,
    solve_theta,
    trainable_only,
    vp_objective,
    vp_site_fitter,
)

from studies import dp_pooled_vp  # noqa: E402

CUT: Final[np.datetime64] = np.datetime64("2024-07-01T00:00:00", "us")
WINDOW_START: Final[np.datetime64] = np.datetime64("2022-07-01T00:00:00", "us")
SETTINGS: Final[VpSettings] = VpSettings(noise_sd=2.0)
FORECAST_YEAR: Final[int] = 2024

pytestmark = pytest.mark.filterwarnings("ignore:Converting a tensor with requires_grad")


def _cache(base: FeatureCache, *, third: str | None) -> FeatureCache:
    """Return the synthetic cache with S1 and S2 (MW) and optionally S3 (`mva` or `mw`)."""
    first = base.sites["S1"]
    generator = np.random.default_rng(7)
    second = replace(
        first,
        table=replace(first.table, site="S2", time_series_id=2),
        y=first.y * 1.2 + generator.normal(scale=1.0, size=first.y.shape),
    )
    sites = {"S1": first, "S2": second}
    if third == "mva":
        sites["S3"] = replace(
            first,
            table=replace(first.table, site="S3", units="MVA", time_series_id=3),
            dark=np.nan_to_num(first.ghi_w_m2) < 1.0,
            y=first.y * 0.7 + 30.0,
        )
    elif third == "mw":
        sites["S3"] = replace(
            first,
            table=replace(first.table, site="S3", time_series_id=3),
            y=first.y * 0.8 + generator.normal(scale=1.0, size=first.y.shape),
        )
    return replace(base, sites=sites, node_ids={name: (name,) for name in sites})


def _batch(cache: FeatureCache, *, sites: tuple[str, ...]) -> PooledBatch:
    return build_pooled_batch(
        cache=cache, sites=sites, cut_end=CUT, after=WINDOW_START, dtype=torch.float64
    )


def _model(
    *,
    batch: PooledBatch,
    hidden: tuple[int, ...] = (),
    shapes: int = 2,
    seed: int = 0,
    map_prior: bool = False,
) -> PooledDemandModel:
    torch.manual_seed(seed)
    scales = PopulationScales(deviation=1.0) if map_prior else None
    model = PooledDemandModel(
        config=DpConfig(shared_shapes=shapes, interaction_hidden=hidden, population_scales=scales),
        n_sites=len(batch.site_labels),
        n_season_years=batch.n_season_years,
        initial_level=torch.full((len(batch.site_labels),), 50.0),
    ).to(dtype=torch.float64)
    model.mark_trained_years(batch=batch)
    model.start_from_batch(batch=batch)
    freeze_operator_for_vp(model=model)
    return model


def _problem(model: PooledDemandModel, batch: PooledBatch) -> VpProblem:
    return make_vp_problem(model=model, batch=batch, settings=SETTINGS)


def _random_theta(*, layout: VpLayout, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    theta = 0.3 * torch.randn(layout.n_columns, dtype=torch.float64, generator=generator)
    theta = torch.where(layout.constrained, theta.abs(), theta)
    # The population means of the slope and the increments are softplus outputs, so non-negative.
    for name, column in layout.means.items():
        if name != "level":
            theta[column] = theta[column].abs()
    for site in layout.sites:
        theta[layout.local_start[site] + layout.local[site]["level"].start] += 50.0
    return theta


def _perturb_phi(*, model: PooledDemandModel, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in phi_parameters(model=model):
            noise = torch.randn(parameter.shape, dtype=parameter.dtype, generator=generator)
            parameter.add_(0.4 * noise)
        model.threshold.add_(15.0)


def _check_design_against_forward(
    *, model: PooledDemandModel, batch: PooledBatch, layout: VpLayout, theta: torch.Tensor
) -> None:
    """Load `theta`, and assert the dense design reproduces `forward` on every substation."""
    load_theta(model=model, layout=layout, theta=theta)
    model.train()
    with torch.no_grad():
        forward = model(batch)
    q = layout.n_shared
    for site, (start, stop) in layout.rows.items():
        design = linear_design(model=model, batch=batch, layout=layout, site=site)
        columns = slice(layout.local_start[site], layout.local_start[site] + design.local.shape[1])
        prediction = design.shared @ theta[:q] + design.local @ theta[columns] + design.offset
        assert torch.allclose(prediction, forward[start:stop], atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("hidden", [(), (8, 8)])
def test_the_linear_design_reproduces_the_pooled_model_forward(
    dp_synthetic_cache: FeatureCache, hidden: tuple[int, ...]
):
    cache = _cache(dp_synthetic_cache, third="mva")
    batch = _batch(cache, sites=("S1", "S2", "S3"))
    model = _model(batch=batch, hidden=hidden)
    _perturb_phi(model=model, seed=1)
    layout = make_vp_layout(model=model, batch=batch)
    _check_design_against_forward(
        model=model, batch=batch, layout=layout, theta=_random_theta(layout=layout, seed=2)
    )


def test_a_nan_in_the_capacity_override_stays_learned_and_a_pinned_year_adds_its_solar_output(
    dp_synthetic_cache: FeatureCache,
):
    cache = _cache(dp_synthetic_cache, third=None)
    batch = _batch(cache, sites=("S1", "S2"))
    model = _model(batch=batch)
    trained_years = [int(y) for y in torch.nonzero(model.trained[0] > 0).flatten()]
    assert len(trained_years) >= 2
    override = torch.full((2, batch.n_season_years), float("nan"), dtype=torch.float64)
    override[0, trained_years[0]] = 7.0
    model.capacity_override = override
    layout = make_vp_layout(model=model, batch=batch)
    # S1's first year is pinned and its later year is learned: its increments reach the later year
    # only, so both stay columns; S2 is fully learned.
    assert layout.capacity_years[0] == tuple(trained_years)
    assert layout.capacity_years[1] == tuple(trained_years)
    _perturb_phi(model=model, seed=11)
    theta = _random_theta(layout=layout, seed=12)
    _check_design_against_forward(model=model, batch=batch, layout=layout, theta=theta)
    pinned_offset = linear_design(model=model, batch=batch, layout=layout, site=0).offset
    learned_offset = linear_design(model=model, batch=batch, layout=layout, site=1).offset
    assert float(pinned_offset.abs().max()) > 0.0
    assert float(learned_offset.abs().max()) == 0.0
    fitted, _ = solve_theta(problem=_problem(model, batch))
    assert bool((fitted[layout.constrained] >= 0).all())


def test_a_fully_pinned_substation_has_no_capacity_columns(dp_synthetic_cache: FeatureCache):
    cache = _cache(dp_synthetic_cache, third=None)
    batch = _batch(cache, sites=("S1", "S2"))
    model = _model(batch=batch)
    model.capacity_override = torch.full((2, batch.n_season_years), 7.0, dtype=torch.float64)
    layout = make_vp_layout(model=model, batch=batch)
    assert all(len(years) == 0 for years in layout.capacity_years.values())
    assert not bool(learned_increment_mask(model=model).any())
    _check_design_against_forward(
        model=model, batch=batch, layout=layout, theta=_random_theta(layout=layout, seed=3)
    )
    offset = linear_design(model=model, batch=batch, layout=layout, site=0).offset
    assert float(offset.abs().max()) > 0.0


def test_a_learned_capacity_with_no_trained_year_to_fit_it_is_refused(
    dp_synthetic_cache: FeatureCache,
):
    cache = _cache(dp_synthetic_cache, third=None)
    batch = _batch(cache, sites=("S1", "S2"))
    model = _model(batch=batch)
    override = torch.full((2, batch.n_season_years), 7.0, dtype=torch.float64)
    override[0, -1] = float("nan")
    model.capacity_override = override
    with pytest.raises(ValueError, match="learned capacity only in season-years without"):
        learned_increment_mask(model=model)


def _dense_design(
    *, model: PooledDemandModel, batch: PooledBatch, layout: VpLayout
) -> torch.Tensor:
    """Stack every substation's columns into one dense matrix (small problems only)."""
    x = torch.zeros((batch.n_rows, layout.n_columns), dtype=torch.float64)
    for site, (start, stop) in layout.rows.items():
        design = linear_design(model=model, batch=batch, layout=layout, site=site)
        columns = slice(layout.local_start[site], layout.local_start[site] + design.local.shape[1])
        x[start:stop, : layout.n_shared] = design.shared
        x[start:stop, columns] = design.local
    return x


def _dense_bounded_reference(
    *, matrix: torch.Tensor, rhs: torch.Tensor, constrained: torch.Tensor
) -> np.ndarray:
    """Solve `min theta A theta - 2 b theta` with bounds by bounded least squares on `chol(A)`."""
    factor = torch.linalg.cholesky(matrix).T
    target = torch.linalg.solve_triangular(factor.T, rhs[:, None], upper=False)[:, 0]
    lower = np.where(constrained.numpy(), 0.0, -np.inf)
    return optimize.lsq_linear(
        factor.numpy(), target.numpy(), bounds=(lower, np.inf), method="bvls", tol=1e-14
    ).x


def test_the_exact_solve_matches_an_independent_dense_bounded_solve(
    dp_synthetic_cache: FeatureCache,
):
    cache = _cache(dp_synthetic_cache, third="mva")
    batch = _batch(cache, sites=("S1", "S2", "S3"))
    model = _model(batch=batch)
    _perturb_phi(model=model, seed=4)
    problem = _problem(model, batch)
    layout = problem.layout
    theta, violation = solve_theta(problem=problem)
    assert violation < 1e-6
    # Dense reference: the normal equations from the explicit design, solved by bounded least
    # squares on the Cholesky factor, which shares no code with `solve_bounded_quadratic`.
    with torch.no_grad():
        x = _dense_design(model=model, batch=batch, layout=layout)
        offsets = [
            linear_design(model=model, batch=batch, layout=layout, site=s).offset for s in range(3)
        ]
    matrix, rhs = assemble_system(problem=problem)
    scale = 2.0 * SETTINGS.noise_sd**2
    assert torch.allclose(matrix - x.T @ x, scale * problem.penalty, atol=1e-6, rtol=1e-9)
    assert torch.allclose(rhs, x.T @ (batch.y - torch.cat(offsets)), rtol=1e-9, atol=1e-6)
    reference = _dense_bounded_reference(matrix=matrix, rhs=rhs, constrained=layout.constrained)
    assert np.allclose(theta.numpy(), reference, atol=1e-7, rtol=1e-8)

    def objective(value: torch.Tensor) -> float:
        return float(value @ matrix @ value - 2.0 * rhs @ value)

    # No projected-gradient run reaches a lower objective than the exact solve.
    value = theta.clone()
    moment = torch.zeros_like(value)
    second = torch.zeros_like(value)
    best = objective(theta)
    for step in range(1, 5001):
        gradient = 2.0 * (matrix @ value - rhs)
        moment = 0.9 * moment + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient**2
        update = 1e-3 * (moment / (1 - 0.9**step)) / ((second / (1 - 0.999**step)).sqrt() + 1e-8)
        value = value - update
        value = torch.where(layout.constrained, value.clamp(min=0.0), value)
        best = min(best, objective(value))
    assert best >= objective(theta) - 1e-9 * abs(objective(theta))


def _matrix_and_model_objective_differ_by_a_constant(
    *, problem: VpProblem, batch: PooledBatch
) -> None:
    matrix, rhs = assemble_system(problem=problem)
    scale = 2.0 * SETTINGS.noise_sd**2
    values = []
    for seed in (6, 7):
        theta = _random_theta(layout=problem.layout, seed=seed)
        load_theta(model=problem.model, layout=problem.layout, theta=theta)
        with torch.no_grad():
            nats = float(vp_objective(model=problem.model, batch=batch, settings=SETTINGS))
        values.append((nats, float(theta @ matrix @ theta - 2.0 * rhs @ theta) / scale))
    # The two differ by a constant (the target's sum of squares and the terms in `phi` alone).
    assert values[0][0] - values[1][0] == pytest.approx(values[0][1] - values[1][1], rel=1e-8)


def test_the_matrix_objective_differences_equal_the_model_objective_differences(
    dp_synthetic_cache: FeatureCache,
):
    """The Gram matrix and the model's own objective describe the same function of `theta`."""
    cache = _cache(dp_synthetic_cache, third="mva")
    batch = _batch(cache, sites=("S1", "S2", "S3"))
    model = _model(batch=batch, hidden=(6,))
    _perturb_phi(model=model, seed=5)
    _matrix_and_model_objective_differ_by_a_constant(problem=_problem(model, batch), batch=batch)


def test_the_objective_leaves_the_stored_centring_means_alone(dp_synthetic_cache: FeatureCache):
    cache = _cache(dp_synthetic_cache, third=None)
    batch = _batch(cache, sites=("S1", "S2"))
    model = _model(batch=batch)
    model.centre_means.copy_(torch.arange(model.centre_means.numel(), dtype=torch.float64))
    before = model.centre_means.clone()
    vp_objective(model=model, batch=batch, settings=SETTINGS)
    assert torch.equal(model.centre_means, before)


def _truth_batch(
    base: FeatureCache, *, seed: int, year_shift: float = 0.0
) -> tuple[PooledBatch, PooledDemandModel]:
    """Replace each site's target with the pooled model's output at known parameters, plus noise."""
    batch = _batch(_cache(base, third=None), sites=("S1", "S2"))
    truth = _model(batch=batch, hidden=(), shapes=2, seed=seed)
    with torch.no_grad():
        truth.threshold.copy_(torch.tensor([14.0, 14.0], dtype=torch.float64))
        truth.site_logits.copy_(torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float64))
        truth.shapes.copy_(3.0 * torch.randn_like(truth.shapes))
        slope = torch.tensor(0.5, dtype=torch.float64).log().expand_as(truth.slope_raw)
        truth.slope_raw.copy_(slope)
        truth.capacity_raw.fill_(1.0)
    truth.train()
    with torch.no_grad():
        clean = truth(batch)
    generator = torch.Generator().manual_seed(seed)
    noise = 0.3 * torch.randn(clean.shape, dtype=torch.float64, generator=generator)
    years = (batch.season_slot % batch.n_season_years).to(torch.float64)
    return replace(batch, y=clean + noise + year_shift * years), truth


def _gradient_probes(
    *, model: PooledDemandModel
) -> list[tuple[torch.nn.Parameter, tuple[int, ...]]]:
    """One entry of each kind of non-linear parameter, picked by identity."""
    assert model.interaction is not None
    return [
        (model.site_logits, (0, 0)),
        (model.threshold, (0,)),
        (model.inertia_logits, (1,)),
        (model.population_mean["threshold"], ()),
        (model.interaction[0].weight, (0, 0)),
    ]


def _finite_difference(
    *,
    problem: VpProblem,
    phi: list[torch.nn.Parameter],
    parameter: torch.nn.Parameter,
    index: tuple[int, ...],
) -> float:
    step = 1e-5
    values = []
    for sign in (1.0, -1.0):
        with torch.no_grad():
            parameter[index] += sign * step
        values.append(_evaluate(problem=problem, phi=phi)[0])
        with torch.no_grad():
            parameter[index] -= sign * step
    return (values[0] - values[1]) / (2.0 * step)


def _assert_gradient_matches_finite_differences(
    *, problem: VpProblem, phi: list[torch.nn.Parameter], skip: int = 0
) -> None:
    _, gradients, _ = _evaluate(problem=problem, phi=phi)
    for parameter, index in _gradient_probes(model=problem.model)[skip:]:
        position = next(i for i, p in enumerate(phi) if p is parameter)
        numeric = _finite_difference(problem=problem, phi=phi, parameter=parameter, index=index)
        assert float(gradients[position][index]) == pytest.approx(numeric, rel=1e-4, abs=1e-8)


def test_the_envelope_gradient_matches_central_finite_differences(dp_synthetic_cache: FeatureCache):
    batch, _ = _truth_batch(dp_synthetic_cache, seed=11, year_shift=4.0)
    model = _model(batch=batch, hidden=(4,), shapes=2, seed=12)
    _perturb_phi(model=model, seed=13)
    with torch.no_grad():
        model.threshold.copy_(torch.tensor([14.5, 15.5], dtype=torch.float64))
    problem = _problem(model, batch)
    phi = phi_parameters(model=model)
    with trainable_only(model=model, parameters=phi):
        theta, _ = solve_theta(problem=problem)
        assert float(theta[problem.layout.constrained].min()) > 0.0
        _assert_gradient_matches_finite_differences(problem=problem, phi=phi)


def test_the_gradient_needs_the_centring_means_recomputed_inside_the_graph(
    dp_synthetic_cache: FeatureCache, monkeypatch: pytest.MonkeyPatch
):
    """A step in level between season-years makes the means matter; detaching them is caught."""
    batch, _ = _truth_batch(dp_synthetic_cache, seed=21, year_shift=6.0)
    model = _model(batch=batch, hidden=(4,), shapes=2, seed=22)
    _perturb_phi(model=model, seed=23)
    problem = _problem(model, batch)
    phi = phi_parameters(model=model)
    assert model.interaction is not None
    hidden_weight = model.interaction[0].weight
    position = next(i for i, p in enumerate(phi) if p is hidden_weight)
    with trainable_only(model=model, parameters=phi):
        _, honest, _ = _evaluate(problem=problem, phi=phi)
        numeric = _finite_difference(
            problem=problem, phi=phi, parameter=hidden_weight, index=(0, 0)
        )
        assert float(honest[position][0, 0]) == pytest.approx(numeric, rel=1e-4, abs=1e-8)
        original = PooledDemandModel._centred

        def detached(
            self: PooledDemandModel, *, values: torch.Tensor, slot: torch.Tensor
        ) -> torch.Tensor:
            return original(self, values=values.detach(), slot=slot) + values - values.detach()

        monkeypatch.setattr(PooledDemandModel, "_centred", detached)
        _, mutant, _ = _evaluate(problem=problem, phi=phi)
    difference = (honest[position] - mutant[position]).abs().max()
    assert float(difference) > 0.05 * float(honest[position].abs().max())


def test_the_envelope_gradient_holds_with_a_bound_active(dp_synthetic_cache: FeatureCache):
    """Targets with no heating signal push some slopes and increments to zero."""
    batch = _batch(_cache(dp_synthetic_cache, third=None), sites=("S1", "S2"))
    generator = torch.Generator().manual_seed(21)
    flat = 40.0 + 2.0 * torch.randn(batch.n_rows, dtype=torch.float64, generator=generator)
    batch = replace(batch, y=flat)
    model = _model(batch=batch, hidden=(4,), shapes=2, seed=22)
    _perturb_phi(model=model, seed=23)
    problem = _problem(model, batch)
    theta, _ = solve_theta(problem=problem)
    assert int(((theta == 0.0) & problem.layout.constrained).sum()) > 0
    phi = phi_parameters(model=model)
    with trainable_only(model=model, parameters=phi):
        _assert_gradient_matches_finite_differences(problem=problem, phi=phi, skip=1)


def _active_set(*, problem: VpProblem) -> tuple[bool, ...]:
    theta, _ = solve_theta(problem=problem)
    return tuple(bool(v) for v in ((theta == 0.0) & problem.layout.constrained).tolist())


def test_the_envelope_gradient_is_continuous_across_an_active_set_change(
    dp_synthetic_cache: FeatureCache,
):
    """Where a slope or an increment leaves zero, the gradient still matches to first order."""
    batch = _batch(_cache(dp_synthetic_cache, third=None), sites=("S1", "S2"))
    generator = torch.Generator().manual_seed(51)
    flat = 40.0 + 2.0 * torch.randn(batch.n_rows, dtype=torch.float64, generator=generator)
    batch = replace(batch, y=flat)
    model = _model(batch=batch, shapes=2, seed=52)
    problem = _problem(model, batch)
    phi = phi_parameters(model=model)
    threshold = model.threshold

    def active_at(value: float) -> tuple[bool, ...]:
        with torch.no_grad():
            threshold[0] = value
        return _active_set(problem=problem)

    with trainable_only(model=model, parameters=phi):
        grid = np.linspace(8.0, 24.0, 17)
        sets = [active_at(float(v)) for v in grid]
        changes = [i for i in range(len(grid) - 1) if sets[i] != sets[i + 1]]
        assert changes, "no active-set change along the threshold scan"
        low, high = float(grid[changes[0]]), float(grid[changes[0] + 1])
        low_set = sets[changes[0]]
        for _ in range(40):
            middle = 0.5 * (low + high)
            if active_at(middle) == low_set:
                low = middle
            else:
                high = middle
        kink = 0.5 * (low + high)
        step = 1e-4
        assert active_at(kink + step) != active_at(kink - step)
        with torch.no_grad():
            threshold[0] = kink
        _, gradients, _ = _evaluate(problem=problem, phi=phi)
        numeric = _finite_difference(problem=problem, phi=phi, parameter=threshold, index=(0,))
    position = next(i for i, p in enumerate(phi) if p is threshold)
    assert float(gradients[position][0]) == pytest.approx(numeric, rel=1e-3, abs=1e-7)


def test_the_fit_reaches_the_objective_of_the_parameters_that_made_the_targets(
    dp_synthetic_cache: FeatureCache,
):
    batch, truth = _truth_batch(dp_synthetic_cache, seed=31)
    with torch.no_grad():
        reference = float(vp_objective(model=truth, batch=batch, settings=SETTINGS))
    model = _model(batch=batch, hidden=(), shapes=2, seed=32)
    with torch.no_grad():
        start = float(vp_objective(model=model, batch=batch, settings=SETTINGS))
    # No Adam fallback: L-BFGS alone has to reach the optimum and say it converged.
    result = fit_pooled_vp(
        model=model, batch=batch, settings=SETTINGS, max_evaluations=150, adam_steps=0
    )
    assert result.converged
    assert result.objective < start - 1e3
    assert result.history[-1] <= result.history[0]
    assert result.kkt_violation < 1e-6
    # The truth is a feasible point of the same objective, so a fit that reaches the optimum
    # scores at least as well. A stalled fit does not.
    assert result.objective <= reference + 1.0
    # The truth has heating thresholds of 14 degrees and slopes of 0.5 in every band.
    assert model.threshold.detach().tolist() == pytest.approx([14.0, 14.0], abs=1.5)
    slope = torch.nn.functional.softplus(model.slope_raw.detach())
    assert float(slope.mean()) == pytest.approx(0.5, rel=0.2)
    with torch.no_grad():
        final = float(vp_objective(model=model, batch=batch, settings=SETTINGS))
    assert final == pytest.approx(result.objective, rel=1e-9)


def test_a_fit_cut_short_says_it_did_not_converge(dp_synthetic_cache: FeatureCache):
    batch, _ = _truth_batch(dp_synthetic_cache, seed=41)
    cut_short = fit_pooled_vp(
        model=_model(batch=batch, shapes=2, seed=42),
        batch=batch,
        settings=SETTINGS,
        max_evaluations=5,
        adam_steps=0,
    )
    assert not cut_short.converged
    assert cut_short.adam_objective is None
    assert cut_short.evaluations <= 5 + 25


def test_the_stored_forecast_season_year_means_are_the_last_trained_years_means(
    dp_synthetic_cache: FeatureCache,
):
    cache = _cache(dp_synthetic_cache, third=None)
    batch = _batch(cache, sites=("S1", "S2"))
    model = _model(batch=batch, hidden=(4,))
    fit_pooled_vp(model=model, batch=batch, settings=SETTINGS, max_evaluations=15, adam_steps=0)
    means = model.centre_means.reshape(model.n_sites, model.n_season_years)
    for site in range(model.n_sites):
        trained = [int(y) for y in torch.nonzero(model.trained[site] > 0).flatten()]
        last = trained[-1]
        assert float(means[site, last]) != 0.0
        for later in range(last + 1, model.n_season_years):
            assert float(means[site, later]) == float(means[site, last])


def test_forecast_rows_never_read_their_targets(dp_synthetic_cache: FeatureCache):
    cache = _cache(dp_synthetic_cache, third=None)
    batch = _batch(cache, sites=("S1", "S2"))
    model = _model(batch=batch, hidden=(4,))
    fit_pooled_vp(model=model, batch=batch, settings=SETTINGS, max_evaluations=15, adam_steps=0)
    forecast = build_pooled_batch(
        cache=cache,
        sites=("S1", "S2"),
        cut_end=np.datetime64("2025-07-01T00:00:00", "us"),
        after=CUT,
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(41)
    noise = 50.0 * torch.randn(forecast.y.shape, dtype=torch.float64, generator=generator)
    model.eval()
    with torch.no_grad():
        first = model(forecast)
        second = model(replace(forecast, y=forecast.y + noise))
    assert torch.equal(first, second)


def test_a_fit_needs_float64(dp_synthetic_cache: FeatureCache):
    cache = _cache(dp_synthetic_cache, third=None)
    batch = build_pooled_batch(cache=cache, sites=("S1", "S2"), cut_end=CUT, after=WINDOW_START)
    model = PooledDemandModel(
        config=DpConfig(),
        n_sites=2,
        n_season_years=batch.n_season_years,
        initial_level=torch.full((2,), 50.0),
    )
    with pytest.raises(ValueError, match="float64"):
        fit_pooled_vp(model=model, batch=batch, settings=SETTINGS)


def test_an_objective_with_a_reactive_mva_operator_is_refused(dp_synthetic_cache: FeatureCache):
    batch = _batch(_cache(dp_synthetic_cache, third="mva"), sites=("S1", "S3"))
    model = _model(batch=batch)
    with torch.no_grad():
        model.tan_phi.fill_(0.2)
    with pytest.raises(ValueError, match="tan"):
        vp_objective(model=model, batch=batch, settings=SETTINGS)


def test_a_variational_model_and_a_site_capacity_prior_are_refused(
    dp_synthetic_cache: FeatureCache,
):
    batch = _batch(_cache(dp_synthetic_cache, third=None), sites=("S1", "S2"))
    variational = PooledDemandModel(
        config=DpConfig(variational=True),
        n_sites=2,
        n_season_years=batch.n_season_years,
        initial_level=torch.full((2,), 50.0),
    ).to(dtype=torch.float64)
    with pytest.raises(ValueError, match="MAP objective"):
        vp_objective(model=variational, batch=batch, settings=SETTINGS)
    model = _model(batch=batch)
    with torch.no_grad():
        model.capacity_prior[0] = 1.0
    with pytest.raises(ValueError, match="capacity prior"):
        vp_objective(model=model, batch=batch, settings=SETTINGS)


def test_rows_not_sorted_by_substation_are_refused(dp_synthetic_cache: FeatureCache):
    batch = _batch(_cache(dp_synthetic_cache, third=None), sites=("S1", "S2"))
    flipped = replace(batch, site=batch.site.flip(0))
    with pytest.raises(ValueError, match="sorted by substation"):
        site_bounds(batch=flipped, n_sites=2)


def test_a_network_that_does_not_end_in_a_linear_layer_is_refused(
    dp_synthetic_cache: FeatureCache,
):
    batch = _batch(_cache(dp_synthetic_cache, third=None), sites=("S1", "S2"))
    model = _model(batch=batch, hidden=(4,))
    model.interaction = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Tanh())
    with pytest.raises(TypeError, match="linear layer"):
        _last_layer(model=model)


def test_a_failed_block_restores_the_training_flags(dp_synthetic_cache: FeatureCache):
    batch, _ = _truth_batch(dp_synthetic_cache, seed=91)
    model = _model(batch=batch, shapes=2, seed=92)
    before = {name: p.requires_grad for name, p in model.named_parameters()}
    with (
        pytest.raises(RuntimeError, match="boom"),
        trainable_only(model=model, parameters=phi_parameters(model=model)),
    ):
        raise RuntimeError("boom")
    assert {name: p.requires_grad for name, p in model.named_parameters()} == before


def test_a_failed_convergence_runs_the_adam_fallback_and_reports_both_objectives(
    dp_synthetic_cache: FeatureCache,
):
    batch, _ = _truth_batch(dp_synthetic_cache, seed=61)
    model = _model(batch=batch, hidden=(), shapes=2, seed=62)
    result = fit_pooled_vp(
        model=model, batch=batch, settings=SETTINGS, max_evaluations=4, adam_steps=5
    )
    assert not result.converged
    assert result.adam_objective is not None
    assert result.objective <= result.lbfgs_objective + 1e-6
    assert result.evaluations <= 4 + 25


def _map_model(batch: PooledBatch) -> PooledDemandModel:
    return PooledDemandModel(
        config=DpConfig(shared_shapes=2, population_scales=PopulationScales(deviation=1.0)),
        n_sites=2,
        n_season_years=batch.n_season_years,
        initial_level=torch.full((2,), 50.0),
    ).to(dtype=torch.float64)


def test_the_adam_fit_on_the_vp_objective_lowers_it(dp_synthetic_cache: FeatureCache):
    batch, _ = _truth_batch(dp_synthetic_cache, seed=71)
    model = _map_model(batch)
    model.mark_trained_years(batch=batch)
    model.start_from_batch(batch=batch)
    freeze_operator_for_vp(model=model)
    with torch.no_grad():
        start = float(vp_objective(model=model, batch=batch, settings=SETTINGS))
    result = fit_pooled_adam_on_vp_objective(
        model=model, batch=batch, settings=SETTINGS, steps=30, learning_rate=0.05
    )
    assert result.losses[-1] < result.losses[0]
    with torch.no_grad():
        end = float(vp_objective(model=model, batch=batch, settings=SETTINGS))
    assert end < start
    assert result.losses[0] == pytest.approx(start / batch.n_rows, rel=1e-6)


def test_an_adam_warm_start_lowers_the_objective_before_the_exact_solves(
    dp_synthetic_cache: FeatureCache,
):
    batch, _ = _truth_batch(dp_synthetic_cache, seed=81)
    warm = fit_pooled_vp(
        model=_map_model(batch),
        batch=batch,
        settings=SETTINGS,
        max_evaluations=6,
        adam_steps=0,
        warm_start_steps=20,
    )
    assert warm.warm_start_objective is not None
    cold = fit_pooled_vp(
        model=_model(batch=batch, hidden=(), shapes=2, seed=82),
        batch=batch,
        settings=SETTINGS,
        max_evaluations=1,
        adam_steps=0,
    )
    assert cold.warm_start_objective is None
    assert warm.history[0] < cold.history[0]


def _onboarding_setup(
    base: FeatureCache,
) -> tuple[PooledDemandModel, PooledBatch, FeatureCache]:
    """Fit S1 and S2 by VP with S3 held out; return the model, S3's last days, and the cache."""
    cache = _cache(base, third="mw")
    batch = _batch(cache, sites=("S1", "S2", "S3"))
    others = select_rows(batch=batch, keep=~site_mask(batch=batch, label="S3"))
    model = _model(batch=batch, hidden=(4,), shapes=2, seed=101, map_prior=True)
    fit_pooled_vp(model=model, batch=others, settings=SETTINGS, max_evaluations=40, adam_steps=0)
    ends = torch.as_tensor(cache.end_times[batch.row.cpu().numpy()] > CUT - np.timedelta64(28, "D"))
    own = select_rows(batch=batch, keep=site_mask(batch=batch, label="S3") & ends)
    return model, own, cache


def _onboard(
    model: PooledDemandModel,
    own: PooledBatch,
    cache: FeatureCache,
    fitter: Callable[[PooledDemandModel, PooledBatch, int], float],
) -> dict[str, float]:
    return onboard_site(
        model=model,
        batch=own,
        label="S3",
        cache=cache,
        history_end=CUT,
        forecast_season_year=FORECAST_YEAR,
        steps=0,
        fitter=fitter,
    )


def test_onboarding_by_variable_projection_changes_only_the_new_substation(
    dp_synthetic_cache: FeatureCache,
):
    model, own, cache = _onboarding_setup(dp_synthetic_cache)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    centres = model.centre_means.clone()
    info = _onboard(model, own, cache, vp_site_fitter(settings=SETTINGS, max_evaluations=40))
    site = 2
    assert info["rows"] == own.n_rows
    assert np.isfinite(info["loss"])
    per_site = ("level", "site_logits", "deviation", "threshold", "slope_raw", "capacity_raw")
    for name, parameter in model.named_parameters():
        change = (parameter.detach() - before[name]).abs()
        if name in per_site:
            others_changed = torch.arange(change.shape[0]) != site
            assert float(change[others_changed].sum()) == 0.0, name
        else:
            assert float(change.sum()) == 0.0, name
    assert float((model.level.detach() - before["level"]).abs()[site]) > 0.0
    assert float((model.threshold.detach() - before["threshold"]).abs()[site]) > 0.0
    kept = torch.arange(model.n_sites * model.n_season_years) // model.n_season_years != site
    assert torch.equal(model.centre_means[kept], centres[kept])
    with torch.no_grad():
        assert torch.isfinite(model(own)).all()


def test_the_frozen_site_solve_matches_an_independent_dense_bounded_solve(
    dp_synthetic_cache: FeatureCache,
):
    model, own, cache = _onboarding_setup(dp_synthetic_cache)
    checked: list[bool] = []

    def probe(model: PooledDemandModel, batch: PooledBatch, site: int) -> float:
        problem = make_vp_problem(model=model, batch=batch, settings=SETTINGS, frozen_shared=True)
        layout = problem.layout
        assert layout.n_shared == 0
        assert layout.sites == (site,)
        theta, violation = solve_theta(problem=problem)
        assert violation < 1e-6
        matrix, rhs = assemble_system(problem=problem)
        reference = _dense_bounded_reference(matrix=matrix, rhs=rhs, constrained=layout.constrained)
        assert np.allclose(theta.numpy(), reference, atol=1e-7, rtol=1e-8)
        # With the shared parts frozen, the fixed population means centre the priors: the matrix
        # objective and the model objective agree up to a constant.
        _matrix_and_model_objective_differ_by_a_constant(problem=problem, batch=batch)
        checked.append(True)
        return 0.0

    _onboard(model, own, cache, probe)
    assert checked


def test_a_modelled_mva_demand_that_is_not_positive_is_refused(dp_synthetic_cache: FeatureCache):
    batch = _batch(_cache(dp_synthetic_cache, third="mva"), sites=("S1", "S3"))
    model = _model(batch=batch)
    with torch.no_grad():
        model.level.fill_(-100.0)
    with pytest.raises(ValueError, match="modelled demand"):
        _check_mva_demand(model=model, batch=batch)


def test_onboarding_by_variable_projection_refuses_a_base_model_with_a_reactive_operator(
    dp_synthetic_cache: FeatureCache,
):
    model, own, cache = _onboarding_setup(dp_synthetic_cache)
    with torch.no_grad():
        model.tan_phi[0] = 0.2
    with pytest.raises(ValueError, match="zero MVA operator"):
        _onboard(model, own, cache, vp_site_fitter(settings=SETTINGS, max_evaluations=5))
    # A fitter that raises leaves no gradient hooks and no changed training flags behind.
    assert all(not p._backward_hooks for p in model.parameters())


def test_the_site_fitter_returns_the_loss_of_the_best_point_it_saw(
    dp_synthetic_cache: FeatureCache, monkeypatch: pytest.MonkeyPatch
):
    model, own, cache = _onboarding_setup(dp_synthetic_cache)
    seen: list[float] = []
    original = dp_pooled_vp._run_lbfgs

    def recording(**kwargs):  # noqa: ANN003, ANN202
        history, best, stopped = original(**kwargs)
        seen.append(best.objective)
        return history, best, stopped

    monkeypatch.setattr(dp_pooled_vp, "_run_lbfgs", recording)
    info = _onboard(model, own, cache, vp_site_fitter(settings=SETTINGS, max_evaluations=25))
    assert info["loss"] == pytest.approx(seen[0], rel=1e-9)


def test_a_training_batch_holds_no_row_after_its_cut(dp_synthetic_cache: FeatureCache):
    cache = _cache(dp_synthetic_cache, third=None)
    batch = _batch(cache, sites=("S1", "S2"))
    assert cache.end_times[batch.row.cpu().numpy()].max() <= CUT
    later = build_pooled_batch(
        cache=cache,
        sites=("S1", "S2"),
        cut_end=CUT + np.timedelta64(1, "D"),
        after=WINDOW_START,
        dtype=torch.float64,
    )
    assert later.n_rows > batch.n_rows
