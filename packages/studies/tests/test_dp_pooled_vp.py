"""Tests of the variable-projection fit of the pooled differentiable-physics model."""

from dataclasses import replace
from typing import Final

import numpy as np
import pytest

torch = pytest.importorskip("torch")
optimize = pytest.importorskip("scipy.optimize")

from studies.dp_features import FeatureCache  # noqa: E402
from studies.dp_pooled import (  # noqa: E402
    DpConfig,
    PooledBatch,
    PooledDemandModel,
    PopulationScales,
    build_pooled_batch,
)
from studies.dp_pooled_vp import (  # noqa: E402
    VpSettings,
    _evaluate,
    assemble_system,
    assert_batch_before_cut,
    fit_pooled_adam_on_vp_objective,
    fit_pooled_vp,
    freeze_operator_for_vp,
    linear_design,
    load_theta,
    make_vp_layout,
    penalty_system,
    phi_parameters,
    site_bounds,
    solve_theta,
    vp_objective,
)

CUT: Final[np.datetime64] = np.datetime64("2024-07-01T00:00:00", "us")
WINDOW_START: Final[np.datetime64] = np.datetime64("2022-07-01T00:00:00", "us")
SETTINGS: Final[VpSettings] = VpSettings(noise_sd=2.0)


def _cache(base: FeatureCache, *, mva: bool) -> FeatureCache:
    """Return the synthetic cache with S1 (MW), S2 (MW, other targets), and optionally S3 (MVA)."""
    first = base.sites["S1"]
    generator = np.random.default_rng(7)
    second = replace(
        first,
        table=replace(first.table, site="S2", time_series_id=2),
        y=first.y * 1.2 + generator.normal(scale=1.0, size=first.y.shape),
    )
    sites = {"S1": first, "S2": second}
    if mva:
        third = replace(
            first,
            table=replace(first.table, site="S3", units="MVA", time_series_id=3),
            dark=np.nan_to_num(first.ghi_w_m2) < 1.0,
            y=first.y * 0.7 + 30.0,
        )
        sites["S3"] = third
    return replace(base, sites=sites, node_ids={name: (name,) for name in sites})


def _batch(cache: FeatureCache, *, sites: tuple[str, ...]) -> PooledBatch:
    return build_pooled_batch(
        cache=cache,
        sites=sites,
        cut_end=CUT,
        after=WINDOW_START,
        dtype=torch.float64,
    )


def _model(
    *, batch: PooledBatch, hidden: tuple[int, ...] = (), shapes: int = 2, seed: int = 0
) -> PooledDemandModel:
    torch.manual_seed(seed)
    model = PooledDemandModel(
        config=DpConfig(shared_shapes=shapes, interaction_hidden=hidden),
        n_sites=len(batch.site_labels),
        n_season_years=batch.n_season_years,
        initial_level=torch.full((len(batch.site_labels),), 50.0),
    ).to(dtype=torch.float64)
    model.mark_trained_years(batch=batch)
    model.start_from_batch(batch=batch)
    freeze_operator_for_vp(model=model)
    return model


def _random_theta(*, layout, seed: int) -> torch.Tensor:  # noqa: ANN001
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
            parameter.add_(
                0.4 * torch.randn(parameter.shape, dtype=parameter.dtype, generator=generator)
            )
        model.threshold.add_(15.0)


@pytest.mark.parametrize("hidden", [(), (8, 8)])
def test_the_linear_design_reproduces_the_pooled_model_forward(
    dp_synthetic_cache: FeatureCache, hidden: tuple[int, ...]
):
    cache = _cache(dp_synthetic_cache, mva=True)
    batch = _batch(cache, sites=("S1", "S2", "S3"))
    model = _model(batch=batch, hidden=hidden)
    _perturb_phi(model=model, seed=1)
    layout = make_vp_layout(model=model, batch=batch)
    theta = _random_theta(layout=layout, seed=2)
    load_theta(model=model, layout=layout, theta=theta)
    model.train()
    with torch.no_grad():
        forward = model(batch)
    q = layout.n_shared
    for site, (start, stop) in enumerate(site_bounds(batch=batch, n_sites=model.n_sites)):
        design = linear_design(model=model, batch=batch, layout=layout, site=site)
        columns = slice(layout.local_start[site], layout.local_start[site] + design.local.shape[1])
        prediction = design.shared @ theta[:q] + design.local @ theta[columns] + design.offset
        assert torch.allclose(prediction, forward[start:stop], atol=1e-6, rtol=0.0)


def test_the_linear_design_reproduces_the_forward_with_a_pinned_capacity(
    dp_synthetic_cache: FeatureCache,
):
    cache = _cache(dp_synthetic_cache, mva=False)
    batch = _batch(cache, sites=("S1", "S2"))
    model = _model(batch=batch)
    pinned = torch.full((2, batch.n_season_years), 7.0, dtype=torch.float64)
    model.capacity_override = pinned
    layout = make_vp_layout(model=model, batch=batch)
    assert all(len(years) == 0 for years in layout.capacity_years.values())
    theta = _random_theta(layout=layout, seed=3)
    load_theta(model=model, layout=layout, theta=theta)
    model.train()
    with torch.no_grad():
        forward = model(batch)
    q = layout.n_shared
    for site, (start, stop) in enumerate(site_bounds(batch=batch, n_sites=model.n_sites)):
        design = linear_design(model=model, batch=batch, layout=layout, site=site)
        columns = slice(layout.local_start[site], layout.local_start[site] + design.local.shape[1])
        prediction = design.shared @ theta[:q] + design.local @ theta[columns] + design.offset
        assert float(design.offset.abs().max()) > 0.0
        assert torch.allclose(prediction, forward[start:stop], atol=1e-6, rtol=0.0)


def _dense_design(*, model, batch, layout) -> torch.Tensor:  # noqa: ANN001
    """Stack every substation's columns into one dense matrix (small problems only)."""
    x = torch.zeros((batch.n_rows, layout.n_columns), dtype=torch.float64)
    for site, (start, stop) in enumerate(site_bounds(batch=batch, n_sites=model.n_sites)):
        design = linear_design(model=model, batch=batch, layout=layout, site=site)
        columns = slice(layout.local_start[site], layout.local_start[site] + design.local.shape[1])
        x[start:stop, : layout.n_shared] = design.shared
        x[start:stop, columns] = design.local
    return x


def test_the_exact_solve_matches_an_independent_dense_bounded_solve(
    dp_synthetic_cache: FeatureCache,
):
    cache = _cache(dp_synthetic_cache, mva=True)
    batch = _batch(cache, sites=("S1", "S2", "S3"))
    model = _model(batch=batch)
    _perturb_phi(model=model, seed=4)
    layout = make_vp_layout(model=model, batch=batch)
    theta, violation = solve_theta(model=model, batch=batch, layout=layout, settings=SETTINGS)
    assert violation < 1e-6
    # Dense reference: the normal equations from the explicit design, solved by bounded least
    # squares on the Cholesky factor, which shares no code with `solve_bounded_quadratic`.
    with torch.no_grad():
        x = _dense_design(model=model, batch=batch, layout=layout)
    offset = torch.cat(
        [
            linear_design(model=model, batch=batch, layout=layout, site=s).offset.detach()
            for s in range(3)
        ]
    )
    matrix, rhs = assemble_system(model=model, batch=batch, layout=layout, settings=SETTINGS)
    gram = x.T @ x
    assert torch.allclose(rhs, x.T @ (batch.y - offset), rtol=1e-9, atol=1e-6)
    expected = (
        2.0
        * SETTINGS.noise_sd**2
        * penalty_system(
            layout=layout, settings=SETTINGS, dtype=torch.float64, device=torch.device("cpu")
        )
    )
    assert torch.allclose(matrix - gram, expected, atol=1e-6, rtol=1e-9)
    factor = torch.linalg.cholesky(matrix).T
    target = torch.linalg.solve_triangular(factor.T, rhs[:, None], upper=False)[:, 0]
    lower = np.where(layout.constrained.numpy(), 0.0, -np.inf)
    reference = optimize.lsq_linear(
        factor.numpy(), target.numpy(), bounds=(lower, np.inf), method="bvls", tol=1e-14
    ).x
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


def test_the_matrix_objective_differences_equal_the_model_objective_differences(
    dp_synthetic_cache: FeatureCache,
):
    """The Gram matrix and the model's own objective describe the same function of `theta`."""
    cache = _cache(dp_synthetic_cache, mva=True)
    batch = _batch(cache, sites=("S1", "S2", "S3"))
    model = _model(batch=batch, hidden=(6,))
    _perturb_phi(model=model, seed=5)
    layout = make_vp_layout(model=model, batch=batch)
    matrix, rhs = assemble_system(model=model, batch=batch, layout=layout, settings=SETTINGS)
    scale = 2.0 * SETTINGS.noise_sd**2
    values = []
    for seed in (6, 7):
        theta = _random_theta(layout=layout, seed=seed)
        load_theta(model=model, layout=layout, theta=theta)
        with torch.no_grad():
            nats = float(vp_objective(model=model, batch=batch, settings=SETTINGS))
        quadratic = float(theta @ matrix @ theta - 2.0 * rhs @ theta)
        values.append((nats, quadratic / scale))
    # The two differ by a constant (the target's sum of squares and the terms in `phi` alone).
    assert values[0][0] - values[1][0] == pytest.approx(values[0][1] - values[1][1], rel=1e-8)


def _truth_batch(base: FeatureCache, *, seed: int) -> tuple[FeatureCache, PooledBatch]:
    """Replace each site's target with the pooled model's output at known parameters, plus noise."""
    cache = _cache(base, mva=False)
    batch = _batch(cache, sites=("S1", "S2"))
    model = _model(batch=batch, hidden=(), shapes=2, seed=seed)
    with torch.no_grad():
        model.threshold.copy_(torch.tensor([14.0, 14.0], dtype=torch.float64))
        model.site_logits.copy_(torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float64))
        model.shapes.copy_(3.0 * torch.randn_like(model.shapes))
        model.slope_raw.copy_(
            torch.tensor(0.5, dtype=torch.float64).log().expand_as(model.slope_raw)
        )
        model.capacity_raw.fill_(1.0)
    model.train()
    with torch.no_grad():
        truth = model(batch)
    generator = torch.Generator().manual_seed(seed)
    noisy = truth + 0.3 * torch.randn(truth.shape, dtype=torch.float64, generator=generator)
    return cache, replace(batch, y=noisy)


def test_the_envelope_gradient_matches_central_finite_differences(dp_synthetic_cache: FeatureCache):
    _, batch = _truth_batch(dp_synthetic_cache, seed=11)
    model = _model(batch=batch, hidden=(4,), shapes=2, seed=12)
    _perturb_phi(model=model, seed=13)
    with torch.no_grad():
        model.threshold.copy_(torch.tensor([14.5, 15.5], dtype=torch.float64))
    layout = make_vp_layout(model=model, batch=batch)
    phi = phi_parameters(model=model)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in phi:
        parameter.requires_grad = True
    _, gradients, _ = _evaluate(model=model, batch=batch, layout=layout, settings=SETTINGS, phi=phi)
    interior = float(
        solve_theta(model=model, batch=batch, layout=layout, settings=SETTINGS)[0][
            layout.constrained
        ].min()
    )
    assert interior > 0.0
    step = 1e-5
    # One entry of each kind of non-linear parameter: loading, threshold, inertia, hidden weight.
    probes = [(0, (0, 0)), (1, (0,)), (2, (1,)), (4, (0, 0))]
    names = ["site_logits", "threshold", "inertia_logits", "hidden weight"]
    assert len(phi) > max(position for position, _ in probes), names
    for position, index in probes:
        values = []
        for sign in (1.0, -1.0):
            with torch.no_grad():
                phi[position][index] += sign * step
            values.append(
                _evaluate(model=model, batch=batch, layout=layout, settings=SETTINGS, phi=phi)[0]
            )
            with torch.no_grad():
                phi[position][index] -= sign * step
        numeric = (values[0] - values[1]) / (2.0 * step)
        analytic = float(gradients[position][index])
        assert analytic == pytest.approx(numeric, rel=1e-4, abs=1e-8)


def test_the_envelope_gradient_holds_with_a_bound_active(dp_synthetic_cache: FeatureCache):
    """Targets with no heating signal push some slopes and increments to zero."""
    cache = _cache(dp_synthetic_cache, mva=False)
    batch = _batch(cache, sites=("S1", "S2"))
    generator = torch.Generator().manual_seed(21)
    flat = 40.0 + 2.0 * torch.randn(batch.n_rows, dtype=torch.float64, generator=generator)
    batch = replace(batch, y=flat)
    model = _model(batch=batch, hidden=(4,), shapes=2, seed=22)
    _perturb_phi(model=model, seed=23)
    layout = make_vp_layout(model=model, batch=batch)
    theta, _ = solve_theta(model=model, batch=batch, layout=layout, settings=SETTINGS)
    at_bound = (theta == 0.0) & layout.constrained
    assert int(at_bound.sum()) > 0
    phi = phi_parameters(model=model)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in phi:
        parameter.requires_grad = True
    _, gradients, _ = _evaluate(model=model, batch=batch, layout=layout, settings=SETTINGS, phi=phi)
    step = 1e-5
    for position, index in [(1, (0,)), (2, (1,)), (4, (0, 0))]:
        values = []
        for sign in (1.0, -1.0):
            with torch.no_grad():
                phi[position][index] += sign * step
            values.append(
                _evaluate(model=model, batch=batch, layout=layout, settings=SETTINGS, phi=phi)[0]
            )
            with torch.no_grad():
                phi[position][index] -= sign * step
        numeric = (values[0] - values[1]) / (2.0 * step)
        assert float(gradients[position][index]) == pytest.approx(numeric, rel=1e-4, abs=1e-8)


def test_the_fit_recovers_the_parameters_that_made_the_targets(dp_synthetic_cache: FeatureCache):
    _, batch = _truth_batch(dp_synthetic_cache, seed=31)
    model = _model(batch=batch, hidden=(), shapes=2, seed=32)
    with torch.no_grad():
        start = float(vp_objective(model=model, batch=batch, settings=SETTINGS))
    result = fit_pooled_vp(model=model, batch=batch, settings=SETTINGS, max_evaluations=120)
    assert result.objective < start - 1e3
    assert result.history[-1] <= result.history[0]
    assert result.kkt_violation < 1e-6
    # The truth has heating thresholds of 14 degrees and slopes of 0.5 in every band.
    assert model.threshold.detach().abs().max() < 30.0
    slope = torch.nn.functional.softplus(model.slope_raw.detach())
    assert float(slope.mean()) == pytest.approx(0.5, rel=0.3)
    # The fit's own objective, evaluated afresh, agrees with the reported value.
    with torch.no_grad():
        final = float(vp_objective(model=model, batch=batch, settings=SETTINGS))
    assert final == pytest.approx(result.objective, rel=1e-9)


def test_a_fit_needs_float64(dp_synthetic_cache: FeatureCache):
    cache = _cache(dp_synthetic_cache, mva=False)
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
    cache = _cache(dp_synthetic_cache, mva=True)
    batch = _batch(cache, sites=("S1", "S3"))
    model = _model(batch=batch)
    with torch.no_grad():
        model.tan_phi.fill_(0.2)
    with pytest.raises(ValueError, match="tan"):
        vp_objective(model=model, batch=batch, settings=SETTINGS)


def test_a_training_row_after_the_cut_is_refused(dp_synthetic_cache: FeatureCache):
    cache = _cache(dp_synthetic_cache, mva=False)
    batch = _batch(cache, sites=("S1", "S2"))
    assert_batch_before_cut(batch=batch, end_times=cache.end_times, cut_end=CUT)
    earlier = np.datetime64("2024-01-01T00:00:00", "us")
    with pytest.raises(ValueError, match="lookahead"):
        assert_batch_before_cut(batch=batch, end_times=cache.end_times, cut_end=earlier)


def test_forecast_rows_are_centred_on_training_means_and_never_read_their_targets(
    dp_synthetic_cache: FeatureCache,
):
    cache, batch = (
        _cache(dp_synthetic_cache, mva=False),
        None,
    )
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
    scrambled = replace(
        forecast,
        y=forecast.y
        + 50.0 * torch.randn(forecast.y.shape, dtype=torch.float64, generator=generator),
    )
    model.eval()
    with torch.no_grad():
        first, second = model(forecast), model(scrambled)
    assert torch.equal(first, second)
    # The stored centring means are the training rows' own, so they do not move with the forecast.
    means = model.centre_means.clone()
    with torch.no_grad():
        model(scrambled)
    assert torch.equal(model.centre_means, means)


def _active_set(*, model, batch, layout) -> tuple[bool, ...]:  # noqa: ANN001
    theta, _ = solve_theta(model=model, batch=batch, layout=layout, settings=SETTINGS)
    return tuple(bool(v) for v in ((theta == 0.0) & layout.constrained).tolist())


def test_the_envelope_gradient_is_continuous_across_an_active_set_change(
    dp_synthetic_cache: FeatureCache,
):
    """Where a slope or an increment leaves zero, the gradient still matches to first order."""
    cache = _cache(dp_synthetic_cache, mva=False)
    batch = _batch(cache, sites=("S1", "S2"))
    generator = torch.Generator().manual_seed(51)
    flat = 40.0 + 2.0 * torch.randn(batch.n_rows, dtype=torch.float64, generator=generator)
    batch = replace(batch, y=flat)
    model = _model(batch=batch, shapes=2, seed=52)
    layout = make_vp_layout(model=model, batch=batch)
    phi = phi_parameters(model=model)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in phi:
        parameter.requires_grad = True
    threshold = model.threshold

    def active_at(value: float) -> tuple[bool, ...]:
        with torch.no_grad():
            threshold[0] = value
        return _active_set(model=model, batch=batch, layout=layout)

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
    values = []
    for sign in (1.0, -1.0):
        with torch.no_grad():
            threshold[0] = kink + sign * step
        values.append(
            _evaluate(model=model, batch=batch, layout=layout, settings=SETTINGS, phi=phi)[0]
        )
    assert active_at(kink + step) != active_at(kink - step)
    with torch.no_grad():
        threshold[0] = kink
    _, gradients, _ = _evaluate(model=model, batch=batch, layout=layout, settings=SETTINGS, phi=phi)
    numeric = (values[0] - values[1]) / (2.0 * step)
    assert float(gradients[1][0]) == pytest.approx(numeric, rel=1e-3, abs=1e-7)


def test_a_failed_convergence_runs_the_adam_fallback_and_reports_both_objectives(
    dp_synthetic_cache: FeatureCache,
):
    _, batch = _truth_batch(dp_synthetic_cache, seed=61)
    model = _model(batch=batch, hidden=(), shapes=2, seed=62)
    result = fit_pooled_vp(
        model=model, batch=batch, settings=SETTINGS, max_evaluations=4, adam_steps=5
    )
    assert not result.converged
    assert result.adam_objective is not None
    assert result.objective <= result.lbfgs_objective + 1e-6
    assert result.evaluations <= 4 + 25


def test_the_adam_fit_on_the_vp_objective_lowers_it(dp_synthetic_cache: FeatureCache):
    _, batch = _truth_batch(dp_synthetic_cache, seed=71)
    model = PooledDemandModel(
        config=DpConfig(shared_shapes=2, population_scales=PopulationScales(deviation=1.0)),
        n_sites=2,
        n_season_years=batch.n_season_years,
        initial_level=torch.full((2,), 50.0),
    ).to(dtype=torch.float64)
    model.mark_trained_years(batch=batch)
    model.start_from_batch(batch=batch)
    with torch.no_grad():
        start = float(vp_objective(model=_frozen(model), batch=batch, settings=SETTINGS))
    result = fit_pooled_adam_on_vp_objective(
        model=model, batch=batch, settings=SETTINGS, steps=30, learning_rate=0.05
    )
    assert result.losses[-1] < result.losses[0]
    with torch.no_grad():
        end = float(vp_objective(model=model, batch=batch, settings=SETTINGS))
    assert end < start
    assert result.losses[0] == pytest.approx(start / batch.n_rows, rel=1e-6)


def _frozen(model: PooledDemandModel) -> PooledDemandModel:
    freeze_operator_for_vp(model=model)
    return model
