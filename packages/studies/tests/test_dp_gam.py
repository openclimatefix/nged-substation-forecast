"""Tests of the convex model of the differentiable-physics series (`studies.dp_gam`).

The fixtures in `conftest.py` build a feature cache from synthetic weather with the real builder;
the tests replace the power with a truth made from the model's own design, so a fit that recovers
the truth shows the design, the penalties, the constraints, and the solver agree.
"""

import itertools
from dataclasses import replace
from datetime import UTC, datetime
from typing import Final

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scipy.optimize import lsq_linear  # noqa: E402
from studies.dp_features import (  # noqa: E402
    FeatureCache,
    input_key,
    load_feature_cache,
    save_feature_cache,
)
from studies.dp_gam import (  # noqa: E402
    GamSpec,
    ResidualSpread,
    TrainingWindow,
    accumulate_gram,
    assert_before_holdout,
    assert_no_lookahead,
    crossfit_cuts,
    crossfit_residuals,
    design_matrix,
    fit_gam,
    month_block_weights,
    penalty_matrix,
    predict_gam,
    prepare_fit,
    solve_bounded_quadratic,
    training_rows,
)

from studies import dp_gam  # noqa: E402

CUT: Final[datetime] = datetime(2024, 7, 1, tzinfo=UTC)
SEASON_YEARS: Final[range] = range(2019, 2026)
NOISE_POINTS: Final[float] = 0.15
N_BANDS: Final[int] = 4


def _truth_theta(*, layout: dp_gam.GamLayout, generator: np.random.Generator) -> np.ndarray:
    """Return coefficients of a plausible substation, in the layout's columns."""
    theta = np.zeros(layout.n_columns)
    profile = layout.slices["profile"]
    half_hours = np.arange(48)
    shape = 40.0 + 6.0 * np.sin(2 * np.pi * half_hours / 48 - 1.0)
    for day_type, offset in enumerate((0.0, -4.0, -6.0, -6.0)):
        theta[profile.start + day_type * 48 : profile.start + (day_type + 1) * 48] = shape + offset
    holiday = layout.slices["holiday"]
    theta[holiday] = 0.5 * generator.normal(size=holiday.stop - holiday.start)
    heating = layout.slices["heating"]
    for band, slope in enumerate((0.8, 1.5, 1.0, 1.8)):
        theta[heating.start + 1 * N_BANDS + band] = slope
    lighting = layout.slices["lighting"]
    theta[lighting] = (0.5, 0.3, 0.2, 0.1)
    capacity = layout.slices["capacity"]
    for year, increment in {2020: 8.0, 2022: 10.0, 2023: 5.0}.items():
        theta[capacity.start + layout.season_years.index(year)] = increment
    if "drift" in layout.slices:
        n_knots = len(layout.knot_days)
        drift = layout.slices["drift"]
        for band in range(N_BANDS):
            path = 1.5 * np.sin(2 * np.pi * np.arange(n_knots) / 24.0 + band)
            theta[drift.start + band * n_knots : drift.start + (band + 1) * n_knots] = path
    return theta


def _with_truth(
    *, cache: FeatureCache, theta: np.ndarray, layout: dp_gam.GamLayout, noise: float, seed: int
) -> FeatureCache:
    """Replace the substation's power with `design @ theta` plus noise on every usable row."""
    features = cache.sites["S1"]
    usable = np.flatnonzero(
        np.isfinite(features.pv_unit_shape)
        & np.isfinite(features.smoothed_temperature_c).all(axis=1)
    )
    y = np.full(len(cache.end_times), np.nan)
    for start in range(0, len(usable), 20000):
        rows = usable[start : start + 20000]
        y[rows] = design_matrix(cache=cache, features=features, layout=layout, index=rows) @ theta
    generator = np.random.default_rng(seed)
    y = y + noise * generator.normal(size=len(y))
    return replace(cache, sites={"S1": replace(features, y=y)})


def _layout_for(*, cache: FeatureCache, spec: GamSpec, window: TrainingWindow | None = None):  # noqa: ANN202
    layout, index = prepare_fit(
        cache=cache,
        site="S1",
        spec=spec,
        cut=CUT,
        window=window or TrainingWindow(),
        season_years=SEASON_YEARS,
    )
    return layout, index


def test_the_gram_solve_matches_numpy_and_a_direct_lbfgs_fit_of_the_ridge_problem(
    dp_synthetic_cache: FeatureCache,
):
    spec = GamSpec(drift="d1")
    window = TrainingWindow(months=12)
    layout, index = _layout_for(cache=dp_synthetic_cache, spec=spec, window=window)
    truth = _truth_theta(layout=layout, generator=np.random.default_rng(2))
    cache = _with_truth(cache=dp_synthetic_cache, theta=truth, layout=layout, noise=1.0, seed=3)
    features = cache.sites["S1"]
    x = design_matrix(cache=cache, features=features, layout=layout, index=index)
    y = features.y[index]
    penalty, _ = penalty_matrix(layout=layout, n_rows=len(index))
    reference = np.linalg.solve(x.T @ x + penalty, x.T @ y)

    gram, rhs = accumulate_gram(cache=cache, features=features, layout=layout, index=index)
    solved, violation = solve_bounded_quadratic(
        matrix=gram + torch.as_tensor(penalty)[None],
        rhs=rhs,
        constrained=torch.zeros(layout.n_columns, dtype=torch.bool),
    )
    coefficients = solved[0].numpy()
    assert violation.max() < 1e-9
    well_determined = np.abs(reference) > 1e-3
    assert np.allclose(
        coefficients[well_determined], reference[well_determined], rtol=1e-6, atol=1e-6
    )
    assert np.abs(x @ coefficients - x @ reference).max() < 1e-4

    # A direct fit of the same ridge problem by L-BFGS, on the residuals and not the Gram matrix,
    # checks that the Gram sums and the penalty are the objective they claim to be. The columns are
    # rescaled by the square root of the Hessian's diagonal, which makes L-BFGS converge.
    x_t = torch.as_tensor(x)
    y_t = torch.as_tensor(y)
    penalty_t = torch.as_tensor(penalty)
    scale = 1.0 / torch.sqrt(torch.diag(x_t.T @ x_t + penalty_t))
    scaled = torch.zeros(layout.n_columns, dtype=torch.float64, requires_grad=True)
    optimiser = torch.optim.LBFGS(
        [scaled],
        max_iter=3000,
        tolerance_grad=1e-12,
        tolerance_change=1e-16,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        theta = scaled * scale
        loss = ((x_t @ theta - y_t) ** 2).sum() + theta @ penalty_t @ theta
        loss.backward()
        return loss

    optimiser.step(closure)
    fitted = (scaled * scale).detach().numpy()
    assert np.abs(x @ fitted - x @ reference).max() < 1e-4


def test_constraints_hold_when_the_truth_breaks_them(dp_synthetic_cache: FeatureCache):
    spec = GamSpec(drift="none")
    layout, _ = _layout_for(cache=dp_synthetic_cache, spec=spec)
    truth = _truth_theta(layout=layout, generator=np.random.default_rng(4))
    heating = layout.slices["heating"]
    truth[heating.start + 1 * N_BANDS + 0] = -1.5  # a heating slope the fit must not follow
    capacity = layout.slices["capacity"]
    truth[capacity] = 0.0
    # Capacities 20 in season-year 2022, then 12 from 2023: a decrease the model cannot represent.
    path = {2022: 20.0, 2023: 12.0, 2024: 12.0, 2025: 12.0}
    previous = 0.0
    for year in SEASON_YEARS:
        value = path.get(year, 0.0)
        truth[capacity.start + layout.season_years.index(year)] = max(value - previous, 0.0)
        previous = max(previous, value)
    cache = _with_truth(
        cache=dp_synthetic_cache, theta=truth, layout=layout, noise=NOISE_POINTS, seed=5
    )
    # The truth is not monotone, so replace the capacity columns' sum by a decreasing shape:
    # subtract the missing drop from y directly.
    features = cache.sites["S1"]
    drop = np.where(
        cache.calendar.season_year >= 2023, 8.0 * np.nan_to_num(features.pv_unit_shape), 0.0
    )
    cache = replace(cache, sites={"S1": replace(features, y=features.y + drop)})
    fit = fit_gam(
        cache=cache,
        site="S1",
        specs=[spec],
        cut=CUT,
        window=TrainingWindow(),
        season_years=SEASON_YEARS,
    )
    capacities = np.array(list(fit.capacity_points().values()))
    assert (np.diff(capacities) >= -1e-12).all()
    assert (fit.theta[0, heating] >= 0.0).all()
    assert fit.theta[0, heating.start + 1 * N_BANDS + 0] == 0.0
    assert fit.violation.max() < 1e-7


def test_a_known_answer_is_recovered_by_the_drift_and_step_arm(dp_synthetic_cache: FeatureCache):
    spec = GamSpec(drift="d1_steps", ridge=0.01, drift_step=3e3, step_penalty=30.0)
    layout, _index = _layout_for(cache=dp_synthetic_cache, spec=spec)
    truth = _truth_theta(layout=layout, generator=np.random.default_rng(6))
    n_knots = len(layout.knot_days)
    step_knot = n_knots // 2
    steps = layout.slices["steps"]
    midday = 2
    truth[steps.start + midday * (n_knots - 1) + (step_knot - 1)] = 3.0
    cache = _with_truth(
        cache=dp_synthetic_cache, theta=truth, layout=layout, noise=NOISE_POINTS, seed=7
    )
    fit = fit_gam(
        cache=cache,
        site="S1",
        specs=[spec],
        cut=CUT,
        window=TrainingWindow(),
        season_years=SEASON_YEARS,
    )
    found = fit.theta[0]
    true_capacity = np.cumsum(truth[layout.slices["capacity"]])
    fitted_capacity = np.cumsum(found[layout.slices["capacity"]])
    trained = [i for i, year in enumerate(SEASON_YEARS) if year in (2021, 2022, 2023)]
    assert np.allclose(fitted_capacity[trained], true_capacity[trained], rtol=0.10)
    heating = layout.slices["heating"]
    assert np.allclose(
        found[heating][1 * N_BANDS : 2 * N_BANDS],
        truth[heating][1 * N_BANDS : 2 * N_BANDS],
        rtol=0.10,
    )

    def paths(theta: np.ndarray) -> np.ndarray:
        drift = theta[layout.slices["drift"]].reshape(N_BANDS, n_knots)
        step_block = theta[layout.slices["steps"]]
        half = step_block.size // 2
        signed = (step_block[:half] - step_block[half:]).reshape(N_BANDS, n_knots - 1)
        level = np.concatenate([np.zeros((N_BANDS, 1)), np.cumsum(signed, axis=1)], axis=1)
        total = drift + level
        return total - total.mean(axis=1, keepdims=True)

    rms = np.sqrt(((paths(found) - paths(truth)) ** 2).mean())
    assert rms < 1.0
    found_signed = found[steps][: steps.stop - steps.start].reshape(2, N_BANDS, n_knots - 1)
    midday_step = found_signed[0, midday] - found_signed[1, midday]
    assert abs(int(np.argmax(np.abs(midday_step))) + 1 - step_knot) <= 1


def test_the_bounded_solver_matches_scipy_on_a_random_problem():
    generator = np.random.default_rng(8)
    size = 40
    factor = generator.normal(size=(120, size))
    target = generator.normal(size=120) + 0.5
    constrained = np.arange(size) % 2 == 0
    lower = np.where(constrained, 0.0, -np.inf)
    reference = lsq_linear(factor, target, bounds=(lower, np.full(size, np.inf)), tol=1e-14)
    solved, violation = solve_bounded_quadratic(
        matrix=torch.as_tensor(factor.T @ factor)[None],
        rhs=torch.as_tensor(factor.T @ target)[None],
        constrained=torch.as_tensor(constrained),
    )
    assert violation.max() < 1e-9
    assert np.allclose(solved[0].numpy(), reference.x, atol=1e-7)
    assert (solved[0].numpy()[constrained] >= 0.0).all()
    assert (solved[0].numpy()[constrained] == 0.0).any()


def test_no_training_row_ends_after_the_cut(dp_synthetic_cache: FeatureCache):
    features = dp_synthetic_cache.sites["S1"]
    index = training_rows(
        features=features, cache=dp_synthetic_cache, cut=CUT, window=TrainingWindow()
    )
    assert dp_synthetic_cache.end_times[index].max() == np.datetime64("2024-07-01T00:00:00", "us")
    assert dp_synthetic_cache.end_times[index].min() > np.datetime64("2021-01-01T00:00:00", "us")
    late = np.flatnonzero(dp_synthetic_cache.end_times > np.datetime64("2024-07-01", "us"))[:5]
    with pytest.raises(ValueError, match="lookahead"):
        assert_no_lookahead(cache=dp_synthetic_cache, index=late, cut=CUT)


def test_a_twelve_month_window_starts_twelve_months_before_the_cut(
    dp_synthetic_cache: FeatureCache,
):
    features = dp_synthetic_cache.sites["S1"]
    index = training_rows(
        features=features, cache=dp_synthetic_cache, cut=CUT, window=TrainingWindow(months=12)
    )
    first = dp_synthetic_cache.end_times[index].min()
    assert first == np.datetime64("2023-07-01T00:30:00", "us")


def test_scoring_refuses_a_row_on_or_after_the_held_out_start(dp_synthetic_cache: FeatureCache):
    shifted = replace(
        dp_synthetic_cache,
        end_times=dp_synthetic_cache.end_times + np.timedelta64(400, "D"),
    )
    with pytest.raises(ValueError, match="held out"):
        assert_before_holdout(cache=shifted, index=np.array([len(shifted.end_times) - 1]))
    assert_before_holdout(cache=dp_synthetic_cache, index=np.array([len(shifted.end_times) - 1]))


def test_the_cross_fit_predicts_each_quarter_from_a_fit_that_ends_at_its_start(
    dp_synthetic_cache: FeatureCache, monkeypatch: pytest.MonkeyPatch
):
    recorded: list[tuple[datetime, int]] = []
    original = dp_gam.fit_gam

    def spy(**kwargs):  # noqa: ANN003, ANN202
        fit = original(**kwargs)
        features = kwargs["cache"].sites[kwargs["site"]]
        index = training_rows(
            features=features, cache=kwargs["cache"], cut=kwargs["cut"], window=kwargs["window"]
        )
        recorded.append(
            (kwargs["cut"], int(kwargs["cache"].end_times[index].max().astype("int64")))
        )
        return fit

    monkeypatch.setattr(dp_gam, "fit_gam", spy)
    residuals, band, quarter = crossfit_residuals(
        cache=dp_synthetic_cache,
        site="S1",
        spec=GamSpec(),
        cut=CUT,
        window=TrainingWindow(months=12),
        season_years=SEASON_YEARS,
    )
    starts = [start for start, _ in crossfit_cuts(cut=CUT)]
    assert [cut for cut, _ in recorded] == starts
    for cut, last_end in recorded:
        assert last_end <= np.datetime64(cut.replace(tzinfo=None), "us").astype("int64")
    assert len(residuals) == len(band) == len(quarter)
    assert set(np.unique(quarter)) == {0, 1, 2, 3}


def test_the_cross_fit_quarters_tile_the_last_twelve_months():
    quarters = crossfit_cuts(cut=CUT)
    assert quarters[0][0] == datetime(2023, 7, 1, tzinfo=UTC)
    assert quarters[-1][1] == CUT
    for (_, end), (start, _) in itertools.pairwise(quarters):
        assert end == start


def test_drift_is_held_at_the_last_knot_after_the_cut():
    knots = np.array([0.0, 30.0, 60.0])
    inside = dp_gam._hat_functions(days=np.array([0.0, 15.0, 60.0]), knot_days=knots)
    assert inside.tolist() == [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]
    held = dp_gam._hat_functions(days=np.array([75.0, 400.0]), knot_days=knots)
    assert held.tolist() == [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]


def test_predictions_after_the_cut_use_the_last_trained_capacity(dp_synthetic_cache: FeatureCache):
    spec = GamSpec(drift="none")
    fit = fit_gam(
        cache=dp_synthetic_cache,
        site="S1",
        specs=[spec],
        cut=CUT,
        window=TrainingWindow(),
        season_years=SEASON_YEARS,
    )
    capacities = fit.capacity_points()
    assert capacities[2025] == capacities[2023]
    scoring = np.flatnonzero(dp_synthetic_cache.end_times > np.datetime64("2025-07-01", "us"))[:48]
    assert np.isfinite(predict_gam(cache=dp_synthetic_cache, fit=fit, index=scoring)).all()


def test_the_design_has_one_profile_column_per_row_and_a_negative_capacity_block(
    dp_synthetic_cache: FeatureCache,
):
    layout, index = _layout_for(cache=dp_synthetic_cache, spec=GamSpec(drift="d1_steps"))
    rows = index[5000:5300]
    x = design_matrix(
        cache=dp_synthetic_cache,
        features=dp_synthetic_cache.sites["S1"],
        layout=layout,
        index=rows,
    )
    assert (x[:, layout.slices["profile"]].sum(axis=1) == 1.0).all()
    assert (x[:, layout.slices["capacity"]] <= 0.0).all()
    drift = x[:, layout.slices["drift"]].reshape(len(rows), N_BANDS, -1)
    assert np.allclose(drift.sum(axis=(1, 2)), 1.0)
    block = x[:, layout.slices["steps"]]
    half = block.shape[1] // 2
    assert np.array_equal(block[:, :half], -block[:, half:])


def test_a_batch_of_penalties_gives_the_same_fits_as_separate_calls(
    dp_synthetic_cache: FeatureCache,
):
    specs = [GamSpec(ridge=1.0), GamSpec(ridge=100.0)]
    together = fit_gam(
        cache=dp_synthetic_cache,
        site="S1",
        specs=specs,
        cut=CUT,
        window=TrainingWindow(months=12),
        season_years=SEASON_YEARS,
    )
    for member, spec in enumerate(specs):
        alone = fit_gam(
            cache=dp_synthetic_cache,
            site="S1",
            specs=[spec],
            cut=CUT,
            window=TrainingWindow(months=12),
            season_years=SEASON_YEARS,
        )
        assert np.allclose(together.theta[member], alone.theta[0], atol=1e-8)
    assert not np.allclose(together.theta[0], together.theta[1])


def test_a_batch_of_month_block_weightings_gives_one_fit_each(dp_synthetic_cache: FeatureCache):
    spec = GamSpec()
    window = TrainingWindow(months=12)
    _layout, index = _layout_for(cache=dp_synthetic_cache, spec=spec, window=window)
    weights = month_block_weights(cache=dp_synthetic_cache, index=index, n_resamples=3, seed=1)
    batched = fit_gam(
        cache=dp_synthetic_cache,
        site="S1",
        specs=[spec],
        cut=CUT,
        window=window,
        season_years=SEASON_YEARS,
        weights=weights,
    )
    assert batched.theta.shape[0] == 3
    single = fit_gam(
        cache=dp_synthetic_cache,
        site="S1",
        specs=[spec],
        cut=CUT,
        window=window,
        season_years=SEASON_YEARS,
        weights=weights[1:2],
    )
    assert np.allclose(batched.theta[1], single.theta[0], atol=1e-7)


def test_month_block_weights_draw_whole_months(dp_synthetic_cache: FeatureCache):
    _, index = _layout_for(
        cache=dp_synthetic_cache, spec=GamSpec(), window=TrainingWindow(months=12)
    )
    weights = month_block_weights(cache=dp_synthetic_cache, index=index, n_resamples=4, seed=9)
    months = dp_synthetic_cache.end_times[index].astype("datetime64[M]")
    assert weights.shape == (4, len(index))
    assert (weights == weights.round()).all()
    for resample in weights:
        # Rows of one month share one weight, and the weights count 12 draws of whole months.
        for month in np.unique(months):
            assert len(np.unique(resample[months == month])) == 1
        drawn = sum(resample[months == month][0] for month in np.unique(months))
        assert drawn == len(np.unique(months))
    again = month_block_weights(cache=dp_synthetic_cache, index=index, n_resamples=4, seed=9)
    other = month_block_weights(cache=dp_synthetic_cache, index=index, n_resamples=4, seed=10)
    assert np.array_equal(weights, again)
    assert not np.array_equal(weights, other)


def test_the_spread_reproduces_the_residual_quantiles_and_never_crosses():
    generator = np.random.default_rng(11)
    n = 40000
    band = generator.integers(0, 4, size=n)
    quarter = generator.integers(0, 4, size=n)
    scale = 1.0 + band
    residuals = generator.laplace(scale=scale)
    spread = ResidualSpread.fit(residuals=residuals, band=band, quarter=quarter)
    fresh_band = generator.integers(0, 4, size=n)
    fresh = generator.laplace(scale=1.0 + fresh_band)
    quantiles = spread.quantiles(
        mean=np.zeros(n), band=fresh_band, quarter=generator.integers(0, 4, size=n)
    )
    assert (np.diff(quantiles, axis=1) >= 0.0).all()
    for position, level in enumerate(dp_gam.QUANTILE_LEVELS):
        assert (fresh <= quantiles[:, position]).mean() == pytest.approx(level, abs=0.01)


def test_a_small_group_falls_back_to_its_band():
    generator = np.random.default_rng(12)
    band = np.zeros(1000, dtype=int)
    quarter = np.where(np.arange(1000) < 5, 3, 0)
    residuals = generator.normal(size=1000)
    spread = ResidualSpread.fit(residuals=residuals, band=band, quarter=quarter)
    assert np.allclose(spread.table[0, 3], np.quantile(residuals, dp_gam.QUANTILE_LEVELS))


def test_the_spread_refuses_to_fit_on_nothing():
    with pytest.raises(ValueError, match="no cross-fitted residuals"):
        ResidualSpread.fit(residuals=np.array([]), band=np.array([]), quarter=np.array([]))


def test_cpu_and_gpu_fits_agree_within_a_hundredth_of_a_point(dp_synthetic_cache: FeatureCache):
    if not torch.cuda.is_available():
        pytest.skip("no GPU in this environment")
    spec = GamSpec(drift="d1")
    window = TrainingWindow(months=12)
    cpu = fit_gam(
        cache=dp_synthetic_cache,
        site="S1",
        specs=[spec],
        cut=CUT,
        window=window,
        season_years=SEASON_YEARS,
    )
    gpu = fit_gam(
        cache=dp_synthetic_cache,
        site="S1",
        specs=[spec],
        cut=CUT,
        window=window,
        season_years=SEASON_YEARS,
        device="cuda",
    )
    scoring = np.flatnonzero(dp_synthetic_cache.end_times > np.datetime64("2024-07-01", "us"))[
        :3000
    ]
    difference = np.abs(
        predict_gam(cache=dp_synthetic_cache, fit=cpu, index=scoring)
        - predict_gam(cache=dp_synthetic_cache, fit=gpu, index=scoring)
    )
    assert difference.max() < 0.01


def test_the_cache_key_changes_with_an_input_file(dp_synthetic_inputs, tmp_path):  # noqa: ANN001
    key = input_key(paths=dp_synthetic_inputs)
    copy = tmp_path / "sites.parquet"
    copy.write_bytes(dp_synthetic_inputs.sites.read_bytes() + b"x")
    changed = replace(dp_synthetic_inputs, sites=copy)
    assert input_key(paths=changed) != key
    assert input_key(paths=dp_synthetic_inputs) == key


def test_the_cache_survives_a_round_trip_through_a_file(dp_synthetic_cache: FeatureCache, tmp_path):  # noqa: ANN001
    path = tmp_path / "cache.npz"
    save_feature_cache(cache=dp_synthetic_cache, path=path)
    loaded = load_feature_cache(path=path)
    original = dp_synthetic_cache.sites["S1"]
    restored = loaded.sites["S1"]
    assert loaded.key == dp_synthetic_cache.key
    assert restored.table == original.table
    assert np.array_equal(restored.y, original.y, equal_nan=True)
    assert np.array_equal(
        restored.smoothed_temperature_c, original.smoothed_temperature_c, equal_nan=True
    )
    assert np.array_equal(loaded.calendar.local_date, dp_synthetic_cache.calendar.local_date)
    assert np.array_equal(loaded.end_times, dp_synthetic_cache.end_times)


def test_the_builder_converts_power_to_points_of_effective_capacity(
    dp_synthetic_cache: FeatureCache,
):
    features = dp_synthetic_cache.sites["S1"]
    reading = features.y[np.isfinite(features.y)]
    assert reading[0] == pytest.approx(100.0 * 5.0 / 10.0)
    assert dp_synthetic_cache.end_times.max() < np.datetime64("2026-07-01", "us")
