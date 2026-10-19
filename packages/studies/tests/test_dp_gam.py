"""Tests of the convex model of the differentiable-physics series (`studies.dp_gam`).

The fixtures in `conftest.py` build a feature cache from synthetic weather with the real builder;
the tests replace the power with a truth made from the model's own design, so a fit that recovers
the truth shows the design, the penalties, the constraints, and the solver agree.
"""

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from functools import partial
from typing import Final
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")

from scipy.optimize import lsq_linear  # noqa: E402
from studies.dp_features import (  # noqa: E402
    FeatureCache,
    default_input_paths,
    load_or_build_feature_cache,
)
from studies.dp_gam import (  # noqa: E402
    GamSpec,
    ResidualSpread,
    TrainingWindow,
    accumulate_gram,
    assert_before_holdout,
    assert_no_lookahead,
    design_matrix,
    fit_gam,
    month_block_weights,
    penalty_matrix,
    predict_gam,
    prepare_fit,
    replica_cut,
    replica_residuals,
    solve_bounded_quadratic,
    training_rows,
)
from studies.dp_pinned_capacity import PinnedCapacity, pinned_capacity  # noqa: E402

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
    if capacity.stop > capacity.start:
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


def _layout_for(
    *, cache: FeatureCache, spec: GamSpec, window: TrainingWindow | None = None
) -> tuple[dp_gam.GamLayout, np.ndarray]:
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


def test_the_default_tolerance_is_tight_in_float64_and_loose_in_float32():
    assert dp_gam.default_tolerance(dtype=torch.float64) == 1e-7
    assert 1e-4 < dp_gam.default_tolerance(dtype=torch.float32) < 1e-2


def test_the_default_tolerance_loosens_in_float32_and_a_wellconditioned_problem_still_solves():
    generator = np.random.default_rng(15)
    factor = generator.normal(size=(60, 12))
    target = generator.normal(size=60)
    constrained = torch.as_tensor(np.arange(12) % 2 == 0)
    gram = torch.as_tensor(factor.T @ factor + np.eye(12), dtype=torch.float32)[None]
    rhs = torch.as_tensor(factor.T @ target, dtype=torch.float32)[None]
    solved, violation = solve_bounded_quadratic(matrix=gram, rhs=rhs, constrained=constrained)
    reference, _ = solve_bounded_quadratic(
        matrix=gram.double(), rhs=rhs.double(), constrained=constrained
    )
    assert violation.max() < 1e-3
    assert np.allclose(solved[0].numpy(), reference[0].numpy(), atol=1e-4)


def test_a_batch_of_large_systems_solves_on_the_cpu_with_several_threads():
    # PyTorch's batched MKL LU solve fails from about 200 by 200 on four or more threads; the solver
    # solves a CPU batch one system at a time. The failure needs an explicit thread count.
    generator = np.random.default_rng(16)
    size = 567
    factor = generator.normal(size=(3, size + 50, size))
    gram = torch.as_tensor(np.einsum("bnp,bnq->bpq", factor, factor) + np.eye(size))
    rhs = torch.as_tensor(generator.normal(size=(3, size)))
    threads = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        solved, violation = solve_bounded_quadratic(
            matrix=gram, rhs=rhs, constrained=torch.zeros(size, dtype=torch.bool)
        )
    finally:
        torch.set_num_threads(threads)
    assert violation.max() < 1e-7
    assert torch.allclose(torch.einsum("bpq,bq->bp", gram, solved), rhs, atol=1e-6)


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


def test_the_replica_fits_twelve_months_before_the_cut_and_predicts_only_the_year_after(
    dp_synthetic_cache: FeatureCache, monkeypatch: pytest.MonkeyPatch
):
    recorded: list[datetime] = []
    original = dp_gam.fit_gam

    def spy(**kwargs):  # noqa: ANN003, ANN202
        recorded.append(kwargs["cut"])
        return original(**kwargs)

    monkeypatch.setattr(dp_gam, "fit_gam", spy)
    replica = replica_residuals(
        cache=dp_synthetic_cache,
        site="S1",
        spec=GamSpec(),
        cut=CUT,
        window=TrainingWindow(months=12),
        season_years=SEASON_YEARS,
    )
    assert recorded == [datetime(2023, 7, 1, tzinfo=UTC)]
    assert replica.fit_cut == replica_cut(cut=CUT)
    ends = dp_synthetic_cache.end_times[replica.index]
    # No residual row is in the replica fit's training window, and none is after the final cut.
    assert ends.min() > np.datetime64("2023-07-01T00:00:00", "us")
    assert ends.max() <= np.datetime64("2024-07-01T00:00:00", "us")
    assert len(replica.residuals) == len(replica.band) == len(replica.quarter) == len(ends)
    assert set(np.unique(replica.quarter)) == {0, 1, 2, 3}


def test_the_replica_residuals_carry_the_staleness_of_a_frozen_forecast(
    dp_synthetic_cache: FeatureCache,
):
    # A level that rises by 0.02 points a day after the replica fit's cut is invisible to a frozen
    # fit, so its residuals must grow through the replica year, as the scored forecast's do.
    features = dp_synthetic_cache.sites["S1"]
    days = (dp_synthetic_cache.end_times - np.datetime64("2023-07-01", "us")) / np.timedelta64(
        1, "D"
    )
    ramp = 0.02 * np.clip(days, 0.0, None)
    truth = replace(features, y=40.0 + ramp)
    cache = replace(dp_synthetic_cache, sites={"S1": truth})
    replica = replica_residuals(
        cache=cache,
        site="S1",
        spec=GamSpec(),
        cut=CUT,
        window=TrainingWindow(months=12),
        season_years=SEASON_YEARS,
    )
    first = replica.residuals[replica.quarter == 2]
    last = replica.residuals[replica.quarter == 1]
    assert last.mean() - first.mean() > 3.0


def test_the_replica_cut_clamps_the_day_of_the_month():
    assert replica_cut(cut=datetime(2025, 3, 31, tzinfo=UTC)) == datetime(2024, 3, 31, tzinfo=UTC)
    assert dp_gam._months_before(moment=datetime(2025, 5, 31, tzinfo=UTC), months=3) == datetime(
        2025, 2, 28, tzinfo=UTC
    )


def test_a_cut_in_another_time_zone_is_converted_and_a_naive_cut_is_refused():
    summer = datetime(2024, 7, 1, 1, 0, tzinfo=ZoneInfo("Europe/London"))
    assert dp_gam.naive_utc(moment=summer) == np.datetime64("2024-07-01T00:00:00", "us")
    with pytest.raises(ValueError, match="time zone"):
        dp_gam.naive_utc(moment=datetime(2024, 7, 1))


def test_drift_is_held_at_the_last_knot_after_the_cut():
    knots = np.array([0.0, 30.0, 60.0])
    inside = dp_gam._hat_functions(days=np.array([0.0, 15.0, 60.0]), knot_days=knots)
    assert inside.tolist() == [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]
    held = dp_gam._hat_functions(days=np.array([75.0, 400.0]), knot_days=knots)
    assert held.tolist() == [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]


def test_a_later_season_year_takes_the_last_trained_capacity_even_if_its_increment_is_nonzero(
    dp_synthetic_cache: FeatureCache,
):
    spec = GamSpec(drift="none", ridge=0.01)
    layout, _index = _layout_for(cache=dp_synthetic_cache, spec=spec)
    truth = _truth_theta(layout=layout, generator=np.random.default_rng(13))
    cache = _with_truth(
        cache=dp_synthetic_cache, theta=truth, layout=layout, noise=NOISE_POINTS, seed=14
    )
    fit = fit_gam(
        cache=cache,
        site="S1",
        specs=[spec],
        cut=CUT,
        window=TrainingWindow(),
        season_years=SEASON_YEARS,
    )
    # The 2024-07-01 cut ends the 2023 season-year. The two half-hours that end at the cut start
    # in British Summer Time on 1 July, and are dark, so they must not make 2024 look trained.
    assert fit.last_season_year == 2023
    assert fit.capacity_points()[2023] > 0.0
    tampered = replace(fit, theta=fit.theta.copy())
    capacity = fit.layout.slices["capacity"]
    tampered.theta[0, capacity.start + fit.layout.season_years.index(2024)] += 7.0
    tampered.theta[0, capacity.start + fit.layout.season_years.index(2025)] += 3.0
    summer_2024 = np.flatnonzero(
        (cache.end_times > np.datetime64("2024-07-01", "us"))
        & (cache.end_times <= np.datetime64("2024-07-08", "us"))
    )
    summer_2023 = np.flatnonzero(
        (cache.end_times > np.datetime64("2023-07-01", "us"))
        & (cache.end_times <= np.datetime64("2023-07-08", "us"))
    )
    after = predict_gam(cache=cache, fit=tampered, index=summer_2024)
    assert np.array_equal(after, predict_gam(cache=cache, fit=fit, index=summer_2024))
    # A trained season-year does use its own increments, so the tampering is not invisible to it.
    tampered.theta[0, capacity.start + fit.layout.season_years.index(2023)] += 2.0
    assert not np.allclose(
        predict_gam(cache=cache, fit=tampered, index=summer_2023),
        predict_gam(cache=cache, fit=fit, index=summer_2023),
    )


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
    spread = ResidualSpread.fit(residuals=residuals, band=band, quarter=quarter, centre=False)
    fresh_band = generator.integers(0, 4, size=n)
    fresh = generator.laplace(scale=1.0 + fresh_band)
    quantiles = spread.quantiles(
        mean=np.zeros(n), band=fresh_band, quarter=generator.integers(0, 4, size=n)
    )
    assert (np.diff(quantiles, axis=1) >= 0.0).all()
    for position, level in enumerate(dp_gam.QUANTILE_LEVELS):
        assert (fresh <= quantiles[:, position]).mean() == pytest.approx(level, abs=0.01)


def test_a_small_group_falls_back_to_its_band_and_a_small_band_to_all_residuals():
    generator = np.random.default_rng(12)
    # Band 0 is wide, band 1 is narrow, and band 2 has too few residuals to stand alone.
    band = np.concatenate([np.zeros(2000, int), np.ones(2000, int), np.full(10, 2)])
    quarter = np.concatenate(
        [np.where(np.arange(2000) < 5, 3, 0), np.zeros(2000, int), np.zeros(10, int)]
    )
    residuals = np.concatenate(
        [
            5.0 * generator.normal(size=2000),
            0.5 * generator.normal(size=2000),
            generator.normal(size=10),
        ]
    )
    spread = ResidualSpread.fit(residuals=residuals, band=band, quarter=quarter, centre=False)
    levels = dp_gam.QUANTILE_LEVELS
    assert np.allclose(spread.table[0, 3], np.quantile(residuals[:2000], levels))
    assert not np.allclose(spread.table[0, 3], np.quantile(residuals, levels))
    assert np.allclose(spread.table[2, 0], np.quantile(residuals, levels))


def test_a_centred_spread_has_median_zero_in_every_group_and_keeps_the_shape():
    generator = np.random.default_rng(13)
    n = 40000
    band = generator.integers(0, 4, size=n)
    quarter = generator.integers(0, 4, size=n)
    biased = generator.laplace(scale=1.0 + band) + 5.0
    raw = ResidualSpread.fit(residuals=biased, band=band, quarter=quarter, centre=False)
    centred = ResidualSpread.fit(residuals=biased, band=band, quarter=quarter, centre=True)
    median = dp_gam.QUANTILE_LEVELS.index(0.5)
    assert np.allclose(centred.table[:, :, median], 0.0)
    assert np.allclose(raw.table[:, :, median], 5.0, atol=0.5)
    assert np.allclose(centred.table, raw.table - raw.table[:, :, [median]])
    mean = np.full(n, 10.0)
    quantiles = centred.quantiles(mean=mean, band=band, quarter=quarter)
    assert np.allclose(quantiles[:, median], 10.0)


def test_a_fallback_group_of_a_centred_spread_is_centred_too():
    generator = np.random.default_rng(14)
    band = np.concatenate([np.zeros(2000, int), np.full(10, 1)])
    quarter = np.zeros(2010, int)
    residuals = np.concatenate([generator.normal(size=2000) + 3.0, generator.normal(size=10) + 3.0])
    spread = ResidualSpread.fit(residuals=residuals, band=band, quarter=quarter, centre=True)
    median = dp_gam.QUANTILE_LEVELS.index(0.5)
    assert spread.table[1, 0, median] == 0.0


def test_the_spread_refuses_to_fit_on_nothing():
    with pytest.raises(ValueError, match="no residuals"):
        ResidualSpread.fit(
            residuals=np.array([]), band=np.array([]), quarter=np.array([]), centre=True
        )


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


@pytest.mark.skipif(
    not default_input_paths().rows.exists(), reason="experiment 103's private row sets are absent"
)
def test_the_gram_solve_matches_numpy_on_one_real_substation_with_drift(tmp_path):  # noqa: ANN001
    # CI has no private data; where it exists this is the plan's test 1 on real tensors.
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=tmp_path)
    spec = GamSpec(drift="d1")
    layout, index = prepare_fit(
        cache=cache,
        site="D10",
        spec=spec,
        cut=datetime(2025, 7, 1, tzinfo=UTC),
        window=TrainingWindow(),
        season_years=SEASON_YEARS,
    )
    features = cache.sites["D10"]
    x = design_matrix(cache=cache, features=features, layout=layout, index=index)
    y = features.y[index]
    penalty, _ = penalty_matrix(layout=layout, n_rows=len(index))
    reference = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    gram, rhs = accumulate_gram(cache=cache, features=features, layout=layout, index=index)
    solved, _ = solve_bounded_quadratic(
        matrix=gram + torch.as_tensor(penalty)[None],
        rhs=rhs,
        constrained=torch.zeros(layout.n_columns, dtype=torch.bool),
    )
    assert np.abs(x @ solved[0].numpy() - x @ reference).max() < 1e-4
    well_determined = np.abs(reference) > 1e-3
    assert np.allclose(solved[0].numpy()[well_determined], reference[well_determined], rtol=1e-6)


def test_a_fit_with_no_solar_signal_in_its_training_rows_names_the_problem(
    dp_synthetic_cache: FeatureCache,
):
    features = dp_synthetic_cache.sites["S1"]
    dark = replace(features, pv_unit_shape=np.zeros_like(features.pv_unit_shape))
    cache = replace(dp_synthetic_cache, sites={"S1": dark})
    with pytest.raises(ValueError, match="solar signal"):
        fit_gam(
            cache=cache,
            site="S1",
            specs=[GamSpec()],
            cut=CUT,
            window=TrainingWindow(months=12),
            season_years=SEASON_YEARS,
        )


def _pinned_capacity(*, cache: FeatureCache, cut: datetime) -> PinnedCapacity:
    """Pin S1's capacity to a weekly vintage that rises from 1 MW at 0.01 MW a week."""
    weeks = []
    week = date(2019, 9, 2)
    while week <= cut.date():
        weeks.append(week)
        week += timedelta(days=7)
    vintages = pl.DataFrame(
        {
            "time_series_id": cache.sites["S1"].table.time_series_id,
            "variant": "frozen",
            "origin_monday": cut.date(),
            "week_start": weeks,
            "capacity_mw": [1.0 + 0.01 * i for i in range(len(weeks))],
            "unobserved": False,
        }
    )
    return pinned_capacity(vintages=vintages, table=cache.sites["S1"].table, cut=cut)


def _pinned_truth_cache(
    *, cache: FeatureCache, spec: GamSpec, pinned: PinnedCapacity
) -> tuple[FeatureCache, dp_gam.GamLayout, np.ndarray]:
    """Return the cache with power made from a pinned-capacity truth, and the truth's layout."""
    layout, _ = _layout_for(cache=cache, spec=spec)
    theta = _truth_theta(layout=layout, generator=np.random.default_rng(11))
    features = cache.sites["S1"]
    usable = np.flatnonzero(
        np.isfinite(features.pv_unit_shape)
        & np.isfinite(features.smoothed_temperature_c).all(axis=1)
    )
    y = np.full(len(cache.end_times), np.nan)
    for start in range(0, len(usable), 20000):
        rows = usable[start : start + 20000]
        y[rows] = design_matrix(cache=cache, features=features, layout=layout, index=rows) @ theta
        y[rows] -= dp_gam.pinned_solar_points(
            cache=cache, features=features, pinned=pinned, index=rows
        )
    y = y + NOISE_POINTS * np.random.default_rng(12).normal(size=len(y))
    return replace(cache, sites={"S1": replace(features, y=y)}), layout, theta


def test_a_pinned_capacity_fit_has_no_capacity_columns_and_recovers_the_truth(
    dp_synthetic_cache: FeatureCache,
):
    spec = GamSpec(capacity="pinned", ridge=0.01)
    pinned = _pinned_capacity(cache=dp_synthetic_cache, cut=CUT)
    cache, layout, truth = _pinned_truth_cache(cache=dp_synthetic_cache, spec=spec, pinned=pinned)
    assert layout.slices["capacity"].stop == layout.slices["capacity"].start
    fit = fit_gam(
        cache=cache,
        site="S1",
        specs=[spec],
        cut=CUT,
        window=TrainingWindow(),
        season_years=SEASON_YEARS,
        pinned=pinned,
    )
    assert fit.capacity_points() == {}
    heating = layout.slices["heating"]
    assert np.allclose(
        fit.theta[0][heating][N_BANDS : 2 * N_BANDS],
        truth[heating][N_BANDS : 2 * N_BANDS],
        rtol=0.10,
    )
    index = np.flatnonzero(
        (cache.end_times > np.datetime64("2024-07-01T00:00", "us"))
        & (cache.end_times <= np.datetime64("2024-09-01T00:00", "us"))
        & np.isfinite(cache.sites["S1"].y)
    )
    predicted = predict_gam(cache=cache, fit=fit, index=index)
    assert np.abs(predicted - cache.sites["S1"].y[index]).mean() < 4 * NOISE_POINTS


def test_a_pinned_spec_needs_a_pinned_capacity_and_a_fitted_spec_refuses_one(
    dp_synthetic_cache: FeatureCache,
):
    pinned = _pinned_capacity(cache=dp_synthetic_cache, cut=CUT)
    fit = partial(
        fit_gam,
        cache=dp_synthetic_cache,
        site="S1",
        cut=CUT,
        window=TrainingWindow(months=12),
        season_years=SEASON_YEARS,
    )
    with pytest.raises(ValueError, match="pinned capacity is required"):
        fit(specs=[GamSpec(capacity="pinned")])
    with pytest.raises(ValueError, match="pinned capacity is required"):
        fit(specs=[GamSpec()], pinned=pinned)
    earlier = _pinned_capacity(cache=dp_synthetic_cache, cut=datetime(2024, 6, 3, tzinfo=UTC))
    with pytest.raises(ValueError, match="was built for"):
        fit(specs=[GamSpec(capacity="pinned")], pinned=earlier)


def test_the_covid_flag_keeps_the_lockdown_rows_and_recovers_an_injected_shift(
    dp_synthetic_cache: FeatureCache,
):
    spec = GamSpec(covid_flag=True)
    in_lockdown = (dp_synthetic_cache.end_times > np.datetime64("2021-01-01T00:00", "us")) & (
        dp_synthetic_cache.end_times <= np.datetime64("2021-07-19T00:00", "us")
    )
    plain = dp_synthetic_cache.sites["S1"]
    features = replace(plain, lockdown=in_lockdown, default_rows=plain.default_rows & ~in_lockdown)
    dp_synthetic_cache = replace(dp_synthetic_cache, sites={"S1": features})
    plain_rows = training_rows(
        features=features, cache=dp_synthetic_cache, cut=CUT, window=TrainingWindow()
    )
    flagged_rows = training_rows(
        features=features,
        cache=dp_synthetic_cache,
        cut=CUT,
        window=TrainingWindow(),
        keep_lockdown=True,
    )
    assert not features.lockdown[plain_rows].any()
    assert features.lockdown[flagged_rows].any()
    layout, _ = _layout_for(cache=dp_synthetic_cache, spec=spec)
    truth = _truth_theta(layout=layout, generator=np.random.default_rng(13))
    truth[layout.slices["covid"]] = (-6.0, -3.0, -2.0, -1.0)
    cache = _with_truth(
        cache=dp_synthetic_cache, theta=truth, layout=layout, noise=NOISE_POINTS, seed=14
    )
    fit = fit_gam(
        cache=cache,
        site="S1",
        specs=[spec],
        cut=CUT,
        window=TrainingWindow(),
        season_years=SEASON_YEARS,
    )
    assert np.allclose(fit.theta[0][layout.slices["covid"]], (-6.0, -3.0, -2.0, -1.0), atol=0.5)


def test_a_pinned_final_knot_is_tied_to_the_one_before_it(dp_synthetic_cache: FeatureCache):
    free = GamSpec(drift="d1", final_knot="free")
    tied = GamSpec(drift="d1", final_knot="pinned")
    layout, _ = _layout_for(cache=dp_synthetic_cache, spec=tied, window=TrainingWindow(months=12))
    truth = _truth_theta(layout=layout, generator=np.random.default_rng(15))
    n_knots = len(layout.knot_days)
    drift = layout.slices["drift"]
    for band in range(N_BANDS):
        truth[drift.start + band * n_knots + n_knots - 1] += 4.0
    cache = _with_truth(
        cache=dp_synthetic_cache, theta=truth, layout=layout, noise=NOISE_POINTS, seed=16
    )
    fits = {
        name: fit_gam(
            cache=cache,
            site="S1",
            specs=[spec],
            cut=CUT,
            window=TrainingWindow(months=12),
            season_years=SEASON_YEARS,
        )
        for name, spec in {"free": free, "pinned": tied}.items()
    }
    for name, expected_tie in (("free", False), ("pinned", True)):
        knots = fits[name].theta[0][drift].reshape(N_BANDS, n_knots)
        last_step = np.abs(knots[:, -1] - knots[:, -2])
        assert (last_step.max() < 0.01) == expected_tie


def test_a_shrunk_final_knot_moves_the_forecast_towards_the_recent_mean_and_zero_shrink_is_free(
    dp_synthetic_cache: FeatureCache,
):
    window = TrainingWindow(months=24)
    layout, _ = _layout_for(cache=dp_synthetic_cache, spec=GamSpec(drift="d1"), window=window)
    truth = _truth_theta(layout=layout, generator=np.random.default_rng(17))
    cache = _with_truth(
        cache=dp_synthetic_cache, theta=truth, layout=layout, noise=NOISE_POINTS, seed=18
    )
    fits = {
        shrink: fit_gam(
            cache=cache,
            site="S1",
            specs=[GamSpec(drift="d1", final_knot="shrunk", final_knot_shrink=shrink)],
            cut=CUT,
            window=window,
            season_years=SEASON_YEARS,
        )
        for shrink in (0.0, 1.0)
    }
    free = fit_gam(
        cache=cache,
        site="S1",
        specs=[GamSpec(drift="d1")],
        cut=CUT,
        window=window,
        season_years=SEASON_YEARS,
    )
    index = np.flatnonzero(
        (cache.end_times > np.datetime64("2024-07-02T00:00", "us"))
        & (cache.end_times <= np.datetime64("2024-07-20T00:00", "us"))
    )
    free_prediction = predict_gam(cache=cache, fit=free, index=index)
    assert np.allclose(predict_gam(cache=cache, fit=fits[0.0], index=index), free_prediction)
    full = predict_gam(cache=cache, fit=fits[1.0], index=index)
    assert np.abs(full - free_prediction).max() > 0.1
    assert np.allclose(free.theta[0], fits[1.0].theta[0])


def test_the_replica_of_a_pinned_arm_needs_the_vintages(dp_synthetic_cache: FeatureCache):
    with pytest.raises(ValueError, match="needs the capacity vintages"):
        replica_residuals(
            cache=dp_synthetic_cache,
            site="S1",
            spec=GamSpec(capacity="pinned"),
            cut=CUT,
            window=TrainingWindow(months=12),
            season_years=SEASON_YEARS,
        )
