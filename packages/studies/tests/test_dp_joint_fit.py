"""Tests of the joint fit of the convex model and its Kalman correction (`studies.dp_joint_fit`).

The central claim is that the filter's day-ahead correction is linear in the residual stream, so
the loss after the correction is an exact quadratic in the coefficients. The first tests check that
claim against `studies.dp_assimilation.run_kalman`, the filter that the joint fit reproduces, on
random streams with missing readings. The later tests run the whole fit on the synthetic cache.
"""

from dataclasses import replace
from datetime import UTC, datetime
from typing import Final

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from studies.dp_assimilation import (  # noqa: E402
    FilterHyperparameters,
    correction_at,
    run_kalman,
)
from studies.dp_features import FeatureCache  # noqa: E402
from studies.dp_gam import (  # noqa: E402
    GamSpec,
    TrainingWindow,
    design_matrix,
    fit_gam,
    penalty_matrix,
    prepare_fit,
)
from studies.dp_joint_fit import (  # noqa: E402
    BURN_IN_DAYS,
    corrected_normal_equations,
    day_ahead_initialisations,
    fit_joint_gam,
    loss_cases,
    stream_design,
    stream_rows,
)

N_BANDS: Final[int] = 4
N_ROWS: Final[int] = 700
CUT: Final[datetime] = datetime(2024, 7, 1, tzinfo=UTC)
SEASON_YEARS: Final[range] = range(2019, 2026)
HYPER: Final[np.ndarray] = np.array([-1.0, 2.0, 0.5, 0.3])
NO_FILTER: Final[np.ndarray] = np.array([-30.0, 2.0, -20.0, 12.0])
"""Settings under which the gain, and so the correction, is about 1e-9 of the residual."""


def test_each_valid_time_has_one_initialisation_with_a_lead_of_12_to_36_hours():
    ends = np.arange(
        np.datetime64("2025-03-01T00:30", "us"),
        np.datetime64("2025-03-06T00:30", "us"),
        np.timedelta64(30, "m"),
    )
    initialisation = day_ahead_initialisations(end_times=ends)
    lead = (ends - initialisation) / np.timedelta64(1, "h")
    assert lead.min() > 12.0
    assert lead.max() <= 36.0
    hours = (initialisation - initialisation.astype("datetime64[D]").astype("datetime64[us]")) / (
        np.timedelta64(1, "h")
    )
    assert (hours == 9.0).all()


def _problem(*, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n_columns = 5
    band = (np.arange(N_ROWS) // 12) % N_BANDS
    present = rng.random(N_ROWS) > 0.2
    rows = np.flatnonzero(present)
    columns = rng.normal(size=(len(rows), n_columns))
    target = rng.normal(size=len(rows)).cumsum() * 0.1 + columns @ rng.normal(size=n_columns)
    case_target = rows[rows > 200][::3]
    case_cutoff = case_target - rng.integers(20, 90, size=len(case_target))
    return {
        "columns": columns,
        "target": target,
        "rows": rows,
        "band": band,
        "case_target": case_target,
        "case_cutoff": case_cutoff,
    }


def test_the_corrected_loss_equals_the_loss_of_the_filter_run_on_the_residual_stream():
    problem = _problem(seed=1)
    normal = corrected_normal_equations(**problem, hyper=HYPER, device="cpu")
    rng = np.random.default_rng(2)
    for _ in range(3):
        theta = rng.normal(size=problem["columns"].shape[1])
        stream = np.full(N_ROWS, np.nan)
        stream[problem["rows"]] = problem["target"] - problem["columns"] @ theta
        trace = run_kalman(
            residual=stream,
            band=problem["band"],
            hyper=FilterHyperparameters.from_array(values=HYPER),
        )
        correction = correction_at(
            trace=trace,
            cutoff=problem["case_cutoff"],
            target=problem["case_target"],
            band=problem["band"][problem["case_target"]],
        ).total
        error = stream[problem["case_target"]] - correction
        quadratic = (
            theta @ normal.gram.numpy() @ theta
            - 2.0 * normal.rhs.numpy() @ theta
            + normal.target_sum_squares
        )
        assert quadratic == pytest.approx(float(error @ error), rel=1e-9)


def test_a_case_whose_target_is_not_a_stream_row_is_refused():
    problem = _problem(seed=3)
    missing = np.setdiff1d(np.arange(problem["rows"][0], problem["rows"][-1]), problem["rows"])
    problem["case_target"] = missing[:3]
    problem["case_cutoff"] = missing[:3] - 30
    with pytest.raises(ValueError, match="stream row"):
        corrected_normal_equations(**problem, hyper=HYPER, device="cpu")


def _truth_cache(*, cache: FeatureCache) -> tuple[FeatureCache, GamSpec]:
    spec = GamSpec(drift="none")
    layout, _ = prepare_fit(
        cache=cache,
        site="S1",
        spec=spec,
        cut=CUT,
        window=TrainingWindow(months=12),
        season_years=SEASON_YEARS,
    )
    features = cache.sites["S1"]
    usable = np.flatnonzero(
        np.isfinite(features.pv_unit_shape)
        & np.isfinite(features.smoothed_temperature_c).all(axis=1)
    )
    rng = np.random.default_rng(4)
    theta = np.abs(rng.normal(size=layout.n_columns)) * 0.1
    y = np.full(len(cache.end_times), np.nan)
    for start in range(0, len(usable), 20000):
        rows = usable[start : start + 20000]
        y[rows] = design_matrix(cache=cache, features=features, layout=layout, index=rows) @ theta
    y = y + 0.5 * rng.normal(size=len(y))
    return replace(cache, sites={"S1": replace(features, y=y)}), spec


def _fit_arguments(*, cache: FeatureCache, spec: GamSpec) -> dict:
    return {
        "cache": cache,
        "site": "S1",
        "spec": spec,
        "cut": CUT,
        "window": TrainingWindow(months=12),
        "season_years": SEASON_YEARS,
    }


def test_a_filter_that_corrects_nothing_gives_the_two_stage_fit(dp_synthetic_cache: FeatureCache):
    cache, spec = _truth_cache(cache=dp_synthetic_cache)
    arguments = _fit_arguments(cache=cache, spec=spec)
    _layout, index = prepare_fit(
        cache=cache,
        site="S1",
        spec=spec,
        cut=CUT,
        window=TrainingWindow(months=12),
        season_years=SEASON_YEARS,
    )
    rows = stream_rows(
        cache=cache, site="S1", cut=CUT, start=cache.end_times[index[0]] - np.timedelta64(30, "m")
    )
    targets, _ = loss_cases(cache=cache, index=index, rows=rows)
    weights = np.isin(index, targets).astype(np.float64)[None]
    two_stage = fit_gam(
        cache=cache,
        site="S1",
        specs=[spec],
        cut=CUT,
        window=TrainingWindow(months=12),
        season_years=SEASON_YEARS,
        weights=weights,
    )
    [joint] = fit_joint_gam(**arguments, hyper=NO_FILTER, device="cpu")
    fitted = np.abs(two_stage.theta[0]) > 1e-3
    assert np.abs(joint.theta[0] - two_stage.theta[0])[fitted].max() < 1e-6


def test_a_heavy_ridge_to_an_anchor_returns_the_anchor(dp_synthetic_cache: FeatureCache):
    cache, spec = _truth_cache(cache=dp_synthetic_cache)
    arguments = _fit_arguments(cache=cache, spec=spec)
    [anchor] = fit_joint_gam(**arguments, hyper=NO_FILTER, device="cpu")
    [pulled, free] = fit_joint_gam(
        **arguments, hyper=HYPER, anchor=anchor, ridges=(1e6, 0.0), device="cpu"
    )
    assert np.abs(pulled.theta - anchor.theta).max() < np.abs(free.theta - anchor.theta).max()
    assert np.abs(pulled.theta - anchor.theta).max() < 1e-2


def test_a_ridge_without_an_anchor_is_refused(dp_synthetic_cache: FeatureCache):
    cache, spec = _truth_cache(cache=dp_synthetic_cache)
    with pytest.raises(ValueError, match="anchor"):
        fit_joint_gam(**_fit_arguments(cache=cache, spec=spec), hyper=HYPER, ridges=(1.0,))


def test_the_gram_matches_a_column_by_column_dense_reference_with_gaps_and_a_binding_floor():
    problem = _problem(seed=5)
    # Long gaps (a stretch with no stream row) and a tiny observation noise make the covariance
    # collapse to the variance floor, the case where the gains are least like a textbook filter.
    keep = ~((problem["rows"] > 350) & (problem["rows"] < 450))
    for name in ("columns", "target", "rows"):
        problem[name] = problem[name][keep]
    inside = np.isin(problem["case_target"], problem["rows"])
    problem["case_target"] = problem["case_target"][inside]
    problem["case_cutoff"] = problem["case_cutoff"][inside]
    hyper = np.array([-6.0, 1.0, -3.0, -8.0])
    normal = corrected_normal_equations(**problem, hyper=hyper, device="cpu")
    settings = FilterHyperparameters.from_array(values=hyper)
    band_of_case = problem["band"][problem["case_target"]]
    position = {int(row): index for index, row in enumerate(problem["rows"])}
    case_rows = np.array([position[int(t)] for t in problem["case_target"]])

    def corrected(values: np.ndarray) -> np.ndarray:
        stream = np.full(N_ROWS, np.nan)
        stream[problem["rows"]] = values
        correction = correction_at(
            trace=run_kalman(residual=stream, band=problem["band"], hyper=settings),
            cutoff=problem["case_cutoff"],
            target=problem["case_target"],
            band=band_of_case,
        ).total
        return values[case_rows] - correction

    z = np.column_stack([corrected(problem["columns"][:, j]) for j in range(5)])
    zy = corrected(problem["target"])
    assert np.allclose(normal.gram.numpy(), z.T @ z, rtol=1e-8, atol=1e-8)
    assert np.allclose(normal.rhs.numpy(), z.T @ zy, rtol=1e-8, atol=1e-8)


def test_windows_of_consecutive_days_tile_the_time_axis():
    ends = np.arange(
        np.datetime64("2025-03-01T00:30", "us"),
        np.datetime64("2025-03-11T00:30", "us"),
        np.timedelta64(30, "m"),
    )
    initialisation = day_ahead_initialisations(end_times=ends)
    assert (np.diff(initialisation.astype(np.int64)) >= 0).all()
    _, counts = np.unique(initialisation, return_counts=True)
    assert (counts[1:-1] == 48).all()


def test_a_non_finite_design_value_is_refused():
    problem = _problem(seed=6)
    problem["columns"][3, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        corrected_normal_equations(**problem, hyper=HYPER, device="cpu")


def test_streams_that_start_late_and_per_band_settings_match_the_dense_reference():
    problem = _problem(seed=7)
    keep = problem["rows"] > 60
    for name in ("columns", "target", "rows"):
        problem[name] = problem[name][keep]
    inside = np.isin(problem["case_target"], problem["rows"])
    problem["case_target"] = problem["case_target"][inside]
    problem["case_cutoff"] = problem["case_cutoff"][inside]
    per_band = np.stack([HYPER + 0.2 * band for band in range(N_BANDS)])
    normal = corrected_normal_equations(**problem, hyper=per_band, device="cpu")
    assert normal.n_cases == len(problem["case_target"])
    stream = np.full(N_ROWS, np.nan)
    theta = np.random.default_rng(8).normal(size=problem["columns"].shape[1])
    stream[problem["rows"]] = problem["target"] - problem["columns"] @ theta
    band_of_case = problem["band"][problem["case_target"]]
    expected = np.zeros(len(band_of_case))
    for band in range(N_BANDS):
        trace = run_kalman(
            residual=stream,
            band=problem["band"],
            hyper=FilterHyperparameters.from_array(values=per_band[band]),
        )
        total = correction_at(
            trace=trace,
            cutoff=problem["case_cutoff"],
            target=problem["case_target"],
            band=band_of_case,
        ).total
        expected = np.where(band_of_case == band, total, expected)
    error = stream[problem["case_target"]] - expected
    quadratic = (
        theta @ normal.gram.numpy() @ theta
        - 2.0 * normal.rhs.numpy() @ theta
        + normal.target_sum_squares
    )
    assert quadratic == pytest.approx(float(error @ error), rel=1e-9)


def test_loss_cases_respect_the_cut_the_delay_and_the_burn_in(dp_synthetic_cache: FeatureCache):
    cache, spec = _truth_cache(cache=dp_synthetic_cache)
    _layout, index = prepare_fit(
        cache=cache,
        site="S1",
        spec=spec,
        cut=CUT,
        window=TrainingWindow(months=12),
        season_years=SEASON_YEARS,
    )
    rows = stream_rows(
        cache=cache, site="S1", cut=CUT, start=cache.end_times[index[0]] - np.timedelta64(30, "m")
    )
    targets, cutoffs = loss_cases(cache=cache, index=index, rows=rows)
    ends = cache.end_times
    initialisation = day_ahead_initialisations(end_times=ends[targets])
    assert np.isin(targets, rows).all()
    assert (cutoffs < targets).all()
    assert (cutoffs >= rows[0]).all()
    assert (ends[cutoffs] <= initialisation - np.timedelta64(3 * 3600 + 1800, "s")).all()
    assert (ends[targets] > ends[rows[0]] + np.timedelta64(BURN_IN_DAYS, "D")).all()
    assert (ends[targets] <= np.datetime64(CUT.replace(tzinfo=None), "us")).all()


def test_an_orthogonal_penalty_and_a_negative_ridge_are_refused(dp_synthetic_cache: FeatureCache):
    cache, spec = _truth_cache(cache=dp_synthetic_cache)
    arguments = _fit_arguments(cache=cache, spec=spec)
    with pytest.raises(ValueError, match="orthogonal"):
        fit_joint_gam(
            **_fit_arguments(cache=cache, spec=replace(spec, orthogonal_penalty=1.0)), hyper=HYPER
        )
    with pytest.raises(ValueError, match="negative"):
        fit_joint_gam(**arguments, hyper=HYPER, ridges=(-1.0,))


def test_the_joint_fit_has_no_larger_true_loss_than_the_two_stage_fit(
    dp_synthetic_cache: FeatureCache,
):
    cache, spec = _truth_cache(cache=dp_synthetic_cache)
    arguments = _fit_arguments(cache=cache, spec=spec)
    [joint] = fit_joint_gam(**arguments, hyper=HYPER, device="cpu")
    two_stage = fit_gam(
        cache=cache,
        site="S1",
        specs=[spec],
        cut=CUT,
        window=TrainingWindow(months=12),
        season_years=SEASON_YEARS,
    )
    _layout, index = prepare_fit(
        cache=cache,
        site="S1",
        spec=spec,
        cut=CUT,
        window=TrainingWindow(months=12),
        season_years=SEASON_YEARS,
    )
    rows = stream_rows(
        cache=cache, site="S1", cut=CUT, start=cache.end_times[index[0]] - np.timedelta64(30, "m")
    )
    columns, target = stream_design(
        cache=cache, site="S1", layout=joint.layout, rows=rows, pinned=None
    )
    targets, cutoffs = loss_cases(cache=cache, index=index, rows=rows)
    matrix, cost = penalty_matrix(layout=joint.layout, n_rows=len(index))

    def penalised_loss(theta: np.ndarray) -> float:
        residual = np.full(len(cache.end_times), np.nan)
        residual[rows] = target - columns @ theta
        trace = run_kalman(
            residual=residual,
            band=cache.calendar.band,
            hyper=FilterHyperparameters.from_array(values=HYPER),
        )
        correction = correction_at(
            trace=trace, cutoff=cutoffs, target=targets, band=cache.calendar.band[targets]
        ).total
        error = residual[targets] - correction
        return float(error @ error + theta @ matrix @ theta + cost @ theta)

    assert penalised_loss(joint.theta[0]) < penalised_loss(two_stage.theta[0])
    assert penalised_loss(joint.theta[0]) < penalised_loss(joint.theta[0] * 1.01)


def test_more_ridge_keeps_the_fit_closer_to_the_anchor(dp_synthetic_cache: FeatureCache):
    cache, spec = _truth_cache(cache=dp_synthetic_cache)
    arguments = _fit_arguments(cache=cache, spec=spec)
    [anchor] = fit_joint_gam(**arguments, hyper=NO_FILTER, device="cpu")
    fits = fit_joint_gam(
        **arguments, hyper=HYPER, anchor=anchor, ridges=(0.0, 0.3, 3.0, 30.0), device="cpu"
    )
    distance = [float(np.linalg.norm(fit.theta - anchor.theta)) for fit in fits]
    assert distance == sorted(distance, reverse=True)
    assert distance[0] > 10.0 * distance[-1] > 0.0
