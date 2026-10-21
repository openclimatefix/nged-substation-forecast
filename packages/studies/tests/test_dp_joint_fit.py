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
    prepare_fit,
)
from studies.dp_joint_fit import (  # noqa: E402
    corrected_normal_equations,
    day_ahead_initialisations,
    fit_joint_gam,
    loss_cases,
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
    joint = fit_joint_gam(**arguments, hyper=NO_FILTER, device="cpu")
    fitted = np.abs(two_stage.theta[0]) > 1e-3
    assert np.abs(joint.theta[0] - two_stage.theta[0])[fitted].max() < 1e-6


def test_a_heavy_ridge_to_an_anchor_returns_the_anchor(dp_synthetic_cache: FeatureCache):
    cache, spec = _truth_cache(cache=dp_synthetic_cache)
    arguments = _fit_arguments(cache=cache, spec=spec)
    anchor = fit_joint_gam(**arguments, hyper=NO_FILTER, device="cpu")
    pulled = fit_joint_gam(**arguments, hyper=HYPER, anchor=anchor, ridge=1e6, device="cpu")
    free = fit_joint_gam(**arguments, hyper=HYPER, device="cpu")
    assert np.abs(pulled.theta - anchor.theta).max() < np.abs(free.theta - anchor.theta).max()
    assert np.abs(pulled.theta - anchor.theta).max() < 1e-2


def test_a_ridge_without_an_anchor_is_refused(dp_synthetic_cache: FeatureCache):
    cache, spec = _truth_cache(cache=dp_synthetic_cache)
    with pytest.raises(ValueError, match="anchor"):
        fit_joint_gam(**_fit_arguments(cache=cache, spec=spec), hyper=HYPER, ridge=1.0)
