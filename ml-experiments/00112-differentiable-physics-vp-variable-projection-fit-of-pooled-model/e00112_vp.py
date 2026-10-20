"""Fit and store the variable-projection (VP) arms and their gradient-descent comparators.

A fit is a saved state dictionary plus a record, keyed by an arm name and a cut. The forecast of a
year needs the arm's fit at the cut and its replica, the fit 12 months earlier, whose residuals give
the spread. The replicate year's cut (2024-07-01) is therefore also the scored year's replica, and
the 2023-07-01 cut is the replicate year's replica.

**Noise rounds.** The noise standard deviation is fixed within a fit. Round 1 is the root mean
square of the per-site convex model's training residuals at the same cut (a 2-year window, no
drift, never the stored scored-year predictions). Round 2 is the root mean square of the round-1
VP-P fit's training residuals; every arm of a cut uses round 2, and the round-1 VP-P fit is the
starting point of the round-2 VP-P fit.
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Final

import e00112_common as common
import numpy as np
import torch
from studies.dp_features import FeatureCache
from studies.dp_gam import GamSpec, TrainingWindow, fit_gam, predict_gam, training_rows
from studies.dp_onboarding import predict_pooled
from studies.dp_pooled import DpConfig, PooledBatch, PooledDemandModel, PopulationScales
from studies.dp_pooled_vp import (
    VpFitResult,
    VpSettings,
    fit_pooled_adam_on_vp_objective,
    fit_pooled_vp,
    freeze_operator_for_vp,
    load_theta,
    make_vp_problem,
    solve_theta,
    vp_objective,
)

RECORDS_PATH: Final[Path] = common.WORK_DIR / "fit_records.json"
DEVICE: Final[str] = "cuda"


def model_config() -> DpConfig:
    """Return the pooled configuration of every VP arm: P-map's MAP prior, K shapes, the network."""
    tuned = common.tuned()
    return DpConfig(
        shared_shapes=int(tuned["k"]),
        interaction_hidden=common.INTERACTION,
        population_scales=PopulationScales(deviation=tuned["map_deviation"]),
    )


def new_model(*, batch: PooledBatch, start: str) -> PooledDemandModel:
    """Return a fresh pooled model for `batch`, on the batch's device in float64.

    Args:
        batch: The training rows, which give the site and season-year layout.
        start: `s0` (seed 0, the default thresholds) or `s1` (seed 1, thresholds at 17 degrees).

    Returns:
        The model, with the MVA operator frozen at zero reactive power.

    Raises:
        ValueError: If `start` is not one of `common.STARTS`.
    """
    if start not in common.STARTS:
        msg = f"unknown start {start!r}"
        raise ValueError(msg)
    torch.manual_seed(0 if start == "s0" else 1)
    model = PooledDemandModel(
        config=model_config(),
        n_sites=len(batch.site_labels),
        n_season_years=batch.n_season_years,
        initial_level=torch.full((len(batch.site_labels),), 50.0),
    ).to(device=batch.y.device, dtype=batch.y.dtype)
    if start == "s1":
        with torch.no_grad():
            model.threshold.fill_(common.START_THRESHOLD_C)
            model.population_mean["threshold"].fill_(common.START_THRESHOLD_C)
    freeze_operator_for_vp(model=model)
    return model


def model_path(*, name: str, cut: datetime) -> Path:
    """Return where the state dictionary of an arm's fit at a cut is saved."""
    return common.MODELS_DIR / f"{name}__{cut.date().isoformat()}.pt"


def save_fit(*, name: str, cut: datetime, model: PooledDemandModel, record: dict) -> None:
    """Save a fitted model's state dictionary, and add its record to the shared record file."""
    common.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path(name=name, cut=cut))
    records = json.loads(RECORDS_PATH.read_text()) if RECORDS_PATH.exists() else {}
    records[f"{name}__{cut.date().isoformat()}"] = record
    common.write_json(name="fit_records.json", payload=records)


def read_record(*, name: str, cut: datetime) -> dict:
    """Return the saved record of an arm's fit at a cut."""
    return json.loads(RECORDS_PATH.read_text())[f"{name}__{cut.date().isoformat()}"]


def has_fit(*, name: str, cut: datetime) -> bool:
    """Return whether an arm's fit at a cut is saved."""
    return model_path(name=name, cut=cut).exists()


def load_fit(
    *, name: str, cut: datetime, batch: PooledBatch, start: str = "s0"
) -> PooledDemandModel:
    """Load a saved fit into a fresh model built for `batch`."""
    model = new_model(batch=batch, start=start)
    model.load_state_dict(torch.load(model_path(name=name, cut=cut), map_location=batch.y.device))
    return model


def record_of(*, result: VpFitResult, rows: int, extra: dict | None = None) -> dict:
    """Return the JSON-ready record of a VP fit."""
    record = asdict(result)
    record["history"] = record["history"][::10]
    record["rows"] = rows
    record["peak_gpu_gb"] = result.peak_memory_bytes / 1e9
    return record | (extra or {})


def training_residual_sd(*, model: PooledDemandModel, batch: PooledBatch) -> float:
    """Return the root mean square of the model's training residuals, in points."""
    prediction = predict_pooled(model=model, batch=batch)
    actual = batch.y.detach().cpu().numpy()
    return float(np.sqrt(np.mean((actual - prediction) ** 2)))


def convex_training_sd(*, cache: FeatureCache, sites: tuple[str, ...], cut: datetime) -> float:
    """Return round 1: the root mean square of the per-site convex model's training residuals.

    The model is Part 2's convex model with its tuned penalties, no drift, on a 2-year window
    ending at `cut`; each MW substation is fitted alone and the residuals are pooled.
    """
    tuned = json.loads((common.PART_2_DIR / "work" / "tuned.json").read_text())
    spec = GamSpec(
        drift="none",
        ridge=tuned["ridge"]["chosen"],
        drift_step=tuned["drift_step"]["chosen"],
        step_penalty=tuned["step_penalty"]["chosen"],
    )
    window = TrainingWindow(months=common.WINDOW_MONTHS)
    squares: list[np.ndarray] = []
    for site in sites:
        fit = fit_gam(
            cache=cache,
            site=site,
            specs=[spec],
            cut=cut,
            window=window,
            season_years=common.SEASON_YEARS,
        )
        index = training_rows(features=cache.sites[site], cache=cache, cut=cut, window=window)
        squares.append(
            (cache.sites[site].y[index] - predict_gam(cache=cache, fit=fit, index=index)) ** 2
        )
    return float(np.sqrt(np.mean(np.concatenate(squares))))


def fit_pooled_arm(
    *,
    batch: PooledBatch,
    settings: VpSettings,
    start: str,
    fallback_steps: int,
    init_state: dict | None = None,
) -> tuple[PooledDemandModel, VpFitResult]:
    """Fit one pooled VP model on `batch`.

    Args:
        batch: The training rows (float64, on the GPU).
        settings: The noise standard deviation and the prior scales.
        start: Which start (`common.STARTS`).
        fallback_steps: Adam steps of the fallback when L-BFGS misses its convergence test.
        init_state: A saved state dictionary to continue from (the round-2 fit and the leave-one-out
            fits); the Adam warm start is skipped then.

    Returns:
        The fitted model and its fit report.
    """
    model = new_model(batch=batch, start=start)
    if init_state is not None:
        model.load_state_dict(init_state)
    result = fit_pooled_vp(
        model=model,
        batch=batch,
        settings=settings,
        adam_steps=fallback_steps,
        warm_start_steps=0 if init_state is not None else common.WARM_START_STEPS,
    )
    return model, result


def fit_adam_same(*, batch: PooledBatch, settings: VpSettings) -> tuple[PooledDemandModel, dict]:
    """Fit P-map-same: experiment 105's Adam fit (steps, schedule) on the VP objective.

    Returns:
        The model and a record holding its VP objective in nats.
    """
    model = new_model(batch=batch, start="s0")
    result = fit_pooled_adam_on_vp_objective(
        model=model,
        batch=batch,
        settings=settings,
        steps=common.FIT_STEPS,
        learning_rate=common.FIT_RATE,
        final_rate_fraction=common.FINAL_RATE_FRACTION,
    )
    with torch.no_grad():
        objective = float(vp_objective(model=model, batch=batch, settings=settings))
    return model, {
        "objective": objective,
        "seconds": result.seconds,
        "peak_gpu_gb": result.peak_memory_bytes / 1e9,
        "rows": batch.n_rows,
    }


def polish(*, model: PooledDemandModel, batch: PooledBatch, settings: VpSettings) -> float:
    """Solve the linear coefficients exactly at the model's non-linear parameters; return nats.

    This is P-map-polish: P-map-same's non-linear parameters are frozen and only the exact solve
    changes the model. The model is changed in place.
    """
    problem = make_vp_problem(model=model, batch=batch, settings=settings)
    theta, _ = solve_theta(problem=problem)
    load_theta(model=model, layout=problem.layout, theta=theta)
    model.finalise_centres(batch=batch)
    with torch.no_grad():
        return float(vp_objective(model=model, batch=batch, settings=settings))
