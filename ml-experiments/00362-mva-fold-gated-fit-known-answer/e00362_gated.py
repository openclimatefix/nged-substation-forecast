r"""Experiment 362: the known-answer test of experiment 358 with a gate on reverse flow.

Each job folds one Primary's signed power to apparent power, fits the signed series (S), the
reading as if signed (N), the reading through the meter (L0, ungated, to replicate experiment 358),
and the same fit with the gate (G), then scores the following year on past weather. The gate
rejects a start of the multi-start fit when more than `UNGATED_SHARE` of the training half-hours
have negative fitted power while the independent solar output is under `GATE_RATIO` times the
fitted demand. Independent solar is the pinned capacity times the solar shape, plus (lab only) the
injected farm's output.

    PYTHONPATH=ml-experiments/00110-*:ml-experiments/00358-* \
        uv run --no-sync python ml-experiments/00362-*/e00362_gated.py --workers 5
"""

import argparse
import json
import multiprocessing
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

import e00110_arms as arms  # ty: ignore[unresolved-import]
import e00110_common as common  # ty: ignore[unresolved-import]
import e00110_inject as inject  # ty: ignore[unresolved-import]
import e00358_known_answer as known  # ty: ignore[unresolved-import]
import numpy as np
from studies.dp_features import FeatureCache
from studies.dp_gam import (
    LAST_TRAINED_MIN_SHAPE,
    GamFit,
    derated_unit_shape,
    fit_gam,
    pinned_solar_points,
    predict_gam,
    prepare_fit,
)
from studies.dp_meter import (
    MeterFit,
    StartSummary,
    _Problem,
    fit_gam_through_meter,
    predict_meter,
)
from studies.dp_operator import tan_phi_from_power_factor
from studies.dp_pinned_capacity import PinnedCapacity, pinned_capacity, read_vintages
from studies.sources import ML_EXPERIMENTS_DATA_DIR

GATE_RATIO: Final[float] = 0.75
POST_HOC_RATIOS: Final[tuple[float, ...]] = (1.0, 1.25)
"""Post hoc gate ratios, run beside the registered 0.75 and labelled as such."""
UNGATED_SHARE: Final[float] = 0.005
DATA_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00362-mva-fold-gated-fit-known-answer" / "work" / "gated"
)


def gated_meter_fit(
    *,
    cache: FeatureCache,
    site: str,
    spec: Any,
    cut: Any,
    window: Any,
    solar: np.ndarray,
    ratio: float = GATE_RATIO,
) -> tuple[MeterFit, dict[str, Any]]:
    """Fit through the meter with zero reactive power, rejecting starts that break the gate.

    Args:
        cache: The cache whose reading is the folded `|S|`.
        site: The substation's label.
        spec: The model spec, with fitted capacity.
        cut: The cut; training rows end no later.
        window: The training-window rule.
        solar: The independent solar output on the cache's grid, in points.
        ratio: The gate closes where solar output is under this multiple of the fitted demand.

    Returns:
        The chosen fit and a record of the gate (starts rejected, whether all were).
    """
    features = cache.sites[site]
    layout, index = prepare_fit(
        cache=cache, site=site, spec=spec, cut=cut, window=window, season_years=common.SEASON_YEARS
    )
    signal = derated_unit_shape(features=features, index=index) > LAST_TRAINED_MIN_SHAPE
    weights = np.ones(len(index))
    problem = _Problem.build(
        cache=cache, features=features, layout=layout, index=index, weights=weights
    )
    starts = problem.starts(features=features, index=index, layout=layout)
    runs = [problem.minimise(theta=theta, mode="zero") for _, theta in starts]
    weight = problem.prior_weight(state=min(runs, key=lambda r: r.squares).state)
    losses = np.array([problem.loss(state=r.state, weight=weight, learned=False) for r in runs])
    shares = []
    for run in runs:
        power = problem.design @ run.theta
        demand = problem.demand_design @ run.theta
        closed = solar[index] < ratio * demand
        shares.append(float(((power < 0.0) & closed).mean()))
    shares_array = np.array(shares)
    allowed = shares_array <= UNGATED_SHARE
    if allowed.any():
        best = int(np.argmin(np.where(allowed, losses, np.inf)))
    else:
        best = int(np.argmin(shares_array))
    chosen = runs[best]
    summaries = tuple(
        StartSummary(
            name=name,
            loss=float(loss),
            final_capacity=float(np.cumsum(run.theta[layout.slices["capacity"]])[-1]),
            tan_phi=run.tan_phi,
            iterations=run.iterations,
            converged=run.converged,
        )
        for (name, _), run, loss in zip(starts, runs, losses, strict=True)
    )
    fit = GamFit(
        layout=layout,
        theta=chosen.theta[None, :],
        cut=cut,
        n_rows=len(index),
        site=site,
        last_season_year=int(cache.calendar.season_year[index[signal]].max()),
        violation=np.zeros(1),
        pinned=None,
    )
    meter = MeterFit(
        fit=fit,
        qmode="zero",
        reactive="zero",
        tan_phi=chosen.tan_phi,
        q0=chosen.q0,
        loss=float(losses[best]),
        reverse_flow_share=float((problem.design @ chosen.theta < 0.0).mean()),
        starts=summaries,
        demand_columns=problem.demand_mask,
    )
    record = {
        "n_starts_rejected": int((~allowed).sum()),
        "all_rejected": bool(not allowed.any()),
        "ungated_shares": [float(x) for x in shares_array],
        "chosen_start": starts[best][0],
    }
    return meter, record


def run_job(job: tuple[str, str, float, str]) -> Path:  # noqa: PLR0915
    """Run one (site, cut, level, reactive) job and return its result file."""
    site, cut_name, level, reactive = job
    path = DATA_DIR / f"{site}_{cut_name}_L{level}_{reactive}.json"
    if path.exists():
        return path
    import torch  # ty: ignore[unresolved-import]

    torch.set_num_threads(2)
    cut = known.CUTS[cut_name]
    cache = common.load_cache()
    part_4 = arms.read_part_4_tuned()
    spec_b = arms.base_specs(part_4=part_4)["b"]
    spec = replace(spec_b, capacity="fitted")
    window = arms.arm_definitions(part_4=part_4, tuned=arms.read_tuned())["lq"].window
    features = cache.sites[site]
    pinned: PinnedCapacity = pinned_capacity(
        vintages=read_vintages(), table=features.table, cut=cut
    )
    fit_b = fit_gam(
        cache=cache,
        site=site,
        specs=[spec_b],
        cut=cut,
        window=window,
        season_years=common.SEASON_YEARS,
        pinned=pinned,
    )
    rows = np.flatnonzero(np.isfinite(features.y) & np.isfinite(features.pv_unit_shape))
    rooftop = np.full(len(cache.end_times), np.nan)
    rooftop[rows] = pinned_solar_points(cache=cache, features=features, pinned=pinned, index=rows)
    demand = np.full(len(cache.end_times), np.nan)
    demand[rows] = predict_gam(cache=cache, fit=fit_b, index=rows) + rooftop[rows]
    scale = inject.midday_median_absolute(cache=cache, site=site, before=cut)
    signed_cache = (
        cache
        if level == 0.0
        else inject.inject_farm(
            cache=cache, site=site, farm=known.FARM, capacity_points=level * scale
        )
    )
    signed = signed_cache.sites[site].y
    injected_output = np.nan_to_num(features.y - signed, nan=0.0)
    solar = np.nan_to_num(rooftop, nan=0.0) + injected_output
    tan_phi = (
        0.0 if reactive == "Q0" else tan_phi_from_power_factor(power_factor=known.POWER_FACTOR)
    )
    folded_cache = inject.inject_meter(
        cache=signed_cache, site=site, signed=signed, demand=demand, tan_phi=tan_phi
    )
    folded = folded_cache.sites[site].y

    def capacity(fit: Any) -> float:
        return float(fit.capacity_points()[fit.last_season_year])

    kwargs: dict[str, Any] = {
        "site": site,
        "cut": cut,
        "window": window,
        "season_years": common.SEASON_YEARS,
    }
    fit_s = fit_gam(cache=signed_cache, specs=[spec], **kwargs)
    fit_n = fit_gam(cache=folded_cache, specs=[spec], **kwargs)
    ungated = fit_gam_through_meter(cache=folded_cache, spec=spec, qmode="zero", **kwargs)
    gated_fits = {
        ratio: gated_meter_fit(
            cache=folded_cache,
            site=site,
            spec=spec,
            cut=cut,
            window=window,
            solar=solar,
            ratio=ratio,
        )
        for ratio in (GATE_RATIO, *POST_HOC_RATIOS)
    }
    gated, record = gated_fits[GATE_RATIO]
    index = common.scoring_index(
        cache=folded_cache, site=site, after=cut, through=common.year_end(cut=cut)
    )
    keep = np.isfinite(signed[index]) & np.isfinite(demand[index]) & ~features.switching_mask[index]
    index = index[keep]
    truth = signed[index]
    reading = folded[index]
    pred_s = predict_gam(cache=signed_cache, fit=fit_s, index=index)
    pred_n = predict_gam(cache=folded_cache, fit=fit_n, index=index)
    pred_u = predict_meter(cache=folded_cache, meter=ungated, index=index)
    pred_g = predict_meter(cache=folded_cache, meter=gated, index=index)
    closed = solar[index] < GATE_RATIO * pred_g.demand
    power_g = np.where(closed, np.maximum(pred_g.real_power, 0.0), pred_g.real_power)
    sunny = known._sunny(cache=cache, site=site, index=index)
    clear = np.abs(truth) >= known.MIN_TRUE_POINTS
    result: dict[str, Any] = {
        "site": site,
        "cut": cut_name,
        "level": level,
        "reactive": reactive,
        "n_rows": len(index),
        "n_sunny": int((sunny & clear).sum()),
        "reverse_flow_share_sunny": float((truth[sunny] < 0).mean()) if sunny.any() else None,
        "capacity_s": capacity(fit_s),
        "capacity_l0": capacity(ungated.fit),
        "capacity_g": capacity(gated.fit),
        "pinned_held_points": float(pinned.held_points),
        **record,
    }
    mask = sunny & clear
    positive = truth[mask] > 0
    post_hoc = []
    for ratio in POST_HOC_RATIOS:
        meter_r, record_r = gated_fits[ratio]
        pred_r = predict_meter(cache=folded_cache, meter=meter_r, index=index)
        closed_r = solar[index] < ratio * pred_r.demand
        label = f"g{ratio:g}".replace(".", "_")
        post_hoc.append(
            (label, np.where(closed_r, np.maximum(pred_r.real_power, 0.0), pred_r.real_power))
        )
        result[f"capacity_{label}"] = capacity(meter_r.fit)
        result[f"n_starts_rejected_{label}"] = record_r["n_starts_rejected"]
        result[f"all_rejected_{label}"] = record_r["all_rejected"]
        result[f"chosen_start_{label}"] = record_r["chosen_start"]
    for arm, predicted in (
        ("s", pred_s),
        ("n", pred_n),
        ("l0", pred_u.real_power),
        ("g", power_g),
        ("g_unclipped", pred_g.real_power),
        *post_hoc,
    ):
        result[f"acc_{arm}_sunny"] = float(((predicted[mask] > 0) == positive).mean())
        result[f"acc_{arm}_all"] = float(((predicted[clear] > 0) == (truth[clear] > 0)).mean())
    for arm, predicted in (
        ("l0", pred_u.apparent_power),
        ("n", pred_n),
        ("g", pred_g.apparent_power),
    ):
        result[f"mae_reading_{arm}_all"] = float(np.abs(predicted - reading).mean())
        result[f"mae_reading_{arm}_sunny"] = float(np.abs(predicted[sunny] - reading[sunny]).mean())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, default=float) + "\n")
    temporary.replace(path)
    return path


def main() -> None:
    """Run every job not yet done, in a spawned pool."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--sites", nargs="*", default=list(known.SITES))
    parser.add_argument("--cuts", nargs="*", default=list(known.CUTS))
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "2"
    common.load_cache()
    jobs = [
        (site, cut, level, reactive)
        for cut in arguments.cuts
        for level in known.LEVELS
        for reactive in known.REACTIVE
        for site in arguments.sites
    ]
    context = multiprocessing.get_context("spawn")
    with context.Pool(arguments.workers) as pool:
        for finished in pool.imap_unordered(run_job, jobs):
            print("done", finished.name, flush=True)


if __name__ == "__main__":
    main()
