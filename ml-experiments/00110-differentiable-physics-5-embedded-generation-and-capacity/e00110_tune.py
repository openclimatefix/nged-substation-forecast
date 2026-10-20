"""Tune experiment 110's settings on 2023-07 to 2024-06, a year that is never scored.

Every fit uses data before 2023-07-01 and is scored on 2023-07-01 to 2024-06-30. The settings are
chosen in this order:

1. **The covariance penalty's weight** for arm K-i: the smallest value on {0.1, 1, 10, 100} at which
   every penalised term's absolute correlation with the unit PV shape falls under 0.05 in every
   daylight quarter-by-half-hour cell at every decision site. If none does, 100, and the residual
   correlations are reported.
2. **The node's step and level penalties** (a 3 by 3 grid) on an injection that gate test 4 never
   uses: flat generators of 6 and 15 points, each with one step and one 14-day outage, injected at
   D5 on dates that gate test 4 does not use. D9 has no usable rows at this cut, so it is not
   injected. The chosen pair is the largest step penalty, then the largest level penalty, at which
   both injections are recovered.
3. **C4's arm**: among K-base, K-i, K-ii, and K-iii, the arm with the best tuning-year score.
4. **The filters' hyperparameters** for every arm, by marginal likelihood on that arm's own
   out-of-sample residuals from the 2023-07-01 fit, under a prior shared across the arm's sites.
   Sites with no tuning-year data take the mean of the others' unconstrained values.

    uv run python ml-experiments/00110-*/e00110_tune.py [--workers 4]
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, Final

import e00110_arms as arms
import e00110_common as common
import e00110_inject as inject
import numpy as np
import polars as pl
from studies.dp_assimilation import fit_filter_hyperparameters
from studies.dp_features import FeatureCache
from studies.dp_gam import (
    LAST_TRAINED_MIN_SHAPE,
    MIN_ORTHOGONAL_CELL_ROWS,
    ORTHOGONAL_BLOCKS,
    GamSpec,
    NodeSpec,
    TrainingWindow,
    derated_unit_shape,
    design_matrix,
    fit_gam,
    prepare_fit,
)
from studies.dp_pinned_capacity import pinned_capacity, read_vintages

LAMBDA_GRID: Final[tuple[float, ...]] = (0.1, 1.0, 10.0, 100.0)
CORRELATION_TARGET: Final[float] = 0.05
STEP_PENALTY_GRID: Final[tuple[float, ...]] = (30.0, 100.0, 300.0)
LEVEL_PENALTY_GRID: Final[tuple[float, ...]] = (0.01, 0.03, 0.1)
NODE_WINDOW: Final[TrainingWindow] = TrainingWindow()
"""The node's tuning fits use all years, as the scored node arms do."""
NODE_INJECTIONS: Final[tuple[dict[str, Any], ...]] = (
    {"level": 6.0, "step": 3.0, "step_date": (2022, 2, 15), "outage": (2022, 6, 8, 14)},
    {"level": 15.0, "step": 6.0, "step_date": (2022, 9, 20), "outage": (2023, 2, 14, 14)},
)
"""The tuning injections: a level, a step, its date, and a 14-day outage's date and length.

The outage is 14 days because the node's blocks are 7 days long, and a shorter outage that straddles
two blocks would be read at half its depth.
"""
STEP_TOLERANCE_DAYS: Final[float] = 7.0
STEP_SIZE_TOLERANCE: Final[float] = 0.30
LEVEL_TOLERANCE: Final[float] = 0.20

_STATE: dict[str, Any] = {}


def _init_worker() -> None:
    """Load the cache and vintages once per worker process, with two threads."""
    import torch

    torch.set_num_threads(2)
    _STATE["cache"] = common.load_cache()
    _STATE["vintages"] = read_vintages()


def max_abs_correlations(
    *, cache: FeatureCache, site: str, spec: GamSpec, theta: np.ndarray
) -> dict[str, float]:
    """Return, per penalised block, the largest absolute correlation with `u` over daylight cells.

    A cell is a calendar quarter by local half-hour of the daylight training rows; a cell with
    fewer than `MIN_ORTHOGONAL_CELL_ROWS` rows is left out, as in the penalty.
    """
    layout, index = prepare_fit(
        cache=cache,
        site=site,
        spec=spec,
        cut=common.TUNING_CUT,
        window=TrainingWindow(),
        season_years=common.SEASON_YEARS,
    )
    features = cache.sites[site]
    shape = derated_unit_shape(features=features, index=index)
    daylight = shape > LAST_TRAINED_MIN_SHAPE
    rows = index[daylight]
    unit = shape[daylight]
    design = design_matrix(cache=cache, features=features, layout=layout, index=rows)
    cell = cache.calendar.quarter[rows].astype(np.int64) * 48 + cache.calendar.half_hour[rows]
    result: dict[str, float] = {}
    for name in ORTHOGONAL_BLOCKS:
        block = layout.slices[name]
        term = design[:, block] @ theta[block]
        worst = 0.0
        for value in np.unique(cell):
            inside = cell == value
            if inside.sum() < MIN_ORTHOGONAL_CELL_ROWS:
                continue
            if np.std(term[inside]) < 1e-12 or np.std(unit[inside]) < 1e-12:
                continue
            worst = max(worst, abs(float(np.corrcoef(term[inside], unit[inside])[0, 1])))
        result[name] = worst
    return result


def _lambda_job(site: str) -> dict[str, dict[str, float]]:
    """Return the largest correlations at one site for the unpenalised fit and each weight."""
    cache: FeatureCache = _STATE["cache"]
    fitted = GamSpec(ridge=common.RIDGE, capacity="fitted")
    specs = [replace(fitted, orthogonal_penalty=value) for value in (0.0, *LAMBDA_GRID)]
    fit = fit_gam(
        cache=cache,
        site=site,
        specs=specs,
        cut=common.TUNING_CUT,
        window=TrainingWindow(),
        season_years=common.SEASON_YEARS,
    )
    return {
        f"{value:g}": max_abs_correlations(
            cache=cache, site=site, spec=specs[member], theta=fit.theta[member]
        )
        for member, value in enumerate((0.0, *LAMBDA_GRID))
    }


def choose_lambda(*, worst: dict[str, float]) -> float:
    """Return the smallest weight whose largest correlation is under the target, else the top."""
    for value in LAMBDA_GRID:
        if worst[f"{value:g}"] <= CORRELATION_TARGET:
            return value
    return LAMBDA_GRID[-1]


def _node_job(job: tuple[float, float, int]) -> dict[str, Any]:
    """Fit the node on the two tuning injections at D5 and read each back."""
    step_penalty, level_penalty, step_days = job
    cache: FeatureCache = _STATE["cache"]
    part_4 = arms.read_part_4_tuned()
    base = arms.base_specs(part_4=part_4)["b_d1"]
    spec = replace(
        base,
        node=NodeSpec(step_penalty=step_penalty, level_penalty=level_penalty, step_days=step_days),
    )
    site = common.TUNING_INJECTION_SITES[0]
    pinned = pinned_capacity(
        vintages=_STATE["vintages"], table=cache.sites[site].table, cut=common.TUNING_CUT
    )
    rows = []
    for injection in NODE_INJECTIONS:
        step_day = inject.day_of(moment=datetime(*injection["step_date"], tzinfo=UTC))
        first = inject.day_of(moment=datetime(*injection["outage"][:3], tzinfo=UTC))
        length = injection["outage"][3]
        output = inject.generator_level(
            cache=cache,
            level=injection["level"],
            step_day=step_day,
            step_size=injection["step"],
            outages=((first, length),),
        )
        injected = inject.inject_generator(cache=cache, site=site, output=output)
        fit = fit_gam(
            cache=injected,
            site=site,
            specs=[spec],
            cut=common.TUNING_CUT,
            window=NODE_WINDOW,
            season_years=common.SEASON_YEARS,
            pinned=pinned,
        )
        readout = inject.node_readout(fit=fit, step_day=step_day, outage=(first, length))
        truth_level = injection["level"] + (injection["step"] if first >= step_day else 0.0)
        recovered = (
            abs(readout.step_error_days) <= STEP_TOLERANCE_DAYS
            and abs(readout.step_size - injection["step"])
            <= STEP_SIZE_TOLERANCE * injection["step"]
            and readout.outage_found
            and abs(readout.outage_level - truth_level) <= LEVEL_TOLERANCE * truth_level
        )
        rows.append(
            {
                "step_penalty": step_penalty,
                "level_penalty": level_penalty,
                "level": injection["level"],
                "step_error_days": readout.step_error_days,
                "step_size": readout.step_size,
                "outage_found": readout.outage_found,
                "outage_level": readout.outage_level,
                "truth_level": truth_level,
                "recovered": recovered,
            }
        )
    return {"rows": rows}


def choose_node(*, results: list[dict[str, Any]]) -> dict[str, float]:
    """Return the largest step penalty, then the largest level penalty, that recovers both.

    If no pair recovers both injections, the smallest pair is returned and the caller reports it.
    """
    passing = [
        (rows[0]["step_penalty"], rows[0]["level_penalty"])
        for rows in (r["rows"] for r in results)
        if all(row["recovered"] for row in rows)
    ]
    if not passing:
        print("WARNING: no pair recovers both injections; the smallest pair is used")
        return {"step_penalty": min(STEP_PENALTY_GRID), "level_penalty": min(LEVEL_PENALTY_GRID)}
    step_penalty, level_penalty = max(passing)
    return {"step_penalty": step_penalty, "level_penalty": level_penalty}


def _score_job(job: tuple[str, str, arms.ArmDefinition]) -> dict[str, Any]:
    """Fit an arm at the tuning cut and return its unfiltered score and residual stream."""
    site, name, definition = job
    cache: FeatureCache = _STATE["cache"]
    try:
        run = arms.run_site(
            cache=cache,
            vintages=_STATE["vintages"],
            site=site,
            cut=common.TUNING_CUT,
            definition=definition,
        )
    except ValueError as error:
        if "no training row" not in str(error):
            raise
        return {"site": site, "arm": name, "score": float("nan"), "stream": np.array([np.nan])}
    keep = ~cache.sites[site].switching_mask[run.index]
    quantiles = arms.uncorrected_quantiles(cache=cache, run=run)[keep]
    score = float(
        common.quantile_score(
            actual=cache.sites[site].y[run.index][keep], quantiles=quantiles
        ).mean()
    )
    return {"site": site, "arm": name, "score": score, "stream": run.residual_stream(cache=cache)}


def main() -> None:
    """Choose the settings and write `work/tuned.json`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--node-step-days", type=int, default=7)
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "2"
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    common.load_cache()
    part_4 = arms.read_part_4_tuned()
    tuned: dict[str, Any] = {"node_step_days": arguments.node_step_days}
    with ProcessPoolExecutor(max_workers=arguments.workers, initializer=_init_worker) as pool:
        by_site = dict(
            zip(
                common.DECISION_SITES,
                pool.map(_lambda_job, common.DECISION_SITES),
                strict=True,
            )
        )
        worst = {
            f"{value:g}": max(
                max(by_site[site][f"{value:g}"].values()) for site in common.DECISION_SITES
            )
            for value in (0.0, *LAMBDA_GRID)
        }
        tuned["orthogonal_lambda"] = choose_lambda(worst=worst)
        tuned["lambda_worst_correlation"] = worst
        tuned["lambda_by_site"] = by_site
        node_jobs = [
            (step, level, arguments.node_step_days)
            for step in STEP_PENALTY_GRID
            for level in LEVEL_PENALTY_GRID
        ]
        node_results = list(pool.map(_node_job, node_jobs))
        tuned["node"] = {
            **choose_node(results=node_results),
            "step_days": arguments.node_step_days,
        }
        tuned["node_grid"] = [row for result in node_results for row in result["rows"]]
        partial = {**tuned}
        (common.WORK_DIR / "tuned_partial.json").write_text(json.dumps(partial, indent=2) + "\n")
        definitions = arms.arm_definitions(part_4=part_4, tuned=tuned)
        k_arms = ("k_base", "k_i", "k_ii")
        jobs = [
            (site, name, definition)
            for name, definition in definitions.items()
            for site in definition.sites
            if site in common.DECISION_SITES or definition.sites == common.MVA_SITES
        ]
        scored = list(pool.map(_score_job, jobs))
    scores: dict[str, dict[str, float]] = {}
    streams: dict[str, dict[str, np.ndarray]] = {}
    for result in scored:
        if np.isfinite(result["stream"]).any():
            scores.setdefault(result["arm"], {})[result["site"]] = result["score"]
            streams.setdefault(result["arm"], {})[result["site"]] = result["stream"]
    tuned["k_scores"] = {name: scores[name] for name in k_arms if name in scores}
    tuned["c4_arm"] = min(
        tuned["k_scores"], key=lambda name: float(np.mean(list(tuned["k_scores"][name].values())))
    )
    cache = common.load_cache()
    hyper: dict[str, dict[str, list[float]]] = {}
    for name, per_site in streams.items():
        fitted = fit_filter_hyperparameters(
            residuals=per_site,
            band=cache.calendar.band,
            end_times=cache.end_times,
            window=arms.tuning_window(),
        )
        values = {site: value.as_array() for site, value in fitted.items()}
        mean = np.mean(list(values.values()), axis=0)
        for site in definitions[name].sites:
            values.setdefault(site, mean)
        hyper[name] = {site: array.tolist() for site, array in values.items()}
    tuned["filter_hyper"] = hyper
    tuned["tuning_scores"] = scores
    (common.WORK_DIR / "tuned.json").write_text(json.dumps(tuned, indent=2) + "\n")
    _print_report(tuned=tuned, node_results=node_results)


def _print_report(*, tuned: dict[str, Any], node_results: list[dict[str, Any]]) -> None:
    """Print the tuning tables."""
    print("== covariance penalty: largest |corr| of a penalised term with u, all decision sites ==")
    for value, corr in tuned["lambda_worst_correlation"].items():
        print(f"  lambda {value:>6}: {corr:.4f}")
    print(f"  chosen lambda: {tuned['orthogonal_lambda']}")
    print("\n== node penalties on the tuning injection at D5 ==")
    print(pl.DataFrame(tuned["node_grid"]))
    print(f"  chosen: {tuned['node']}")
    print("\n== tuning-year quantile score (unfiltered), mean over sites ==")
    for name, per_site in tuned["tuning_scores"].items():
        print(f"  {name:>8} {np.mean(list(per_site.values())):7.3f}  ({len(per_site)} sites)")
    print(f"\nC4's arm (best tuning-year score among the K arms): {tuned['c4_arm']}")


if __name__ == "__main__":
    main()
