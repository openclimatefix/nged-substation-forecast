"""Tuning, C1, and C2 of experiment 124's pilot, on real readings.

    uv run python ml-experiments/00124-.../e00124_run.py tune
    uv run python ml-experiments/00124-.../e00124_run.py score replicate
    uv run python ml-experiments/00124-.../e00124_run.py score scored

`tune` fits the pilot at the tuning cut for the whole weight grid, injects metering errors into the
tuning year's readings (seeds disjoint from the scored cuts'), and writes W* to `tuning.json`
under the rule fixed in the plan: the weight with the largest summed damage reduction whose clean
score is at most 0.023 points worse than weight 0. `score` fits at a cut, runs the C1 and C2
injections for weight 0, W*, 30, and the oracle, and appends one JSON line per fit.
"""

import json
import sys
import time
from datetime import datetime
from typing import Final, TextIO

import numpy as np
import torch
from e00124_common import CUTS, WORK_DIR, load_bsps, load_cache
from e00124_inject import KINDS, ScoreRows, draw_injection, mae, monthly_mae, score_rows
from e00124_pilot import SEASON_YEARS, WEIGHT_GRID, Pilot, build_pilot
from studies.dp_features import FeatureCache
from studies.dp_hierarchy import JointAssembly, RemainderSpec, assemble_joint, solve_joint

CLEAN_COST_LIMIT: Final[float] = 0.023
"""Part 4's noise reference for a mean score, in points (97.5% level)."""
TUNING_SEEDS: Final[int] = 10
SCORED_SEEDS: Final[int] = 10
MAX_WEIGHT: Final[float] = 30.0
DEVICE: Final[str] = "cuda" if torch.cuda.is_available() else "cpu"


def _setup(*, cut_name: str) -> tuple[FeatureCache, Pilot, JointAssembly, datetime]:
    cache = load_cache()
    bsps = load_bsps(cache=cache)
    cut = CUTS[cut_name]
    pilot = build_pilot(cache=cache, bsps=bsps, cut=cut)
    assembly = assemble_joint(
        cache=cache,
        specs=pilot.specs,
        bsps=[pilot.bsp],
        remainder_spec=RemainderSpec(),
        sigma_points=pilot.sigma_points,
        cut=cut,
        window=pilot.windows,
        season_years=SEASON_YEARS,
        device=DEVICE,
    )
    return cache, pilot, assembly, cut


def _inject_all(
    *,
    cache: FeatureCache,
    pilot: Pilot,
    assembly: JointAssembly,
    cut: datetime,
    weights: list[float],
    seeds: int,
    seed_offset: int,
    scores: dict[str, ScoreRows],
    out: TextIO,
    with_oracle: bool,
) -> None:
    """Run every injection cell at every MW Primary and append the damage rows to `out`."""
    rng_base = seed_offset
    started = time.time()
    count = 0
    for site_number, site in enumerate(pilot.mw_sites):
        for cell_number, cell in enumerate(KINDS):
            for seed in range(seeds):
                rng = np.random.default_rng([rng_base, cell_number, seed, site_number])
                injection = draw_injection(
                    cache=cache, assembly=assembly, site=site, cell=cell, cut=cut, rng=rng
                )
                if injection is None:
                    continue
                excluded = {site: injection.rows} if injection.kind == "gap" else None
                power = None if injection.kind == "gap" else {site: injection.power}
                fit = solve_joint(
                    assembly=assembly,
                    weights=weights,
                    primary_power=power,
                    excluded_rows=excluded,
                )
                row = {"site": site, "cell": injection.label, "kind": injection.kind, "seed": seed}
                row["mae"] = mae(score=scores[site], fit=fit, site=site).tolist()
                row["siblings"] = {
                    other: mae(score=scores[other], fit=fit, site=other).tolist()
                    for other in pilot.mw_sites
                    if other != site
                }
                if injection.kind != "gap" and with_oracle:
                    oracle = solve_joint(
                        assembly=assembly,
                        weights=[weights[1]],
                        primary_power={site: injection.power},
                        excluded_rows={site: injection.rows},
                    )
                    row["oracle"] = float(mae(score=scores[site], fit=oracle, site=site)[0])
                out.write(json.dumps(row) + "\n")
                out.flush()
                count += 1
        sys.stdout.write(
            f"Primary {site_number}: {count} injections, {time.time() - started:.0f} s\n"
        )
        sys.stdout.flush()


def tune() -> None:
    """Choose W* on the tuning year's injections and write it with the clean costs."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    cache, pilot, assembly, cut = _setup(cut_name="tuning")
    clean = solve_joint(assembly=assembly, weights=WEIGHT_GRID)
    scores = {
        s: score_rows(cache=cache, assembly=assembly, fit=clean, site=s, cut=cut)
        for s in pilot.mw_sites
    }
    clean_mae = {s: mae(score=scores[s], fit=clean, site=s) for s in pilot.mw_sites}
    path = WORK_DIR / "tuning_injections.jsonl"
    with path.open("w") as out:
        _inject_all(
            cache=cache,
            pilot=pilot,
            assembly=assembly,
            cut=cut,
            weights=list(WEIGHT_GRID),
            seeds=TUNING_SEEDS,
            seed_offset=1000,
            scores=scores,
            out=out,
            with_oracle=False,
        )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    damage = np.zeros(len(WEIGHT_GRID))
    for row in rows:
        damage += np.array(row["mae"]) - np.array(clean_mae[row["site"]])
    cost = np.mean([clean_mae[s] - clean_mae[s][0] for s in pilot.mw_sites], axis=0)
    reduction = damage[0] - damage
    allowed = cost <= CLEAN_COST_LIMIT
    best = int(np.argmax(np.where(allowed, reduction, -np.inf)))
    result = {
        "weights": list(WEIGHT_GRID),
        "summed_damage": damage.tolist(),
        "summed_reduction": reduction.tolist(),
        "clean_cost": cost.tolist(),
        "w_star": WEIGHT_GRID[best],
        "n_injections": len(rows),
    }
    (WORK_DIR / "tuning.json").write_text(json.dumps(result, indent=2))
    sys.stdout.write(json.dumps(result, indent=2) + "\n")


def score(*, cut_name: str) -> None:
    """Fit at a cut and run the C1 and C2 injections for weight 0, W*, and 30."""
    w_star = json.loads((WORK_DIR / "tuning.json").read_text())["w_star"]
    weights = [0.0, w_star, MAX_WEIGHT]
    cache, pilot, assembly, cut = _setup(cut_name=cut_name)
    clean = solve_joint(assembly=assembly, weights=weights)
    scores = {
        s: score_rows(cache=cache, assembly=assembly, fit=clean, site=s, cut=cut)
        for s in pilot.mw_sites
    }
    summary = {
        "cut": cut_name,
        "weights": weights,
        "clean_mae": {s: mae(score=scores[s], fit=clean, site=s).tolist() for s in pilot.mw_sites},
        "monthly_mae": {
            s: monthly_mae(cache=cache, score=scores[s], fit=clean, site=s) for s in pilot.mw_sites
        },
    }
    (WORK_DIR / f"clean_{cut_name}.json").write_text(json.dumps(summary, indent=2))
    with (WORK_DIR / f"injections_{cut_name}.jsonl").open("w") as out:
        _inject_all(
            cache=cache,
            pilot=pilot,
            assembly=assembly,
            cut=cut,
            weights=weights,
            seeds=SCORED_SEEDS,
            seed_offset=2000 + len(cut_name),
            scores=scores,
            out=out,
            with_oracle=True,
        )


if __name__ == "__main__":
    if sys.argv[1] == "tune":
        tune()
    else:
        score(cut_name=sys.argv[2])
