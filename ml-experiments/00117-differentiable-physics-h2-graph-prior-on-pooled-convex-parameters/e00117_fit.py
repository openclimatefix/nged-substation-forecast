"""Fit every arm at the requested cuts and store the predictions of the scored sites.

An arm is a penalty on the joint fit of one cut's nodes (`e00117_arms`). The graph weights and the
holiday weight come from `work/tuned.json` (`e00117_tune.py`), fitted on the tuning year only. The
three H-perm arms are H on a degree-preserving permutation of the graph, which tells a width gain
from neighbours being alike apart from a width gain from any extra shrinkage.

Each set is stored as `<cut>_<arm>`, and the MVA nodes' dark-row predictions (descriptive) as
`<cut>_<arm>_mvadark`.

    uv run python ml-experiments/00117-*/e00117_fit.py --cuts tuning
"""

import argparse
import json
from dataclasses import replace

import e00117_arms as arms
import e00117_common as common
import numpy as np
import torch
from studies.dp_gam_graph import GraphPenalty, degree_preserving_permutation
from studies.dp_pinned_capacity import read_vintages

N_PERMUTATIONS = 3
PERMUTATION_SEED = 20261020


def arm_catalogue(*, tuned: dict) -> dict[str, GraphPenalty]:
    """Return the penalty of every arm except the permuted ones."""
    decision = tuned["decision"]
    arms_ = {
        "g": GraphPenalty.none(),
        "n": GraphPenalty.of(fleet_rows=common.FLEET_ROWS),
        "h_lo": GraphPenalty.of(graph_rows=decision["gamma_lo"]),
        "nh_lo": GraphPenalty.of(fleet_rows=common.FLEET_ROWS, graph_rows=decision["gamma_lo"]),
        "phol": GraphPenalty.of(holiday_fleet_rows=decision["lambda_hol"]),
    }
    if decision["gamma_hi"] is not None:
        arms_["h_hi"] = GraphPenalty.of(graph_rows=decision["gamma_hi"])
        arms_["nh_hi"] = GraphPenalty.of(
            fleet_rows=common.FLEET_ROWS, graph_rows=decision["gamma_hi"]
        )
    return arms_


def permuted_context(*, context: arms.CutContext, index: int) -> arms.CutContext:
    """Return the context with the graph of the cut and of its replica permuted."""

    def permuted(plan: common.NodePlan) -> common.NodePlan:
        rng = np.random.default_rng([PERMUTATION_SEED, index])
        return replace(plan, graph=degree_preserving_permutation(graph=plan.graph, rng=rng))

    return replace(
        context, plan=permuted(context.plan), replica_plan=permuted(context.replica_plan)
    )


def main() -> None:
    """Fit and store the arms of the requested cuts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuts", nargs="+", required=True)
    parser.add_argument("--threads", type=int, default=6)
    arguments = parser.parse_args()
    torch.set_num_threads(arguments.threads)
    tuned = json.loads((common.WORK_DIR / "tuned.json").read_text())
    catalogue = arm_catalogue(tuned=tuned)
    cache = common.load_cache()
    vintages = read_vintages()
    for cut_name in arguments.cuts:
        context = arms.build_context(cache=cache, vintages=vintages, cut=common.CUTS[cut_name])
        five = arms.build_context(
            cache=cache, vintages=vintages, cut=common.CUTS[cut_name], labels=common.DECISION_SITES
        )
        jobs = [
            (name, five if name == "phol" else context, penalty)
            for name, penalty in catalogue.items()
        ]
        gamma = tuned["decision"]["gamma_lo"]
        jobs += [
            (
                f"h_perm{k}",
                permuted_context(context=context, index=k),
                GraphPenalty.of(graph_rows=gamma),
            )
            for k in range(N_PERMUTATIONS)
        ]
        for name, job_context, penalty in jobs:
            if (common.DATA_DIR / "predictions" / f"{cut_name}_{name}").exists():
                continue
            sites, dark, _ = arms.predict_arm(context=job_context, name=name, penalty=penalty)
            common.store_arm(name=f"{cut_name}_{name}", frame=sites)
            if not dark.is_empty():
                common.store_arm(name=f"{cut_name}_{name}_mvadark", frame=dark)
            print(f"{cut_name}_{name}: {sites.height} rows", flush=True)


if __name__ == "__main__":
    main()
