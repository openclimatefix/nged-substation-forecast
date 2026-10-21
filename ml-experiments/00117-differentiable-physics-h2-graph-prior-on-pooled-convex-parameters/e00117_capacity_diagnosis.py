"""Diagnose which graph-penalty family damages the intervals, on four synthetic smooth fleets.

For each family alone (and no penalty) at gamma = 10,000, the script fits four smooth-cell fleets
and reports the coverage, the mean absolute error, and the mean width of each parameter. It also
prints the capacity family's scale `g_cap`, the weight it gives an interconnection, and the
information the data hold about a substation's capacity sum. Four fleets are too few to read a
coverage to better than about 10 points, so the result is a diagnosis and not a gate.

    uv run python ml-experiments/00117-*/e00117_capacity_diagnosis.py
"""

import json

import e00117_common as common
import e00117_synth as synth
import numpy as np
import polars as pl
import torch
from studies.dp_gam_graph import GraphPenalty, family_scales, solve_graph_fit

GAMMA = 10_000.0
N_FLEETS = 4
FAMILIES = ("none", "capacity", "heating", "profile", "holiday")


def main() -> None:
    """Fit the fleets and write the table."""
    torch.set_num_threads(2)
    templates = synth.build_templates(cache=common.load_cache())
    coverage = synth._coverage_table(templates=templates)
    rows = []
    information = []
    scale = 0.0
    for index in range(N_FLEETS):
        seed = (9, 0, index)
        fleet = synth.draw_fleet(templates=templates, coverage=coverage, kind="smooth", seed=seed)
        counts = synth.fleet_counts(fleet=fleet, seed=seed)
        scale = family_scales(designs=fleet.designs)["capacity"]
        for design in fleet.designs:
            identified = design.identified[design.layout.slices["capacity"]]
            if identified.any():
                gram = design.month_gram.sum(dim=0)[
                    design.layout.slices["capacity"], design.layout.slices["capacity"]
                ]
                ones = torch.ones(gram.shape[0], dtype=torch.float64)
                information.append(float(1.0 / (ones @ torch.linalg.pinv(gram) @ ones)))
        for family in FAMILIES:
            penalty = (
                GraphPenalty.none()
                if family == "none"
                else GraphPenalty(fleet={}, graph={family: GAMMA})
            )
            solution = solve_graph_fit(
                designs=fleet.designs,
                graph=fleet.graph,
                penalty=penalty,
                scales=family_scales(designs=fleet.designs),
                axis=fleet.axis,
                counts=counts,
                month_rhs=fleet.month_rhs,
                device="cuda",
            )
            for design in fleet.designs:
                values = synth.functional_values(design=design, theta=solution.thetas[design.site])
                truth = synth.functional_values(
                    design=design, theta=fleet.truth[design.site][None]
                )[0]
                flags = synth.identified_functionals(design=design)
                median = np.median(values[1:], axis=0)
                low, high = np.quantile(
                    values[1:], [synth.INTERVAL_LOW, synth.INTERVAL_HIGH], axis=0
                )
                for k, name in enumerate(synth.FUNCTIONALS):
                    if not flags[k]:
                        continue
                    lo = median[k] - synth.INFLATE * (median[k] - low[k])
                    hi = median[k] + synth.INFLATE * (high[k] - median[k])
                    rows.append(
                        {
                            "penalty": family,
                            "functional": name,
                            "cover": float(lo <= truth[k] <= hi),
                            "error": abs(float(values[0, k] - truth[k])),
                            "width": float(hi - lo),
                        }
                    )
    table = (
        pl.DataFrame(rows)
        .group_by("penalty", "functional")
        .agg(pl.col("cover").mean(), pl.col("error").mean(), pl.col("width").mean())
        .sort("functional", "penalty")
    )
    report = {
        "capacity_scale": scale,
        "weight_per_interconnection": GAMMA * scale,
        "capacity_sum_information_median": float(np.median(information)),
        "table": table.to_dicts(),
    }
    (common.WORK_DIR / "capacity_diagnosis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "table"}, indent=2))
    print(table.filter(pl.col("functional").is_in(["capacity", "slope_3"])))


if __name__ == "__main__":
    main()
