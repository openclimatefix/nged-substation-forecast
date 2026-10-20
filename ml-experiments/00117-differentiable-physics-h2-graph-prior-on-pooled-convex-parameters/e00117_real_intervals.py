"""Real-data parameter intervals: the width role of the graph prior, with a permuted-graph placebo.

For each arm (G, N, H, NH, and H on three degree-preserving permutations of the graph) the script
fits the joint model of a cut on the full sample and on 40 joint month-block resamples, and writes
the 80% interval of every parameter that a node identifies. It also prints the edge-similarity
ratio of the unpenalised (G) parameters, which says whether interconnected substations are alike,
and G's residual correlation along the graph's edges.

A width gain that H shares with H-perm is the prior shrinking variance, not neighbours being alike.

    uv run python ml-experiments/00117-*/e00117_real_intervals.py --cut scored
"""

import argparse
import json
import os
from typing import Any

import e00117_arms as arms
import e00117_common as common
import e00117_fit as fit_module
import e00117_synth as synth
import numpy as np
import polars as pl
import torch
from studies.dp_gam_graph import (
    GraphPenalty,
    edge_similarity_ratio,
    family_scales,
    joint_month_counts,
    month_axis_of,
    solve_graph_fit,
)
from studies.dp_pinned_capacity import read_vintages

SEED = 20261021


def interval_frame(*, context: arms.CutContext, arm: str, penalty: GraphPenalty) -> pl.DataFrame:
    """Return the 80% interval of every identified parameter of every node for one arm."""
    designs = context.designs
    axis = month_axis_of(designs=designs)
    counts = joint_month_counts(
        axis=axis, n_resamples=synth.N_BOOTSTRAP, rng=np.random.default_rng(SEED)
    )
    solution = solve_graph_fit(
        designs=designs,
        graph=context.plan.graph,
        penalty=penalty,
        scales=family_scales(designs=designs),
        axis=axis,
        counts=counts,
        device=os.environ.get("E00117_DEVICE", "cpu"),
    )
    rows = []
    for design in designs:
        values = synth.functional_values(design=design, theta=solution.thetas[design.site])
        flags = synth.identified_functionals(design=design)
        median = np.median(values[1:], axis=0)
        low, high = np.quantile(values[1:], [synth.INTERVAL_LOW, synth.INTERVAL_HIGH], axis=0)
        for k, name in enumerate(synth.FUNCTIONALS):
            rows.append(
                {
                    "arm": arm,
                    "node": design.site,
                    "functional": name,
                    "family": synth.WIDTH_FAMILY[name],
                    "identified": bool(flags[k]),
                    "point": float(values[0, k]),
                    "lo": float(median[k] - synth.INFLATE * (median[k] - low[k])),
                    "hi": float(median[k] + synth.INFLATE * (high[k] - median[k])),
                }
            )
    return pl.DataFrame(rows).with_columns(width=pl.col("hi") - pl.col("lo"))


def residual_correlation(*, context: arms.CutContext) -> float:
    """Return G's mean residual correlation along the graph's edges, on rows both ends hold."""
    solution = solve_graph_fit(
        designs=context.designs,
        graph=context.plan.graph,
        penalty=GraphPenalty.none(),
        scales=family_scales(designs=context.designs),
    )
    by_site = {d.site: d for d in context.designs}
    grid = len(context.plan.cache.end_times)
    residual: dict[str, np.ndarray] = {}
    for site, design in by_site.items():
        series = np.full(grid, np.nan)
        target = context.plan.cache.sites[site].y[design.index]
        series[design.index] = target - design.x @ solution.thetas[site][0]
        residual[site] = series
    values = []
    for first, second in context.plan.graph.edges:
        both = np.isfinite(residual[first]) & np.isfinite(residual[second])
        if both.sum() > 1000:
            values.append(float(np.corrcoef(residual[first][both], residual[second][both])[0, 1]))
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    """Write the real intervals and the placebo comparison of a cut."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cut", default="scored")
    arguments = parser.parse_args()
    torch.set_num_threads(6)
    tuned = json.loads((common.WORK_DIR / "tuned.json").read_text())
    catalogue = fit_module.arm_catalogue(tuned=tuned)
    cache = common.load_cache()
    context = arms.build_context(
        cache=cache, vintages=read_vintages(), cut=common.CUTS[arguments.cut]
    )
    gamma = tuned["decision"]["gamma_lo"]
    jobs: list[tuple[str, arms.CutContext, GraphPenalty]] = [
        (name, context, catalogue[name]) for name in ("g", "n", "h_lo", "nh_lo")
    ]
    jobs += [
        (
            f"h_perm{k}",
            fit_module.permuted_context(context=context, index=k),
            GraphPenalty.of(graph_rows=gamma),
        )
        for k in range(fit_module.N_PERMUTATIONS)
    ]
    frames = [interval_frame(context=c, arm=name, penalty=p) for name, c, p in jobs]
    intervals = pl.concat(frames).filter(pl.col("identified"))
    wide = intervals.pivot(on="arm", index=["node", "functional", "family"], values="width")
    report: dict[str, Any] = {"cut": arguments.cut, "gamma": gamma}
    ratios = {}
    for arm in wide.columns[3:]:
        if arm == "g":
            continue
        ratios[arm] = {
            family: common.as_float(
                wide.filter(pl.col("family") == family)
                .select(ratio=pl.col(arm) / pl.col("g"))["ratio"]
                .median()
            )
            for family in wide["family"].unique().sort().to_list()
        }
    perm_columns = [c for c in wide.columns if c.startswith("h_perm")]
    report["width_ratio_to_g"] = ratios
    report["h_over_h_perm"] = {
        family: common.as_float(
            wide.filter(pl.col("family") == family)
            .select(ratio=pl.col("h_lo") / pl.concat_list(perm_columns).list.mean())["ratio"]
            .median()
        )
        for family in wide["family"].unique().sort().to_list()
    }
    solution = solve_graph_fit(
        designs=context.designs,
        graph=context.plan.graph,
        penalty=GraphPenalty.none(),
        scales=family_scales(designs=context.designs),
    )
    similarity = {}
    for k, name in enumerate(synth.FUNCTIONALS):
        values = {
            design.site: float(
                synth.functional_values(design=design, theta=solution.thetas[design.site])[0, k]
            )
            for design in context.designs
            if synth.identified_functionals(design=design)[k]
        }
        nodes = tuple(values)
        graph = context.plan.graph
        sub = type(graph).from_pairs(
            nodes=nodes, pairs=[e for e in graph.edges if e[0] in values and e[1] in values]
        )
        if len(nodes) >= 3 and sub.edges:
            similarity[name] = edge_similarity_ratio(
                graph=sub, values=values, rng=np.random.default_rng(SEED)
            )
    report["edge_similarity_ratio"] = similarity
    report["g_residual_correlation_along_edges"] = residual_correlation(context=context)
    intervals.write_parquet(common.WORK_DIR / f"real_intervals_{arguments.cut}.parquet")
    (common.WORK_DIR / f"real_intervals_{arguments.cut}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
