"""The plumbing controls P1 to P5 on the real data, before any tuning or scoring.

P1: with no penalty each node equals `dp_gam.fit_gam`; the fleet-mean fit equals
`dp_gam_pooled.fit_pooled`; a graph with no edges equals the fit without a graph.
P2: the joint solver matches a dense program on a small graph and ignores node order (the module's
own tests; this script also checks that node order changes nothing at the real size).
P3: as the graph weight grows, two connected nodes' identified parameters converge.
P4: changing any reading, weather value, or solar shape of one node after the cut changes no node's
fit and no node's replica spread.
P5: the fleet-mean arm reproduces experiment 109's stored scores on the five decision sites to
0.005 points, with 109's pool and its penalty of 3,000 rows.

Every control prints its numbers. A failure makes the gate fail; the script exits non-zero.

    uv run python ml-experiments/00117-*/e00117_controls.py
"""

import json
import sys
from dataclasses import replace
from datetime import datetime
from typing import Any

import e00117_arms as arms
import e00117_common as common
import numpy as np
import polars as pl
import torch
from studies.dp_features import FeatureCache
from studies.dp_gam import fit_gam
from studies.dp_gam_graph import GraphPenalty, InterconnectionGraph
from studies.dp_gam_pooled import PoolingStrength, fit_pooled
from studies.dp_pinned_capacity import read_vintages
from studies.sources import ML_EXPERIMENTS_DATA_DIR

TOLERANCE_EXACT = 1e-8
TOLERANCE_POOLED = 1e-6
TOLERANCE_SCORE = 0.005
STORED_109 = "00109-differentiable-physics-n0-partially-pooled-convex-model"
POOL_109 = ("D5", "D9", "D10", "D11", "D14", "D16", "D18")


def p1(*, context: arms.CutContext) -> dict[str, Any]:
    """Return P1's maximum differences."""
    plan = context.plan
    none = arms.solve(designs=context.designs, plan=plan, penalty=GraphPenalty.none())
    worst_gam = 0.0
    for site in plan.sites:
        if site.site not in common.DECISION_SITES:
            continue
        alone = fit_gam(
            cache=plan.cache,
            site=site.site,
            specs=[site.spec],
            cut=plan.cut,
            window=site.window,
            season_years=site.season_years,
            pinned=site.pinned,
        )
        worst_gam = max(worst_gam, float(np.abs(none.thetas[site.site] - alone.theta).max()))
    fleet = GraphPenalty.of(fleet_rows=common.FLEET_ROWS)
    mine = arms.solve(designs=context.designs, plan=plan, penalty=fleet)
    pooled = fit_pooled(
        cache=plan.cache,
        sites=list(plan.sites),
        cut=plan.cut,
        strengths=[PoolingStrength.uniform(value=common.FLEET_ROWS)],
    )
    worst_pooled = max(
        float(np.abs(mine.thetas[s.site] - pooled.fits[s.site].theta).max()) for s in plan.sites
    )
    no_edges = replace(plan, graph=InterconnectionGraph(nodes=plan.graph.nodes, edges=()))
    h = arms.solve(
        designs=context.designs, plan=no_edges, penalty=GraphPenalty.of(graph_rows=3_000.0)
    )
    nh = arms.solve(
        designs=context.designs,
        plan=no_edges,
        penalty=GraphPenalty.of(fleet_rows=common.FLEET_ROWS, graph_rows=3_000.0),
    )
    worst_h = max(float(np.abs(h.thetas[s] - none.thetas[s]).max()) for s in none.thetas)
    worst_nh = max(float(np.abs(nh.thetas[s] - mine.thetas[s]).max()) for s in none.thetas)
    return {
        "h0_vs_fit_gam": worst_gam,
        "n_vs_fit_pooled": worst_pooled,
        "no_edges_h_vs_g": worst_h,
        "no_edges_nh_vs_n": worst_nh,
        "passed": worst_gam < TOLERANCE_EXACT
        and worst_pooled < TOLERANCE_POOLED
        and worst_h < TOLERANCE_EXACT
        and worst_nh < TOLERANCE_EXACT,
    }


def p2(*, context: arms.CutContext) -> dict[str, Any]:
    """Return the largest change when the graph's node order is reversed."""
    penalty = GraphPenalty.of(fleet_rows=common.FLEET_ROWS, graph_rows=3_000.0)
    forward = arms.solve(designs=context.designs, plan=context.plan, penalty=penalty)
    graph = context.plan.graph
    backward_graph = InterconnectionGraph(nodes=tuple(reversed(graph.nodes)), edges=graph.edges)
    backward = arms.solve(
        designs=context.designs, plan=replace(context.plan, graph=backward_graph), penalty=penalty
    )
    worst = max(float(np.abs(forward.thetas[s] - backward.thetas[s]).max()) for s in forward.thetas)
    return {"node_order": worst, "passed": worst < TOLERANCE_EXACT}


def p3(*, context: arms.CutContext) -> dict[str, Any]:
    """Return the heating gap between two connected decision sites as the graph weight grows."""
    first, second = "D5", "D16"
    layout = next(d.layout for d in context.designs if d.site == first)
    heating = layout.slices["heating"]
    gaps = []
    grid = (0.0, 1e3, 1e5, 1e7)
    for gamma in grid:
        solution = arms.solve(
            designs=context.designs, plan=context.plan, penalty=GraphPenalty.of(graph_rows=gamma)
        )
        gaps.append(
            float(
                np.abs(
                    solution.thetas[first][0, heating] - solution.thetas[second][0, heating]
                ).max()
            )
        )
    return {
        "grid": list(grid),
        "gaps": gaps,
        "passed": gaps == sorted(gaps, reverse=True) and gaps[-1] < 0.05 * gaps[0],
    }


def perturbed_cache(*, cache: FeatureCache, site: str, cut: datetime) -> FeatureCache:
    """Return the cache with one node's readings, weather, and solar shape changed after `cut`."""
    features = cache.sites[site]
    after = cache.end_times > common.to_64(moment=cut)
    y = features.y.copy()
    y[after] = np.nan_to_num(y[after]) + 1000.0
    temperature = features.smoothed_temperature_c.copy()
    temperature[after] += 25.0
    shape = features.pv_unit_shape.copy()
    shape[after] *= 3.0
    changed = replace(
        features,
        y=y,
        smoothed_temperature_c=temperature,
        temperature_c=features.temperature_c + 25.0 * after,
        pv_unit_shape=shape,
    )
    return replace(cache, sites={**cache.sites, site: changed})


def p4(*, cache: FeatureCache, vintages: pl.DataFrame, cut: datetime) -> dict[str, Any]:
    """Return the largest change in any other node's fit or replica spread."""
    penalty = GraphPenalty.of(fleet_rows=common.FLEET_ROWS, graph_rows=3_000.0)
    base = arms.build_context(cache=cache, vintages=vintages, cut=cut)
    changed_cache = perturbed_cache(cache=cache, site="D14", cut=cut)
    changed = arms.build_context(cache=changed_cache, vintages=vintages, cut=cut)
    first = arms.solve(designs=base.designs, plan=base.plan, penalty=penalty)
    second = arms.solve(designs=changed.designs, plan=changed.plan, penalty=penalty)
    worst_fit = max(float(np.abs(first.thetas[s] - second.thetas[s]).max()) for s in first.thetas)
    spreads = arms.replica_spreads(context=base, penalty=penalty)
    other = arms.replica_spreads(context=changed, penalty=penalty)
    worst_spread = max(
        float(np.abs(spreads[s].table - other[s].table).max()) for s in spreads if s != "D14"
    )
    return {
        "fit": worst_fit,
        "replica_spread_other_nodes": worst_spread,
        "passed": worst_fit < 1e-10 and worst_spread < 1e-10,
    }


def p5(*, cache: FeatureCache, vintages: pl.DataFrame, cut: datetime) -> dict[str, Any]:
    """Return N's score difference from experiment 109's stored scores at the five sites."""
    context = arms.build_context(
        cache=cache, vintages=vintages, cut=cut, labels=POOL_109, allow_excluded=("D18",)
    )
    penalty = GraphPenalty.of(fleet_rows=common.FLEET_ROWS)
    mine, _, _ = arms.predict_arm(context=context, name="n_109_pool", penalty=penalty)
    stored = pl.read_parquet(
        ML_EXPERIMENTS_DATA_DIR / STORED_109 / "predictions" / "scored_n" / "predictions.parquet"
    )
    keep = mine.select("time_series_id", "valid_time")
    stored = stored.join(keep, on=["time_series_id", "valid_time"], how="semi")
    mine = mine.join(
        stored.select("time_series_id", "valid_time"),
        on=["time_series_id", "valid_time"],
        how="semi",
    )
    mine_scores = arms.score_by_site(frame=mine)
    stored_scores = arms.score_by_site(
        frame=stored.with_columns(switching_mask=pl.col("switching_mask"))
    )
    joined = mine_scores.join(stored_scores, on="time_series_id", suffix="_109")
    joined = joined.with_columns(difference=pl.col("score") - pl.col("score_109"))
    by_id = {context.cache.sites[s].table.time_series_id: s for s in common.DECISION_SITES}
    five = joined.filter(pl.col("time_series_id").is_in(list(by_id)))
    mean_difference = common.as_float(five["difference"].mean())
    return {
        "rows_compared": mine.height,
        "score_mine": common.as_float(five["score"].mean()),
        "score_109": common.as_float(five["score_109"].mean()),
        "per_site": {by_id[row["time_series_id"]]: row["difference"] for row in five.to_dicts()},
        "mean_difference": mean_difference,
        "passed": abs(mean_difference) < TOLERANCE_SCORE,
        "note": "N here uses experiment 109's pool, including its seventh series; H2's own N pools "
        "the decision graph's nodes.",
    }


def main() -> None:
    """Run P1 to P5 and write the report."""
    torch.set_num_threads(6)
    cache = common.load_cache()
    vintages = read_vintages()
    edges = common.decision_edges(cache=cache)
    report: dict[str, Any] = {
        "edges": len(edges),
        "edge_digest": common.edge_list_digest(edges=edges),
        "run": common.run_metadata(),
    }
    counts = {}
    for name, cut in common.CUTS.items():
        plan = common.prepare_nodes(cache=cache, vintages=vintages, cut=cut)
        counts[name] = {
            "nodes": len(plan.sites),
            "mva_nodes": sum(cache.sites[s.site].table.units == "MVA" for s in plan.sites),
            "edges": len(plan.graph.edges),
            "left_out": plan.left_out,
        }
    report["counts"] = counts
    print(json.dumps(counts, indent=2), flush=True)
    only = sys.argv[1:] or ["p1", "p2", "p3", "p4", "p5"]
    path = common.WORK_DIR / "controls.json"
    if path.exists() and sys.argv[1:]:
        report = json.loads(path.read_text()) | {"run": report["run"]}
    if {"p1", "p2", "p3"} & set(only):
        context = arms.build_context(cache=cache, vintages=vintages, cut=common.SCORED_CUT)
    if "p1" in only:
        report["p1"] = p1(context=context)
        print("P1", report["p1"], flush=True)
    if "p2" in only:
        report["p2"] = p2(context=context)
        print("P2", report["p2"], flush=True)
    if "p3" in only:
        report["p3"] = p3(context=context)
        print("P3", report["p3"], flush=True)
    if "p4" in only:
        report["p4"] = p4(cache=cache, vintages=vintages, cut=common.SCORED_CUT)
        print("P4", report["p4"], flush=True)
    if "p5" in only:
        report["p5"] = p5(cache=cache, vintages=vintages, cut=common.SCORED_CUT)
        print("P5", report["p5"], flush=True)
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    failed = [k for k in ("p1", "p2", "p3", "p4", "p5") if not report.get(k, {}).get("passed")]
    print("failed:", failed or "none")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
