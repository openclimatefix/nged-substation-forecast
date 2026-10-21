"""Leave-one-out checks for the audit: do R1 and R2 survive without any one site, edge, or node?

Dropping a decision site removes its rows from the paired contrasts. Dropping an edge or a node
refits the arms on the graph without it. The script prints each variant's R1 (H minus G) and R2
(NH minus N) estimate and 98.75% interval in the scored year, and says whether the sign or the
non-inferiority reading (upper end below +0.03) changes.

    uv run python ml-experiments/00117-*/e00117_leave_one_out.py
"""

import json
from dataclasses import replace
from datetime import datetime

import e00117_arms as arms
import e00117_common as common
import e00117_report as report
import polars as pl
import torch
from studies.dp_features import FeatureCache
from studies.dp_gam_graph import GraphPenalty, InterconnectionGraph
from studies.dp_pinned_capacity import read_vintages

GAMMA = 10_000.0
MARGIN = 0.03


def contrasts_from(*, frames: dict[str, pl.DataFrame], ids: list[int]) -> dict[str, dict]:
    """Return R1 and R2 from stored or fresh prediction frames on the given series."""
    losses = {
        name: report.row_losses(frame=frame.filter(pl.col("time_series_id").is_in(ids)))
        for name, frame in frames.items()
    }
    out = {}
    for spec in report.CONTRASTS[:2]:
        out[spec.name] = report.contrast_result(
            treatment=losses[spec.treatment],
            reference=losses[spec.reference],
            spec=spec,
            metric="qs",
        )
    return out


def verdict_reading(*, result: dict) -> str:
    """Return a short reading of one contrast: non-inferior, undecided, or worse."""
    if result.get("upper") is None:
        return "no interval"
    if result["upper"] <= MARGIN:
        return "non-inferior"
    if result["lower"] > MARGIN:
        return "worse than the margin"
    return "undecided"


def refit_variant(
    *,
    cache: FeatureCache,
    vintages: pl.DataFrame,
    cut: datetime,
    drop_node: str | None,
    drop_edge: tuple[str, str] | None,
) -> dict[str, pl.DataFrame]:
    """Refit G, N, H, and NH on the graph without a node or an edge."""
    nodes = None
    if drop_node is not None:
        base = common.decision_edges(cache=cache)
        nodes = tuple(sorted({n for e in base for n in e} - {drop_node}))
    context = arms.build_context(cache=cache, vintages=vintages, cut=cut, labels=nodes)
    if drop_edge is not None:
        graph = context.plan.graph
        kept = tuple(e for e in graph.edges if e != drop_edge)
        context = replace(
            context,
            plan=replace(context.plan, graph=InterconnectionGraph(nodes=graph.nodes, edges=kept)),
            replica_plan=replace(
                context.replica_plan,
                graph=InterconnectionGraph(
                    nodes=context.replica_plan.graph.nodes,
                    edges=tuple(e for e in context.replica_plan.graph.edges if e != drop_edge),
                ),
            ),
        )
    penalties = {
        "g": GraphPenalty.none(),
        "n": GraphPenalty.of(fleet_rows=common.FLEET_ROWS),
        "h_lo": GraphPenalty.of(graph_rows=GAMMA),
        "nh_lo": GraphPenalty.of(fleet_rows=common.FLEET_ROWS, graph_rows=GAMMA),
    }
    return {
        name: arms.predict_arm(context=context, name=name, penalty=penalty)[0]
        for name, penalty in penalties.items()
    }


def main() -> None:
    """Run every variant and write the table."""
    torch.set_num_threads(6)
    cache = common.load_cache()
    vintages = read_vintages()
    cut = common.SCORED_CUT
    ids = {s: cache.sites[s].table.time_series_id for s in common.DECISION_SITES}
    stored = {
        name: pl.read_parquet(
            common.DATA_DIR / "predictions" / f"scored_{name}" / "predictions.parquet"
        )
        for name in ("g", "n", "h_lo", "nh_lo")
    }
    rows: list[tuple[str, dict[str, dict]]] = [
        ("all", contrasts_from(frames=stored, ids=list(ids.values())))
    ]
    for site in common.DECISION_SITES:
        keep = [i for s, i in ids.items() if s != site]
        rows.append((f"drop site {site}", contrasts_from(frames=stored, ids=keep)))
    edges = common.decision_edges(cache=cache)
    for edge in (("D10", "D11"), ("D16", "D5")):
        pair = (min(edge), max(edge))
        if pair in edges:
            fresh = refit_variant(
                cache=cache, vintages=vintages, cut=cut, drop_node=None, drop_edge=pair
            )
            rows.append(
                (
                    f"drop edge {pair[0]}-{pair[1]}",
                    contrasts_from(frames=fresh, ids=list(ids.values())),
                )
            )
    mva = sorted({n for e in edges for n in e if cache.sites[n].table.units == "MVA"})
    for node in mva:
        fresh = refit_variant(
            cache=cache, vintages=vintages, cut=cut, drop_node=node, drop_edge=None
        )
        rows.append(
            (
                f"drop MVA node {mva.index(node) + 1} of {len(mva)}",
                contrasts_from(frames=fresh, ids=list(ids.values())),
            )
        )
    path = common.WORK_DIR / "leave_one_out.json"
    table = []
    for variant, result in rows:
        r1, r2 = result["R1 H - G"], result["R2 NH - N"]
        read1, read2 = verdict_reading(result=r1), verdict_reading(result=r2)
        table.append(
            {"variant": variant, "R1": r1, "R1 reading": read1, "R2": r2, "R2 reading": read2}
        )
        print(variant, _format(result=r1, reading=read1), _format(result=r2, reading=read2))
    path.write_text(json.dumps(table, indent=2, default=float) + "\n")


def _format(*, result: dict, reading: str) -> str:
    return (
        f"{result.get('difference', float('nan')):+.3f} "
        f"[{result.get('lower', float('nan')):+.3f}, {result.get('upper', float('nan')):+.3f}] "
        f"{reading};"
    )


if __name__ == "__main__":
    main()
