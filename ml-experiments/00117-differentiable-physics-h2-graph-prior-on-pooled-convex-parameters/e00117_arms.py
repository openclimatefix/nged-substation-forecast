"""The arms of experiment 117 on real data: fits, replica spreads, and quantile predictions.

An arm is a penalty (`studies.dp_gam_graph.GraphPenalty`) on the joint fit of one cut's nodes. G has
no penalty, N the fleet-mean penalty, H the graph penalty, NH both, and P-hol a fleet-mean penalty
on the holiday family only. All arms of a cut share the nodes, the designs, and the replica
designs, so the arms differ in the penalty alone.

**Capacity.** A decision site's capacity is pinned to the real-time vintage
(`studies.dp_pinned_capacity`), except where the rule falls back to a vintage fitted after the cut,
and an MVA node fits its capacity from dark rows (which carry none). A node whose points of
effective capacity were rebuilt (D9 on a partial window, and every MVA node) is scored in original
points after its forecast is divided by the rebuilt scale.

**Spread.** Each node takes the median-centred spread of its own residuals from the same arm fitted
12 months before the cut and predicted over the 12 months that follow.
"""

import os
from dataclasses import dataclass
from datetime import datetime

import e00117_common as common
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache
from studies.dp_gam import (
    ReplicaResiduals,
    ResidualSpread,
    predict_gam,
    replica_cut,
    replica_rows,
)
from studies.dp_gam_graph import (
    GraphPenalty,
    GraphSolution,
    NodeDesign,
    build_node_design,
    family_scales,
    gam_fits_from_solution,
    solve_graph_fit,
)


@dataclass(frozen=True)
class CutContext:
    """The nodes and designs of one cut and of its replica.

    Attributes:
        cache: The cache in original points.
        plan: The nodes of the cut.
        designs: Each node's design at the cut.
        replica_plan: The nodes of the replica cut, 12 months earlier.
        replica_designs: Each node's design at the replica cut.
    """

    cache: FeatureCache
    plan: common.NodePlan
    designs: list[NodeDesign]
    replica_plan: common.NodePlan
    replica_designs: list[NodeDesign]


def build_context(
    *,
    cache: FeatureCache,
    vintages: pl.DataFrame,
    cut: datetime,
    labels: tuple[str, ...] | None = None,
    allow_excluded: tuple[str, ...] = (),
) -> CutContext:
    """Build the nodes and designs of a cut and of its replica.

    Args:
        cache: The cache in original points.
        vintages: The real-time capacity vintages.
        cut: The cut.
        labels: The nodes, by default every node of the decision graph that joins the cut.
        allow_excluded: Excluded series that `labels` may name, for the reproduction check only.

    Returns:
        The context.
    """
    edges = None
    if labels is not None:
        edges = tuple(e for e in common.private_edges() if e[0] in labels and e[1] in labels)
    plan = common.prepare_nodes(
        cache=cache,
        vintages=vintages,
        cut=cut,
        labels=labels,
        edges=edges,
        allow_excluded=allow_excluded,
    )
    replica = common.prepare_nodes(
        cache=cache,
        vintages=vintages,
        cut=replica_cut(cut=cut),
        labels=labels,
        edges=edges,
        allow_excluded=allow_excluded,
    )
    return CutContext(
        cache=cache,
        plan=plan,
        designs=[build_node_design(cache=plan.cache, site=s, cut=cut) for s in plan.sites],
        replica_plan=replica,
        replica_designs=[
            build_node_design(cache=replica.cache, site=s, cut=replica.cut) for s in replica.sites
        ],
    )


def solve(
    *, designs: list[NodeDesign], plan: common.NodePlan, penalty: GraphPenalty
) -> GraphSolution:
    """Fit the joint model of a cut's nodes under a penalty (full sample only)."""
    return solve_graph_fit(
        designs=designs,
        graph=plan.graph,
        penalty=penalty,
        scales=family_scales(designs=designs),
        device=os.environ.get("E00117_DEVICE", "cpu"),
    )


def replica_spreads(*, context: CutContext, penalty: GraphPenalty) -> dict[str, ResidualSpread]:
    """Return each node's median-centred spread from the arm fitted 12 months before the cut.

    Args:
        context: The cut's context.
        penalty: The arm's penalty, used again for the replica fit.

    Returns:
        A spread per node of the replica, in original points.
    """
    cut = context.plan.cut
    fit_cut = context.replica_plan.cut
    solution = solve(designs=context.replica_designs, plan=context.replica_plan, penalty=penalty)
    fits = gam_fits_from_solution(
        cache=context.replica_plan.cache,
        designs=context.replica_designs,
        solution=solution,
        cut=fit_cut,
    )
    out: dict[str, ResidualSpread] = {}
    for design in context.replica_designs:
        site = design.site
        cache = context.replica_plan.cache
        index = replica_rows(cache=cache, site=site, fit_cut=fit_cut, cut=cut)
        predicted = predict_gam(cache=cache, fit=fits[site], index=index)
        residual = cache.sites[site].y[index] - predicted
        residual = residual / context.replica_plan.scale.get(site, 1.0)
        replica = ReplicaResiduals(
            residuals=residual,
            band=cache.calendar.band[index],
            quarter=cache.calendar.quarter[index],
            index=index,
            fit_cut=fit_cut,
            switching=cache.sites[site].switching_mask[index],
        )
        out[site] = ResidualSpread.fit(
            residuals=replica.residuals, band=replica.band, quarter=replica.quarter, centre=True
        )
    return out


def predict_arm(
    *, context: CutContext, name: str, penalty: GraphPenalty
) -> tuple[pl.DataFrame, pl.DataFrame, GraphSolution]:
    """Fit an arm at the cut and return its predictions for every node.

    Args:
        context: The cut's context.
        name: The arm's name, written into the frames.
        penalty: The arm's penalty.

    Returns:
        The predictions of the decision sites and D9, the predictions of the MVA nodes on dark
        rows (descriptive), and the fit's solution.
    """
    plan = context.plan
    cut = plan.cut
    solution = solve(designs=context.designs, plan=plan, penalty=penalty)
    fits = gam_fits_from_solution(
        cache=plan.cache, designs=context.designs, solution=solution, cut=cut
    )
    spreads = replica_spreads(context=context, penalty=penalty)
    sites_frames, dark_frames = [], []
    for design in context.designs:
        site = design.site
        if site not in spreads:
            continue
        units = context.cache.sites[site].table.units
        index = common.scoring_index(
            cache=context.cache,
            site=site,
            after=cut,
            through=common.year_end(cut=cut),
            cut=cut,
        )
        if units == "MVA":
            index = index[context.cache.sites[site].dark[index]]
        mean = predict_gam(cache=plan.cache, fit=fits[site], index=index)
        mean = mean / plan.scale.get(site, 1.0)
        quantiles = spreads[site].quantiles(
            mean=mean,
            band=context.cache.calendar.band[index],
            quarter=context.cache.calendar.quarter[index],
        )
        frame = common.prediction_frame(
            cache=context.cache,
            site=site,
            index=index,
            quantiles=quantiles,
            mean=mean,
            arm=name,
            cut=cut,
        )
        (dark_frames if units == "MVA" else sites_frames).append(frame)
    dark = pl.concat(dark_frames) if dark_frames else pl.DataFrame()
    return pl.concat(sites_frames), dark, solution


def score_by_site(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return each site's mean quantile score over the rows outside the switching events.

    Args:
        frame: Predictions from `predict_arm`.

    Returns:
        Columns `time_series_id`, `score`, and `n_rows`.
    """
    scores = []
    for series_id in frame["time_series_id"].unique().sort():
        part = frame.filter((pl.col("time_series_id") == series_id) & ~pl.col("switching_mask"))
        quantiles = part.select(common.QUANTILE_COLUMNS).to_numpy().astype(np.float64)
        row = common.quantile_score(
            actual=part["actual"].to_numpy().astype(np.float64), quantiles=quantiles
        )
        scores.append({"time_series_id": series_id, "score": float(row.mean()), "n_rows": len(row)})
    return pl.DataFrame(scores)
