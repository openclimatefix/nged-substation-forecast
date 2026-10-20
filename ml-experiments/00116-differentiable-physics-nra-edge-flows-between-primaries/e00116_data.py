"""Residual streams, noise scales, and the decision graph for experiment 116.

A year's data are the residuals (observed minus forecast) of a frozen fit at the year's cut, in
megawatts, on the whole grid with `NaN` where there is no reading. The rows before the cut are the
fit's own training window and are used only for noise scales; the rows after it are out of sample.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Final

import e00116_common as common
import e00116_gam as gam
import numpy as np
import polars as pl
from studies.dp_edge_flow import (
    FlowFit,
    PathSpec,
    block_mean_residuals,
    build_paths,
    decision_edges,
    edge_name,
    expand_blocks,
    fit_flows,
    robust_residual_scale,
)
from studies.dp_features import FeatureCache

SWEEP_TOLERANCE: Final[float] = 1e-3
"""Block-coordinate sweeps stop when no path moves by more than 1 kW and the objective's relative
change is under this; the residuals' scale is 0.2 to 0.8 MW."""
MAX_SWEEPS: Final[int] = 12
BLOCK_HALF_HOURS: Final[int] = 6
"""The 3-hour block of arm A2-b."""
POINTS_PER_CAPACITY: Final[float] = 100.0
"""A point is one percent of a substation's effective capacity."""


@dataclass(frozen=True)
class YearData:
    """One cut's residual streams.

    Attributes:
        cut_name: `tuning`, `replicate`, or `scored`.
        cut: The cut.
        arm: The convex-model arm (`a0` or `a1`).
        sites: The substations with a fit at this cut, in name order.
        capacity_mw: Each site's effective capacity in megawatts.
        residual_mw: Each site's residual on the whole grid in megawatts.
        cut_position: The grid position of the last row at or before the cut.
        first: The first grid position of the scored year.
        stop: One past the last grid position of the scored year.
    """

    cut_name: str
    cut: datetime
    arm: gam.ArmType
    sites: tuple[str, ...]
    capacity_mw: dict[str, float]
    residual_mw: dict[str, np.ndarray]
    cut_position: int
    first: int
    stop: int

    def year(self) -> dict[str, np.ndarray]:
        """Return each site's residual over the scored year only."""
        return {site: r[self.first : self.stop] for site, r in self.residual_mw.items()}

    def noise_mw(self, *, block: int = 1) -> dict[str, float]:
        """Return each site's robust residual scale from rows at or before the cut.

        Args:
            block: Rows per block; a scale above 1 is the scale of the block means.

        Returns:
            The scale of each site in megawatts.
        """
        return {
            site: robust_residual_scale(
                residual=_blocked(stream=r[: self.cut_position + 1], block=block)
            )
            for site, r in self.residual_mw.items()
        }

    def edge_scale_mw(self, *, node_a: str, node_b: str, block: int = 1) -> float:
        """Return the robust scale of `(r_a - r_b) / 2` on rows at or before the cut.

        Args:
            node_a: One end of the edge.
            node_b: The other end.
            block: Rows per block; a scale above 1 is the scale of the block means.

        Returns:
            The scale in megawatts.
        """
        target = 0.5 * (self.residual_mw[node_a] - self.residual_mw[node_b])
        return robust_residual_scale(
            residual=_blocked(stream=target[: self.cut_position + 1], block=block)
        )


def _blocked(*, stream: np.ndarray, block: int) -> np.ndarray:
    """Return a stream's block means, or the stream itself for a block of 1."""
    if block == 1:
        return stream
    means, _ = block_mean_residuals(residuals={"x": stream}, block_size=block)
    return means["x"]


def load_year(*, cache: FeatureCache, cut_name: str, arm: gam.ArmType) -> YearData:
    """Read the stored fits of an arm at a cut and build the residual streams.

    Args:
        cache: The feature cache (for the grid and the effective capacities).
        cut_name: `tuning`, `replicate`, or `scored`.
        arm: `a0` or `a1`.

    Returns:
        The year's data, with the sites that had enough history at the cut.
    """
    cut = common.CUTS[cut_name]
    year = common.year_slice(cache=cache, cut=cut)
    residuals: dict[str, np.ndarray] = {}
    capacity: dict[str, float] = {}
    for site in gam.SITES:
        path = gam.job_path(arm=arm, site=site, cut=cut)
        if not path.exists():
            continue
        frame = pl.read_parquet(path)
        stream = np.full(len(cache.end_times), np.nan)
        scale = float(cache.sites[site].table.effective_capacity_mw) / POINTS_PER_CAPACITY
        stream[frame["position"].to_numpy()] = (frame["y"] - frame["mean"]).to_numpy() * scale
        residuals[site] = stream
        capacity[site] = float(cache.sites[site].table.effective_capacity_mw)
    return YearData(
        cut_name=cut_name,
        cut=cut,
        arm=arm,
        sites=tuple(sorted(residuals)),
        capacity_mw=capacity,
        residual_mw=residuals,
        cut_position=year.start - 1,
        first=year.start,
        stop=year.stop,
    )


def decision_graph(*, data: YearData) -> tuple[tuple[str, str], ...]:
    """Return the decision edges among a year's sites, read from the private interconnections.

    The edges are the metered pairs that touch neither series labelled G-c. The count is printed
    by the caller before any residual is read.
    """
    return decision_edges(
        pairs=common.private_edges(),
        metered=set(data.sites),
        excluded=set(common.GENERATOR_SERIES),
    )


@dataclass(frozen=True)
class FlowConfig:
    """The tuned settings of the flow fit.

    Attributes:
        step_c: The step penalty is `step_c / sigma_e` for each edge.
        level_c: The level penalty is `level_c / sigma_e`.
        outside_factor: The multiple of the edge penalties charged on an outside path.
        block: Rows per block (1 for half-hourly flows, 6 for the 3-hour blocks of A2-b).
    """

    step_c: float
    level_c: float
    outside_factor: float = 1.0
    block: int = 1

    def label(self) -> str:
        """Return a short label for file names."""
        return f"c{self.step_c:g}_l{self.level_c:g}_r{self.outside_factor:g}_b{self.block}"


def build_graph_paths(
    *,
    data: YearData,
    edges: tuple[tuple[str, str], ...],
    config: FlowConfig,
    extra_nodes: tuple[str, ...] = (),
) -> tuple[PathSpec, ...]:
    """Build the edge and outside paths of a graph with the scales and penalties of a config."""
    nodes = sorted({node for edge in edges for node in edge} | set(extra_nodes))
    scale = {
        edge_name(node_a=a, node_b=b): data.edge_scale_mw(node_a=a, node_b=b, block=config.block)
        for a, b in edges
    }
    return build_paths(
        edges=list(edges),
        nodes=nodes,
        step_penalty_by_edge={name: config.step_c / s for name, s in scale.items()},
        level_penalty_by_edge={name: config.level_c / s for name, s in scale.items()},
        outside_factor=config.outside_factor,
    )


def fit_year_flows(
    *,
    paths: tuple[PathSpec, ...],
    residuals: dict[str, np.ndarray],
    noise: dict[str, float],
    config: FlowConfig,
    start: dict[str, np.ndarray] | None = None,
) -> FlowFit:
    """Fit the flows of a graph on residuals given on one grid, half-hourly or on 3-hour blocks.

    Args:
        paths: The graph's paths.
        residuals: Each node's residual on one grid in megawatts.
        noise: Each node's robust scale in megawatts, of the block means when `config.block > 1`.
        config: The settings; `config.block > 1` fits on block means.
        start: Optional half-hourly warm start for each path.

    Returns:
        The fit, with flows on the half-hourly grid.
    """
    nodes = {path.node_a for path in paths} | {path.node_b for path in paths if path.node_b}
    chosen = {node: residuals[node] for node in nodes}
    if config.block == 1:
        return fit_flows(
            paths=paths,
            residuals=chosen,
            noise=noise,
            start=start,
            max_sweeps=MAX_SWEEPS,
            sweep_tolerance=SWEEP_TOLERANCE,
        )
    length = len(next(iter(chosen.values())))
    means, reliability = block_mean_residuals(residuals=chosen, block_size=config.block)
    block_start = None
    if start is not None:
        block_start = {name: flow[:: config.block] for name, flow in start.items()}
    fit = fit_flows(
        paths=paths,
        residuals=means,
        noise=noise,
        reliability=reliability,
        start=block_start,
        max_sweeps=MAX_SWEEPS,
        sweep_tolerance=SWEEP_TOLERANCE,
    )
    flows = {
        name: expand_blocks(block_values=flow, block_size=config.block, length=length)
        for name, flow in fit.flows.items()
    }
    return FlowFit(flows=flows, sweeps=fit.sweeps, converged=fit.converged, objective=fit.objective)
