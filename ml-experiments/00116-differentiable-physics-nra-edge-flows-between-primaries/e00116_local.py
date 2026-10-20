"""Fit the flows of the neighbourhood of one injected transfer, for experiment 116.

A hindsight fit of the whole graph over a year costs minutes, and the gate, the tuning, and C1
need hundreds of injected transfers. A transfer changes the residuals of its two ends only, so this
module fits the paths that touch those two ends on a window around the transfer, with every other
path set to zero. The window starts 3 days before the transfer and ends 3 days after the
transfer's 48-hour post-event rows. The transfer's true flow is added to the baseline residuals of
the two ends.
"""

from dataclasses import dataclass
from typing import Final

import e00116_data as data_module
import e00116_inject as inject
import numpy as np
from studies.dp_edge_flow import FlowFit, PathSpec, edge_name
from studies.dp_features import FeatureCache

PAD_HALF_HOURS: Final[int] = 3 * 48
"""Rows before the transfer, and after its post-event rows, that the window includes: 3 days."""


@dataclass(frozen=True)
class LocalWindow:
    """One transfer's window and what was fitted in it.

    Attributes:
        transfer: The transfer's row.
        first: The first grid position of the window.
        stop: One past the last grid position.
        true_flow: The injected flow on the transfer's edge over the window, positive onto the
            edge's first node.
        fit: The fit of the neighbourhood's paths over the window.
        edge: The edge's path name.
    """

    transfer: dict
    first: int
    stop: int
    true_flow: np.ndarray
    fit: FlowFit
    edge: str

    @property
    def onset(self) -> int:
        """Return the transfer's onset as a row of the window."""
        return int(self.transfer["onset"]) - self.first

    @property
    def end(self) -> int:
        """Return the first row after the transfer as a row of the window."""
        return int(self.transfer["end"]) - self.first

    def fitted_edge(self) -> np.ndarray:
        """Return the fitted flow on the transfer's edge over the window."""
        return self.fit.flows[self.edge]


def neighbourhood(*, paths: tuple[PathSpec, ...], nodes: tuple[str, str]) -> tuple[PathSpec, ...]:
    """Return the paths that touch either of two nodes."""
    return tuple(path for path in paths if path.node_a in nodes or path.node_b in nodes)


def injected_residuals(
    *,
    cache: FeatureCache,
    data: data_module.YearData,
    transfer: dict,
    first: int,
    stop: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return the residual streams over a window with the transfer added, and its true edge flow.

    Args:
        cache: The feature cache, for the measured powers that shape the transfer.
        data: The year's residuals.
        transfer: One row of the transfers table.
        first: The first grid position of the window.
        stop: One past the last grid position of the window.

    Returns:
        Each site's residual over the window in megawatts with the transfer added at its two ends,
        and the true flow on the edge over the window.
    """
    onset, end = int(transfer["onset"]), int(transfer["end"])
    scale = {node: data.capacity_mw[node] / data_module.POINTS_PER_CAPACITY for node in data.sites}
    power = {
        node: cache.sites[node].y[onset:end] * scale[node]
        for node in (transfer["donor"], transfer["receiver"])
    }
    moved = inject.moved_mw(
        transfer=transfer,
        donor_power_mw=power[transfer["donor"]],
        receiver_power_mw=power[transfer["receiver"]],
    )
    residuals = {node: r[first:stop].copy() for node, r in data.residual_mw.items()}
    residuals[transfer["receiver"]][onset - first : end - first] += moved
    residuals[transfer["donor"]][onset - first : end - first] -= moved
    flow = inject.edge_flow(
        transfer={**transfer, "onset": onset - first, "end": end - first},
        moved=moved,
        length=stop - first,
    )
    return residuals, flow


def fit_local(
    *,
    cache: FeatureCache,
    data: data_module.YearData,
    paths: tuple[PathSpec, ...],
    config: data_module.FlowConfig,
    transfer: dict,
    inject_transfer: bool = True,
) -> LocalWindow:
    """Fit the paths at a transfer's two ends over its window.

    Args:
        cache: The feature cache.
        data: The year's residuals.
        paths: The graph's paths, built with the config's penalties.
        config: The flow settings.
        transfer: One row of the transfers table.
        inject_transfer: False to fit the same window on the uninjected residuals, as a baseline.

    Returns:
        The window with the fitted flows of the neighbourhood.
    """
    first = max(int(transfer["onset"]) - PAD_HALF_HOURS, data.first)
    stop = min(int(transfer["end"]) + inject.POST_HALF_HOURS + PAD_HALF_HOURS, data.stop)
    residuals, true_flow = injected_residuals(
        cache=cache, data=data, transfer=transfer, first=first, stop=stop
    )
    if not inject_transfer:
        residuals = {node: r[first:stop].copy() for node, r in data.residual_mw.items()}
        true_flow = np.zeros(stop - first)
    local_paths = neighbourhood(paths=paths, nodes=(transfer["node_a"], transfer["node_b"]))
    fit = data_module.fit_year_flows(
        paths=local_paths,
        residuals=residuals,
        noise=data.noise_mw(block=config.block),
        config=config,
    )
    return LocalWindow(
        transfer=transfer,
        first=first,
        stop=stop,
        true_flow=true_flow,
        fit=fit,
        edge=edge_name(node_a=transfer["node_a"], node_b=transfer["node_b"]),
    )


def fit_window(
    *,
    data: data_module.YearData,
    paths: tuple[PathSpec, ...],
    config: data_module.FlowConfig,
    residuals: dict[str, np.ndarray],
    nodes: tuple[str, str],
) -> FlowFit:
    """Fit the paths that touch two nodes on residuals given over one window.

    Args:
        data: The year's data, for the noise scales.
        paths: The graph's paths, possibly with a placebo path added.
        config: The flow settings.
        residuals: Each node's residual over the window in megawatts.
        nodes: The two nodes whose neighbourhood is fitted.

    Returns:
        The fit of the neighbourhood's paths.
    """
    return data_module.fit_year_flows(
        paths=neighbourhood(paths=paths, nodes=nodes),
        residuals=residuals,
        noise=data.noise_mw(block=config.block),
        config=config,
    )


def placebo_tiles(*, data: data_module.YearData) -> list[tuple[int, int]]:
    """Return non-overlapping windows that tile the scored year, for the placebo check."""
    length = 6 * PAD_HALF_HOURS
    return [
        (first, min(first + length, data.stop)) for first in range(data.first, data.stop, length)
    ]


def placebo_mw_days(
    *,
    data: data_module.YearData,
    paths: tuple[PathSpec, ...],
    config: data_module.FlowConfig,
    pair: tuple[str, str],
) -> float:
    """Return the megawatt-days of flow a fit puts on a pair with no interconnection.

    The placebo path joins the pair and is fitted beside the real paths at the pair's two nodes,
    tile by tile over the year. It never enters the decision graph.

    Args:
        data: The year's data.
        paths: The decision graph's paths.
        config: The flow settings.
        pair: Two nodes with no interconnection.

    Returns:
        The sum of the placebo path's absolute flow over the year, in megawatt-days.
    """
    node_a, node_b = sorted(pair)
    scale = data.edge_scale_mw(node_a=node_a, node_b=node_b, block=config.block)
    placebo = PathSpec(
        name=f"placebo:{node_a}-{node_b}",
        node_a=node_a,
        node_b=node_b,
        step_penalty=config.step_c / scale,
        level_penalty=config.level_c / scale,
    )
    total = 0.0
    for first, stop in placebo_tiles(data=data):
        residuals = {node: r[first:stop].copy() for node, r in data.residual_mw.items()}
        fit = fit_window(
            data=data, paths=(*paths, placebo), config=config, residuals=residuals, nodes=pair
        )
        total += float(np.sum(np.abs(fit.flows[placebo.name]))) / 48.0
    return total


def injected_streams(
    *, cache: FeatureCache, data: data_module.YearData, transfer: dict
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return every site's residual stream on the whole grid with a transfer added.

    Args:
        cache: The feature cache.
        data: The year's residuals.
        transfer: One row of the transfers table.

    Returns:
        The streams in megawatts, and the injected megawatts at each of the two ends (positive at
        the receiver, negative at the donor, zero elsewhere), also on the whole grid.
    """
    streams = {node: r.copy() for node, r in data.residual_mw.items()}
    onset, end = int(transfer["onset"]), int(transfer["end"])
    scale = {node: data.capacity_mw[node] / data_module.POINTS_PER_CAPACITY for node in data.sites}
    power = {
        node: cache.sites[node].y[onset:end] * scale[node]
        for node in (transfer["donor"], transfer["receiver"])
    }
    moved = inject.moved_mw(
        transfer=transfer,
        donor_power_mw=power[transfer["donor"]],
        receiver_power_mw=power[transfer["receiver"]],
    )
    added = {node: np.zeros(len(next(iter(streams.values())))) for node in streams}
    added[transfer["receiver"]][onset:end] = moved
    added[transfer["donor"]][onset:end] = -moved
    for node in (transfer["receiver"], transfer["donor"]):
        streams[node] = streams[node] + added[node]
    return streams, added
