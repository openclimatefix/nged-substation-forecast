"""Fit the flows of the neighbourhood of one injected transfer, for experiment 131.

A hindsight fit of the whole graph over a year costs minutes, and the gate, the tuning, and C1
need hundreds of injected transfers. A transfer changes the residuals of its two ends only, so this
module fits the paths that touch those two ends on a window around the transfer, with every other
path set to zero. The window starts 7 days before the transfer and ends 7 days after the
transfer's 48-hour post-event rows. The transfer's true flow is added to the baseline residuals of
the two ends.
"""

from dataclasses import dataclass
from typing import Final

import e00131_data as data_module
import e00131_inject as inject
import numpy as np
from studies.dp_edge_flow import FlowFit, PathSpec, edge_name
from studies.dp_features import FeatureCache

PAD_HALF_HOURS: Final[int] = 7 * 48
"""Rows before the transfer, and after its post-event rows, that the window includes: 7 days."""
PLACEBO_TILE_HALF_HOURS: Final[int] = 72 * 48
"""A placebo tile is 72 days, long enough for a drift term to matter."""
SCREEN_TILES: Final[int] = 4
"""The tuning screen fits this many evenly spaced placebo tiles."""


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


def drift_for(
    *, data: data_module.YearData, paths: tuple[PathSpec, ...], config: data_module.FlowConfig
) -> dict[str, float] | None:
    """Return the drift variance of every node the paths touch, or `None` for no drift term."""
    nodes = {path.node_a for path in paths} | {path.node_b for path in paths if path.node_b}
    return data_module.drift_variances(data=data, nodes=nodes, drift_q=config.drift_q)


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
        drift_step_variance=drift_for(data=data, paths=local_paths, config=config),
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
    local_paths = neighbourhood(paths=paths, nodes=nodes)
    return data_module.fit_year_flows(
        paths=local_paths,
        residuals=residuals,
        noise=data.noise_mw(block=config.block),
        config=config,
        drift_step_variance=drift_for(data=data, paths=local_paths, config=config),
    )


def placebo_tiles(*, data: data_module.YearData) -> list[tuple[int, int]]:
    """Return non-overlapping 72-day windows that tile the scored year, for the placebo check."""
    return [
        (first, min(first + PLACEBO_TILE_HALF_HOURS, data.stop))
        for first in range(data.first, data.stop, PLACEBO_TILE_HALF_HOURS)
    ]


def screen_tiles(*, data: data_module.YearData) -> list[tuple[int, int]]:
    """Return `SCREEN_TILES` evenly spaced whole tiles of the year, for the tuning screen."""
    whole = [
        tile for tile in placebo_tiles(data=data) if tile[1] - tile[0] == PLACEBO_TILE_HALF_HOURS
    ]
    picks = np.linspace(0, len(whole) - 1, SCREEN_TILES).round().astype(int)
    return [whole[i] for i in picks]


def placebo_mw_days(
    *,
    data: data_module.YearData,
    paths: tuple[PathSpec, ...],
    config: data_module.FlowConfig,
    pair: tuple[str, str],
    placebo_first: bool,
    tiles: list[tuple[int, int]] | None = None,
) -> float:
    """Return the megawatt-days of flow a fit puts on a pair with no interconnection, per year.

    The placebo path joins the pair and is fitted beside the real paths at the pair's two nodes,
    tile by tile. It never enters the decision graph. **The path order matters when the level
    penalty cannot break a tie, so the caller reads the placebo with the placebo path first and
    last, and the larger flow counts.**

    Args:
        data: The year's data.
        paths: The decision graph's paths.
        config: The flow settings.
        pair: Two nodes with no interconnection.
        placebo_first: Whether the placebo path is listed before the real paths.
        tiles: The windows to fit; every tile of the year if `None`.

    Returns:
        The sum of the placebo path's absolute flow over the tiles in megawatt-days, scaled by the
        year's rows over the tiles' rows to a full year.
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
    ordered = (placebo, *paths) if placebo_first else (*paths, placebo)
    tiles = placebo_tiles(data=data) if tiles is None else tiles
    total = 0.0
    for first, stop in tiles:
        residuals = {node: r[first:stop].copy() for node, r in data.residual_mw.items()}
        fit = fit_window(data=data, paths=ordered, config=config, residuals=residuals, nodes=pair)
        total += float(np.sum(np.abs(fit.flows[placebo.name]))) / 48.0
    covered = sum(stop - first for first, stop in tiles)
    return total * (data.stop - data.first) / covered


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
    streams = {node: r.copy() for node, r in data.out_of_sample().items()}
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
