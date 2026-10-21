"""The causal pipelines of experiment 116: plain filter (A0, A4), causal flows (A5), and the oracle.

At an origin `tau`, a forecast may see only readings that ended 3.5 hours before `tau` or earlier.
**A5** refits the flows of the paths at the scored nodes on the trailing 30 days of residuals,
removes each node's flow from its residual, runs the Kalman filter over those 30 days from the
state the previous origin's run held at the window's start, and forecasts the mean as

    GAM + level + phi**h * anomaly + f_now * S(a + h) / S(a)

where `f_now` is the node's newest flow, `a` the age of its current run, `h` the rows ahead, and
`S` the Kaplan-Meier survival of the hindsight flows' run lengths. **The plain filter** is the same
run with every flow forced to zero, so it equals `studies.dp_assimilation`'s stream filter.
**The oracle** removes the true injected flow and adds back the true flow while the transfer lasts.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import e00116_common as common
import e00116_data as data_module
import e00116_gam as gam
import numpy as np
import polars as pl
from studies.dp_assimilation import (
    DEFAULT_DELAY_HOURS,
    FilterHyperparameters,
    correction_at,
    fit_filter_hyperparameters,
)
from studies.dp_edge_flow import (
    FilterCarry,
    PathSpec,
    SurvivalCurve,
    carry_at,
    causal_window_fit,
    fit_survival,
    node_flow,
    run_known_input_filter,
    run_lengths,
    runs_to_trace,
)
from studies.dp_features import FeatureCache

WINDOW_HALF_HOURS: Final[int] = 30 * 48
"""The causal window: the trailing 30 days."""
HORIZON_HALF_HOURS: Final[int] = 96
"""Forecast rows per origin: 0.5 to 48 hours ahead."""
DELAY_HALF_HOURS: Final[int] = round(DEFAULT_DELAY_HOURS * 2)
"""Rows between the newest usable reading and the origin."""
ACTIVE_FLOW_MW: Final[float] = 0.1
"""A flow is part of a run when its magnitude is over this."""
HYPER_PATH: Final[Path] = common.WORK_DIR / "filter_hyper.json"


def fit_hyper(*, cache: FeatureCache, arm: gam.ArmType) -> dict[str, FilterHyperparameters]:
    """Fit every site's filter hyperparameters on the tuning year's out-of-sample residuals.

    Args:
        cache: The feature cache.
        arm: The base arm whose residuals are filtered.

    Returns:
        Each site's hyperparameters, written to `HYPER_PATH` on the first call.
    """
    stored = json.loads(HYPER_PATH.read_text()) if HYPER_PATH.exists() else {}
    if arm in stored:
        return {
            site: FilterHyperparameters.from_array(values=np.array(values))
            for site, values in stored[arm].items()
        }
    data = data_module.load_year(cache=cache, cut_name="tuning", arm=arm)
    streams = {}
    for site in data.sites:
        scale = data.capacity_mw[site] / data_module.POINTS_PER_CAPACITY
        stream = np.full(len(cache.end_times), np.nan)
        stream[data.first : data.stop] = data.residual_mw[site][data.first : data.stop] / scale
        streams[site] = stream
    fitted = fit_filter_hyperparameters(
        residuals=streams,
        band=cache.calendar.band,
        end_times=cache.end_times,
        window=(cache.end_times[data.first - 1], cache.end_times[data.stop - 1]),
    )
    stored[arm] = {site: hyper.as_array().tolist() for site, hyper in fitted.items()}
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    HYPER_PATH.write_text(json.dumps(stored, indent=2))
    return fitted


def survival_from_flows(*, flows: dict[str, np.ndarray]) -> SurvivalCurve:
    """Fit the Kaplan-Meier survival of the run lengths of the edge flows, pooled over edges.

    Args:
        flows: Hindsight flows by path name; the `position` column and outside paths are skipped.

    Returns:
        The survival curve in rows.
    """
    lengths, censored = [], []
    for name, flow in flows.items():
        if name == "position" or name.startswith("outside"):
            continue
        run, cut_short = run_lengths(flow=flow, threshold=ACTIVE_FLOW_MW)
        lengths.append(run)
        censored.append(cut_short)
    return fit_survival(lengths=np.concatenate(lengths), censored=np.concatenate(censored))


@dataclass(frozen=True)
class Context:
    """What the causal pipelines share for one year and one base arm.

    Attributes:
        cache: The feature cache.
        data: The year's data for the base arm.
        paths: The graph's paths.
        noise: Each node's robust residual scale in megawatts, from pre-cut rows.
        hyper: Each node's filter hyperparameters.
        survival: The run-length survival curve.
        mean_points: Each node's frozen-fit mean in points, on the whole grid (`NaN` elsewhere).
        scale: Each node's megawatts per point.
        huber_sigmas: The Huber threshold of the flow fit, in noise scales.
    """

    cache: FeatureCache
    data: data_module.YearData
    paths: tuple[PathSpec, ...]
    noise: dict[str, float]
    hyper: dict[str, FilterHyperparameters]
    survival: SurvivalCurve
    mean_points: dict[str, np.ndarray]
    scale: dict[str, float]
    huber_sigmas: float

    def local_paths(self, *, nodes: tuple[str, ...]) -> tuple[PathSpec, ...]:
        """Return the paths that touch any of the nodes."""
        return tuple(p for p in self.paths if p.node_a in nodes or p.node_b in nodes)


@dataclass(frozen=True)
class OriginForecast:
    """One origin's corrections at one node, over the 96 forecast rows.

    Attributes:
        cutoff: The newest usable grid position.
        target: The grid position of each forecast row.
        level: The filter's persistent part in points.
        anomaly: The filter's decayed transient part in points.
        flow: The flow term in points.
    """

    cutoff: int
    target: np.ndarray
    level: np.ndarray
    anomaly: np.ndarray
    flow: np.ndarray

    @property
    def correction(self) -> np.ndarray:
        """Return the total correction to the frozen mean, in points."""
        return self.level + self.anomaly + self.flow


def burn_in_first(*, cutoffs: list[int]) -> int:
    """Return where every pipeline's filter starts cold: 30 days before the first window.

    All pipelines start at the same row, so a pipeline with every flow set to zero gives the same
    corrections as the plain filter.
    """
    return cutoffs[0] - 2 * WINDOW_HALF_HOURS + 1


def targets_of(*, cutoff: int) -> np.ndarray:
    """Return the grid positions of the 96 rows an origin forecasts."""
    return cutoff + DELAY_HALF_HOURS + 1 + np.arange(HORIZON_HALF_HOURS)


def _corrections(
    *, runs: tuple, hyper: FilterHyperparameters, cutoff: int, band: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the level, anomaly, and target positions of one origin from a filter run."""
    target = targets_of(cutoff=cutoff)
    found = correction_at(
        trace=runs_to_trace(runs=runs, hyper=hyper),
        cutoff=np.full(len(target), cutoff),
        target=target,
        band=band[target],
    )
    return found.level, found.anomaly, target


def plain_forecasts(
    *, context: Context, streams_mw: dict[str, np.ndarray], node: str, cutoffs: list[int]
) -> list[OriginForecast]:
    """Return the plain filter's forecasts at several origins of one node.

    The filter runs once over the span from the first origin's window to the last origin, which
    gives the same states as running it origin by origin from a carried state.

    Args:
        context: The year's shared objects.
        streams_mw: Each node's residual stream on the grid in megawatts.
        node: The node.
        cutoffs: The origins' newest usable positions, ascending.

    Returns:
        One forecast per origin, with a zero flow term.
    """
    scale = context.scale[node]
    points = streams_mw[node] / scale
    first, last = burn_in_first(cutoffs=cutoffs), cutoffs[-1]
    runs = run_known_input_filter(
        residual=points,
        flow=np.zeros(last - first + 1),
        band=context.cache.calendar.band,
        first=first,
        last=last,
        hyper=context.hyper[node],
        carry=None,
    )
    out = []
    for cutoff in cutoffs:
        level, anomaly, target = _corrections(
            runs=runs, hyper=context.hyper[node], cutoff=cutoff, band=context.cache.calendar.band
        )
        out.append(
            OriginForecast(
                cutoff=cutoff,
                target=target,
                level=level,
                anomaly=anomaly,
                flow=np.zeros_like(level),
            )
        )
    return out


def oracle_forecasts(
    *,
    context: Context,
    streams_mw: dict[str, np.ndarray],
    node: str,
    true_node_flow_mw: np.ndarray,
    cutoffs: list[int],
) -> list[OriginForecast]:
    """Return the oracle's forecasts: the true flow is removed and added back while it lasts.

    Args:
        context: The year's shared objects.
        streams_mw: Each node's residual stream on the grid in megawatts, with the transfer.
        node: The node.
        true_node_flow_mw: The node's true injected flow on the whole grid in megawatts.
        cutoffs: The origins' newest usable positions, ascending.

    Returns:
        One forecast per origin, whose flow term is the true flow at each target.
    """
    scale = context.scale[node]
    first, last = burn_in_first(cutoffs=cutoffs), cutoffs[-1]
    runs = run_known_input_filter(
        residual=streams_mw[node] / scale,
        flow=true_node_flow_mw[first : last + 1] / scale,
        band=context.cache.calendar.band,
        first=first,
        last=last,
        hyper=context.hyper[node],
        carry=None,
    )
    out = []
    for cutoff in cutoffs:
        level, anomaly, target = _corrections(
            runs=runs, hyper=context.hyper[node], cutoff=cutoff, band=context.cache.calendar.band
        )
        flow = true_node_flow_mw[target] / scale
        out.append(
            OriginForecast(cutoff=cutoff, target=target, level=level, anomaly=anomaly, flow=flow)
        )
    return out


def _warm_start(
    *, previous: dict[str, np.ndarray] | None, shift: int
) -> dict[str, np.ndarray] | None:
    """Shift the previous window's flows left by `shift` rows, holding the last value."""
    if previous is None or shift >= len(next(iter(previous.values()))):
        return None
    return {
        name: np.concatenate([flow[shift:], np.full(shift, flow[-1])])
        for name, flow in previous.items()
    }


def causal_forecasts(
    *,
    context: Context,
    streams_mw: dict[str, np.ndarray],
    nodes: tuple[str, ...],
    cutoffs: list[int],
) -> dict[str, list[OriginForecast]]:
    """Return A5's forecasts at several origins for several nodes.

    Each origin refits the paths at the scored nodes on its trailing window, then runs each node's
    filter over the window with the node's flow taken out, from the state the previous origin
    held at the window's start. The first origin starts from the plain filter's state there.

    Args:
        context: The year's shared objects.
        streams_mw: Each node's residual stream on the grid in megawatts.
        nodes: The nodes forecast.
        cutoffs: The origins' newest usable positions, ascending.

    Returns:
        For each node, one forecast per origin.
    """
    paths = context.local_paths(nodes=nodes)
    # Only edge flows are taken out and added back; an outside path's level also absorbs growth and
    # new connections, so it stays in the residual for the filter, as in the A4 refit.
    edge_paths = tuple(path for path in paths if path.node_b is not None)
    band = context.cache.calendar.band
    points = {node: streams_mw[node] / context.scale[node] for node in nodes}
    graph_nodes = {p.node_a for p in paths} | {p.node_b for p in paths if p.node_b}
    window_streams = {node: streams_mw[node] for node in graph_nodes}
    results: dict[str, list[OriginForecast]] = {node: [] for node in nodes}
    carry: dict[str, FilterCarry | None] = {}
    for node in nodes:
        first_window = cutoffs[0] - WINDOW_HALF_HOURS + 1
        base = run_known_input_filter(
            residual=points[node],
            flow=np.zeros(WINDOW_HALF_HOURS),
            band=band,
            first=first_window - WINDOW_HALF_HOURS,
            last=first_window - 1,
            hyper=context.hyper[node],
            carry=None,
        )
        carry[node] = carry_at(runs=base, position=first_window - 1, fallback=None)
    previous: dict[str, np.ndarray] | None = None
    previous_cutoff = cutoffs[0]
    for step, cutoff in enumerate(cutoffs):
        fit, first = causal_window_fit(
            paths=paths,
            residuals=window_streams,
            noise=context.noise,
            cutoff=cutoff,
            window_length=WINDOW_HALF_HOURS,
            start=_warm_start(previous=previous, shift=cutoff - previous_cutoff),
            max_sweeps=data_module.MAX_SWEEPS,
            sweep_tolerance=data_module.SWEEP_TOLERANCE,
            huber_sigmas=context.huber_sigmas,
        )
        previous, previous_cutoff = fit.flows, cutoff
        for node in nodes:
            scale = context.scale[node]
            flow_mw = node_flow(paths=edge_paths, flows=fit.flows, node=node)
            runs = run_known_input_filter(
                residual=points[node],
                flow=flow_mw / scale,
                band=band,
                first=first,
                last=cutoff,
                hyper=context.hyper[node],
                carry=carry[node],
            )
            level, anomaly, target = _corrections(
                runs=runs, hyper=context.hyper[node], cutoff=cutoff, band=band
            )
            age = _current_age(flow_mw=flow_mw)
            now = flow_mw[-1] / scale if age > 0 else 0.0
            held = context.survival.held_share(age=age, ahead=target - cutoff)
            results[node].append(
                OriginForecast(
                    cutoff=cutoff, target=target, level=level, anomaly=anomaly, flow=now * held
                )
            )
            if step + 1 < len(cutoffs):
                carry[node] = carry_at(
                    runs=runs,
                    position=cutoffs[step + 1] - WINDOW_HALF_HOURS,
                    fallback=carry[node],
                )
    return results


def _current_age(*, flow_mw: np.ndarray) -> int:
    """Return how many rows the flow has been active, counting back from its last row."""
    active = np.abs(flow_mw) > ACTIVE_FLOW_MW
    if not active[-1]:
        return 0
    inactive = np.flatnonzero(~active)
    return len(active) - 1 - int(inactive[-1]) if len(inactive) else len(active)


def build_context(
    *,
    cache: FeatureCache,
    cut_name: str,
    arm: gam.ArmType,
    config: data_module.FlowConfig,
    survival: SurvivalCurve,
    hyper_arm: gam.ArmType = "a0",
) -> Context:
    """Build the shared objects of one year and base arm.

    Args:
        cache: The feature cache.
        cut_name: `tuning`, `replicate`, or `scored`.
        arm: The base arm whose frozen fit supplies the mean and the residuals.
        config: The flow settings.
        survival: The run-length survival curve.
        hyper_arm: The arm whose tuning-year residuals fit the filter hyperparameters.

    Returns:
        The context.

    Raises:
        ValueError: If `config.block` is not 1, because the causal path fits half-hourly rows.
    """
    if config.block != 1:
        raise ValueError(f"The causal path fits half-hourly rows, not blocks of {config.block}.")
    data = data_module.load_year(cache=cache, cut_name=cut_name, arm=arm)
    edges = data_module.decision_graph(data=data)
    paths = data_module.build_graph_paths(data=data, edges=edges, config=config)
    mean_points = {}
    for site in data.sites:
        frame = gam_frame(arm=arm, site=site, cut=data.cut)
        stream = np.full(len(cache.end_times), np.nan)
        stream[frame["position"].to_numpy()] = frame["mean"].to_numpy()
        mean_points[site] = stream
    return Context(
        cache=cache,
        data=data,
        paths=paths,
        noise=data.noise_mw(block=config.block),
        hyper=fit_hyper(cache=cache, arm=hyper_arm),
        survival=survival,
        mean_points=mean_points,
        scale={s: data.capacity_mw[s] / data_module.POINTS_PER_CAPACITY for s in data.sites},
        huber_sigmas=config.huber_sigmas,
    )


def gam_frame(*, arm: gam.ArmType, site: str, cut: datetime) -> pl.DataFrame:
    """Read an arm's stored fit of a site at a cut."""
    return pl.read_parquet(gam.job_path(arm=arm, site=site, cut=cut))
