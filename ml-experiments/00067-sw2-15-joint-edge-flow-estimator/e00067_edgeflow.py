"""The joint edge-flow estimator: one convex fit of block-shaped flows across every boundary.

The residual of every Primary substation is averaged to hourly bins and divided by its noise
scale. A window of hourly residuals is explained as a flow across each boundary between neighbouring
Primaries, plus a slack term per Primary for load moved to a neighbour that is not metered. A flow
that leaves one Primary enters its neighbour, so conservation of power is built in. Primaries
metered only in MVA carry no direction, so their residual is explained by their slack alone.

The fit minimises a Huber loss, plus a level penalty (flows and slacks are zero almost always), plus
a fused penalty (each flow and each slack changes rarely, so it is piecewise constant). A group
version that ties the changes at one Primary together solved 25 times slower, so it is not used. Nothing here
reads a switching log. Run it with `uv run --with cvxpy`, since CVXPY is not a locked dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Final

import cvxpy as cp  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00067_common import ADJACENCY, GRID_LENGTH, ML_EXPERIMENTS_DATA_DIR, SERIES_IDS, TUNING_MONTHS
from e00067_common import stamp_months
from e00067_intervals import Interval

BIN: Final[int] = 2
"""Half-hours per hourly bin."""
HOURS_PER_DAY: Final[int] = 24
WINDOW_BINS: Final[int] = 70 * HOURS_PER_DAY
PRE_BINS: Final[int] = 35 * HOURS_PER_DAY
"""A scored transfer's fit window starts 35 days before its onset and spans 70 days."""
STRIDE_BINS: Final[int] = 28 * HOURS_PER_DAY
KEEP_OFFSET_BINS: Final[int] = 21 * HOURS_PER_DAY
"""A real-residual window keeps the runs that start in bins 21 to 49 days after its start."""
HUBER_THRESHOLD: Final[float] = 3.0
ACTIVE_THRESHOLD: Final[float] = 0.1
"""A flow or slack counts as non-zero above this many hourly noise standard deviations."""
MAX_GAP_BINS: Final[int] = 3
MIN_RUN_BINS: Final[int] = 24
METADATA_PATH: Final = ML_EXPERIMENTS_DATA_DIR.parent / "NGED" / "metadata.parquet"


@cache
def mva_series() -> frozenset[int]:
    """The series ids metered only in MVA: a magnitude with no direction."""
    metadata = pl.read_parquet(METADATA_PATH, columns=["time_series_id", "units"])
    return frozenset(
        metadata.filter(pl.col("units").cast(pl.String) == "MVA")["time_series_id"].to_list()
    )


@dataclass(frozen=True)
class Network:
    """The Primaries, the boundaries between them, and the matrix that maps flows to nodes."""

    series: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    directional: tuple[bool, ...]

    @property
    def incidence(self) -> np.ndarray:
        """Per node and edge: -1 where the flow leaves the node, +1 where it enters, else 0."""
        out = np.zeros((len(self.series), len(self.edges)))
        index = {s: i for i, s in enumerate(self.series)}
        for k, (a, b) in enumerate(self.edges):
            out[index[a], k], out[index[b], k] = -1.0, 1.0
        return out


def build_network() -> Network:
    """The 16 trial Primaries and every boundary of `ADJACENCY` between two of them."""
    series = tuple(sorted(SERIES_IDS))
    edges = tuple(
        sorted(
            {
                (min(a, b), max(a, b))
                for a, neighbours in ADJACENCY.items()
                for b in neighbours
                if a in series and b in series
            }
        )
    )
    mva = mva_series()
    return Network(series=series, edges=edges, directional=tuple(s not in mva for s in series))


def hourly(*, values: np.ndarray) -> np.ndarray:
    """Average each pair of half-hours; NaN where both are missing."""
    pairs = values.reshape(-1, BIN)
    present = np.isfinite(pairs)
    total = np.where(present, pairs, 0.0).sum(axis=1)
    count = present.sum(axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def hourly_panel(*, residual: dict[int, np.ndarray], network: Network) -> np.ndarray:
    """The hourly residual of every series, as an array of shape (series, bins)."""
    return np.vstack([hourly(values=residual[s]) for s in network.series])


def noise_scales(*, panel: np.ndarray) -> np.ndarray:
    """Robust standard deviation of each series' hourly residual, on odd months only."""
    months = stamp_months()[::BIN]
    odd = np.isin(months, list(TUNING_MONTHS))
    out = np.empty(panel.shape[0])
    for i, row in enumerate(panel):
        chosen = row[odd & np.isfinite(row)]
        out[i] = 1.4826 * np.median(np.abs(chosen - np.median(chosen)))
    return out


@dataclass(frozen=True)
class Penalties:
    """The three tuned weights: level `flow` and `slack`, and the `fused` penalty on changes."""

    flow: float
    slack: float
    fused: float

    @staticmethod
    def from_knobs(*, level: float, fused_ratio: float, slack_multiplier: float) -> Penalties:
        """`level` times (1, slack multiplier, fused ratio)."""
        return Penalties(flow=level, slack=level * slack_multiplier, fused=level * fused_ratio)


class EdgeFlowProblem:
    """The convex fit for one window, built afresh from constants for each window.

    A matrix-shaped CVXPY parameter multiplying the model makes the compiled problem far too large,
    so each solve builds its problem from numpy constants.
    """

    def __init__(self, *, network: Network, sigma: np.ndarray, bins: int = WINDOW_BINS) -> None:
        self.network, self.sigma, self.bins = network, sigma, bins
        edge_sigma = np.array(
            [
                np.sqrt(sigma[network.series.index(a)] * sigma[network.series.index(b)])
                for a, b in network.edges
            ]
        )
        scale = edge_sigma[None, :] / sigma[:, None]
        mask = np.array(network.directional)[:, None]
        self.edge_sigma = edge_sigma
        self._mixing = network.incidence * scale * mask

    def solve(self, *, z: np.ndarray, penalties: Penalties) -> tuple[np.ndarray, np.ndarray]:
        """Fit one window of normalised hourly residuals; return flows and slacks in noise units."""
        observed = np.isfinite(z)
        flow = cp.Variable((len(self.network.edges), z.shape[1]))
        slack = cp.Variable(z.shape)
        # A missing bin has z = 0 and no model term, so it adds no loss.
        explained = np.where(observed, z, 0.0) - cp.multiply(
            observed.astype(float), self._mixing @ flow + slack
        )
        objective = (
            cp.sum(cp.huber(explained, HUBER_THRESHOLD))
            + penalties.flow * cp.sum(cp.abs(flow))
            + penalties.slack * cp.sum(cp.abs(slack))
            + penalties.fused
            * (cp.sum(cp.abs(cp.diff(flow, axis=1))) + cp.sum(cp.abs(cp.diff(slack, axis=1))))
        )
        problem = cp.Problem(cp.Minimize(objective))
        problem.solve(solver=cp.CLARABEL)
        if problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"edge-flow fit failed: {problem.status}")
        self.last_status, self.last_solve_seconds = problem.status, problem.solver_stats.solve_time
        return flow.value, slack.value


def find_runs(*, values: np.ndarray) -> list[tuple[int, int]]:
    """Runs of bins where |values| is above `ACTIVE_THRESHOLD`, as (first bin, one past the last).

    Gaps of up to `MAX_GAP_BINS` between active bins are bridged, and runs shorter than
    `MIN_RUN_BINS` are dropped.
    """
    active = np.flatnonzero(np.abs(values) > ACTIVE_THRESHOLD)
    runs: list[tuple[int, int]] = []
    for bin_ in active:
        if runs and bin_ - runs[-1][1] <= MAX_GAP_BINS:
            runs[-1] = (runs[-1][0], int(bin_) + 1)
        else:
            runs.append((int(bin_), int(bin_) + 1))
    return [(a, b) for a, b in runs if b - a >= MIN_RUN_BINS]


def window_intervals(
    *,
    network: Network,
    sigma: np.ndarray,
    edge_sigma: np.ndarray,
    flow: np.ndarray,
    slack: np.ndarray,
    first_stamp: int,
) -> list[Interval]:
    """Switching intervals from one fitted window: a run on a boundary marks both its series."""
    out: list[Interval] = []
    for k, (a, b) in enumerate(network.edges):
        for start, stop in find_runs(values=flow[k]):
            step = float(np.mean(flow[k, start:stop]) * edge_sigma[k])
            onset, end = first_stamp + BIN * start, first_stamp + BIN * stop
            for series in (a, b):
                if network.directional[network.series.index(series)]:
                    out.append(Interval(series, onset, end, end, a, "EF", "edge", step))
    for i, series in enumerate(network.series):
        for start, stop in find_runs(values=slack[i]):
            step = float(np.mean(slack[i, start:stop]) * sigma[i])
            onset, end = first_stamp + BIN * start, first_stamp + BIN * stop
            out.append(Interval(series, onset, end, end, series, "EF", "slack", step))
    return out


def window_starts(*, total_bins: int) -> list[tuple[int, int, int]]:
    """Real-residual windows: (first bin, first kept onset bin, one past the last kept bin).

    Windows of `WINDOW_BINS` start every `STRIDE_BINS`, the last one shifted back to end at the end
    of the grid. A run belongs to the window whose kept range holds its onset, and the kept ranges
    tile the grid.
    """
    starts = list(range(0, total_bins - WINDOW_BINS + 1, STRIDE_BINS))
    if starts[-1] + WINDOW_BINS < total_bins:
        starts.append(total_bins - WINDOW_BINS)
    keeps = [0] + [s + KEEP_OFFSET_BINS for s in starts[1:-1]]
    if len(starts) > 1:
        keeps.append(starts[-2] + KEEP_OFFSET_BINS + STRIDE_BINS)
    keeps.append(total_bins)
    return [(s, keeps[i], keeps[i + 1]) for i, s in enumerate(starts)]


def fit_window(
    *, problem: EdgeFlowProblem, z: np.ndarray, first_bin: int, penalties: Penalties
) -> list[Interval]:
    """Fit the `WINDOW_BINS` hourly bins from `first_bin` and return the runs as intervals."""
    flow, slack = problem.solve(z=z[:, first_bin : first_bin + WINDOW_BINS], penalties=penalties)
    return window_intervals(
        network=problem.network,
        sigma=problem.sigma,
        edge_sigma=problem.edge_sigma,
        flow=flow,
        slack=slack,
        first_stamp=BIN * first_bin,
    )


def first_bin_around(*, onset_stamp: int, total_bins: int) -> int:
    """The first bin of the window that starts `PRE_BINS` before an onset, clipped to the grid."""
    return int(np.clip(onset_stamp // BIN - PRE_BINS, 0, total_bins - WINDOW_BINS))


assert GRID_LENGTH % BIN == 0
