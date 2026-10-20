"""The convex model of `studies.dp_gam`, fitted jointly with a penalty along interconnection edges.

**Each substation keeps its own coefficients, and a penalty pulls the coefficients of two
interconnected substations towards each other.** For an edge `e = (i, j)` of the interconnection
graph and a pooled family `f`, the joint objective gains

    gamma_f * g_f * ||C_f M_e,f (theta[i, f] - theta[j, f])||^2

where `g_f` and `C_f` are those of `studies.dp_gam_pooled` (the family's scale per training row and
its centring projector) and `M_e,f` keeps only the coordinates that both ends identify. The weight
`gamma_f` then reads as rows of evidence, as the fleet-mean weight of `dp_gam_pooled` does. The
objective stays a positive-definite convex quadratic, with the same box constraints as `dp_gam`, so
the joint fit has one optimum.

- **Graph families:** the heating slopes, the profile, the holiday periods, and the capacity.
  The capacity term acts on the sum of a substation's capacity increments (its latest capacity in
  points of effective capacity), and only on an edge whose two ends both fit a capacity.
- **Per-edge weight:** the penalty is summed over edges, so a substation with `d` edges is pulled
  `d` times as hard as one with a single edge, and a complete graph on `S` substations with weight
  `gamma` equals the fleet-mean penalty with weight `S * gamma`. The weight reads as rows of
  evidence per edge.
- **Identified coordinates:** a substation identifies a coordinate if the rows it trains on make the
  design column non-zero often enough (`MIN_IDENTIFYING_ROWS` and `RELATIVE_IDENTIFYING_SHARE`). A
  substation that trains on dark rows only has no non-zero daytime heating, profile, or capacity
  columns, so an edge to it penalises only the coordinates both ends hold. The profile's centring
  runs over those shared coordinates only, so a level shift of a band is never penalised.
- **Fleet-mean penalty:** `GraphPenalty.fleet` adds the deviation penalty of `dp_gam_pooled` for the
  families it names, so one fit can hold the fleet-mean penalty, the graph penalty, both, or only a
  holiday fleet mean.
- **Scale readouts:** the profile's scale is the root-mean-square of the centred profile, so
  smoothing the scale through the centred profile vector smooths the profile's shape too.

**The bootstrap is a month-block bootstrap with one draw of calendar months for the whole fleet.**
Each substation keeps a Gram matrix and a right-hand side per calendar month, so a resample is a
vector of month counts and costs one weighted sum. `joint_month_counts` draws the same calendar
months at every substation, within calendar quarters, so the correlation between neighbouring
substations survives into the intervals.

**Stand-alone synthetic fleets** reuse the real design matrices with a chosen truth and a chosen
noise: `month_cross_products` gives the right-hand side for any target.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Literal

import numpy as np
import torch

from studies.dp_calendar import DAY_TYPES, HALF_HOURS_PER_DAY, band_of_hour
from studies.dp_features import FeatureCache
from studies.dp_gam import (
    DEFAULT_MAX_ITERATIONS,
    N_BANDS,
    PIVOT_PATIENCE,
    GamFit,
    GamLayout,
    accumulate_gram,
    check_pinned_capacity,
    default_tolerance,
    design_matrix,
    last_signal_season_year,
    penalty_matrix,
    prepare_fit,
    solve_bounded_quadratic,
)
from studies.dp_gam_pooled import (
    FLEET_MEAN_RIDGE,
    POOLED_FAMILIES,
    PooledFamilyType,
    PooledSite,
    centring_matrix,
)

GraphFamilyType = Literal["heating", "profile", "holiday", "capacity"]
"""The coefficient families a graph penalty can act on."""

GRAPH_FAMILIES: Final[tuple[GraphFamilyType, ...]] = ("heating", "profile", "holiday", "capacity")
"""The coefficient families a graph penalty can act on."""

MIN_IDENTIFYING_ROWS: Final[dict[GraphFamilyType, int]] = {"heating": 1_000, "capacity": 1_000}
"""A heating or capacity column is identified if it is non-zero in this many rows.

These columns are dense (a heating column is non-zero in a quarter of the rows), so 1,000 rows
separates a band the substation trains on from one it does not.
"""

RELATIVE_IDENTIFYING_SHARE: Final[float] = 0.05
"""A profile or holiday column is identified if its non-zero rows reach this share of the median.

The median is taken over the family's columns that are non-zero in the substation's own design. A
profile column (one half-hour of one day type) is non-zero in about 100 to 500 rows over two years
and a holiday column in 48 to 96, so a fixed threshold of 1,000 rows would leave both families
without an identified coordinate, and a share of the substation's own typical column keeps the test
the same for a 12-month and a 24-month window. A column that is non-zero in a handful of twilight
rows is not identified.
"""


@dataclass(frozen=True)
class InterconnectionGraph:
    """An undirected graph whose nodes are substations and whose edges are interconnections.

    Attributes:
        nodes: The substation labels, each once.
        edges: The edges as pairs of node labels, each pair once with the first label sorting
            before the second, and no edge from a node to itself.
    """

    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        """Raise unless the nodes are distinct and every edge is a sorted pair of known nodes.

        Raises:
            ValueError: If a node repeats, an edge repeats, joins a node to itself, names an
                unknown node, or is not sorted.
        """
        if len(set(self.nodes)) != len(self.nodes):
            msg = "each node label may appear once"
            raise ValueError(msg)
        if len(set(self.edges)) != len(self.edges):
            msg = "each edge may appear once"
            raise ValueError(msg)
        for first, second in self.edges:
            if first == second:
                msg = f"the edge {first}-{second} joins a node to itself"
                raise ValueError(msg)
            if first not in self.nodes or second not in self.nodes:
                msg = f"the edge {first}-{second} names a node that is not in the graph"
                raise ValueError(msg)
            if first > second:
                msg = f"the edge {first}-{second} must list the smaller label first"
                raise ValueError(msg)

    @staticmethod
    def from_pairs(
        *, nodes: Sequence[str], pairs: Sequence[tuple[str, str]]
    ) -> InterconnectionGraph:
        """Build a graph from unordered pairs, dropping duplicates and self-pairs.

        Args:
            nodes: The node labels.
            pairs: Pairs of node labels in either order, possibly repeated.

        Returns:
            The graph, keeping only pairs whose two ends are in `nodes`.
        """
        known = set(nodes)
        edges = {(min(a, b), max(a, b)) for a, b in pairs if a != b and a in known and b in known}
        return InterconnectionGraph(nodes=tuple(nodes), edges=tuple(sorted(edges)))

    def degrees(self) -> dict[str, int]:
        """Return the number of edges at each node.

        Returns:
            A map from node label to degree.
        """
        degree = dict.fromkeys(self.nodes, 0)
        for first, second in self.edges:
            degree[first] += 1
            degree[second] += 1
        return degree

    def laplacian(self) -> np.ndarray:
        """Return the graph Laplacian in the order of `nodes`.

        Returns:
            A symmetric matrix of shape (nodes, nodes), degree on the diagonal, minus one for each
            edge.
        """
        position = {node: i for i, node in enumerate(self.nodes)}
        laplacian = np.zeros((len(self.nodes), len(self.nodes)))
        for first, second in self.edges:
            a, b = position[first], position[second]
            laplacian[a, a] += 1.0
            laplacian[b, b] += 1.0
            laplacian[a, b] -= 1.0
            laplacian[b, a] -= 1.0
        return laplacian


def ring_with_chords(
    *, nodes: Sequence[str], n_chords: int, rng: np.random.Generator
) -> InterconnectionGraph:
    """Return a ring through all the nodes plus random chords.

    Args:
        nodes: The node labels, in ring order.
        n_chords: The number of extra edges, each between two nodes not already joined.
        rng: The random generator, which fixes the chords.

    Returns:
        The graph, with `len(nodes) + n_chords` edges.

    Raises:
        ValueError: If fewer than three nodes are given or the ring leaves too few free pairs.
    """
    count = len(nodes)
    if count < 3:
        msg = "a ring needs at least three nodes"
        raise ValueError(msg)
    pairs = {(nodes[i], nodes[(i + 1) % count]) for i in range(count)}
    normalised = {(min(a, b), max(a, b)) for a, b in pairs}
    free = [
        (min(a, b), max(a, b))
        for i, a in enumerate(nodes)
        for b in nodes[i + 1 :]
        if (min(a, b), max(a, b)) not in normalised
    ]
    if n_chords > len(free):
        msg = f"only {len(free)} pairs are free for {n_chords} chords"
        raise ValueError(msg)
    chosen = rng.choice(len(free), size=n_chords, replace=False)
    normalised |= {free[int(i)] for i in chosen}
    return InterconnectionGraph(nodes=tuple(nodes), edges=tuple(sorted(normalised)))


def degree_preserving_permutation(
    *, graph: InterconnectionGraph, rng: np.random.Generator, swaps_per_edge: int = 10
) -> InterconnectionGraph:
    """Return a graph with the same node degrees and different edges, by random edge swaps.

    A swap replaces the edges (a, b) and (c, d) with (a, d) and (c, b), and is kept only if the new
    edges are not self-loops and not already present.

    Args:
        graph: The graph to permute.
        rng: The random generator.
        swaps_per_edge: How many swaps to try per edge.

    Returns:
        The permuted graph, with the same degree at every node.
    """
    edges = list(graph.edges)
    present = set(edges)
    for _ in range(swaps_per_edge * max(len(edges), 1)):
        if len(edges) < 2:
            break
        first, second = (int(i) for i in rng.choice(len(edges), size=2, replace=False))
        a, b = edges[first]
        c, d = edges[second]
        if rng.random() < 0.5:
            c, d = d, c
        new_first = (min(a, d), max(a, d))
        new_second = (min(c, b), max(c, b))
        if a == d or c == b or new_first == new_second:
            continue
        if new_first in present or new_second in present:
            continue
        present -= {edges[first], edges[second]}
        present |= {new_first, new_second}
        edges[first], edges[second] = new_first, new_second
    return InterconnectionGraph(nodes=graph.nodes, edges=tuple(sorted(edges)))


def smooth_field_covariance(*, graph: InterconnectionGraph, kappa: float) -> np.ndarray:
    """Return the correlation matrix of a Gaussian Markov random field with precision `I + kappa L`.

    Args:
        graph: The graph.
        kappa: The coupling; 0 gives independent nodes.

    Returns:
        The correlation matrix (unit diagonal) in the order of `graph.nodes`.
    """
    precision = np.eye(len(graph.nodes)) + kappa * graph.laplacian()
    covariance = np.linalg.solve(precision, np.eye(len(graph.nodes)))
    scale = np.sqrt(np.diag(covariance))
    return covariance / scale[:, None] / scale[None, :]


def mean_neighbour_correlation(*, graph: InterconnectionGraph, kappa: float) -> float:
    """Return the mean correlation of the field across the graph's edges.

    Args:
        graph: The graph, with at least one edge.
        kappa: The coupling of `smooth_field_covariance`.

    Returns:
        The mean over edges of the correlation between the two ends.
    """
    correlation = smooth_field_covariance(graph=graph, kappa=kappa)
    position = {node: i for i, node in enumerate(graph.nodes)}
    return float(np.mean([correlation[position[a], position[b]] for a, b in graph.edges]))


def kappa_for_neighbour_correlation(*, graph: InterconnectionGraph, target: float) -> float:
    """Return the coupling whose mean neighbour correlation is `target`, by bisection.

    Args:
        graph: The graph, with at least one edge.
        target: The wanted mean correlation across edges, between 0 and 1.

    Returns:
        The coupling `kappa`.

    Raises:
        ValueError: If `target` is outside (0, 1).
    """
    if not 0.0 < target < 1.0:
        msg = f"the target correlation must lie in (0, 1), not {target}"
        raise ValueError(msg)
    low, high = 0.0, 1.0
    while mean_neighbour_correlation(graph=graph, kappa=high) < target:
        high *= 2.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        if mean_neighbour_correlation(graph=graph, kappa=middle) < target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def sample_smooth_fields(
    *, graph: InterconnectionGraph, kappa: float, n_fields: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw standardised Gaussian Markov random fields on the graph.

    Args:
        graph: The graph.
        kappa: The coupling of `smooth_field_covariance`.
        n_fields: How many independent fields to draw.
        rng: The random generator.

    Returns:
        An array of shape (nodes, n_fields), each node with unit marginal variance.
    """
    correlation = smooth_field_covariance(graph=graph, kappa=kappa)
    factor = np.linalg.cholesky(correlation + 1e-12 * np.eye(len(graph.nodes)))
    return factor @ rng.normal(size=(len(graph.nodes), n_fields))


def edge_similarity_ratio(
    *, graph: InterconnectionGraph, values: Mapping[str, float], rng: np.random.Generator
) -> float:
    """Return the mean squared difference along edges over the mean over random node pairs.

    A ratio below 1 means interconnected substations are more alike than random pairs.

    Args:
        graph: The graph.
        values: One value per node (a parameter such as a heating slope).
        rng: The random generator, which picks the random pairs.

    Returns:
        The ratio. The random pairs are 2,000 draws of two different nodes.

    Raises:
        ValueError: If the graph has no edge, fewer than three nodes, or `values` lacks a node.
    """
    if not graph.edges or len(graph.nodes) < 3:
        msg = "the ratio needs at least one edge and three nodes"
        raise ValueError(msg)
    missing = set(graph.nodes) - set(values)
    if missing:
        msg = f"no value for {sorted(missing)}"
        raise ValueError(msg)
    along = np.mean([(values[a] - values[b]) ** 2 for a, b in graph.edges])
    draws = np.array([rng.choice(len(graph.nodes), size=2, replace=False) for _ in range(2000)])
    ordered = [values[node] for node in graph.nodes]
    across = np.mean((np.take(ordered, draws[:, 0]) - np.take(ordered, draws[:, 1])) ** 2)
    return float(along / across)


@dataclass(frozen=True)
class GraphPenalty:
    """The weights of one fit, in rows of evidence.

    Attributes:
        fleet: The fleet-mean deviation weight of each pooled family; a missing family has weight
            0. With all weights 0 the fit has no fleet mean.
        graph: The graph weight of each graph family; a missing family has weight 0.
    """

    fleet: Mapping[PooledFamilyType, float]
    graph: Mapping[GraphFamilyType, float]

    @staticmethod
    def none() -> GraphPenalty:
        """Return the penalty of the per-site fit.

        Returns:
            A penalty with no weight on anything.
        """
        return GraphPenalty(fleet={}, graph={})

    @staticmethod
    def of(
        *,
        fleet_rows: float = 0.0,
        graph_rows: float = 0.0,
        holiday_fleet_rows: float = 0.0,
    ) -> GraphPenalty:
        """Return a penalty from the three weights the experiment uses.

        Args:
            fleet_rows: The fleet-mean weight on every pooled family.
            graph_rows: The graph weight on every graph family.
            holiday_fleet_rows: The fleet-mean weight on the holiday family alone, used when
                `fleet_rows` is 0.

        Returns:
            The penalty.
        """
        fleet: dict[PooledFamilyType, float] = {}
        if fleet_rows:
            fleet = dict.fromkeys(POOLED_FAMILIES, fleet_rows)
        elif holiday_fleet_rows:
            fleet = {"holiday": holiday_fleet_rows}
        graph: dict[GraphFamilyType, float] = {}
        if graph_rows:
            graph = dict.fromkeys(GRAPH_FAMILIES, graph_rows)
        return GraphPenalty(fleet=fleet, graph=graph)

    def is_separable(self) -> bool:
        """Return True if no weight couples two substations.

        Returns:
            True when every fleet and graph weight is 0.
        """
        return not any(self.fleet.values()) and not any(self.graph.values())


@dataclass(frozen=True)
class MonthAxis:
    """The calendar months a fleet spans, shared by every substation.

    Attributes:
        months: The months, ascending, as `datetime64[M]`.
        quarter: The calendar quarter (0 to 3) of each month.
    """

    months: np.ndarray
    quarter: np.ndarray

    @staticmethod
    def of(*, months: np.ndarray) -> MonthAxis:
        """Build the axis from any array of months.

        Args:
            months: Months as `datetime64[M]`, possibly repeated.

        Returns:
            The sorted distinct months with their quarters.
        """
        distinct = np.unique(months)
        quarter = ((distinct.astype(np.int64) % 12) // 3).astype(np.int64)
        return MonthAxis(months=distinct, quarter=quarter)


@dataclass(frozen=True)
class NodeDesign:
    """One substation's design at one cut, split by calendar month.

    Attributes:
        inputs: The substation's inputs to the fit.
        layout: The columns.
        index: The training rows.
        n_rows: The number of training rows.
        months: The calendar months the training rows span, ascending, as `datetime64[M]`.
        month_of_row: For each training row, its position in `months`.
        x: The design matrix, shape (n_rows, p), kept to build right-hand sides of other targets.
        month_gram: `X_m^T X_m` per month, shape (months, p, p), float64.
        month_rhs: `X_m^T y_m` per month for the substation's own target (with the pinned solar
            output added back for a pinned substation), shape (months, p).
        penalty: The base penalty matrix of `dp_gam.penalty_matrix`.
        cost: The base penalty's linear term.
        identified: The design columns the substation's rows identify, from `identified_columns`.
    """

    inputs: PooledSite
    layout: GamLayout
    index: np.ndarray
    n_rows: int
    months: np.ndarray
    month_of_row: np.ndarray
    x: np.ndarray
    month_gram: torch.Tensor
    month_rhs: torch.Tensor
    penalty: np.ndarray
    cost: np.ndarray
    identified: np.ndarray

    @property
    def site(self) -> str:
        """Return the substation's label."""
        return self.inputs.site


def _check_site(*, site: PooledSite, cache: FeatureCache, cut: datetime) -> None:
    """Raise if a substation's inputs cannot join a graph fit at `cut`."""
    spec = site.spec
    if spec.drift != "none" or spec.covid_flag or spec.final_knot != "free":
        msg = f"{site.site}: a graph fit needs drift 'none', no COVID flag, and a free final knot"
        raise ValueError(msg)
    check_pinned_capacity(
        capacity=spec.capacity,
        pinned=site.pinned,
        site=site.site,
        features=cache.sites[site.site],
        cut=cut,
    )
    if (
        site.pinned is not None
        and site.pinned.source == "growth_prior"
        and not site.allow_late_vintage
    ):
        msg = (
            f"{site.site}: the pinned capacity is a vintage fitted long after {cut}; give the "
            "substation a fitted capacity, or set allow_late_vintage for an oracle arm"
        )
        raise ValueError(msg)


def identified_columns(*, x: np.ndarray, layout: GamLayout) -> np.ndarray:
    """Return the design columns that a substation's own rows identify.

    Args:
        x: The substation's design matrix, shape (rows, p).
        layout: The columns.

    Returns:
        A boolean array of length p. A heating or capacity column is identified if it is non-zero in
        at least `MIN_IDENTIFYING_ROWS` rows; a profile or holiday column if it is non-zero in at
        least `RELATIVE_IDENTIFYING_SHARE` of the median non-zero count of the family's non-zero
        columns. Columns of no graph family are False.
    """
    non_zero_rows = (x != 0.0).sum(axis=0)
    identified = np.zeros(layout.n_columns, dtype=bool)
    for family, minimum in MIN_IDENTIFYING_ROWS.items():
        columns = layout.slices[family]
        identified[columns] = non_zero_rows[columns] >= minimum
    for family in ("profile", "holiday"):
        columns = layout.slices[family]
        counts = non_zero_rows[columns]
        held = counts[counts > 0]
        if len(held) == 0:
            continue
        identified[columns] = (counts > 0) & (
            counts >= RELATIVE_IDENTIFYING_SHARE * np.median(held)
        )
    return identified


def build_node_design(*, cache: FeatureCache, site: PooledSite, cut: datetime) -> NodeDesign:
    """Build a substation's design, monthly Gram matrices, and base penalty at a cut.

    Args:
        cache: The cache.
        site: The substation's inputs; the same rules as `dp_gam_pooled.fit_pooled` apply.
        cut: The cut; no training row ends after it.

    Returns:
        The design.
    """
    _check_site(site=site, cache=cache, cut=cut)
    features = cache.sites[site.site]
    layout, index = prepare_fit(
        cache=cache,
        site=site.site,
        spec=site.spec,
        cut=cut,
        window=site.window,
        season_years=site.season_years,
    )
    x = design_matrix(cache=cache, features=features, layout=layout, index=index)
    month_keys = cache.end_times[index].astype("datetime64[M]")
    months, month_of_row = np.unique(month_keys, return_inverse=True)
    indicator = (month_of_row[None, :] == np.arange(len(months))[:, None]).astype(np.float64)
    gram, rhs = accumulate_gram(
        cache=cache,
        features=features,
        layout=layout,
        index=index,
        weights=indicator,
        pinned=site.pinned,
    )
    penalty, cost = penalty_matrix(layout=layout, n_rows=len(index))
    return NodeDesign(
        inputs=site,
        layout=layout,
        index=index,
        n_rows=len(index),
        months=months,
        month_of_row=month_of_row,
        x=x,
        month_gram=gram,
        month_rhs=rhs,
        penalty=penalty,
        cost=cost,
        identified=identified_columns(x=x, layout=layout),
    )


def month_cross_products(*, design: NodeDesign, target: np.ndarray) -> torch.Tensor:
    """Return `X_m^T y_m` for each month of a design, for any target.

    Args:
        design: The design.
        target: A target for the design's training rows, shape (n_rows,), in points.

    Returns:
        A tensor of shape (months, p), float64.
    """
    out = torch.zeros((len(design.months), design.layout.n_columns), dtype=torch.float64)
    for position in range(len(design.months)):
        rows = design.month_of_row == position
        out[position] = torch.as_tensor(design.x[rows].T @ target[rows])
    return out


def month_axis_of(*, designs: Sequence[NodeDesign]) -> MonthAxis:
    """Return the shared month axis of a fleet's designs.

    Args:
        designs: The fleet's designs.

    Returns:
        The axis over the union of their months.
    """
    return MonthAxis.of(months=np.concatenate([d.months for d in designs]))


def joint_month_counts(
    *, axis: MonthAxis, n_resamples: int, rng: np.random.Generator
) -> np.ndarray:
    """Return month counts of a month-block bootstrap with one draw for the whole fleet.

    Each resample draws, with replacement and within each calendar quarter, as many of the
    quarter's months as the quarter holds, and every substation takes its own months' counts. A
    row of ones is the full sample.

    Args:
        axis: The fleet's month axis.
        n_resamples: The number of resamples.
        rng: The random generator.

    Returns:
        Counts of shape (1 + n_resamples, months): the full sample first, then the resamples.
    """
    counts = np.zeros((n_resamples, len(axis.months)))
    for quarter in range(4):
        members = np.flatnonzero(axis.quarter == quarter)
        if len(members) == 0:
            continue
        draws = rng.integers(0, len(members), size=(n_resamples, len(members)))
        for resample in range(n_resamples):
            counts[resample, members] = np.bincount(draws[resample], minlength=len(members))
    return np.concatenate([np.ones((1, len(axis.months))), counts])


def family_scales(
    *, designs: Sequence[NodeDesign]
) -> dict[GraphFamilyType | PooledFamilyType, float]:
    """Return `g_f`, the mean diagonal of each family's Gram matrix per training row.

    The pooled families follow `dp_gam_pooled`: pooled over every substation's rows and the
    family's columns. The capacity scale is the diagonal sum over the capacity columns of the
    substations that identify a capacity, divided by their number of capacity columns times their
    rows, so a pinned substation (no capacity column) and a dark-only substation (capacity columns
    that are all zero) do not enter it.

    Args:
        designs: The fleet's designs, whose full-sample Gram matrices set the scale.

    Returns:
        A map from family to scale.
    """
    total_rows = sum(design.n_rows for design in designs)
    scales: dict[GraphFamilyType | PooledFamilyType, float] = {}
    diagonals = [torch.diagonal(design.month_gram.sum(dim=0)) for design in designs]
    reference = designs[0].layout.slices
    for family in POOLED_FAMILIES:
        columns = reference[family]
        diagonal_sum = sum(float(d[columns].sum()) for d in diagonals)
        scales[family] = diagonal_sum / ((columns.stop - columns.start) * total_rows)
    capacity_sum, capacity_cells = 0.0, 0
    for design, diagonal in zip(designs, diagonals, strict=True):
        columns = design.layout.slices["capacity"]
        if not design.identified[columns].any():
            continue
        capacity_sum += float(diagonal[columns].sum())
        capacity_cells += (columns.stop - columns.start) * design.n_rows
    scales["capacity"] = capacity_sum / capacity_cells if capacity_cells else 0.0
    return scales


def _masked_centring(*, family: GraphFamilyType, both: np.ndarray) -> np.ndarray:
    """Return the centring projector over the shared coordinates only, shape (m, m).

    The profile's centring removes each hour band's mean over the band's shared columns; every other
    family uses the identity.
    """
    shared = int(both.sum())
    if family != "profile":
        return np.eye(shared)
    band_of_half_hour = band_of_hour(hour=np.arange(HALF_HOURS_PER_DAY) // 2)
    band = np.tile(band_of_half_hour, len(DAY_TYPES))[both]
    indicator = (band[:, None] == np.arange(N_BANDS)[None, :]).astype(np.float64)
    sizes = np.maximum(indicator.sum(axis=0), 1.0)
    return np.eye(shared) - indicator @ (indicator / sizes[None, :]).T


def _edge_block(
    *, family: GraphFamilyType, first: NodeDesign, second: NodeDesign
) -> tuple[slice, slice, np.ndarray] | None:
    """Return the two ends' column slices of a family and the edge's quadratic form.

    The form acts on the difference of the two ends' coefficients: `(C D)^T (C D)` over the
    coordinates both ends identify, with the centring `C` taken over those coordinates only, or
    `a a^T` for the capacity functional. `None` if the edge has nothing to penalise in this family.
    """
    if family == "capacity":
        left, right = first.layout.slices["capacity"], second.layout.slices["capacity"]
        if left.stop == left.start or right.stop == right.start:
            return None
        if not (first.identified[left].any() and second.identified[right].any()):
            return None
        return left, right, np.ones((1, 1))
    left, right = first.layout.slices[family], second.layout.slices[family]
    size = left.stop - left.start
    both = first.identified[left] & second.identified[right]
    if not both.any():
        return None
    centring = _masked_centring(family=family, both=both)
    form = np.zeros((size, size))
    form[np.ix_(both, both)] = centring.T @ centring
    return left, right, form


def _add_coupling(
    *,
    matrix: torch.Tensor,
    designs: Sequence[NodeDesign],
    graph: InterconnectionGraph,
    penalty: GraphPenalty,
    scales: Mapping[GraphFamilyType | PooledFamilyType, float],
    offsets: Mapping[PooledFamilyType, slice],
    starts: np.ndarray,
    mu_start: int,
) -> None:
    """Add the quadratic form of the fleet-mean and graph penalties to `matrix`, in place."""
    reference = designs[0].layout.slices
    for family, rows in penalty.fleet.items():
        weight = rows * scales[family]
        if weight == 0.0:
            continue
        size = reference[family].stop - reference[family].start
        coupling = torch.as_tensor(weight * centring_matrix(family=family, size=size))
        mu_columns = slice(mu_start + offsets[family].start, mu_start + offsets[family].stop)
        for start in starts[:-1]:
            columns = slice(
                int(start) + reference[family].start, int(start) + reference[family].stop
            )
            matrix[columns, columns] += coupling
            matrix[mu_columns, mu_columns] += coupling
            matrix[columns, mu_columns] -= coupling
            matrix[mu_columns, columns] -= coupling
    position = {design.site: i for i, design in enumerate(designs)}
    for family, rows in penalty.graph.items():
        weight = rows * scales[family]
        if weight == 0.0:
            continue
        for first_label, second_label in graph.edges:
            i, j = position[first_label], position[second_label]
            block = _edge_block(family=family, first=designs[i], second=designs[j])
            if block is None:
                continue
            left, right, form = block
            a = slice(int(starts[i]) + left.start, int(starts[i]) + left.stop)
            b = slice(int(starts[j]) + right.start, int(starts[j]) + right.stop)
            if family == "capacity":
                # The penalty acts on the sum of each end's increments, so every entry of the
                # blocks (a, a), (b, b), (a, b), and (b, a) takes the same weight.
                matrix[a, a] += weight
                matrix[b, b] += weight
                matrix[a, b] -= weight
                matrix[b, a] -= weight
                continue
            form_tensor = torch.as_tensor(weight * form)
            matrix[a, a] += form_tensor
            matrix[b, b] += form_tensor
            matrix[a, b] -= form_tensor
            matrix[b, a] -= form_tensor


@dataclass(frozen=True)
class GraphSolution:
    """The solution of a batch of joint fits that share a graph and a penalty.

    Attributes:
        thetas: Each substation's coefficients, shape (batch, p).
        fleet_mean: The fleet mean of the pooled families, shape (batch, m), or `None` if the
            penalty has no fleet-mean weight.
        violation: Each fit's optimality violation, shape (batch,).
    """

    thetas: dict[str, np.ndarray]
    fleet_mean: np.ndarray | None
    violation: np.ndarray


def _check_same_pooled_layout(*, designs: Sequence[NodeDesign]) -> None:
    """Raise unless every substation holds the same pooled-family columns in the same places."""
    reference = {f: designs[0].layout.slices[f] for f in POOLED_FAMILIES}
    for design in designs[1:]:
        if {f: design.layout.slices[f] for f in POOLED_FAMILIES} != reference:
            msg = f"{design.site} has pooled-family columns that differ from {designs[0].site}'s"
            raise ValueError(msg)


def _slice_counts(
    *, design: NodeDesign, axis: MonthAxis | None, counts: np.ndarray | None
) -> np.ndarray:
    """Return a substation's month counts, shape (batch, months), from the fleet's counts."""
    if counts is None or axis is None:
        return np.ones((1, len(design.months)))
    positions = np.searchsorted(axis.months, design.months)
    return counts[:, positions]


def solve_graph_fit(
    *,
    designs: Sequence[NodeDesign],
    graph: InterconnectionGraph,
    penalty: GraphPenalty,
    scales: Mapping[GraphFamilyType | PooledFamilyType, float],
    axis: MonthAxis | None = None,
    counts: np.ndarray | None = None,
    month_rhs: Mapping[str, torch.Tensor] | None = None,
    device: torch.device | str = "cpu",
) -> GraphSolution:
    """Solve the joint convex program for every weighting of a batch.

    Args:
        designs: The fleet's designs, one per node of the graph (extra nodes are refused).
        graph: The graph; a node absent from `designs` is refused.
        penalty: The fleet-mean and graph weights, in rows of evidence.
        scales: The family scales `family_scales` returns for the full sample. They are passed in
            so that every resample of a bootstrap shares the full sample's penalty.
        axis: The fleet's month axis; required with `counts`.
        counts: Month counts of shape (batch, months), as `joint_month_counts` returns, or `None`
            for the full sample alone.
        month_rhs: A right-hand side per month and substation to use instead of the designs' own,
            for a synthetic target.
        device: Where the solve runs.

    Returns:
        The solution, with one batch member per row of `counts`.

    Raises:
        ValueError: If the graph and the designs name different nodes, or `counts` is given
            without `axis`.
    """
    if set(graph.nodes) != {design.site for design in designs}:
        msg = "the graph's nodes and the designs' substations must be the same"
        raise ValueError(msg)
    if counts is not None and axis is None:
        msg = "month counts need the fleet's month axis"
        raise ValueError(msg)
    ordered = [next(d for d in designs if d.site == node) for node in graph.nodes]
    _check_same_pooled_layout(designs=ordered)
    batch = 1 if counts is None else counts.shape[0]
    grams, rhss = [], []
    for design in ordered:
        node_counts = torch.as_tensor(_slice_counts(design=design, axis=axis, counts=counts))
        source_rhs = design.month_rhs if month_rhs is None else month_rhs[design.site]
        grams.append(torch.einsum("bm,mpq->bpq", node_counts, design.month_gram))
        rhss.append(node_counts @ source_rhs)
    if penalty.is_separable():
        return _solve_separable(designs=ordered, grams=grams, rhss=rhss, device=device)
    return _solve_coupled(
        designs=ordered,
        graph=graph,
        penalty=penalty,
        scales=scales,
        grams=grams,
        rhss=rhss,
        batch=batch,
        device=device,
    )


def _solve_separable(
    *,
    designs: Sequence[NodeDesign],
    grams: Sequence[torch.Tensor],
    rhss: Sequence[torch.Tensor],
    device: torch.device | str,
) -> GraphSolution:
    """Solve each substation on its own, which a penalty with no coupling allows."""
    thetas: dict[str, np.ndarray] = {}
    violation = np.zeros(grams[0].shape[0])
    for design, gram, rhs in zip(designs, grams, rhss, strict=True):
        penalty = torch.as_tensor(design.penalty, dtype=torch.float64)
        cost = torch.as_tensor(design.cost, dtype=torch.float64)
        solution, worst = solve_bounded_quadratic(
            matrix=(gram + penalty[None]).to(device),
            rhs=(rhs - 0.5 * cost[None]).to(device),
            constrained=torch.as_tensor(design.layout.constrained),
        )
        thetas[design.site] = solution.cpu().numpy()
        violation = np.maximum(violation, worst.cpu().numpy())
    return GraphSolution(thetas=thetas, fleet_mean=None, violation=violation)


def _kkt_violation(
    *, theta: torch.Tensor, gradient: torch.Tensor, constrained: torch.Tensor
) -> float:
    """Return the largest violation of the optimality conditions of one problem."""
    free = ~constrained | (theta > 0.0)
    free_violation = torch.where(free, gradient.abs(), torch.zeros_like(gradient)).max()
    bound_violation = torch.where(
        ~free, (-gradient).clamp(min=0.0), torch.zeros_like(gradient)
    ).max()
    return float(torch.maximum(free_violation, bound_violation))


def solve_bounded_cholesky(
    *,
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    constrained: torch.Tensor,
    start_free: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Minimise `theta^T A theta - 2 b^T theta` with `theta_j >= 0` where `constrained[j]`.

    Block principal pivoting as in `dp_gam.solve_bounded_quadratic`, for one problem, with a
    Cholesky factorisation of the free block in place of an LU solve of the whole matrix and a
    starting free set the caller supplies. The members of a bootstrap batch share nearly the same
    active set, so a start from the previous member's free set usually needs one or two
    factorisations, where the LU version needs three or more.

    Args:
        matrix: `A`, shape (p, p), positive definite.
        rhs: `b`, shape (p,).
        constrained: Boolean, shape (p,), True for the non-negative parameters.
        start_free: Boolean, shape (p,), the starting free set; unconstrained parameters are
            always free.

    Returns:
        `(theta, free, violation)`: the solution, the free set it ended on, and the largest
        violation of the optimality conditions relative to the largest entry of `b`.

    Raises:
        ValueError: If the problem has not converged.
        torch.linalg.LinAlgError: If a free block is not positive definite in floating point.
    """
    p = rhs.shape[0]
    epsilon = float(torch.finfo(matrix.dtype).eps)
    tolerance = default_tolerance(dtype=matrix.dtype)
    scale = float(rhs.abs().max().clamp(min=1.0))
    indices = torch.arange(p, device=rhs.device)
    free = start_free | ~constrained
    best_broken, patience = p + 1, 0
    theta = torch.zeros_like(rhs)
    for _ in range(DEFAULT_MAX_ITERATIONS):
        held = torch.nonzero(free).squeeze(1)
        reduced = matrix.index_select(0, held).index_select(1, held)
        factor, info = torch.linalg.cholesky_ex(reduced)
        if int(info) != 0:
            msg = "a free block is not positive definite"
            raise torch.linalg.LinAlgError(msg)
        theta = torch.zeros_like(rhs)
        theta[held] = torch.cholesky_solve(rhs[held][:, None], factor)[:, 0]
        gradient = matrix @ theta - rhs
        theta_tolerance = 100.0 * epsilon * float(theta.abs().max().clamp(min=1.0))
        broken = (free & constrained & (theta < -theta_tolerance)) | (
            ~free & (gradient < -tolerance * scale)
        )
        count = int(broken.sum())
        if count == 0:
            break
        if count < best_broken:
            best_broken, patience = count, 0
        else:
            patience += 1
        if patience >= PIVOT_PATIENCE:
            largest = int(torch.where(broken, indices, -torch.ones_like(indices)).max())
            broken = indices == largest
        free = free ^ broken
    else:
        msg = "the bounded Cholesky solver did not converge"
        raise ValueError(msg)
    theta = torch.where(constrained, theta.clamp(min=0.0), theta)
    gradient = matrix @ theta - rhs
    violation = _kkt_violation(theta=theta, gradient=gradient, constrained=constrained) / scale
    if violation > 10.0 * tolerance:
        msg = f"the bounded Cholesky solver did not converge: worst violation {violation:.3g}"
        raise ValueError(msg)
    return theta, free, violation


def _solve_member(
    *,
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    constrained: torch.Tensor,
    start_free: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Solve one member by Cholesky, falling back to the LU solver if that fails."""
    try:
        return solve_bounded_cholesky(
            matrix=matrix, rhs=rhs, constrained=constrained, start_free=start_free
        )
    except torch.linalg.LinAlgError, ValueError:
        solution, violation = solve_bounded_quadratic(
            matrix=matrix[None], rhs=rhs[None], constrained=constrained
        )
        theta = solution[0]
        return theta, (theta > 0.0) | ~constrained, float(violation[0])


def _solve_coupled(
    *,
    designs: Sequence[NodeDesign],
    graph: InterconnectionGraph,
    penalty: GraphPenalty,
    scales: Mapping[GraphFamilyType | PooledFamilyType, float],
    grams: Sequence[torch.Tensor],
    rhss: Sequence[torch.Tensor],
    batch: int,
    device: torch.device | str,
) -> GraphSolution:
    """Solve the joint problem in `[theta_1, ..., theta_S, mu]`, in chunks of the batch."""
    reference = designs[0].layout.slices
    offsets: dict[PooledFamilyType, slice] = {}
    cursor = 0
    for family in POOLED_FAMILIES:
        size = reference[family].stop - reference[family].start
        offsets[family] = slice(cursor, cursor + size)
        cursor += size
    has_fleet = any(penalty.fleet.values())
    sizes = [design.layout.n_columns for design in designs]
    starts = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    mu_start = int(starts[-1])
    total = mu_start + (cursor if has_fleet else 0)
    constrained = np.zeros(total, dtype=bool)
    fixed = torch.zeros((total, total), dtype=torch.float64)
    fixed_rhs = torch.zeros(total, dtype=torch.float64)
    for design, start in zip(designs, starts[:-1], strict=True):
        stop = int(start) + design.layout.n_columns
        fixed[start:stop, start:stop] = torch.as_tensor(design.penalty)
        fixed_rhs[start:stop] = -0.5 * torch.as_tensor(design.cost)
        constrained[start:stop] = design.layout.constrained
    if has_fleet:
        fixed[mu_start:, mu_start:] += FLEET_MEAN_RIDGE * torch.eye(cursor, dtype=torch.float64)
    _add_coupling(
        matrix=fixed,
        designs=designs,
        graph=graph,
        penalty=penalty,
        scales=scales,
        offsets=offsets,
        starts=starts,
        mu_start=mu_start,
    )
    fixed = fixed.to(device)
    constrained_tensor = torch.as_tensor(constrained, device=device)
    pieces, violations = [], []
    free = torch.ones(total, dtype=torch.bool, device=device)
    for member in range(batch):
        matrix = fixed.clone()
        rhs = fixed_rhs.to(device).clone()
        for design, gram, node_rhs, start in zip(designs, grams, rhss, starts[:-1], strict=True):
            stop = int(start) + design.layout.n_columns
            matrix[start:stop, start:stop] += gram[member].to(device)
            rhs[start:stop] += node_rhs[member].to(device)
        theta, free, violation = _solve_member(
            matrix=matrix, rhs=rhs, constrained=constrained_tensor, start_free=free
        )
        pieces.append(theta.cpu().numpy())
        violations.append(violation)
    solved = np.stack(pieces)
    thetas = {
        design.site: solved[:, int(start) : int(start) + design.layout.n_columns]
        for design, start in zip(designs, starts[:-1], strict=True)
    }
    return GraphSolution(
        thetas=thetas,
        fleet_mean=solved[:, mu_start:] if has_fleet else None,
        violation=np.array(violations),
    )


def gam_fits_from_solution(
    *, cache: FeatureCache, designs: Sequence[NodeDesign], solution: GraphSolution, cut: datetime
) -> dict[str, GamFit]:
    """Wrap a solution as one `dp_gam.GamFit` per substation, for `dp_gam.predict_gam`.

    Args:
        cache: The cache.
        designs: The fleet's designs.
        solution: The solution.
        cut: The fit's cut.

    Returns:
        Each substation's fit, with one batch member per member of the solution.
    """
    return {
        design.site: GamFit(
            layout=design.layout,
            theta=solution.thetas[design.site],
            cut=cut,
            n_rows=design.n_rows,
            site=design.site,
            last_season_year=last_signal_season_year(
                cache=cache,
                site=design.site,
                index=design.index,
                cut=cut,
                pinned=design.inputs.pinned,
            ),
            violation=solution.violation,
            pinned=design.inputs.pinned,
        )
        for design in designs
    }


def heating_band_slopes(*, layout: GamLayout, theta: np.ndarray) -> np.ndarray:
    """Return each hour band's total heating slope, summed over the smoothing windows.

    Args:
        layout: The columns.
        theta: Coefficients, shape (batch, p).

    Returns:
        An array of shape (batch, `N_BANDS`), in points per 10 degrees Celsius as the layout scales.
        The number of smoothing windows is read from the layout.
    """
    columns = layout.slices["heating"]
    windows = (columns.stop - columns.start) // N_BANDS
    block = theta[:, columns].reshape(theta.shape[0], windows, N_BANDS)
    return block.sum(axis=1)


def profile_scale(*, layout: GamLayout, theta: np.ndarray) -> np.ndarray:
    """Return the root-mean-square of the centred profile.

    Args:
        layout: The columns.
        theta: Coefficients, shape (batch, p).

    Returns:
        An array of shape (batch,).
    """
    columns = layout.slices["profile"]
    size = columns.stop - columns.start
    centred = theta[:, columns] @ centring_matrix(family="profile", size=size).T
    return np.sqrt((centred**2).mean(axis=1))


def latest_capacity(*, layout: GamLayout, theta: np.ndarray) -> np.ndarray:
    """Return the sum of the capacity increments, the latest season-year's capacity in points.

    Args:
        layout: The columns.
        theta: Coefficients, shape (batch, p).

    Returns:
        An array of shape (batch,); zeros if the layout has no capacity column.
    """
    return theta[:, layout.slices["capacity"]].sum(axis=1)


def holiday_level(*, layout: GamLayout, theta: np.ndarray) -> np.ndarray:
    """Return the mean holiday coefficient, the level shared across holiday periods.

    Args:
        layout: The columns.
        theta: Coefficients, shape (batch, p).

    Returns:
        An array of shape (batch,).
    """
    return theta[:, layout.slices["holiday"]].mean(axis=1)


__all__ = [
    "GRAPH_FAMILIES",
    "MIN_IDENTIFYING_ROWS",
    "RELATIVE_IDENTIFYING_SHARE",
    "GraphFamilyType",
    "GraphPenalty",
    "GraphSolution",
    "InterconnectionGraph",
    "MonthAxis",
    "NodeDesign",
    "build_node_design",
    "degree_preserving_permutation",
    "edge_similarity_ratio",
    "family_scales",
    "gam_fits_from_solution",
    "heating_band_slopes",
    "holiday_level",
    "identified_columns",
    "joint_month_counts",
    "kappa_for_neighbour_correlation",
    "latest_capacity",
    "mean_neighbour_correlation",
    "month_axis_of",
    "month_cross_products",
    "profile_scale",
    "ring_with_chords",
    "sample_smooth_fields",
    "smooth_field_covariance",
    "solve_graph_fit",
]
