"""The synthetic fleets of the known-answer stage (S1) and the arms that run on them.

A fleet is 15 synthetic substations: three copies of each of the five decision substations' real
design (calendar, weather, and PV shape, all from rows before the 2024-07-01 cut), joined on a ring
with 6 random chords, 4 of them (random) trained on dark rows only. Each node gets a truth whose
heating slopes, profile scale, capacity, and holiday level follow one of the truth cells, and a
noise made of whole months of real residuals, drawn jointly across the fleet.

**Nothing after the 2024-07-01 cut enters the generator.** The real fits behind the truths, the
donor residuals, and the designs come from the 24 months before it.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final

import e00117_common as common
import numpy as np
import polars as pl
import torch
from studies.dp_features import FeatureCache
from studies.dp_gam import N_BANDS
from studies.dp_gam_graph import (
    GraphPenalty,
    InterconnectionGraph,
    MonthAxis,
    NodeDesign,
    build_node_design,
    family_scales,
    heating_band_slopes,
    holiday_level,
    joint_month_counts,
    kappa_for_neighbour_correlation,
    latest_capacity,
    month_axis_of,
    month_cross_products,
    profile_scale,
    ring_with_chords,
    sample_smooth_fields,
    solve_graph_fit,
)
from studies.dp_gam_pooled import PooledSite, centring_matrix

N_COPIES: Final[int] = 3
N_CHORDS: Final[int] = 6
N_DARK_NODES: Final[int] = 4
N_BOOTSTRAP: Final[int] = 40
SMOOTH_CORRELATION: Final[float] = 0.8
"""The mean correlation across an edge of the smooth truth cell."""
RELATIVE_SD: Final[float] = 0.6 / np.sqrt(12.0)
"""The standard deviation of a factor drawn from U(0.7, 1.3), which experiment 113 draws."""
HEATING_SPLIT: Final[tuple[float, ...]] = (0.2, 0.1, 0.3, 0.4)
"""Each band's total heating slope split over the 6, 24, 72, and 168 hour smoothings."""
MIN_BAND_SLOPE: Final[float] = 2.0
"""The floor of a truth band slope, in points per 10 degrees C."""
CAPACITY_FIRST_MEAN: Final[float] = 15.0
CAPACITY_FIRST_SD: Final[float] = 20.0 / np.sqrt(12.0)
"""The first increment follows U(5, 25) in mean and spread."""
CAPACITY_REST_RANGE: Final[tuple[float, float]] = (1.0, 5.0)
HOLIDAY_OFFSET_SHARE: Final[float] = 0.25
"""The holiday offset's standard deviation as a share of the fleet holiday coefficients' RMS."""
INTERVAL_LOW: Final[float] = 0.1
INTERVAL_HIGH: Final[float] = 0.9
INFLATE: Final[float] = float(np.sqrt(6.0 / 5.0))
"""The month-block interval is widened by sqrt(m / (m - 1)) with m = 6 months per quarter."""
MIN_DONOR_COVERAGE: Final[float] = 0.25
"""A donor month is a candidate only if this share of its half-hours hold a residual."""
HALF_HOURS_PER_DAY: Final[int] = 48
FIELD_COUNT: Final[int] = N_BANDS + 3
"""Fields per fleet: one per heating band, the profile scale, the capacity, and the holiday."""

CELL_ARMS: Final[dict[str, tuple[str, ...]]] = {
    "smooth": ("g", "n", "h_lo", "nh_lo", "h_hi", "nh_hi"),
    "unlike": ("g", "n", "h_lo", "nh_lo", "h_hi", "nh_hi", "h_planted"),
    "wrong": ("g", "n", "h_lo", "nh_lo"),
    "corr03": ("g", "h_lo"),
    "corr06": ("g", "h_lo"),
}
"""The arms each truth cell runs; `lo` and `hi` are the tuned gamma and the next grid value."""
CELL_NOISE_RHO: Final[dict[str, float]] = {"corr03": 0.3, "corr06": 0.6}
"""The shared noise component of the report-only cells."""


@dataclass(frozen=True)
class Templates:
    """The real designs and real fits the generator builds on.

    Attributes:
        full: The five decision sites' full-window designs, by site.
        dark: The same sites' dark-only designs, by site.
        real_theta: The real per-site fit of each full design, by site.
        residual: The real residual of each site on the grid (NaN elsewhere).
        end_times: The grid's half-hour end times.
        dark_theta: The real fit's coefficients re-laid on each dark design, by site.
    """

    full: dict[str, NodeDesign]
    dark: dict[str, NodeDesign]
    real_theta: dict[str, np.ndarray]
    residual: dict[str, np.ndarray]
    end_times: np.ndarray
    dark_theta: dict[str, np.ndarray]


def _site(*, label: str, dark: bool) -> PooledSite:
    from studies.dp_gam import GamSpec

    return PooledSite(
        site=label,
        spec=GamSpec(ridge=common.BASE_RIDGE, capacity="fitted"),
        window=common.DARK_WINDOW if dark else common.FULL_WINDOW,
        season_years=common.SEASON_YEARS,
    )


def build_templates(*, cache: FeatureCache) -> Templates:
    """Build the designs and the real fits of the five decision sites at the 2024-07-01 cut.

    Args:
        cache: The feature cache.

    Returns:
        The templates.
    """
    cut = common.REPLICATE_CUT
    full = {
        s: build_node_design(cache=cache, site=_site(label=s, dark=False), cut=cut)
        for s in common.DECISION_SITES
    }
    dark = {
        s: build_node_design(cache=cache, site=_site(label=s, dark=True), cut=cut)
        for s in common.DECISION_SITES
    }
    graph = InterconnectionGraph(nodes=common.DECISION_SITES, edges=())
    designs = list(full.values())
    solution = solve_graph_fit(
        designs=designs,
        graph=graph,
        penalty=GraphPenalty.none(),
        scales=family_scales(designs=designs),
    )
    real_theta = {s: solution.thetas[s][0] for s in common.DECISION_SITES}
    residual: dict[str, np.ndarray] = {}
    for site, design in full.items():
        target = cache.sites[site].y[design.index]
        grid = np.full(len(cache.end_times), np.nan)
        grid[design.index] = target - design.x @ real_theta[site]
        residual[site] = grid
    dark_theta = {s: _relay(theta=real_theta[s], source=full[s], target=dark[s]) for s in full}
    return Templates(
        full=full,
        dark=dark,
        real_theta=real_theta,
        residual=residual,
        end_times=cache.end_times,
        dark_theta=dark_theta,
    )


def _relay(*, theta: np.ndarray, source: NodeDesign, target: NodeDesign) -> np.ndarray:
    """Copy a coefficient vector between layouts, family by family, leaving capacity at zero."""
    out = np.zeros(target.layout.n_columns)
    for name, columns in source.layout.slices.items():
        other = target.layout.slices.get(name)
        if other is None or name == "capacity":
            continue
        if columns.stop - columns.start == other.stop - other.start:
            out[other] = theta[columns]
    return out


@dataclass(frozen=True)
class Fleet:
    """One synthetic fleet: its graph, nodes, truths, and noises.

    Attributes:
        graph: The synthetic graph the arms are fitted with.
        designs: One design per node, labelled by the node.
        truth: Each node's true coefficients.
        month_rhs: Each node's right-hand side per month: the truth's plus the noise's.
        axis: The fleet's month axis.
        dark: The nodes that train on dark rows only.
        source: Each node's real decision site.
        noise_sd: The noise's standard deviation per node, in points.
    """

    graph: InterconnectionGraph
    designs: list[NodeDesign]
    truth: dict[str, np.ndarray]
    month_rhs: dict[str, torch.Tensor]
    axis: MonthAxis
    dark: frozenset[str]
    source: dict[str, str]
    noise_sd: dict[str, float]


def node_labels() -> tuple[str, ...]:
    """Return the synthetic node labels, N00 to N14 (they name no real substation)."""
    return tuple(f"N{k:02d}" for k in range(N_COPIES * len(common.DECISION_SITES)))


def _coverage_table(*, templates: Templates) -> dict[tuple[str, int], float]:
    """Return the share of each (site, month) whose half-hours hold a residual."""
    table: dict[tuple[str, int], float] = {}
    months = np.unique(templates.end_times.astype("datetime64[M]"))
    for site, residual in templates.residual.items():
        finite = np.isfinite(residual)
        keys = templates.end_times.astype("datetime64[M]")
        for month in months:
            rows = keys == month
            if not rows.any():
                continue
            days = (month + 1).astype("datetime64[D]") - month.astype("datetime64[D]")
            held = float(finite[rows].sum())
            table[(site, int(month.astype(np.int64)))] = held / (
                days.astype(int) * HALF_HOURS_PER_DAY
            )
    return table


def draw_noise(
    *,
    templates: Templates,
    coverage: Mapping[tuple[str, int], float],
    designs: Mapping[str, NodeDesign],
    source: Mapping[str, str],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Draw real-residual noise jointly across a fleet.

    Each calendar month of the fleet takes one donor month and one sign for every node. Each node
    then takes its own donor substation, never the substation its design comes from.

    Args:
        templates: The real residuals.
        coverage: The share of each (site, month) that holds a residual.
        designs: The nodes' designs, by node label.
        source: Each node's real decision site (empty string for no exclusion).
        rng: The random generator.

    Returns:
        For each node, the noise on its design's rows, in points.
    """
    axis = month_axis_of(designs=list(designs.values()))
    end = templates.end_times
    out = {label: np.zeros(design.n_rows) for label, design in designs.items()}
    for month in axis.months:
        options = [
            month + np.timedelta64(12 * years, "M")
            for years in range(-7, 8)
            if sum(
                coverage.get((s, int((month + 12 * years).astype(np.int64))), 0.0)
                >= MIN_DONOR_COVERAGE
                for s in templates.residual
            )
            >= 2
        ]
        if not options:
            continue
        donor_month = options[rng.integers(len(options))]
        sign = 1.0 if rng.random() < 0.5 else -1.0
        key = int(donor_month.astype(np.int64))
        for label, design in designs.items():
            donors = [
                s
                for s in templates.residual
                if s != source[label] and coverage.get((s, key), 0.0) >= MIN_DONOR_COVERAGE
            ]
            if not donors:
                continue
            donor = donors[rng.integers(len(donors))]
            position = int(np.searchsorted(design.months, month))
            if position >= len(design.months) or design.months[position] != month:
                continue
            rows = np.flatnonzero(design.month_of_row == position)
            offset = end[design.index[rows]] - month.astype("datetime64[us]")
            wanted = donor_month.astype("datetime64[us]") + offset
            slot = np.clip(np.searchsorted(end, wanted), 0, len(end) - 1)
            exact = (end[slot] == wanted) & (end[slot].astype("datetime64[M]") == donor_month)
            values = np.where(exact, templates.residual[donor][slot], np.nan)
            out[label][rows] = sign * np.where(np.isfinite(values), values, 0.0)
    return out


def _fields(
    *,
    kind: str,
    graph_true: InterconnectionGraph,
    rng: np.random.Generator,
    nodes: Sequence[str],
) -> np.ndarray:
    """Return the standardised fields, shape (nodes, FIELD_COUNT), for a truth cell."""
    if kind == "unlike":
        return rng.normal(size=(len(nodes), FIELD_COUNT))
    if kind == "wrong":
        shuffled = list(nodes)
        rng.shuffle(shuffled)
        other = ring_with_chords(nodes=shuffled, n_chords=N_CHORDS, rng=rng)
        kappa = kappa_for_neighbour_correlation(graph=other, target=SMOOTH_CORRELATION)
        smooth = sample_smooth_fields(graph=other, kappa=kappa, n_fields=FIELD_COUNT, rng=rng)
        order = [other.nodes.index(node) for node in nodes]
        return smooth[order]
    kappa = kappa_for_neighbour_correlation(graph=graph_true, target=SMOOTH_CORRELATION)
    smooth = sample_smooth_fields(graph=graph_true, kappa=kappa, n_fields=FIELD_COUNT, rng=rng)
    order = [graph_true.nodes.index(node) for node in nodes]
    return smooth[order]


def _truth_theta(
    *,
    design: NodeDesign,
    own_theta: np.ndarray,
    fleet_theta: np.ndarray,
    shape: np.ndarray,
    totals: np.ndarray,
    field: np.ndarray,
    rest: np.ndarray,
    holiday_sd: float,
) -> np.ndarray:
    """Return one node's true coefficients from the fleet means and its field values."""
    layout = design.layout
    theta = own_theta.copy()
    heating = layout.slices["heating"]
    band_total = np.maximum(MIN_BAND_SLOPE, totals * (1.0 + RELATIVE_SD * field[:N_BANDS]))
    for smoothing, share in enumerate(HEATING_SPLIT):
        theta[heating.start + smoothing * N_BANDS : heating.start + (smoothing + 1) * N_BANDS] = (
            band_total * share
        )
    profile = layout.slices["profile"]
    size = profile.stop - profile.start
    centre = centring_matrix(family="profile", size=size)
    band_means = own_theta[profile] - centre @ own_theta[profile]
    theta[profile] = band_means + (1.0 + RELATIVE_SD * field[N_BANDS]) * shape
    holiday = layout.slices["holiday"]
    theta[holiday] = fleet_theta + holiday_sd * field[N_BANDS + 2]
    capacity = layout.slices["capacity"]
    increments = np.zeros(capacity.stop - capacity.start)
    held = np.flatnonzero(design.identified[capacity])
    if len(held):
        increments[held[0]] = max(1.0, CAPACITY_FIRST_MEAN + CAPACITY_FIRST_SD * field[N_BANDS + 1])
        increments[held[1:]] = rest[: len(held) - 1]
    theta[capacity] = increments
    return theta


def draw_fleet(
    *,
    templates: Templates,
    coverage: Mapping[tuple[str, int], float],
    kind: str,
    seed: tuple[int, ...],
    noise_rho: float = 0.0,
) -> Fleet:
    """Draw one fleet of a truth cell.

    Args:
        templates: The real designs and fits.
        coverage: The donor coverage table.
        kind: `smooth`, `unlike`, or `wrong` (the truth cell's field).
        seed: Entropy for the fleet, such as `(set, cell, index)`.
        noise_rho: The weight of a noise component shared by all nodes (a report-only cell).

    Returns:
        The fleet.
    """
    rng = np.random.default_rng(np.random.SeedSequence([common.EXPERIMENT_ID, *seed]))
    nodes = node_labels()
    graph = ring_with_chords(nodes=nodes, n_chords=N_CHORDS, rng=rng)
    dark = frozenset(rng.choice(nodes, size=N_DARK_NODES, replace=False).tolist())
    source = {label: common.DECISION_SITES[k // N_COPIES] for k, label in enumerate(nodes)}
    designs = {}
    for label in nodes:
        template = (templates.dark if label in dark else templates.full)[source[label]]
        designs[label] = replace(template, inputs=replace(template.inputs, site=label))
    fields = _fields(kind=kind, graph_true=graph, rng=rng, nodes=nodes)
    sites = common.DECISION_SITES
    totals = np.mean(
        [
            heating_band_slopes(
                layout=templates.full[s].layout, theta=templates.real_theta[s][None]
            )[0]
            for s in sites
        ],
        axis=0,
    )
    shapes = []
    for s in sites:
        columns = templates.full[s].layout.slices["profile"]
        centre = centring_matrix(family="profile", size=columns.stop - columns.start)
        shapes.append(centre @ templates.real_theta[s][columns])
    shape = np.mean(shapes, axis=0)
    holiday_columns = templates.full[sites[0]].layout.slices["holiday"]
    fleet_holiday = np.mean([templates.real_theta[s][holiday_columns] for s in sites], axis=0)
    holiday_sd = HOLIDAY_OFFSET_SHARE * float(np.sqrt(np.mean(fleet_holiday**2)))
    rest = rng.uniform(*CAPACITY_REST_RANGE, size=len(common.SEASON_YEARS))
    truth = {}
    for k, label in enumerate(nodes):
        own = (templates.dark_theta if label in dark else templates.real_theta)[source[label]]
        truth[label] = _truth_theta(
            design=designs[label],
            own_theta=own,
            fleet_theta=fleet_holiday,
            shape=shape,
            totals=totals,
            field=fields[k],
            rest=rest,
            holiday_sd=holiday_sd,
        )
    noise = draw_noise(
        templates=templates, coverage=coverage, designs=designs, source=source, rng=rng
    )
    if noise_rho > 0.0:
        shared = draw_noise(
            templates=templates,
            coverage=coverage,
            designs=designs,
            source=dict.fromkeys(nodes, ""),
            rng=rng,
        )
        noise = {
            label: np.sqrt(noise_rho) * shared[label] + np.sqrt(1.0 - noise_rho) * noise[label]
            for label in nodes
        }
    month_rhs = {}
    for label in nodes:
        design = designs[label]
        gram_truth = torch.einsum(
            "mpq,q->mp", design.month_gram, torch.as_tensor(truth[label], dtype=torch.float64)
        )
        month_rhs[label] = gram_truth + month_cross_products(design=design, target=noise[label])
    return Fleet(
        graph=graph,
        designs=list(designs.values()),
        truth=truth,
        month_rhs=month_rhs,
        axis=month_axis_of(designs=list(designs.values())),
        dark=dark,
        source=source,
        noise_sd={label: float(noise[label].std()) for label in nodes},
    )


def arm_penalty(*, arm: str, gamma_lo: float, gamma_hi: float) -> GraphPenalty:
    """Return the penalty of an arm name.

    Args:
        arm: `g`, `n`, `h_lo`, `nh_lo`, `h_hi`, `nh_hi`, `h_planted`, or `h@<gamma>` (the graph
            penalty alone at a stated weight), `heat@<gamma>` or `hol@<gamma>` (the graph
            penalty on the heating or the holiday family alone).
        gamma_lo: The tuned graph weight.
        gamma_hi: The next grid value above it.

    Returns:
        The penalty.

    Raises:
        ValueError: If the arm name is unknown.
    """
    table = {
        "g": GraphPenalty.none(),
        "n": GraphPenalty.of(fleet_rows=common.FLEET_ROWS),
        "h_lo": GraphPenalty.of(graph_rows=gamma_lo),
        "nh_lo": GraphPenalty.of(fleet_rows=common.FLEET_ROWS, graph_rows=gamma_lo),
        "h_hi": GraphPenalty.of(graph_rows=gamma_hi),
        "nh_hi": GraphPenalty.of(fleet_rows=common.FLEET_ROWS, graph_rows=gamma_hi),
        "h_planted": GraphPenalty.of(graph_rows=common.PLANTED_GAMMA),
    }
    if arm.startswith("h@"):
        return GraphPenalty.of(graph_rows=float(arm[2:]))
    if arm.startswith("heat@"):
        return GraphPenalty(fleet={}, graph={"heating": float(arm[5:])})
    if arm.startswith("hol@"):
        return GraphPenalty(fleet={}, graph={"holiday": float(arm[4:])})
    if arm not in table:
        msg = f"unknown arm {arm}"
        raise ValueError(msg)
    return table[arm]


FUNCTIONALS: Final[tuple[str, ...]] = (
    "slope_0",
    "slope_1",
    "slope_2",
    "slope_3",
    "scale",
    "capacity",
    "holiday",
)
"""The parameters whose intervals the gate reads."""
WIDTH_FAMILY: Final[dict[str, str]] = {
    **{f"slope_{b}": "slope" for b in range(N_BANDS)},
    "scale": "scale",
    "capacity": "capacity",
    "holiday": "holiday",
}
"""The family of each functional; slope, scale, and capacity are the three width families."""


def functional_values(*, design: NodeDesign, theta: np.ndarray) -> np.ndarray:
    """Return the functionals of coefficients of shape (batch, p), shape (batch, 7)."""
    slopes = heating_band_slopes(layout=design.layout, theta=theta)
    return np.column_stack(
        [
            slopes,
            profile_scale(layout=design.layout, theta=theta),
            latest_capacity(layout=design.layout, theta=theta),
            holiday_level(layout=design.layout, theta=theta),
        ]
    )


def identified_functionals(*, design: NodeDesign) -> np.ndarray:
    """Return which functionals the node's rows identify, shape (7,)."""
    layout = design.layout
    heating = layout.slices["heating"]
    flags = []
    for band in range(N_BANDS):
        columns = [heating.start + s * N_BANDS + band for s in range(len(HEATING_SPLIT))]
        flags.append(bool(design.identified[columns].all()))
    flags.append(bool(design.identified[layout.slices["profile"]].all()))
    flags.append(bool(design.identified[layout.slices["capacity"]].any()))
    flags.append(bool(design.identified[layout.slices["holiday"]].mean() >= 0.5))
    return np.array(flags)


def run_arm(
    *,
    fleet: Fleet,
    arm: str,
    gamma_lo: float,
    gamma_hi: float,
    counts: np.ndarray,
    graph: InterconnectionGraph | None = None,
    device: str = "cpu",
) -> pl.DataFrame:
    """Fit an arm on a fleet for the full sample and every resample, and read the intervals.

    Args:
        fleet: The fleet.
        arm: An arm name of `arm_penalty`.
        gamma_lo: The tuned graph weight.
        gamma_hi: The next grid value above it.
        counts: Joint month counts, full sample first.
        graph: The graph to fit with; by default the fleet's own (the wrong-graph cell passes the
            fleet's graph too, because its truth was drawn on another).
        device: Where the coupled solve runs, `cpu` or `cuda`.

    Returns:
        One row per node and functional: truth, point, the 80% interval, and whether the node
        identifies the functional.
    """
    designs = fleet.designs
    solution = solve_graph_fit(
        designs=designs,
        graph=fleet.graph if graph is None else graph,
        penalty=arm_penalty(arm=arm, gamma_lo=gamma_lo, gamma_hi=gamma_hi),
        scales=family_scales(designs=designs),
        axis=fleet.axis,
        counts=counts,
        month_rhs=fleet.month_rhs,
        device=device,
    )
    rows = []
    for design in designs:
        label = design.site
        values = functional_values(design=design, theta=solution.thetas[label])
        truth = functional_values(design=design, theta=fleet.truth[label][None])[0]
        flags = identified_functionals(design=design)
        median = np.median(values[1:], axis=0)
        low, high = np.quantile(values[1:], [INTERVAL_LOW, INTERVAL_HIGH], axis=0)
        for k, name in enumerate(FUNCTIONALS):
            rows.append(
                {
                    "arm": arm,
                    "node": label,
                    "dark": label in fleet.dark,
                    "functional": name,
                    "family": WIDTH_FAMILY[name],
                    "identified": bool(flags[k]),
                    "truth": float(truth[k]),
                    "point": float(values[0, k]),
                    "lo": float(median[k] - INFLATE * (median[k] - low[k])),
                    "hi": float(median[k] + INFLATE * (high[k] - median[k])),
                    "violation": float(solution.violation[0]),
                }
            )
    return pl.DataFrame(rows)


def fleet_counts(*, fleet: Fleet, seed: tuple[int, ...]) -> np.ndarray:
    """Return the joint month counts of a fleet: the full sample, then `N_BOOTSTRAP` resamples."""
    rng = np.random.default_rng(np.random.SeedSequence([common.EXPERIMENT_ID, 7, *seed]))
    return joint_month_counts(axis=fleet.axis, n_resamples=N_BOOTSTRAP, rng=rng)
