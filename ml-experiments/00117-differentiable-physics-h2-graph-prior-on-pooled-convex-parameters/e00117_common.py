"""Constants and helpers shared by the scripts of experiment 117 (DP H2, a graph prior).

The experiment fits the joint convex model of `studies.dp_gam_graph` at three cuts: 2023-07-01
(tuning), 2024-07-01 (replicate), and 2025-07-01 (scored). It scores the 11 quantiles of experiment
87 as the quantile score (twice the mean pinball loss) in points of effective capacity, on the rows
outside the hand-filtered switching events and outside the clock-change days.

**Lead.** A frozen fit on past weather does not depend on an initialisation, so each valid time is
written once, against a day-ahead initialisation at 09:00 UTC on the previous day (leads of 15.5 to
39 hours), as experiment 109 does.

**Privacy.** The two series of the plan's "G-c" are generator data. They are in `EXCLUDED_SERIES`
and never enter a fit, a graph, or an output of this experiment.
"""

import calendar
import hashlib
import subprocess
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final, cast

import numpy as np
import polars as pl
from studies.dp_features import FeatureCache, default_input_paths, load_or_build_feature_cache
from studies.dp_gam import (
    QUANTILE_COLUMNS,
    QUANTILE_LEVELS,
    GamSpec,
    TrainingWindow,
    training_rows,
)
from studies.dp_gam_graph import InterconnectionGraph
from studies.dp_gam_pooled import (
    PooledSite,
    rebuilt_effective_capacity_mw,
    with_effective_capacity,
)
from studies.dp_pinned_capacity import pinned_capacity
from studies.experiment_store import HOLDOUT_START, write_predictions
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

EXPERIMENT_ID: Final[int] = 117
SLUG: Final[str] = "differentiable-physics-h2-graph-prior-on-pooled-convex-parameters"
ROW_SET_ID: Final[str] = "dp_mw_v1"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
"""Small result files (tuning, gate, reports); private and never committed."""
CACHE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
"""The feature cache that experiment 106 built; the same inputs hash to the same file."""
HIERARCHY_DIR: Final[Path] = REPO_DATA_DIR / "NGED" / "substation_hierarchy_2026-10-15"
"""Private topology files; never commit anything read from here."""
SERIES_PATH: Final[Path] = (
    REPO_DATA_DIR / "studies/per_study/stage1_pilot/demand/inputs/series.parquet"
)

DECISION_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
"""The five sites that carry the verdicts."""
PARTIAL_SITES: Final[tuple[str, ...]] = ("D9",)
"""A site reported beside the decision sites; it holds part of a window at the earlier cuts."""
EXCLUDED_SERIES: Final[tuple[str, ...]] = ("D15", "D18")
"""The two series labelled G-c on the page: never in a graph, a fit, or an output."""
EXPECTED_EDGES: Final[int] = 21
"""The decision graph's metered edges that do not touch an excluded series."""

TUNING_CUT: Final[datetime] = datetime(2023, 7, 1, tzinfo=UTC)
REPLICATE_CUT: Final[datetime] = datetime(2024, 7, 1, tzinfo=UTC)
SCORED_CUT: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
CUTS: Final[dict[str, datetime]] = {
    "tuning": TUNING_CUT,
    "replicate": REPLICATE_CUT,
    "scored": SCORED_CUT,
}
SEASON_YEARS: Final[tuple[int, ...]] = tuple(range(2019, 2026))
FULL_WINDOW: Final[TrainingWindow] = TrainingWindow(months=24)
"""Two years before each cut, no drift, for every arm."""
DARK_WINDOW: Final[TrainingWindow] = TrainingWindow(months=24, dark_only=True)
"""An MVA node trains on dark rows only."""
BASE_RIDGE: Final[float] = 0.1
"""The ridge of every substation: experiment 104's tuned value."""
PARTIAL_HISTORY_ROWS: Final[int] = 28_000
"""A site with fewer usable rows in its two-year window holds only part of it, and its points of
effective capacity are rebuilt from the rows it holds."""
MIN_TRAINING_ROWS: Final[int] = 1344
"""A site enters a cut only with at least four weeks of usable rows."""
MIN_DARK_SHARE: Final[float] = 0.5
"""An MVA node joins a cut only if it holds at least this share of its window's dark half-hours,
which is 12 months of dark rows in a 24-month window."""

FLEET_ROWS: Final[float] = 3_000.0
"""Experiment 109's fleet-mean penalty, in rows of evidence."""
GAMMA_GRID: Final[tuple[float, ...]] = (0.0, 300.0, 1_000.0, 3_000.0, 10_000.0, 30_000.0)
"""The graph weights tried on the tuning year, in rows of evidence per edge."""
PLANTED_GAMMA: Final[float] = 100_000.0
"""The planted failure's graph weight, off the grid so that a tuned gamma cannot hide it."""
LAMBDA_HOLIDAY_GRID: Final[tuple[float, ...]] = (0.0, 300.0, 1_000.0, 3_000.0, 10_000.0)
"""The holiday-only fleet-mean weights tried on the tuning year."""

HOURS_AHEAD: Final[int] = 9
"""The day-ahead initialisation hour (UTC)."""
WINTER_MONTHS: Final[tuple[int, ...]] = (11, 12, 1, 2, 3)
"""The winter rows are valid times in these months."""
EVENING_PEAK_STARTS: Final[tuple[float, float]] = (16.0, 19.5)
"""The evening peak is the half-hours that start from 16:00 to 19:30 UTC."""
COLD_SNAP_QUANTILE: Final[float] = 0.10
"""A cold-snap day has a mean temperature at or below this quantile of winter daily means."""
MIN_COLD_SNAP_DAYS: Final[int] = 5
"""A site-year with fewer cold-snap days is reported as "too few days"."""


def load_cache() -> FeatureCache:
    """Return the feature cache, building and saving it on the first call."""
    return load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)


def to_64(*, moment: datetime) -> np.datetime64:
    """Return a timezone-aware datetime as a naive UTC `datetime64[us]`."""
    return np.datetime64(moment.replace(tzinfo=None), "us")


def year_end(*, cut: datetime) -> datetime:
    """Return the end of the scored year that starts at `cut`, never past the held-out start."""
    return min(cut.replace(year=cut.year + 1), HOLDOUT_START)


def private_edges() -> tuple[tuple[str, str], ...]:
    """Read the private interconnection file and return its distinct pairs as D-label pairs.

    Returns:
        Sorted pairs, the smaller label first, among the series in the series table.
    """
    series = pl.read_parquet(SERIES_PATH).select("time_series_id", "site")
    monitors = pl.read_csv(HIERARCHY_DIR / "Monitor Direction.csv").select(
        "TimeSeriesID", "SubstationNumber"
    )
    label = {
        row[0]: row[1]
        for row in series.join(monitors, left_on="time_series_id", right_on="TimeSeriesID")
        .select("SubstationNumber", "site")
        .rows()
    }
    pairs = pl.read_csv(HIERARCHY_DIR / "Primary Substation Interconnections.csv").rows()
    return tuple(
        sorted(
            {
                (min(label[a], label[b]), max(label[a], label[b]))
                for a, b in pairs
                if a in label and b in label and label[a] != label[b]
            }
        )
    )


def decision_edges(*, cache: FeatureCache) -> tuple[tuple[str, str], ...]:
    """Return the decision graph's edges: the interconnections that touch no excluded series.

    Args:
        cache: The cache, whose sites bound the nodes.

    Returns:
        The sorted pairs.

    Raises:
        ValueError: If the count differs from `EXPECTED_EDGES`, which means the interconnection file
            is not the one the plan counted.
    """
    edges = tuple(
        (a, b)
        for a, b in private_edges()
        if a in cache.sites
        and b in cache.sites
        and a not in EXCLUDED_SERIES
        and b not in EXCLUDED_SERIES
    )
    if len(edges) != EXPECTED_EDGES:
        msg = f"the decision graph holds {len(edges)} edges, not {EXPECTED_EDGES}"
        raise ValueError(msg)
    return edges


def edge_list_digest(*, edges: tuple[tuple[str, str], ...]) -> str:
    """Return a short hash of an edge list, so a stored result names the graph it used."""
    text = ";".join(f"{a}-{b}" for a, b in edges)
    return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:10]


@dataclass(frozen=True)
class NodePlan:
    """The nodes of one cut, ready for `studies.dp_gam_graph.build_node_design`.

    Attributes:
        cut: The cut.
        sites: The substations of the joint fit, in graph order.
        cache: The cache with the scale of a partial-window or MVA node rebuilt.
        graph: The decision graph restricted to the nodes that join this cut.
        scale: For each node whose scale was rebuilt, the original points divided by the rebuilt
            points; a forecast in rebuilt points is divided by it to return to original points.
        capacity_sources: How each node's capacity was set: `pinned`, `fitted_late_vintage`, or
            `fitted_dark`.
        left_out: The graph nodes that did not join, with the reason.
    """

    cut: datetime
    sites: tuple[PooledSite, ...]
    cache: FeatureCache
    graph: InterconnectionGraph
    scale: dict[str, float]
    capacity_sources: dict[str, str]
    left_out: dict[str, str]


def _window_start(*, cut: datetime, months: int) -> np.datetime64:
    """Return the instant `months` calendar months before `cut` as `datetime64[us]`."""
    month = to_64(moment=cut).astype("datetime64[M]") - months
    return month.astype("datetime64[us]")


def dark_half_hours_in_window(*, cache: FeatureCache, site: str, cut: datetime) -> int:
    """Return the dark half-hours of the site in the 24 months before the cut, readings or not."""
    start = _window_start(cut=cut, months=24)
    rows = (cache.end_times > start) & (cache.end_times <= to_64(moment=cut))
    return int(np.sum(cache.sites[site].dark & rows))


def prepare_nodes(
    *,
    cache: FeatureCache,
    vintages: pl.DataFrame,
    cut: datetime,
    labels: tuple[str, ...] | None = None,
    edges: tuple[tuple[str, str], ...] | None = None,
    allow_excluded: tuple[str, ...] = (),
) -> NodePlan:
    """Choose the nodes that join a cut and build their inputs.

    A MW node joins with at least `MIN_TRAINING_ROWS` usable rows in its window. An MVA node joins
    with at least `MIN_DARK_SHARE` of its window's dark half-hours as usable dark rows. A MW node
    that holds fewer than `PARTIAL_HISTORY_ROWS` rows, and every MVA node, has its points of
    effective capacity rebuilt from the rows it holds, so no row after the cut sets a scale.

    Args:
        cache: The cache in original points.
        vintages: The real-time capacity vintages.
        cut: The cut.
        labels: The candidate nodes; by default every node of the decision edges.
        edges: The graph's edges; by default `decision_edges`.
        allow_excluded: Excluded series that may be listed in `labels`, for the reproduction
            check of experiment 109 only.

    Returns:
        The plan.

    Raises:
        ValueError: If `labels` names an excluded series that `allow_excluded` does not allow.
    """
    chosen_edges = decision_edges(cache=cache) if edges is None else edges
    candidates = labels or tuple(sorted({node for edge in chosen_edges for node in edge}))
    for label in candidates:
        if label in EXCLUDED_SERIES and label not in allow_excluded:
            msg = f"{label} is an excluded series and may not join a fit"
            raise ValueError(msg)
    used = cache
    sites: list[PooledSite] = []
    scale: dict[str, float] = {}
    sources: dict[str, str] = {}
    left_out: dict[str, str] = {}
    for label in candidates:
        features = used.sites[label]
        mva = features.table.units == "MVA"
        window = DARK_WINDOW if mva else FULL_WINDOW
        rows = training_rows(features=features, cache=used, cut=cut, window=window)
        if mva:
            dark_total = dark_half_hours_in_window(cache=used, site=label, cut=cut)
            if len(rows) < MIN_DARK_SHARE * max(dark_total, 1):
                left_out[label] = f"{len(rows)} dark rows of {dark_total}"
                continue
        elif len(rows) < MIN_TRAINING_ROWS:
            left_out[label] = f"{len(rows)} usable rows"
            continue
        if mva or len(rows) < PARTIAL_HISTORY_ROWS:
            rebuilt = rebuilt_effective_capacity_mw(cache=used, site=label, cut=cut, window=window)
            scale[label] = features.table.effective_capacity_mw / rebuilt
            used = with_effective_capacity(cache=used, site=label, capacity_mw=rebuilt)
        pinned = (
            None
            if mva
            else pinned_capacity(vintages=vintages, table=used.sites[label].table, cut=cut)
        )
        use_pinned = pinned is not None and pinned.source != "growth_prior"
        sources[label] = "fitted_dark" if mva else "pinned" if use_pinned else "fitted_late_vintage"
        sites.append(
            PooledSite(
                site=label,
                spec=GamSpec(ridge=BASE_RIDGE, capacity="pinned" if use_pinned else "fitted"),
                window=window,
                season_years=SEASON_YEARS,
                pinned=pinned if use_pinned else None,
            )
        )
    joined = tuple(site.site for site in sites)
    graph = InterconnectionGraph.from_pairs(nodes=joined, pairs=list(chosen_edges))
    return NodePlan(
        cut=cut,
        sites=tuple(sites),
        cache=used,
        graph=graph,
        scale=scale,
        capacity_sources=sources,
        left_out=left_out,
    )


def last_sunday(*, year: int, month: int) -> date:
    """Return the date of the last Sunday of a month."""
    last = calendar.monthrange(year, month)[1]
    day = date(year, month, last)
    return day - timedelta(days=(day.weekday() + 1) % 7)


def clock_change_days(*, cut: datetime) -> tuple[date, ...]:
    """Return the days excluded from the scored rows of the year that starts at `cut`.

    These are the autumn clock-change day, the spring clock-change day, and the day before it, in
    UTC dates, as the plan fixes them (26 October 2025, 28 and 29 March 2026 for the scored year).

    Args:
        cut: The cut, on 1 July.

    Returns:
        The three dates.
    """
    autumn = last_sunday(year=cut.year, month=10)
    spring = last_sunday(year=cut.year + 1, month=3)
    return autumn, spring - timedelta(days=1), spring


def scoring_index(
    *, cache: FeatureCache, site: str, after: datetime, through: datetime, cut: datetime
) -> np.ndarray:
    """Return the grid rows a site is scored on in (`after`, `through`].

    Args:
        cache: The cache.
        site: The substation's label.
        after: Rows must end after this instant.
        through: Rows must end no later than this instant.
        cut: The cut, which names the year whose clock-change days are left out.

    Returns:
        Sorted grid positions: rows with a reading, outside the clock-change days, never at or
        after the held-out start.
    """
    features = cache.sites[site]
    end = cache.end_times
    keep = (
        features.present
        & np.isfinite(features.y)
        & (end > to_64(moment=after))
        & (end <= to_64(moment=min(through, HOLDOUT_START - timedelta(minutes=30))))
        & ~excluded_day_mask(end_times=end, cut=cut)
    )
    return np.flatnonzero(keep)


def start_dates(*, end_times: np.ndarray) -> np.ndarray:
    """Return the UTC date each half-hour starts on, as `datetime64[D]`."""
    return (end_times - np.timedelta64(30, "m")).astype("datetime64[D]")


def excluded_day_mask(*, end_times: np.ndarray, cut: datetime) -> np.ndarray:
    """Return True for the half-hours that start on a clock-change day of the year after `cut`."""
    days = start_dates(end_times=end_times)
    mask = np.zeros(len(end_times), dtype=bool)
    for day in clock_change_days(cut=cut):
        mask |= days == np.datetime64(day, "D")
    return mask


def winter_mask(*, end_times: np.ndarray) -> np.ndarray:
    """Return True for the half-hours that start in November to March (UTC)."""
    months = (end_times - np.timedelta64(30, "m")).astype("datetime64[M]").astype(int) % 12 + 1
    return np.isin(months, WINTER_MONTHS)


def evening_peak_mask(*, end_times: np.ndarray) -> np.ndarray:
    """Return True for the half-hours that start from 16:00 to 19:30 UTC."""
    starts = end_times - np.timedelta64(30, "m")
    hour = (starts - starts.astype("datetime64[D]")) / np.timedelta64(1, "h")
    return (hour >= EVENING_PEAK_STARTS[0]) & (hour <= EVENING_PEAK_STARTS[1])


def cold_snap_days(*, cache: FeatureCache, site: str, cut: datetime) -> np.ndarray:
    """Return the UTC dates of the cold-snap days of the year after `cut`.

    A cold-snap day is a winter day whose mean ERA5 temperature at the site is at or below the
    `COLD_SNAP_QUANTILE` quantile of winter daily means over the 24 months before the cut.

    Args:
        cache: The cache.
        site: The substation's label.
        cut: The cut, which fixes the reference window and the scored year.

    Returns:
        Sorted dates as `datetime64[D]`.
    """
    temperature = cache.sites[site].temperature_c
    days = start_dates(end_times=cache.end_times)
    winter = winter_mask(end_times=cache.end_times) & np.isfinite(temperature)
    reference = (
        winter
        & (cache.end_times > _window_start(cut=cut, months=24))
        & (cache.end_times <= to_64(moment=cut))
    )
    scored = (
        winter
        & (cache.end_times > to_64(moment=cut))
        & (cache.end_times <= to_64(moment=year_end(cut=cut)))
    )

    def daily_means(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        unique, inverse = np.unique(days[rows], return_inverse=True)
        sums = np.bincount(inverse, weights=temperature[rows])
        counts = np.bincount(inverse)
        return unique, sums / counts

    _, reference_means = daily_means(reference)
    threshold = float(np.quantile(reference_means, COLD_SNAP_QUANTILE))
    scored_days, scored_means = daily_means(scored)
    return scored_days[scored_means <= threshold]


def event_day_mask(*, cache: FeatureCache, cut: datetime) -> np.ndarray:
    """Return True for the half-hours of a holiday-period day that is not a clock-change day."""
    return (cache.calendar.holiday_period >= 0) & ~excluded_day_mask(
        end_times=cache.end_times, cut=cut
    )


def day_ahead_initialisation(*, end_times: np.ndarray) -> np.ndarray:
    """Return 09:00 UTC on the day before each half-hour's start day, as `datetime64[us]`."""
    day = start_dates(end_times=end_times)
    return (day - np.timedelta64(1, "D")).astype("datetime64[us]") + np.timedelta64(
        HOURS_AHEAD, "h"
    )


def prediction_frame(
    *,
    cache: FeatureCache,
    site: str,
    index: np.ndarray,
    quantiles: np.ndarray,
    mean: np.ndarray,
    arm: str,
    cut: datetime,
) -> pl.DataFrame:
    """Return one substation's predictions as rows with the columns the scorers read.

    Args:
        cache: The cache (in original points of effective capacity).
        site: The substation's label.
        index: Grid rows predicted.
        quantiles: Shape (len(index), 11), the 11 quantiles in original points.
        mean: The mean forecast the quantiles are built around, in original points.
        arm: The arm's name.
        cut: The cut of the scored year, which fixes the cold-snap days.

    Returns:
        Columns `time_series_id`, `valid_time`, `power_fcst_init_time`, `lead_hours`, `seed`,
        `arm`, `q01` to `q99`, `actual`, `mean`, `switching_mask`, `event_day`, `winter`,
        `evening_peak`, and `cold_snap` (the last four are row labels fixed before scoring).

    Raises:
        ValueError: If the quantiles' shape is wrong.
    """
    features = cache.sites[site]
    if quantiles.shape != (len(index), len(QUANTILE_LEVELS)):
        expected = (len(index), len(QUANTILE_LEVELS))
        msg = f"expected quantiles of shape {expected}, got {quantiles.shape}"
        raise ValueError(msg)
    ordered = np.sort(quantiles, axis=1)
    valid = cache.end_times[index]
    init = day_ahead_initialisation(end_times=valid)
    snap = cold_snap_days(cache=cache, site=site, cut=cut)
    return pl.DataFrame(
        {
            "time_series_id": np.full(len(index), features.table.time_series_id, dtype=np.int32),
            "valid_time": pl.Series(valid).dt.replace_time_zone("UTC"),
            "power_fcst_init_time": pl.Series(init).dt.replace_time_zone("UTC"),
            "lead_hours": ((valid - init) / np.timedelta64(1, "h")).astype(np.float64),
            "seed": np.zeros(len(index), dtype=np.int8),
            "arm": np.full(len(index), arm),
            **{
                column: ordered[:, position].astype(np.float32)
                for position, column in enumerate(QUANTILE_COLUMNS)
            },
            "actual": features.y[index].astype(np.float32),
            "mean": mean.astype(np.float32),
            "switching_mask": features.switching_mask[index],
            "event_day": cache.calendar.holiday_period[index] >= 0,
            "winter": winter_mask(end_times=valid),
            "evening_peak": evening_peak_mask(end_times=valid),
            "cold_snap": np.isin(start_dates(end_times=valid), snap),
        }
    )


def store_arm(*, name: str, frame: pl.DataFrame) -> Path:
    """Write an arm's predictions once; a second call returns the existing folder.

    Args:
        name: The set's name.
        frame: The predictions of every scored substation.

    Returns:
        The set's folder.
    """
    folder = DATA_DIR / "predictions" / name
    if folder.exists():
        return folder
    return write_predictions(
        experiment_id=EXPERIMENT_ID, name=name, frame=frame, kind="quantiles", row_set_id=ROW_SET_ID
    )


def quantile_score(*, actual: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Return twice the mean pinball loss over the 11 levels, per row, in points.

    Args:
        actual: Measured values, shape (n,).
        quantiles: Predicted quantiles, shape (n, 11).

    Returns:
        The quantile score of each row.
    """
    levels = np.array(QUANTILE_LEVELS)[None, :]
    error = actual[:, None] - quantiles
    return 2.0 * np.mean(np.maximum(levels * error, (levels - 1.0) * error), axis=1)


def run_metadata() -> dict[str, str]:
    """Return the commit of the checkout and of the last change to the graph module.

    Returns:
        The two commit hashes, so a result records the code that produced it.
    """

    def last_commit(*arguments: str) -> str:
        return subprocess.run(
            ["git", "log", "-1", "--format=%H", *arguments],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    return {
        "checkout": last_commit(),
        "graph_module": last_commit("--", "packages/studies/src/studies/dp_gam_graph.py"),
    }


def as_float(value: object) -> float:
    """Return a polars scalar (typed as a wide union) as a float."""
    return float(cast("float", value))
