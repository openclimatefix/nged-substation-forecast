"""Plant losses in a record and score how well an estimator recovers them, for issue #1143.

**Stage A of the study plants known capacity losses in the measured power and asks each estimator
to find them.** A loss is a step, a ramp, or an intermittent loss with its own size, and every draw
plants three events (one of each shape) at every plant. This module draws the events, applies them
to a `CapacityRecord`, scores an estimator's recovery, gates out estimators that move without a
planted loss, and picks the estimator by a rule fixed before any forecast is fitted.

**Footprints keep the events apart.** An event's footprint is the 30 days before its month (the
longest estimator look-back), the month, and the 60 days after. Two events at one plant never share
footprint days. An event's month never falls inside another plant's footprint, so a
neighbour-based estimator never reads a neighbour that is itself deflated. No footprint touches the
reference year that calibrates the estimators.
"""

import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, NamedTuple

import numpy as np
import polars as pl
from scipy import stats

from studies.capacity_estimators import (
    ESTIMATOR_FAMILY,
    REFERENCE_END,
    REFERENCE_ESTIMATORS,
    REFERENCE_START,
    CapacityRecord,
    FamilyType,
    IssueScale,
    valid_days,
)
from studies.guards import refuse_to_overwrite

EventShapeType = Literal["step", "ramp", "intermittent"]
"""The three shapes of planted loss."""

EVENT_SHAPES: Final[tuple[EventShapeType, ...]] = ("step", "ramp", "intermittent")
"""Each draw plants one event of each shape at every plant."""

SIZE_RANGE: Final[tuple[float, float]] = (0.05, 0.40)
"""Each event's size, the share of power lost, is drawn uniformly from here."""

FOOTPRINT_BEFORE_DAYS: Final[int] = 30
FOOTPRINT_AFTER_DAYS: Final[int] = 60
"""The days before an event's month and after it that its footprint covers."""

MIN_VALID_SHARE: Final[float] = 0.5
"""A footprint needs at least this share of valid days at its plant."""

MIN_HISTORY_DAYS: Final[int] = 60
"""An event starts at least this many days after its plant's first valid day."""

MAX_PLACEMENT_ATTEMPTS: Final[int] = 400
"""How many random months to try for one event before the draw raises."""

FALSE_MOVEMENT_THRESHOLD: Final[float] = 0.05
"""A day moves falsely when the estimate is more than this share from its trailing median."""

FALSE_MOVEMENT_TRAILING_DAYS: Final[int] = 90
"""The trailing median covers this many days before the day."""

FALSE_MOVEMENT_GATE: Final[float] = 0.10
"""An estimator whose share of falsely moving days exceeds this is out of the selection."""

INTERVAL_Z_90: Final[float] = 1.6449
"""The standard normal quantile of a two-sided 90% interval."""

CONTRAST_CONFIDENCE: Final[float] = 0.975
"""The coverage of each of the two estimator-level contrasts' intervals (Bonferroni over two)."""


@dataclass(frozen=True)
class PlantedEvent:
    """One planted loss.

    Attributes:
        plant: The plant's index in the record.
        shape: `step`, `ramp`, or `intermittent`.
        size: The share of power lost, at the deepest point.
        first_day: The index of the first day of the event's calendar month.
        month_days: The number of days in that month.
        hit_days: For an intermittent loss, the days of the month (from 0) that lose power.
    """

    plant: int
    shape: EventShapeType
    size: float
    first_day: int
    month_days: int
    hit_days: tuple[int, ...] = ()

    @property
    def footprint(self) -> tuple[int, int]:
        """The first day (inclusive) and last day (exclusive) the event's footprint covers."""
        return (
            self.first_day - FOOTPRINT_BEFORE_DAYS,
            self.first_day + self.month_days + FOOTPRINT_AFTER_DAYS,
        )

    @property
    def scored_days(self) -> tuple[int, int]:
        """The first day (inclusive) and last day (exclusive) scored: the month and what follows."""
        return self.first_day, self.first_day + self.month_days + FOOTPRINT_AFTER_DAYS


def event_factor(*, event: PlantedEvent, days: int) -> np.ndarray:
    """Return the share of power left on each day because of one event.

    Args:
        event: The event.
        days: The number of days in the record.

    Returns:
        1 outside the event's month. Inside it, a step is `1 - size`, a ramp falls linearly from 1
        on the first day to `1 - size` on the last, and an intermittent loss is `1 - size` on its
        hit days and 1 on the others.
    """
    factor = np.ones(days)
    month = slice(event.first_day, event.first_day + event.month_days)
    if event.shape == "step":
        factor[month] = 1.0 - event.size
    elif event.shape == "ramp":
        factor[month] = 1.0 - event.size * np.arange(event.month_days) / max(
            event.month_days - 1, 1
        )
    else:
        hit = np.asarray(event.hit_days, dtype=int) + event.first_day
        factor[hit] = 1.0 - event.size
    return factor


def expected_factor(*, event: PlantedEvent, days: int) -> np.ndarray:
    """Return the factor an estimator can be expected to find on each day.

    An intermittent loss is scored against its month's mean, `1 - size / 2`, because no daily
    estimate can know which days were hit. A step and a ramp are scored against their daily factor.

    Args:
        event: The event.
        days: The number of days in the record.

    Returns:
        One value per day.
    """
    factor = event_factor(event=event, days=days)
    if event.shape == "intermittent":
        factor[event.first_day : event.first_day + event.month_days] = 1.0 - event.size / 2.0
    return factor


def factor_matrix(*, events: Sequence[PlantedEvent], plants: int, days: int) -> np.ndarray:
    """Return the share of power left at every plant on every day.

    Args:
        events: The planted events.
        plants: The number of plants.
        days: The number of days.

    Returns:
        An array of shape (plants, days).
    """
    factor = np.ones((plants, days))
    for event in events:
        factor[event.plant] *= event_factor(event=event, days=days)
    return factor


def plant_losses(*, record: CapacityRecord, events: Sequence[PlantedEvent]) -> CapacityRecord:
    """Plant the events in the measured power, leaving every other input alone.

    The loss multiplies all of a day's power, so it is a proportional loss of direct-current and
    alternating-current capacity together.

    Args:
        record: The record.
        events: The events.

    Returns:
        The record with the planted power.
    """
    factor = factor_matrix(events=events, plants=len(record.sites), days=len(record.dates))
    return record.with_power(power_mw=record.power_mw * factor[:, :, None])


def calendar_month_spans(*, record: CapacityRecord) -> dict[str, tuple[int, int]]:
    """Return the first day index and length of every whole calendar month in the record.

    Args:
        record: The record.

    Returns:
        A mapping from the `%Y-%m` label to (first day index, number of days). A month the record
        starts or ends part-way through is left out.
    """
    months = record.dates.astype("datetime64[M]")
    spans: dict[str, tuple[int, int]] = {}
    for month in np.unique(months):
        indices = np.flatnonzero(months == month)
        first_of_month = month.astype("datetime64[D]")
        length = int(
            ((month + np.timedelta64(1, "M")).astype("datetime64[D]") - first_of_month).astype(int)
        )
        if record.dates[indices[0]] == first_of_month and len(indices) == length:
            spans[str(month)] = (int(indices[0]), length)
    return spans


def _footprints_clash(*, first: PlantedEvent, second: PlantedEvent) -> bool:
    """Say whether two events break the footprint rules.

    Args:
        first: One event.
        second: Another.

    Returns:
        True if they share a plant and their footprints overlap, or they are at different plants and
        either one's month overlaps the other's footprint.
    """

    def overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return a[0] < b[1] and b[0] < a[1]

    month_first = (first.first_day, first.first_day + first.month_days)
    month_second = (second.first_day, second.first_day + second.month_days)
    if first.plant == second.plant:
        return overlap(first.footprint, second.footprint)
    return overlap(month_first, second.footprint) or overlap(month_second, first.footprint)


DRAW_RETRIES: Final[int] = 60
"""How many times a draw restarts its search before it gives up."""

SEARCH_BUDGET: Final[int] = 20000
"""How many assignments one backtracking search tries before it gives up and restarts."""


def _eligible_spans(
    *,
    record: CapacityRecord,
    plant: int,
    lowest: int,
    reserved: Sequence[PlantedEvent],
) -> list[tuple[int, int]]:
    """List the calendar months in which an event at `plant` obeys every rule but the other draws.

    Args:
        record: The record.
        plant: The plant's index.
        lowest: The earliest day index a month may start on.
        reserved: Events that already exist.

    Returns:
        (first day, number of days) of each eligible month.
    """
    days = len(record.dates)
    valid = valid_days(record=record)[plant]
    reference = (record.dates >= np.datetime64(REFERENCE_START)) & (
        record.dates < np.datetime64(REFERENCE_END)
    )
    first_valid = int(np.argmax(valid)) if valid.any() else days
    eligible = []
    for first_day, month_days in calendar_month_spans(record=record).values():
        probe = PlantedEvent(plant, "step", 0.0, first_day, month_days)
        start, stop = probe.footprint
        if first_day < max(lowest, first_valid + MIN_HISTORY_DAYS) or start < 0 or stop > days:
            continue
        if reference[start:stop].any() or valid[start:stop].mean() < MIN_VALID_SHARE:
            continue
        if any(_footprints_clash(first=probe, second=other) for other in reserved):
            continue
        eligible.append((first_day, month_days))
    return eligible


def _assign_months(
    *,
    wanted: Sequence[tuple[int, EventShapeType]],
    domains: dict[tuple[int, EventShapeType], list[tuple[int, int]]],
    rng: np.random.Generator,
) -> dict[tuple[int, EventShapeType], tuple[int, int]] | None:
    """Give every wanted (plant, shape) a month so that no two events break the footprint rules.

    A backtracking search with forward checking, trying the variable with the fewest options first
    and its options in random order.

    Args:
        wanted: The (plant, shape) pairs.
        domains: Each pair's eligible months.
        rng: The generator.

    Returns:
        The assignment, or `None` if the search spends `SEARCH_BUDGET` tries without finding one.
    """
    budget = [SEARCH_BUDGET]

    def clash(plant: int, span: tuple[int, int], other_plant: int, other: tuple[int, int]) -> bool:
        return _footprints_clash(
            first=PlantedEvent(plant, "step", 0.0, *span),
            second=PlantedEvent(other_plant, "step", 0.0, *other),
        )

    def search(
        assigned: dict[tuple[int, EventShapeType], tuple[int, int]],
        remaining: dict[tuple[int, EventShapeType], list[tuple[int, int]]],
    ) -> dict[tuple[int, EventShapeType], tuple[int, int]] | None:
        if not remaining:
            return assigned
        variable = min(remaining, key=lambda v: len(remaining[v]))
        options = list(remaining[variable])
        rng.shuffle(options)
        for span in options:
            budget[0] -= 1
            if budget[0] < 0:
                return None
            narrowed = {
                other: [s for s in domain if not clash(variable[0], span, other[0], s)]
                for other, domain in remaining.items()
                if other != variable
            }
            if any(not domain for domain in narrowed.values()):
                continue
            found = search({**assigned, variable: span}, narrowed)
            if found is not None:
                return found
        return None

    return search({}, {variable: list(domains[variable]) for variable in wanted})


def draw_events(
    *,
    record: CapacityRecord,
    seed: int,
    reserved: Sequence[PlantedEvent] = (),
    earliest_first_day: Mapping[int, int] | None = None,
    shapes: Sequence[EventShapeType] = EVENT_SHAPES,
    may_have_fewer: Collection[int] = (),
) -> list[PlantedEvent]:
    """Draw one event of each shape at every plant, obeying the footprint rules.

    **The rules pack tightly: any two events, at any plants, end up at least 60 days apart.** A
    random placement therefore fails often, so the months come from a backtracking search that
    tries a different random order on each restart. The result is a function of the seed alone.

    Args:
        record: The record.
        seed: The draw's seed.
        reserved: Events that already exist (the real low months, as steps) and so block their
            footprints, though they are not returned.
        earliest_first_day: For a plant, the earliest day index an event's month may start.
        shapes: The shapes to plant at each plant.
        may_have_fewer: Plants that may get fewer events than shapes, because too few of their
            days are valid for every event to fit (the ANM plant's export cap removes its summer
            hours). Each such plant still gets at least one.

    Returns:
        The events, in plant order then shape order.

    Raises:
        ValueError: If `DRAW_RETRIES` restarts all fail to place the events.
    """
    plants = range(len(record.sites))
    domains = {
        (plant, shape): _eligible_spans(
            record=record,
            plant=plant,
            lowest=(earliest_first_day or {}).get(plant, 0),
            reserved=reserved,
        )
        for plant in plants
        for shape in shapes
    }
    for attempt in range(DRAW_RETRIES):
        rng = np.random.default_rng([seed, attempt])
        shapes_by_plant = {p: list(rng.permutation(np.array(shapes, dtype=object))) for p in plants}
        lowest_kept = 1 if may_have_fewer else len(shapes)
        for kept in range(len(shapes), lowest_kept - 1, -1):
            wanted = [
                (plant, shape)
                for plant in plants
                for shape in shapes_by_plant[plant][
                    : kept if plant in may_have_fewer else len(shapes)
                ]
            ]
            assignment = _assign_months(wanted=wanted, domains=domains, rng=rng)
            if assignment is not None:
                return _events_from(assignment=assignment, rng=rng)
    msg = f"cannot place the events for seed {seed}"
    raise ValueError(msg)


def _events_from(
    *, assignment: Mapping[tuple[int, EventShapeType], tuple[int, int]], rng: np.random.Generator
) -> list[PlantedEvent]:
    """Give each assigned month a size and, for an intermittent loss, its hit days.

    Args:
        assignment: Each (plant, shape)'s month.
        rng: The generator.

    Returns:
        The events in plant order then shape order.
    """
    events = []
    for (plant, shape), (first_day, month_days) in sorted(
        assignment.items(), key=lambda item: (item[0][0], EVENT_SHAPES.index(item[0][1]))
    ):
        size = float(rng.uniform(*SIZE_RANGE))
        hit = (
            tuple(sorted(rng.permutation(month_days)[: month_days // 2].tolist()))
            if shape == "intermittent"
            else ()
        )
        events.append(PlantedEvent(plant, shape, size, first_day, month_days, hit))
    return events


MIN_CONTROL_VALID_SHARE: Final[float] = 0.8
"""A control month needs at least this share of valid days at its plant."""


def draw_control_steps(
    *,
    record: CapacityRecord,
    months_per_plant: int,
    size: float,
    scored_months: Sequence[str],
    barred: Mapping[int, Sequence[str]],
    seed: int,
) -> list[PlantedEvent]:
    """Plant a step of one size in a few scored months at every plant, in distinct months.

    No month is used twice, so the neighbours of a planted plant are never themselves planted. A
    plant is planted only in months where at least `MIN_CONTROL_VALID_SHARE` of its days are valid,
    so a plant with little data in a month is not asked to show a loss there. A plant with fewer
    eligible months than `months_per_plant` gets fewer steps.

    Args:
        record: The record.
        months_per_plant: The most months to plant at each plant.
        size: The share of power lost.
        scored_months: The `%Y-%m` labels the frames score, which are the candidates.
        barred: For a plant index, the months it may not be planted in (its own real low months).
        seed: The generator's seed.

    Returns:
        The step events, in plant order then month order.

    Raises:
        ValueError: If a plant has no eligible month.
    """
    spans = calendar_month_spans(record=record)
    valid = valid_days(record=record)
    eligible = {
        plant: [
            month
            for month in scored_months
            if month in spans
            and month not in barred.get(plant, ())
            and valid[plant, spans[month][0] : spans[month][0] + spans[month][1]].mean()
            >= MIN_CONTROL_VALID_SHARE
        ]
        for plant in range(len(record.sites))
    }
    empty = [record.sites[p] for p, months in eligible.items() if not months]
    if empty:
        msg = f"plants {empty} have no scored month with enough valid days"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    used: set[str] = set()
    chosen: dict[int, list[str]] = {plant: [] for plant in eligible}
    for _ in range(months_per_plant):
        for plant in rng.permutation(list(eligible)):
            free = [
                m for m in eligible[int(plant)] if m not in used and m not in chosen[int(plant)]
            ]
            if free:
                month = str(free[int(rng.integers(len(free)))])
                used.add(month)
                chosen[int(plant)].append(month)
    return [
        PlantedEvent(plant, "step", size, spans[month][0], spans[month][1])
        for plant in sorted(chosen)
        for month in sorted(chosen[plant])
    ]


def event_ratios(
    *, events: Sequence[PlantedEvent], planted: IssueScale, original: IssueScale
) -> list[np.ndarray]:
    """Return, for each event, the planted scale over the original scale on its scored days.

    Args:
        events: The events.
        planted: The scale from the record with the losses planted.
        original: The scale from the record without them.

    Returns:
        One array per event, covering `event.scored_days`.
    """
    ratios = []
    for event in events:
        start, stop = event.scored_days
        ratios.append(
            planted.scale_mw[event.plant, start:stop] / original.scale_mw[event.plant, start:stop]
        )
    return ratios


def relative_capacity_error(*, event: PlantedEvent, ratio: np.ndarray, days: int) -> float:
    """Return the mean absolute gap between the scale ratio and the expected factor.

    Args:
        event: The event.
        ratio: The planted scale over the original scale on the event's scored days.
        days: The number of days in the record.

    Returns:
        The mean over the scored days of `|ratio - expected factor|`.
    """
    start, stop = event.scored_days
    return float(np.mean(np.abs(ratio - expected_factor(event=event, days=days)[start:stop])))


def detection_delay(*, event: PlantedEvent, ratio: np.ndarray) -> float:
    """Return the days from a step's start until the scale ratio first reaches halfway.

    Args:
        event: The event, which should be a step.
        ratio: The planted scale over the original scale on the event's scored days.

    Returns:
        The delay in days, or NaN if the ratio never reaches `1 - size / 2` in the scored days.
    """
    reached = np.flatnonzero(ratio <= 1.0 - event.size / 2.0)
    return float(reached[0]) if len(reached) else float("nan")


def interval_coverage(
    *, event: PlantedEvent, ratio: np.ndarray, relative_sd: np.ndarray, days: int
) -> float:
    """Return the share of scored days whose 90% interval covers the expected factor.

    Args:
        event: The event.
        ratio: The planted scale over the original scale on the event's scored days.
        relative_sd: The planted scale's relative standard deviation on the same days.
        days: The number of days in the record.

    Returns:
        The share over the days with a finite standard deviation, or NaN if there are none.
    """
    start, stop = event.scored_days
    expected = expected_factor(event=event, days=days)[start:stop]
    usable = np.isfinite(relative_sd)
    if not usable.any():
        return float("nan")
    half_width = INTERVAL_Z_90 * relative_sd * ratio
    covered = np.abs(ratio - expected) <= half_width
    return float(covered[usable].mean())


def score_events(
    *,
    events: Sequence[PlantedEvent],
    planted: IssueScale,
    original: IssueScale,
    days: int,
) -> pl.DataFrame:
    """Score every event of one draw for one estimator.

    Args:
        events: The draw's events.
        planted: The scale from the record with the losses planted.
        original: The scale from the record without them.
        days: The number of days in the record.

    Returns:
        One row per event with `plant`, `shape`, `size`, `error`, `delay` (steps only, else null),
        and `coverage` (null for an estimator without a standard deviation).
    """
    rows = []
    for event, ratio in zip(
        events, event_ratios(events=events, planted=planted, original=original), strict=True
    ):
        start, stop = event.scored_days
        rows.append(
            {
                "plant": event.plant,
                "shape": event.shape,
                "size": event.size,
                "error": relative_capacity_error(event=event, ratio=ratio, days=days),
                "delay": detection_delay(event=event, ratio=ratio)
                if event.shape == "step"
                else None,
                "coverage": interval_coverage(
                    event=event,
                    ratio=ratio,
                    relative_sd=planted.relative_sd[event.plant, start:stop],
                    days=days,
                ),
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "plant": pl.Int64,
            "shape": pl.String,
            "size": pl.Float64,
            "error": pl.Float64,
            "delay": pl.Float64,
            "coverage": pl.Float64,
        },
    ).with_columns(pl.col("delay").fill_nan(None), pl.col("coverage").fill_nan(None))


def draw_primary_metric(*, scores: pl.DataFrame) -> float:
    """Return a draw's primary metric: the mean relative capacity error over all its events.

    Args:
        scores: `score_events`' result for the draw.

    Returns:
        The mean of `error`, which weighs every event and every plant alike.
    """
    return float(np.mean(scores["error"].to_numpy()))


def false_movement_windows(
    *,
    ratios: pl.DataFrame,
    record: CapacityRecord,
    per_plant: int,
    seed: int,
    lower: float = 0.95,
    upper: float = 1.05,
) -> list[tuple[int, int, int]]:
    """Choose the plant-months on which the real-data estimate should not move.

    Args:
        ratios: Rows with `site`, `month` (`%Y-%m`), and `ratio`, each plant-month's output over its
            neighbours'.
        record: The record, for the site order and the month spans.
        per_plant: The most months to choose at each plant.
        seed: The generator's seed.
        lower: The lowest ratio of a quiet month.
        upper: The highest ratio of a quiet month.

    Returns:
        (plant, first day, number of days) triples, in plant order.
    """
    spans = calendar_month_spans(record=record)
    rng = np.random.default_rng(seed)
    windows: list[tuple[int, int, int]] = []
    for plant, site in enumerate(record.sites):
        quiet = ratios.filter((pl.col("site") == site) & pl.col("ratio").is_between(lower, upper))[
            "month"
        ].to_list()
        quiet = [month for month in quiet if month in spans]
        chosen = rng.permutation(quiet)[:per_plant] if quiet else []
        windows += [(plant, *spans[str(month)]) for month in sorted(chosen)]
    return windows


def false_movement_days(*, windows: Sequence[tuple[int, int, int]]) -> list[np.ndarray]:
    """Return the estimate days the false-movement metric needs, per window.

    A target day `T` reads the estimate for day `T - 2` and compares it with the median of the 90
    target days before, so a window starting on `first` needs estimates from `first - 92`.

    Args:
        windows: `false_movement_windows`' result.

    Returns:
        One array of day indices per window, ending two days before the window's last day.
    """
    return [
        np.arange(first - FALSE_MOVEMENT_TRAILING_DAYS - 2, first + length - 2)
        for _, first, length in windows
    ]


def false_movement_share(*, windows: Sequence[tuple[int, int, int]], original: IssueScale) -> float:
    """Return the share of window days on which the estimate moves falsely.

    A day moves falsely when the estimate for it is more than `FALSE_MOVEMENT_THRESHOLD` away from
    the median of the estimates for the `FALSE_MOVEMENT_TRAILING_DAYS` target days before.

    Args:
        windows: `false_movement_windows`' result.
        original: The estimator's scale on the original record.

    Returns:
        The share over all window days. A day on which the estimator fell back to the static power
        counts as moving, because a constant fallback is a failure to estimate and must not read
        as stability.
    """
    moved, total = 0, 0
    for plant, first, length in windows:
        series = original.scale_mw[plant]
        for day in range(first, first + length):
            trailing = np.median(series[day - FALSE_MOVEMENT_TRAILING_DAYS : day])
            fell_back = original.fallback_reason[plant, day] != 0
            moved += int(fell_back or abs(series[day] / trailing - 1.0) > FALSE_MOVEMENT_THRESHOLD)
            total += 1
    return moved / total


def fallback_share_on_events(*, events: Sequence[PlantedEvent], planted: IssueScale) -> float:
    """Return the share of the events' scored days on which the estimator fell back.

    Args:
        events: The draw's events.
        planted: The estimator's scale on the record with the losses planted.

    Returns:
        The share over all scored days of all events.
    """
    reasons = np.concatenate(
        [planted.fallback_reason[e.plant, e.scored_days[0] : e.scored_days[1]] for e in events]
    )
    return float(np.mean(reasons != 0))


FALLBACK_SHARE_GATE: Final[float] = 0.25
"""An estimator that falls back to the static power on more than this share of event days is out."""

StatusType = Literal["gated", "ungated_pick", "none_beat_s0"]
"""How the selected estimator was reached: it passed every gate; it is the best that beats S0 and
passes the fallback gate although none passed the false-movement gate; or nothing beat S0."""


class SelectionInputs(NamedTuple):
    """What the selection rule reads, for every estimator that was run."""

    tuning_error: Mapping[str, float]
    selection_errors: Mapping[str, np.ndarray]
    false_movement: Mapping[str, float]
    fallback_share: Mapping[str, float]


@dataclass(frozen=True)
class Contrast:
    """A difference in the primary metric between two representatives, with its interval."""

    name: str
    treatment: str
    reference: str
    difference: float
    lower: float
    upper: float
    verdict: str


@dataclass(frozen=True)
class Selection:
    """The result of the selection rule.

    Attributes:
        selected: The estimator the forecast contrast uses, or `None` if none beat S0.
        status: `gated`, `ungated_pick`, or `none_beat_s0`.
        survivors: The candidates that passed every gate.
        shortlist: The best surviving estimator of each family, selected one included.
        representatives: Each family group's representative, chosen on the tuning draw.
        contrasts: E1 and E2, or an empty list where a representative is missing.
        missing: A note for each contrast that cannot be computed.
        s0_error: The no-skill reference's mean primary metric over the selection draws.
    """

    selected: str | None
    status: StatusType
    survivors: tuple[str, ...]
    shortlist: tuple[str, ...]
    representatives: Mapping[str, str]
    contrasts: tuple[Contrast, ...]
    missing: tuple[str, ...]
    s0_error: float


def t_interval(*, differences: np.ndarray, confidence: float) -> tuple[float, float, float]:
    """Return the mean of paired differences and a two-sided t-interval.

    Args:
        differences: One difference per draw.
        confidence: The interval's coverage, such as 0.975.

    Returns:
        The mean, the lower bound, and the upper bound.
    """
    n = len(differences)
    mean = float(np.mean(differences))
    half_width = float(
        stats.t.ppf(0.5 + confidence / 2.0, df=n - 1) * np.std(differences, ddof=1) / np.sqrt(n)
    )
    return mean, mean - half_width, mean + half_width


def verdict_of(*, lower: float, upper: float) -> str:
    """Name an interval's verdict on a difference where a lower error is better.

    Args:
        lower: The interval's lower bound.
        upper: The interval's upper bound.

    Returns:
        `beats` if the interval lies wholly below zero, `loses` if wholly above, and `no detectable
        difference` otherwise.
    """
    if upper < 0:
        return "beats"
    if lower > 0:
        return "loses"
    return "no detectable difference"


def _contrast(
    *, name: str, treatment: str, reference: str, errors: Mapping[str, np.ndarray]
) -> Contrast:
    """Build one estimator-level contrast from the selection draws' errors."""
    mean, lower, upper = t_interval(
        differences=errors[treatment] - errors[reference], confidence=CONTRAST_CONFIDENCE
    )
    return Contrast(
        name=name,
        treatment=treatment,
        reference=reference,
        difference=mean,
        lower=lower,
        upper=upper,
        verdict=verdict_of(lower=lower, upper=upper),
    )


def choose_estimators(*, inputs: SelectionInputs) -> Selection:
    """Apply the selection rule.

    **The references S0 and EM are never candidates.** A candidate must beat S0's mean primary
    metric over the selection draws, pass the false-movement gate (`FALSE_MOVEMENT_GATE`), and pass
    the fallback gate (`FALLBACK_SHARE_GATE`). Each family's representative is its lowest-error
    survivor on the tuning draw, so neither contrast is scored on the draw that chose it. The
    selected estimator is the survivor with the lowest mean primary metric over the selection
    draws, and the shortlist holds the best survivor of each family by the same mean.

    **If no candidate passes every gate, the selection is an ungated pick:** the lowest-error
    candidate that beats S0 and passes the fallback gate. If none beats S0, nothing is selected and
    the finding is that no estimator beat doing nothing.

    E1 is the convex representative minus the representative of the cheap and wacky families. E2 is
    the neural-network estimator minus the convex representative.

    Args:
        inputs: The tuning-draw errors, the selection draws' errors, the false-movement shares, and
            the fallback shares on event days.

    Returns:
        The selection.

    Raises:
        ValueError: If `inputs` hold no estimator or lack S0.
    """
    names = sorted(inputs.selection_errors)
    if "S0" not in names:
        msg = "the no-skill reference S0 was not run"
        raise ValueError(msg)
    mean_error = {name: float(np.mean(inputs.selection_errors[name])) for name in names}
    candidates = [name for name in names if name not in REFERENCE_ESTIMATORS]
    if not candidates:
        msg = "no candidate estimator was run"
        raise ValueError(msg)
    usable = [
        name
        for name in candidates
        if mean_error[name] < mean_error["S0"]
        and inputs.fallback_share[name] <= FALLBACK_SHARE_GATE
    ]
    survivors = [name for name in usable if inputs.false_movement[name] <= FALSE_MOVEMENT_GATE]
    status: StatusType = "gated" if survivors else ("ungated_pick" if usable else "none_beat_s0")
    eligible = survivors or usable
    selected = min(eligible, key=lambda name: mean_error[name]) if eligible else None
    families: dict[FamilyType, list[str]] = {}
    for name in survivors:
        families.setdefault(ESTIMATOR_FAMILY[name], []).append(name)
    shortlist = tuple(
        min(members, key=lambda name: mean_error[name]) for members in families.values()
    )
    if status == "ungated_pick" and selected is not None:
        shortlist = (selected,)

    def representative(group: Sequence[str]) -> str | None:
        members = [name for name in eligible if name in group]
        return min(members, key=lambda name: inputs.tuning_error[name]) if members else None

    cheap_and_wacky = [n for n in candidates if ESTIMATOR_FAMILY[n] in ("cheap", "wacky")]
    convex = [n for n in candidates if ESTIMATOR_FAMILY[n] == "convex"]
    pytorch = [n for n in candidates if ESTIMATOR_FAMILY[n] == "pytorch"]
    reps = {
        "cheap_and_wacky": representative(cheap_and_wacky),
        "convex": representative(convex),
        "pytorch": representative(pytorch),
    }
    contrasts: list[Contrast] = []
    missing: list[str] = []
    if reps["convex"] and reps["cheap_and_wacky"]:
        contrasts.append(
            _contrast(
                name="E1",
                treatment=reps["convex"],
                reference=reps["cheap_and_wacky"],
                errors=inputs.selection_errors,
            )
        )
    else:
        missing.append("E1: no convex or no cheap-or-wacky estimator was eligible")
    if reps["pytorch"] and reps["convex"]:
        contrasts.append(
            _contrast(
                name="E2",
                treatment=reps["pytorch"],
                reference=reps["convex"],
                errors=inputs.selection_errors,
            )
        )
    else:
        missing.append("E2: no neural-network or no convex estimator was eligible")
    return Selection(
        selected=selected,
        status=status,
        survivors=tuple(survivors),
        shortlist=shortlist,
        representatives={k: v for k, v in reps.items() if v is not None},
        contrasts=tuple(contrasts),
        missing=tuple(missing),
        s0_error=mean_error["S0"],
    )


POSTHOC_GATE_THRESHOLDS: Final[tuple[float, ...]] = (0.05, 0.10, 0.15, 0.20, 0.30)
"""The false-movement thresholds the post hoc sensitivity table tries."""


def gate_sensitivity(
    *, inputs: SelectionInputs, thresholds: Sequence[float] = POSTHOC_GATE_THRESHOLDS
) -> dict[float, tuple[str, ...]]:
    """Return the candidates that pass every gate at each false-movement threshold.

    **This is a post hoc, descriptive table, not part of the selection rule.** A candidate passes
    at a threshold if it beats S0, stays within the fallback gate, and its false-movement share is
    at most the threshold.

    Args:
        inputs: The selection inputs.
        thresholds: The false-movement thresholds to try.

    Returns:
        The passing candidates, sorted, for each threshold.
    """
    s0 = float(np.mean(inputs.selection_errors["S0"]))
    candidates = [
        name for name in sorted(inputs.selection_errors) if name not in REFERENCE_ESTIMATORS
    ]
    return {
        threshold: tuple(
            name
            for name in candidates
            if float(np.mean(inputs.selection_errors[name])) < s0
            and inputs.fallback_share[name] <= FALLBACK_SHARE_GATE
            and inputs.false_movement[name] <= threshold
        )
        for threshold in thresholds
    }


def posthoc_contrasts(*, inputs: SelectionInputs) -> tuple[Contrast, ...]:
    """Return E1 and E2 without any gate, as a post hoc analysis.

    **The representatives are the lowest tuning-draw error of each family group among all
    candidates, whether or not they beat S0 or pass a gate.** The neural-network representative
    is DP1. The contrasts are labelled `E1_posthoc` and `E2_posthoc`.

    Args:
        inputs: The selection inputs.

    Returns:
        The contrasts whose two representatives exist.
    """
    candidates = [name for name in inputs.selection_errors if name not in REFERENCE_ESTIMATORS]

    def lowest(families: tuple[str, ...]) -> str | None:
        members = [name for name in candidates if ESTIMATOR_FAMILY[name] in families]
        return min(members, key=lambda name: inputs.tuning_error[name]) if members else None

    cheap_and_wacky, convex, pytorch = (
        lowest(("cheap", "wacky")),
        lowest(("convex",)),
        lowest(("pytorch",)),
    )
    contrasts: list[Contrast] = []
    if convex and cheap_and_wacky:
        contrasts.append(
            _contrast(
                name="E1_posthoc",
                treatment=convex,
                reference=cheap_and_wacky,
                errors=inputs.selection_errors,
            )
        )
    if pytorch and convex:
        contrasts.append(
            _contrast(
                name="E2_posthoc",
                treatment=pytorch,
                reference=convex,
                errors=inputs.selection_errors,
            )
        )
    return tuple(contrasts)


def write_selection(*, path: Path, selection: Selection, inputs: SelectionInputs) -> None:
    """Write `selection.json`, refusing to overwrite an existing file.

    Args:
        path: The file to write.
        selection: `choose_estimators`' result.
        inputs: The inputs it was chosen from, which the file records.

    Raises:
        FileExistsError: If `path` exists (from `refuse_to_overwrite`).
    """
    refuse_to_overwrite(paths=[path])
    payload = {
        "selected": selection.selected,
        "status": selection.status,
        "s0_error": selection.s0_error,
        "fallback_share_gate": FALLBACK_SHARE_GATE,
        "fallback_share": dict(inputs.fallback_share),
        "survivors": list(selection.survivors),
        "shortlist": list(selection.shortlist),
        "representatives": dict(selection.representatives),
        "contrasts": [vars(contrast) for contrast in selection.contrasts],
        "contrasts_not_computed": list(selection.missing),
        "false_movement_gate": FALSE_MOVEMENT_GATE,
        "tuning_error": dict(inputs.tuning_error),
        "mean_selection_error": {k: float(np.mean(v)) for k, v in inputs.selection_errors.items()},
        "selection_errors": {k: v.tolist() for k, v in inputs.selection_errors.items()},
        "false_movement": dict(inputs.false_movement),
        "posthoc_gate_sensitivity": {
            str(threshold): list(passing)
            for threshold, passing in gate_sensitivity(inputs=inputs).items()
        },
        "posthoc_contrasts": [vars(contrast) for contrast in posthoc_contrasts(inputs=inputs)],
    }
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.rename(path)
