"""Score a labeller's knots against planted losses, for the study in issue #1147.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1147>.

**A knot series is scored two ways.** The capacity error is the first study's: the labeller's
capacity on the planted record over its capacity on the original record, against the factor the
planted event applied (`studies.capacity_recovery.score_events`). The event score is new: whether
the series holds a drop segment near the event's start with the right sign and size and a recovery
segment near its end, how many segments no event explains (false alarms), and the date and size
errors of the hits.

**A segment is the stretch between two consecutive knots.** Its change is the end level over the
start level minus 1. A segment is a step when the change is at least `STEP_MINIMUM` in size.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from statistics import NormalDist
from typing import Final

import numpy as np
import polars as pl

from studies.capacity_estimators import CapacityRecord, IssueScale, valid_days
from studies.capacity_labeller import Answer, Knot, capacity_series
from studies.capacity_recovery import MIN_CONTROL_VALID_SHARE, PlantedEvent, calendar_month_spans

EVENT_WINDOW_DAYS: Final[int] = 14
"""A drop or recovery segment counts when it starts or ends this close to the event's start or
end."""

SIZE_TOLERANCE: Final[float] = 0.5
"""A drop's size must be within this share of the event's size."""

STEP_MINIMUM: Final[float] = 0.03
"""A segment whose change is smaller than this is not a step."""

BOOTSTRAP_RESAMPLES: Final[int] = 2000
"""How many resamples the event bootstrap draws."""


@dataclass(frozen=True)
class Segment:
    """The stretch between two consecutive knots.

    Attributes:
        start: The first knot's day index in the record.
        end: The second knot's day index.
        change: The end level over the start level, minus 1.
    """

    start: int
    end: int
    change: float


def segments_of(*, knots: Sequence[Knot], dates: np.ndarray) -> list[Segment]:
    """List the segments of a knot series.

    Args:
        knots: The knots in date order.
        dates: The record's dates.

    Returns:
        One segment per consecutive pair of knots.
    """
    pairs = [(int((np.datetime64(knot.day) - dates[0]).astype(int)), knot) for knot in knots]
    return [
        Segment(start=a, end=b, change=after.capacity_mw / before.capacity_mw - 1.0)
        for (a, before), (b, after) in pairwise(pairs)
    ]


def effective_size(*, event: PlantedEvent) -> float:
    """Return the size a labeller is asked to find: the window mean for an intermittent loss.

    Args:
        event: The event.

    Returns:
        The event's size, halved for an intermittent loss (a step is `size` and a ramp's deepest
        point is `size`).
    """
    return event.size / 2.0 if event.shape == "intermittent" else event.size


def _near(*, segment: Segment, day: int) -> bool:
    """Say whether a segment starts or ends within the event window of `day`."""
    return min(abs(segment.start - day), abs(segment.end - day)) <= EVENT_WINDOW_DAYS


@dataclass(frozen=True)
class EventScore:
    """One planted event scored against one knot series.

    Attributes:
        drop_found: Whether a drop segment with the right sign and size lies near the start.
        recovery_found: Whether a rising segment lies near the end.
        hit: Whether both were found.
        date_error_days: The drop segment's smallest distance to the start, or NaN.
        size_error: The drop's size error as a share of the event's size, or NaN.
    """

    drop_found: bool
    recovery_found: bool
    hit: bool
    date_error_days: float
    size_error: float


def score_event(*, event: PlantedEvent, segments: Sequence[Segment]) -> EventScore:
    """Score one event against the segments of the series at its plant.

    Args:
        event: The planted event.
        segments: The plant's segments.

    Returns:
        The event score.
    """
    size = effective_size(event=event)
    start, end = event.first_day, event.first_day + event.month_days
    drops = [
        s
        for s in segments
        if s.change < 0
        and _near(segment=s, day=start)
        and abs(-s.change - size) <= SIZE_TOLERANCE * size
    ]
    recoveries = [s for s in segments if s.change > 0 and _near(segment=s, day=end)]
    if drops:
        best = min(drops, key=lambda s: min(abs(s.start - start), abs(s.end - start)))
        date_error = float(min(abs(best.start - start), abs(best.end - start)))
        size_error = abs(-best.change - size) / size
    else:
        date_error, size_error = float("nan"), float("nan")
    return EventScore(
        drop_found=bool(drops),
        recovery_found=bool(recoveries),
        hit=bool(drops) and bool(recoveries),
        date_error_days=date_error,
        size_error=size_error,
    )


def false_alarms(
    *,
    events: Sequence[PlantedEvent],
    planted: Sequence[Segment],
    original: Sequence[Segment],
) -> int:
    """Count the planted-run segments that no planted event and no original-run segment explains.

    A segment is explained if it is not a step, if it starts or ends within the event window of an
    event's start or end, or if the original-record series holds a segment of the same sign that
    ends within the event window of its end.

    Args:
        events: The planted events at the plant.
        planted: The plant's segments on the planted record.
        original: The plant's segments on the original record, same sample.

    Returns:
        The number of unexplained step segments.
    """
    anchors = [
        day for event in events for day in (event.first_day, event.first_day + event.month_days)
    ]
    count = 0
    for segment in planted:
        if abs(segment.change) < STEP_MINIMUM:
            continue
        if any(_near(segment=segment, day=day) for day in anchors):
            continue
        if any(
            np.sign(other.change) == np.sign(segment.change)
            and abs(other.end - segment.end) <= EVENT_WINDOW_DAYS
            for other in original
        ):
            continue
        count += 1
    return count


def answers_to_scale(*, answers: Mapping[int, Answer], record: CapacityRecord) -> IssueScale:
    """Turn per-plant answers into a scale for every plant and day.

    Args:
        answers: The answer for each plant index; a plant without one keeps its static power.
        record: The record, for the dates and static powers.

    Returns:
        The scale (the capacity the knots describe, on the day itself), with no standard
        deviation and no fallback.
    """
    plants, days = len(record.sites), len(record.dates)
    scale = np.stack(
        [
            capacity_series(
                knots=answers[plant].knots if plant in answers else (),
                dates=record.dates,
                static_p99_mw=float(record.static_p99_mw[plant]),
            )
            for plant in range(plants)
        ]
    )
    return IssueScale(
        scale_mw=scale,
        relative_sd=np.full((plants, days), np.nan),
        fallback_reason=np.zeros((plants, days), dtype=np.int8),
    )


def score_draw(
    *,
    events: Sequence[PlantedEvent],
    planted: Mapping[int, Answer],
    original: Mapping[int, Answer],
    record: CapacityRecord,
) -> pl.DataFrame:
    """Score every event of one draw for one sample of one arm.

    Args:
        events: The draw's events.
        planted: Each plant's answer on the planted record.
        original: Each plant's answer on the original record, same sample.
        record: The record, for the dates.

    Returns:
        One row per event with `plant`, `shape`, `size`, `drop_found`, `recovery_found`, `hit`,
        `date_error_days`, `size_error`, and `false_alarms` (the plant's count, repeated on each of
        the plant's events).
    """
    rows = []
    for event in events:
        planted_segments = segments_of(knots=planted[event.plant].knots, dates=record.dates)
        original_segments = segments_of(knots=original[event.plant].knots, dates=record.dates)
        score = score_event(event=event, segments=planted_segments)
        plant_events = [e for e in events if e.plant == event.plant]
        rows.append(
            {
                "plant": event.plant,
                "shape": event.shape,
                "size": event.size,
                "drop_found": score.drop_found,
                "recovery_found": score.recovery_found,
                "hit": score.hit,
                "date_error_days": score.date_error_days,
                "size_error": score.size_error,
                "false_alarms": false_alarms(
                    events=plant_events, planted=planted_segments, original=original_segments
                ),
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "plant": pl.Int64,
            "shape": pl.String,
            "size": pl.Float64,
            "drop_found": pl.Boolean,
            "recovery_found": pl.Boolean,
            "hit": pl.Boolean,
            "date_error_days": pl.Float64,
            "size_error": pl.Float64,
            "false_alarms": pl.Int64,
        },
    ).with_columns(pl.col("date_error_days").fill_nan(None), pl.col("size_error").fill_nan(None))


def false_alarms_per_plant(*, scores: pl.DataFrame) -> int:
    """Return the total false alarms of a draw, counting each plant once.

    Args:
        scores: `score_draw`'s result.

    Returns:
        The sum over plants of the plant's false-alarm count.
    """
    return int(scores.group_by("plant").agg(pl.col("false_alarms").first())["false_alarms"].sum())


def event_bootstrap_interval(
    *, differences: np.ndarray, confidence: float, seed: int, resamples: int = BOOTSTRAP_RESAMPLES
) -> tuple[float, float, float]:
    """Return the mean paired difference and its bootstrap interval over events and samples.

    Each resample draws as many (event, sample) pairs as there are events, drawing the event and the
    sample index together, so the interval covers which events were drawn and which sample was read.

    Args:
        differences: The per-event, per-sample differences, shape (events, samples).
        confidence: The two-sided coverage, for example 0.975.
        seed: The generator's seed.
        resamples: The number of resamples.

    Returns:
        The mean difference, then the lower and upper ends of the interval.
    """
    events, samples = differences.shape
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, events, size=(resamples, events))
    columns = rng.integers(0, samples, size=(resamples, events))
    means = differences[rows, columns].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [tail, 1.0 - tail])
    return float(differences.mean()), float(lower), float(upper)


NULL_MIN_SHAPE: Final[float] = 0.02
"""Hours with less modelled output than this carry no relative residual."""

NULL_MIN_FINITE_HOURS: Final[int] = 6
"""A quiet day is a resampling candidate when it has this many finite relative residuals."""


def synthetic_null_record(
    *, record: CapacityRecord, windows: Sequence[tuple[int, int, int]], seed: int
) -> CapacityRecord:
    """Build a record in which no plant ever changes, from each plant's median ratio and noise.

    Each plant's power is its median ratio of power to modelled output times the modelled output,
    times one plus a relative residual. Residuals are whole days (24 hours) resampled with
    replacement from the quiet plant-months, so the day-to-day correlation of the satellite's
    errors is kept. Hours missing in the original stay missing. The right answer on this record is
    no change at any plant.

    Args:
        record: The original record.
        windows: The quiet plant-months as (plant, first day, number of days) triples.
        seed: The generator's seed.

    Returns:
        The record with synthetic power.
    """
    rng = np.random.default_rng(seed)
    plants = len(record.sites)
    power = np.full(record.power_mw.shape, np.nan)
    for plant in range(plants):
        shape = record.cams_shape[plant]
        usable = np.isfinite(record.power_mw[plant]) & (shape >= NULL_MIN_SHAPE)
        ratio = float(np.median(record.power_mw[plant][usable] / shape[usable]))
        pool = []
        for window_plant, first, length in windows:
            if window_plant != plant:
                continue
            block = slice(first, first + length)
            ok = usable[block]
            with np.errstate(invalid="ignore", divide="ignore"):
                residual = np.where(
                    ok, record.power_mw[plant][block] / (ratio * shape[block]) - 1.0, np.nan
                )
            pool += [row for row in residual if np.isfinite(row).sum() >= NULL_MIN_FINITE_HOURS]
        if not pool:
            msg = f"plant {record.sites[plant]} has no quiet day to resample"
            raise ValueError(msg)
        chosen = np.stack([pool[i] for i in rng.integers(0, len(pool), size=len(record.dates))])
        modelled = ratio * np.nan_to_num(shape, nan=0.0)
        synthetic = np.clip(modelled * (1.0 + np.nan_to_num(chosen, nan=0.0)), 0.0, None)
        power[plant] = np.where(np.isfinite(record.power_mw[plant]), synthetic, np.nan)
    return record.with_power(power_mw=power)


MOVEMENT_THRESHOLD: Final[float] = 0.05
"""A day moves when the capacity is more than this share from its trailing median."""

MOVEMENT_TRAILING_DAYS: Final[int] = 90
"""The trailing days whose median a day's capacity is compared with."""


def movement_by_window(
    *, windows: Sequence[tuple[int, int, int]], scale_mw: np.ndarray
) -> np.ndarray:
    """Count, per quiet plant-month, the days whose capacity moves from its trailing median.

    The rule is `studies.capacity_recovery.false_movement_share`'s, without its fallback clause
    (a knot series has no fallback), so the first study's gate and threshold carry over.

    Args:
        windows: The quiet plant-months as (plant, first day, number of days) triples.
        scale_mw: The capacity on each day, shape (plants, days).

    Returns:
        An array of shape (windows, 2): days that moved, and days in the window.
    """
    rows = []
    for plant, first, length in windows:
        series = scale_mw[plant]
        moved = 0
        for day in range(first, first + length):
            trailing = float(np.median(series[day - MOVEMENT_TRAILING_DAYS : day]))
            moved += int(abs(series[day] / trailing - 1.0) > MOVEMENT_THRESHOLD)
        rows.append((moved, length))
    return np.array(rows, dtype=float)


def movement_interval(
    *, per_window: np.ndarray, confidence: float, seed: int, resamples: int = BOOTSTRAP_RESAMPLES
) -> tuple[float, float, float]:
    """Return the false-movement share and its interval from resampling the plant-months.

    Args:
        per_window: `movement_by_window`'s result, optionally stacked over samples as shape
            (samples, windows, 2) and averaged by the caller into (windows, 2).
        confidence: The two-sided coverage.
        seed: The generator's seed.
        resamples: The number of resamples.

    Returns:
        The share, then the lower and upper ends of the interval.
    """
    rng = np.random.default_rng(seed)
    count = len(per_window)
    picks = rng.integers(0, count, size=(resamples, count))
    shares = per_window[picks, 0].sum(axis=1) / per_window[picks, 1].sum(axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(shares, [tail, 1.0 - tail])
    return float(per_window[:, 0].sum() / per_window[:, 1].sum()), float(lower), float(upper)


def draw_nonadjacent_control_steps(
    *,
    record: CapacityRecord,
    months_per_plant: int,
    size: float,
    scored_months: Sequence[str],
    barred: Mapping[int, Sequence[str]],
    seed: int,
) -> list[PlantedEvent]:
    """Plant a step of one size in a few scored months per plant, never in adjacent months.

    The rule is `studies.capacity_recovery.draw_control_steps`' (distinct months across plants, only
    months with enough valid days, a plant's own low months barred) with one addition: no two steps
    at the same plant fall in consecutive calendar months, so each loss is a separate event to find.

    Args:
        record: The record.
        months_per_plant: The most months to plant at each plant.
        size: The share of power lost.
        scored_months: The `%Y-%m` labels that are the candidates.
        barred: For a plant index, the months it may not be planted in.
        seed: The generator's seed.

    Returns:
        The step events in plant order then month order; a plant gets fewer steps if too few
        non-adjacent months are eligible.
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
    rng = np.random.default_rng(seed)
    used: set[str] = set()
    chosen: dict[int, list[str]] = {plant: [] for plant in eligible}
    for _ in range(months_per_plant):
        for plant in rng.permutation(list(eligible)):
            taken = [np.datetime64(m, "M") for m in chosen[int(plant)]]
            free = [
                m
                for m in eligible[int(plant)]
                if m not in used
                and all(abs((np.datetime64(m, "M") - t).astype(int)) >= 2 for t in taken)
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


def wilson_interval(
    *, successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Return the Wilson score interval of a proportion.

    Args:
        successes: The number of hits.
        trials: The number of events.
        confidence: The two-sided coverage.

    Returns:
        The lower and upper ends; `(0.0, 1.0)` for no trials.
    """
    if trials == 0:
        return 0.0, 1.0
    z = float(NormalDist().inv_cdf(0.5 + confidence / 2.0))
    share = successes / trials
    centre = (share + z * z / (2 * trials)) / (1 + z * z / trials)
    half = (
        z
        * np.sqrt(share * (1 - share) / trials + z * z / (4 * trials * trials))
        / (1 + z * z / trials)
    )
    return float(max(centre - half, 0.0)), float(min(centre + half, 1.0))


def exact_upper_bound_for_no_hits(*, trials: int, confidence: float = 0.975) -> float:
    """Return the highest hit rate that `trials` events with no hit cannot rule out.

    Args:
        trials: The number of events, none of which was a hit.
        confidence: The one-sided coverage.

    Returns:
        `1 - (1 - confidence) ** (1 / trials)`, the exact one-sided upper bound.
    """
    return float(1.0 - (1.0 - confidence) ** (1.0 / trials))


SIZE_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("under 15%", 0.0, 0.15),
    ("15% to 25%", 0.15, 0.25),
    ("25% and over", 0.25, 1.01),
)
"""The planted-size bands the hit rate is reported by."""


def size_band(*, size: float) -> str:
    """Return the label of the size band a planted loss falls in."""
    return next(label for label, low, high in SIZE_BANDS if low <= size < high)


def placebo_hits(
    *, events: Sequence[PlantedEvent], answers: Mapping[int, Answer], dates: np.ndarray
) -> pl.DataFrame:
    """Score a record's answers against events that were never planted in it.

    A null-record answer is scored against another draw's event dates. The hit rate is what a
    labeller would score by marking steps near those dates by chance, so it is the baseline for
    that draw's real hit rate.

    Args:
        events: The other draw's events.
        answers: Each plant's answer on the record with no change.
        dates: The record's dates.

    Returns:
        One row per event with `hit`, `drop_found`, `recovery_found`, and `size`.
    """
    rows = []
    for event in events:
        score = score_event(
            event=event, segments=segments_of(knots=answers[event.plant].knots, dates=dates)
        )
        rows.append(
            {
                "size": event.size,
                "hit": score.hit,
                "drop_found": score.drop_found,
                "recovery_found": score.recovery_found,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "size": pl.Float64,
            "hit": pl.Boolean,
            "drop_found": pl.Boolean,
            "recovery_found": pl.Boolean,
        },
    )


def events_bootstrap_interval(
    *,
    differences: np.ndarray,
    confidence: float,
    seed: int,
    clusters: np.ndarray | None = None,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float, float]:
    """Return the mean paired difference and an interval that resamples whole events or clusters.

    Each resample draws events with replacement and keeps every sample of an event together. With
    `clusters` (for example each event's calendar month), whole clusters are drawn instead, which
    allows for events that share weather.

    Args:
        differences: The per-event, per-sample differences, shape (events, samples).
        confidence: The two-sided coverage.
        seed: The generator's seed.
        clusters: One label per event, or `None` to resample events.
        resamples: The number of resamples.

    Returns:
        The mean difference, then the lower and upper ends of the interval.
    """
    per_event = differences.mean(axis=1)
    rng = np.random.default_rng(seed)
    if clusters is None:
        picks = rng.integers(0, len(per_event), size=(resamples, len(per_event)))
        means = per_event[picks].mean(axis=1)
    else:
        labels = np.unique(clusters)
        sums = np.array([per_event[clusters == label].sum() for label in labels])
        counts = np.array([(clusters == label).sum() for label in labels])
        picks = rng.integers(0, len(labels), size=(resamples, len(labels)))
        means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [tail, 1.0 - tail])
    return float(per_event.mean()), float(lower), float(upper)


def events_needed(
    *, sd_points: float, effect_points: float, alpha: float = 0.025, power: float = 0.8
) -> int:
    """Return the events a paired contrast needs to detect an effect at a one-sided level.

    Args:
        sd_points: The standard deviation of one event's paired difference.
        effect_points: The effect to detect, in the same units.
        alpha: The one-sided level (0.025 for the 97.5% interval's lower bound).
        power: The power.

    Returns:
        `((z_alpha + z_power) * sd / effect) ** 2`, rounded up.
    """
    z = NormalDist().inv_cdf(1.0 - alpha) + NormalDist().inv_cdf(power)
    return int(np.ceil((z * sd_points / abs(effect_points)) ** 2))


EASY_MIN_START_DAY: Final[int] = 400
"""The earliest day an easy-case loss may start, so every plant has a year of history behind it."""

EASY_MIN_VALID_SHARE: Final[float] = 0.8
"""An easy-case window and its 30 days either side need this share of valid days."""


def draw_easy_events(
    *,
    record: CapacityRecord,
    size: float,
    length: int,
    seed: int,
    avoid: Mapping[int, Sequence[tuple[int, int]]],
    barred_months: Mapping[int, Sequence[str]],
    earliest_first_day: Mapping[int, int],
) -> list[PlantedEvent]:
    """Plant one large, long step at every plant, starting on any day, clear of other windows.

    An easy case asks whether a labeller follows a clear signal at all: a loss of `size` for
    `length` days, beginning on a random day (not the first of a month). A plant's window avoids
    its `avoid` windows (an earlier easy record's, with 30 days of margin), its barred calendar
    months (the plant's real low months), and any day before its `earliest_first_day`.

    Args:
        record: The record.
        size: The share of power lost.
        length: The loss's length in days.
        seed: The generator's seed.
        avoid: For a plant index, windows (first day, last day) to stay clear of.
        barred_months: For a plant index, `YYYY-MM` months the window may not touch.
        earliest_first_day: For a plant index, the earliest day the loss may start.

    Returns:
        One step event per plant, in plant order.

    Raises:
        ValueError: If a plant has no eligible start day.
    """
    valid = valid_days(record=record)
    days = len(record.dates)
    months = record.dates.astype("datetime64[M]").astype(str)
    rng = np.random.default_rng(seed)
    margin = 30
    events = []
    for plant in range(len(record.sites)):
        eligible = []
        for start in range(
            max(EASY_MIN_START_DAY, earliest_first_day.get(plant, 0)), days - length - 60
        ):
            window = slice(start - margin, start + length + margin)
            if valid[plant, window].mean() < EASY_MIN_VALID_SHARE:
                continue
            if any(m in barred_months.get(plant, ()) for m in set(months[start : start + length])):
                continue
            if any(
                start - margin < b and start + length + margin > a for a, b in avoid.get(plant, ())
            ):
                continue
            eligible.append(start)
        if not eligible:
            msg = f"plant {record.sites[plant]} has no eligible easy-case start day"
            raise ValueError(msg)
        events.append(
            PlantedEvent(
                plant, "step", size, int(eligible[int(rng.integers(len(eligible)))]), length
            )
        )
    return events
