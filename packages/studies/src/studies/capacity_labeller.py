"""Knots, case files, prompts, and the tools-only labeller, for the study in issue #1147.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1147>. A
labeller (a language model, or the fixed rule `tools_only_answer`) reads one plant's case file and
writes dated capacity knots. This module holds the knot schema and its validation, the conversion of
knots to a daily capacity series, the case-file text and its two charts, the fixed system prompt,
the tools-only rule, and the cache key.

**A case file is a deterministic function of the evidence** (`studies.capacity_evidence`), so the
case file for a plant and a cut-off day is the same every time and is identical on a planted record
and on a copy with the same power. The charts fix their axis domains from the cut-off day and from
constants, so no axis reveals a later day.

**Knots are levels at dates.** Capacity is linear between consecutive knots and constant before the
first and after the last. A step is therefore two knots one day apart. With no knots the capacity
is the plant's static 99th-percentile power.
"""

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import pairwise
from typing import Final, Literal

import altair as alt
import numpy as np
import vl_convert as vlc

from studies.capacity_evidence import Changepoint, Evidence

CauseType = Literal[
    "string_failure", "soiling", "degradation", "curtailment", "model_bias", "data_fault"
]
"""The causes a knot may name."""

CAUSES: Final[tuple[str, ...]] = (
    "string_failure",
    "soiling",
    "degradation",
    "curtailment",
    "model_bias",
    "data_fault",
)
"""The runtime tuple of `CauseType`'s values."""

MAX_KNOTS: Final[int] = 24
"""The most knots an answer may hold."""

TOOLS_ONLY_MAX_P: Final[float] = 0.01
"""The tools-only rule writes a knot at a changepoint only when its p-value is below this."""

TOOLS_ONLY_MIN_SIZE: Final[float] = 0.08
"""The tools-only rule ignores a changepoint smaller than this fraction."""

TOOLS_V2_MAX_EPISODES: Final[int] = 5
"""Round 2's tools-only rule marks at most this many episodes (the deepest), so that its knots,
four to an episode and one opening knot, stay within `MAX_KNOTS`."""

TOOLS_ONLY_MIN_LEVEL: Final[float] = 0.25
TOOLS_ONLY_MAX_LEVEL: Final[float] = 1.05
"""The tools-only rule keeps each segment's level, a share of the static power, between these, so
that its knots pass the bounds check by construction."""

CAP_OVERLAP_SHARE: Final[float] = 0.2
"""The tools-only rule calls a change curtailment when the cap binds on this share of daylight."""

CAP_OVERLAP_DAYS: Final[int] = 14
"""The days either side of a change over which the tools-only rule looks for a binding cap."""

HOLE_FRACTION: Final[float] = 0.75
"""An hour-of-day value below this share of its month's median marks a hole."""

HOURS_SHOWN: Final[tuple[int, ...]] = tuple(range(4, 20))
"""The hours (0 to 23, the hour of the day on the record's clock) the hour-of-day table lists."""

SYSTEM_PROMPT: Final[str] = """You label changes in the capacity of one solar power plant. You are \
given a case file of measurements and the output of analysis tools for the plant, and you reply \
with one JSON object and nothing else.

How to read a case file:
- Power is the plant's measured output. The physics model turns satellite irradiance into the \
output a plant of fixed capacity would give. A ratio of power to that output near 100 percent \
means the plant behaves as it did in its reference year; a lasting fall means lost capacity.
- The seasonal correction divides out the physics model's own seasonal error: the shared column \
is what the other plants' ratio did on the same day, and the own column is this plant's profile \
from earlier years. A move in the shared column is the physics model or the weather, not this plant.
- Gap days have too little data. Stuck days hold a run of identical readings, which points to a \
metering fault. The cap columns, if present, show an export cap from network management: when it \
binds, lower power is curtailment, not lost capacity.
- The hour-of-day table shows the corrected ratio by month and hour. A hole in a few hours, in the \
morning or the evening, points to a failed string or shading; a flat fall across all hours \
points to soiling or degradation. Compare a month with the same month in other years.
- Changepoint candidates come from a statistical test and can be wrong. A candidate marked \
unconfirmed is within 14 days of the end of the data.

Rules:
1. The expected answer for most periods is no change: set "no_change" to true and "knots" to an \
empty list. Write a knot only when several tools agree on a change that lasts.
2. A knot is a date and a capacity in megawatts. Capacity is linear between consecutive knots, \
constant before the first knot at the first knot's level, and constant after the last knot. With \
no knots the capacity is the static power given in the case file.
3. To write a step change, write two knots one day apart: the old level, then the new level. To \
write the end of a loss, write the lower level and then the recovered level in the same way. If \
you write any knot, start with one at the first date the case file allows, at the level you \
believe is normal.
4. Knots must be in strictly increasing date order and inside the dates the case file allows \
("Knot dates may run from ... to ..."). Write at most 24 knots.
5. Each knot names one cause from: string_failure, soiling, degradation, curtailment, model_bias, \
data_fault. The evidence list quotes short facts from the case file, such as a changepoint line.

Reply with this JSON shape only:
{"plant": "<label>", "period": ["<first allowed date>", "<last allowed date>"], \
"no_change": <true or false>, \
"knots": [{"date": "YYYY-MM-DD", "capacity_mw": <number>, "cause": "<cause>", \
"evidence": ["<short fact>"]}]}"""
"""The fixed system prompt, identical for every call of every arm."""


_V2_RULE_OLD: Final[str] = "Write a knot only when several tools agree on a change that lasts."
_V2_RULE_NEW: Final[str] = (
    "Write a knot pair when several tools agree on a change. A loss that lasts a month or more is "
    "a legitimate change when the local-scan table and the monthly table both show a fall of "
    "similar size at the same dates, even if the changepoint list lists nothing there. Write the "
    "drop and the recovery as two knot pairs."
)
_V2_READING_OLD: Final[str] = "- Changepoint candidates come from"
_V2_READING_NEW: Final[str] = (
    "- The local scan compares every 30-day window with the 30 days either side of it, on the "
    "corrected clear-sky ratio and on the all-sky ratio of energy to modelled energy. An episode "
    "lists the days it spans, the deepest window's fall or rise, and the all-sky figure for that "
    "window. A one-month loss shows as a down episode with a matching fall in the monthly table. "
    "Some episodes are noise, so check each against the monthly table.\n"
    "- Changepoint candidates come from"
)
SYSTEM_PROMPT_V2: Final[str] = (
    SYSTEM_PROMPT.replace(_V2_RULE_OLD, _V2_RULE_NEW)
    .replace(_V2_READING_OLD, _V2_READING_NEW)
    .replace(
        '"no_change": <true or false>, ',
        '"no_change": <true or false>, "note": "<one sentence on why, when there is no change>", ',
    )
)
"""The round-2 system prompt: round 1's with a softened rule for losses that last a month, the local
scan described, and a one-sentence note on why there is no change."""

assert _V2_RULE_NEW in SYSTEM_PROMPT_V2
assert "local scan" in SYSTEM_PROMPT_V2
assert '"note"' in SYSTEM_PROMPT_V2


class AnswerError(ValueError):
    """A labeller's reply is not a valid answer."""


@dataclass(frozen=True)
class Knot:
    """One dated capacity level.

    Attributes:
        day: The date.
        capacity_mw: The capacity in megawatts.
        cause: The cause the labeller names.
        evidence: Short facts that support the knot.
    """

    day: date
    capacity_mw: float
    cause: CauseType
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class Answer:
    """A labeller's validated reply.

    Attributes:
        plant: The plant's label.
        first_day: The first date of the period.
        last_day: The last date of the period.
        no_change: Whether the labeller found no change.
        knots: The knots, in strictly increasing date order.
    """

    plant: str
    first_day: date
    last_day: date
    no_change: bool
    knots: tuple[Knot, ...]


def _as_date(*, value: object, name: str) -> date:
    """Parse an ISO date or raise `AnswerError`."""
    try:
        return date.fromisoformat(str(value))
    except ValueError as error:
        msg = f"{name} {value!r} is not an ISO date"
        raise AnswerError(msg) from error


def _parse_knot(*, raw: object, first_day: date, last_day: date) -> Knot:
    """Validate one knot of a reply."""
    if not isinstance(raw, dict):
        msg = f"a knot must be an object, not {raw!r}"
        raise AnswerError(msg)
    day = _as_date(value=raw.get("date"), name="knot date")
    if not first_day <= day <= last_day:
        msg = f"knot date {day} is outside the period {first_day} to {last_day}"
        raise AnswerError(msg)
    capacity = raw.get("capacity_mw")
    if isinstance(capacity, bool) or not isinstance(capacity, int | float) or not capacity > 0:
        msg = f"capacity_mw {capacity!r} must be a positive number"
        raise AnswerError(msg)
    cause = raw.get("cause")
    if cause not in CAUSES:
        msg = f"cause {cause!r} is not one of {CAUSES}"
        raise AnswerError(msg)
    evidence = raw.get("evidence", [])
    if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
        msg = "evidence must be a list of strings"
        raise AnswerError(msg)
    return Knot(day=day, capacity_mw=float(capacity), cause=cause, evidence=tuple(evidence))


def parse_answer(*, text: str, plant: str, first_day: date, last_day: date) -> Answer:
    """Parse and validate a labeller's reply.

    Args:
        text: The reply, possibly wrapped in prose or a code fence.
        plant: The plant the case file was about.
        first_day: The first date of the period.
        last_day: The last date of the period.

    Returns:
        The answer.

    Raises:
        AnswerError: If the reply holds no JSON object, names another plant, leaves the period,
            lists knots out of order, holds more than `MAX_KNOTS` knots, or contradicts itself
            about `no_change`.
    """
    start, stop = text.find("{"), text.rfind("}")
    if start < 0 or stop < start:
        msg = "the reply holds no JSON object"
        raise AnswerError(msg)
    try:
        payload = json.loads(text[start : stop + 1])
    except json.JSONDecodeError as error:
        msg = f"the reply is not valid JSON: {error}"
        raise AnswerError(msg) from error
    if not isinstance(payload, dict) or payload.get("plant") != plant:
        msg = f"the reply is not an object about plant {plant}"
        raise AnswerError(msg)
    raw_knots = payload.get("knots")
    if not isinstance(raw_knots, list) or len(raw_knots) > MAX_KNOTS:
        msg = f"knots must be a list of at most {MAX_KNOTS} entries"
        raise AnswerError(msg)
    knots = tuple(_parse_knot(raw=raw, first_day=first_day, last_day=last_day) for raw in raw_knots)
    if any(a.day >= b.day for a, b in pairwise(knots)):
        msg = "knots must be in strictly increasing date order"
        raise AnswerError(msg)
    no_change = payload.get("no_change")
    if not isinstance(no_change, bool) or no_change == bool(knots):
        msg = "no_change must be true exactly when knots is empty"
        raise AnswerError(msg)
    return Answer(
        plant=plant, first_day=first_day, last_day=last_day, no_change=no_change, knots=knots
    )


def capacity_series(
    *, knots: Sequence[Knot], dates: np.ndarray, static_p99_mw: float
) -> np.ndarray:
    """Return the daily capacity the knots describe.

    Args:
        knots: The knots in date order.
        dates: One calendar day per entry, as `datetime64[D]`.
        static_p99_mw: The capacity used when there are no knots.

    Returns:
        Linear interpolation between knots, constant before the first and after the last.
    """
    if not knots:
        return np.full(len(dates), static_p99_mw)
    positions = np.array([(np.datetime64(knot.day) - dates[0]).astype(int) for knot in knots])
    levels = np.array([knot.capacity_mw for knot in knots])
    return np.interp(np.arange(len(dates)), positions, levels)


def answer_to_json(*, answer: Answer) -> str:
    """Write an answer in the shape the system prompt asks for.

    Args:
        answer: The answer.

    Returns:
        A JSON string.
    """
    return json.dumps(
        {
            "plant": answer.plant,
            "period": [answer.first_day.isoformat(), answer.last_day.isoformat()],
            "no_change": answer.no_change,
            "knots": [
                {
                    "date": knot.day.isoformat(),
                    "capacity_mw": round(knot.capacity_mw, 3),
                    "cause": knot.cause,
                    "evidence": list(knot.evidence),
                }
                for knot in answer.knots
            ],
        }
    )


def _hole_in_month(*, evidence: Evidence, day: date) -> bool:
    """Say whether the hour-of-day table shows a hole in the month of `day`."""
    label = str(np.datetime64(day, "M"))
    if label not in evidence.months:
        return False
    row = evidence.hour_table[evidence.months.index(label)][list(HOURS_SHOWN)]
    row = row[np.isfinite(row)]
    if len(row) < 4:
        return False
    return bool(np.sum(row < HOLE_FRACTION * np.median(row)) >= 2)


def _near(*, evidence: Evidence, day: date) -> slice:
    """Return the slice of day indices within `CAP_OVERLAP_DAYS` of `day`."""
    centre = int((np.datetime64(day) - evidence.dates[0]).astype(int))
    return slice(max(centre - CAP_OVERLAP_DAYS, 0), centre + CAP_OVERLAP_DAYS + 1)


def tools_only_cause(*, evidence: Evidence, change: Changepoint) -> CauseType:
    """Return the cause the tools-only decision table gives a changepoint.

    The table runs in this order: a binding export cap over `CAP_OVERLAP_SHARE` of daylight near the
    change is curtailment, a stuck day near the change is a data fault, a hole in the hour-of-day
    table is a string failure, any other fall is degradation, and any other rise is model bias.

    Args:
        evidence: The evidence.
        change: The changepoint.

    Returns:
        The cause.
    """
    near = _near(evidence=evidence, day=change.day)
    if evidence.cap_binding_share is not None:
        share = evidence.cap_binding_share[near]
        if np.isfinite(share).any() and float(np.nanmean(share)) >= CAP_OVERLAP_SHARE:
            return "curtailment"
    if bool(evidence.stuck[near].any()):
        return "data_fault"
    if change.direction == "up":
        return "model_bias"
    return "string_failure" if _hole_in_month(evidence=evidence, day=change.day) else "degradation"


def tools_only_answer(*, evidence: Evidence, first_day: date, last_day: date) -> Answer:
    """Write a knot at each strong changepoint, with the cause from the decision table.

    Args:
        evidence: The evidence.
        first_day: The first date of the period.
        last_day: The last date of the period.

    Returns:
        The answer. A strong changepoint has a p-value below `TOOLS_ONLY_MAX_P` and a size above
        `TOOLS_ONLY_MIN_SIZE`. Each stretch between strong changepoints has the median corrected
        clear-sky ratio of its days as its level, as a share of the static power, and each step is
        two knots one day apart. The first knot, at the first date, holds the opening level.
    """
    strong = [
        change
        for change in evidence.changepoints
        if change.p_value < TOOLS_ONLY_MAX_P and abs(change.size) > TOOLS_ONLY_MIN_SIZE
    ]
    if not strong:
        return Answer(
            plant=evidence.plant, first_day=first_day, last_day=last_day, no_change=True, knots=()
        )
    static = evidence.static_p99_mw
    edges = [0, *(change.day_index for change in strong), len(evidence.dates)]
    levels = []
    for start, stop in pairwise(edges):
        block = evidence.clear_corrected[start:stop]
        block = block[np.isfinite(block)]
        level = float(np.median(block)) if len(block) else (levels[-1] if levels else 1.0)
        levels.append(min(max(level, TOOLS_ONLY_MIN_LEVEL), TOOLS_ONLY_MAX_LEVEL))
    knots = [
        Knot(
            day=first_day,
            capacity_mw=static * levels[0],
            cause="model_bias",
            evidence=("opening level",),
        )
    ]
    for index, change in enumerate(strong, start=1):
        cause = tools_only_cause(evidence=evidence, change=change)
        note = (
            f"changepoint {change.day} {change.direction} {change.size:+.0%} p={change.p_value:.3f}"
        )
        before = change.day - timedelta(days=1)
        if before <= knots[-1].day:
            continue
        knots.append(
            Knot(day=before, capacity_mw=static * levels[index - 1], cause=cause, evidence=(note,))
        )
        knots.append(
            Knot(day=change.day, capacity_mw=static * levels[index], cause=cause, evidence=(note,))
        )
    return Answer(
        plant=evidence.plant,
        first_day=first_day,
        last_day=last_day,
        no_change=False,
        knots=tuple(knots),
    )


def tools_only_v2_answer(*, evidence: Evidence, first_day: date, last_day: date) -> Answer:
    """Write a knot pair at the start and the end of each local-scan episode (round 2's rule).

    Each episode becomes the old level the day before it starts, the level its deepest window shows
    (the static power times one plus the effect) on its first day, the same level on its last day,
    and the static power the day after. The cause comes from the same decision table as round 1.
    Only the `TOOLS_V2_MAX_EPISODES` deepest episodes are used. Episodes that overlap an earlier
    episode's knots are skipped, and the answer opens with a knot
    at the first date at the static power.

    Args:
        evidence: The evidence.
        first_day: The first date a knot may take.
        last_day: The last date a knot may take.

    Returns:
        The answer; `no_change` when the scan lists no episode.
    """
    static = evidence.static_p99_mw
    knots: list[Knot] = []
    deepest = sorted(evidence.local_episodes, key=lambda e: -abs(e.effect))[:TOOLS_V2_MAX_EPISODES]
    for episode in sorted(deepest, key=lambda e: e.first_day):
        start = evidence.dates[episode.first_day].astype(date)
        end = evidence.dates[min(episode.last_day, len(evidence.dates) - 1)].astype(date)
        if start - timedelta(days=1) <= (knots[-1].day if knots else first_day) or end >= last_day:
            continue
        level = min(max(1.0 + episode.effect, TOOLS_ONLY_MIN_LEVEL), TOOLS_ONLY_MAX_LEVEL)
        change = Changepoint(
            day_index=episode.first_day,
            day=start,
            direction=episode.direction,
            size=episode.effect,
            p_value=0.0,
            unconfirmed=False,
        )
        cause = tools_only_cause(evidence=evidence, change=change)
        note = (evidence.dates[episode.deepest_start].astype(date).isoformat(),)
        evidence_text = (f"local scan {start} to {end} {episode.effect:+.0%}", *note)
        knots += [
            Knot(
                day=start - timedelta(days=1),
                capacity_mw=static,
                cause=cause,
                evidence=evidence_text,
            ),
            Knot(day=start, capacity_mw=static * level, cause=cause, evidence=evidence_text),
            Knot(day=end, capacity_mw=static * level, cause=cause, evidence=evidence_text),
            Knot(
                day=end + timedelta(days=1), capacity_mw=static, cause=cause, evidence=evidence_text
            ),
        ]
    if not knots:
        return Answer(
            plant=evidence.plant, first_day=first_day, last_day=last_day, no_change=True, knots=()
        )
    opening = Knot(
        day=first_day, capacity_mw=static, cause="model_bias", evidence=("opening level",)
    )
    if knots[0].day > first_day:
        knots.insert(0, opening)
    return Answer(
        plant=evidence.plant,
        first_day=first_day,
        last_day=last_day,
        no_change=False,
        knots=tuple(knots),
    )


def _cell(*, value: float, scale: float = 100.0) -> str:
    """Format a ratio as an integer percentage, or `-` where it is NaN."""
    return "-" if not np.isfinite(value) else str(round(value * scale))


def _monthly_median(*, values: np.ndarray, months: np.ndarray, month: np.datetime64) -> float:
    """Return the median of the finite values in one month, NaN if there are none."""
    block = values[months == month]
    block = block[np.isfinite(block)]
    return float(np.median(block)) if len(block) else float("nan")


def monthly_table(*, evidence: Evidence, shown_months: int | None) -> str:
    """Write the monthly summary table.

    Args:
        evidence: The evidence.
        shown_months: How many of the latest months to list, or `None` for all.

    Returns:
        A comma-separated table with a header line.
    """
    months = evidence.dates.astype("datetime64[M]")
    labels = evidence.months if shown_months is None else evidence.months[-shown_months:]
    has_cap = evidence.cap_binding_share is not None
    header = "month,gap_days,stuck_days,clear_days,raw,corrected,shared,own,neighbour,plateau,clip"
    lines = [header + (",cap_binding,cap_mw" if has_cap else "")]
    for label in labels:
        month = np.datetime64(label)
        in_month = months == month
        index = evidence.months.index(label)
        cells = [
            label,
            str(int(evidence.gap[in_month].sum())),
            str(int(evidence.stuck[in_month].sum())),
            str(int(np.isfinite(evidence.clear_raw[in_month]).sum())),
            _cell(value=_monthly_median(values=evidence.clear_raw, months=months, month=month)),
            _cell(
                value=_monthly_median(values=evidence.clear_corrected, months=months, month=month)
            ),
            _cell(value=_monthly_median(values=evidence.shared, months=months, month=month)),
            _cell(value=_monthly_median(values=evidence.own, months=months, month=month)),
            _cell(value=_monthly_median(values=evidence.neighbour, months=months, month=month)),
            _cell(value=float(evidence.plateau[index])),
            _cell(value=float(evidence.clip_share[index])),
        ]
        if has_cap:
            assert evidence.cap_binding_share is not None
            assert evidence.cap_mean_mw is not None
            cells += [
                _cell(
                    value=_monthly_median(
                        values=evidence.cap_binding_share, months=months, month=month
                    )
                ),
                _fmt_mw(
                    value=_monthly_median(values=evidence.cap_mean_mw, months=months, month=month)
                ),
            ]
        lines.append(",".join(cells))
    return "\n".join(lines)


def _fmt_mw(*, value: float) -> str:
    """Format a megawatt value with one decimal, or `-` where it is NaN."""
    return "-" if not np.isfinite(value) else f"{value:.1f}"


def hour_table_text(*, evidence: Evidence, shown_months: int | None) -> str:
    """Write the hour-of-day table, one row per month.

    Args:
        evidence: The evidence.
        shown_months: How many of the latest months to list, or `None` for all.

    Returns:
        A comma-separated table of corrected ratios in percent, one column per hour in
        `HOURS_SHOWN`.
    """
    start = 0 if shown_months is None else max(len(evidence.months) - shown_months, 0)
    lines = ["month," + ",".join(f"h{hour:02d}" for hour in HOURS_SHOWN)]
    for row in range(start, len(evidence.months)):
        cells = [_cell(value=float(evidence.hour_table[row, hour])) for hour in HOURS_SHOWN]
        lines.append(evidence.months[row] + "," + ",".join(cells))
    return "\n".join(lines)


def envelope_text(*, evidence: Evidence, shown_months: int | None) -> str:
    """Write the monthly 95th percentile of power per irradiance bin, in percent of static power.

    Args:
        evidence: The evidence.
        shown_months: How many of the latest months to list, or `None` for all.

    Returns:
        A comma-separated table, one column per irradiance bin of 100 W/m2 from 100 to 800.
    """
    start = 0 if shown_months is None else max(len(evidence.months) - shown_months, 0)
    bins = evidence.envelope.shape[1]
    lines = ["month," + ",".join(f"ghi{100 * (column + 1)}" for column in range(bins))]
    for row in range(start, len(evidence.months)):
        cells = [_cell(value=float(value)) for value in evidence.envelope[row]]
        lines.append(evidence.months[row] + "," + ",".join(cells))
    return "\n".join(lines)


def changepoint_text(*, evidence: Evidence) -> str:
    """Write the changepoint candidates, one per line."""
    if not evidence.changepoints:
        return "none found"
    return "\n".join(
        f"{change.day} {change.direction} {change.size:+.0%} p={change.p_value:.3f}"
        + (" unconfirmed" if change.unconfirmed else "")
        for change in evidence.changepoints
    )


def raw_summary_text(*, evidence: Evidence) -> str:
    """Write weekly totals of measured and modelled energy and the data-quality flags.

    Args:
        evidence: The evidence.

    Returns:
        A comma-separated table: week start, measured megawatt-hours, modelled megawatt-hours at the
        fitted direct-current capacity, gap days, and stuck days.
    """
    lines = ["week_start,measured_mwh,modelled_mwh,gap_days,stuck_days"]
    for start in range(0, len(evidence.dates), 7):
        block = slice(start, start + 7)
        lines.append(
            ",".join(
                [
                    str(evidence.dates[start]),
                    _fmt_mw(value=float(np.nansum(evidence.energy_mwh[block]))),
                    _fmt_mw(value=float(np.nansum(evidence.modelled_mwh[block]))),
                    str(int(evidence.gap[block].sum())),
                    str(int(evidence.stuck[block].sum())),
                ]
            )
        )
    return "\n".join(lines)


ALLOWED_WINDOW_PREFIX: Final[str] = "Knot dates may run from "
"""The start of the case-file line that states the dates a knot may take."""

_ALLOWED_WINDOW: Final[re.Pattern[str]] = re.compile(
    re.escape(ALLOWED_WINDOW_PREFIX) + r"(\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})\."
)


def allowed_window(*, text: str) -> tuple[date, date]:
    """Read the dates a knot may take from a case file's text.

    Args:
        text: The case-file text.

    Returns:
        The first and last allowed dates. `parse_answer` is given these, so the prompt and the
        parser cannot disagree about the window.

    Raises:
        ValueError: If the text states no window.
    """
    found = _ALLOWED_WINDOW.search(text)
    if found is None:
        msg = "the case file states no window of allowed knot dates"
        raise ValueError(msg)
    return date.fromisoformat(found.group(1)), date.fromisoformat(found.group(2))


CaseKindType = Literal["tools", "raw"]
"""Which tools a case file shows: the six tools, or only weekly energy totals and data flags."""


def local_scan_text(*, evidence: Evidence) -> str:
    """Write the local scan's episodes, one per line.

    Args:
        evidence: The evidence.

    Returns:
        Lines of first day, last day, direction, the deepest window's effect and start, and the
        all-sky effect, or `none found`.
    """
    if not evidence.local_episodes:
        return "none found"
    lines = []
    for episode in evidence.local_episodes:
        first = evidence.dates[episode.first_day]
        last = evidence.dates[min(episode.last_day, len(evidence.dates) - 1)]
        all_sky = (
            "n/a" if not np.isfinite(episode.all_sky_effect) else f"{episode.all_sky_effect:+.0%}"
        )
        lines.append(
            f"{first} to {last} {episode.direction} {episode.effect:+.0%} "
            f"(deepest window starts {evidence.dates[episode.deepest_start]}; all-sky {all_sky})"
        )
    return "\n".join(lines)


def case_file_text(
    *,
    evidence: Evidence,
    kind: CaseKindType,
    shown_months: int | None,
    current_knots: Sequence[Knot] | None = None,
    issue_day: date | None = None,
    period_first: date | None = None,
    local_scan: bool = False,
) -> str:
    """Write a case file as text.

    Args:
        evidence: The evidence, already cut at the latest day the labeller may read.
        kind: `tools` for the six tools, `raw` for weekly totals and flags only.
        shown_months: How many of the latest months the monthly tables list, or `None` for all.
        current_knots: The labeller's previous estimate, for a causal call, or `None`.
        issue_day: The day the labeller is asked, for a causal call, or `None`.
        period_first: The first date a knot may take, or `None` for the first day of the data.
        local_scan: Whether to add the local-scan table (round 2's seventh evidence table).

    Returns:
        The case file.
    """
    first, last = evidence.dates[0].astype(object), evidence.dates[-1].astype(object)
    window_first = first if period_first is None else period_first
    fit = evidence.fit
    static_text = (
        f"Static power (99th percentile of the readings in this case file): "
        f"{evidence.static_p99_mw:.2f} MW."
    )
    if fit is not None:
        static_text += (
            f" Fitted AC capacity {fit.ac_capacity_mw:.2f} MW, "
            f"DC capacity {fit.dc_capacity_mw:.2f} MW, "
            f"tilt {fit.tilt_deg:.0f} deg, azimuth {fit.azimuth_deg:.0f} deg."
        )
    parts = [
        f"# Case file for plant {evidence.plant}",
        f"Data: {first} to {last} ({len(evidence.dates)} days).",
        f"{ALLOWED_WINDOW_PREFIX}{window_first} to {last}.",
        static_text,
    ]
    if issue_day is not None:
        parts.append(
            f"Today is {issue_day}. The data end on {last}, two days earlier than the latest "
            "whole day, because the satellite data arrive late. Your current estimate is below; "
            "revise it only if the evidence supports a change."
        )
        knots_json = json.dumps(
            [
                {"date": k.day.isoformat(), "capacity_mw": round(k.capacity_mw, 3)}
                for k in current_knots or ()
            ]
        )
        parts.append(f"Current knots: {knots_json}")
    if kind == "raw":
        parts.append("## Weekly totals and data flags\n" + raw_summary_text(evidence=evidence))
        return "\n\n".join(parts)
    parts += [
        "Ratios are integer percentages of the plant's own normal level.",
        "## Monthly summary\nraw is the clear-sky ratio before the seasonal correction; "
        "corrected is after it; shared and own are the two correction terms; neighbour is the "
        "7-day output over the other plants'; plateau is the 99th percentile of power over static "
        "power; clip is the share of daylight hours within 2 percent of the plateau.\n"
        + monthly_table(evidence=evidence, shown_months=shown_months),
        "## Hour of day\nCorrected ratio by month (rows) and hour (columns).\n"
        + hour_table_text(evidence=evidence, shown_months=shown_months),
        "## Power envelope\n95th percentile of power over static power, by satellite irradiance "
        "(W/m2).\n" + envelope_text(evidence=evidence, shown_months=shown_months),
        "## Changepoint candidates\n" + changepoint_text(evidence=evidence),
    ]
    if local_scan:
        parts.append("## Local scan\n" + local_scan_text(evidence=evidence))
    return "\n\n".join(parts)


CHART_RATIO_DOMAIN: Final[tuple[float, float]] = (0.3, 1.5)
"""The fixed vertical domain of the ratio chart."""

CHART_HEAT_DOMAIN: Final[tuple[float, float]] = (0.5, 1.5)
"""The fixed colour domain of the hour-of-day chart."""


def ratio_chart_spec(*, evidence: Evidence) -> dict[str, object]:
    """Return the Vega-Lite spec of the corrected clear-sky ratio against date.

    Args:
        evidence: The evidence.

    Returns:
        The spec. Its horizontal domain is the cut record's first and last dates and its vertical
        domain is a constant, so no axis depends on a later day.
    """
    finite = np.isfinite(evidence.clear_corrected)
    points = [
        {"date": str(day), "ratio": float(value)}
        for day, value in zip(evidence.dates[finite], evidence.clear_corrected[finite], strict=True)
    ]
    rules = [
        {"date": str(change.day), "direction": change.direction} for change in evidence.changepoints
    ]
    domain = [str(evidence.dates[0]), str(evidence.dates[-1])]
    x = alt.X("date:T", scale=alt.Scale(domain=domain), title=None)
    layers = [
        alt.Chart(alt.Data(values=points))
        .mark_point(size=6, opacity=0.5, color="#555555")
        .encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y=alt.Y(
                "ratio:Q",
                scale=alt.Scale(domain=list(CHART_RATIO_DOMAIN)),
                title="corrected clear-sky ratio",
            ),
        )
    ]
    if rules:
        layers.append(
            alt.Chart(alt.Data(values=rules))
            .mark_rule(color="#d62728", strokeDash=[4, 3])
            .encode(x=x)  # ty: ignore[unresolved-attribute]
        )
    if evidence.cap_binding_share is not None:
        capped = [
            {"date": str(day), "y": CHART_RATIO_DOMAIN[0] + 0.03}
            for day, share in zip(evidence.dates, evidence.cap_binding_share, strict=True)
            if np.isfinite(share) and share >= 0.5
        ]
        if capped:
            layers.append(
                alt.Chart(alt.Data(values=capped))
                .mark_tick(color="#ff7f0e", thickness=2, size=8)
                .encode(x=x, y=alt.Y("y:Q", scale=alt.Scale(domain=list(CHART_RATIO_DOMAIN))))  # ty: ignore[unresolved-attribute]
            )
    return alt.layer(*layers).properties(width=640, height=240).to_dict()


def heatmap_spec(*, evidence: Evidence, shown_months: int | None) -> dict[str, object]:
    """Return the Vega-Lite spec of the hour-of-day corrected ratio by month.

    Args:
        evidence: The evidence.
        shown_months: How many of the latest months to draw, or `None` for all.

    Returns:
        The spec, with a constant colour domain.
    """
    start = 0 if shown_months is None else max(len(evidence.months) - shown_months, 0)
    cells = [
        {
            "month": evidence.months[row],
            "hour": f"{hour:02d}",
            "ratio": float(evidence.hour_table[row, hour]),
        }
        for row in range(start, len(evidence.months))
        for hour in HOURS_SHOWN
        if np.isfinite(evidence.hour_table[row, hour])
    ]
    cells = cells or [{"month": evidence.months[-1], "hour": "12", "ratio": 1.0}]
    return (
        alt.Chart(alt.Data(values=cells))
        .mark_rect()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("hour:O", title="hour"),
            y=alt.Y("month:O", title=None, sort=evidence.months[start:]),
            color=alt.Color(
                "ratio:Q",
                scale=alt.Scale(domain=list(CHART_HEAT_DOMAIN), scheme="redyellowblue", clamp=True),
                legend=None,
            ),
        )
        .properties(width=300, height=max(120, 6 * len(evidence.months[start:])))
        .to_dict()
    )


def chart_pngs(*, evidence: Evidence, shown_months: int | None) -> tuple[bytes, bytes]:
    """Render the two case-file charts to PNG bytes.

    Args:
        evidence: The evidence.
        shown_months: How many of the latest months the heatmap draws, or `None` for all.

    Returns:
        The ratio chart and the heatmap.
    """
    return (
        vlc.vegalite_to_png(ratio_chart_spec(evidence=evidence), scale=1.0),
        vlc.vegalite_to_png(heatmap_spec(evidence=evidence, shown_months=shown_months), scale=1.0),
    )


DATA_PATH_PATTERN: Final[re.Pattern[str]] = re.compile(r"(^|[\s'\"(/])data/", re.MULTILINE)
"""Matches a path under `data/`, which no prompt may hold."""


@dataclass(frozen=True)
class Prompt:
    """Everything one call sends.

    Attributes:
        system: The system prompt.
        text: The case file and the request.
        images: PNG bytes of the case-file charts.
    """

    system: str
    text: str
    images: tuple[bytes, ...]


def build_prompt(*, text: str, images: Sequence[bytes], system: str = SYSTEM_PROMPT) -> Prompt:
    """Wrap a case file in the fixed system prompt, after checking it names no path under `data/`.

    Args:
        text: The case-file text.
        images: The chart PNGs.
        system: The system prompt, round 1's unless a round asks for another.

    Returns:
        The prompt.

    Raises:
        ValueError: If the text or the system prompt holds a path under `data/`.
    """
    for part in (system, text):
        if DATA_PATH_PATTERN.search(part):
            msg = "a prompt must not name a path under data/"
            raise ValueError(msg)
    return Prompt(system=system, text=text, images=tuple(images))


def cache_key(*, model: str, prompt: Prompt, sample_index: int) -> str:
    """Return the SHA-256 key of a call: the model, the prompts, the image bytes, and the sample.

    Args:
        model: The exact model identifier.
        prompt: The prompt.
        sample_index: Which sample of the same prompt this is.

    Returns:
        A hexadecimal digest that changes if any input changes by one character or byte.
    """
    digest = hashlib.sha256()
    for part in (model.encode(), prompt.system.encode(), prompt.text.encode()):
        digest.update(hashlib.sha256(part).digest())
    for image in prompt.images:
        digest.update(hashlib.sha256(image).digest())
    digest.update(str(sample_index).encode())
    return digest.hexdigest()
