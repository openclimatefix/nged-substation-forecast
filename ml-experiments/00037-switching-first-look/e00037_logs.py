"""Parse NGED switching logs into intervals during which a normal open point was closed.

The workbook's `Manual Time` is Europe/London local time. A telemetry stamp marks the end of its
half-hour, so an action at local time `t` first affects the stamps after `t` in UTC.
"""

import re
from datetime import UTC, datetime
from typing import Final, Literal

import polars as pl
from e00037_grid import GRID_LENGTH, stamp_index

ActionKindType = Literal["open", "close", "earth", "other"]

GROUPING_HOURS: Final[float] = 2.0
LONG_INTERVAL_HOURS: Final[float] = 60 * 24

_CLOSE_PATTERN: Final[str] = (
    r"^close|close to make|close to energise|close \(dead"
    r"|remove safety lock & caution notice close|^confirm closed$"
)
_OPEN_PATTERN: Final[str] = (
    r"^open|open to break|open to de-energise|open \(dead|^confirm open|^confirm opened"
)
_SWITCH_CODE: Final[re.Pattern[str]] = re.compile(r"(\d{2}[A-Z]{1,2}\d{3,4}|ABI 90 \d+|\b9\d{5}\b)")
_LOCATION_PREFIX: Final[re.Pattern[str]] = re.compile(r"^\s*(\d+|[a-z])\s*[\)\-]\s*")


def classify_action(*, action: str | None) -> ActionKindType:
    """Classify a logged action string as opening, closing, earthing, or other."""
    text = re.sub(r"\s+", " ", (action or "").lower())
    if "earth" in text:
        return "earth"
    if re.search(_CLOSE_PATTERN, text):
        return "close"
    if re.search(_OPEN_PATTERN, text):
        return "open"
    return "other"


def switch_key(*, location: str | None) -> str | None:
    """Return the switch code in a logged location, or the text before its first comma."""
    if location is None:
        return None
    stripped = _LOCATION_PREFIX.sub("", location.strip())
    match = _SWITCH_CODE.search(stripped)
    return match[1] if match else stripped.split(",")[0].strip()


def to_utc(*, local: datetime) -> datetime | None:
    """Convert a naive Europe/London time to UTC; the October repeated hour takes the earlier."""
    frame = pl.DataFrame({"t": [local]}).select(
        pl.col("t")
        .dt.replace_time_zone("Europe/London", ambiguous="earliest", non_existent="null")
        .dt.convert_time_zone("UTC")
    )
    value = frame.item()
    return None if value is None else value.astimezone(UTC)


def interval_stamps(*, close_utc: datetime, open_utc: datetime) -> tuple[int, int]:
    """Return the half-open range of period-ending stamp indices affected by a closure.

    Affected stamps `tau` satisfy `close_utc < tau <= open_utc + 30 minutes`.
    """
    return stamp_index(utc_time=close_utc) + 1, stamp_index(utc_time=open_utc) + 2


def confirmed_actions(*, raw: pl.DataFrame) -> pl.DataFrame:
    """Keep the confirmed open and close rows, with a switch key and an action time."""
    kinds = raw["action"].map_elements(
        lambda action: classify_action(action=action), return_dtype=pl.String
    )
    keys = raw["location"].map_elements(
        lambda location: switch_key(location=location), return_dtype=pl.String
    )
    return (
        raw.with_columns(kind=kinds, switch=keys, t=pl.col("manual").fill_null(pl.col("recorded")))
        .filter(pl.col("state") == "Confirmed", pl.col("kind").is_in(["open", "close"]))
        .unique(subset=["sheet", "switch", "kind", "t"])
        .sort("sheet", "switch", "t")
    )


def reconstruct_intervals(*, confirmed: pl.DataFrame) -> pl.DataFrame:
    """Build closed-interval rows per (sheet, switch) from the sorted confirmed actions.

    The normal state is open. A repeated close or open is ignored. A close with no later open
    becomes an open-ended interval (`end` null).
    """
    rows: list[dict] = []
    for (sheet, switch), group in confirmed.group_by(["sheet", "switch"], maintain_order=True):
        start, job = None, None
        for action in group.iter_rows(named=True):
            if action["kind"] == "close" and start is None:
                start, job = action["t"], action["job"]
            elif action["kind"] == "open" and start is not None:
                rows.append(
                    {
                        "sheet": sheet,
                        "switch": switch,
                        "start": start,
                        "end": action["t"],
                        "job": job,
                    }
                )
                start = None
        if start is not None:
            rows.append({"sheet": sheet, "switch": switch, "start": start, "end": None, "job": job})
    schema = {
        "sheet": pl.String,
        "switch": pl.String,
        "start": pl.Datetime("us"),
        "end": pl.Datetime("us"),
        "job": pl.String,
    }
    return pl.DataFrame(rows, schema=schema).with_columns(
        hours=(pl.col("end") - pl.col("start")).dt.total_minutes() / 60,
        side_a=pl.col("sheet").str.split("-").list.get(0),
        side_b=pl.col("sheet").str.split("-").list.get(1, null_on_oob=True),
    )


def add_stamps(*, intervals: pl.DataFrame) -> pl.DataFrame:
    """Add `start_utc`, `end_utc`, and the stamp range `first_stamp` to `last_stamp_exclusive`.

    Open-ended intervals and intervals outside the grid are dropped.
    """
    rows: list[dict] = []
    for row in intervals.filter(pl.col("end").is_not_null()).iter_rows(named=True):
        close_utc, open_utc = to_utc(local=row["start"]), to_utc(local=row["end"])
        if close_utc is None or open_utc is None:
            continue
        first, last = interval_stamps(close_utc=close_utc, open_utc=open_utc)
        if last <= 0 or first >= GRID_LENGTH:
            continue
        rows.append(
            {
                **row,
                "start_utc": close_utc,
                "end_utc": open_utc,
                "first_stamp": max(first, 0),
                "last_stamp_exclusive": min(last, GRID_LENGTH),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def group_operations(*, intervals: pl.DataFrame) -> pl.DataFrame:
    """Group closures that share a Primary and start within `GROUPING_HOURS` into operations.

    Adds `operation`, an integer id. Intervals are linked when they share a series on either side
    and start within the window of one another (single-linkage per series, then merged).
    """
    rows = intervals.sort("start_utc").to_dicts()
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    window = GROUPING_HOURS * 3600
    for i, first in enumerate(rows):
        sides_i = {first["side_a"], first["side_b"]} - {None, "unknown"}
        for j in range(i + 1, len(rows)):
            second = rows[j]
            if (second["start_utc"] - first["start_utc"]).total_seconds() > window:
                break
            sides_j = {second["side_a"], second["side_b"]} - {None, "unknown"}
            if sides_i & sides_j:
                parent[find(j)] = find(i)
    for i, row in enumerate(rows):
        row["operation"] = find(i)
    return pl.DataFrame(rows, infer_schema_length=None)
