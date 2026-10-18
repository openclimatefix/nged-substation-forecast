"""Experiment 41's labelled switching events as per-series arrays on the dense half-hourly grid.

For each Primary series the module builds three arrays aligned with `common.grid_times()`:

- `in_event`: the half-hour lies inside a labelled visible event of the series, as donor or as
  metered receiver. Onset and end are both inclusive.
- `signed_mw`: the signed MW that the events moved at that half-hour, as a fraction of the
  series' effective capacity. A donor gets minus the event's median moved MW. A metered receiver
  gets plus the median moved MW times the receiver's share. Overlapping events add.
- `closed`: the half-hour lies inside the logged closure interval of any real logged card that
  names the series as one of its two sides.

The labelled set holds events from January 2022 only, so earlier half-hours are never marked.

`load_xl_labels` builds the same arrays from the XL detector's intervals (experiment 00082's
`xl_intervals.parquet`) instead of NGED's logs. Its `closed` array is the oracle's, because the
scored rows are defined by NGED's logged closures whichever labels the model sees.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00088_common as common
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

LABELS_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00041-sw2-0a-labeller-gate" / "real" / "labelled_set.parquet"
)
XL_INTERVALS_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00082-sw2-18-19-combined-refit-offset-and-same-sign-count"
    / "xl_intervals.parquet"
)
XL_GRID_START: Final[datetime] = datetime(2022, 1, 1, tzinfo=UTC)
"""Experiment 00082's stamp 0, a half-hour on the 2022-01-01 grid."""
EVENT_KINDS: Final[tuple[str, ...]] = (
    "real_operation",
    "real_short",
    "real_edge_close",
    "real_edge_open",
    "alarm",
)
"""Label kinds that describe real events; sentinel and placebo cards are not events."""
CLOSURE_KINDS: Final[tuple[str, ...]] = (
    "real_operation",
    "real_short",
    "real_edge_close",
    "real_edge_open",
)
"""Kinds whose logged close and open times come from NGED's switching log."""
VISIBLE: Final[tuple[str, ...]] = ("conserved", "one_sided")


@dataclass(frozen=True)
class SeriesLabels:
    """One series' label arrays on the dense grid.

    Attributes:
        in_event: True where a labelled visible event covers the half-hour.
        signed_mw: Signed MW moved by events over effective capacity, 0.0 where none.
        closed: True where a logged closure of the series covers the half-hour.
    """

    in_event: np.ndarray
    signed_mw: np.ndarray
    closed: np.ndarray


def _span(*, start: object, stop: object) -> slice:
    """Return the grid slice of the half-hours from `start` to `stop`, both inclusive."""
    first = max(int(np.ceil(_index(time=start))), 0)
    last = int(np.floor(_index(time=stop)))
    return slice(first, last + 1)


def _index(*, time: object) -> float:
    """Return a timestamp's fractional grid index."""
    return (time - common.GRID_START) / common.STEP  # ty: ignore[unsupported-operator]


def load_labels(*, series: pl.DataFrame) -> dict[str, SeriesLabels]:
    """Return each series' label arrays for the series of the table.

    Args:
        series: `common.series_table()` restricted to the series to label.

    Returns:
        The `SeriesLabels` of each site label.

    Raises:
        ValueError: If a visible event lacks an onset, an end, or a moved MW.
    """
    size = common.grid_times().len()
    by_id = {int(row["time_series_id"]): row for row in series.iter_rows(named=True)}
    out = {
        row["site"]: SeriesLabels(
            in_event=np.zeros(size, dtype=bool),
            signed_mw=np.zeros(size, dtype=np.float32),
            closed=np.zeros(size, dtype=bool),
        )
        for row in series.iter_rows(named=True)
    }
    labels = pl.read_parquet(LABELS_PATH)
    events = labels.filter(pl.col("kind").is_in(EVENT_KINDS) & pl.col("visibility").is_in(VISIBLE))
    for event in events.iter_rows(named=True):
        if event["onset_utc"] is None or event["end_utc"] is None or event["moved_mw_p50"] is None:
            msg = f"visible event {event['card_id']} lacks an onset, an end, or a moved MW"
            raise ValueError(msg)
        span = _span(start=event["onset_utc"], stop=event["end_utc"])
        moved = float(event["moved_mw_p50"])
        parts: list[tuple[int, float]] = []
        if event["donor"] is not None:
            parts.append((int(event["donor"]), -moved))
        parts.extend(
            (int(receiver["series"]), moved * float(receiver["share"]))
            for receiver in json.loads(event["receivers"] or "[]")
        )
        for series_id, megawatts in parts:
            row = by_id.get(series_id)
            if row is None:
                continue
            out[row["site"]].in_event[span] = True
            out[row["site"]].signed_mw[span] += np.float32(megawatts / row["effective_capacity_mw"])
    for card in labels.filter(pl.col("kind").is_in(CLOSURE_KINDS)).iter_rows(named=True):
        span = _span(start=card["logged_close_utc"], stop=card["logged_open_utc"])
        for side in (card["side_a"], card["side_b"]):
            row = by_id.get(int(side)) if side is not None else None
            if row is not None:
                out[row["site"]].closed[span] = True
    return out


def load_xl_labels(
    *, series: pl.DataFrame, oracle: dict[str, SeriesLabels]
) -> dict[str, SeriesLabels]:
    """Return each series' label arrays from the XL detector's intervals.

    An interval `[onset, end)` of experiment 00082's stamps marks `in_event` on the half-hours
    from `onset` to `end - 1`. Its signed MW (the series' own 24-hour residual step, negative on a
    donor) is divided by the series' effective capacity and added over the span. Overlapping
    intervals add.

    Args:
        series: `common.series_table()` restricted to the series to label.
        oracle: `load_labels` output, whose `closed` arrays are reused.

    Returns:
        The `SeriesLabels` of each site label.
    """
    size = common.grid_times().len()
    by_id = {int(row["time_series_id"]): row for row in series.iter_rows(named=True)}
    out = {
        site: SeriesLabels(
            in_event=np.zeros(size, dtype=bool),
            signed_mw=np.zeros(size, dtype=np.float32),
            closed=labels.closed,
        )
        for site, labels in oracle.items()
    }
    offset = (XL_GRID_START - common.GRID_START) / common.STEP
    if offset != int(offset):
        msg = "the XL grid must lie on a whole number of half-hours from the grid start"
        raise ValueError(msg)
    for interval in pl.read_parquet(XL_INTERVALS_PATH).iter_rows(named=True):
        row = by_id.get(int(interval["series"]))
        if row is None:
            continue
        span = slice(int(interval["onset"]) + int(offset), int(interval["end"]) + int(offset))
        out[row["site"]].in_event[span] = True
        out[row["site"]].signed_mw[span] += np.float32(
            interval["step_mw"] / row["effective_capacity_mw"]
        )
    return out
