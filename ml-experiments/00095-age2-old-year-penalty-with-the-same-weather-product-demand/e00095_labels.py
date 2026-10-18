"""Experiment 41's labelled switching events as per-series arrays on the dense half-hourly grid.

Copied from experiment 00089's `e00089_labels.py`, without its detector-label variant. The arrays
are `in_event` (a labelled visible event covers the half-hour), `signed_mw` (the signed MW moved,
over effective capacity), and `closed` (a logged closure of the series covers the half-hour). The
labelled set holds events from January 2022 only, so earlier half-hours are never marked.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00095_common as common
import numpy as np
import polars as pl

LABELS_PATH: Final[Path] = common.LABELS_PATH
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
