# ruff: noqa: E501
"""Alarm selection at a fixed alarm budget and scoring against the labelled real transfers and injected ones.

Copies the scoring helpers of experiment 195's `e00195_run.py` so that this folder imports only itself.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import polars as pl

BUDGET: Final[int] = 80
ONSET_TOLERANCE: Final[int] = 4
END_MINIMUM: Final[int] = 12
END_FRACTION: Final[float] = 0.10


def threshold(*, score: np.ndarray, table: pl.DataFrame, budget: int = BUDGET) -> float:
    """Score above which exactly `budget` unexplained odd-month candidates remain."""
    pool = score[table["odd"].to_numpy() & ~table["explained"].to_numpy() & np.isfinite(score)]
    ordered = np.sort(pool)[::-1]
    return float(ordered[budget]) if len(ordered) > budget else -np.inf


def flags(*, score: np.ndarray, thr: float) -> np.ndarray:
    """Candidates whose score is above `thr`."""
    return np.isfinite(score) & (score > thr)


def clusters(*, table: pl.DataFrame, flagged: np.ndarray) -> int:
    """Unexplained odd-month flagged candidates, merging those within 2 hours on one series."""
    rows = table.filter(
        pl.Series(flagged & table["odd"].to_numpy() & ~table["explained"].to_numpy())
    ).sort("series", "stamp")
    count, last = 0, (None, -99)
    for series, stamp in rows.select("series", "stamp").iter_rows():
        if series != last[0] or stamp - last[1] > ONSET_TOLERANCE:
            count += 1
        last = (series, stamp)
    return count


def end_tolerance(*, onset: int, end: int) -> float:
    """The larger of 6 hours and 10% of the duration, in half-hours."""
    return max(float(END_MINIMUM), END_FRACTION * (end - onset))


def real_hits(*, table: pl.DataFrame, flagged: np.ndarray, events: pl.DataFrame) -> pl.DataFrame:
    """For each labelled transfer: onset found, found (start and end), and the onset error in hours."""
    alarms = table.filter(pl.Series(flagged))
    by_series = {s: g for (s,), g in alarms.group_by("series")}
    rows = []
    for e in events.iter_rows(named=True):
        onset_found, found, errors = False, False, []
        for s in e["series"]:
            g = by_series.get(s)
            if g is None:
                continue
            for stamp, end_est in zip(g["stamp"].to_list(), g["end_est"].to_list(), strict=True):
                gap = abs(stamp - e["onset"])
                if gap <= 48:
                    errors.append(gap)
                if gap <= ONSET_TOLERANCE:
                    onset_found = True
                    if abs(end_est - e["end"]) <= end_tolerance(onset=e["onset"], end=e["end"]):
                        found = True
        rows.append(
            {
                "card_id": e["card_id"],
                "onset_found": onset_found,
                "found": found,
                "onset_error_h": min(errors) / 2 if errors else None,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "card_id": pl.String,
            "onset_found": pl.Boolean,
            "found": pl.Boolean,
            "onset_error_h": pl.Float64,
        },
    )


def injected_hits(
    *, table: pl.DataFrame, flagged: np.ndarray, events: pl.DataFrame
) -> pl.DataFrame:
    """For each injected event: found and onset found, from the flagged candidates that mark it."""
    marks = table.filter(pl.Series(flagged & (table["event_id"].to_numpy() >= 0)))
    by_event = {k: g for (k,), g in marks.group_by("event_id")}
    rows = []
    for e in events.iter_rows(named=True):
        g = by_event.get(e["event_id"])
        onset_found = g is not None
        found = bool(
            onset_found
            and any(
                abs(x - e["end"]) <= end_tolerance(onset=e["onset"], end=e["end"])
                for x in g["end_est"].to_list()
            )
        )
        rows.append({"event_id": e["event_id"], "onset_found": onset_found, "found": found})
    return pl.DataFrame(
        rows, schema={"event_id": pl.Int64, "onset_found": pl.Boolean, "found": pl.Boolean}
    )
