"""The real record's events on the day grid: block days, D8 drops, real transfers, closures.

Each event is a dict with `cls` (block, drop, transfer), `sites` (labels), `start` and `end`
(inclusive grid days), and, for transfers, `moved_mw`. Sources: experiment 175 (block days),
experiment 159 (D8 drops), experiment 41 (the labelled real transfers over 1 day), and experiment
37 (logged closures).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00298_common as common
import e00298_detect as dt
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

GRID_START: Final[datetime] = datetime(2022, 1, 1, tzinfo=UTC)
FIRST_EVAL_DAY: Final[np.datetime64] = np.datetime64("2022-01-01", "D")


def _day(*, moment) -> int:  # noqa: ANN001
    """Grid day of a datetime or date."""
    stamp = np.datetime64(moment.replace(tzinfo=None) if hasattr(moment, "tzinfo") else moment, "D")
    return int((stamp - dt.DAY0).astype(int))


def stamp_day(*, stamp: int) -> int:
    """Grid day of a 30-minute stamp counted from 2022-01-01."""
    return _day(moment=GRID_START + timedelta(minutes=30 * int(stamp)))


def load_events() -> list[dict]:
    """Return every block day, D8 drop and real transfer as an event dict."""
    root = ML_EXPERIMENTS_DATA_DIR
    events: list[dict] = []
    blocks = pl.read_parquet(root / "00175-dp-unlabelled-day-shapes/work/day_shapes.parquet")
    for row in blocks.filter(pl.col("shape") == "a_block").iter_rows(named=True):
        d = _day(moment=row["local_date"])
        events.append({"cls": "block", "sites": [row["site"]], "start": d, "end": d, "moved_mw": np.nan})
    drops = pl.read_parquet(
        root / "00159-switching-d8-multi-day-drops-recurrence-and-report/drops.parquet"
    )
    for row in drops.iter_rows(named=True):
        events.append(
            {"cls": "drop", "sites": ["D8"], "start": _day(moment=row["start_date"]),
             "end": _day(moment=row["end_date"]), "moved_mw": row["depth_mw"]}  # fmt: skip
        )
    labelled = pl.read_parquet(root / "00041-sw2-0a-labeller-gate/real/labelled_set.parquet")
    labelled = labelled.filter(
        pl.col("kind") == "real_operation",
        (pl.col("end_utc") - pl.col("onset_utc")).dt.total_seconds() > 86400,
    )
    for row in labelled.iter_rows(named=True):
        sites = [common.SERIES_TO_SITE[int(s)] for s in (row["side_a"], row["side_b"]) if s is not None]
        events.append(
            {"cls": "transfer", "sites": sites, "start": _day(moment=row["onset_utc"]),
             "end": _day(moment=row["end_utc"]), "moved_mw": row["moved_mw_p50"]}  # fmt: skip
        )
    return events


def load_closures() -> list[dict]:
    """Return every logged closure as {sites, start, end} on the day grid."""
    path = ML_EXPERIMENTS_DATA_DIR / "00037-switching-first-look" / "intervals.parquet"
    out = []
    for row in pl.read_parquet(path).iter_rows(named=True):
        sites = [
            common.SERIES_TO_SITE[int(s)]
            for s in (row["side_a"], row["side_b"])
            if s is not None and str(s).isdigit() and int(s) in common.SERIES_TO_SITE
        ]
        out.append(
            {"sites": sites, "start": stamp_day(stamp=row["first_stamp"]),
             "end": stamp_day(stamp=row["last_stamp_exclusive"] - 1)}  # fmt: skip
        )
    return out


def month_of_day(*, day: int) -> int:
    """Calendar month (1 to 12) of a grid day."""
    return int(str(dt.DAY0 + np.timedelta64(day, "D"))[5:7])
