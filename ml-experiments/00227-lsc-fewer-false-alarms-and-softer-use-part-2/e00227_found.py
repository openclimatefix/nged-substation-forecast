# ruff: noqa: E501, PERF401
"""Alarms per matched logged load shift, for every source and threshold of part (a).

A logged load shift lasting over 1 day is matched when an alarm on one of its series has its step
within 2 hours of the logged onset and its estimated end within the larger of 6 hours and 10% of
the duration of the logged end (experiment 195's rule). A learned source is scored for each load
shift with the fold that holds out the load shift's Primary pair. Alarms are the rows of the
interval table (alarms on one series that start within 24 hours merge) over all 16 series and the
whole real record; a learned source's count is the mean over its three folds, because each fold's
model raises alarms on the whole record.

    OMP_NUM_THREADS=1 uv run --no-sync python e00227_found.py
"""

from __future__ import annotations

from typing import Final

import e00227_alarms as al
import numpy as np
import polars as pl
from e00227_common import WORK_DIR
from e00227_s_common import pair_folds
from e00227_s_run import end_tolerance

ONSET_TOLERANCE: Final[int] = 4
REAL_RESULTS: Final = (
    WORK_DIR.parent / "00130-ro-nc-detector-on-the-gam-k-residual" / "B" / "real_results.parquet"
)


def labelled() -> pl.DataFrame:
    """The 35 labelled real load shifts lasting over 1 day: card, onset, end, hours, pair, series."""
    return (
        pl.read_parquet(REAL_RESULTS)
        .filter((pl.col("setting") == "XL") & (pl.col("hours") > 24))
        .select("card_id", "onset", "end", "hours", pair=pl.col("series"))
        .with_columns(series=pl.col("pair").str.split(",").list.eval(pl.element().cast(pl.Int64)))
    )


def matched(*, table: pl.DataFrame, event: dict) -> bool:
    """Whether `table` holds an alarm that matches the load shift."""
    hits = table.filter(pl.col("series").is_in(event["series"]))
    for onset, end in zip(hits["onset"].to_list(), hits["end"].to_list(), strict=True):
        if abs(onset - event["onset"]) <= ONSET_TOLERANCE and abs(
            end - event["end"]
        ) <= end_tolerance(onset=event["onset"], end=event["end"]):
            return True
    return False


def per_source() -> pl.DataFrame:
    """One row per (source, load shift): matched, with the source's mean alarm count."""
    folds = pair_folds()
    events = labelled()
    rows = []
    for source in al.source_ids():
        by_fold = al.tables(source=source)
        alarms = float(np.mean([t.height for t in by_fold.values()]))
        for e in events.iter_rows(named=True):
            rows.append(
                {
                    "source": source,
                    "card": e["card_id"],
                    "matched": matched(table=by_fold[folds[e["pair"]]], event=e),
                    "alarms": alarms,
                }
            )
    return pl.DataFrame(rows)


def main() -> None:
    """Print and store the table."""
    pl.Config.set_tbl_rows(100)
    out = per_source()
    out.write_parquet(WORK_DIR / "found35.parquet")
    summary = (
        out.group_by("source", maintain_order=True)
        .agg(matched=pl.col("matched").sum(), alarms=pl.col("alarms").first())
        .with_columns(alarms_per_match=pl.col("alarms") / pl.col("matched"))
    )
    print(summary)


if __name__ == "__main__":
    main()
