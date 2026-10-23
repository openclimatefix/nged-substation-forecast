"""Interval tables for the masked-forecast harness, from this experiment's flagged alarms.

An arm's alarms are the real candidates above the arm's threshold (written by `e00302_score.py`),
known 192 half-hours after the step (experiment 227's learned-score lag), merged within 24 hours.
"""

from __future__ import annotations

from typing import Final

import e00302_h_events as ev
import polars as pl
from e00302_common import E195_DIR, WORK_DIR

N_FOLDS: Final[int] = 3
LAG_R: Final[int] = 192
END_LAG_UNMATCHED: Final[int] = 96
ARMS: Final[tuple[str, ...]] = ("L0", "L1", "L2", "L3", "L3_5", "L3_2")


def _live() -> pl.DataFrame:
    return (
        pl.read_parquet(ev.INTERVALS_RONC)
        .filter(pl.col("setting") == "R0_M48")
        .group_by("series", "onset")
        .agg(live_known=pl.col("known_from").min())
        .rename({"onset": "stamp"})
    )


def _interval_table(*, alarms: pl.DataFrame) -> pl.DataFrame:
    table = alarms.select(
        "series",
        pl.col("series").alias("alarm_series"),
        pl.lit("alarm").alias("role"),
        pl.lit("R0_M48").alias("setting"),
        onset=pl.col("stamp"),
        end=pl.col("end_est"),
        step_mw=pl.col("sign") * pl.col("step_mw24"),
        onset_known=pl.col("stamp") + LAG_R,
        known_from=pl.max_horizontal(
            pl.col("live_known").fill_null(pl.col("end_est") + END_LAG_UNMATCHED), pl.col("end_est")
        ),
    ).drop_nulls("step_mw")
    return ev.union_intervals(ronc=table, pair=table.clear())


def tables(*, arm: str) -> dict[int, pl.DataFrame]:
    """Each fold's interval table for one arm."""
    cands = pl.read_parquet(E195_DIR / "real_candidates.parquet").select(
        "series", "stamp", "source", "sign", "step_mw24"
    )
    live = _live()
    out = {}
    for fold in range(N_FOLDS):
        alarms = (
            pl.read_parquet(WORK_DIR / f"alarms_{arm}_f{fold}.parquet")
            .join(cands, on=["series", "stamp", "source"], how="left", validate="1:1")
            .join(live, on=["series", "stamp"], how="left")
        )
        out[fold] = _interval_table(alarms=alarms)
    return out
