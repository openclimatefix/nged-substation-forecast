"""The 17 sharp-edged unlabelled worst days of experiment 175, placed on the RO-NC residual panel."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Final

import numpy as np
import polars as pl
from e00181_common import GRID_LENGTH, GRID_START, ML_EXPERIMENTS_DATA_DIR

E174_DAYS: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00174-dp-unlabelled-worst-days-weather-shared-or-single-site"
    / "work"
    / "worst_days_split.parquet"
)
E175_SHAPES: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00175-dp-unlabelled-day-shapes" / "work" / "day_shapes.parquet"
)
BG2_PREDICTIONS: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00087-bg2-best-guess-assembled-with-drop-one-ablations"
    / "predictions"
    / "p_s6_bg2"
    / "predictions.parquet"
)
E175_GRID_FIRST: Final = datetime(2025, 6, 1, tzinfo=UTC)
OFFSET_STAMPS: Final[int] = int((E175_GRID_FIRST - GRID_START).total_seconds() // 1800)
"""Experiment 175's grid position 0 is this detector stamp."""


def site_series() -> dict[str, int]:
    """Each D-label mapped to its `time_series_id`, the detectors' series number."""
    pairs = pl.read_parquet(BG2_PREDICTIONS, columns=["site", "time_series_id"]).unique()
    return {r["site"]: int(r["time_series_id"]) for r in pairs.iter_rows(named=True)}


def block_days() -> pl.DataFrame:
    """The 17 unlabelled worst days that experiment 175 classed as sharp-edged blocks."""
    unlabelled = pl.read_parquet(E174_DAYS).filter(~pl.col("labelled")).select("site", "local_date")
    days = (
        pl.read_parquet(E175_SHAPES)
        .join(unlabelled, on=["site", "local_date"])
        .filter(pl.col("shape") == "a_block")
        .sort("site", "local_date")
    )
    series = site_series()
    days = days.with_columns(
        series=pl.col("site").replace_strict(series, return_dtype=pl.Int64),
        onset_stamp=pl.col("edge_position") + OFFSET_STAMPS,
    )
    if days.height != 17:
        raise ValueError(f"expected 17 block days, found {days.height}")
    return days


def stamp_time(*, stamp: int) -> datetime:
    """UTC start of the half-hour at a detector stamp."""
    return datetime.fromtimestamp(GRID_START.timestamp() + 1800 * stamp, tz=UTC)


if __name__ == "__main__":
    pl.Config.set_tbl_cols(20)
    print(block_days().select("site", "series", "local_date", "onset_stamp", "edge_shift"))
    print(GRID_LENGTH)
