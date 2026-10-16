"""Paths, series, and logged intervals shared by the scripts of experiment 00041."""

import os
from pathlib import Path
from typing import Final

import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00041-sw2-0a-labeller-gate"
"""Private tables, cards, and labels; never committed."""
FEATURES_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00006-stage1a-demand-input-feature-sweep"
    / "inputs"
    / "features.parquet"
)
LOGGED_INTERVALS_PATH: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "sw1b_scratch" / "intervals.parquet"
"""Every closure reconstructed from the confirmed logs, local times, any duration."""
TEST_SET, BIG_SET = os.environ.get("E41_SETS", "test2,testbig2").split(",")
"""The gate's test cards and the extra large-transfer cards (`E41_SETS=test,testbig`: first set)."""
SERIES_IDS: Final[tuple[int, ...]] = (3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 25, 26)
MVA_SERIES: Final[frozenset[int]] = frozenset({3, 9, 11, 14, 15, 24, 25, 26})
"""Non-directional apparent-power series."""
EARLIEST_LOG: Final[str] = "2019-01-01"
"""Logged times before this date are typing errors (the sheet holds a 2008 row)."""


def logged_intervals() -> pl.DataFrame:
    """Return every closed logged interval with UTC `start_utc`, `end_utc`, and sheet sides."""
    return (
        pl.read_parquet(LOGGED_INTERVALS_PATH)
        .filter(
            pl.col("end").is_not_null(), pl.col("start") >= pl.lit(EARLIEST_LOG).str.to_datetime()
        )
        .with_columns(
            start_utc=pl.col("start")
            .dt.replace_time_zone("Europe/London", ambiguous="earliest", non_existent="null")
            .dt.convert_time_zone("UTC"),
            end_utc=pl.col("end")
            .dt.replace_time_zone("Europe/London", ambiguous="earliest", non_existent="null")
            .dt.convert_time_zone("UTC"),
        )
        .drop_nulls(["start_utc", "end_utc"])
        .filter(pl.col("end_utc") > pl.col("start_utc"))
    )
