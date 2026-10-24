"""A month-by-half-hour climatology reference for experiment 00288.

For each farm, the 11 quantiles of the farm's output over capacity at every (UTC month, UTC
half-hour of the day) cell of the training window (valid times before 2024-04-01). It uses no
weather, so every lead gets the same quantiles. The reference row of `result.md` and the
experiment's stop rule use it.
"""

from datetime import UTC, datetime
from typing import Final

import e00288_common as common
import numpy as np
import polars as pl

TRAIN_END: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)


def climatology_table(*, training: pl.DataFrame) -> pl.DataFrame:
    """Return the quantiles per farm, month, and half-hour from `farm`, `valid_time`, `target`."""
    return (
        training.filter(pl.col("valid_time") < TRAIN_END, pl.col("target").is_finite())
        .with_columns(
            month=pl.col("valid_time").dt.month(),
            slot=pl.col("valid_time").dt.hour() * 2 + pl.col("valid_time").dt.minute() // 30,
        )
        .group_by("farm", "month", "slot")
        .agg(
            [
                pl.col("target").quantile(float(level), interpolation="linear").alias(name)
                for name, level in zip(common.LEVEL_COLUMNS, common.LEVELS, strict=True)
            ]
        )
    )


def climatology_quantiles(*, cases: pl.DataFrame, table: pl.DataFrame) -> np.ndarray:
    """Return shape (cases, 11) climatology quantiles for the cases' farm and valid time."""
    keyed = cases.select(
        "farm",
        month=pl.col("valid_time").dt.month(),
        slot=pl.col("valid_time").dt.hour() * 2 + pl.col("valid_time").dt.minute() // 30,
    ).join(table, on=["farm", "month", "slot"], how="left", maintain_order="left")
    return np.sort(keyed.select(list(common.LEVEL_COLUMNS)).to_numpy(), axis=1)
