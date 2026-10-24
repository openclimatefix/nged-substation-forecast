# ruff: noqa: E501
"""Recalibration table (arm 1): offsets per farm, lead day, and quantile level, fitted rolling-origin.

For each scored month (the month of the run), the offset of a cell is the empirical quantile, at the
cell's nominal level, of (actual - forecast) over rows whose valid time precedes the month's first
run. A cell with fewer than `MIN_CELL_ROWS` earlier rows gets no offset. The table is applied to the
sorted forecast, clipped at zero, and re-sorted. Variants: `shrunk` multiplies the offsets by
k / (k + SHRINK_MONTHS) for k earlier months, `stale7` fits on rows at least 7 days older, and
`swapped` applies the next farm's table (a placebo for the farm-specific part).
"""

from datetime import timedelta
from typing import Final

import e00310_common as c310  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.tail_calibration import level_column

QCOLS: Final[list[str]] = [level_column(level=lv) for lv in c310.LEVELS]
KEYS: Final[list[str]] = ["time_series_id", "time", "init_time"]
MIN_CELL_ROWS: Final[int] = 30
SHRINK_MONTHS: Final[int] = 3
STALE_DAYS: Final[int] = 7
VARIANTS: Final[tuple[str, ...]] = ("recal", "recal_shrunk", "recal_stale7", "recal_swapped")


def load_stored(*, band: str, setting: str, arm: str) -> pl.DataFrame:
    """Return experiment 310's stored per-farm arm (`blend`, `mdn_ens`) with float columns."""
    files = sorted((c310.PRED_DIR / band / setting).glob(f"{arm}_s0_?.parquet"))
    frame = pl.concat(
        [pl.read_parquet(f).select(*KEYS, "lead_hours", "actual", *QCOLS) for f in files]
    )
    return frame.with_columns(
        pl.col(QCOLS).cast(pl.Float64),
        pl.col("actual").cast(pl.Float64),
        pl.col("lead_hours").cast(pl.Float64),
    )


def add_day(*, frame: pl.DataFrame, lead_origin_hours: float = 12.0) -> pl.DataFrame:
    """Add the lead day of each row; day 1 starts `lead_origin_hours` after the run."""
    return frame.with_columns(
        day=((pl.col("lead_hours") - lead_origin_hours) / 24.0).floor().cast(pl.Int64) + 1,
        month=pl.col("init_time").dt.strftime("%Y-%m"),
    )


def cell_offsets(*, history: pl.DataFrame) -> dict[tuple[str, int], np.ndarray]:
    """Return, per (farm, lead day), the offset at each of the 11 levels, from the history rows."""
    out: dict[tuple[str, int], np.ndarray] = {}
    for (farm, day), part in history.group_by("time_series_id", "day"):
        if part.height < MIN_CELL_ROWS:
            continue
        actual = part["actual"].to_numpy()
        q = np.sort(part.select(QCOLS).to_numpy(), axis=1)
        out[(farm, day)] = np.array(
            [np.quantile(actual - q[:, k], level) for k, level in enumerate(c310.LEVELS)]
        )
    return out


def recalibrate(
    *, frame: pl.DataFrame, variant: str, lead_origin_hours: float = 12.0
) -> pl.DataFrame:
    """Return the frame (sorted by key) with the variant's rolling-origin recalibrated quantiles."""
    frame = add_day(frame=frame, lead_origin_hours=lead_origin_hours).sort(KEYS)
    farms = sorted(frame["time_series_id"].unique().to_list())
    q_all = np.sort(frame.select(QCOLS).to_numpy(), axis=1)
    out = q_all.copy()
    months = sorted(frame["month"].unique().to_list())
    time_col = frame["time"].dt.epoch("us").to_numpy()
    init_col = frame["init_time"].dt.epoch("us").to_numpy()
    month_col = frame["month"].to_numpy()
    farm_col = frame["time_series_id"].to_numpy()
    day_col = frame["day"].to_numpy()
    stale_us = (
        int(timedelta(days=STALE_DAYS).total_seconds() * 1e6) if variant == "recal_stale7" else 0
    )
    for month in months:
        in_month = month_col == month
        cutoff = init_col[in_month].min() - stale_us
        history = frame.filter(pl.Series(time_col < cutoff))
        if history.is_empty():
            continue
        k_months = history["month"].n_unique()
        weight = k_months / (k_months + SHRINK_MONTHS) if variant == "recal_shrunk" else 1.0
        table = cell_offsets(history=history)
        for farm in farms:
            source = (
                farms[(farms.index(farm) + 1) % len(farms)] if variant == "recal_swapped" else farm
            )
            for day in np.unique(day_col[in_month & (farm_col == farm)]):
                offset = table.get((source, int(day)))
                if offset is None:
                    continue
                rows = in_month & (farm_col == farm) & (day_col == day)
                out[rows] = q_all[rows] + weight * offset
    out = np.sort(np.maximum(out, 0.0), axis=1)
    return frame.with_columns(
        *[pl.Series(c, out[:, k]) for k, c in enumerate(QCOLS)],
    )
