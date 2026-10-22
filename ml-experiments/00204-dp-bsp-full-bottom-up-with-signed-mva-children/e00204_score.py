"""Load, align, and score the arms of experiment 204.

A row is scored when the BSP and every child that enters any arm are outside the hand-filtered
switching events (the frames' union mask), the reading is finite, and the lead is at most 168
hours. Every arm is restricted to the rows all arms hold, so the arms are scored on the same rows.
"""

from typing import Final

import e00204_common as common
import numpy as np
import polars as pl
from studies.experiment_store import losses_for_bootstrap

KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
TAIL_QUANTILES: Final[tuple[tuple[str, float], ...]] = (
    ("q01", 0.01), ("q05", 0.05), ("q95", 0.95), ("q99", 0.99),
)  # fmt: skip
ARMS: Final[tuple[str, ...]] = (
    "a_td",
    "b_partial",
    "b_partial_tdspread",
    "c",
    "d",
    "dp50",
    "oracle",
)
"""(a) top-down; (b) experiment 161's partial bottom-up with its own table and with the top-down
table; (c) full naive; (d) full fold-aware; (d-p50) the same at probability 0.5; the oracle."""


def arm_path(*, arm: str, bsp: str, period: str):  # noqa: ANN201
    """Return the file of an arm at a BSP for a period."""
    if arm == "a_td":
        return common.E161_OUT_DIR / f"d_{period}_{bsp}.parquet"
    if arm == "b_partial":
        return common.E161_OUT_DIR / f"bu_r_{period}_{bsp}.parquet"
    if arm == "b_partial_tdspread":
        return common.E161_OUT_DIR / f"bu_r_tdspread_{period}_{bsp}.parquet"
    return common.OUT_DIR / f"bu_{arm}_{period}_{bsp}.parquet"


def read_arm(*, arm: str, bsp: str, period: str) -> pl.DataFrame:
    """Return one arm's frame at a BSP for a period."""
    return pl.read_parquet(arm_path(arm=arm, bsp=bsp, period=period))


def scorable(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Keep rows outside the switching events with a finite reading and a lead up to 168 hours."""
    return frame.filter(
        ~pl.col("switching_mask")
        & pl.col("actual").is_finite()
        & (pl.col("lead_hours") <= common.HORIZON_HOURS)
    )


def common_rows(*, frames: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Restrict every frame to the rows all frames hold."""
    shared = None
    for frame in frames.values():
        current = frame.select(KEYS).unique()
        shared = current if shared is None else shared.join(current, on=KEYS, how="inner")
    assert shared is not None
    return {
        name: frame.join(shared, on=KEYS, how="inner").sort(KEYS) for name, frame in frames.items()
    }


def with_losses(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the quantile score, the absolute error of the mean and median, and the tail losses."""
    actual = frame["actual"].to_numpy().astype(np.float64)
    quantiles = frame.select(list(common.QUANTILES)).to_numpy().astype(np.float64)
    error = actual[:, None] - quantiles
    levels = common.LEVELS[None, :]
    score = 2.0 * np.mean(np.maximum(levels * error, (levels - 1.0) * error), axis=1)
    extra = {
        "qs": score,
        "abs_mean": np.abs(actual - frame["mean"].to_numpy().astype(np.float64)),
        "abs_median": np.abs(actual - frame["q50"].to_numpy().astype(np.float64)),
    }
    for name, level in TAIL_QUANTILES:
        value = frame[name].to_numpy().astype(np.float64)
        diff = actual - value
        extra[f"pin_{name}"] = np.maximum(level * diff, (level - 1.0) * diff)
        extra[f"cov_{name}"] = (actual <= value).astype(np.float64)
    return frame.with_columns([pl.Series(k, v) for k, v in extra.items()])


def loss_frame(*, bsp: str, period: str, arms: tuple[str, ...] = ARMS) -> pl.DataFrame:
    """Return the bootstrap loss frame of the arms on their shared rows at a BSP."""
    frames = common_rows(
        frames={arm: scorable(frame=read_arm(arm=arm, bsp=bsp, period=period)) for arm in arms}
    )
    keep = [*KEYS, "seed", "lead_hours", *common.QUANTILES, "actual", "mean", "hour_band"]
    stacked = losses_for_bootstrap(
        arm_losses={name: with_losses(frame=f.select(keep)) for name, f in frames.items()}
    )
    return stacked.with_columns(site=pl.lit(bsp))
