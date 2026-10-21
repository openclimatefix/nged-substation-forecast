"""Load, align, and score the arms of experiment 161.

A row is scored when the BSP and its three covered children are outside the hand-filtered switching
events (the frames' union mask), the reading is finite, and the lead is at most 168 hours. Every
arm is restricted to the rows all arms hold, so the arms are scored on the same rows.
"""

from typing import Final

import e00161_common as common
import numpy as np
import polars as pl
from studies.experiment_store import losses_for_bootstrap

KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
TAIL_QUANTILES: Final[tuple[tuple[str, float], ...]] = (
    ("q01", 0.01), ("q05", 0.05), ("q95", 0.95), ("q99", 0.99),
)  # fmt: skip
ARMS: Final[tuple[str, ...]] = ("td", "bu_r", "bu_s", "bu_f", "bu_r_tdspread", "bu_oracle")
FILES: Final[dict[str, str]] = {
    "td": "td",
    "bu_r": "bu_r",
    "bu_s": "bu_s",
    "bu_f": "bu_f",
    "bu_r_tdspread": "bu_r_tdspread",
    "bu_oracle": "bu_oracle",
}


def read_arm(*, arm: str, bsp: str, period: str) -> pl.DataFrame:
    """Return one arm's frame at a BSP for a period."""
    return pl.read_parquet(common.OUT_DIR / f"{FILES[arm]}_{period}_{bsp}.parquet")


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
