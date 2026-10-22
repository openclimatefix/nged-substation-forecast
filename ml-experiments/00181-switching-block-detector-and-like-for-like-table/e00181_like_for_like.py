"""Part 3: RO-NC on the XGBoost residual against RO-NC on the convex-model residual, 35 transfers.

Both detectors are experiment 82's RO-NC, scored by the same `match_event` rule (an alarm within 2
hours of the labelled start and an end within the larger of 6 hours and 10% of the duration) on the
same 35 labelled real transfers lasting over 1 day. Reads experiment 130's stored tables for arm A
(XGBoost) and arm B (convex model) and experiment 101's table for the readings present. Nothing is
fitted.

    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00181-*/e00181_like_for_like.py
"""

from __future__ import annotations

from typing import Final

import numpy as np
import polars as pl
from e00181_common import ML_EXPERIMENTS_DATA_DIR, WORK_DIR

RUN_130: Final = ML_EXPERIMENTS_DATA_DIR / "00130-ro-nc-detector-on-the-gam-k-residual"
EVENTS_101: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00101-ro-nc-detector-on-real-hand-filtered-switching-events"
    / "event_table.parquet"
)
N_RESAMPLES: Final[int] = 20_000
SEED: Final[int] = 181


def load() -> pl.DataFrame:
    """One row per transfer: found by each arm, moved MW, and the XGBoost panel's coverage."""

    def arm(name: str) -> pl.DataFrame:
        return (
            pl.read_parquet(RUN_130 / name / "real_results.parquet")
            .filter(pl.col("setting") == "XL", pl.col("hours") > 24)
            .select("card_id", "series", "hours", "moved_mw_p50", "found")
            .sort("card_id")
        )

    a, b = arm("A15"), arm("B")
    if not (a["card_id"] == b["card_id"]).all() or a.height != 35:
        raise ValueError("the two arms do not hold the same 35 transfers")
    coverage = pl.read_parquet(EVENTS_101).select("card_id", "power_present_share")
    return (
        a.rename({"found": "found_xgboost"})
        .join(b.select("card_id", found_convex="found"), on="card_id")
        .join(coverage, on="card_id")
        .with_columns(xgboost_has_readings=pl.col("power_present_share") >= 0.5)
    )


def summarise(*, table: pl.DataFrame, label: str) -> None:
    """Print found counts, MW-weighted recall, and the paired MW-weighted gap with its interval."""
    weight = table["moved_mw_p50"].to_numpy()
    fa, fb = table["found_xgboost"].to_numpy(), table["found_convex"].to_numpy()
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(weight), size=(N_RESAMPLES, len(weight)))
    gap = ((weight[picks] * fb[picks]).sum(1) - (weight[picks] * fa[picks]).sum(1)) / weight[
        picks
    ].sum(1)
    lo, hi = np.percentile(gap, [2.5, 97.5])
    print(
        f"{label}: {len(weight)} transfers, {weight.sum():.1f} MW. "
        f"XGBoost found {fa.sum()} (MW-weighted {np.sum(weight * fa) / weight.sum():.3f}); "
        f"convex found {fb.sum()} ({np.sum(weight * fb) / weight.sum():.3f}). "
        f"Paired MW-weighted gap {np.mean(gap):+.3f} (95% {lo:+.3f} to {hi:+.3f}). "
        f"Both {int((fa & fb).sum())}, only XGBoost {int((fa & ~fb).sum())}, "
        f"only convex {int((~fa & fb).sum())}, neither {int((~fa & ~fb).sum())}."
    )


def main() -> None:
    """Print the three rows of the like-for-like table and write the per-transfer table."""
    table = load()
    table.write_parquet(WORK_DIR / "like_for_like.parquet")
    summarise(table=table, label="All 35")
    summarise(table=table.filter(pl.col("xgboost_has_readings")), label="Readings in both panels")
    summarise(
        table=table.filter(~pl.col("xgboost_has_readings")), label="No readings in XGBoost panel"
    )


if __name__ == "__main__":
    main()
