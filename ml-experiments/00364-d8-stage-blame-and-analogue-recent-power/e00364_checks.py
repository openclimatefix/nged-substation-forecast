# ruff: noqa: E501
"""Print tail calibration and the median-minus-mean check for each D8 arm, from `work/wide_D8_weekly.parquet`."""

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG = "00364-d8-stage-blame-and-analogue-recent-power"
LEVELS = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}


def main() -> None:
    """Print, per arm: q05 to q95 coverage, shares below q01 and above q99, pinball at four tails."""
    wide = pl.read_parquet(ML_EXPERIMENTS_DATA_DIR / SLUG / "work" / "wide_D8_weekly.parquet")
    actual = wide["actual"].to_numpy().astype(np.float64)
    arms = sorted({c.split("__")[0] for c in wide.columns if c.endswith("__q50")})
    print("arm | q05-q95 coverage | below q01 | above q99 | pinball q01, q05, q95, q99 (points)")
    for arm in arms:
        q = {k: wide[f"{arm}__{k}"].to_numpy().astype(np.float64) for k in (*LEVELS, "q50")}
        cover = float(((actual >= q["q05"]) & (actual <= q["q95"])).mean())
        pin = [
            float(np.mean(np.maximum(tau * (actual - q[k]), (tau - 1) * (actual - q[k]))))
            for k, tau in LEVELS.items()
        ]
        print(
            f"{arm:12s} | {cover:.3f} | {(actual < q['q01']).mean():.3f} | {(actual > q['q99']).mean():.3f} | "
            + ", ".join(f"{p:.3f}" for p in pin)
        )
    print("\nmean of (mean forecast column minus q50), points, for the stages that carry a mean")
    for arm in arms:
        if f"{arm}__mean" in wide.columns:
            gap = (wide[f"{arm}__mean"] - wide[f"{arm}__q50"]).to_numpy()
            print(f"{arm:12s} {np.nanmean(gap):.4f} max abs {np.nanmax(np.abs(gap)):.4f}")


if __name__ == "__main__":
    main()
