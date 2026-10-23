"""Print the numbers of experiment 290: bias of a 30-day, 25% transfer by fit-cut date and offset.

uv run --no-sync python e00290_report.py > printed_report.txt
"""

from __future__ import annotations

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROWS = ML_EXPERIMENTS_DATA_DIR / "00290-lsc-long-shift-cut-date-robustness" / "rows"


def main() -> None:
    """Print bias and learned share by end date, cut offset, and window."""
    rows = pl.read_parquet(ROWS / "*.parquet").with_columns(
        share=pl.col("mean_bias") / pl.col("mean_transfer")
    )
    print(f"jobs: {rows.select('site', 'end').unique().height}")
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    summary = (
        rows.filter((pl.col("cut_offset_hours") == 0) & (pl.col("window") == "after_48h"))
        .group_by("end")
        .agg(
            bias=pl.col("mean_bias").mean(),
            transfer=pl.col("mean_transfer").mean(),
            share_pct=pl.col("share").mean() * 100,
            min_bias=pl.col("mean_bias").min(),
            max_bias=pl.col("mean_bias").max(),
        )
        .sort("end")
    )
    print("## 48 h after the end, fit cut at the end, by end date (mean over 10 substations)")
    print(summary)
    shares = np.asarray(summary["share_pct"].to_list(), dtype=float)
    half = 2.201 * shares.std(ddof=1) / len(shares) ** 0.5  # t multiplier, 11 df
    print(f"mean share over end dates (%): {shares.mean():.2f}, sd {shares.std(ddof=1):.2f}")
    print(f"95% t interval (%): {shares.mean() - half:.2f} to {shares.mean() + half:.2f}")
    print(f"mean bias over end dates: {summary['bias'].mean():.2f}")
    print(f"mean transfer over end dates: {summary['transfer'].mean():.2f}")
    print("share range over end dates (%):", shares.min(), shares.max())
    print("bias range over end dates:", summary["bias"].min(), summary["bias"].max())
    for window in ("days_3_to_10", "days_10_to_17", "days_3_to_33"):
        print(f"## {window}: share (%) by cut offset (rows) and end date (columns)")
        print(
            rows.filter(pl.col("window") == window)
            .group_by("end", "cut_offset_hours")
            .agg(share_pct=pl.col("share").mean() * 100)
            .sort("end", "cut_offset_hours")
            .pivot(on="end", index="cut_offset_hours", values="share_pct")
            .sort("cut_offset_hours")
            .with_columns(pl.exclude("cut_offset_hours").round(1))
        )
    jobs = (
        rows.filter(pl.col("window") == "days_10_to_17")
        .pivot(on="cut_offset_hours", index=["site", "end"], values="mean_bias")
        .with_columns(change_12h=(pl.col("12") - pl.col("0")).abs())
    )
    print(f"## largest change in bias from a 12-hour later cut, over {jobs.height} jobs (points)")
    print(jobs["change_12h"].max())
    print("## mean share (%) over end dates, by cut offset and window")
    print(
        rows.filter(pl.col("window") != "after_48h")
        .group_by("cut_offset_hours", "window")
        .agg(share_pct=pl.col("share").mean() * 100, bias=pl.col("mean_bias").mean())
        .sort("window", "cut_offset_hours")
    )
    print("## training rows (mean) by cut offset: does the cut change the training set?")
    print(
        rows.filter(pl.col("window") == "days_3_to_10")
        .group_by("cut_offset_hours")
        .agg(n=pl.col("n_train_rows").mean())
        .sort("cut_offset_hours")
    )
    print("## by substation at cut 0 (48 h after), mean over end dates")
    print(
        rows.filter((pl.col("cut_offset_hours") == 0) & (pl.col("window") == "after_48h"))
        .group_by("site")
        .agg(bias=pl.col("mean_bias").mean(), share_pct=pl.col("share").mean() * 100)
        .sort("site")
    )


if __name__ == "__main__":
    main()
