"""Print the numbers of experiment 290: bias of a 30-day, 25% transfer by fit-cut date and offset.

uv run --no-sync python e00290_report.py > printed_report.txt
"""

from __future__ import annotations

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
    print("share range over end dates (%):", summary["share_pct"].min(), summary["share_pct"].max())
    print("bias range over end dates:", summary["bias"].min(), summary["bias"].max())
    for window in ("days_3_to_10", "days_10_to_17", "days_3_to_33"):
        print(f"## {window}: share (%) by cut offset (rows) and end date (columns)")
        print(
            rows.filter(pl.col("window") == window)
            .group_by("end", "cut_offset_hours")
            .agg(share_pct=pl.col("share").mean() * 100)
            .pivot(on="end", index="cut_offset_hours", values="share_pct")
            .sort("cut_offset_hours")
            .with_columns(pl.exclude("cut_offset_hours").round(1))
        )
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
