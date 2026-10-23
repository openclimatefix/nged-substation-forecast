"""Draw the figure of experiment 290 as SVG beside the write-up.

OMP_NUM_THREADS=1 uv run --no-sync python e00290_charts.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.charts import CONTENT_WIDTH_PX
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROWS: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00290-lsc-long-shift-cut-date-robustness" / "rows"
OUT: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00290-lsc-long-shift-cut-date-robustness"
)


def main() -> None:
    """Plot the share of a 30-day transfer learned as normal, by end date, with the 274 value."""
    rows = (
        pl.read_parquet(ROWS / "*.parquet")
        .filter((pl.col("cut_offset_hours") == 0) & (pl.col("window") == "after_48h"))
        .with_columns(share=pl.col("mean_bias") / pl.col("mean_transfer") * 100)
        .group_by("end")
        .agg(
            mean=pl.col("share").mean(),
            low=pl.col("share").min(),
            high=pl.col("share").max(),
        )
        .sort("end")
        .with_columns(date=pl.col("end").dt.strftime("%Y-%m-%d"))
        .to_pandas()
    )
    x = alt.X("date:O", title="End of the transfer (fit cut)", axis=alt.Axis(labelAngle=-45))
    bars = (
        alt.Chart(rows)
        .mark_rule(strokeWidth=2, color=ocf.DATA_SKY)
        .encode(  # ty: ignore[unresolved-attribute]
            x=x, y=alt.Y("low:Q", title="Share learned as normal (%)"), y2="high:Q"
        )
    )
    points = (
        alt.Chart(rows)
        .mark_point(filled=True, size=80, color=ocf.DATA_BLUE)
        .encode(x=x, y="mean:Q")  # ty: ignore[unresolved-attribute]
    )
    reference = (
        alt.Chart(alt.Data(values=[{"y": 19.1}]))
        .mark_rule(strokeDash=[4, 3], color=ocf.BRAND_ORANGE)
        .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    chart = (bars + points + reference).properties(width=CONTENT_WIDTH_PX - 120, height=260)
    chart.configure_view(stroke=None).save(OUT / "share_by_end_date.svg")


if __name__ == "__main__":
    main()
