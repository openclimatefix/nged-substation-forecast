"""Median q05-q95 width and q95 coverage by lead day for arm A2, per substation (descriptive).

Scored year, rows outside switching events, lead of at most 168 hours. Lead day `d` holds leads
above `24 (d - 1)` and at most `24 d` hours.

    uv run python ml-experiments/00136-*/e00136_width.py
"""

import json

import e00136_common as common
import e00136_report as rep
import numpy as np
import polars as pl


def main() -> None:
    """Print and store the width and coverage table."""
    frame = rep.load(arm="a2", period="scored")
    assert frame is not None
    frame = frame.with_columns(
        lead_day=(pl.col("lead_hours") / 24.0).ceil().cast(pl.Int32),
        width=pl.col("q95") - pl.col("q05"),
        covered=(pl.col("actual") <= pl.col("q95")).cast(pl.Float64),
    )
    table = (
        frame.group_by(["site", "lead_day"])
        .agg(
            pl.col("width").median().alias("median_width"),
            pl.col("covered").mean().alias("q95_coverage"),
            pl.len().alias("rows"),
        )
        .sort(["site", "lead_day"])
    )
    for site in common.SCORED_SITES:
        part = table.filter(pl.col("site") == site)
        widths = " ".join(f"{w:5.1f}" for w in part["median_width"].to_numpy())
        coverage = " ".join(f"{c:5.3f}" for c in part["q95_coverage"].to_numpy())
        print(f"{site:4s} width {widths} | q95 coverage {coverage}")
    common.write_json_atomic(
        path=common.WORK_DIR / "report" / "width_by_lead_day.json",
        content={"rows": table.to_dicts()},
    )


if __name__ == "__main__":
    main()
