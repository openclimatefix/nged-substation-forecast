"""Experiment 00341, step 5: the private half-hour capacity-state table.

Expands the capacity-state segments of arm D (14-day blocks) to half-hours and marks the half-hours
inside experiment 210's idle runs as `out`. Writes `work/capacity_state.parquet` with the columns
`site`, `valid_time`, `state` (`full`, `reduced`, `out`), and `reduced_share`. The table is private
and a later experiment tests it as a training mask and as an as-of feature.

    uv run --no-sync python ml-experiments/00341-*/e00341_state_table.py
"""

from datetime import timedelta

import e00341_common as common
import polars as pl


def main() -> None:
    """Write the table and print its row counts."""
    segments = pl.read_parquet(common.WORK_DIR / "capacity_state_segments.parquet")
    outage = pl.read_parquet(common.E210_DIR / "outage_runs.parquet")
    table = pl.read_parquet(common.WORK_DIR / "half_hours.parquet").select("site", "valid_time")
    day = (
        (pl.col("valid_time") - pl.lit(common.ERA5_START)).dt.total_minutes() // (48 * 30)
    ).alias("day")
    table = table.with_columns(day)
    frames = []
    for site in common.SITES:
        seg = segments.filter(pl.col("site") == site)
        part = table.filter(pl.col("site") == site)
        joined = part.join_asof(
            seg.sort("first_day"), left_on="day", right_on="first_day", by="site"
        )
        joined = joined.with_columns(
            state=pl.when(pl.col("day") <= pl.col("last_day"))
            .then(pl.col("state"))
            .otherwise(pl.lit("full")),
            reduced_share=pl.when(pl.col("day") <= pl.col("last_day"))
            .then(pl.col("reduced_share"))
            .otherwise(0.0),
        )
        runs = outage.filter(pl.col("site") == site)
        out_mask = pl.lit(False)
        for r in runs.iter_rows(named=True):
            out_mask = out_mask | pl.col("valid_time").is_between(
                r["start"], r["end"] + timedelta(minutes=0)
            )
        joined = joined.with_columns(
            state=pl.when(out_mask).then(pl.lit("out")).otherwise(pl.col("state"))
        )
        frames.append(joined.select("site", "valid_time", "state", "reduced_share"))
    result = pl.concat(frames)
    result.write_parquet(common.WORK_DIR / "capacity_state.parquet")
    print(result.group_by("site", "state").len().sort("site", "state"))


if __name__ == "__main__":
    main()
