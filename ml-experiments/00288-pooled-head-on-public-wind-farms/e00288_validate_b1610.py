"""Validate the B1610 download of experiment 00288 per BMU (the `data-validation` checklist).

Checks per BMU: first and last half-hour, rows against the rows a gap-free series would hold between
them, duplicate keys, nulls and NaNs, negative readings, the largest reading against the BMU's
register capacity, the share of exact zeros, the longest run of one repeated non-zero value, and a
month-by-month mean for a step. Writes `work/b1610_validation.parquet` and prints a summary.

    uv run --no-sync python ml-experiments/00288-*/e00288_validate_b1610.py
"""

from typing import Final

import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

WORK_DIR: Final = ML_EXPERIMENTS_DATA_DIR / "00288-pooled-head-on-public-wind-farms" / "work"
B1610: Final = (
    REPO_DATA_DIR / "studies/downloads/market_wind00288/elexon_b1610/elexon_b1610.parquet"
)


def main() -> None:
    """Print and write the per-BMU validation table."""
    frame = pl.read_parquet(B1610).with_columns(mw=pl.col("output_mwh") * 2)
    candidates = pl.read_parquet(WORK_DIR / "farm_candidates.parquet").select(
        "elexon_bmu_id", "generation_capacity_mw", "capacity_mw"
    )
    duplicates = (
        frame.group_by("elexon_bmu_id", "half_hour_end_time").len().filter(pl.col("len") > 1).height
    )
    print(
        "duplicate keys:",
        duplicates,
        "| nulls:",
        frame.null_count().row(0),
        "| NaN mw:",
        frame["mw"].is_nan().sum(),
    )
    runs = (
        frame.filter(pl.col("mw") > 0)
        .sort("elexon_bmu_id", "half_hour_end_time")
        .with_columns(run=(pl.col("mw") != pl.col("mw").shift(1)).cum_sum().over("elexon_bmu_id"))
        .group_by("elexon_bmu_id", "run")
        .len()
        .group_by("elexon_bmu_id")
        .agg(longest_constant_run=pl.col("len").max())
    )
    per_bmu = (
        frame.group_by("elexon_bmu_id")
        .agg(
            first=pl.col("half_hour_end_time").min(),
            last=pl.col("half_hour_end_time").max(),
            rows=pl.len(),
            max_mw=pl.col("mw").max(),
            negative_share=(pl.col("mw") < 0).mean(),
            zero_share=(pl.col("mw") == 0).mean(),
            mean_mw=pl.col("mw").mean(),
        )
        .with_columns(
            expected=((pl.col("last") - pl.col("first")).dt.total_minutes() // 30 + 1),
        )
        .with_columns(missing_share=1 - pl.col("rows") / pl.col("expected"))
        .join(runs, on="elexon_bmu_id", how="left")
        .join(candidates, on="elexon_bmu_id", how="left")
        .with_columns(max_over_capacity=pl.col("max_mw") / pl.col("capacity_mw"))
        .sort("elexon_bmu_id")
    )
    per_bmu.write_parquet(WORK_DIR / "b1610_validation.parquet")
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(14)
    print(
        per_bmu.select(
            "elexon_bmu_id",
            "first",
            "last",
            "rows",
            "missing_share",
            "max_over_capacity",
            "negative_share",
            "zero_share",
            "longest_constant_run",
        )
    )
    monthly = (
        frame.with_columns(month=pl.col("half_hour_end_time").dt.truncate("1mo"))
        .group_by("month")
        .agg(bmus=pl.col("elexon_bmu_id").n_unique(), rows=pl.len())
        .sort("month")
    )
    print(monthly.head(6), monthly.tail(6))


if __name__ == "__main__":
    main()
