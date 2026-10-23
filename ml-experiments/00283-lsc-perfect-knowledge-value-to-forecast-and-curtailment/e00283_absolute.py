"""Absolute no-knowledge (K0) scores by phase, in points of effective capacity, for the page.

uv run --no-sync python e00283_absolute.py
"""

import polars as pl
from e00283_common import WORK_DIR

lazy = pl.scan_parquet(WORK_DIR / "scores" / "D*_*[0-9].parquet").filter(pl.col("arm") == "K0")
print(
    lazy.group_by("phase")
    .agg(
        n=pl.len(),
        crps_nra=pl.col("qs_nra").mean(),
        crps_obs=pl.col("qs_obs").mean(),
        mae_median_nra=pl.col("ae_median_nra").mean(),
        mae_mean_nra=pl.col("ae_mean_nra").mean(),
        tw95_nra=pl.col("tw95_nra").mean(),
        tw99_nra=pl.col("tw99_nra").mean(),
    )
    .sort("phase")
    .collect()
)
