"""Experiment 177, stage 2: score the public-data indicators (DFS events, price spikes).

Reads the files `e00177_fetch.py` cached and the day lists of experiment 175, and compares the 37
unlabelled worst days with 1,000 draws of 37 matched ordinary days, as `e00177_run.py` does.
"""

import json
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SEED: Final[int] = 177
N_DRAWS: Final[int] = 1000
BASE: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00177-dp-worst-days-other-data-sources"
E172: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00172-dp-error-budget-worst-days" / "work"
E175: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00175-dp-unlabelled-day-shapes" / "work"
DFS_LAST_COVERED: Final[str] = "2026-03-27"


def main() -> None:
    """Score the public-data indicators."""
    prices = pl.concat(
        [pl.read_parquet(f) for f in sorted((BASE / "public" / "system_prices").glob("*.parquet"))]
    )
    prices = prices.with_columns(
        date=pl.col("settlementDate").str.to_date(),
        price=pl.max_horizontal("systemSellPrice", "systemBuyPrice"),
    )
    p95 = prices["price"].quantile(0.95)
    p995 = prices["price"].quantile(0.995)
    spike = prices.group_by("date").agg(
        p2=(pl.col("price") >= p95).any(), p2_top05=(pl.col("price") >= p995).any()
    )
    dfs = pl.read_csv(BASE / "public/dfs_service_requirement.csv").with_columns(
        date=pl.col("Delivery Date").str.to_date("%d/%m/%Y")
    )
    dfs_days = dfs.select("date").unique().with_columns(p1=pl.lit(True))
    days = (
        pl.read_parquet(E172 / "days.parquet")
        .filter(pl.col("n_rows") >= 24)
        .select("site", "local_date")
    )
    flags = (
        days.join(spike, left_on="local_date", right_on="date", how="left")
        .join(dfs_days, left_on="local_date", right_on="date", how="left")
        .with_columns(pl.col("p1", "p2", "p2_top05").fill_null(False))
        .with_columns(
            p1_in_dfs_coverage=pl.col("local_date") <= pl.lit(DFS_LAST_COVERED).str.to_date()
        )
    )
    key = ["site", "local_date"]
    target = pl.read_parquet(E175 / "day_shapes.parquet").select(key)
    pool = pl.read_parquet(E175 / "ordinary_day_shapes.parquet").select(key).join(flags, on=key)
    t = target.join(flags, on=key)
    cols = ["p1", "p2", "p2_top05"]
    rng = np.random.default_rng(SEED)
    draws = {c: [] for c in [*cols, "union"]}
    for _ in range(N_DRAWS):
        picks = []
        for r in target.iter_rows(named=True):
            cand = pool.filter(
                pl.col("site") == r["site"],
                pl.col("local_date").dt.month() == r["local_date"].month,
            )
            if cand.height == 0:
                cand = pool.filter(pl.col("site") == r["site"])
            picks.append(cand.row(int(rng.integers(cand.height)), named=True))
        d = pl.DataFrame(picks).with_columns(union=pl.col("p1") | pl.col("p2"))
        for c in list(draws):
            draws[c].append(int(d[c].sum()))
    t = t.with_columns(union=pl.col("p1") | pl.col("p2"))
    flags = flags.with_columns(union=pl.col("p1") | pl.col("p2"))
    report = {"p95_price": p95, "p995_price": p995}
    for c in list(draws):
        arr = np.array(draws[c])
        obs = int(t[c].sum())
        report[c] = {
            "observed_of_37": obs,
            "ordinary_mean_of_37": float(arr.mean()),
            "ordinary_95pct_range": [
                float(np.percentile(arr, 2.5)),
                float(np.percentile(arr, 97.5)),
            ],
            "one_sided_p": float((arr >= obs).mean()),
            "all_scored_days_rate": int(flags[c].sum()) / flags.height,
        }
    report["target_days_after_dfs_archive_ends"] = int((~t["p1_in_dfs_coverage"]).sum())
    (BASE / "work" / "report_stage2.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
