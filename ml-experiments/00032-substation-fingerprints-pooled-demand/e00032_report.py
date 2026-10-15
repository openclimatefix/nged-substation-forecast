"""Print every number `result.md` quotes from experiment 00032's saved results.

uv run python ml-experiments/00032-substation-fingerprints-pooled-demand/e00032_report.py
"""

from pathlib import Path
from typing import Final

import e00032_common as common
import polars as pl

RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"


def main() -> None:
    """Print fit cost, absolute scores, per-series scores, and contrasts."""
    with pl.Config(tbl_rows=400, tbl_cols=20, fmt_str_lengths=90):
        times = pl.read_ndjson(common.WORK_DIR / "fit_times.jsonl")
        print(
            times.group_by("arm").agg(models=pl.len(), seconds=pl.col("seconds").sum()).sort("arm")
        )
        absolute = pl.read_parquet(RESULTS_DIR / "absolute.parquet").filter(
            pl.col("scope") == "all leads"
        )
        print(
            absolute.filter(pl.col("metric") == "quantile_score")
            .select("table", "group", "arm", "value", "lower_95", "upper_95")
            .sort("table", "group", "value")
        )
        print(
            absolute.filter(
                (pl.col("metric") == "coverage")
                & pl.col("group").is_in(["Primary (16)", "Held-out (8)"])
            )
            .select(
                "table", "group", "arm", bias="bias", coverage_90="value", coverage_80="coverage_80"
            )
            .sort("table", "group", "arm")
        )
        print(
            pl.read_parquet(RESULTS_DIR / "per_series.parquet")
            .filter(pl.col("arm").str.ends_with("_stack") | pl.col("arm").str.starts_with("manual"))
            .pivot(on="arm", index=["table", "site", "group"], values="value")
            .sort("table", "group", "site")
        )
        contrasts = pl.read_parquet(RESULTS_DIR / "contrasts.parquet")
        print(
            contrasts.filter(pl.col("planned") & (pl.col("scope") == "all leads"))
            .select("label", "metric", "difference", "lower", "upper", "level", "months_negative")
            .sort("label", "metric")
        )
        print(
            contrasts.filter(
                ~pl.col("planned")
                & (pl.col("scope") == "all leads")
                & (pl.col("metric") == "quantile_score")
            )
            .select("label", "difference", "lower", "upper", "months_negative")
            .sort("label")
        )
        print(
            contrasts.filter(
                pl.col("planned")
                & (pl.col("scope") != "all leads")
                & (pl.col("metric") == "quantile_score")
            )
            .select("label", "scope", "difference", "lower", "upper")
            .sort("label", "scope")
        )
        gap = absolute.filter(
            (pl.col("metric") == "quantile_score")
            & (pl.col("group") == "Primary (16)")
            & (pl.col("table") == "warm")
        ).select("arm", "value")
        print(gap.sort("value"))


if __name__ == "__main__":
    main()
