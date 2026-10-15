"""Print every number `result.md` quotes from experiment 00024's saved results.

uv run python ml-experiments/00024-pooling-groupings-demand-xgboost/e00024_report.py
"""

import json
from pathlib import Path
from typing import Final

import e00024_common as common
import polars as pl

RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"


def main() -> None:
    """Print group membership, fit cost, absolute scores, per-series scores, and contrasts."""
    series = common.series_table()
    with pl.Config(tbl_rows=400, tbl_cols=20, fmt_str_lengths=90):
        for grouping in ("size", "random"):
            labels = common.group_labels(series=series, grouping=grouping)
            print(
                grouping,
                {g: sorted(s for s, n in labels.items() if n == g) for g in set(labels.values())},
            )
        times = pl.read_ndjson(common.WORK_DIR / "fit_times.jsonl")
        print(
            times.group_by("arm", "seed")
            .agg(models=pl.len(), seconds=pl.col("seconds").sum())
            .group_by("arm")
            .agg(
                models=pl.col("models").first(),
                seconds_per_seed=pl.col("seconds").mean(),
                min_seconds=pl.col("seconds").min(),
                max_seconds=pl.col("seconds").max(),
            )
            .sort("models", descending=True)
        )
        sizes = {
            arm: sum(
                p.stat().st_size for p in (common.WORK_DIR / "models" / arm).glob("*_seed0.ubj")
            )
            / 1e6
            for arm in common.ARMS
        }
        print("model megabytes per seed:", json.dumps(sizes, indent=1))
        absolute = pl.read_parquet(RESULTS_DIR / "absolute.parquet").filter(
            pl.col("scope") == "all leads"
        )
        print(
            absolute.filter(pl.col("metric") == "quantile_score")
            .select("group", "arm", "value", "lower_95", "upper_95")
            .sort("group", "value")
        )
        print(
            absolute.filter(pl.col("metric") == "coverage")
            .select("group", "arm", bias="bias", coverage_90="value", coverage_80="coverage_80")
            .filter(pl.col("group").is_in(["Primary (16)", "All (20)"]))
            .sort("group", "arm")
        )
        print(
            pl.read_parquet(RESULTS_DIR / "per_series.parquet")
            .filter(
                pl.col("arm").is_in(
                    [
                        "own_stack",
                        "type_id_stack",
                        "all_id_stack",
                        "all_noid_stack",
                        "size_id_stack",
                        "random_id_stack",
                        "manual_heuristic",
                        "climatology",
                    ]
                )
            )
            .pivot(on="arm", index=["site", "group"], values="value")
            .sort("group", "site")
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


if __name__ == "__main__":
    main()
