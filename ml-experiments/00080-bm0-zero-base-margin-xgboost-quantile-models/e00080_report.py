"""Print every number that experiment 00080's write-up quotes, from the saved results.

uv run python ml-experiments/00080-*/e00080_report.py
"""

import e00080_common as common
import polars as pl


def main() -> None:
    """Print the pooled contrasts, the absolute scores, the per-series C1, and the checks."""
    results = common.RESULTS_DIR
    contrasts = pl.read_parquet(results / "contrasts.parquet")
    pooled = contrasts.filter(
        (pl.col("site") == "pooled") & (pl.col("scope") == "all leads") & (pl.col("metric") == "qs")
    )
    with pl.Config(tbl_rows=200, tbl_cols=12, tbl_width_chars=200, fmt_str_lengths=70):
        print("Pooled contrasts, quantile score, all leads (planned rows at 98.75%, others 95%)")
        print(pooled.select("id", "level_label", "difference", "lower", "upper", "cases", "months"))
        print("\nC1 on each series alone (95%)")
        print(
            contrasts.filter(
                (pl.col("id") == "C1")
                & (pl.col("site") != "pooled")
                & (pl.col("scope") == "all leads")
                & (pl.col("level_label") == "95")
            )
            .select("site", "difference", "lower", "upper")
            .sort("difference")
        )
        print("\nAbsolute quantile score by arm, all leads (95% interval over months)")
        print(
            pl.read_parquet(results / "absolute.parquet")
            .filter(pl.col("scope") == "all leads")
            .select("group", "arm", "qs", "qs_lower", "qs_upper", "cases")
            .sort("group", "arm")
        )
        print("\nReproduction of the estimated start")
        print(pl.read_parquet(results / "reproduction.parquet"))
        print("\nBest early-stopping round, by series and start, and by group")
        curves = pl.concat(
            [
                pl.read_parquet(path)
                for path in sorted((common.WORK_DIR / "curves").glob("*.parquet"))
            ],
            how="diagonal_relaxed",
        )
        print(
            curves.sort("valid")
            .group_by("group", "site", "start")
            .agg(best_round=pl.col("round").first())
            .sort("group", "site", "start")
        )
        print(
            curves.sort("valid")
            .group_by("group", "site", "start")
            .agg(best_round=pl.col("round").first())
            .group_by("group")
            .agg(
                fewest=pl.col("best_round").min(),
                median=pl.col("best_round").median(),
                most=pl.col("best_round").max(),
            )
        )
        print(
            curves.filter(pl.col("round").is_in([10, 50, 100, 250, 500]))
            .group_by("group", "start", "round")
            .agg(mean_held_out_loss=pl.col("valid").mean())
            .sort("group", "round", "start")
        )


if __name__ == "__main__":
    main()
