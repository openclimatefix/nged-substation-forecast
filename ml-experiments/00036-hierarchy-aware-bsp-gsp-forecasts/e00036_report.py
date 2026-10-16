"""Print every number that experiment 00036's page quotes, from the saved results.

uv run python ml-experiments/00036-*/e00036_report.py
"""

import e00036_common as common
import polars as pl

RESULTS_DIR = common.WORK_DIR / "results"


def main() -> None:
    """Print the absolute scores, the contrasts, and the case counts."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    absolute = pl.read_parquet(RESULTS_DIR / "absolute.parquet")
    for metric in ("quantile_score", "median_error"):
        print(f"\n{metric}, all leads, by series and arm")
        print(
            absolute.filter((pl.col("metric") == metric) & (pl.col("scope") == "all leads"))
            .select("site", "arm", pl.col("value").round(3))
            .pivot(on="arm", index="site", values="value")
        )
    contrasts = pl.read_parquet(RESULTS_DIR / "contrasts.parquet").filter(
        pl.col("metric") == "quantile_score"
    )
    print("\ncontrasts on the quantile score")
    print(
        contrasts.select(
            "planned",
            "treatment",
            "reference",
            "site",
            "scope",
            pl.col("difference", "lower", "upper").round(3),
            "level",
            "months_negative",
        ).sort("planned", "treatment", "site", descending=[True, False, False])
    )
    print("\ncases per series and arm")
    print(
        pl.read_parquet(RESULTS_DIR / "cases.parquet").pivot(on="arm", index="label", values="rows")
    )


if __name__ == "__main__":
    main()
