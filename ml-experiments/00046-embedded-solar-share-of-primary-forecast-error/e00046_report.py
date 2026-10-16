"""Print every number the write-up of experiment 00046 quotes, and save the text beside the results.

Reads `work/results/*.parquet` (see `e00046_stats.py`) and the solar calibration, and writes
`work/results/report.txt`. The write-up quotes only numbers printed here.

    uv run python ml-experiments/00046-*/e00046_report.py
"""

import contextlib
import io
from typing import Final

import e00046_common as common
import polars as pl

SPLIT_SCOPES: Final[tuple[str, ...]] = (
    "verdict series",
    "without D9",
    "verdict series and D5",
    "verdict series, all 12 months",
)


def build_report() -> None:
    """Print the tables the write-up quotes."""
    results = common.RESULTS_DIR
    with pl.Config(tbl_rows=300, tbl_cols=30, fmt_float="mixed", tbl_width_chars=220):
        print("Calibration slope of the CAMS anomaly on the ENS ensemble-mean anomaly (training)")
        print(pl.read_parquet(common.SOLAR_DIR / "calibration.parquet"))
        print("\nAbsolute scores on the verdict series (points of capacity)")
        print(pl.read_parquet(results / "absolute.parquet"))
        print("\nCoefficient diagnostic")
        print(pl.read_parquet(results / "diagnostic.parquet"))
        print("\nCoverage of the 0.05 and 0.95 quantiles")
        print(pl.read_parquet(results / "coverage.parquet"))
        contrasts = pl.read_parquet(results / "contrasts.parquet")
        columns = [
            "contrast",
            "treatment",
            "reference",
            "block",
            "level",
            "n_resamples",
            "difference",
            "lower",
            "upper",
            "half_width",
            "seed_spread",
            "rows_per_seed",
            "blocks",
        ]
        print("\nThe four planned contrasts (deciding C3 and C4 at 97.5%, C5 and C6 at 95%)")
        print(
            contrasts.filter(
                (pl.col("scope") == "verdict series")
                & (pl.col("block") == "month")
                & (pl.col("n_resamples") == 50_000)
            ).select(columns)
        )
        for scope in SPLIT_SCOPES[1:]:
            print(f"\nContrasts, scope: {scope} (exploratory, 95%)")
            print(contrasts.filter(pl.col("scope") == scope).select(columns))
        print("\nWeek-block reads of C3 to C6 (exploratory)")
        print(contrasts.filter(pl.col("block") == "week").select(columns))
        for kind in ("lead band", "clear-sky index", "series"):
            print(f"\nContrasts split by {kind} (exploratory, 95%)")
            print(
                contrasts.filter(pl.col("scope").str.starts_with(kind)).select(["scope", *columns])
            )
        print("\nMean paired difference by valid month")
        print(
            pl.read_parquet(results / "by_month.parquet").pivot(
                on="valid_month", index="contrast", values="difference"
            )
        )


def main() -> None:
    """Print the report and save a copy."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        build_report()
    text = buffer.getvalue()
    print(text)
    (common.RESULTS_DIR / "report.txt").write_text(text)


if __name__ == "__main__":
    main()
