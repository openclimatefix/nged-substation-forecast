"""Print every number `result.md` quotes from experiment 00019's saved results.

uv run python ml-experiments/00019-primary-demand-stack-vs-manual-heuristic/e00019_report.py
"""

from pathlib import Path
from typing import Final

import e00019_common as common
import polars as pl

RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"


def main() -> None:
    """Print the absolute scores, the contrasts, and the correction's size."""
    with pl.Config(tbl_rows=200, tbl_cols=20, fmt_str_lengths=80):
        print(
            pl.read_parquet(RESULTS_DIR / "absolute.parquet")
            .filter(pl.col("metric") != "coverage")
            .select("arm", "scope", "metric", "value", "lower_95", "upper_95")
            .sort("metric", "scope", "arm")
        )
        print(
            pl.read_parquet(RESULTS_DIR / "contrasts.parquet")
            .select("label", "metric", "scope", "difference", "lower", "upper", "level")
            .sort("label", "metric", "scope")
        )
        corrections = pl.read_parquet(RESULTS_DIR / "corrections.parquet").filter(
            (pl.col("fit") == "n1") & (pl.col("seed") == 0)
        )
        print(
            corrections.select(
                mean_correction=pl.col("correction").mean(),
                mean_absolute_correction=pl.col("correction").abs().mean(),
                runs_without_correction=(pl.col("history_count") < 144).sum(),
                runs=pl.len(),
            )
        )


if __name__ == "__main__":
    main()
