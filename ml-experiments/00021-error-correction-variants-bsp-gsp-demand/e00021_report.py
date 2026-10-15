"""Print every number `result.md` quotes from experiment 00021's saved results.

uv run python ml-experiments/00021-error-correction-variants-bsp-gsp-demand/e00021_report.py
"""

from pathlib import Path
from typing import Final

import e00021_common as common
import polars as pl

RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"


def main() -> None:
    """Print the absolute scores, bias, contrasts, and the size of the corrections."""
    with pl.Config(tbl_rows=300, tbl_cols=20, fmt_str_lengths=80):
        absolute = pl.read_parquet(RESULTS_DIR / "absolute.parquet").filter(
            pl.col("scope") == "all leads"
        )
        print(
            absolute.filter(pl.col("metric") != "coverage")
            .select("group", "arm", "metric", "value", "lower_95", "upper_95")
            .sort("metric", "group", "arm")
        )
        print(
            absolute.filter(pl.col("metric") == "coverage")
            .select(
                "group",
                "arm",
                bias=pl.col("bias"),
                coverage_90=pl.col("value"),
                c80=pl.col("coverage_80"),
            )
            .sort("group", "arm")
        )
        print(
            pl.read_parquet(RESULTS_DIR / "contrasts.parquet")
            .select(
                "label",
                "metric",
                "scope",
                "difference",
                "lower",
                "upper",
                "level",
                "months_negative",
            )
            .sort("label", "metric", "scope")
        )
        corrections = pl.read_parquet(RESULTS_DIR / "corrections.parquet").filter(
            pl.col("seed") == 0
        )
        print(
            corrections.group_by("variant").agg(
                mean_shift=pl.col("shift").mean(),
                mean_absolute_shift=pl.col("shift").abs().mean(),
                runs_without_correction=(pl.col("history_count") < 144).sum(),
                runs=pl.len(),
            )
        )


if __name__ == "__main__":
    main()
