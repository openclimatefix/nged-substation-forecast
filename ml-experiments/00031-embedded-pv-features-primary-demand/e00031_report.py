"""Print every number `result.md` quotes from experiment 00031's saved results.

uv run python ml-experiments/00031-embedded-pv-features-primary-demand/e00031_report.py
"""

from pathlib import Path
from typing import Final

import e00031_common as common
import polars as pl

RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"


def main() -> None:
    """Print the absolute scores, the contrasts, the per-series scores, and the history size."""
    with pl.Config(tbl_rows=300, tbl_cols=20, fmt_str_lengths=80):
        absolute = pl.read_parquet(RESULTS_DIR / "absolute.parquet")
        print(absolute.filter(pl.col("group") == "Primary").sort("subset", "arm"))
        print(absolute.filter(pl.col("group") != "Primary").sort("group", "subset", "arm"))
        print(
            pl.read_parquet(RESULTS_DIR / "contrasts.parquet")
            .select("label", "metric", "subset", "difference", "lower", "upper", "level")
            .sort("label", "metric", "subset")
        )
        print(pl.read_parquet(RESULTS_DIR / "per_site.parquet").sort("site", "arm"))
        history = pl.read_parquet(RESULTS_DIR / "history.parquet").filter(pl.col("seed") == 0)
        print(
            history.select(
                mean_shape_slope=pl.col("slope").mean(),
                runs_with_slope=(pl.col("slope") != 0).sum(),
                runs=pl.len(),
            )
        )


if __name__ == "__main__":
    main()
