"""Print every number `result.md` quotes from experiment 00030's saved results.

The seed-0 screen (arms A0 to A5, one seed) is in `work/screen_a0_a5`; the 3-seed scoring of A0,
A1, A3, and A7 is in `work/final`.

    uv run python ml-experiments/00030-lagged-power-feature-sets-primary-demand/e00030_report.py
"""

from pathlib import Path
from typing import Final

import e00030_common as common
import polars as pl

SCREEN_DIR: Final[Path] = common.WORK_DIR / "screen_a0_a5"
FINAL_DIR: Final[Path] = common.WORK_DIR / "final"


def main() -> None:
    """Print the absolute scores and the contrasts of both scorings."""
    with pl.Config(tbl_rows=300, tbl_cols=20, fmt_str_lengths=90, tbl_width_chars=200):
        for name, folder in (("seed-0 screen", SCREEN_DIR), ("3 seeds", FINAL_DIR)):
            print(f"== {name}: absolute scores")
            print(
                pl.read_parquet(folder / "absolute.parquet")
                .filter(pl.col("metric") != "coverage")
                .select("arm", "scope", "metric", "value", "lower_95", "upper_95")
                .sort("metric", "scope", "arm")
            )
            print(f"== {name}: contrasts")
            print(
                pl.read_parquet(folder / "contrasts.parquet")
                .select("label", "metric", "scope", "difference", "lower", "upper", "level")
                .sort("label", "metric", "scope")
            )


if __name__ == "__main__":
    main()
