"""Print every number `result.md` quotes from experiment 00022's saved results.

uv run python ml-experiments/00022-era5-weather-ceiling-primary-demand/e00022_report.py
"""

from pathlib import Path
from typing import Final

import e00022_common as common
import polars as pl

RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"


def main() -> None:
    """Print the absolute scores, the contrasts, and the weather columns' means and maxima."""
    with pl.Config(tbl_rows=200, tbl_cols=20, fmt_str_lengths=80):
        print(
            pl.read_parquet(RESULTS_DIR / "absolute.parquet")
            .select("arm", "scope", "value", "lower_95", "upper_95", "coverage_90", "coverage_80")
            .sort("scope", "arm")
        )
        print(
            pl.read_parquet(RESULTS_DIR / "contrasts.parquet")
            .select("treatment", "reference", "scope", "difference", "lower", "upper", "level")
            .sort("treatment", "scope")
        )
        era5 = pl.read_parquet(common.ERA5_PATH).filter(
            (pl.col("valid_time") >= common.VALIDATION_START)
            & (pl.col("valid_time") < common.GRID_END)
        )
        ens = pl.concat(
            [
                pl.read_parquet(path, columns=list(common.WEATHER_COLUMNS))
                for path in sorted(common.VALIDATION_DIR.glob("init=*.parquet"))[::6]
            ]
        )
        print(
            pl.DataFrame(
                {
                    "column": list(common.WEATHER_COLUMNS),
                    "era5_mean": [era5[c].mean() for c in common.WEATHER_COLUMNS],
                    "ens_mean": [ens[c].mean() for c in common.WEATHER_COLUMNS],
                    "era5_max": [era5[c].max() for c in common.WEATHER_COLUMNS],
                    "ens_max": [ens[c].max() for c in common.WEATHER_COLUMNS],
                }
            )
        )


if __name__ == "__main__":
    main()
