"""Build arm D (experiment 155's `d_spread_a`) from arm A2 at each chain series.

Arm D is arm A2's forecast shifted by the winter night offset and then by recipe R fitted on the
offset forecast's own errors, with arm A2's spread table. Every quantile and the mean move by the
same shift.

    uv run --no-sync python ml-experiments/00161-*/e00161_arms.py [--sites D13 D13r ...]
"""

import argparse
import os
from pathlib import Path
from typing import Final

import e00161_common as common
import e00161_corr as corr
import numpy as np
import polars as pl

BASE_COLUMNS: Final[tuple[str, ...]] = (
    "time_series_id", "run", "power_fcst_init_time", "valid_time", "lead_hours", "seed",
    *common.QUANTILES, "actual", "mean", "hour_band", "quarter", "switching_mask",
)  # fmt: skip


def a2_path(*, period: str, site: str) -> str:
    """Return the file of arm A2 for one period at one series."""
    return str(common.OUT_DIR / f"a2_{period}_{site}.parquet")


def d_path(*, period: str, site: str) -> str:
    """Return the file of arm D for one period at one series."""
    return str(common.OUT_DIR / f"d_{period}_{site}.parquet")


def read_base(*, site: str) -> pl.DataFrame:
    """Return arm A2 at a series: all three periods, ascending by origin."""
    pieces = [
        pl.read_parquet(a2_path(period=period, site=site))
        .select(list(BASE_COLUMNS))
        .with_columns(period=pl.lit(period))
        for period in common.PERIODS
    ]
    return pl.concat(pieces).sort(["power_fcst_init_time", "valid_time"])


def build_d(*, base: pl.DataFrame) -> pl.DataFrame:
    """Return arm D: arm A2 shifted by the offset and recipe R, keeping arm A2's spread table."""
    mean_a = base["mean"].to_numpy().astype(np.float64)
    offset, _ = corr.winter_offset_shift(frame=base, variant="winter")
    mean_b = mean_a + offset
    shift = offset + corr.recipe_r_shift(frame=base, forecast=mean_b)
    return base.with_columns(
        pl.Series("mean", (mean_a + shift).astype(np.float32)),
        *[(pl.col(q) + pl.Series(shift)).cast(pl.Float32) for q in common.QUANTILES],
    )


def write_d(*, site: str) -> int:
    """Build and write arm D at a series; return the number of rows."""
    base = corr.add_calendar(frame=read_base(site=site))
    frame = build_d(base=base)
    rows = 0
    for period in common.PERIODS:
        part = frame.filter(pl.col("period") == period).select(list(BASE_COLUMNS))
        target = d_path(period=period, site=site)
        temporary = f"{target}.{os.getpid()}.tmp"
        part.write_parquet(temporary)
        Path(temporary).replace(target)
        rows += part.height
    return rows


def main() -> None:
    """Build arm D at every chain series."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="+", default=list(common.CHAIN_SITES))
    args = parser.parse_args()
    for site in args.sites:
        print(f"d {site}: {write_d(site=site):,} rows", flush=True)


if __name__ == "__main__":
    main()
