"""Apply experiment 155's winter night offset to each raw arm of experiment 281.

| Arm | Built from | Night term |
|---|---|---|
| `a0` | `k0` (T-all, experiment 136's A2) | winter night offset |
| `a1` | `e1` (T-all plus 20 UTC night levels) | winter night offset |
| `a2` | `e2` (`e1` plus the monotone temperature step) | winter night offset |

The raw arms `k0`, `e1`, and `e2` stay as references. Each arm's quantiles are its corrected mean
plus a spread table refitted on its own tuning-month forecasts, every factor 1, as experiment 155
builds its arm B. The output is in `e00281_run`'s `dp_preds` layout.

    uv run --no-sync python ml-experiments/00281-*/e00281_arms.py [--workers 4]
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Final

import e00281_c155 as common
import e00281_common as paths
import e00281_corrections as corr
import e00281_run as run
import polars as pl

RAW_OF: Final[dict[str, str]] = {"a0": "k0", "a1": "e1", "a2": "e2"}
OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "time_series_id", "run", "power_fcst_init_time", "valid_time", "lead_hours", "seed", "arm",
    *common.QUANTILES, "actual", "mean", "hour_band", "quarter", "switching_mask",
)  # fmt: skip


def read_raw(*, arm: str, site: str) -> pl.DataFrame:
    """Return a raw arm's forecasts at a site: all three periods, ascending by origin."""
    pieces = [
        pl.read_parquet(run.out_path(arm=arm, period=period, site=site))
        .select([c for c in OUTPUT_COLUMNS if c != "arm"])
        .with_columns(period=pl.lit(period))
        for period in common.PERIODS
    ]
    return pl.concat(pieces).sort(["power_fcst_init_time", "valid_time"])


def _write(*, arm: str, site: str, frame: pl.DataFrame) -> None:
    """Write one arm's rows per period, atomically."""
    for period in common.PERIODS:
        part = frame.filter(pl.col("period") == period).with_columns(arm=pl.lit(arm))
        target = run.out_path(arm=arm, period=period, site=site)
        temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        part.select(list(OUTPUT_COLUMNS)).write_parquet(temporary)
        temporary.rename(target)


def _job(site: str) -> str:
    for name, raw in RAW_OF.items():
        base = corr.add_calendar(frame=read_raw(arm=raw, site=site))
        offset, _ = corr.winter_offset_shift(frame=base, variant="winter")
        _write(
            arm=name, site=site, frame=corr.apply_shift_and_refit_spread(frame=base, shift=offset)
        )
    return f"{site}: done"


def main() -> None:
    """Build the offset arms at every substation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(_job, paths.SCORED_SITES):
            print(line, flush=True)


if __name__ == "__main__":
    main()
