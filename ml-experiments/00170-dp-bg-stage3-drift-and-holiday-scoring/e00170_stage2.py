"""Build the stage 2 arms of experiment 170 at every substation and write the prediction frames.

Each stage 1 arm of `e00170_run.py` gets four stage 2 arms, named with a prefix for the stage 1 arm
(`r` for `a2_repro`, `s24` for `m24_d1`, `sall` for `all_d1`):

| Arm | Built from | Correction |
|---|---|---|
| `<prefix>_a` | the stage 1 arm | none (spread as stored) |
| `<prefix>_b` | `<prefix>_a` | winter night offset |
| `<prefix>_c` | `<prefix>_a` | recipe R at every lead |
| `<prefix>_d` | `<prefix>_b` | recipe R at every lead, on the offset forecast's own past errors |

So `r_a`, `r_b`, `r_c`, and `r_d` repeat experiment 155's arms A, B, C, and D with this folder's
fitting code. Each arm's quantiles are its corrected mean plus a spread table refitted on its own
tuning-month forecasts, every factor 1. The output is in the `dp_preds` layout.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00170-*/e00170_stage2.py [--workers 6]
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Final

import e00170_corrections as corr
import e00170_stage2_common as common
import numpy as np
import polars as pl

STAGE1_PREFIX: Final[dict[str, str]] = {"a2_repro": "r", "m24_d1": "s24", "all_d1": "sall"}
SUFFIXES: Final[tuple[str, ...]] = ("a", "b", "c", "d")
ARM_NAMES: Final[tuple[str, ...]] = tuple(
    f"{prefix}_{suffix}" for prefix in STAGE1_PREFIX.values() for suffix in SUFFIXES
)
OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "time_series_id", "run", "power_fcst_init_time", "valid_time", "lead_hours", "seed", "arm",
    *common.QUANTILES, "actual", "mean", "hour_band", "quarter", "switching_mask",
)  # fmt: skip


def build_arms(*, site: str, stage1: str) -> dict[str, pl.DataFrame]:
    """Return the four stage 2 arms of a stage 1 arm at a substation, all periods together.

    Args:
        site: The substation's label.
        stage1: The stage 1 arm's name (a key of `STAGE1_PREFIX`).

    Returns:
        The frames by arm name, each with the base frame's rows and a `period` column.
    """
    prefix = STAGE1_PREFIX[stage1]
    base = corr.add_calendar(frame=common.read_base(site=site, base=stage1))
    mean_a = base["mean"].to_numpy().astype(np.float64)
    offset, _ = corr.winter_offset_shift(frame=base, variant="winter")
    mean_b = mean_a + offset
    shift_d = offset + corr.recipe_r_shift(frame=base, forecast=mean_b)
    return {
        f"{prefix}_a": base,
        f"{prefix}_b": corr.apply_shift_and_refit_spread(frame=base, shift=offset),
        f"{prefix}_c": corr.apply_shift_and_refit_spread(
            frame=base, shift=corr.recipe_r_shift(frame=base, forecast=mean_a)
        ),
        f"{prefix}_d": corr.apply_shift_and_refit_spread(frame=base, shift=shift_d),
    }


def _write(*, arm: str, site: str, frame: pl.DataFrame) -> int:
    """Write one arm's rows per period, atomically, and return the number of rows written."""
    rows = 0
    for period in common.PERIODS:
        part = frame.filter(pl.col("period") == period).with_columns(arm=pl.lit(arm))
        target = common.out_path(arm=arm, period=period, site=site)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        part.select(list(OUTPUT_COLUMNS)).write_parquet(temporary)
        temporary.rename(target)
        rows += part.height
    return rows


def _job(site: str) -> str:
    started = time.monotonic()
    arms = {}
    for stage1 in STAGE1_PREFIX:
        arms.update(build_arms(site=site, stage1=stage1))
    rows = {arm: _write(arm=arm, site=site, frame=frame) for arm, frame in arms.items()}
    return f"{site}: {sum(rows.values()):,} rows, {time.monotonic() - started:.0f} s"


def main() -> None:
    """Build every arm at every substation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--sites", nargs="+", default=list(common.SCORED_SITES))
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(_job, args.sites):
            print(line, flush=True)


if __name__ == "__main__":
    main()
