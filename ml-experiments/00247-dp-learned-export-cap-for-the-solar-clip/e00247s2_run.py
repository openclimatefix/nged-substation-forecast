"""Build every arm of experiment 155 at every substation and write the prediction frames.

| Arm | Built from | Correction |
|---|---|---|
| `a` | experiment 136's A2 | none |
| `b` | `a` | winter night offset |
| `c` | `a` | recipe R at every lead |
| `d` | `b` | recipe R at every lead, on `b`'s own past errors |
| `c48` | `a` | recipe R at leads of 48 hours or less |
| `d48` | `b` | recipe R at leads of 48 hours or less |
| `e` | `b` | ANALOG+R (experiment 145's tuned values) at every lead |
| `b0` | `a` | the offset's constants and weekend flag only (ablation) |
| `*_spread_a` | `a` | `b`, `c`, `d` with arm `a`'s spread table (sensitivity) |

Each arm's quantiles are its corrected mean plus a spread table refitted on its own tuning-month
forecasts, every factor 1. The output is in the `dp_preds` layout.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00155-*/e00247s2_run.py [--workers 6]
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Final

import e00247s2_common as common
import e00247s2_corrections as corr
import numpy as np
import polars as pl

ARM_NAMES: Final[tuple[str, ...]] = (
    "a",
    "b",
    "c",
    "d",
    "c48",
    "d48",
    "e",
    "b0",
    "b_spread_a",
    "c_spread_a",
    "d_spread_a",
)
OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "time_series_id", "run", "power_fcst_init_time", "valid_time", "lead_hours", "seed", "arm",
    *common.QUANTILES, "actual", "mean", "hour_band", "quarter", "switching_mask",
)  # fmt: skip


def build_arms(*, site: str, frame: pl.DataFrame | None = None) -> dict[str, pl.DataFrame]:
    """Return every arm's frame at a substation, all three periods together.

    Args:
        site: The substation's label.
        frame: Arm A's frame with the calendar columns, or `None` to read and annotate it.

    Returns:
        The frames by arm name, each with the base frame's rows and a `period` column.
    """
    base = frame if frame is not None else corr.add_calendar(frame=common.read_base(site=site))
    mean_a = base["mean"].to_numpy().astype(np.float64)
    offset, _ = corr.winter_offset_shift(frame=base, variant="winter")
    offset_const, _ = corr.winter_offset_shift(frame=base, variant="const")
    mean_b = mean_a + offset
    arms = {"a": base}
    arms["b"] = corr.apply_shift_and_refit_spread(frame=base, shift=offset)
    arms["b0"] = corr.apply_shift_and_refit_spread(frame=base, shift=offset_const)
    shift_c = corr.recipe_r_shift(frame=base, forecast=mean_a)
    shift_d = offset + corr.recipe_r_shift(frame=base, forecast=mean_b)
    mean_c, mean_d = mean_a + shift_c, mean_a + shift_d
    arms["c"] = corr.apply_shift_and_refit_spread(frame=base, shift=shift_c)
    arms["c48"] = corr.apply_shift_and_refit_spread(
        frame=base, shift=corr.recipe_r_shift(frame=base, forecast=mean_a, short_only=True)
    )
    arms["d"] = corr.apply_shift_and_refit_spread(frame=base, shift=shift_d)
    arms["d48"] = corr.apply_shift_and_refit_spread(
        frame=base,
        shift=offset + corr.recipe_r_shift(frame=base, forecast=mean_b, short_only=True),
    )
    for name, mean in (("b", mean_b), ("c", mean_c), ("d", mean_d)):
        arms[f"{name}_spread_a"] = base.with_columns(
            pl.Series("mean", mean.astype(np.float32)),
            *[(pl.col(q) + pl.Series(mean - mean_a)).cast(pl.Float32) for q in common.QUANTILES],
        )
    return arms


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
    arms = build_arms(site=site)
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
