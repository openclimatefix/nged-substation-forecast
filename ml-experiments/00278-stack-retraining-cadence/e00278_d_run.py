"""Build arm D (experiment 155's winter night offset and recipe R) on each convex-model cadence arm.

For each cadence arm (`a2` monthly, `a3` weekly, `a4` quarterly; `e00278_g_run.py` writes their
forecasts) this writes `<cadence>_d_<period>_<site>.parquet`: the cadence arm's mean plus the winter
night offset, then recipe R, with the spread table refitted on the arm's own tuning months. The
rebuilt monthly arm is gated against experiment 155's stored arm D by `e00278_score.py`.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00278-*/e00278_d_run.py [--workers 6]
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Final

import e00278_d_common as common
import e00278_d_corrections as corr
import polars as pl

CADENCE_ARMS: Final[tuple[str, ...]] = ("a2", "a3", "a4")
OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "time_series_id", "run", "power_fcst_init_time", "valid_time", "lead_hours", "seed", "arm",
    *common.QUANTILES, "actual", "mean", "hour_band", "quarter", "switching_mask",
)  # fmt: skip


def build_arm_d(*, site: str, cadence_arm: str) -> pl.DataFrame:
    """Return arm D on a cadence arm at a substation, all three periods together."""
    base = corr.add_calendar(frame=common.read_base(site=site, cadence_arm=cadence_arm))
    mean_a = base["mean"].to_numpy().astype("float64")
    offset, _ = corr.winter_offset_shift(frame=base, variant="winter")
    mean_b = mean_a + offset
    shift_d = offset + corr.recipe_r_shift(frame=base, forecast=mean_b)
    return corr.apply_shift_and_refit_spread(frame=base, shift=shift_d)


def _write(*, arm: str, site: str, frame: pl.DataFrame) -> int:
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


def _job(job: tuple[str, str]) -> str:
    cadence_arm, site = job
    started = time.monotonic()
    frame = build_arm_d(site=site, cadence_arm=cadence_arm)
    rows = _write(arm=f"{cadence_arm}_d", site=site, frame=frame)
    return f"{cadence_arm} {site}: {rows:,} rows, {time.monotonic() - started:.0f} s"


def main() -> None:
    """Build arm D on every cadence arm at every substation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--arms", nargs="+", default=list(CADENCE_ARMS))
    args = parser.parse_args()
    jobs = [(arm, site) for arm in args.arms for site in common.SCORED_SITES]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(_job, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
