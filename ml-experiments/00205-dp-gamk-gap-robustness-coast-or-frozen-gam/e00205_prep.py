"""Store each substation's unfiltered forecast and filter input, and check they rebuild arm A.

For each decision substation this writes, under `work/prep/`, the unfiltered forecast of every run
(`<site>_base.parquet`, as `e00205_base.build_base` makes it) and the filter's residual stream
(`<site>_stream.npz`: the stream, the hour band of every grid row, and the grid's end times). It
then runs experiment 136's A2 filter (the POOL setting) on the stream and compares the rebuilt mean
forecast with the stored A2 forecast of experiment 136, which is the first control.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00205-*/e00205_prep.py [--workers 4]
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Final

import e00205_base as base
import e00205_common as common
import e00205_filters as filters
import numpy as np
import polars as pl
import torch

PREP_DIR: Final[Path] = common.WORK_DIR / "prep"
MEAN_TOLERANCE: Final[float] = 1e-3
"""The rebuilt mean may differ from the stored mean by this many points (float32 storage)."""


def base_path(*, site: str) -> Path:
    """Return the file that holds a substation's unfiltered forecasts."""
    return PREP_DIR / f"{site}_base.parquet"


def stream_path(*, site: str) -> Path:
    """Return the file that holds a substation's residual stream."""
    return PREP_DIR / f"{site}_stream.npz"


def read_stream(*, site: str) -> tuple[np.ndarray, np.ndarray]:
    """Return a substation's residual stream and the hour band of every grid row."""
    with np.load(stream_path(site=site)) as stored:
        return stored["stream"], stored["band"]


def _job(site: str) -> str:
    torch.set_num_threads(1)
    started = time.monotonic()
    context = base.load_site(site=site)
    frame = base.build_base(context=context)
    stream = base.residual_stream(context=context)
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(base_path(site=site))
    band = context.cache.calendar.band.astype(np.int8)
    np.savez(stream_path(site=site), stream=stream, band=band)
    run = filters.filter_stream(residual=stream, band=band, settings=base.pool_settings())
    mean = base.arm_frame(mean_frame=base.arm_mean_frame(base=frame, trace=run.trace))
    stored = pl.concat(
        [
            pl.read_parquet(common.E136_WORK_DIR / "dp_preds" / f"a2_{period}_{site}.parquet")
            for period in base.PERIODS
        ]
    ).select("run", "valid_time", stored_mean="mean", stored_q50="q50")
    joined = mean.join(stored, on=["run", "valid_time"], how="inner")
    gap = (joined["mean"] - joined["stored_mean"]).abs()
    median_gap = (joined["q50"] - joined["stored_q50"]).abs()
    return (
        f"{site}: {frame.height:,} base rows, {joined.height:,} matched to the stored A2, "
        f"max |mean - stored| {gap.max():.2e}, max |q50 - stored| {median_gap.max():.2e}, "
        f"{time.monotonic() - started:.0f} s"
    )


def main() -> None:
    """Prepare every decision substation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sites", nargs="+", default=list(common.DECISION_SITES))
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(_job, args.sites):
            print(line, flush=True)


if __name__ == "__main__":
    main()
