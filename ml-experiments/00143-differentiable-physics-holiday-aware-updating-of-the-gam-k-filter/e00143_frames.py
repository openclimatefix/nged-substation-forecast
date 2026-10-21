r"""Build each arm's correction and forecast frame, and check K0 against experiment 115's frame.

The four arms share K0's spread tables and factors (see `e00143_arms`), so a frame differs between
arms only through the correction.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --no-sync python \\
        ml-experiments/00143-*/e00143_frames.py [--workers 3]
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import e00143_arms as arms
import e00143_common as common
import e00143_site as site_module
import numpy as np
import polars as pl

REPRODUCTION_TOLERANCE: float = 1e-3
"""K0's quantiles must match experiment 115's stored frame to this many points (float32 storage)."""


@lru_cache(maxsize=2)
def get_site(site: str) -> site_module.SiteData:
    """Load a site once per worker process."""
    return site_module.load_site(site=site)


def _site_job(site: str) -> str:
    data = get_site(site)
    spread = arms.k0_spread(site=site)
    arms.FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    for arm in arms.ARMS:
        correction = arms.arm_correction(data=data, arm=arm)
        frame = arms.frame_for_arm(data=data, arm=arm, correction=correction, spread=spread)
        frame = arms.replace_period_masks(frame=frame, data=data)
        frame.write_parquet(arms.frame_path(arm=arm, site=site))
        lines.append(f"{arm} {site}: {frame.height:,} rows")
        if arm == "K0":
            lines.append(_reproduction_line(site=site, frame=frame))
    return "\n".join(lines)


def _reproduction_line(*, site: str, frame: pl.DataFrame) -> str:
    """Compare K0's frame with experiment 115's stored K0 frame on shared rows."""
    stored = pl.read_parquet(arms.SRC_115_FRAMES_DIR / f"K0_{site}.parquet")
    keys = ["power_fcst_init_time", "valid_time"]
    both = frame.join(stored, on=keys, how="inner", suffix="_115")
    worst = max(
        float(np.nanmax(np.abs(both[q].to_numpy() - both[f"{q}_115"].to_numpy())))
        for q in ("q01", "q05", "q50", "q95", "q99")
    )
    status = "OK" if worst <= REPRODUCTION_TOLERANCE else "MISMATCH"
    return (
        f"K0 {site} reproduces experiment 115: {both.height:,} of {stored.height:,} rows, "
        f"largest quantile difference {worst:.2e} points {status}"
    )


def main() -> None:
    """Build every arm's frames."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for text in pool.map(_site_job, common.SCORED_SITES):
            print(text, flush=True)


if __name__ == "__main__":
    main()
