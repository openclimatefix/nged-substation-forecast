# ruff: noqa: PTH207
"""Post hoc: does soft down-weighting work at all with a much larger noise factor?

Reads the `injected_soft_large/*.parquet` files (worlds 9200 to 9203, all six substations) and
prints, for the AFTER phase of transfers of 25% of load or more that last over 1 day, the gain over
K0 of each arm and its share of the perfect-knowledge MASK gain.

    OMP_NUM_THREADS=1 uv run --no-sync python e00227_soft_large.py
"""

from __future__ import annotations

import glob

import polars as pl
from e00227_common import WORK_DIR


def main() -> None:
    """Print the table."""
    frames = [
        pl.read_parquet(p)
        for p in sorted(glob.glob(str(WORK_DIR / "injected_soft_large" / "*.parquet")))
    ]
    f = pl.concat(frames, how="diagonal_relaxed").filter(
        (pl.col("phase") == "AFTER") & (pl.col("fraction") >= 0.25) & (pl.col("event_hours") > 24)
    )
    k0 = f.filter(pl.col("arm") == "K0|NONE").select("site", "world", "row", k0=pl.col("qs"))
    out = (
        f.join(k0, on=["site", "world", "row"])
        .group_by("arm")
        .agg(gain=(pl.col("k0") - pl.col("qs")).mean(), n=pl.len())
        .sort("arm")
    )
    ref = out.filter(pl.col("arm") == "PERFECT|MASK")["gain"][0]
    print(
        out.with_columns(share_of_perfect_mask=pl.col("gain") / ref).with_columns(
            pl.col(pl.Float64).round(4)
        )
    )


if __name__ == "__main__":
    main()
