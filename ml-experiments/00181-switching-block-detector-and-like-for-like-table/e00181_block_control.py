"""Part 1, step 2 controls: the threshold curve on the 17 block days and an injected-block control.

Reads the alarm table written by `e00181_block.py`. The injected-block control adds a rectangular
block of known height to the real residual of one series at a time, in even-numbered months, and
asks whether the budgeted detector alarms inside the block's span (plus 3 hours either side).

    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00181-*/e00181_block_control.py
"""

from __future__ import annotations

import json
from typing import Final

import numpy as np
import polars as pl
from e00181_block import ONSET_TOLERANCE, best_z, series_scores
from e00181_blocks17 import block_days
from e00181_common import GRID_LENGTH, TUNING_MONTHS, WORK_DIR, stamp_months
from e00181_variants import load_both

HEIGHTS: Final[tuple[float, ...]] = (3.0, 4.0, 5.0, 6.0, 8.0)
"""Injected block heights in noise units of the series' single-edge contrast at that length."""
INJECT_LENGTHS: Final[tuple[int, ...]] = (12, 24, 48, 96)
PER_CELL: Final[int] = 10
SEED: Final[int] = 181


def main() -> None:
    """Print the threshold curve and the recall of injected blocks."""
    summary = json.loads((WORK_DIR / "block_summary.json").read_text())
    threshold = summary["S0"]["threshold"]
    alarms = pl.read_parquet(WORK_DIR / "block_alarms_S0.parquet")
    print("Found block days and unexplained odd-month alarms, by threshold (S0):")
    days = block_days()
    candidates = pl.read_parquet(WORK_DIR / "block_candidates_S0.parquet")
    for tau in (2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, threshold):
        kept = candidates.filter(pl.col("z") > tau)
        found = sum(
            kept.filter(
                pl.col("series") == d["series"],
                pl.col("stamp").is_between(
                    d["day_first_stamp"] - ONSET_TOLERANCE, d["day_last_stamp"] + ONSET_TOLERANCE
                ),
            ).height
            > 0
            for d in days.iter_rows(named=True)
        )
        unexplained_odd = kept.filter(pl.col("odd"), ~pl.col("explained")).height
        unexplained_even = kept.filter(~pl.col("odd"), ~pl.col("explained")).height
        print(
            f"  tau {tau:.2f}: found {found} of 17; unexplained alarms odd {unexplained_odd}, "
            f"even {unexplained_even}"
        )
    del alarms

    panel, _ = load_both()
    residual = {s: v for s, v in panel["residual"].items() if np.isfinite(v).any()}
    base = {s: series_scores(values=v) for s, v in residual.items()}
    months = stamp_months()
    even_ok = ~np.isin(months, list(TUNING_MONTHS))
    rng = np.random.default_rng(SEED)
    rows = []
    for s, values in residual.items():
        _, scales = base[s]
        for length in INJECT_LENGTHS:
            starts = np.flatnonzero(
                even_ok[: GRID_LENGTH - length - 100]
                & np.isfinite(values[: GRID_LENGTH - length - 100])
            )
            starts = starts[(starts > 100) & (starts < GRID_LENGTH - length - 100)]
            for height in HEIGHTS:
                hits = 0
                for start in rng.choice(starts, size=PER_CELL, replace=False):
                    injected = values.copy()
                    injected[start : start + length] += height * scales[length] / 1.0
                    scores, _ = series_scores(values=injected, scales=scales)
                    z, _ = best_z(scores=scores)
                    window = np.abs(z[start - ONSET_TOLERANCE : start + length + ONSET_TOLERANCE])
                    hits += int(np.nanmax(window) > threshold)
                rows.append({"series": s, "length_h": length / 2, "height": height, "hits": hits})
    control = pl.DataFrame(rows)
    table = (
        control.group_by("length_h", "height")
        .agg(hits=pl.col("hits").sum(), n=pl.len() * PER_CELL)
        .with_columns(recall=pl.col("hits") / pl.col("n"))
        .sort("length_h", "height")
    )
    table.write_parquet(WORK_DIR / "block_injection_control.parquet")
    with pl.Config(tbl_rows=40):
        print(table)


if __name__ == "__main__":
    main()
