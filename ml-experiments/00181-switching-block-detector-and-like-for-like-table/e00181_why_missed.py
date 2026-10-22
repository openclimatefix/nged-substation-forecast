"""Part 1, step 1: why neither RO-NC nor the pair alarm fired on the 17 block days.

For each block day, reads the arm B residual panel (RO-NC's input) and reports the step in noise
units at four window lengths, the block's length on that residual, whether a neighbour stepped the
other way, and the nearest stored RO-NC alarm. Nothing is fitted.

    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00181-*/e00181_why_missed.py
"""

from __future__ import annotations

from typing import Final

import numpy as np
import polars as pl
from e00181_blocks17 import block_days, stamp_time
from e00181_common import ADJACENCY, ML_EXPERIMENTS_DATA_DIR, WORK_DIR
from e00181_detectors import level_shift, robust_scale
from e00181_variants import load_both, scales_for, shifts

RUN_130_B: Final = ML_EXPERIMENTS_DATA_DIR / "00130-ro-nc-detector-on-the-gam-k-residual" / "B"
SEARCH: Final[int] = 4
"""Stamps (2 hours) either side of the block's onset searched for the largest step."""
WINDOWS: Final[tuple[int, ...]] = (8, 12, 24, 48)
"""Half-hours in the step statistic's two windows: 4, 6, 12, and 24 hours."""
LEVEL_WINDOW: Final[int] = 12
MAX_BLOCK: Final[int] = 7 * 48


def block_extent(*, residual: np.ndarray, onset: int) -> tuple[float, float, float]:
    """Return the step in MW and the block's length in hours (NaN if it never returns)."""
    pre = np.nanmedian(residual[onset - LEVEL_WINDOW : onset])
    after = np.nanmedian(residual[onset : onset + LEVEL_WINDOW])
    step = float(after - pre)
    for t in range(onset + LEVEL_WINDOW, onset + MAX_BLOCK):
        window = residual[t : t + LEVEL_WINDOW]
        if np.isfinite(window).sum() < LEVEL_WINDOW // 2:
            continue
        if (np.nanmedian(window) - pre) * np.sign(step) < 0.5 * abs(step):
            return step, (t - onset) / 2, float(np.nanmedian(window) - pre)
    return step, float("nan"), float("nan")


def main() -> None:
    """Print and write one row per block day."""
    panel, offset_panel = load_both()
    residual = panel["residual"]
    scales = {
        w: {s: robust_scale(shift=level_shift(values=v, window=w)) for s, v in residual.items()}
        for w in WINDOWS
    }
    shift_by_window = {
        w: {s: level_shift(values=v, window=w) for s, v in residual.items()} for w in WINDOWS
    }
    x_scale = scales_for(variant="X", residual=residual, offset=offset_panel["residual"])
    x_shift = shifts(variant="X", residual=residual, offset=offset_panel["residual"])
    alarms = pl.read_parquet(RUN_130_B / "alarms.parquet")
    pair = pl.read_parquet(
        ML_EXPERIMENTS_DATA_DIR
        / "00156-switching-faster-onset-alarm-from-opposite-steps"
        / "detect"
        / "real_W8_alarms.parquet"
    )
    rows = []
    for day in block_days().iter_rows(named=True):
        s, onset = day["series"], day["onset_stamp"]
        step_mw, length_h, _ = block_extent(residual=residual[s], onset=onset)
        row = {
            "site": day["site"],
            "series": s,
            "local_date": day["local_date"],
            "onset_utc": stamp_time(stamp=onset),
            "step_mw": step_mw,
            "block_hours": length_h,
        }
        sign = np.sign(step_mw)
        for w in WINDOWS:
            z = shift_by_window[w][s][onset - SEARCH : onset + SEARCH + 1] / scales[w][s]
            row[f"z_w{w}"] = float(np.nanmax(sign * z)) if np.isfinite(z).any() else float("nan")
        zx = x_shift[s][onset - SEARCH : onset + SEARCH + 1] / x_scale[s]
        row["z_ronc_statistic"] = (
            float(np.nanmax(sign * zx)) if np.isfinite(zx).any() else float("nan")
        )
        neighbours = [n for n in ADJACENCY.get(s, ()) if n in residual]
        metered = [n for n in neighbours if np.isfinite(residual[n][onset - 48 : onset + 48]).any()]
        best = 0.0
        for n in metered:
            z = shift_by_window[8][n][onset - SEARCH : onset + SEARCH + 1] / scales[8][n]
            if np.isfinite(z).any():
                best = max(best, float(np.nanmax(-sign * z)))
        row["n_neighbours"], row["n_metered_neighbours"] = len(neighbours), len(metered)
        row["best_opposite_neighbour_z_w8"] = best
        for name, table in (("ronc_alarm", alarms.filter(pl.col("setting") == "XL")), ("pair", pair)):
            near = table.filter(pl.col("series") == s).with_columns(
                gap=(pl.col("stamp") - onset) / 2
            )
            row[f"nearest_{name}_hours"] = (
                float(near.sort(pl.col("gap").abs())["gap"][0]) if near.height else float("nan")
            )
        rows.append(row)
    table = pl.DataFrame(rows)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    table.write_parquet(WORK_DIR / "why_missed.parquet")
    with pl.Config(tbl_rows=30, tbl_cols=30, tbl_width_chars=300):
        print(table)
        print(table.select(pl.all().exclude("site", "local_date", "onset_utc").median()))


if __name__ == "__main__":
    main()
