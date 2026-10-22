"""Part 1, step 1: why neither RO-NC nor the pair alarm fired on the 17 block days.

For each block day, reads the arm B residual panel (RO-NC's input) and reports the largest block
score of the day (the block's length and size in noise units), the largest step at four window
lengths, RO-NC's own 24-hour step statistic against its threshold, whether a neighbour stepped the
other way, and the nearest stored RO-NC and pair alarms. Nothing is fitted.

    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00181-*/e00181_why_missed.py
"""

from __future__ import annotations

from typing import Final

import numpy as np
import polars as pl
from e00181_block import LENGTHS, best_z, series_scores
from e00181_blocks17 import block_days, stamp_time
from e00181_common import ADJACENCY, ML_EXPERIMENTS_DATA_DIR, WORK_DIR
from e00181_detectors import level_shift, robust_scale
from e00181_variants import load_both, scales_for, shifts

RUN_130_B: Final = ML_EXPERIMENTS_DATA_DIR / "00130-ro-nc-detector-on-the-gam-k-residual" / "B"
MARGIN: Final[int] = 6
"""Stamps (3 hours) added either side of the local day, as in experiment 175."""
STEP_WINDOWS: Final[tuple[int, ...]] = (8, 12, 24, 48)
"""Half-hours in the two windows of a step statistic: 4, 6, 12, and 24 hours."""
X_THRESHOLD: Final[float] = 4.35
"""The threshold of RO-NC's refit-offset step statistic (experiment 130, arm B)."""


def main() -> None:
    """Print and write one row per block day."""
    panel, offset_panel = load_both()
    residual = {s: v for s, v in panel["residual"].items() if np.isfinite(v).any()}
    block_scores = {s: series_scores(values=v)[0] for s, v in residual.items()}
    scales = {
        w: {s: robust_scale(shift=level_shift(values=v, window=w)) for s, v in residual.items()}
        for w in STEP_WINDOWS
    }
    shift_by_window = {
        w: {s: level_shift(values=v, window=w) for s, v in residual.items()} for w in STEP_WINDOWS
    }
    x_scale = scales_for(variant="X", residual=panel["residual"], offset=offset_panel["residual"])
    x_shift = shifts(variant="X", residual=panel["residual"], offset=offset_panel["residual"])
    alarms = pl.read_parquet(RUN_130_B / "alarms.parquet").filter(pl.col("setting") == "XL")
    pair = pl.read_parquet(
        ML_EXPERIMENTS_DATA_DIR
        / "00156-switching-faster-onset-alarm-from-opposite-steps"
        / "detect"
        / "real_W8_alarms.parquet"
    )
    rows = []
    for day in block_days().iter_rows(named=True):
        s = day["series"]
        lo, hi = day["day_first_stamp"] - MARGIN, day["day_last_stamp"] + MARGIN + 1
        z, length = best_z(scores=block_scores[s])
        pick = lo + int(np.argmax(np.abs(z[lo:hi])))
        sign = float(np.sign(z[pick]))
        row = {
            "site": day["site"],
            "series": s,
            "local_date": day["local_date"],
            "block_start_utc": stamp_time(stamp=pick),
            "block_hours": int(length[pick]) / 2,
            "block_z": abs(float(z[pick])),
            "block_sign": sign,
        }
        row["sign_agrees_with_experiment_175"] = bool(sign == np.sign(day["edge_shift"]))
        for w in STEP_WINDOWS:
            zw = sign * shift_by_window[w][s][lo:hi] / scales[w][s]
            row[f"step_z_{w // 2}h_window"] = (
                float(np.nanmax(np.abs(zw))) if np.isfinite(zw).any() else float("nan")
            )
        zx = np.abs(x_shift[s][lo:hi] / x_scale[s])
        row["ronc_statistic_z"] = float(np.nanmax(zx)) if np.isfinite(zx).any() else float("nan")
        neighbours = [n for n in ADJACENCY.get(s, ()) if n in residual]
        metered = [n for n in neighbours if np.isfinite(residual[n][lo:hi]).sum() > 24]
        best = 0.0
        for n in metered:
            segment = np.concatenate(
                [-sign * block_scores[n][length_][lo:hi] for length_ in LENGTHS]
            )
            if np.isfinite(segment).any():
                best = max(best, float(np.nanmax(segment)))
        row["n_neighbours"], row["n_metered_neighbours"] = len(neighbours), len(metered)
        row["best_opposite_neighbour_block_z"] = best
        for name, table in (("ronc", alarms), ("pair", pair)):
            near = table.filter(pl.col("series") == s, pl.col("stamp").is_between(lo, hi - 1))
            row[f"{name}_alarms_in_day"] = near.height
        rows.append(row)
    table = pl.DataFrame(rows)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    table.write_parquet(WORK_DIR / "why_missed.parquet")
    with pl.Config(tbl_rows=30, tbl_cols=30, tbl_width_chars=320):
        print(
            table.select(
                "site",
                "local_date",
                "block_hours",
                "block_z",
                "sign_agrees_with_experiment_175",
                "step_z_4h_window",
                "step_z_24h_window",
                "ronc_statistic_z",
                "n_metered_neighbours",
                "best_opposite_neighbour_block_z",
                "ronc_alarms_in_day",
                "pair_alarms_in_day",
            )
        )
        print(
            table.select(
                "block_hours",
                "block_z",
                "step_z_4h_window",
                "step_z_24h_window",
                "ronc_statistic_z",
                "best_opposite_neighbour_block_z",
            ).median()
        )


if __name__ == "__main__":
    main()
