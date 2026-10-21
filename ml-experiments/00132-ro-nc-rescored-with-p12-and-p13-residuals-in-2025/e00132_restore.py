r"""Re-read the zero readings that `substation_zero` flags inside the six transfers at P12 and P13.

For each of the six labelled transfers over 1 day at P12 and P13, every half-hour within
`RESTORE_PAD` of the labelled window that the raw `power_time_series` table holds and the
`cleaned_power_time_series` table flags is written, unchanged, to the private `restored.parquet`.
No value is interpolated. The raw store and the cleaned table are only read. The script also prints,
per window and series, the half-hours present in the raw table, flagged by cleaning, null in
experiment 6's `features.parquet`, and restored.

    systemd-run --user --scope -p MemoryMax=4G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00132-*/e00132_restore.py
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Final

import polars as pl
from e00132_common import BLOCKED_SERIES, FEATURES_PATH, GRID_START, RESTORE_PAD, WORK_DIR
from e00132_fit import transfer_windows
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NGED_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR.parent / "NGED"
"""The data folder's NGED tables; this script reads them and never writes them."""
E00101_EVENTS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00101-ro-nc-detector-on-real-hand-filtered-switching-events"
    / "event_table.parquet"
)


def main() -> None:
    """Write `restored.parquet` and print the per-window counts."""
    raw = (
        pl.scan_delta(str(NGED_DIR / "power_time_series.delta"))
        .filter(pl.col("time_series_id").is_in(BLOCKED_SERIES))
        .collect()
    )
    cleaned = (
        pl.scan_delta(str(NGED_DIR / "cleaned_power_time_series.delta"))
        .filter(pl.col("time_series_id").is_in(BLOCKED_SERIES))
        .collect()
    )
    features = (
        pl.scan_parquet(FEATURES_PATH)
        .filter(pl.col("time_series_id").is_in(BLOCKED_SERIES))
        .select("time_series_id", "valid_time", "target")
        .collect()
    )
    events = pl.read_parquet(E00101_EVENTS).filter(
        pl.col("series") == "12,13", pl.col("hours") > 24
    )
    flagged = cleaned.filter(pl.col("drop_reason") == "substation_zero")
    pieces, rows = [], []
    for event, (start, end) in zip(
        events.sort("onset")["event"], sorted(transfer_windows()), strict=True
    ):
        for series in BLOCKED_SERIES:
            in_window = (pl.col("time") >= start) & (pl.col("time") < end)
            raw_rows = raw.filter(pl.col("time_series_id") == series).filter(in_window)
            flagged_rows = flagged.filter(pl.col("time_series_id") == series).filter(in_window)
            null_target = features.filter(
                pl.col("time_series_id") == series,
                pl.col("valid_time") >= start,
                pl.col("valid_time") < end,
                pl.col("target").is_null(),
            ).height
            restored = raw_rows.join(
                flagged_rows.select("time_series_id", "time"), on=["time_series_id", "time"]
            ).with_columns(event=pl.lit(event))
            pieces.append(restored)
            rows.append(
                {
                    "event": event,
                    "series": series,
                    "window_half_hours": int((end - start) / timedelta(minutes=30)),
                    "in_raw": raw_rows.height,
                    "flagged_zero": flagged_rows.height,
                    "null_in_features": null_target,
                    "restored": restored.height,
                    "restored_max_abs_mw": float(restored["power"].abs().max() or 0.0),
                }
            )
    out = (
        pl.concat(pieces)
        .unique(["time_series_id", "time"])
        .with_columns(
            stamp=(pl.col("time").dt.epoch("s") - int(GRID_START.timestamp())) // 1800,
            power=pl.col("power").cast(pl.Float64),
            time_series_id=pl.col("time_series_id").cast(pl.Int64),
        )
        .sort("time_series_id", "stamp")
    )
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    out.write_parquet(WORK_DIR / "restored.parquet")
    pl.Config.set_tbl_rows(40)
    table = pl.DataFrame(rows)
    table.write_csv(WORK_DIR / "restore_counts.csv")
    print(table)
    print(
        f"restored {out.height} half-hours in total; any nonzero value: {(out['power'] != 0).sum()}"
    )
    print(f"half-hours of the window pad: {RESTORE_PAD}")


if __name__ == "__main__":
    main()
