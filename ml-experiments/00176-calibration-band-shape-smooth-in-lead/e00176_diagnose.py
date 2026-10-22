"""Print D5's spread table and the median q05 to q95 width by lead day for every substation.

Reads experiment 169's arm A and its reference (experiment 160's arm I). Shows the widths on rows
outside the switching events and on all rows, binned by `ceil(lead / 24)` and by `floor(lead / 24)`,
so a reader sees which binning and which row set produce a dip or a jump.

    uv run --no-sync python ml-experiments/00176-*/e00176_diagnose.py
"""

import e00176_common as common
import numpy as np
import polars as pl

np.set_printoptions(precision=1, suppress=True, linewidth=160)


def day_widths(*, frame: pl.DataFrame, floor: bool) -> list[float]:
    """Return the median q05 to q95 width on each lead day."""
    lead = frame["lead_hours"].to_numpy()
    day = np.floor(lead / 24.0).astype(int) if floor else common.lead_day(lead_hours=lead)
    width = (frame["q95"] - frame["q05"]).to_numpy()
    return [float(np.median(width[day == d])) for d in np.unique(day)]


def main() -> None:
    """Print the diagnosis."""
    table = np.load(common.table_path(site="D5"))
    width = table[..., 9] - table[..., 1]
    print(
        "D5 stored q05-q95 width by (hour band, quarter); lead bands 0-24, 24-48, 48-144, 144-168 h"
    )
    for band in range(common.N_HOUR_BANDS):
        for quarter in range(common.N_QUARTERS):
            print(f"  band {band} quarter {quarter}", width[band, quarter])
    for period in ("scored",):
        for site in common.SCORED_SITES:
            for arm in ("ref", "a"):
                frame = pl.read_parquet(common.stored_path(arm=arm, period=period, site=site))
                frame = frame.filter(pl.col("lead_hours") <= common.HORIZON_HOURS)
                kept = frame.filter(~pl.col("switching_mask"))
                print(
                    f"{site} {arm:3s} {period} outside switching events, ceil:",
                    np.array(day_widths(frame=kept, floor=False)),
                )
                print(
                    f"{site} {arm:3s} {period} all rows, ceil:                ",
                    np.array(day_widths(frame=frame, floor=False)),
                )
                print(
                    f"{site} {arm:3s} {period} all rows, floor (8 days):      ",
                    np.array(day_widths(frame=frame, floor=True)),
                )
    frame = pl.read_parquet(common.stored_path(arm="a", period="scored", site="D5"))
    exact = frame.filter(pl.col("lead_hours") == common.HORIZON_HOURS)
    print(
        "D5 rows at lead exactly 168.0 h:",
        exact.height,
        "hour bands:",
        exact["hour_band"].unique().to_list(),
    )
    print(
        "D5 masked share by hour band:",
        frame.group_by("hour_band")
        .agg(pl.col("switching_mask").mean())
        .sort("hour_band")
        .to_dicts(),
    )


if __name__ == "__main__":
    main()
