"""Summarise G-skf's pass rows from the harness files `work/skf_gate_<cut>.parquet`.

uv run --no-sync python ml-experiments/00126-*/e00126_skf_summary.py [--cuts 2023-07-01]
"""

import argparse

import e00126_common as common
import polars as pl


def main() -> None:
    """Print false-alarm rates and per-cell recall, delay, and recovery for the plain filter and G-skf."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuts", nargs="+", default=["2023-07-01"])
    arguments = parser.parse_args()
    frame = pl.concat(
        [pl.read_parquet(common.WORK_DIR / f"skf_gate_{cut}.parquet") for cut in arguments.cuts]
    )
    clean = frame.filter(pl.col("cell") == "clean")
    print("== alarms per 100 days on the clean stream ==")
    print(
        clean.group_by("detector").agg(
            (pl.col("alarms").sum() / pl.col("days").sum() * 100).alias("all non-event days"),
            (pl.col("cold_alarms").sum() / pl.col("cold_days").sum() * 100).alias("cold-snap days"),
            (pl.col("holiday_alarms").sum() / pl.col("holiday_days").sum() * 100).alias("holidays"),
        )
    )
    events = frame.filter(pl.col("cell") != "clean")
    table = (
        events.group_by("cell", "amplitude", "detector")
        .agg(
            pl.col("detected").mean().alias("recall"),
            pl.col("delay_hours").median().alias("median delay h"),
            pl.col("recovery_hours").median().alias("median recovery h"),
        )
        .sort("cell", "amplitude", "detector")
    )
    with pl.Config(tbl_rows=60):
        print(table)
    transfers = events.filter(pl.col("cell").str.starts_with("transfer"))
    print("== pooled over the three transfer durations ==")
    with pl.Config(tbl_rows=60):
        print(
            transfers.group_by("amplitude", "detector")
            .agg(
                pl.col("detected").mean().alias("recall"),
                pl.col("delay_hours").median().alias("median delay h"),
                pl.col("recovery_hours").median().alias("median recovery h"),
            )
            .sort("amplitude", "detector")
        )


if __name__ == "__main__":
    main()
