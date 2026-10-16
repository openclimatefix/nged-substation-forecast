"""Print the numbers that `result.md` quotes: Tier A and B contrasts, injection, Q3 tables."""

import polars as pl
from e00038_base import SOURCE_DIR, WORK_DIR

KEY = ["contrast", "n_rows", "reference_mean", "arm_mean", "change_pct", "lo", "hi"]


def main() -> None:
    """Print each table the write-up cites."""
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_fmt_str_lengths(60)
    contrasts = pl.read_parquet(WORK_DIR / "tier_a_contrasts.parquet")
    for band in ("lead 4 to 48 h", "lead 144 to 192 h"):
        for view in ("out_of_event", "quiet", "all"):
            print(f"## Tier A, tw, {band}, {view}")
            print(
                contrasts.filter(
                    pl.col("band") == band, pl.col("metric") == "tw", pl.col("view") == view
                ).select(
                    *KEY,
                    "odd_pct",
                    "even_pct",
                    "without_largest_pct",
                    "largest_month",
                    "largest_month_share_of_gain",
                )
            )
    print("## Tier A, mean absolute error (points), 3.5 to 48 h")
    print(
        contrasts.filter(
            pl.col("band") == "lead 4 to 48 h",
            pl.col("metric") == "ae",
            pl.col("view").is_in(["out_of_event", "quiet", "all"]),
        ).select("view", "contrast", "reference_mean", "arm_mean", "change_points", "change_pct")
    )
    events = pl.read_parquet(WORK_DIR / "tier_a_events.parquet")
    print("## long events")
    print(events.select("sheet", "hours", "share", "n_rows", pl.col("^tw_.*$").round(3)))
    print("## Tier B")
    print(
        pl.read_parquet(WORK_DIR / "tier_b_contrasts.parquet").select(
            "view", "metric", *KEY, "without_largest_pct", "largest_month"
        )
    )
    inj = pl.read_parquet(WORK_DIR / "injection_results.parquet")
    print("## injection recall by phi and n_receivers")
    print(
        inj.group_by("variant", "phi", "n_receivers")
        .agg(n=pl.len(), recall=pl.col("hit").mean())
        .sort("variant", "phi", "n_receivers")
    )
    found = inj.filter("hit")
    print("## injection, found events")
    print(
        found.group_by("variant", "phi")
        .agg(
            n=pl.len(),
            iou=pl.col("iou").mean(),
            end_abs_h=pl.col("end_error_hours").abs().median(),
            split_mae=pl.col("split_mae").mean(),
            receivers_matched=pl.col("receivers_matched").mean(),
        )
        .sort("variant", "phi")
    )
    long_found = found.filter(pl.col("hours") >= 48)
    print(
        long_found.group_by("variant").agg(
            n=pl.len(),
            end_within_24h=(pl.col("end_error_hours").abs() <= 24).mean(),
        )
    )
    varying = found.filter(pl.col("varies"), pl.col("true_range") >= 0.3)
    print(varying.group_by("variant").agg(n=pl.len(), flagged=(pl.col("est_range") > 0.15).mean()))
    print("## Q3")
    for name in ("d1", "d1fx"):
        q3 = pl.read_parquet(WORK_DIR / f"q3_table_{name}.parquet")
        print(
            name,
            q3.height,
            "intervals; overlap logged",
            q3["overlaps_logged"].mean(),
            "; alarms",
            q3.filter(pl.col("role") == "alarm").height,
            "; median hours",
            q3["hours"].median(),
            "; share at the 60-day cap",
            (q3["hours"] >= 1440).mean(),
        )
    oracle = pl.read_parquet(SOURCE_DIR / "oracle_real.parquet").filter("primary")
    print(
        "confirmed >= 24 h:",
        oracle.filter("confirmed", pl.col("hours") >= 24).height,
        "of",
        oracle.filter("confirmed").height,
    )
    fractions = pl.read_parquet(WORK_DIR / "split_fractions_d1fx.parquet")
    per_event = fractions.group_by("alarm_series", "onset", "partner").agg(
        spread=pl.col("fraction").max() - pl.col("fraction").min()
    )
    print("split-fraction within-day range, real events:", per_event["spread"].describe())


if __name__ == "__main__":
    main()
