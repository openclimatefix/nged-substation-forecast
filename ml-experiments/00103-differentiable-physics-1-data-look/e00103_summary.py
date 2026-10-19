"""Apply experiment 103's remaining decision rules to the saved tables.

Adds the results to decisions.json.

Prints the clip decision per site, the timestamp peak per site and side, D18's negative dark rows by
month, and the long-history counts at each tuning origin. The generators print as G-a and G-b.

    POLARS_MAX_THREADS=4 uv run python ml-experiments/00103-*/e00103_summary.py
"""

from __future__ import annotations

import json

import polars as pl
from e00103_common import MW_PRIMARIES, WORK_DIR, load_series
from e00103_stats import load_panel

TABLES = WORK_DIR / "tables"


def clip_decisions(*, sites: dict) -> pl.DataFrame:
    """Clip yes when, in 2 or more summers, the slope above 600 exceeds -1 and below 500 is < -4."""
    clip = pl.read_parquet(TABLES / "clip.parquet").with_columns(
        site=pl.col("time_series_id").replace_strict(sites, return_dtype=pl.String),
        clipped=(pl.col("above_600") > -1) & (pl.col("below_500") < -4),
    )
    return (
        clip.filter(pl.col("site").is_in(MW_PRIMARIES))
        .group_by("site")
        .agg(clipped_summers=pl.col("clipped").sum(), summers=pl.len())
        .with_columns(clip=pl.col("clipped_summers") >= 2)
        .sort("site")
    )


def timestamp_peaks(*, sites: dict) -> pl.DataFrame:
    """The most negative shift per MW Primary and side, and the change at plus or minus 30."""
    table = pl.read_parquet(TABLES / "timestamp.parquet").with_columns(
        site=pl.col("time_series_id").replace_strict(sites, return_dtype=pl.String)
    )
    rows = []
    for (site, after), group in table.group_by("site", "after_fix"):
        part = group.sort("shift_minutes")
        best = part.sort("correlation").row(0, named=True)
        by_shift = dict(part.select("shift_minutes", "correlation").iter_rows())
        rows.append(
            {
                "site": site,
                "after_fix": after,
                "best_shift": best["shift_minutes"],
                "corr_at_0": by_shift[0],
                "change_to_minus_30": by_shift[-30] - by_shift[0],
                "change_to_plus_30": by_shift[30] - by_shift[0],
            }
        )
    return pl.DataFrame(rows).sort("site", "after_fix")


def main() -> None:
    """Print and save the decisions."""
    series = load_series()
    sites = dict(series.select("time_series_id", "site").iter_rows())
    ids = {s: i for i, s in sites.items()}
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(220)
    clip = clip_decisions(sites=sites)
    print("== clip decision ==")
    print(clip)
    peaks = timestamp_peaks(sites=sites)
    print("== timestamp peaks ==")
    print(
        peaks.filter(pl.col("site").is_in(MW_PRIMARIES)).with_columns(pl.col(pl.Float64).round(4))
    )
    means = (
        pl.read_parquet(TABLES / "timestamp.parquet")
        .filter(pl.col("time_series_id").is_in([ids[s] for s in MW_PRIMARIES]))
        .group_by("after_fix", "shift_minutes")
        .agg(pl.col("correlation").mean())
    )
    print(
        "== timestamp: vertex of the parabola through the mean correlation at -30, 0, +30 minutes "
        "=="
    )
    for after in (False, True):
        part = means.filter(pl.col("after_fix") == after)
        c = {r["shift_minutes"]: r["correlation"] for r in part.iter_rows(named=True)}
        vertex = 30 * (c[-30] - c[30]) / (2 * (c[-30] - 2 * c[0] + c[30]))
        print(
            f"after_fix={after}: vertex at {vertex:+.1f} minutes; mean correlation at -30/0/+30 ="
            f" {c[-30]:.3f}/{c[0]:.3f}/{c[30]:.3f}"
        )
    panel = load_panel()
    d18 = panel.filter(
        pl.col("time_series_id") == ids["D18"],
        pl.col("clean"),
        pl.col("dark"),
        ~pl.col("switching_mask"),
    )
    by_month = (
        d18.group_by(pl.col("local_date").dt.truncate("1mo").alias("month"))
        .agg(dark=pl.len(), negative=(pl.col("power") < 0).sum())
        .sort("month")
    )
    print("== D18 negative dark rows by month ==")
    print(by_month.filter(pl.col("negative") > 0))
    after = d18.filter(pl.col("local_date") >= pl.date(2025, 2, 1))
    share_after = (after["power"] < 0).sum() / after.height
    print(f"D18 negative dark share from 2025-02-01: {share_after:.4%} ({after.height} dark rows)")
    sites_table = pl.read_parquet(WORK_DIR / "sites.parquet")
    long_history = {
        col: int(((sites_table["role"].is_in(["mw_long"])) & (sites_table[col] >= 12)).sum())
        for col in sites_table.columns
        if col.startswith("months_before")
    }
    print("long-history sites holding 12+ months before each origin:", long_history)
    decisions = json.loads((WORK_DIR / "decisions.json").read_text())
    decisions["clip"] = {r["site"]: r["clip"] for r in clip.iter_rows(named=True)}
    decisions["timestamp_best_shift_minutes"] = {
        f"{r['site']}_{'after' if r['after_fix'] else 'before'}": r["best_shift"]
        for r in peaks.filter(pl.col("site").is_in(MW_PRIMARIES)).iter_rows(named=True)
    }
    decisions["d18_negative_dark_share_from_2025_02"] = share_after
    decisions["long_history_counts_by_origin"] = long_history
    (WORK_DIR / "decisions.json").write_text(json.dumps(decisions, indent=2, default=str))


if __name__ == "__main__":
    main()
