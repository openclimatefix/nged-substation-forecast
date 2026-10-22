# ruff: noqa: E501
"""Experiment 191: the compact tables the write-up quotes, printed from the stored predictions.

uv run --no-sync python ml-experiments/00191-*/e00191_report.py
"""

import json

import e00191_common as common
import e00191_score as score
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ARMS = ("a0", "h", "m", "p", "t", "jb")


def contrast_table(*, variant: str, sites: tuple[str, ...], cut: str) -> None:
    """Print mean absolute error and quantile score contrasts against a0 as percentages."""
    report = json.loads((common.WORK_DIR / f"score_{variant}.json").read_text())
    print(f"\n[{variant}] cut={cut}: difference from a0 as % of a0, with 95% interval")
    for site in sites:
        row = report[site][cut]
        base = row["a0"]
        print(f"{site}: a0 mae {base['ae']:.3f} qs {base['qs']:.3f} cov90 {base['cov90']:.2f}")
        for arm in ARMS[1:]:
            ae, qs = row[f"{arm}_minus_a0_ae"], row[f"{arm}_minus_a0_qs"]
            print(
                f"   {arm:2s} mae {ae['pct']:+6.2f}% [{ae['low'] / base['ae'] * 100:+6.2f},"
                f" {ae['high'] / base['ae'] * 100:+6.2f}]   qs {qs['pct']:+6.2f}% "
                f"[{qs['low'] / base['qs'] * 100:+6.2f}, {qs['high'] / base['qs'] * 100:+6.2f}]"
            )


def tails(*, variant: str, sites: tuple[str, ...]) -> None:
    """Print tail pinball loss and the share of readings below each tail quantile, a0 and h."""
    report = json.loads((common.WORK_DIR / f"score_{variant}.json").read_text())
    print(f"\n[{variant}] tail calibration, all rows (pinball loss; share of readings below q)")
    for site in sites:
        for arm in ("a0", "h"):
            r = report[site]["all"][arm]
            print(
                f"{site} {arm}: "
                + "  ".join(
                    f"{q}: pin {r[f'pin_{q}']:.3f}, below {r[f'below_{q}']:.3f}"
                    for q in ("q01", "q05", "q95", "q99")
                )
            )


def median_minus_mean(*, variant: str, sites: tuple[str, ...]) -> None:
    """Print the mean of (q50 minus mean forecast) per site and arm, in points."""
    print(f"\n[{variant}] median forecast minus mean forecast, points")
    for site in sites:
        frame = score.load(variant=variant, site=site)
        by_arm = frame.group_by("arm").agg((pl.col("q50") - pl.col("mean")).mean().alias("d"))
        print(site, {r["arm"]: round(r["d"], 3) for r in by_arm.sort("arm").iter_rows(named=True)})


def against_experiment_155() -> None:
    """Compare a0 at D14 with experiment 155's arm D and 136's A2 on the rows they share."""
    mine = score.losses(frame=score.load(variant="gated", site="D14")).filter(pl.col("arm") == "a0")
    for name, path in (
        ("155 arm D", ML_EXPERIMENTS_DATA_DIR / "00155-dp-bg-stage2-assembly-night-offset-and-error-correction/work/dp_preds/d_scored_D14.parquet"),
        ("136 arm A2", ML_EXPERIMENTS_DATA_DIR / "00136-dp-bg-stage1-assembly-on-the-ensemble-mean/work/dp_preds/a2_scored_D14.parquet"),
    ):  # fmt: skip
        other = pl.read_parquet(path).with_columns(
            month=pl.col("valid_time").dt.month() + 100 * pl.col("valid_time").dt.year(),
            arm=pl.lit("other"),
            gate=pl.lit(0.0),
            spread_positive=pl.lit(False),
        )
        other = other.filter(
            (pl.col("lead_hours") > 0) & (pl.col("lead_hours") <= 168) & ~pl.col("switching_mask")
        )
        other = score.losses(frame=other)
        joined = mine.select("run", "valid_time", a_qs="qs", a_ae="ae").join(
            other.select("run", "valid_time", o_qs="qs", o_ae="ae"), on=["run", "valid_time"]
        )
        print(
            f"\nD14, rows shared with {name}: {joined.height:,}; a0 qs {joined['a_qs'].mean():.4f} "
            f"mae {joined['a_ae'].mean():.4f}; {name} qs {joined['o_qs'].mean():.4f} "
            f"mae {joined['o_ae'].mean():.4f}"
        )


def main() -> None:
    """Print every table."""
    for variant, sites in (("gated", common.ALL_SITES), ("ungated", common.ALL_SITES)):
        for cut in ("all", "high_spread"):
            contrast_table(variant=variant, sites=sites, cut=cut)
    contrast_table(variant="gated_daily", sites=common.ASSET_SITES, cut="all")
    for cut in ("0-24 h", "24-72 h", "72-168 h"):
        contrast_table(variant="gated", sites=common.ASSET_SITES, cut=cut)
    tails(variant="gated", sites=common.ASSET_SITES)
    median_minus_mean(variant="gated", sites=common.ALL_SITES)
    against_experiment_155()
    shares = []
    for site in common.ASSET_SITES:
        frame = score.load(variant="gated", site=site).filter(pl.col("arm") == "a0")
        shares.append(
            f"{site}: months in the scored rows {frame['month'].n_unique()}, "
            f"spread-positive share {frame['spread_positive'].mean():.2f}"
        )
    print("\n" + "\n".join(shares))
    print("Largest absolute q50-mean gap shown above; numpy", np.__version__)


if __name__ == "__main__":
    main()
