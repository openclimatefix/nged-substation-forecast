"""Summarise experiment 286: recovery, detection against each farm's own noise, minimum drift."""

from pathlib import Path
from typing import Final

import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00286-metered-farm-drift-soiling-injection"
LOW, HIGH = 0.7, 1.3
"""The plan's range for a median recovery (recovered over injected)."""
MIN_YEARS: Final[float] = 4.0


def per_group(
    frame: pl.DataFrame, *, by: list[str], shift: pl.Expr, half_width: pl.Expr
) -> pl.DataFrame:
    """Return per group the median and 10th-to-90th percentile of recovery, and detection rates.

    Pre-registered detection: the injected series' own 90% interval excludes zero. Post hoc
    detection: the shift from the un-injected estimate exceeds the 90% interval's half-width.
    """
    return (
        frame.with_columns(shift=shift, half_width=half_width)
        .group_by(by)
        .agg(
            pl.col("recovery").median().alias("median"),
            pl.col("recovery").quantile(0.1).alias("p10"),
            pl.col("recovery").quantile(0.9).alias("p90"),
            pl.col("ci_excludes_zero").mean().alias("detected_vs_zero"),
            (pl.col("shift").abs() > pl.col("half_width")).mean().alias("detected_vs_baseline"),
            pl.col("half_width").first().alias("half_width"),
            pl.len().alias("n"),
        )
        .with_columns(
            ok=pl.col("median").is_between(LOW, HIGH) & (pl.col("detected_vs_zero") >= 0.5),
            ok_vs_baseline=pl.col("median").is_between(LOW, HIGH)
            & (pl.col("detected_vs_baseline") >= 0.5),
        )
        .sort(by)
    )


def main() -> None:  # noqa: D103
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(20)
    drift = pl.read_csv(DATA_DIR / "drift_injection.csv")
    soil = pl.read_csv(DATA_DIR / "soiling_injection.csv")
    base = pl.read_csv(DATA_DIR / "baseline_trends.csv").with_columns(
        half_width=(pl.col("hi") - pl.col("lo")) / 2
    )
    print(base.select("farm", "method", "years", "months", "trend", "lo", "hi", "half_width"))
    wide_farms = (
        base.filter(pl.col("years") >= MIN_YEARS)
        .group_by("method")
        .agg(
            pl.col("half_width").median().alias("median_half_width"),
            pl.col("half_width").max().alias("max_half_width"),
            pl.len().alias("farms"),
        )
    )
    print(wide_farms)
    d = per_group(
        drift,
        by=["method", "rate", "farm"],
        shift=pl.col("trend") - pl.col("base_trend"),
        half_width=(pl.col("base_hi") - pl.col("base_lo")) / 2,
    )
    print(d)
    print(
        d.group_by("method", "rate")
        .agg(
            pl.col("ok").sum().alias("farms_ok_vs_zero"),
            pl.col("ok_vs_baseline").sum().alias("farms_ok_vs_baseline"),
            pl.col("median").median().alias("median_recovery"),
        )
        .sort("method", "rate")
    )
    s = per_group(
        soil,
        by=["method", "farm"],
        shift=pl.col("beta") - pl.col("base_beta"),
        half_width=(pl.col("base_hi") - pl.col("base_lo")) / 2,
    )
    print(s)
    print(
        s.group_by("method").agg(
            pl.col("ok").sum().alias("farms_ok_vs_zero"),
            pl.col("ok_vs_baseline").sum().alias("farms_ok_vs_baseline"),
        )
    )
    # Minimum detectable drift: the baseline half-width over the shift per percent per year.
    shift_per_rate = (
        drift.with_columns(unit=(pl.col("trend") - pl.col("base_trend")) / pl.col("rate"))
        .group_by("method", "farm")
        .agg(pl.col("unit").median())
    )
    mdd = base.join(shift_per_rate, on=["method", "farm"]).with_columns(
        mdd=pl.col("half_width") / pl.col("unit").abs()
    )
    print(mdd.select("farm", "method", "years", "half_width", "unit", "mdd"))
    d.write_csv(DATA_DIR / "drift_summary.csv")
    s.write_csv(DATA_DIR / "soiling_summary.csv")
    mdd.write_csv(DATA_DIR / "minimum_detectable_drift.csv")


if __name__ == "__main__":
    main()


def extra_tables() -> None:
    """Print the real-series soiling slope and the shift-over-half-width at 1 %/yr."""
    soil = pl.read_csv(DATA_DIR / "soiling_injection.csv")
    print(
        soil.group_by("method", "farm")
        .agg(
            pl.col("base_beta").first(),
            pl.col("base_lo").first(),
            pl.col("base_hi").first(),
            pl.col("truth").median().alias("injected_slope"),
        )
        .sort("method", "farm")
    )
    drift = pl.read_csv(DATA_DIR / "drift_injection.csv").filter(pl.col("rate") == 1.0)
    print(
        drift.with_columns(
            ratio=(pl.col("trend") - pl.col("base_trend")).abs()
            / ((pl.col("base_hi") - pl.col("base_lo")) / 2)
        )
        .group_by("method", "farm")
        .agg(pl.col("ratio").median())
        .sort("method", "farm")
    )


if __name__ == "__main__":
    extra_tables()
