"""Rank arm D's days by excess loss and print the error budget (experiment 172).

Run it under `systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue` with
`uv run --no-sync python` and write the printed report to `printed_report.txt`.
"""

import e00172_common as common
import e00172_days as days_module
import e00172_stats as stats
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache

pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(220)
SITES = common.DECISION_SITES
LABELLED = pl.col("label") == "labelled"


def build(
    *, lead_band: tuple[float, float], drop_masked: bool, cache: FeatureCache
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the day table and the rows for one lead band and masking choice."""
    ids = common.series_ids()
    rows = common.read_rows(lead_band=lead_band, drop_masked=drop_masked)
    rows = common.weather_and_calendar(rows=rows, cache=cache)
    days = days_module.build_days(
        rows=rows,
        switching=common.switching_days(series_of_site=ids),
        flagged=common.flagged_days(series_of_site=ids),
    )
    return days, rows


def fmt(*, draws: np.ndarray) -> str:
    """Return a bootstrap interval as text."""
    low, high = stats.interval(draws=draws)
    return f"95% interval {low:.3f} to {high:.3f}"


def report_headline(*, days: pl.DataFrame, worst: pl.DataFrame, boot: dict) -> None:
    """Print the pooled share, its per-site and leave-one-out versions, and the labels."""
    share = stats.pooled_share(days=days, sites=SITES)
    print(
        f"\n== HEADLINE: worst 5% of days carry {share:.3f} of the excess "
        f"({fmt(draws=boot['share'])})"
    )
    for site in common.SITES:
        print(f"  {site}: {stats.pooled_share(days=days, sites=(site,)):.3f}")
    for site in SITES:
        rest = tuple(s for s in SITES if s != site)
        print(f"  leave out {site}: {stats.pooled_share(days=days, sites=rest):.3f}")
    arrays = [days.filter(pl.col("site") == s)["excess"].to_numpy() for s in SITES]
    print("  worst 10%:", f"{stats.share_of_worst(excess_by_site=arrays, fraction=0.10):.3f}")
    table = stats.label_table(days=days, worst=worst, sites=SITES)
    table.write_parquet(common.WORK_DIR / "label_table.parquet")
    print("\n== Labels (decision sites)\n", table)
    print(worst.group_by("site").agg(n=pl.len(), labelled=pl.col("labelled").mean()).sort("site"))
    unlabelled = worst.filter(pl.col("site").is_in(SITES), ~pl.col("labelled"))
    print("== Unlabelled worst days by local month")
    print(unlabelled.group_by("month").len().sort("month"))


def report_cold(*, days: pl.DataFrame, rows: pl.DataFrame, worst: pl.DataFrame, rng) -> None:  # noqa: ANN001
    """Print the cold-night questions at the pre-registered and post hoc thresholds."""
    print("\n== Cold nights")
    for threshold in (common.COLD_DAY_C, 3.0, 5.0):
        blocks = stats.month_blocks(days=days, rows=rows, sites=SITES, cold_c=threshold)
        parts = [blocks[m][s] for m in blocks for s in SITES]
        boot = stats.bootstrap(blocks=blocks, sites=SITES, rng=rng)
        scope = days.filter(pl.col("site").is_in(SITES))
        n_cold = int((scope["day_temperature_c"] < threshold).sum())
        n_worst = int(
            (worst.filter(pl.col("site").is_in(SITES))["day_temperature_c"] < threshold).sum()
        )
        tag = "pre-registered" if threshold == common.COLD_DAY_C else "post hoc"
        print(
            f"-- day mean below {threshold} C ({tag}): {n_cold} of {scope.height} days, "
            f"{n_worst} of the worst days"
        )
        for name, value in (
            ("row-level ratio (pre-registered)", stats.row_ratio(parts=parts)),
            ("day-level ratio (post hoc)", stats.day_ratio(parts=parts)),
            ("night score ratio (post hoc)", stats.night_ratio(parts=parts)),
        ):
            key = {"row": "row_ratio", "day": "day_ratio", "nig": "night_ratio"}[name[:3]]
            print(f"   {name}: {value:.3f} ({fmt(draws=boot[key])})")


def report_controls(*, days: pl.DataFrame, rng) -> None:  # noqa: ANN001
    """Print the random-day and injected-excess controls."""
    print("\n== Random-day control", stats.random_control(days=days, sites=SITES, rng=rng))
    recovered = stats.injected_control(days=days, sites=SITES, rng=rng)
    print(
        f"== Injected control, pooled: {recovered['pooled']:.3f} of injected days in the worst 5%"
    )
    print("   by site:", {k: round(v, 3) for k, v in recovered.items() if k != "pooled"})
    larger = stats.injected_control(days=days, sites=SITES, rng=rng, points=10.0)
    print("== Injected control, 10 points (post hoc):", {k: round(v, 3) for k, v in larger.items()})


def report_checks(*, days: pl.DataFrame, rows: pl.DataFrame, worst: pl.DataFrame) -> None:
    """Print the median-minus-mean forecast and the ranking by the mean forecast's error."""
    print("\n== Checks")
    print(
        rows.group_by("site")
        .agg(
            median_minus_mean=(pl.col("q50") - pl.col("mean")).mean(),
            mean_abs_error_of_mean=(pl.col("actual") - pl.col("mean")).abs().mean(),
            mean_abs_error_of_median=(pl.col("actual") - pl.col("q50")).abs().mean(),
            max_abs_median_minus_mean=(pl.col("q50") - pl.col("mean")).abs().max(),
            first_day=pl.col("local_date").min(),
            last_day=pl.col("local_date").max(),
        )
        .sort("site")
    )
    by_error = days.with_columns(
        excess=pl.col("mean_abs_error") - pl.col("mean_abs_error").median().over("site")
    )
    share = stats.pooled_share(days=by_error, sites=SITES)
    print(f"share of the mean forecast's absolute error carried by its worst 5%: {share:.3f}")
    keys = ["site", "local_date"]
    both = worst.select(keys).join(days_module.worst_days(days=by_error).select(keys), on=keys)
    print(
        f"worst pinball days also among the worst mean-error days: {both.height} of {worst.height}"
    )


def report_sensitivities(*, cache: FeatureCache) -> None:
    """Print the share and the labelled fraction for the alternative designs."""
    print("\n== Sensitivities")
    for name, band, masked in (
        ("all leads 0 to 168 h", (0.0, 168.0), True),
        ("D+1, switching rows kept", common.D1_LEAD_HOURS, False),
    ):
        days, _ = build(lead_band=band, drop_masked=masked, cache=cache)
        worst = days_module.worst_days(days=days)
        table = stats.label_table(days=days, worst=worst, sites=SITES)
        labelled = table.filter(LABELLED)["share_of_worst_days"][0]
        share = stats.pooled_share(days=days, sites=SITES)
        print(f"-- {name}: share {share:.3f}; worst days labelled {labelled:.3f}")


def main() -> None:
    """Run the analysis, write the tables, and print the report."""
    rng = np.random.default_rng(common.SEED)
    cache = common.load_cache()
    days, rows = build(lead_band=common.D1_LEAD_HOURS, drop_masked=True, cache=cache)
    worst = days_module.worst_days(days=days)
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    days.write_parquet(common.WORK_DIR / "days.parquet")
    worst.sort("excess", descending=True).write_parquet(common.WORK_DIR / "worst_days.parquet")
    print("== Days per site (at least 24 rows)")
    print(
        days.group_by("site")
        .agg(
            n_days=pl.len(),
            median_score=pl.col("site_median").first(),
            mean_score=pl.col("score").mean(),
        )
        .sort("site")
    )
    blocks = stats.month_blocks(days=days, rows=rows, sites=SITES, cold_c=common.COLD_DAY_C)
    boot = stats.bootstrap(blocks=blocks, sites=SITES, rng=rng)
    report_headline(days=days, worst=worst, boot=boot)
    report_cold(days=days, rows=rows, worst=worst, rng=rng)
    report_controls(days=days, rng=rng)
    for site in (None, *common.SITES):
        sites = SITES if site is None else (site,)
        frame = stats.curve(days=days, sites=sites)
        frame.write_parquet(common.WORK_DIR / f"curve_{site or 'pooled'}.parquet")
    pooled = pl.read_parquet(common.WORK_DIR / "curve_pooled.parquet")
    at = [
        round(
            float(
                pooled.filter((pl.col("fraction_of_days") - f).abs() < 1e-9)["share_of_excess"][0]
            ),
            3,
        )
        for f in (0.05, 0.10, 0.25, 0.50)
    ]
    print("\n== Pooled curve at 5%, 10%, 25%, 50% of days:", at)
    print("\n== Worst days")
    print(
        worst.sort("excess", descending=True).select(
            "site",
            "local_date",
            "score",
            "excess",
            *common.LABELS,
            "day_temperature_c",
            "step_share",
        )
    )
    report_checks(days=days, rows=rows, worst=worst)
    report_sensitivities(cache=cache)


if __name__ == "__main__":
    main()
