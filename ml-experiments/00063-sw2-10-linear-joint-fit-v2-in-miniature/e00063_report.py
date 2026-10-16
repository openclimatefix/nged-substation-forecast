"""Print every number the write-up of experiment 00063 quotes, and write the tables behind them.

Reads the tables of `e00063_fit.py`, `e00063_inject.py`, and `e00063_real.py` from the results
folder. Errors are in points of capacity: the absolute error in MW divided by the series'
`effective_capacity_mw`, times 100. Every interval is a 95% bootstrap that resamples whole calendar
months within each series and pools the series by rows (50,000 resamples), so a series with more
scored rows weighs more in the pooled mean.

    uv run python ml-experiments/00063-*/e00063_report.py | tee <work>/report.txt
"""

from __future__ import annotations

from typing import Final

import e00063_common as common
import numpy as np
import polars as pl
from studies.bootstrap import BOOTSTRAP_SEED

N_RESAMPLES: Final[int] = 50_000
CHUNK: Final[int] = 5_000
THRESHOLD_POINTS: Final[float] = 0.35
"""The smallest gain worth porting: C1 must lie wholly below minus this for `supported`."""
S1_GATE: Final[float] = 0.15
S2_GATE: Final[float] = 0.05
S2_PLACEBO_GATE: Final[float] = 0.10
SUBSETS: Final[dict[str, str]] = {
    "all daylight": "all",
    "April to September": "summer",
    "October to March": "winter",
}
ERROR_ARMS: Final[tuple[str, ...]] = ("v1", "control", "c1", "c2", "c3", "c4", "c5")
ARM_LABELS: Final[dict[str, str]] = {
    "v1": "frozen v1 baseline (with radiation)",
    "control": "baseline without radiation",
    "c1": "joint fit, season-year constant capacity, fitted scale",
    "c2": "joint fit with the scale fixed at 1",
    "c3": "joint fit, weekly capacity (post hoc pair a=2, r=10)",
    "c4": "joint fit without the transfer terms",
    "c5": "joint fit, previous season-year's constant capacity",
}
CONTRASTS: Final[tuple[tuple[str, str, str, bool], ...]] = (
    ("C1", "c1", "v1", False),
    ("control minus v1 (radiation removed, no solar term)", "control", "v1", False),
    ("c1 minus control (what the solar term buys)", "c1", "control", False),
    ("C2 (fixed scale 1 minus fitted scale)", "c2", "c1", False),
    ("C3 (weekly capacity minus season-year constant)", "c3", "c1", False),
    ("C4 (without transfer terms minus with them)", "c4", "c1", False),
    ("C5 (previous season-year's constant minus in-sample constant)", "c5", "c1", True),
    ("C5 rows only: c1 minus v1", "c1", "v1", True),
    ("C5 rows only: c5 minus v1", "c5", "v1", True),
)
"""(label, minuend arm, subtrahend arm, restricted to the rows where C5 is defined)."""


def pooled_bootstrap(
    *, values: pl.DataFrame, n_resamples: int = N_RESAMPLES
) -> tuple[float, float, float, int, int]:
    """Return the pooled mean, its 95% interval, the row count, and the month count.

    Args:
        values: Columns `series`, `month_id`, and `value` (one row per scored half-hour).
        n_resamples: How many resamples to draw.

    Returns:
        The pooled mean over all rows, the 2.5th and 97.5th percentiles of the resampled pooled
        mean, the number of rows, and the number of series-months.
    """
    per_month = (
        values.group_by("series", "month_id")
        .agg(total=pl.col("value").sum(), n=pl.len())
        .sort("series", "month_id")
    )
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draws_total = np.zeros(n_resamples)
    draws_count = np.zeros(n_resamples)
    for _, group in per_month.group_by("series", maintain_order=True):
        totals = group["total"].to_numpy()
        counts = group["n"].to_numpy().astype(float)
        for start in range(0, n_resamples, CHUNK):
            size = min(CHUNK, n_resamples - start)
            drawn = generator.integers(0, totals.size, size=(size, totals.size))
            draws_total[start : start + size] += totals[drawn].sum(axis=1)
            draws_count[start : start + size] += counts[drawn].sum(axis=1)
    resampled = draws_total / draws_count
    return (
        _mean(values["value"]),
        float(np.percentile(resampled, 2.5)),
        float(np.percentile(resampled, 97.5)),
        values.height,
        per_month.height,
    )


def _points(*, rows: pl.DataFrame, expression: pl.Expr, norms: dict[int, float]) -> pl.DataFrame:
    """Return `series`, `month_id`, and `value` = `expression` in points of capacity."""
    return rows.select(
        "series",
        "month_id",
        value=expression / pl.col("series").replace_strict(norms, return_dtype=pl.Float64) * 100.0,
    )


def _mean(series: pl.Series) -> float:
    """Return the mean of a series as a float."""
    return float(np.mean(series.to_numpy()))


def _median(series: pl.Series) -> float:
    """Return the median of a series as a float."""
    return float(np.median(series.to_numpy()))


def _subset(*, rows: pl.DataFrame, name: str) -> pl.DataFrame:
    if name == "summer":
        return rows.filter(pl.col("summer"))
    if name == "winter":
        return rows.filter(~pl.col("summer"))
    return rows


def _fmt(mean: float, lo: float, hi: float) -> str:
    return f"{mean:+.3f} ({lo:+.3f} to {hi:+.3f})"


def verdict(*, c1_low: float, c1_high: float, s1_holds: bool, s2_holds: bool) -> str:
    """Return the verdict that the plan's decision rule gives, tested in the plan's order."""
    if c1_low > 0.0:
        return "refuted"
    if not (s1_holds and s2_holds):
        return "inconclusive"
    if c1_high < -THRESHOLD_POINTS:
        return "supported"
    if c1_high < 0.0:
        return "small gain"
    if c1_low > -THRESHOLD_POINTS:
        return "refuted"
    return "inconclusive"


def report_rows() -> None:
    """Print the rows-per-series table."""
    print("\n== ROWS PER SERIES ==")
    counts = pl.read_parquet(common.RESULTS_DIR / "row_counts.parquet").with_columns(
        name=pl.col("series").replace_strict(common.SERIES_NAMES, return_dtype=pl.String)
    )
    rows = pl.read_parquet(common.RESULTS_DIR / "c_rows.parquet")
    summer = rows.group_by("series").agg(
        scored_summer=pl.col("summer").sum(),
        scored_c5=pl.col("abs_error_c5").is_not_null().sum(),
        months=pl.col("month_id").n_unique(),
    )
    table = counts.join(summer, on="series", how="left").sort("series")
    with pl.Config(tbl_rows=50, tbl_cols=20, tbl_width_chars=250):
        print(table)
    print(
        f"total scored rows {int(table['scored'].sum())}; first window day "
        f"{common.WINDOW_START.date()}, last {common.WINDOW_END.date()} (exclusive)"
    )
    print("capacity statuses allowed:", common.ALLOWED_STATUSES)
    statuses = (
        pl.read_parquet(common.CAPACITY_PATH)
        .group_by("time_series_id")
        .agg(status=pl.col("status").first())
        .sort("time_series_id")
    )
    print(
        "status of every series:",
        dict(zip(statuses["time_series_id"], statuses["status"], strict=True)),
    )


def report_c(
    *, rows: pl.DataFrame, norms: dict[int, float]
) -> dict[str, tuple[float, float, float]]:
    """Print arms' absolute errors and every contrast; return the C1 interval per subset."""
    print("\n== ABSOLUTE ERROR OF THE MEDIAN, points of capacity (mean absolute error) ==")
    out_rows: list[dict[str, float | str | int]] = []
    for subset_label, subset_name in SUBSETS.items():
        sub = _subset(rows=rows, name=subset_name)
        print(f"-- {subset_label}: {sub.height} rows")
        for arm in ERROR_ARMS:
            data = sub.filter(pl.col(f"abs_error_{arm}").is_not_null())
            values = _points(rows=data, expression=pl.col(f"abs_error_{arm}"), norms=norms)
            mean, lo, hi, n_rows, _ = pooled_bootstrap(values=values)
            print(f"  {arm:8s} {ARM_LABELS[arm]:62s} {mean:7.3f} ({lo:.3f} to {hi:.3f}) n={n_rows}")
            out_rows.append(
                {"subset": subset_label, "arm": arm, "mae": mean, "lo": lo, "hi": hi, "n": n_rows}
            )
    pl.DataFrame(out_rows).write_parquet(common.RESULTS_DIR / "maes.parquet")

    print("\n== CONTRASTS, points of capacity (negative: the first arm has the smaller error) ==")
    contrast_rows: list[dict[str, float | str | int]] = []
    c1_intervals: dict[str, tuple[float, float, float]] = {}
    for label, minuend, subtrahend, c5_only in CONTRASTS:
        print(f"-- {label}")
        for subset_label, subset_name in SUBSETS.items():
            sub = _subset(rows=rows, name=subset_name)
            if c5_only:
                sub = sub.filter(pl.col("abs_error_c5").is_not_null())
            expression = pl.col(f"abs_error_{minuend}") - pl.col(f"abs_error_{subtrahend}")
            values = _points(rows=sub, expression=expression, norms=norms)
            mean, lo, hi, n_rows, n_months = pooled_bootstrap(values=values)
            print(
                f"  {subset_label:22s} {_fmt(mean, lo, hi)}  "
                f"rows {n_rows}, series-months {n_months}"
            )
            contrast_rows.append(
                {
                    "contrast": label,
                    "subset": subset_label,
                    "difference": mean,
                    "lo": lo,
                    "hi": hi,
                    "n_rows": n_rows,
                    "n_series_months": n_months,
                }
            )
            if label == "C1":
                c1_intervals[subset_name] = (mean, lo, hi)
    pl.DataFrame(contrast_rows).write_parquet(common.RESULTS_DIR / "contrasts.parquet")

    print("\n== C1 BY SERIES (points of capacity; v1 and c1 mean absolute error; no interval) ==")
    for series in sorted(rows["series"].unique().to_list()):
        sub = rows.filter(pl.col("series") == series)
        norm = norms[series]
        v1 = _mean(sub["abs_error_v1"]) / norm * 100.0
        c1 = _mean(sub["abs_error_c1"]) / norm * 100.0
        control = _mean(sub["abs_error_control"]) / norm * 100.0
        print(
            f"  {series:3d} {common.SERIES_NAMES[series]:20s} v1 {v1:6.3f}  "
            f"control {control:6.3f}  c1 {c1:6.3f}  C1 {c1 - v1:+.3f}  rows {sub.height}"
        )

    print("\n== PINBALL LOSS, points of capacity (secondary view; difference c1 minus v1) ==")
    for quantile in ("q10", "q90"):
        for arm in ("v1", "control", "c1"):
            values = _points(rows=rows, expression=pl.col(f"pinball_{quantile}_{arm}"), norms=norms)
            print(f"  {quantile} {arm:8s} mean {values['value'].mean():.3f}")
        expression = pl.col(f"pinball_{quantile}_c1") - pl.col(f"pinball_{quantile}_v1")
        mean, lo, hi, _, _ = pooled_bootstrap(
            values=_points(rows=rows, expression=expression, norms=norms)
        )
        print(f"  {quantile} c1 minus v1: {_fmt(mean, lo, hi)}")
    return c1_intervals


def report_folds() -> None:
    """Print the fitted scale k: its sign, spread, and the events in each fit."""
    folds = pl.read_parquet(common.RESULTS_DIR / "c_folds.parquet")
    print("\n== FITTED SCALE k (c1) PER HELD-OUT MONTH ==")
    print(
        f"folds {folds.height}; k median {folds['k_c1'].median():.3f}, "
        f"10th to 90th percentile {folds['k_c1'].quantile(0.1):.3f} to "
        f"{folds['k_c1'].quantile(0.9):.3f}, "
        f"share of folds with k > 0 {(folds['k_c1'] > 0).mean():.3f}"
    )
    by_series = (
        folds.group_by("series")
        .agg(
            k_c1=pl.col("k_c1").median(),
            k_c3=pl.col("k_c3").median(),
            k_c4=pl.col("k_c4").median(),
            k_c5=pl.col("k_c5").median(),
            min_k=pl.col("k_c1").min(),
            max_k=pl.col("k_c1").max(),
            n_events_in_fit=pl.col("n_events_in_fit").median(),
            folds=pl.len(),
        )
        .sort("series")
    )
    with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=250):
        print(by_series)


def report_s1() -> bool:
    """Print S1 and return whether its gate holds."""
    s1 = pl.read_parquet(common.RESULTS_DIR / "s1.parquet").with_columns(
        miss=(pl.col("ratio") - 1.0).abs()
    )
    print("\n== S1: INJECTED SOLAR FARM OUTPUT, recovered capacity / injected capacity ==")
    print(f"injections {s1.filter(pl.col('season') == 'all').height} (series x farm x share)")
    holds = False
    for season in ("all", "summer", "winter"):
        sub = s1.filter(pl.col("season") == season)
        median_miss = _median(sub["miss"])
        print(
            f"  {season:7s} median ratio {sub['ratio'].median():.3f}, 10th to 90th percentile "
            f"{sub['ratio'].quantile(0.1):.3f} to {sub['ratio'].quantile(0.9):.3f}, "
            f"median |ratio - 1| {median_miss:.3f} -> gate "
            f"{'holds' if median_miss <= S1_GATE else 'FAILS'}"
        )
        if season == "all":
            holds = median_miss <= S1_GATE
    by_share = (
        s1.filter(pl.col("season") == "all")
        .group_by("share")
        .agg(median_ratio=pl.col("ratio").median(), median_miss=pl.col("miss").median(), n=pl.len())
        .sort("share")
    )
    by_series = (
        s1.filter(pl.col("season") == "all")
        .group_by("series")
        .agg(median_ratio=pl.col("ratio").median(), median_miss=pl.col("miss").median(), n=pl.len())
        .sort("series")
    )
    by_farm = (
        s1.filter(pl.col("season") == "all")
        .group_by("farm")
        .agg(median_ratio=pl.col("ratio").median(), median_miss=pl.col("miss").median(), n=pl.len())
        .sort("farm")
    )
    with pl.Config(tbl_rows=30, tbl_cols=20):
        print("by share:", by_share, "by series:", by_series, "by farm:", by_farm, sep="\n")
    s1.write_parquet(common.RESULTS_DIR / "s1_with_miss.parquet")
    return holds


def report_s2() -> bool:
    """Print S2 and the placebo intervals, and return whether both gates hold."""
    s2 = pl.read_parquet(common.RESULTS_DIR / "s2.parquet")
    injected = s2.filter(pl.col("kind") == "injected")
    placebo = s2.filter(pl.col("kind") == "placebo")
    failed = s2.filter(pl.col("beta_hat").is_null() | pl.col("beta_hat").is_nan())
    print("\n== S2: INJECTED TRANSFERS (recovered share against injected share) ==")
    print(
        f"injected intervals {injected.height}, placebo intervals {placebo.height}, "
        f"intervals that could not be placed {failed.height}"
    )
    valid = injected.filter(pl.col("beta_hat").is_not_null() & pl.col("beta_hat").is_not_nan())
    valid = valid.with_columns(abs_error=pl.col("error").abs())
    median_error = _median(valid["abs_error"])
    holds_median = median_error <= S2_GATE
    print(
        f"  all: median absolute share error {median_error:.4f} (gate {S2_GATE}) -> "
        f"{'holds' if holds_median else 'FAILS'}; mean error {valid['error'].mean():+.4f}; "
        f"90th percentile of absolute error {valid['abs_error'].quantile(0.9):.4f}"
    )
    with pl.Config(tbl_rows=40, tbl_cols=20):
        for key in ("season", "beta", "duration_hours", "series"):
            table = (
                valid.group_by(key)
                .agg(
                    median_abs_error=pl.col("abs_error").median(),
                    mean_error=pl.col("error").mean(),
                    n=pl.len(),
                )
                .sort(key)
            )
            print(f"by {key}:", table, sep="\n")
        for season in ("summer", "winter"):
            sub = valid.filter(pl.col("season") == season)
            print(
                f"  {season}: median absolute error {sub['abs_error'].median():.4f} "
                f"-> {'holds' if _median(sub['abs_error']) <= S2_GATE else 'FAILS'}"
            )
    placebo_valid = placebo.filter(
        pl.col("beta_hat").is_not_null() & pl.col("beta_hat").is_not_nan()
    )
    absolute = placebo_valid["beta_hat"].abs()
    p95 = float(np.percentile(absolute.to_numpy(), 95))
    holds_placebo = p95 <= S2_PLACEBO_GATE
    print(
        f"  placebo (true share 0): n {placebo_valid.height}, 95th percentile of |share| {p95:.4f} "
        f"(gate {S2_PLACEBO_GATE}) -> {'holds' if holds_placebo else 'FAILS'}; "
        f"median |share| {absolute.median():.4f}; "
        f"mean share {placebo_valid['beta_hat'].mean():+.4f}"
    )
    for season in ("summer", "winter"):
        sub = placebo_valid.filter(pl.col("season") == season)["beta_hat"].abs()
        print(
            f"  placebo {season}: 95th percentile {sub.quantile(0.95):.4f}, "
            f"median {sub.median():.4f}, n {sub.len()}"
        )
    s2.write_parquet(common.RESULTS_DIR / "s2_all.parquet")
    return holds_median and holds_placebo


def report_r1() -> None:
    """Print R1: variance explained inside each real event against shifted placebo windows."""
    pairs = pl.read_parquet(common.RESULTS_DIR / "r1_pairs.parquet")
    skipped = pl.read_parquet(common.RESULTS_DIR / "r1_skipped.parquet")
    placebos = pl.read_parquet(common.RESULTS_DIR / "r1_placebos.parquet")
    n_events = len(common.real_visible_events())
    print("\n== R1: REAL LABELLED EVENTS (descriptive) ==")
    print(f"real visible events with a time span: {n_events}")
    if skipped.height:
        print("skipped (event, reason) counts:")
        with pl.Config(tbl_rows=60, fmt_str_lengths=90, tbl_width_chars=200):
            print(skipped.group_by("reason").len().sort("len", descending=True))
    print(f"event-Primary pairs fitted: {pairs.height}; events: {pairs['card_id'].n_unique()}")
    if pairs.height == 0:
        return
    print(
        f"  event R-squared: median {pairs['r_squared'].median():.3f}, "
        f"mean {pairs['r_squared'].mean():.3f}, 10th to 90th percentile "
        f"{pairs['r_squared'].quantile(0.1):.3f} to {pairs['r_squared'].quantile(0.9):.3f}"
    )
    print(
        f"  placebo R-squared (all windows pooled, n {placebos.height}): median "
        f"{placebos['r_squared'].median():.3f}, mean {placebos['r_squared'].mean():.3f}, "
        f"95th percentile {placebos['r_squared'].quantile(0.95):.3f}"
    )
    print(
        f"  pairs whose R-squared exceeds their own placebo 95th percentile: "
        f"{int(pairs['exceeds_placebo_p95'].sum())} of {pairs.height} "
        f"({pairs['exceeds_placebo_p95'].mean():.3f}; chance at 5%: {0.05 * pairs.height:.1f})"
    )
    print(
        f"  mean rank of the event among its placebos {pairs['placebo_rank'].mean():.3f} "
        f"(0.5 expected with no effect); median share {pairs['share'].median():+.3f}; "
        f"share positive {(pairs['share'] > 0).mean():.3f}"
    )
    print("  by role:")
    print(
        pairs.group_by("role")
        .agg(
            n=pl.len(),
            r_squared=pl.col("r_squared").median(),
            share=pl.col("share").median(),
            exceeds=pl.col("exceeds_placebo_p95").sum(),
        )
        .sort("role")
    )
    with pl.Config(tbl_rows=80, tbl_cols=20, tbl_width_chars=250):
        print(pairs.sort("r_squared", descending=True))


def main() -> None:
    """Print the whole report."""
    print(f"EXPERIMENT 00063 REPORT. Results folder: {common.RESULTS_DIR}")
    norms = common.effective_capacity_mw()
    print(
        "effective_capacity_mw (MW, the normaliser):",
        {s: round(norms[s], 3) for s in common.SERIES_IDS},
    )
    report_rows()
    rows = pl.read_parquet(common.RESULTS_DIR / "c_rows.parquet")
    c1 = report_c(rows=rows, norms=norms)
    report_folds()
    s1_holds = report_s1()
    s2_holds = report_s2()
    report_r1()
    mean, lo, hi = c1["all"]
    print("\n== VERDICT BY THE PLAN'S DECISION RULE ==")
    print(
        f"C1 {_fmt(mean, lo, hi)}; S1 {'holds' if s1_holds else 'fails'}; S2 "
        f"{'holds' if s2_holds else 'fails'}; threshold {THRESHOLD_POINTS} points -> "
        f"{verdict(c1_low=lo, c1_high=hi, s1_holds=s1_holds, s2_holds=s2_holds)}"
    )


if __name__ == "__main__":
    main()
