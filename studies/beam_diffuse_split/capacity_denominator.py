"""Explain the clearest week's per-site bias, and test what a different capacity denominator does.

Two panels of the clearest-week figure prompt the question. Site A's models overpredict it and site
B's underpredict it, and the natural suspicion is that normalising by each site's 99th percentile of
metered output sets the wrong scale. This prints what the bias actually is, where in the output
range it lives, and what the headline contrast becomes under the 99.9th percentile and the highest
reading instead.

Run it from this directory as:

```bash
uv run python capacity_denominator.py --source cams
```
"""

import argparse
import pathlib
import sys
from typing import Final

import polars as pl
from commissioning import drop_commissioning_ramp
from run_experiment import (
    _add_time_features,
    _bootstrap_difference,
    dataset_path_for,
)
from sources import SOURCE_CHOICES, STUDY_DATA_DIR

CLEAREST_WEEK: Final[str] = "2026-04-20"
"""The week the clearest-week panels draw, as `make_figures._chosen_weeks` picks it."""

HEADLINE: Final[tuple[str, str]] = ("C_era5_split", "B_erbs")
"""The contrast to recompute under each denominator."""

DECILES: Final[int] = 10
"""How many equal-count bins of measured output the bias is broken down over."""

PERCENT: Final[float] = 100.0


def _results_dir(*, source: str) -> pathlib.Path:
    """Return where the XGBoost results for one source live."""
    return STUDY_DATA_DIR / "ERA5" / f"beam_diffuse_results_{source}"


def main() -> int:
    """Print the per-site scale, the clearest week's bias, and the headline per denominator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCE_CHOICES, default="cams")
    arguments = parser.parse_args()

    measured = drop_commissioning_ramp(
        dataset=pl.read_parquet(dataset_path_for(source=arguments.source))
    )
    scales = (
        measured.group_by("site")
        .agg(
            p99=pl.col("power_mw").quantile(0.99),
            p999=pl.col("power_mw").quantile(0.999),
            highest=pl.col("power_mw").max(),
            denominator=pl.col("effective_capacity_mw").first(),
        )
        .sort("site")
    )
    print("\n### What the capacity denominator is, and what the alternatives would be\n")
    print(
        "| Site | Denominator in use (MW) | Daylight P99 / it "
        "| Daylight P99.9 / it | Highest reading / it |"
    )
    print("|---|---|---|---|---|")
    for row in scales.iter_rows(named=True):
        print(
            f"| {row['site']} | {row['denominator']:.3f} "
            f"| {row['p99'] / row['denominator']:.3f} "
            f"| {row['p999'] / row['denominator']:.3f} "
            f"| {row['highest'] / row['denominator']:.3f} |"
        )

    losses = pl.read_parquet(_results_dir(**vars(arguments)) / "per_row_losses.parquet").filter(
        (pl.col("setting") == "primary") & (pl.col("target") == "power_mw")
    )
    treatment = losses.filter(pl.col("arm") == HEADLINE[0])
    per_hour = (
        treatment.group_by("site", "time")
        .agg(
            signed_error_mw=pl.col("signed_error_capped_mw").mean(),
            effective_capacity_mw=pl.col("effective_capacity_mw").first(),
        )
        .join(measured.select("site", "time", "power_mw"), on=["site", "time"], how="inner")
    )

    week = per_hour.filter(
        pl.col("time").dt.truncate("1w")
        == pl.lit(CLEAREST_WEEK).str.to_datetime("%Y-%m-%d").dt.replace_time_zone("UTC")
    )
    print(f"\n### Mean signed error in the week beginning {CLEAREST_WEEK}, arm {HEADLINE[0]}\n")
    print(
        "| Site | That week (% of P99) | Whole record (% of P99) | Week's peak output (% of P99) |"
    )
    print("|---|---|---|---|")
    for site in sorted(per_hour["site"].unique().to_list()):
        rows = week.filter(pl.col("site") == site)
        whole = per_hour.filter(pl.col("site") == site)
        if rows.is_empty():
            continue
        share_of_capacity = pl.col("signed_error_mw") / pl.col("effective_capacity_mw")
        bias = rows.select(share_of_capacity.mean()).item() * PERCENT
        overall = whole.select(share_of_capacity.mean()).item() * PERCENT
        peak = (
            rows.select((pl.col("power_mw") / pl.col("effective_capacity_mw")).max()).item()
            * PERCENT
        )
        print(f"| {site} | {bias:+.2f} | {overall:+.2f} | {peak:.1f} |")

    print("\n### Mean signed error by decile of measured output, pooled and at sites A and B\n")
    binned = per_hour.with_columns(
        decile=(pl.col("power_mw").rank("ordinal").over("site") * DECILES - 1)
        // pl.len().over("site"),
        bias=pl.col("signed_error_mw") / pl.col("effective_capacity_mw") * PERCENT,
    )
    print("| Decile of output | All six sites | Site A | Site B |")
    print("|---|---|---|---|")
    for decile in range(DECILES):
        rows = binned.filter(pl.col("decile") == decile)
        pooled = rows["bias"].mean()
        site_a = rows.filter(pl.col("site") == "A")["bias"].mean()
        site_b = rows.filter(pl.col("site") == "B")["bias"].mean()
        print(f"| {decile + 1} | {pooled:+.2f} | {site_a:+.2f} | {site_b:+.2f} |")

    print("\n### The headline contrast under four capacity denominators\n")
    print("| Denominator | ΔMAE (pp) | 95% interval | Relative |")
    print("|---|---|---|---|")
    for label, column in (
        ("The 99th percentile of the whole record, in use", "denominator"),
        ("The 99th percentile of the daylight rows", "p99"),
        ("The 99.9th percentile of the daylight rows", "p999"),
        ("The highest reading", "highest"),
    ):
        rescaled = _add_time_features(
            dataset=losses.join(scales.select("site", column), on="site", how="inner")
        ).with_columns(metric=pl.col("absolute_error_capped_mw") / pl.col(column))
        result = _bootstrap_difference(
            losses=rescaled, treatment=HEADLINE[0], reference=HEADLINE[1], metric="metric"
        )
        reference_level = (
            rescaled.filter(pl.col("arm") == HEADLINE[1]).select(pl.col("metric").mean()).item()
            * PERCENT
        )
        point = result["difference"] * PERCENT
        print(
            f"| {label} | {point:+.4f} "
            f"| [{result['lower_95'] * PERCENT:+.4f}, {result['upper_95'] * PERCENT:+.4f}] "
            f"| {point / reference_level * PERCENT:+.2f}% |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
