"""Bound what a dynamic effective-capacity estimate could buy this experiment.

Site A's error drifts from −2.3% of P99 output in 2023 to +4.1% in 2026, which a dynamic capacity
estimate would track and the static full-history 99th percentile does not. The question that
matters here is not whether tracking it would lower the error — it plainly would — but whether it
would change any contrast, because every contrast on this page is a paired difference between two
arms scored on the same rows.

This removes the bias an oracle estimator would remove and no more: for each arm and seed, the mean
signed error inside each site-year, and then inside each site-month, is subtracted from every
prediction in that block before the absolute error is retaken. Using the scored rows' own mean is
what makes it an oracle, and therefore an upper bound on what any real estimator could deliver.

Run it from this directory as:

```bash
uv run \
    python oracle_capacity.py --source cams
```
"""

import argparse
import sys
from typing import Final

import polars as pl
from run_experiment import _bootstrap_difference
from sources import SOURCE_CHOICES, STUDY_DATA_DIR

CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("C_era5_split", "B_erbs"),
    ("C_era5_split", "A_global_only"),
    ("B_erbs", "A_global_only"),
)
"""The headline, the whole-split contrast, and the negative control."""

PERCENT: Final[float] = 100.0


def main() -> int:
    """Print each contrast as published, and after an oracle removes the per-block bias."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCE_CHOICES, default="cams")
    arguments = parser.parse_args()

    results = STUDY_DATA_DIR / "ERA5" / f"beam_diffuse_results_{arguments.source}"
    losses = (
        pl.read_parquet(results / "per_row_losses.parquet")
        .filter((pl.col("setting") == "primary") & (pl.col("target") == "power_mw"))
        .with_columns(
            year=pl.col("time").dt.year(),
            published=pl.col("absolute_error_capped_mw") / pl.col("effective_capacity_mw"),
        )
    )

    # An oracle capacity estimate can only remove the block's mean signed error. Subtracting it and
    # retaking the absolute error is what the arm would have scored had the block been unbiased.
    for name, keys in (("year", ["site", "year"]), ("month", ["site", "month"])):
        column = f"debiased_by_{name}"
        losses = losses.with_columns(
            (
                (
                    pl.col("signed_error_capped_mw")
                    - pl.col("signed_error_capped_mw").mean().over([*keys, "arm", "seed"])
                ).abs()
                / pl.col("effective_capacity_mw")
            ).alias(column)
        )

    metrics = ("published", "debiased_by_year", "debiased_by_month")
    print(f"\n### Arm error levels, {arguments.source}\n")
    print("| Arm | As published | Oracle per site-year | Oracle per site-month |")
    print("|---|---|---|---|")
    for arm in sorted(losses["arm"].unique().to_list()):
        rows = losses.filter(pl.col("arm") == arm)
        cells = " | ".join(
            f"{rows.select(pl.col(metric).mean()).item() * PERCENT:.3f}" for metric in metrics
        )
        print(f"| {arm} | {cells} |")

    print("\n### Contrasts under each correction\n")
    print("| Contrast | Correction | ΔMAE (pp of P99 output) | 95% interval | Relative |")
    print("|---|---|---|---|---|")
    for treatment, reference in CONTRASTS:
        for metric in metrics:
            result = _bootstrap_difference(
                losses=losses, treatment=treatment, reference=reference, metric=metric
            )
            level = (
                losses.filter(pl.col("arm") == reference).select(pl.col(metric).mean()).item()
                * PERCENT
            )
            point = result["difference"] * PERCENT
            print(
                f"| {treatment} − {reference} | {metric} | {point:+.4f} "
                f"| [{result['lower_95'] * PERCENT:+.4f}, {result['upper_95'] * PERCENT:+.4f}] "
                f"| {point / level * PERCENT:+.2f}% |"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
