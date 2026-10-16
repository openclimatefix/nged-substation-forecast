"""Print every number that experiment 00048's `result.md` quotes, from the saved results.

The numbers are the absolute scores by group, lead band, and series; the nine planned contrasts at
99.44% with the verdict that the decision rule gives each; the same contrasts at 95%, by lead
band, and on each series alone; the descriptive contrasts; and the reproduction check on the 73
runs of experiment 00034.

    uv run python ml-experiments/00048-*/e00048_report.py [--smoke]
"""

import argparse
from typing import Final

import e00048_common as common
import e00048_stats as stats
import polars as pl

HEADLINE_ARMS: Final[dict[str, tuple[str, ...]]] = {
    "Primary": (
        "frozen_cs",
        "frozen_s",
        "monthly_s",
        "monthly_cs",
        "weekly_s",
        "weekly_cs",
        "weekly_c7s",
        "roll12_s",
        "roll12_cs",
        "manual_heuristic",
        "climatology",
    ),
    "BSP": ("frozen_cs", "frozen_s", "monthly_s", "weekly_s", "weekly_cs", "manual_heuristic"),
    "GSP": ("frozen_cs", "frozen_s", "monthly_s", "weekly_s", "weekly_cs", "manual_heuristic"),
    "solar": ("gen_frozen_s", "gen_monthly_s", "gen_frozen", "gen_monthly", "manual_heuristic"),
    "wind": ("gen_frozen_s", "gen_monthly_s", "gen_frozen", "gen_monthly", "manual_heuristic"),
}
SIGN_RULE: Final[frozenset[str]] = frozenset({"C1", "C2", "C3", "C6", "C7"})
CORRECTION_RULE: Final[frozenset[str]] = frozenset({"C4", "C5"})
NO_CHANGE_RULE: Final[frozenset[str]] = frozenset({"C8s", "C8w"})


def _below_or_above(*, lower: float, upper: float) -> str:
    """Return `below`, `above`, or `across` for an interval and zero."""
    if upper < 0:
        return "below"
    if lower > 0:
        return "above"
    return "across"


def verdict(*, contrast_id: str, lower: float, upper: float) -> str:
    """Return the decision rule's reading of one planned contrast at the Bonferroni level.

    Args:
        contrast_id: One of `C1` to `C8w`.
        lower: The interval's lower bound.
        upper: The interval's upper bound.

    Returns:
        The reading, in the plan's words.
    """
    side = _below_or_above(lower=lower, upper=upper)
    if contrast_id in SIGN_RULE:
        readings = {
            "below": "supported (interval wholly below zero)",
            "above": "refuted (interval wholly above zero)",
            "across": "inconclusive (interval includes zero)",
        }
    elif contrast_id in CORRECTION_RULE:
        readings = {
            "below": "correction still helping (interval wholly below zero)",
            "above": "correction no longer helping (interval wholly above zero)",
            "across": "inconclusive (interval includes zero)",
        }
    elif contrast_id in NO_CHANGE_RULE:
        inside = lower >= -stats.NO_CHANGE_MARGIN and upper <= stats.NO_CHANGE_MARGIN
        if inside:
            return "no change (interval inside plus or minus 0.05)"
        readings = {
            "below": "a change (interval excludes zero)",
            "above": "a change (interval excludes zero)",
            "across": "inconclusive (interval includes zero, wider than plus or minus 0.05)",
        }
    else:
        return ""
    return readings[side]


def main() -> None:
    """Print the absolute scores, the contrasts, the per-series results, and the reproduction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    smoke = parser.parse_args().smoke
    folder = common.RESULTS_DIR.with_name("results_smoke") if smoke else common.RESULTS_DIR
    pl.Config.set_tbl_rows(300)
    pl.Config.set_tbl_cols(14)
    pl.Config.set_fmt_str_lengths(40)
    absolute = pl.read_parquet(folder / "absolute.parquet")
    for group, arms in HEADLINE_ARMS.items():
        print(
            f"ABSOLUTE, {group}, all leads (quantile score with 95% interval; MAE; 0.95 exceedance)"
        )
        print(
            absolute.filter(
                (pl.col("group") == group)
                & (pl.col("scope") == "all leads")
                & pl.col("arm").is_in(arms)
            )
            .select("arm", "cases", "qs", "qs_lower", "qs_upper", "mae", "tw_p99", "ex95", "cov90")
            .sort("qs")
            .with_columns(pl.col(pl.Float64).round(3))
        )
        print(f"ABSOLUTE, {group}, quantile score by lead band")
        print(
            absolute.filter((pl.col("group") == group) & pl.col("arm").is_in(arms))
            .pivot(on="scope", index="arm", values="qs")
            .with_columns(pl.col(pl.Float64).round(2))
        )
    series_scores = pl.read_parquet(folder / "absolute_series.parquet")
    print("QUANTILE SCORE BY SERIES, key arms")
    print(
        series_scores.filter(
            pl.col("arm").is_in(
                ["frozen_cs", "monthly_s", "weekly_s", "weekly_cs", "gen_frozen_s", "gen_monthly_s"]
            )
        )
        .pivot(on="arm", index=["group", "site"], values="qs")
        .sort("group", "site")
        .with_columns(pl.col(pl.Float64).round(2))
    )
    contrasts = pl.read_parquet(folder / "contrasts.parquet")
    planned = contrasts.filter(
        (pl.col("family") == "planned")
        & (pl.col("site") == "pooled")
        & (pl.col("scope") == "all leads")
        & (pl.col("metric") == "qs")
    )
    print("PLANNED CONTRASTS at 99.44%, quantile score, all leads (points of capacity)")
    for row in planned.filter(pl.col("level_label") == "planned").iter_rows(named=True):
        print(
            f"{row['id']}: {row['treatment']} minus {row['reference']}: "
            f"{row['difference']:+.3f} [{row['lower']:+.3f}, {row['upper']:+.3f}] "
            f"({row['cases']:,} cases, {row['months']} months): "
            f"{verdict(contrast_id=row['id'], lower=row['lower'], upper=row['upper'])}"
        )
    print("PLANNED CONTRASTS at 95%")
    print(
        planned.filter(pl.col("level_label") == "95")
        .select("id", "treatment", "reference", "difference", "lower", "upper")
        .with_columns(pl.col(pl.Float64).round(3))
    )
    print(
        "PLANNED CONTRASTS, other metrics, all leads (MAE of the median; twCRPS at historical_p99)"
    )
    print(
        contrasts.filter(
            (pl.col("family") == "planned")
            & (pl.col("site") == "pooled")
            & (pl.col("scope") == "all leads")
            & (pl.col("metric") != "qs")
            & (pl.col("level_label") == "95")
        )
        .select("id", "metric", "difference", "lower", "upper")
        .with_columns(pl.col(pl.Float64).round(3))
    )
    print("PLANNED CONTRASTS by lead band, quantile score, 95%")
    print(
        contrasts.filter(
            (pl.col("family") == "planned")
            & (pl.col("site") == "pooled")
            & (pl.col("scope") != "all leads")
            & (pl.col("level_label") == "95")
        )
        .select("id", "scope", "difference", "lower", "upper")
        .with_columns(pl.col(pl.Float64).round(3))
    )
    print("PLANNED CONTRASTS on each series alone, quantile score, 95% (descriptive)")
    print(
        contrasts.filter(
            (pl.col("family") == "planned")
            & (pl.col("site") != "pooled")
            & (pl.col("scope") == "all leads")
            & (pl.col("level_label") == "95")
        )
        .select("id", "site", "difference", "lower", "upper")
        .with_columns(pl.col(pl.Float64).round(3))
        .sort("id", "site")
    )
    print("DESCRIPTIVE CONTRASTS, pooled, all leads, quantile score, 95%")
    print(
        contrasts.filter(
            (pl.col("family") == "descriptive")
            & (pl.col("site") == "pooled")
            & (pl.col("scope") == "all leads")
            & (pl.col("metric") == "qs")
        )
        .select("id", "treatment", "reference", "difference", "lower", "upper", "cases")
        .with_columns(pl.col(pl.Float64).round(3))
    )
    print("REPRODUCTION on the 73 runs of experiment 00034 (tolerance 0.001 points)")
    print(
        pl.read_parquet(folder / "reproduction.parquet").with_columns(pl.col(pl.Float64).round(4))
    )


if __name__ == "__main__":
    main()
