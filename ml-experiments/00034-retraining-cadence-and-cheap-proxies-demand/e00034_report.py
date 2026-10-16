"""Print every number that experiment 00034's `result.md` quotes, from the saved results.

uv run python ml-experiments/00034-*/e00034_report.py
"""

from typing import Final

import e00034_common as common
import polars as pl

RESULTS_DIR: Final = common.WORK_DIR / "results"
ARMS: Final[tuple[str, ...]] = (
    "n1_cs",
    "yearly_s",
    "yearly_cs",
    "half_yearly_s",
    "half_yearly_cs",
    "three_a_year_s",
    "three_a_year_cs",
    "quarterly_s",
    "quarterly_cs",
    "monthly_s",
    "monthly_cs",
    "weekly_s",
    "weekly_cs",
    "monthly_2y_s",
    "roll12_s",
    "warm20_monthly_s",
    "warm50_monthly_s",
    "warm20_weekly_s",
    "warm50_weekly_s",
    "interp_s",
    "manual_heuristic",
    "climatology",
    "production",
)


def main() -> None:
    """Print the absolute scores, the contrasts, the rank test, and the cost."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(12)
    pl.Config.set_fmt_str_lengths(34)
    absolute = pl.read_parquet(RESULTS_DIR / "absolute.parquet")
    print("ABSOLUTE, Primary, all leads (quantile score, 95% interval, seeds)")
    print(
        absolute.filter(
            (pl.col("group") == "Primary")
            & (pl.col("scope") == "all leads")
            & pl.col("arm").is_in(ARMS)
        )
        .select("arm", "seeds", "qs", "qs_lower", "qs_upper", "mae", "ex95")
        .sort("qs")
        .with_columns(pl.col(pl.Float64).round(3))
    )
    print("ABSOLUTE by lead band, Primary, quantile score")
    print(
        absolute.filter(
            (pl.col("group") == "Primary")
            & pl.col("arm").is_in(["n1_cs", "yearly_s", "monthly_s", "weekly_s"])
        )
        .pivot(on="scope", index="arm", values="qs")
        .with_columns(pl.col(pl.Float64).round(2))
    )
    print("ABSOLUTE, BSP and GSP, all leads")
    print(
        absolute.filter(
            pl.col("group").is_in(["BSP", "GSP"])
            & (pl.col("scope") == "all leads")
            & pl.col("arm").is_in(["n1", "n1_cs", "yearly_s", "monthly_s", "monthly_cs"])
        )
        .select("group", "arm", "seeds", "qs")
        .sort("group", "qs")
        .with_columns(pl.col(pl.Float64).round(2))
    )
    for name in ("planned", "exploratory"):
        paths = [
            RESULTS_DIR / f"contrasts_{name}.parquet",
            RESULTS_DIR / f"contrasts_{name}_ids.parquet",
        ]
        contrasts = pl.concat([pl.read_parquet(p) for p in paths if p.exists()])
        print(f"CONTRASTS ({name}), all leads, quantile score")
        print(
            contrasts.filter((pl.col("scope") == "all leads") & (pl.col("metric") == "qs"))
            .select("id", "level_label", "difference", "lower", "upper", "cases", "seeds")
            .with_columns(pl.col(pl.Float64).round(3))
        )
    keys = (
        pl.read_parquet(common.WORK_DIR / "metrics" / "daily_s.parquet")
        .filter((pl.col("group") == "Primary") & (pl.col("seed") == 0))
        .select(*common.KEYS, "seed")
    )
    print("QUANTILE SCORE on the daily arm's rows (October 2025, January 2026, April 2026), seed 0")
    for arm in ("daily_s", "daily_cs", "weekly_s", "weekly_cs", "monthly_s", "yearly_s", "n1_cs"):
        rows = (
            pl.read_parquet(common.WORK_DIR / "metrics" / f"{arm}.parquet")
            .filter((pl.col("group") == "Primary") & (pl.col("seed") == 0))
            .join(keys, on=[*common.KEYS, "seed"], how="semi")
        )
        print(f"{arm}: {rows.height} cases, {float(rows['qs'].mean()):.3f}")  # ty: ignore[invalid-argument-type]
    rank = pl.read_parquet(RESULTS_DIR / "contrasts_rank.parquet")
    print("RANK TEST, all leads, 95% intervals, seed 0")
    print(
        rank.filter(
            (pl.col("scope") == "all leads")
            & (pl.col("metric") == "qs")
            & (pl.col("level_label") == "95")
        )
        .select("id", "difference", "lower", "upper")
        .with_columns(pl.col(pl.Float64).round(3))
    )


if __name__ == "__main__":
    main()
