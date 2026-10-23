# ruff: noqa: E501
"""Leave-one-farm-out and per-farm checks for experiment 300's headline contrasts.

uv run --no-sync python ml-experiments/00300-*/e00300_checks.py
"""

import e00300_common as common
import polars as pl


def main() -> None:
    """Print per-farm scores and the candidate-minus-XGBoost change with each farm dropped in turn."""
    rows = pl.read_parquet(common.DATA_DIR / "row_losses.parquet")
    pl.Config.set_tbl_rows(60)
    for setting in common.SETTINGS:
        sub = rows.filter(pl.col("setting") == setting)
        farm = sub.group_by("arm", "time_series_id").agg(
            score=pl.col("score").mean(), pin99=pl.col("pin99").mean()
        )
        print(setting, "per-farm score")
        print(
            farm.filter(pl.col("arm").is_in(["blend", "xgb", "p2", "floor_heuristic"]))
            .pivot(on="arm", index="time_series_id", values="score")
            .sort("time_series_id")
            .with_columns(pl.all().exclude("time_series_id").round(3))
        )
        out = []
        for drop in common.FARMS:
            kept = (
                farm.filter(pl.col("time_series_id") != drop)
                .group_by("arm")
                .agg(score=pl.col("score").mean(), pin99=pl.col("pin99").mean())
            )
            b = kept.filter(pl.col("arm") == "blend").row(0, named=True)
            x = kept.filter(pl.col("arm") == "xgb").row(0, named=True)
            out.append(
                {
                    "dropped": drop,
                    "score_change_pct": round(100 * (b["score"] - x["score"]) / x["score"], 2),
                    "pin99_change_pct": round(100 * (b["pin99"] - x["pin99"]) / x["pin99"], 2),
                }
            )
        print(setting, "leave one farm out: candidate minus XGBoost")
        print(pl.DataFrame(out))
        monthly = (
            sub.filter(pl.col("arm").is_in(["blend", "xgb"]))
            .group_by("arm", "month")
            .agg(score=pl.col("score").mean())
            .pivot(on="arm", index="month", values="score")
            .sort("month")
            .with_columns(diff=pl.col("blend") - pl.col("xgb"))
        )
        print(
            setting,
            "months where the candidate is worse:",
            monthly.filter(pl.col("diff") > 0)["month"].to_list(),
        )


if __name__ == "__main__":
    main()
