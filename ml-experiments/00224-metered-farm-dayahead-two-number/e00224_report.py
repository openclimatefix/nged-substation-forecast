"""Print every number the write-up of experiment 224 quotes beyond `e00224_score.py`'s tables.

Per-farm gains over the ensemble-mean arm, the leave-one-farm-out check of the two headline
contrasts, the MAE of the median and of the mean forecast, the median minus the mean forecast, the
tail table, and the stop-rule count, all for the scored year's headline lead band (12 to 36 h).

    uv run --no-sync python ml-experiments/00224-*/e00224_report.py
"""

import e00224_common as common
import polars as pl

pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(20)
KEY_ARMS = ["a2_mean", "a3_two_number", "a4_members", "a5_xgboost", "floor_heuristic"]


def main() -> None:
    """Print the tables."""
    farms = pl.read_csv(common.DATA_DIR / "per_farm.csv").filter(
        (pl.col("fold") == "scored_year") & (pl.col("band") == common.HEADLINE_BAND)
    )
    score = farms.pivot(on="arm", index="farm", values="score")
    gains = score.select(
        "farm",
        c0_percent=(pl.col("a4_members") / pl.col("a2_mean") - 1) * 100,
        c1_percent=(pl.col("a3_two_number") / pl.col("a2_mean") - 1) * 100,
        two_number_vs_xgboost_percent=(pl.col("a3_two_number") / pl.col("a5_xgboost") - 1) * 100,
    )
    print(gains)
    print("farms with C0 gain under 2%:", gains.filter(pl.col("c0_percent") > -2.0).height)
    print("--- leave-one-farm-out (mean of the other five farms)")
    for left_out in common.FARMS:
        rest = score.filter(pl.col("farm") != left_out)
        mean = rest.select(pl.col(c).mean() for c in score.columns if c != "farm")
        c0 = (mean["a4_members"][0] / mean["a2_mean"][0] - 1) * 100
        c1 = (mean["a3_two_number"][0] / mean["a2_mean"][0] - 1) * 100
        c3 = (mean["a3_two_number"][0] / mean["a5_xgboost"][0] - 1) * 100
        print(left_out, f"C0 {c0:.2f}  C1 {c1:.2f}  share {c1 / c0:.2f}  C3 {c3:.2f}")
    for column in ("mae_median", "mae_mean", "median_minus_mean"):
        print("---", column, "(points of capacity), per farm and mean over farms")
        table = farms.pivot(on="arm", index="farm", values=column).select("farm", *KEY_ARMS)
        print(table)
        print(table.select(pl.col(c).mean() for c in KEY_ARMS))
    tails = pl.read_csv(common.DATA_DIR / "tails.csv").filter(
        (pl.col("fold") == "scored_year")
        & (pl.col("band") == common.HEADLINE_BAND)
        & pl.col("arm").is_in([*KEY_ARMS, "floor_climatology"])
    )
    print(
        tails.select(
            "arm", "level", "coverage", "pinball", "pinball_lower_95", "pinball_upper_95"
        ).sort("level", "arm")
    )


if __name__ == "__main__":
    main()
