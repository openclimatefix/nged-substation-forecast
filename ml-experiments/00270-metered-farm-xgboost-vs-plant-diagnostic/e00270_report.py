# ruff: noqa: E501
"""Print the numbers experiment 270's write-up quotes beyond the scorer.

Shares of the gain by rung, location versus width, per-farm and leave-one-farm-out contrasts, the
Checks (median minus mean forecast, MAE of the mean), the raw crossing rate, and the tail
calibration table of `studies.tail_calibration`.

    uv run --no-sync python ml-experiments/00270-*/e00270_report.py
"""

import e00270_common as common
import polars as pl
from studies.tail_calibration import tail_calibration_table

pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(30)
KEY = [
    "plant",
    "p_cams",
    "p_ens",
    "pq_elev",
    "pq_csi",
    "pq_spread",
    "x2_plant_forecast_only",
    "x1_plant_inputs",
    "l1",
    "l2",
    "l3",
    "l4",
    "l5",
    "l6",
    "gam",
]
TAIL_ARMS = ["plant", "x1_plant_inputs", "l1", "l3", "l6", "gam", "p_ens"]


def main() -> None:
    """Print the tables."""
    scores = pl.read_csv(common.DATA_DIR / "scores.csv")
    contrasts = pl.read_csv(common.DATA_DIR / "contrasts.csv")
    rows = pl.read_parquet(common.DATA_DIR / "row_losses.parquet")
    allrows = pl.col("slice") == "all"
    for setting in common.SETTINGS:
        print(f"=== {setting}")
        wide = (
            scores.filter(allrows & (pl.col("setting") == setting))
            .pivot(on="metric", index="arm", values="value")
            .filter(pl.col("arm").is_in(KEY + [f"s{k}" for k in range(2, 7)]))
        )
        print("--- absolute scores (points of capacity)")
        print(wide.sort("score"))
        base = scores.filter(
            allrows & (pl.col("setting") == setting) & (pl.col("metric") == "score")
        )
        values = dict(zip(base["arm"], base["value"], strict=True))
        total = values["plant"] - values["l6"]
        print(f"--- shares of the gain l6 versus plant ({total:.3f} points)")
        steps = [
            ("plant", "l1"),
            ("l1", "l2"),
            ("l2", "l3"),
            ("l3", "l4"),
            ("l4", "l5"),
            ("l5", "l6"),
        ]
        for a, b in steps:
            print(
                f"{a} -> {b}: {values[a] - values[b]:+.3f} points, {100 * (values[a] - values[b]) / total:+.0f}% of the total"
            )
        print(
            f"plant -> x1_plant_inputs: {values['plant'] - values['x1_plant_inputs']:+.3f} points, {100 * (values['plant'] - values['x1_plant_inputs']) / total:+.0f}% of the total"
        )
        print("--- per-farm quantile score")
        farm = pl.read_csv(common.DATA_DIR / "per_farm.csv").filter(pl.col("setting") == setting)
        wide_farm = farm.pivot(on="arm", index="time_series_id", values="score").sort(
            "time_series_id"
        )
        print(wide_farm.select("time_series_id", *[a for a in KEY if a in wide_farm.columns]))
        print(
            "--- leave-one-farm-out: percent change against the plant (mean over the other five farms)"
        )
        for arm in ("x1_plant_inputs", "l6", "gam"):
            out = []
            for left in common.FARMS:
                sub = wide_farm.filter(pl.col("time_series_id") != left)
                arm_mean, plant_mean = float(sub[arm].mean()), float(sub["plant"].mean())  # ty: ignore[invalid-argument-type]
                out.append(f"{left}: {100 * (arm_mean - plant_mean) / plant_mean:+.2f}")
            print(arm, " ".join(out))
        print("--- contrasts by slice (quantile score, percent change)")
        for arm, ref in (("x1_plant_inputs", "plant"), ("l6", "plant"), ("gam", "plant")):
            print(
                contrasts.filter(
                    (pl.col("setting") == setting)
                    & (pl.col("metric") == "score")
                    & (pl.col("arm") == arm)
                    & (pl.col("reference") == ref)
                    & (pl.col("slice_kind") != "all")
                ).select("slice_kind", "slice", "arm", "reference", "percent_change", "lo", "hi")
            )
        print(
            "--- Checks: median minus mean forecast and MAE of the mean, per arm (points of capacity)"
        )
        check = (
            rows.filter((pl.col("setting") == setting) & pl.col("arm").is_in(KEY))
            .group_by("arm")
            .agg(
                median_minus_mean=pl.col("median_minus_mean").mean(),
                ae_mean=pl.col("ae_mean").mean(),
                ae_median=pl.col("ae_median").mean(),
            )
            .sort("arm")
        )
        print(check)
        print("--- median minus mean per farm (plant, l6)")
        print(
            rows.filter((pl.col("setting") == setting) & pl.col("arm").is_in(["plant", "l6"]))
            .group_by("arm", "time_series_id")
            .agg(pl.col("median_minus_mean").mean())
            .pivot(on="arm", index="time_series_id", values="median_minus_mean")
            .sort("time_series_id")
        )
    print("=== raw crossing rate per arm")
    print(pl.read_csv(common.DATA_DIR / "crossing.csv").filter(pl.col("arm").is_in(KEY)))
    print("=== tail calibration, own_history, seed 0 (pooled rows)")
    for arm in TAIL_ARMS:
        frame = pl.concat(
            [
                pl.read_parquet(common.PRED_DIR / "own_history" / f"{arm}_s0_{s}.parquet")
                for s in common.FARMS
            ]
        ).rename({"init_time": "power_fcst_init_time"})
        print(arm)
        print(tail_calibration_table(frame=frame.with_columns(pl.col("actual").cast(pl.Float64))))


if __name__ == "__main__":
    main()
