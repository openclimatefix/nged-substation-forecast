# ruff: noqa: E501
"""Print the numbers experiment 243's write-up quotes beyond the scorer.

The numbers are the seed spread, per-farm contrasts, leave-one-farm-out, the Checks (mean forecast, median minus mean), and coverage.

    uv run --no-sync python ml-experiments/00243-*/e00243_report.py
"""

import e00243_common as common
import numpy as np
import polars as pl
from studies.tail_calibration import level_column

pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(20)
QCOLS = [level_column(level=lv) for lv in common.LEVELS]
MASSES = np.diff(np.concatenate([[0.0], (common.LEVELS[:-1] + common.LEVELS[1:]) / 2, [1.0]]))
KEY = [
    "plant",
    "xgboost",
    "nn_pure",
    "h1_scale",
    "h2_residual",
    "h3_adapter",
    "h4_both",
    "h5_quantile",
    "h6_blind",
]


def main() -> None:
    """Print the tables."""
    rows = pl.read_parquet(common.DATA_DIR / "row_losses.parquet")
    farm = rows.group_by("setting", "arm", "time_series_id").agg(pl.col("score").mean())
    print("--- per-farm change in quantile score (%), own_history, against the plant and XGBoost")
    for setting in common.SETTINGS:
        wide = (
            farm.filter(pl.col("setting") == setting)
            .pivot(on="arm", index="time_series_id", values="score")
            .sort("time_series_id")
        )
        for ref in ("plant", "xgboost"):
            print(setting, "vs", ref)
            print(
                wide.select(
                    "time_series_id",
                    *[((pl.col(a) / pl.col(ref) - 1) * 100).alias(a) for a in KEY if a != ref],
                )
            )
        print("--- leave-one-farm-out of h5 vs plant and vs xgboost,", setting)
        for left in common.FARMS:
            rest = wide.filter(pl.col("time_series_id") != left).select(
                pl.col(a).mean() for a in KEY
            )
            print(
                left,
                f"h5/plant {100 * (rest['h5_quantile'][0] / rest['plant'][0] - 1):.2f}  h5/xgb {100 * (rest['h5_quantile'][0] / rest['xgboost'][0] - 1):.2f}  nn/xgb {100 * (rest['nn_pure'][0] / rest['xgboost'][0] - 1):.2f}",
            )
    print("--- seed spread of the farm-mean score (own_history)")
    seeds = []
    for setting in common.SETTINGS:
        for f in sorted((common.PRED_DIR / setting).glob("*_s*_*.parquet")):
            d = pl.read_parquet(f).with_columns(
                pl.col(QCOLS).cast(pl.Float64), pl.col("actual").cast(pl.Float64)
            )
            q = d.select(QCOLS).to_numpy()
            a = d["actual"].to_numpy()
            err = a[:, None] - q
            pin = np.maximum(common.LEVELS * err, (common.LEVELS - 1) * err)
            mean_fc = q @ MASSES
            seeds.append(
                {
                    "setting": setting,
                    "arm": d["arm"][0],
                    "seed": d["seed"][0],
                    "site": d["time_series_id"][0],
                    "score": 200 * pin.mean(),
                    "mae_mean": 100 * np.abs(a - mean_fc).mean(),
                    "mae_median": 100 * np.abs(err[:, 5]).mean(),
                    "median_minus_mean": 100 * (q[:, 5] - mean_fc).mean(),
                    "cover01": (a <= q[:, 0]).mean(),
                    "cover99": (a <= q[:, 10]).mean(),
                    "cover05": (a <= q[:, 1]).mean(),
                    "cover95": (a <= q[:, 9]).mean(),
                }
            )
    s = pl.DataFrame(seeds)
    per_seed = s.group_by("setting", "arm", "seed").agg(pl.col("score").mean())
    print(
        per_seed.group_by("setting", "arm")
        .agg(mean=pl.col("score").mean(), sd=pl.col("score").std(), n=pl.len())
        .sort("setting", "mean")
    )
    print(
        "--- Checks: mean-forecast MAE, median MAE, median minus mean (points of capacity), coverage; mean over farms and seeds"
    )
    print(
        s.group_by("setting", "arm")
        .agg(
            *[
                pl.col(c).mean()
                for c in (
                    "mae_mean",
                    "mae_median",
                    "median_minus_mean",
                    "cover01",
                    "cover05",
                    "cover95",
                    "cover99",
                )
            ]
        )
        .sort("setting", "arm")
    )


if __name__ == "__main__":
    main()
