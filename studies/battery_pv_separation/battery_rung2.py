"""Rung 2: predict a battery's output from prices alone, on held-out months.

Three predictors, all fitted on three of the four folds and scored on the fourth:

- `time_of_day`: the mean output of each half-hour of the UTC day.
- `price_rank`: a least-squares fit to indicator columns for the half-hour of day, for the
  half-hour's day-ahead price rank within its day (eight equal bins), for the day-ahead price level
  (four bins cut at the training rows' quartiles), and for each rank-bin and level-bin pair.
- `xgboost_price`: an XGBoost model (squared error, `colsample_bytree=1`, 200 rounds of depth 4)
  given the half-hour of day, the rank, the price, and the day's mean price and price range.
  Exploratory.

Planned contrast B1: `price_rank` minus `time_of_day` in mean absolute error, as a percentage of the
battery's 99th-percentile absolute output, with a 95% interval from resampling whole calendar
months. Saves the out-of-fold predictions and the monthly error table, and writes
`report_rung2.md`.

Run: `OMP_NUM_THREADS=2 uv run python studies/battery_pv_separation/battery_rung2.py`.
"""

from typing import Final

import numpy as np
import polars as pl
import xgboost as xgb
from battery_inputs import (
    BATTERIES,
    HALF_HOURS_PER_DAY,
    NAMES,
    OUTPUT_DIR,
    battery_frame,
    p99_output_mw,
)

N_RANK_BINS: Final[int] = 8
N_LEVEL_BINS: Final[int] = 4
N_RESAMPLES: Final[int] = 2000
RESAMPLE_SEED: Final[int] = 20261008
PERCENT: Final[float] = 100.0
ARMS: Final[tuple[str, ...]] = ("time_of_day", "price_rank", "xgboost_price")
CONTRASTS: Final[tuple[tuple[str, str, str], ...]] = (
    ("price_rank", "time_of_day", "B1, planned"),
    ("xgboost_price", "time_of_day", "exploratory"),
    ("xgboost_price", "price_rank", "exploratory"),
)
"""Treatment, reference, and whether the contrast was planned."""
XGB_FEATURES: Final[tuple[str, ...]] = (
    "tod",
    "rank_pct",
    "day_ahead_gbp_per_mwh",
    "day_mean_price",
    "day_price_range",
)
XGB_PARAMETERS: Final[dict[str, float | int | str]] = {
    "objective": "reg:squarederror",
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "min_child_weight": 20,
    "tree_method": "hist",
    "nthread": 2,
    "seed": 0,
}
XGB_ROUNDS: Final[int] = 200


def design_matrix(
    *, frame: pl.DataFrame, level_edges: np.ndarray | None, with_price: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Build the indicator columns of one predictor.

    Args:
        frame: A battery frame from `battery_frame`.
        level_edges: The price-level bin edges, or `None` to cut them at this frame's quartiles.
        with_price: Whether to add the price columns to the half-hour-of-day columns.

    Returns:
        The design matrix and the level edges used.
    """
    tod = frame["tod"].to_numpy()
    columns = [(tod[:, None] == np.arange(HALF_HOURS_PER_DAY)[None, :]).astype(float)]
    price = frame["day_ahead_gbp_per_mwh"].to_numpy()
    edges = (
        np.quantile(price, np.linspace(0, 1, N_LEVEL_BINS + 1)[1:-1])
        if level_edges is None
        else level_edges
    )
    if with_price:
        rank_bin = np.minimum(
            (frame["rank_pct"].to_numpy() * N_RANK_BINS).astype(int), N_RANK_BINS - 1
        )
        level_bin = np.digitize(price, edges)
        rank_hot = (rank_bin[:, None] == np.arange(N_RANK_BINS)[None, :]).astype(float)
        level_hot = (level_bin[:, None] == np.arange(N_LEVEL_BINS)[None, :]).astype(float)
        pair = (rank_hot[:, :, None] * level_hot[:, None, :]).reshape(len(frame), -1)
        columns += [rank_hot, level_hot, pair]
    return np.hstack(columns), edges


def out_of_fold(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Predict every row from a fit that never saw the row's fold.

    Args:
        frame: A battery frame from `battery_frame`.

    Returns:
        The frame with `time_of_day` and `price_rank` prediction columns in megawatts.
    """
    frame = frame.with_columns(
        day_mean_price=pl.col("day_ahead_gbp_per_mwh").mean().over("date"),
        day_price_range=pl.col("day_ahead_gbp_per_mwh").max().over("date")
        - pl.col("day_ahead_gbp_per_mwh").min().over("date"),
    )
    predictions = {name: np.zeros(frame.height) for name in ARMS}
    fold = frame["fold"].to_numpy()
    target = frame["output_mw"].to_numpy()
    for held_out in range(4):
        train = fold != held_out
        for name, with_price in (("time_of_day", False), ("price_rank", True)):
            x_train, edges = design_matrix(
                frame=frame.filter(pl.Series(train)), level_edges=None, with_price=with_price
            )
            coefficients, *_ = np.linalg.lstsq(x_train, target[train], rcond=None)
            x_test, _ = design_matrix(
                frame=frame.filter(pl.Series(~train)), level_edges=edges, with_price=with_price
            )
            predictions[name][~train] = x_test @ coefficients
        booster = xgb.train(
            XGB_PARAMETERS,
            xgb.DMatrix(
                frame.filter(pl.Series(train)).select(XGB_FEATURES).to_numpy(), target[train]
            ),
            num_boost_round=XGB_ROUNDS,
        )
        predictions["xgboost_price"][~train] = booster.predict(
            xgb.DMatrix(frame.filter(pl.Series(~train)).select(XGB_FEATURES).to_numpy())
        )
    return frame.with_columns(**{name: pl.Series(values) for name, values in predictions.items()})


def month_resampled_interval(*, monthly: pl.DataFrame, column: str) -> tuple[float, float, float]:
    """Interval for a mean of a per-row quantity by resampling whole calendar months.

    Args:
        monthly: One row per month with `{column}_sum` and `n` (the rows in the month).
        column: The quantity.

    Returns:
        The mean over all rows, and the 2.5th and 97.5th percentiles of the resampled means.
    """
    totals = monthly[f"{column}_sum"].to_numpy()
    counts = monthly["n"].to_numpy().astype(float)
    rng = np.random.default_rng(RESAMPLE_SEED)
    draws = rng.integers(0, len(totals), size=(N_RESAMPLES, len(totals)))
    resampled = totals[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.percentile(resampled, [2.5, 97.5])
    return float(totals.sum() / counts.sum()), float(low), float(high)


def main() -> None:
    """Run the held-out comparison for every battery and write the tables and report."""
    lines = [
        "## Rung 2: predict the battery from prices alone",
        "",
        (
            "Four contiguous three-month folds. Errors are % of the battery's p99 absolute output. "
            "Intervals resample whole calendar months (12 months, 2,000 resamples)."
        ),
        "",
    ]
    predictions, interval_rows = [], []
    monthly_by_battery = {}
    for bmu_id in BATTERIES:
        frame = out_of_fold(frame=battery_frame(bmu_id=bmu_id))
        scale = p99_output_mw(frame=frame)
        frame = frame.with_columns(
            **{
                f"error_{arm}": (pl.col("output_mw") - pl.col(arm)).abs() / scale * PERCENT
                for arm in ARMS
            }
        ).with_columns(
            **{
                f"difference_{treatment}_{reference}": pl.col(f"error_{treatment}")
                - pl.col(f"error_{reference}")
                for treatment, reference, _ in CONTRASTS
            }
        )
        predictions.append(
            frame.with_columns(bmu_id=pl.lit(bmu_id)).select(
                "bmu_id",
                "time",
                "fold",
                "month",
                "output_mw",
                *ARMS,
                *[f"error_{arm}" for arm in ARMS],
            )
        )
        monthly = (
            frame.group_by("month")
            .agg(
                pl.len().alias("n"),
                *[
                    pl.col(c).sum().alias(f"{c}_sum")
                    for c in frame.columns
                    if c.startswith(("error_", "difference_"))
                ],
            )
            .sort("month")
        )
        monthly_by_battery[bmu_id] = monthly
        lines.append(f"### {NAMES[bmu_id]} ({bmu_id}); p99 absolute output {scale:.1f} MW")
        lines.append("")
        sse_mean = float(((frame["output_mw"] - frame["output_mw"].mean()) ** 2).sum())
        for arm in ARMS:
            mean, low, high = month_resampled_interval(monthly=monthly, column=f"error_{arm}")
            interval_rows.append(
                {"bmu_id": bmu_id, "quantity": arm, "mean": mean, "lower_95": low, "upper_95": high}
            )
            sse = float(((frame["output_mw"] - frame[arm]) ** 2).sum())
            lines.append(
                f"- {arm}: {mean:.2f} [{low:.2f}, {high:.2f}] % of p99; "
                f"out-of-fold R2 (vs the year mean) {1 - sse / sse_mean:.3f}"
            )
        for treatment, reference, kind in CONTRASTS:
            column = f"difference_{treatment}_{reference}"
            mean, low, high = month_resampled_interval(monthly=monthly, column=column)
            label = f"{treatment} minus {reference} ({kind})"
            interval_rows.append(
                {
                    "bmu_id": bmu_id,
                    "quantity": label,
                    "mean": mean,
                    "lower_95": low,
                    "upper_95": high,
                }
            )
            better = int((monthly[f"{column}_sum"] < 0).sum())
            lines.append(
                f"- {label}: {mean:.2f} [{low:.2f}, {high:.2f}] % of p99; "
                f"negative in {better} of {monthly.height} months"
            )
        lines.append("")
    # The four batteries' average difference, resampling the same months together.
    value_columns = [c for c in monthly_by_battery[BATTERIES[0]].columns if c.endswith("_sum")]
    pooled = (
        pl.concat(list(monthly_by_battery.values()))
        .group_by("month")
        .agg(pl.col("n").sum(), *[pl.col(c).sum() for c in value_columns])
        .sort("month")
    )
    lines += ["### The four batteries pooled (errors already in % of each battery's own p99)", ""]
    for arm in ARMS:
        mean, low, high = month_resampled_interval(monthly=pooled, column=f"error_{arm}")
        interval_rows.append(
            {"bmu_id": "pooled", "quantity": arm, "mean": mean, "lower_95": low, "upper_95": high}
        )
        lines.append(f"- {arm}: {mean:.2f} [{low:.2f}, {high:.2f}] % of p99")
    for treatment, reference, kind in CONTRASTS:
        mean, low, high = month_resampled_interval(
            monthly=pooled, column=f"difference_{treatment}_{reference}"
        )
        label = f"{treatment} minus {reference} ({kind})"
        interval_rows.append(
            {"bmu_id": "pooled", "quantity": label, "mean": mean, "lower_95": low, "upper_95": high}
        )
        lines.append(f"- {label}: {mean:.2f} [{low:.2f}, {high:.2f}] % of p99")
    lines.append("")
    pl.concat(predictions).write_parquet(OUTPUT_DIR / "rung2_out_of_fold.parquet")
    pl.DataFrame(interval_rows).write_parquet(OUTPUT_DIR / "rung2_intervals.parquet")
    (OUTPUT_DIR / "report_rung2.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
