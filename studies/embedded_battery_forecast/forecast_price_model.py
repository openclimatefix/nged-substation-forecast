"""The price model: a forecast of tomorrow's N2EX day-ahead price, available at 06:00 UTC.

The `DA-early` issue time is before the N2EX auction result, so a forecaster there has a price
forecast, not the price. This script fits an XGBoost point model of the half-hourly day-ahead price
given what is known at 06:00 UTC on the day before the target day:

- the half-hour of day and the day of week;
- the NESO wind forecast of the target hour, as published at or before 06:00 on the day before
  (`forecast_inputs.wind_forecast_at_issue`);
- the price seven days earlier (the `naive` price) and one day earlier (the auction for the day
  before the target day cleared at about 10:00 UTC two days before the target day, so it is known);
- the mean price of the seven days before the target day.

The auction's delivery day is the UK local day, so in summer the UTC hour 23:00 to 24:00 of the
day before the target day is priced by the target day's own auction, which has not cleared at
06:00. In the one-day-earlier price and the seven-day mean, that hour takes the `naive` price.

The NESO demand forecasts are not used: the last publication before 06:00 reaches only about 03:00
on the target day.

**The forecast is nested, so no training row's price forecast depends on the held-out fold.** For
each held-out fold `f`, the forecast of fold `f` comes from a model trained on the other three
folds, and the forecast of each other fold `g` comes from a model trained on the two folds that
are neither `f` nor `g`. The quantile model of fold `f` is fitted on those forecasts
(`forecast_fit.run_arm`).

Run: `OMP_NUM_THREADS=2 uv run python studies/embedded_battery_forecast/forecast_price_model.py`.
Writes `price_model_by_fold.parquet` and `price_model_report.md`.
"""

from typing import Final

import numpy as np
import polars as pl
import xgboost as xgb
from forecast_inputs import (
    SCORING_START,
    half_hour_grid,
    price_sources,
    wind_forecast_at_issue,
)
from studies.battery_forecast import next_uk_day_hour
from studies.battery_market import FOLD_MONTHS
from studies.bootstrap import bootstrap_row_difference
from studies.cross_validation import PRIMARY_HYPER_PARAMETERS, booster_parameters
from studies.sources import EMBEDDED_BATTERY_FORECAST_DIR

PRICE_MODEL_PATH: Final = EMBEDDED_BATTERY_FORECAST_DIR / "price_model_by_fold.parquet"
FEATURES: Final[tuple[str, ...]] = (
    "tod",
    "day_of_week",
    "wind_forecast_mw",
    "price_naive",
    "price_lag_1_day",
    "price_mean_previous_7_days",
)
N_FOLDS: Final[int] = 4
WEAK_RATIO: Final[float] = 0.8
"""A mean absolute error above this share of the `naive` price's marks the price model as weak."""


def price_frame() -> pl.DataFrame:
    """Return the price model's features and target on the half-hour grid.

    The `price_known` column replaces the next UK day's hour on every day, including the days
    where that hour was public in time. That is conservative, and it keeps the rule one line.

    Returns:
        Columns `time`, `fold`, `month`, `price_actual` (the target), and `FEATURES`.
    """
    grid = half_hour_grid()
    prices = (
        price_sources(grid=grid)
        .select("time", "price_actual", "price_naive")
        .with_columns(
            price_known=pl.when(next_uk_day_hour(time=pl.col("time")))
            .then(pl.col("price_naive"))
            .otherwise(pl.col("price_actual"))
        )
    )
    wind = wind_forecast_at_issue(issue="DA-early", grid=grid)
    day = pl.col("time").dt.truncate("1d")
    daily = (
        prices.with_columns(day=day)
        .group_by("day")
        .agg(day_mean=pl.col("price_known").mean())
        .sort("day")
        .with_columns(
            price_mean_previous_7_days=pl.col("day_mean").rolling_mean(
                window_size=7, min_samples=7
            ),
            day=pl.col("day") + pl.duration(days=1),
        )
        .select("day", "price_mean_previous_7_days")
    )
    lag_one = prices.select(
        time=pl.col("time") + pl.duration(days=1), price_lag_1_day=pl.col("price_known")
    )
    return (
        prices.join(wind, on="time", how="left")
        .join(lag_one, on="time", how="left")
        .with_columns(day=day)
        .join(daily, on="day", how="left")
        .with_columns(
            tod=pl.col("time").dt.hour().cast(pl.Int64) * 2 + pl.col("time").dt.minute() // 30,
            day_of_week=pl.col("time").dt.weekday().cast(pl.Int64),
            month=pl.col("time").dt.strftime("%Y-%m"),
            fold=pl.col("time").dt.month().replace_strict(FOLD_MONTHS, return_dtype=pl.Int64),
        )
        .select("time", "fold", "month", "price_actual", *FEATURES)
    )


def _fit_predict(*, frame: pl.DataFrame, train_folds: list[int], predict_fold: int) -> np.ndarray:
    """Fit on some folds' rows and predict one fold's rows."""
    train = frame.filter(pl.col("fold").is_in(train_folds))
    parameters = booster_parameters(hyper_parameters=PRIMARY_HYPER_PARAMETERS, seed=0)
    parameters["nthread"] = 2
    parameters["objective"] = "reg:absoluteerror"
    model = xgb.train(
        parameters,
        xgb.DMatrix(train.select(FEATURES).to_numpy(), label=train["price_actual"].to_numpy()),
        num_boost_round=PRIMARY_HYPER_PARAMETERS["num_boost_round"],
    )
    test = frame.filter(pl.col("fold") == predict_fold)
    return model.predict(xgb.DMatrix(test.select(FEATURES).to_numpy()))


def nested_forecasts(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the price forecast of every half-hour as it stood for each held-out fold.

    Args:
        frame: The frame `price_frame` returns.

    Returns:
        Columns `outer_fold`, `time`, and `price_model`: for each `outer_fold`, one row per
        half-hour. The rows of fold `outer_fold` itself are the plain out-of-fold forecast.
    """
    pieces = []
    for outer in range(N_FOLDS):
        others = [f for f in range(N_FOLDS) if f != outer]
        for target_fold in range(N_FOLDS):
            train_folds = (
                others if target_fold == outer else [f for f in others if f != target_fold]
            )
            predicted = _fit_predict(frame=frame, train_folds=train_folds, predict_fold=target_fold)
            pieces.append(
                frame.filter(pl.col("fold") == target_fold)
                .select("time")
                .with_columns(
                    outer_fold=pl.lit(outer, dtype=pl.Int64),
                    price_model=pl.Series(predicted, dtype=pl.Float64),
                )
            )
    return pl.concat(pieces).select("outer_fold", "time", "price_model")


def model_price_by_fold() -> dict[int, pl.DataFrame]:
    """Return the saved nested forecasts as `forecast_fit.run_arm` reads them."""
    saved = pl.read_parquet(PRICE_MODEL_PATH)
    return {
        outer: saved.filter(pl.col("outer_fold") == outer).select("time", "price_model")
        for outer in range(N_FOLDS)
    }


def plain_out_of_fold(*, saved: pl.DataFrame, frame: pl.DataFrame) -> pl.DataFrame:
    """Return, for each half-hour, the forecast from the model that did not see its fold."""
    return (
        saved.join(frame.select("time", "fold"), on="time")
        .filter(pl.col("outer_fold") == pl.col("fold"))
        .select("time", "price_model")
    )


def plain_forecast() -> pl.DataFrame:
    """Return the plain out-of-fold `price_model` series, which decides the scored rows."""
    frame = price_frame()
    return plain_out_of_fold(saved=pl.read_parquet(PRICE_MODEL_PATH), frame=frame)


def report_lines(*, frame: pl.DataFrame, plain: pl.DataFrame) -> list[str]:
    """Return the price model's error against the `naive` and one-day-lag prices."""
    scored = (
        frame.join(plain, on="time")
        .filter(pl.col("time") >= SCORING_START)
        .drop_nulls(["price_actual", "price_naive", "price_model", "price_lag_1_day"])
    )
    error_model = (scored["price_model"] - scored["price_actual"]).abs().to_numpy()
    error_naive = (scored["price_naive"] - scored["price_actual"]).abs().to_numpy()
    error_lag = (scored["price_lag_1_day"] - scored["price_actual"]).abs().to_numpy()
    months = scored["month"].to_numpy()
    versus_naive = bootstrap_row_difference(values=error_model - error_naive, months=months)
    ratio = float(error_model.mean() / error_naive.mean())
    rmse_ratio = float(
        np.sqrt(np.mean((scored["price_model"] - scored["price_actual"]).to_numpy() ** 2))
        / np.sqrt(np.mean((scored["price_naive"] - scored["price_actual"]).to_numpy() ** 2))
    )
    wind_null = int(frame["wind_forecast_mw"].null_count())
    lines = [
        "# The price model",
        "",
        (
            f"- Rows scored (from {SCORING_START:%Y-%m-%d}, all features and the target "
            f"present): {scored.height}. Wind forecast nulls on the whole grid: {wind_null} of "
            f"{frame.height}."
        ),
        f"- Features: {', '.join(FEATURES)}.",
        "- Hyperparameters: `PRIMARY_HYPER_PARAMETERS`, objective `reg:absoluteerror`, seed 0.",
        "",
        "| Forecast | Mean absolute error, GBP per MWh | Ratio to `naive` |",
        "|---|---|---|",
        f"| `naive` (price seven days earlier) | {error_naive.mean():.2f} | 1.000 |",
        (
            f"| Price one day earlier | {error_lag.mean():.2f} | "
            f"{error_lag.mean() / error_naive.mean():.3f} |"
        ),
        f"| `model` (XGBoost, nested out of fold) | {error_model.mean():.2f} | {ratio:.3f} |",
        "",
        (
            "- Mean absolute error, `model` minus `naive`: "
            f"{versus_naive['difference']:+.2f} GBP per MWh "
            f"(95% interval {versus_naive['lower_95']:+.2f} to {versus_naive['upper_95']:+.2f}, "
            f"{versus_naive['n_months']} months resampled)."
        ),
        f"- Root-mean-square error ratio, `model` to `naive`: {rmse_ratio:.3f}.",
        (
            f"- Verdict: the price model is {'weak' if ratio > WEAK_RATIO else 'not weak'} by "
            f"the plan's rule (a mean-absolute-error ratio above {WEAK_RATIO} marks it weak)."
        ),
        "",
        "| Month | Mean absolute error of `model` | of `naive` | Ratio |",
        "|---|---|---|---|",
    ]
    for month in sorted(set(months.tolist())):
        selected = months == month
        model_error = error_model[selected].mean()
        naive_error = error_naive[selected].mean()
        lines.append(
            f"| {month} | {model_error:.2f} | {naive_error:.2f} | {model_error / naive_error:.3f} |"
        )
    return lines


def main() -> None:
    """Fit the nested price forecasts, save them, and write `price_model_report.md`."""
    frame = price_frame()
    saved = nested_forecasts(frame=frame)
    saved.write_parquet(PRICE_MODEL_PATH)
    lines = report_lines(frame=frame, plain=plain_out_of_fold(saved=saved, frame=frame))
    (EMBEDDED_BATTERY_FORECAST_DIR / "price_model_report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
