r"""Fit experiment 15's four arms for each demand series and write the validation predictions.

Every arm is the experiment 00006 Primary input set (weather block, hour and weekday, COVID
indicator, days to Christmas inside a 14-day window) with recency weights and the median objective
(`reg:absoluteerror`), and the arms differ only in the added columns:

- `BASE`: nothing added.
- `PRICE`: the price of the target half-hour.
- `CONTROL`: national demand and wind of the target half-hour.
- `SHUFFLED`: the price of another week's same weekday and half-hour.

One unit is a fold, a series, and a seed. A unit writes one parquet file and is skipped if the file
exists, so a crash costs one unit. Training ends one purge before the fold's validation start.

    uv run python ml-experiments/00015-same-time-price-stage1-demand/e00015_fit.py \\
        --fold select_2024
"""

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00015_inputs as inputs
import numpy as np
import polars as pl
import xgboost as xgb

FITS_DIR: Final[Path] = inputs.EXPERIMENT_DIR / "fits"
N_ROUNDS: Final[int] = 500
PURGE: Final[timedelta] = timedelta(days=7)
MIN_TRAIN_ROWS: Final[int] = 17_520
"""A series needs a year of observed half-hours before the purge to be fitted in a fold."""
RECENCY_HALF_LIFE_YEARS: Final[float] = 2.0
CHRISTMAS_WINDOW_DAYS: Final[int] = 14
TRAIN_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
PARAMETERS: Final[dict[str, float | int | str]] = {
    "eta": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "colsample_bynode": 1.0,
    "tree_method": "hist",
    "objective": "reg:absoluteerror",
}
"""Identical in every arm, and the settings experiment 00006 used. Column subsampling is off so
that an arm with more columns gets no free win."""
WEATHER_BLOCK: Final[tuple[str, ...]] = (
    "shortwave",
    "longwave",
    "temperature_2m",
    "dew_point_temperature_2m",
    "pressure_surface",
    "precipitation",
    "wind_speed_10m",
    "solar_zenith",
    "solar_azimuth",
)
BASE_COLUMNS: Final[tuple[str, ...]] = (
    *WEATHER_BLOCK,
    "hour_local",
    "day_of_week_local",
    "covid_indicator",
    "days_to_christmas_window",
)
ARM_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "BASE": BASE_COLUMNS,
    "PRICE": (*BASE_COLUMNS, *inputs.PRICE_COLUMNS),
    "CONTROL": (*BASE_COLUMNS, *inputs.CONTROL_COLUMNS),
    "SHUFFLED": (*BASE_COLUMNS, *inputs.SHUFFLED_COLUMNS),
}


@dataclass(frozen=True)
class Fold:
    """A validation window and the training window that ends a purge before it."""

    valid_start: datetime
    valid_end: datetime

    @property
    def train_end(self) -> datetime:
        """The exclusive end of training: one purge before validation starts."""
        return self.valid_start - PURGE


FOLDS: Final[dict[str, Fold]] = {
    "select_2024": Fold(datetime(2024, 7, 1, tzinfo=UTC), datetime(2025, 7, 1, tzinfo=UTC)),
    "confirm_2025": Fold(datetime(2025, 7, 1, tzinfo=UTC), datetime(2026, 7, 1, tzinfo=UTC)),
}
"""The selection fold decides the verdict. The confirmation fold is scored once."""


def christmas_window() -> pl.Expr:
    """Return the signed days to 25 December inside 14 days of it, and null elsewhere."""
    local = pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.date()
    year = local.dt.year()
    offsets = [(pl.date(year + shift, 12, 25) - local).dt.total_days() for shift in (-1, 0, 1)]
    nearest = pl.min_horizontal([o.abs() for o in offsets])
    signed = (
        pl.when(offsets[0].abs() == nearest)
        .then(offsets[0])
        .when(offsets[1].abs() == nearest)
        .then(offsets[1])
        .otherwise(offsets[2])
    )
    return pl.when(nearest <= CHRISTMAS_WINDOW_DAYS).then(signed).otherwise(None)


def unit_path(*, fold_name: str, unit: int, seed: int) -> Path:
    """Return where a unit's predictions are written."""
    return FITS_DIR / fold_name / f"seed{seed}" / f"{unit}.parquet"


def run_unit(*, fold_name: str, unit: int, seed: int) -> str:
    """Fit every arm for one series and seed, skipping a unit whose file exists."""
    path = unit_path(fold_name=fold_name, unit=unit, seed=seed)
    if path.exists():
        return f"{path}: exists"
    fold = FOLDS[fold_name]
    frame = (
        pl.scan_parquet(inputs.INPUTS_PATH)
        .filter(pl.col("time_series_id") == unit)
        .collect()
        .with_columns(days_to_christmas_window=christmas_window())
    )
    train = frame.filter(
        pl.col("target").is_not_null(),
        pl.col("valid_time") >= TRAIN_START,
        pl.col("valid_time") < fold.train_end,
    )
    if train.height < MIN_TRAIN_ROWS:
        return f"{unit}: only {train.height} training rows in {fold_name}"
    valid = frame.filter(
        pl.col("target").is_not_null(),
        pl.col("valid_time") >= fold.valid_start,
        pl.col("valid_time") < fold.valid_end,
    )
    age_years = (fold.train_end - train["valid_time"]).dt.total_days() / 365.25
    weights = 0.5 ** (age_years.to_numpy() / RECENCY_HALF_LIFE_YEARS)
    out = valid.select("time_series_id", "valid_time", "target")
    for arm, columns in ARM_COLUMNS.items():
        matrix = xgb.DMatrix(train.select(columns).cast(pl.Float32).to_numpy(), weight=weights)
        matrix.set_label(train["target"].to_numpy())
        booster = xgb.train(
            {**PARAMETERS, "seed": seed, "nthread": 1}, matrix, num_boost_round=N_ROUNDS
        )
        predicted = booster.predict(xgb.DMatrix(valid.select(columns).cast(pl.Float32).to_numpy()))
        out = out.with_columns(pl.Series(arm, predicted.astype(np.float32)))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(path.with_suffix(".tmp"))
    path.with_suffix(".tmp").rename(path)
    return f"{path}: {out.height:,} rows"


def main() -> None:
    """Run every series and seed of one fold in parallel processes, one thread per fit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=sorted(FOLDS), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    series = pl.read_parquet(inputs.EXPERIMENT_00006_DIR / "series.parquet")["time_series_id"]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = [
            pool.submit(run_unit, fold_name=args.fold, unit=int(unit), seed=seed)
            for seed in args.seeds
            for unit in series.to_list()
        ]
        for future in futures:
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
