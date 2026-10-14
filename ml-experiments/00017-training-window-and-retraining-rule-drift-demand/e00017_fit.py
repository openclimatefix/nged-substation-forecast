"""Fit experiment 17's arms and write monthly rolling-origin predictions, one substation at a time.

Each validation month from `FIRST_MONTH` to `LAST_MONTH` is forecast by models trained on rows that
end `PURGE` before the month starts. The arms differ only in the training window, the row weights,
and how often the model is refitted (see the experiment's `result.md`). One file is written per
series, and a series whose file exists is skipped.

    uv run python ml-experiments/00017-*/e00017_fit.py
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import xgboost as xgb
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SIXTH_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00006-stage1a-demand-input-feature-sweep"
"""Experiment 6's private folder, whose ERA5 feature table this experiment reads."""
EXPERIMENT_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00017-training-window-and-retraining-rule-drift-demand"
)
FITS_DIR: Final[Path] = EXPERIMENT_DIR / "fits"
TRAIN_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
FIRST_MONTH: Final[datetime] = datetime(2022, 1, 1, tzinfo=UTC)
LAST_MONTH: Final[datetime] = datetime(2026, 6, 1, tzinfo=UTC)
"""The last validation month. No experiment scores a valid time on or after 2026-07-01."""
PURGE: Final[timedelta] = timedelta(days=7)
MIN_TRAIN_ROWS: Final[int] = 17_520
"""A series needs a year of observed half-hours before the purge to enter."""
N_ROUNDS: Final[int] = 500
CHRISTMAS_WINDOW_DAYS: Final[int] = 14
PARAMETERS: Final[dict[str, float | int | str]] = {
    "eta": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "colsample_bynode": 1.0,
    "tree_method": "hist",
    "objective": "reg:absoluteerror",
    "seed": 0,
}
COLUMNS: Final[tuple[str, ...]] = (
    "shortwave",
    "longwave",
    "temperature_2m",
    "dew_point_temperature_2m",
    "pressure_surface",
    "precipitation",
    "wind_speed_10m",
    "solar_zenith",
    "solar_azimuth",
    "hour_local",
    "day_of_week_local",
    "covid_indicator",
    "days_to_christmas_window",
)
"""Experiment 6's Primary input set: weather block, clock, COVID flag, windowed Christmas days."""
WINDOW_DAYS: Final[dict[str, int]] = {"W6": 183, "W12": 365, "W18": 548}
HALF_LIFE_YEARS: Final[dict[str, float]] = {"H0.5": 0.5, "H1": 1.0, "H2": 2.0}
ARMS: Final[tuple[str, ...]] = ("A", *WINDOW_DAYS, *HALF_LIFE_YEARS, "F", "Y")
"""The arms that are fitted models. K (the correction) is computed from A's predictions."""
YEARLY_EVERY_MONTHS: Final[int] = 12


def _christmas_window() -> pl.Expr:
    """Return days to the nearest 25 December, signed, null outside the 14-day window."""
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


def series_ids() -> list[int]:
    """Return the Primary substations' ids."""
    series = pl.read_parquet(SIXTH_DIR / "inputs" / "series.parquet")
    return sorted(series.filter(pl.col("substation_type") == "Primary")["time_series_id"].to_list())


def months() -> list[datetime]:
    """Return the first instant of each validation month."""
    out, month = [], FIRST_MONTH
    while month <= LAST_MONTH:
        out.append(month)
        month = datetime(month.year + month.month // 12, month.month % 12 + 1, 1, tzinfo=UTC)
    return out


def _matrix(*, frame: pl.DataFrame) -> xgb.DMatrix:
    return xgb.DMatrix(frame.select([pl.col(c).cast(pl.Float32) for c in COLUMNS]).to_numpy())


def _fit(*, frame: pl.DataFrame, train_end: datetime, arm: str, nthread: int) -> xgb.Booster:
    """Fit one arm on the rows before `train_end`; the window and weights depend on the arm."""
    start = TRAIN_START
    if arm in WINDOW_DAYS:
        start = max(start, train_end - timedelta(days=WINDOW_DAYS[arm]))
    rows = frame.filter(pl.col("valid_time") >= start, pl.col("valid_time") < train_end)
    matrix = _matrix(frame=rows)
    matrix.set_label(rows["target"].to_numpy())
    if arm in HALF_LIFE_YEARS:
        age_years = (train_end - rows["valid_time"]).dt.total_days().to_numpy() / 365.25
        matrix.set_weight(0.5 ** (age_years / HALF_LIFE_YEARS[arm]))
    return xgb.train({**PARAMETERS, "nthread": nthread}, matrix, num_boost_round=N_ROUNDS)


def fit_series(*, time_series_id: int, nthread: int = 1) -> str:
    """Fit every arm for each month of one series and write its predictions."""
    path = FITS_DIR / f"{time_series_id}.parquet"
    if path.exists():
        return f"{path.name}: exists"
    frame = (
        pl.scan_parquet(SIXTH_DIR / "inputs" / "features.parquet")
        .filter(pl.col("time_series_id") == time_series_id, pl.col("target").is_not_null())
        .with_columns(days_to_christmas_window=_christmas_window())
        .select("time_series_id", "valid_time", "target", *COLUMNS)
        .collect()
    )
    pieces: list[pl.DataFrame] = []
    frozen: xgb.Booster | None = None
    yearly: xgb.Booster | None = None
    month_number = 0
    for month in months():
        train_end = month - PURGE
        if frame.filter(pl.col("valid_time") < train_end).height < MIN_TRAIN_ROWS:
            continue
        next_month = datetime(month.year + month.month // 12, month.month % 12 + 1, 1, tzinfo=UTC)
        valid = frame.filter(pl.col("valid_time") >= month, pl.col("valid_time") < next_month)
        if valid.is_empty():
            continue
        matrix = _matrix(frame=valid)
        predictions: dict[str, np.ndarray] = {}
        for arm in ("A", *WINDOW_DAYS, *HALF_LIFE_YEARS):
            booster = _fit(frame=frame, train_end=train_end, arm=arm, nthread=nthread)
            predictions[arm] = booster.predict(matrix)
            if arm == "A":
                if frozen is None:
                    frozen = booster
                if yearly is None or month_number % YEARLY_EVERY_MONTHS == 0:
                    yearly = booster
        predictions["F"] = frozen.predict(matrix)  # ty: ignore[unresolved-attribute]
        predictions["Y"] = yearly.predict(matrix)  # ty: ignore[unresolved-attribute]
        pieces.append(
            valid.select("time_series_id", "valid_time", "target").with_columns(
                pl.Series(arm, values.astype(np.float32)) for arm, values in predictions.items()
            )
        )
        month_number += 1
    if not pieces:
        return f"{time_series_id}: never has a year of training data"
    FITS_DIR.mkdir(parents=True, exist_ok=True)
    out = pl.concat(pieces)
    out.write_parquet(path.with_suffix(".tmp"))
    path.with_suffix(".tmp").rename(path)
    return f"{path.name}: {out.height:,} rows, {len(pieces)} months"


def main() -> None:
    """Fit every Primary substation, 10 processes with one thread each (slot B)."""
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=10, mp_context=context) as pool:
        for line in pool.map(_run, series_ids()):
            print(line, flush=True)


def _run(time_series_id: int) -> str:
    return fit_series(time_series_id=time_series_id)


if __name__ == "__main__":
    main()
