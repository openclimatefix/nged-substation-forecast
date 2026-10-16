"""Fit the residual's XGBoost model with refits on the 16th of each month, one series at a time.

Experiment 17's arm `H0.5` (recency weights with a half-life of 0.5 years, refitted on the 1st of
each month) gave the stored residual. This script repeats that arm with every setting unchanged,
except that the origin of each refit is the 16th, so the two models' refits are half a month
apart. A model with origin `o` trains on rows before `o` minus 7 days and predicts `o` up to the
next origin. One file is written per series, and a series whose file exists is skipped.

    uv run python ml-experiments/00071-*/e00071_fit_offset.py fit
    uv run python ml-experiments/00071-*/e00071_fit_offset.py check
"""

from __future__ import annotations

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import xgboost as xgb
from e00071_common import SERIES_IDS, WORK_DIR
from studies.sources import ML_EXPERIMENTS_DATA_DIR

FEATURES_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00006-stage1a-demand-input-feature-sweep"
    / "inputs"
    / "features.parquet"
)
"""Experiment 6's ERA5 feature table, which experiment 17 fitted on."""
EXPERIMENT_17_FITS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00017-training-window-and-retraining-rule-drift-demand" / "fits"
)
FITS_DIR: Final[Path] = WORK_DIR / "offset_fits"
TRAIN_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
FIRST_ORIGIN: Final[datetime] = datetime(2021, 12, 16, tzinfo=UTC)
LAST_ORIGIN: Final[datetime] = datetime(2026, 6, 16, tzinfo=UTC)
SCORE_END: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
"""No experiment scores a valid time on or after 2026-07-01."""
PURGE: Final[timedelta] = timedelta(days=7)
MIN_TRAIN_ROWS: Final[int] = 17_520
N_ROUNDS: Final[int] = 500
HALF_LIFE_YEARS: Final[float] = 0.5
CHRISTMAS_WINDOW_DAYS: Final[int] = 14
N_WORKERS: Final[int] = 4
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
"""Experiment 17's input columns, copied so the two models differ only in refit date."""


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


def origins(*, first: datetime, last: datetime, day: int) -> list[datetime]:
    """Return the first instant of the given `day` of each month from `first` to `last`."""
    out, month = [], datetime(first.year, first.month, day, tzinfo=UTC)
    while month <= last:
        out.append(month)
        month = datetime(month.year + month.month // 12, month.month % 12 + 1, day, tzinfo=UTC)
    return out


def _matrix(*, frame: pl.DataFrame) -> xgb.DMatrix:
    return xgb.DMatrix(frame.select([pl.col(c).cast(pl.Float32) for c in COLUMNS]).to_numpy())


def _fit(*, frame: pl.DataFrame, train_end: datetime, nthread: int) -> xgb.Booster:
    """Fit on the rows before `train_end`, weighted by age with a half-life of 0.5 years."""
    rows = frame.filter(pl.col("valid_time") >= TRAIN_START, pl.col("valid_time") < train_end)
    matrix = _matrix(frame=rows)
    matrix.set_label(rows["target"].to_numpy())
    age_years = (train_end - rows["valid_time"]).dt.total_days().to_numpy() / 365.25
    matrix.set_weight(0.5 ** (age_years / HALF_LIFE_YEARS))
    return xgb.train({**PARAMETERS, "nthread": nthread}, matrix, num_boost_round=N_ROUNDS)


def _series_frame(*, time_series_id: int) -> pl.DataFrame:
    return (
        pl.scan_parquet(FEATURES_PATH)
        .filter(pl.col("time_series_id") == time_series_id, pl.col("target").is_not_null())
        .with_columns(days_to_christmas_window=_christmas_window())
        .select("time_series_id", "valid_time", "target", *COLUMNS)
        .collect()
    )


def predict_origin(*, frame: pl.DataFrame, origin: datetime, end: datetime) -> pl.DataFrame | None:
    """Predictions for `origin` up to `end`, or None if the series has too little history yet."""
    train_end = origin - PURGE
    if frame.filter(pl.col("valid_time") < train_end).height < MIN_TRAIN_ROWS:
        return None
    valid = frame.filter(pl.col("valid_time") >= origin, pl.col("valid_time") < end)
    if valid.is_empty():
        return None
    booster = _fit(frame=frame, train_end=train_end, nthread=1)
    prediction = booster.predict(_matrix(frame=valid)).astype(np.float32)
    return valid.select("time_series_id", "valid_time", "target").with_columns(
        H0_5=pl.Series(prediction)
    )


def fit_series(*, time_series_id: int) -> str:
    """Fit every refit origin of one series and write its predictions."""
    path = FITS_DIR / f"{time_series_id}.parquet"
    if path.exists():
        return f"{path.name}: exists"
    frame = _series_frame(time_series_id=time_series_id)
    starts = origins(first=FIRST_ORIGIN, last=LAST_ORIGIN, day=16)
    pieces = []
    for origin, following in zip(starts, [*starts[1:], SCORE_END], strict=True):
        piece = predict_origin(frame=frame, origin=origin, end=min(following, SCORE_END))
        if piece is not None:
            pieces.append(piece)
    if not pieces:
        return f"{time_series_id}: never has a year of training data"
    FITS_DIR.mkdir(parents=True, exist_ok=True)
    out = pl.concat(pieces)
    out.write_parquet(path.with_suffix(".tmp"))
    path.with_suffix(".tmp").rename(path)
    return f"{path.name}: {out.height:,} rows, {len(pieces)} refits"


def check() -> None:
    """Refit experiment 17's origin of 2025-03-01 for series 3 and compare it with its stored fit.

    The rows must match to within float32 rounding of a one-thread refit; a larger gap means this
    script does not repeat experiment 17's model.
    """
    origin = datetime(2025, 3, 1, tzinfo=UTC)
    frame = _series_frame(time_series_id=3)
    mine = predict_origin(frame=frame, origin=origin, end=datetime(2025, 4, 1, tzinfo=UTC))
    if mine is None:
        raise SystemExit("STOP: series 3 has no fit at 2025-03-01")
    stored = pl.read_parquet(EXPERIMENT_17_FITS / "3.parquet").select(
        "valid_time", stored=pl.col("H0.5")
    )
    joined = mine.join(stored, on="valid_time", validate="1:1")
    difference = (joined["H0_5"] - joined["stored"]).abs()
    print(
        f"{joined.height} rows; largest difference {difference.max():.6f}, "
        f"mean {difference.mean():.6f}"
    )
    print(f"mean absolute target {joined['target'].abs().mean():.4f}")
    typical = float(np.mean(np.abs(joined["target"].to_numpy())))
    if not float(np.max(difference.to_numpy())) <= 1e-3 * typical:
        raise SystemExit("STOP: this script does not repeat experiment 17's model")


def _run(time_series_id: int) -> str:
    return fit_series(time_series_id=time_series_id)


def main() -> None:
    """Fit every series (`fit`) or check the repetition of experiment 17 (`check`)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("fit", "check"))
    args = parser.parse_args()
    if args.stage == "check":
        check()
        return
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=context) as pool:
        for line in pool.map(_run, SERIES_IDS):
            print(line, flush=True)


if __name__ == "__main__":
    main()
