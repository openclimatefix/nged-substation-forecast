"""Refit experiment 17's `H0.5` arm with zero weight on detected switching intervals.

The model, inputs, hyperparameters, monthly refit, 7-day purge, and recency weights (half-life 0.5
years) are experiment 17's `H0.5` arm. The only change is a mask: a training row inside an interval
that the previous round's detector found gets weight zero. A refit for month `M` is trained on rows
before `M` minus 7 days (`train_end`). It masks only intervals whose end lies at least 24 hours
before `train_end`, so no mask depends on a stamp the refit could not have seen.

One file per round and series is written atomically, and a file that exists is skipped, so a crash
costs one series.

    uv run python ml-experiments/00068-*/e00068_fit.py round1
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
from e00068_common import BLOCK, GRID_LENGTH, GRID_START, SERIES_IDS, WORK_DIR
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SIXTH_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00006-stage1a-demand-input-feature-sweep"
FEATURES_PATH: Final[Path] = SIXTH_DIR / "inputs" / "features.parquet"
TRAIN_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
FIRST_MONTH: Final[datetime] = datetime(2022, 1, 1, tzinfo=UTC)
LAST_MONTH: Final[datetime] = datetime(2026, 6, 1, tzinfo=UTC)
"""The last validation month. No experiment scores a valid time on or after 2026-07-01."""
PURGE: Final[timedelta] = timedelta(days=7)
MIN_TRAIN_ROWS: Final[int] = 17_520
N_ROUNDS: Final[int] = 500
HALF_LIFE_YEARS: Final[float] = 0.5
CHRISTMAS_WINDOW_DAYS: Final[int] = 14
N_WORKERS: Final[int] = 6
"""Slot E's cores; one thread per worker."""
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
"""Experiment 17's input set: weather block, clock, COVID flag, windowed Christmas days."""
KNOWN_BEFORE: Final[int] = 15 * BLOCK
"""An interval is masked only when its end is this many stamps before `train_end`.

That is 14 days, which the end search looks ahead, plus the 24 hours the step window needs.
"""
WORLD_B_DELTA: Final[Path] = WORK_DIR / "world_b_delta.parquet"
"""Injected MW per series and stamp for world B (columns `series`, `stamp`, `delta_mw`)."""


def round_dir(*, name: str) -> Path:
    """The private folder of one round's refits."""
    return WORK_DIR / "rounds" / name


def intervals_path(*, name: str) -> Path:
    """The parquet of the intervals whose mask the next round's refit uses."""
    return round_dir(name=name) / "mask_intervals.parquet"


def _christmas_window() -> pl.Expr:
    """Days to the nearest 25 December, signed, null outside the 14-day window."""
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


def months() -> list[datetime]:
    """The first instant of each validation month."""
    out, month = [], FIRST_MONTH
    while month <= LAST_MONTH:
        out.append(month)
        month = datetime(month.year + month.month // 12, month.month % 12 + 1, 1, tzinfo=UTC)
    return out


def stamp_of(when: datetime) -> int:
    """The half-hour index of `when` from the grid start."""
    return int((when - GRID_START).total_seconds() // 1800)


def masked_stamps(*, intervals: np.ndarray, train_end: datetime) -> np.ndarray:
    """A boolean array over the grid: True inside intervals known to be over before `train_end`.

    `intervals` has one row per interval, `(onset, end)` in stamps, `end` exclusive.
    """
    out = np.zeros(GRID_LENGTH, dtype=bool)
    last = stamp_of(train_end) - KNOWN_BEFORE
    for onset, end in intervals:
        if end <= last:
            out[onset:end] = True
    return out


def with_injected_target(*, frame: pl.DataFrame, time_series_id: int) -> pl.DataFrame:
    """World B: add the injected MW, in units of capacity, to the training target."""
    delta = pl.read_parquet(WORLD_B_DELTA).filter(pl.col("series") == time_series_id)
    capacity = (
        pl.scan_parquet(FEATURES_PATH)
        .filter(pl.col("time_series_id") == time_series_id)
        .select(pl.col("effective_capacity_mw").median())
        .collect()
        .item()
    )
    return (
        frame.with_columns(
            stamp=((pl.col("valid_time") - GRID_START).dt.total_seconds() // 1800).cast(pl.Int64)
        )
        .join(delta.select("stamp", "delta_mw"), on="stamp", how="left", maintain_order="left")
        .with_columns(fitted_target=pl.col("target") + pl.col("delta_mw").fill_null(0.0) / capacity)
        .drop("stamp", "delta_mw")
    )


def _matrix(*, frame: pl.DataFrame) -> xgb.DMatrix:
    return xgb.DMatrix(frame.select([pl.col(c).cast(pl.Float32) for c in COLUMNS]).to_numpy())


def fit_one(
    *, frame: pl.DataFrame, train_end: datetime, intervals: np.ndarray, nthread: int
) -> xgb.Booster:
    """Fit on rows before `train_end` with recency weights, zero inside the masked intervals."""
    rows = frame.filter(pl.col("valid_time") >= TRAIN_START, pl.col("valid_time") < train_end)
    stamps = ((rows["valid_time"] - GRID_START).dt.total_seconds() // 1800).to_numpy().astype(int)
    keep = ~masked_stamps(intervals=intervals, train_end=train_end)[
        np.clip(stamps, 0, GRID_LENGTH - 1)
    ]
    age_years = (train_end - rows["valid_time"]).dt.total_days().to_numpy() / 365.25
    weight = 0.5 ** (age_years / HALF_LIFE_YEARS) * keep
    matrix = _matrix(frame=rows)
    label = "fitted_target" if "fitted_target" in rows.columns else "target"
    matrix.set_label(rows[label].to_numpy())
    matrix.set_weight(weight)
    return xgb.train({**PARAMETERS, "nthread": nthread}, matrix, num_boost_round=N_ROUNDS)


def fit_series(*, name: str, time_series_id: int, nthread: int = 1) -> str:
    """Write one series' monthly predictions for one round, with that round's mask."""
    path = round_dir(name=name) / f"{time_series_id}.parquet"
    if path.exists():
        return f"{name}/{path.name}: exists"
    table = (
        pl.read_parquet(intervals_path(name=name)) if intervals_path(name=name).exists() else None
    )
    intervals = np.zeros((0, 2), dtype=int)
    if table is not None:
        mine = table.filter(pl.col("series") == time_series_id)
        intervals = mine.select("onset", "end").to_numpy()
    frame = (
        pl.scan_parquet(FEATURES_PATH)
        .filter(pl.col("time_series_id") == time_series_id, pl.col("target").is_not_null())
        .with_columns(days_to_christmas_window=_christmas_window())
        .select("time_series_id", "valid_time", "target", "effective_capacity_mw", *COLUMNS)
        .collect()
    )
    if name.startswith("B"):
        frame = with_injected_target(frame=frame, time_series_id=time_series_id)
    pieces: list[pl.DataFrame] = []
    for month in months():
        train_end = month - PURGE
        if frame.filter(pl.col("valid_time") < train_end).height < MIN_TRAIN_ROWS:
            continue
        next_month = datetime(month.year + month.month // 12, month.month % 12 + 1, 1, tzinfo=UTC)
        valid = frame.filter(pl.col("valid_time") >= month, pl.col("valid_time") < next_month)
        if valid.is_empty():
            continue
        booster = fit_one(frame=frame, train_end=train_end, intervals=intervals, nthread=nthread)
        pieces.append(
            valid.select(
                "time_series_id", "valid_time", "target", "effective_capacity_mw"
            ).with_columns(pred=pl.Series(booster.predict(_matrix(frame=valid)).astype(np.float32)))
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pl.concat(pieces)
    out.write_parquet(path.with_suffix(".tmp"))
    path.with_suffix(".tmp").rename(path)
    return f"{name}/{path.name}: {out.height:,} rows, {len(pieces)} months"


def _run(args: tuple[str, int]) -> str:
    return fit_series(name=args[0], time_series_id=args[1])


def fit_round(*, name: str) -> None:
    """Fit every series of one round in worker processes."""
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=context) as pool:
        for line in pool.map(_run, [(name, s) for s in SERIES_IDS]):
            print(line, flush=True)


def main() -> None:
    """Fit the named round."""
    parser = argparse.ArgumentParser()
    parser.add_argument("round_name")
    fit_round(name=parser.parse_args().round_name)


if __name__ == "__main__":
    main()
