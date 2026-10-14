"""Fit experiment 11's arms and write each unit's validation predictions.

An arm is a named set of input columns. The base columns are the weather block, the local hour and
weekday, and the days to Christmas inside a 14-day window. A unit is a fold, a series, and a
fitting seed. It fits every arm and writes one parquet file holding the observed target and one
prediction column per arm, and a unit whose file exists is skipped, so a crash costs one unit.

The features come from experiment 00006's private feature table, read only. The term and
return-to-work columns join on the local date of each half-hour (the date of the period that ends at
`valid_time`, so 00:00 belongs to the previous day).

    cd ml-experiments/00011-school-term-and-return-to-work-columns-stage1a-demand
    uv run python e00011_fit.py --workers 10 --seeds 0 1 2
"""

import argparse
import multiprocessing
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Final

import e00011_term_dates as term_dates
import numpy as np
import polars as pl
import xgboost as xgb
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00011-school-term-and-return-to-work-columns-stage1a-demand"
)
FITS_DIR: Final[Path] = EXPERIMENT_DIR / "fits"
FEATURES_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00006-stage1a-demand-input-feature-sweep" / "inputs"
)
"""Experiment 00006's feature and series tables, which this experiment reads and never writes."""

N_ROUNDS: Final[int] = 500
CHRISTMAS_WINDOW_DAYS: Final[int] = 14
PURGE: Final[timedelta] = timedelta(days=7)
MIN_TRAIN_ROWS: Final[int] = 17_520
"""A series needs a year of observed half-hours between `SHORT_TRAIN_START` and a fold's purge."""
SHORT_TRAIN_START: Final[datetime] = datetime(2021, 9, 1, tzinfo=UTC)
FULL_TRAIN_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
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
"""Identical in every arm, and identical to experiment 00006's. Column subsampling is off so a
wider arm gets no free win."""

WEATHER: Final[tuple[str, ...]] = (
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
BASE: Final[tuple[str, ...]] = (
    *WEATHER,
    "hour_local",
    "day_of_week_local",
    "days_to_christmas_window",
)
RETURN: Final[tuple[str, ...]] = ("days_from_return_w",)
TERM: Final[tuple[str, ...]] = (
    "school_term_flag",
    "days_from_term_start_w",
    "days_from_term_end_w",
)


def _placebo(*, prefix: str) -> tuple[str, ...]:
    return tuple(f"{prefix}{c}" for c in (*RETURN, *TERM))


@dataclass(frozen=True)
class Arm:
    """A named set of input columns and the first valid time its training rows may have."""

    name: str
    columns: tuple[str, ...]
    train_start: datetime = SHORT_TRAIN_START


ARMS: Final[tuple[Arm, ...]] = (
    Arm(name="base", columns=BASE),
    Arm(name="return", columns=(*BASE, *RETURN)),
    Arm(name="term", columns=(*BASE, *TERM)),
    Arm(name="both", columns=(*BASE, *RETURN, *TERM)),
    Arm(name="placebo_a", columns=(*BASE, *_placebo(prefix="placebo_a_"))),
    Arm(name="placebo_b", columns=(*BASE, *_placebo(prefix="placebo_b_"))),
    Arm(name="one_harmonic", columns=(*BASE, "doy_sin_1", "doy_cos_1")),
    Arm(
        name="reference",
        columns=(
            *WEATHER,
            "hour_local",
            "day_of_week_local",
            "covid_indicator",
            "days_to_christmas_window",
        ),
        train_start=FULL_TRAIN_START,
    ),
)
"""`reference` is experiment 00006's recommended set trained on the full history."""


@dataclass(frozen=True)
class Fold:
    """A validation year and the training window that ends a purge before it."""

    valid_start: datetime
    valid_end: datetime

    @property
    def train_end(self) -> datetime:
        """The exclusive end of training: one week before validation starts."""
        return self.valid_start - PURGE


FOLDS: Final[dict[str, Fold]] = {
    "select_2023": Fold(
        valid_start=datetime(2023, 7, 1, tzinfo=UTC), valid_end=datetime(2024, 7, 1, tzinfo=UTC)
    ),
    "select_2024": Fold(
        valid_start=datetime(2024, 7, 1, tzinfo=UTC), valid_end=datetime(2025, 7, 1, tzinfo=UTC)
    ),
    "confirm_2025": Fold(
        valid_start=datetime(2025, 7, 1, tzinfo=UTC), valid_end=datetime(2026, 7, 1, tzinfo=UTC)
    ),
}
"""The three validation years of experiment 00006, in the order they were scored there."""


def local_date_of_period(*, valid_time: pl.Expr) -> pl.Expr:
    """Return the London date of the half-hour ending at `valid_time` (00:00 is the day before)."""
    return (valid_time.dt.convert_time_zone("Europe/London") - pl.duration(minutes=1)).dt.date()


def christmas_window(*, local: pl.Expr) -> pl.Expr:
    """Return the signed days to the nearest 25 December inside 14 days of it, else null.

    Matches experiment 00006's `days_to_christmas_window`, which uses the date of `valid_time`
    itself, so the new columns and this column differ by one half-hour at midnight only.
    """
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


@cache
def term_calendar() -> pl.DataFrame:
    """Return the validated daily calendar, built once per process."""
    terms = term_dates.read_terms()
    term_dates.validate(terms=terms)
    return term_dates.calendar_frame(terms=terms)


def unit_path(*, fold_name: str, series_id: int, seed: int) -> Path:
    """Return where a unit's predictions are written."""
    return FITS_DIR / fold_name / f"seed{seed}" / f"{series_id}.parquet"


def load_series(*, series_id: int, calendar: pl.DataFrame) -> pl.DataFrame:
    """Return one series' rows with the calendar columns and the Christmas window joined on."""
    columns = {c for arm in ARMS for c in arm.columns}
    stored = set(pl.scan_parquet(FEATURES_DIR / "features.parquet").collect_schema().names())
    computed = {"days_to_christmas_window", *term_calendar().columns}
    missing = columns - stored - computed
    if missing:
        msg = f"arm columns found nowhere: {sorted(missing)}"
        raise ValueError(msg)
    wanted = sorted((columns & stored) | {"time_series_id", "valid_time", "target"})
    frame = (
        pl.scan_parquet(FEATURES_DIR / "features.parquet")
        .filter(pl.col("time_series_id") == series_id)
        .select(wanted)
        .collect()
    )
    local_period = local_date_of_period(valid_time=pl.col("valid_time"))
    local_clock = pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.date()
    return (
        frame.with_columns(
            local_date=local_period,
            days_to_christmas_window=christmas_window(local=local_clock),
        )
        .join(calendar, on="local_date", how="left", validate="m:1")
        .sort("valid_time")
    )


def _matrix(*, frame: pl.DataFrame, columns: Sequence[str]) -> xgb.DMatrix:
    return xgb.DMatrix(frame.select([pl.col(c).cast(pl.Float32) for c in columns]).to_numpy())


def fit_arm(*, arm: Arm, fold: Fold, frame: pl.DataFrame, seed: int) -> np.ndarray:
    """Fit one arm on the rows before the fold's training end and predict the validation rows."""
    rows = frame.filter(
        pl.col("target").is_not_null(),
        pl.col("valid_time") >= arm.train_start,
        pl.col("valid_time") < fold.train_end,
    )
    train = _matrix(frame=rows, columns=arm.columns)
    train.set_label(rows["target"].to_numpy())
    booster = xgb.train({**PARAMETERS, "seed": seed, "nthread": 1}, train, num_boost_round=N_ROUNDS)
    valid = frame.filter(
        pl.col("target").is_not_null(),
        pl.col("valid_time") >= fold.valid_start,
        pl.col("valid_time") < fold.valid_end,
    )
    return booster.predict(_matrix(frame=valid, columns=arm.columns))


def run_unit(*, fold_name: str, series_id: int, seed: int) -> str:
    """Fit every arm for one fold, series, and seed, skipping a file that exists."""
    path = unit_path(fold_name=fold_name, series_id=series_id, seed=seed)
    if path.exists():
        return f"{path}: exists"
    fold = FOLDS[fold_name]
    frame = load_series(series_id=series_id, calendar=term_calendar())
    enough = frame.filter(
        pl.col("target").is_not_null(),
        pl.col("valid_time") >= SHORT_TRAIN_START,
        pl.col("valid_time") < fold.train_end,
    ).height
    if enough < MIN_TRAIN_ROWS:
        return f"{series_id}: only {enough} training rows in {fold_name}; skipped"
    valid = frame.filter(
        pl.col("target").is_not_null(),
        pl.col("valid_time") >= fold.valid_start,
        pl.col("valid_time") < fold.valid_end,
    ).select("time_series_id", "valid_time", "target")
    predictions = {
        arm.name: fit_arm(arm=arm, fold=fold, frame=frame, seed=seed).astype(np.float32)
        for arm in ARMS
    }
    out = valid.with_columns(*[pl.Series(k, v) for k, v in predictions.items()])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(path.with_suffix(".tmp"))
    path.with_suffix(".tmp").rename(path)
    return f"{path}: {out.height:,} rows"


def main() -> None:
    """Run every unit in parallel processes, one thread per fit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--folds", nargs="+", default=list(FOLDS))
    parser.add_argument("--limit", type=int, default=None, help="fit only this many series")
    args = parser.parse_args()
    term_calendar()
    series = pl.read_parquet(FEATURES_DIR / "series.parquet")["time_series_id"].sort().to_list()
    series = series[: args.limit] if args.limit is not None else series
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = [
            pool.submit(run_unit, fold_name=fold, series_id=sid, seed=seed)
            for seed in args.seeds
            for fold in args.folds
            for sid in series
        ]
        for future in futures:
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
