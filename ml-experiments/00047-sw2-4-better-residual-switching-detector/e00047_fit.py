r"""Refit experiment 17's arm H0.5 monthly, with and without a bank-holiday column.

The arms are the baseline `H0.5` (experiment 17's arm unchanged), `HOL` (`H0.5` plus
`days_to_nearest_holiday`), and `PLA` (`H0.5` plus the same column built from holiday dates moved 28
days later). Each month from `FIRST_MONTH` to `LAST_MONTH` is forecast by a model trained on rows
that end `PURGE` before the month starts, as in experiment 17. One job is one arm, seed, and series,
and writes one file atomically, so a crashed run resumes by skipping the files that exist.

    uv run python ml-experiments/00047-*/e00047_fit.py run            # all jobs, 4 workers
    uv run python ml-experiments/00047-*/e00047_fit.py time           # time a few fits
    uv run python ml-experiments/00047-*/e00047_fit.py check-stored   # H0.5 seed 0 against 17
    uv run python ml-experiments/00047-*/e00047_fit.py run --series 10 --max-months 2 \\
        --out /path/to/scratch                                         # smoke test
"""

from __future__ import annotations

import argparse
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, date, datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Final, Literal

import holidays
import numpy as np
import polars as pl
import xgboost as xgb
from e00047_common import FEATURES_PATH, FITS_17_DIR, SERIES_IDS, WORK_DIR

ArmType = Literal["H0.5", "HOL", "PLA"]
FITS_DIR: Final[Path] = WORK_DIR / "fits"
TRAIN_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
FIRST_MONTH: Final[datetime] = datetime(2022, 1, 1, tzinfo=UTC)
LAST_MONTH: Final[datetime] = datetime(2026, 6, 1, tzinfo=UTC)
"""The last validation month. No experiment scores a valid time on or after 2026-07-01."""
PURGE: Final[timedelta] = timedelta(days=7)
MIN_TRAIN_ROWS: Final[int] = 17_520
"""A series needs a year of observed half-hours before the purge to enter."""
N_ROUNDS: Final[int] = 500
HALF_LIFE_YEARS: Final[float] = 0.5
CHRISTMAS_WINDOW_DAYS: Final[int] = 14
HOLIDAY_YEARS: Final[range] = range(2018, 2029)
HOLIDAY_CLIP_DAYS: Final[int] = 7
PLACEBO_SHIFT_DAYS: Final[int] = 28
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
BASE_COLUMNS: Final[tuple[str, ...]] = (
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
"""Experiment 6's Primary input set, which is experiment 17's arm H0.5."""
HOLIDAY_COLUMN: Final[str] = "days_to_nearest_holiday"
FITTED_JOBS: Final[tuple[tuple[ArmType, int], ...]] = (
    ("H0.5", 1),
    ("H0.5", 2),
    ("HOL", 0),
    ("HOL", 1),
    ("HOL", 2),
    ("PLA", 0),
)
"""The arm-seeds this experiment fits. `H0.5` seed 0 is experiment 17's stored fit."""
SLOW_FIT_SECONDS: Final[float] = 15.0
"""A fit slower than this stops the run for a re-plan (the plan's compute check)."""
STORED_TOLERANCE: Final[float] = 1e-4


def columns_for(*, arm: ArmType) -> tuple[str, ...]:
    """Return the input columns of an arm."""
    return BASE_COLUMNS if arm == "H0.5" else (*BASE_COLUMNS, HOLIDAY_COLUMN)


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


def bank_holiday_dates() -> list[date]:
    """Return the bank holidays in England, from the `holidays` package, sorted."""
    return sorted(holidays.country_holidays("GB", subdiv="ENG", years=HOLIDAY_YEARS))


@cache
def holiday_gaps(*, shift_days: int = 0) -> pl.DataFrame:
    """Return, for each calendar date, the signed days to the nearest bank holiday.

    The sign is positive before the holiday and negative after it, clipped at 7 days either way, as
    experiment 33's `holiday_gaps` computes it. With `shift_days`, every holiday date moves that
    many days later first (the placebo calendar, which keeps each date's weekday at 28 days).
    """
    holiday_dates = np.array(
        [d + timedelta(days=shift_days) for d in bank_holiday_dates()], dtype="datetime64[D]"
    )
    days = pl.date_range(
        date(HOLIDAY_YEARS.start, 1, 1), date(HOLIDAY_YEARS.stop - 1, 12, 31), eager=True
    )
    gaps = (holiday_dates[None, :] - days.to_numpy()[:, None]).astype("timedelta64[D]").astype(int)
    nearest = gaps[np.arange(days.len()), np.abs(gaps).argmin(axis=1)]
    return pl.DataFrame(
        {
            "date": days,
            HOLIDAY_COLUMN: np.clip(nearest, -HOLIDAY_CLIP_DAYS, HOLIDAY_CLIP_DAYS).astype(
                np.float64
            ),
        }
    )


def months() -> list[datetime]:
    """Return the first instant of each validation month."""
    out, month = [], FIRST_MONTH
    while month <= LAST_MONTH:
        out.append(month)
        month = _next_month(month=month)
    return out


def _next_month(*, month: datetime) -> datetime:
    return datetime(month.year + month.month // 12, month.month % 12 + 1, 1, tzinfo=UTC)


def load_frame(*, time_series_id: int, arm: ArmType) -> pl.DataFrame:
    """Return one series' rows with a target, the arm's columns, and the holiday column of the arm.

    The holiday column is looked up on the local date of the period-ending valid time, as
    experiment 33 does. `PLA` uses the placebo calendar.
    """
    shift = PLACEBO_SHIFT_DAYS if arm == "PLA" else 0
    return (
        pl.scan_parquet(FEATURES_PATH)
        .filter(pl.col("time_series_id") == time_series_id, pl.col("target").is_not_null())
        .with_columns(
            days_to_christmas_window=_christmas_window(),
            date=pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.date(),
        )
        .select("valid_time", "target", "date", *BASE_COLUMNS)
        .collect()
        .join(
            holiday_gaps(shift_days=shift),
            on="date",
            how="left",
            validate="m:1",
            maintain_order="left",
        )
        .drop("date")
    )


def _matrix(*, frame: pl.DataFrame, columns: tuple[str, ...]) -> xgb.DMatrix:
    return xgb.DMatrix(frame.select([pl.col(c).cast(pl.Float32) for c in columns]).to_numpy())


def _fit(
    *, frame: pl.DataFrame, train_end: datetime, columns: tuple[str, ...], seed: int, nthread: int
) -> xgb.Booster:
    """Fit on the rows before `train_end`, weighted by a half-life of 0.5 years of age."""
    rows = frame.filter(pl.col("valid_time") >= TRAIN_START, pl.col("valid_time") < train_end)
    matrix = _matrix(frame=rows, columns=columns)
    matrix.set_label(rows["target"].to_numpy())
    age_years = (train_end - rows["valid_time"]).dt.total_days().to_numpy() / 365.25
    matrix.set_weight(0.5 ** (age_years / HALF_LIFE_YEARS))
    parameters = {**PARAMETERS, "seed": seed, "nthread": nthread}
    return xgb.train(parameters, matrix, num_boost_round=N_ROUNDS)


def fit_months(
    *,
    frame: pl.DataFrame,
    arm: ArmType,
    seed: int,
    nthread: int = 1,
    only_months: set[datetime] | None = None,
    max_months: int | None = None,
) -> pl.DataFrame:
    """Return `valid_time`, `target`, and `prediction` for the validation months of one series.

    `only_months` keeps only those months (the injection check); `max_months` keeps the first few
    months that qualify (the smoke test). The qualifying rule is experiment 17's: a year of rows
    before the purge, and at least one row in the month.
    """
    columns = columns_for(arm=arm)
    pieces: list[pl.DataFrame] = []
    for month in months():
        train_end = month - PURGE
        if frame.filter(pl.col("valid_time") < train_end).height < MIN_TRAIN_ROWS:
            continue
        valid = frame.filter(
            pl.col("valid_time") >= month, pl.col("valid_time") < _next_month(month=month)
        )
        if valid.is_empty():
            continue
        if max_months is not None and len(pieces) >= max_months:
            break
        if only_months is not None and month not in only_months:
            continue
        booster = _fit(
            frame=frame, train_end=train_end, columns=columns, seed=seed, nthread=nthread
        )
        prediction = booster.predict(_matrix(frame=valid, columns=columns))
        pieces.append(
            valid.select("valid_time", "target").with_columns(
                prediction=pl.Series(prediction.astype(np.float32))
            )
        )
    if not pieces:
        return pl.DataFrame(
            schema={
                "valid_time": pl.Datetime("us", "UTC"),
                "target": pl.Float32,
                "prediction": pl.Float32,
            }
        )
    return pl.concat(pieces)


def job_path(*, out: Path, arm: ArmType, seed: int, time_series_id: int) -> Path:
    """Return the file one job writes."""
    return out / f"{arm}_s{seed}" / f"{time_series_id}.parquet"


def run_job(
    *,
    arm: ArmType,
    seed: int,
    time_series_id: int,
    out: Path,
    max_months: int | None = None,
) -> str:
    """Fit one arm-seed for one series and write its file, unless the file exists."""
    path = job_path(out=out, arm=arm, seed=seed, time_series_id=time_series_id)
    if path.exists():
        return f"{arm} s{seed} {time_series_id}: exists"
    started = time.monotonic()
    frame = load_frame(time_series_id=time_series_id, arm=arm)
    result = fit_months(frame=frame, arm=arm, seed=seed, max_months=max_months)
    if result.is_empty():
        return f"{arm} s{seed} {time_series_id}: never has a year of training data"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    result.with_columns(time_series_id=pl.lit(time_series_id, dtype=pl.Int32)).write_parquet(
        temporary
    )
    temporary.rename(path)
    seconds = time.monotonic() - started
    per_fit = seconds / max(result["valid_time"].dt.truncate("1mo").n_unique(), 1)
    flag = "  SLOW: re-plan" if per_fit > SLOW_FIT_SECONDS else ""
    return f"{arm} s{seed} {time_series_id}: {result.height:,} rows, {per_fit:.1f} s a fit{flag}"


def _run(args: tuple[ArmType, int, int, Path, int | None]) -> str:
    arm, seed, time_series_id, out, max_months = args
    return run_job(
        arm=arm, seed=seed, time_series_id=time_series_id, out=out, max_months=max_months
    )


def main() -> None:
    """Run the jobs, time a few fits, or check `H0.5` seed 0 against experiment 17."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "time", "check-stored"))
    parser.add_argument("--series", type=int, nargs="*", default=list(SERIES_IDS))
    parser.add_argument("--max-months", type=int, default=None)
    parser.add_argument("--out", type=Path, default=FITS_DIR)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "time":
        frame = load_frame(time_series_id=args.series[0], arm="HOL")
        started = time.monotonic()
        count = args.max_months or 3
        fit_months(frame=frame, arm="HOL", seed=0, max_months=count)
        print(f"{(time.monotonic() - started) / count:.1f} s a fit (series {args.series[0]})")
    elif args.command == "check-stored":
        check_stored(time_series_id=args.series[0], max_months=args.max_months)
    else:
        jobs = [
            (arm, seed, series, args.out, args.max_months)
            for arm, seed in FITTED_JOBS
            for series in args.series
        ]
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            for line in pool.map(_run, jobs):
                print(line, flush=True)


def check_stored(*, time_series_id: int, max_months: int | None) -> None:
    """Refit `H0.5` seed 0 for one series and compare with experiment 17's stored predictions."""
    frame = load_frame(time_series_id=time_series_id, arm="H0.5")
    fresh = fit_months(frame=frame, arm="H0.5", seed=0, max_months=max_months)
    stored = pl.read_parquet(FITS_17_DIR / f"{time_series_id}.parquet").select(
        "valid_time", stored="H0.5"
    )
    joined = fresh.join(stored, on="valid_time", how="left", validate="1:1")
    worst = float((joined["prediction"] - joined["stored"]).abs().max())  # ty: ignore[invalid-argument-type]
    print(f"series {time_series_id}: {joined.height:,} rows, largest difference {worst:.2e}")
    if not worst <= STORED_TOLERANCE:
        raise SystemExit(f"STOP: refit differs from experiment 17 by more than {STORED_TOLERANCE}")


if __name__ == "__main__":
    main()
