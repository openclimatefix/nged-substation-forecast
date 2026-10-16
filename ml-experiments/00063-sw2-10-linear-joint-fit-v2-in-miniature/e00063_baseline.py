"""Leave-one-month-out XGBoost baseline without radiation inputs, for the 8 Primaries.

The procedure is that of experiment 00041's hindsight baseline with `shortwave` and `longwave`
removed from the 13 input columns: for each series and each held-out month from 2022-01 to
2026-06, fit on every other month of 2019-09 to 2026-06 (past and future), with every logged
closure that touches the series (any duration, padded 24 hours) removed from training, and a
symmetric recency weight centred on the held-out month with a half-life of 0.5 years. The held-out
month is predicted at q10, q50, q90. One file per series is written atomically, and a series whose
file exists is skipped.

    uv run python ml-experiments/00063-*/e00063_baseline.py [--timing] [--check-frozen]

`--check-frozen` refits the first two months of series 8 with all 13 columns and prints the largest
difference from experiment 00041's stored medians (zero when the procedure is reproduced).
"""

from __future__ import annotations

import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00063_common as common
import numpy as np
import polars as pl
import xgboost as xgb

FROZEN_COLUMNS: Final[tuple[str, ...]] = (
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
"""Experiment 00041's 13 input columns."""
RADIATION_COLUMNS: Final[tuple[str, ...]] = ("shortwave", "longwave")
COLUMNS: Final[tuple[str, ...]] = tuple(c for c in FROZEN_COLUMNS if c not in RADIATION_COLUMNS)
"""The 11 input columns of the baseline without radiation."""
QUANTILES: Final[tuple[float, ...]] = (0.1, 0.5, 0.9)
FIRST_MONTH: Final[datetime] = datetime(2022, 1, 1, tzinfo=UTC)
LAST_MONTH: Final[datetime] = datetime(2026, 6, 1, tzinfo=UTC)
PAD: Final[timedelta] = timedelta(hours=24)
HALF_LIFE_YEARS: Final[float] = 0.5
PARAMETERS: Final[dict[str, float | int | str | list[float]]] = {
    "eta": 0.1,
    "max_depth": 5,
    "min_child_weight": 5,
    "subsample": 0.8,
    "tree_method": "hist",
    "max_bin": 64,
    "objective": "reg:quantileerror",
    "quantile_alpha": list(QUANTILES),
    "seed": 0,
    "nthread": 1,
}
N_ROUNDS: Final[int] = 150
N_WORKERS: Final[int] = 4
"""Four workers of one thread each: slot E has four cores."""


def _christmas_window() -> pl.Expr:
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
    return pl.when(nearest <= 14).then(signed).otherwise(None)


def _months() -> list[datetime]:
    out, month = [], FIRST_MONTH
    while month <= LAST_MONTH:
        out.append(month)
        month = datetime(month.year + month.month // 12, month.month % 12 + 1, 1, tzinfo=UTC)
    return out


def _matrix(*, frame: pl.DataFrame, columns: tuple[str, ...]) -> xgb.DMatrix:
    return xgb.DMatrix(frame.select([pl.col(c).cast(pl.Float32) for c in columns]).to_numpy())


def fit_series(
    *,
    time_series_id: int,
    columns: tuple[str, ...] = COLUMNS,
    out_dir: Path = common.BASELINE_DIR,
    first_months_only: int | None = None,
) -> tuple[str, pl.DataFrame]:
    """Fit all held-out months of one series; write the predictions (MW) unless a check run.

    Args:
        time_series_id: The series.
        columns: The input columns.
        out_dir: Where the series' file goes.
        first_months_only: Fit only this many first months and write nothing (a timing run).

    Returns:
        A one-line message and the predictions (`valid_time`, `series`, `obs_mw`, `q10_mw`,
        `q50_mw`, `q90_mw`, `in_logged`).
    """
    path = out_dir / f"{time_series_id}.parquet"
    if path.exists() and first_months_only is None:
        return f"{path.name}: exists", pl.DataFrame()
    frame = (
        pl.scan_parquet(common.FEATURES_PATH)
        .filter(pl.col("time_series_id") == time_series_id, pl.col("target").is_not_null())
        .with_columns(days_to_christmas_window=_christmas_window())
        .select("valid_time", "target", "effective_capacity_mw", *FROZEN_COLUMNS)
        .sort("valid_time")
        .collect()
    )
    times = frame["valid_time"].dt.epoch("s").to_numpy()
    keep = np.ones(frame.height, dtype=bool)
    side = str(time_series_id)
    logged = common.logged_intervals().filter((pl.col("a") == side) | (pl.col("b") == side))
    for row in logged.iter_rows(named=True):
        lo = (row["start_utc"] - PAD).timestamp()
        hi = (row["end_utc"] + PAD).timestamp()
        keep &= ~((times >= lo) & (times <= hi))
    pieces: list[pl.DataFrame] = []
    for month in _months()[:first_months_only]:
        following = datetime(month.year + month.month // 12, month.month % 12 + 1, 1, tzinfo=UTC)
        held = (times >= month.timestamp()) & (times < following.timestamp())
        if not held.any():
            continue
        train = frame.filter(pl.Series(keep & ~held))
        centre = (month.timestamp() + following.timestamp()) / 2
        age_years = np.abs(train["valid_time"].dt.epoch("s").to_numpy() - centre) / (365.25 * 86400)
        matrix = _matrix(frame=train, columns=columns)
        matrix.set_label(train["target"].to_numpy())
        matrix.set_weight(0.5 ** (age_years / HALF_LIFE_YEARS))
        booster = xgb.train(PARAMETERS, matrix, num_boost_round=N_ROUNDS)
        valid = frame.filter(pl.Series(held))
        predicted = np.sort(booster.predict(_matrix(frame=valid, columns=columns)), axis=1)
        capacity = valid["effective_capacity_mw"].to_numpy()
        pieces.append(
            valid.select("valid_time").with_columns(
                series=pl.lit(time_series_id).cast(pl.Int64),
                obs_mw=pl.Series(valid["target"].to_numpy() * capacity),
                q10_mw=pl.Series(predicted[:, 0] * capacity),
                q50_mw=pl.Series(predicted[:, 1] * capacity),
                q90_mw=pl.Series(predicted[:, 2] * capacity),
                in_logged=pl.Series(~keep[held]),
            )
        )
    out = pl.concat(pieces)
    if first_months_only is None:
        out_dir.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        out.write_parquet(temporary)
        temporary.rename(path)
    return f"{time_series_id}: {out.height:,} rows", out


def _run(time_series_id: int) -> str:
    start = time.time()
    message, _ = fit_series(time_series_id=time_series_id)
    return f"{message} ({time.time() - start:.0f} s)"


def check_frozen() -> None:
    """Refit series 8's first two months with 13 columns and compare with experiment 00041."""
    _, refit = fit_series(time_series_id=8, columns=FROZEN_COLUMNS, first_months_only=2)
    stored = pl.read_parquet(common.FROZEN_BASELINE_DIR / "8.parquet")
    both = refit.join(stored, on="valid_time", suffix="_stored")
    for quantile in (10, 50, 90):
        worst = (both[f"q{quantile}_mw"] - both[f"q{quantile}_mw_stored"]).abs().max()
        print(f"q{quantile}: {both.height} rows compared, largest difference {worst} MW")


def main() -> None:
    """Fit every series in four single-thread workers, or run a timing or check run."""
    print("baseline columns:", ", ".join(COLUMNS), f"({len(COLUMNS)} of {len(FROZEN_COLUMNS)})")
    if "--timing" in sys.argv:
        start = time.time()
        message, _ = fit_series(time_series_id=8, first_months_only=2)
        print(message, f"{time.time() - start:.1f} s for two months")
        return
    if "--check-frozen" in sys.argv:
        check_frozen()
        return
    start = time.time()
    with ProcessPoolExecutor(
        max_workers=N_WORKERS, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for line in pool.map(_run, common.SERIES_IDS):
            print(line, flush=True)
    print(f"total {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
