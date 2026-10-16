"""Hindsight quantile model: leave-one-month-out XGBoost on 00017's 13 columns.

For each series and each held-out month from 2022-01 to 2026-06, fit on every other month of
2019-09 to 2026-06 (past and future), with every logged closure that touches the series (any
duration, padded 24 hours) removed from training, and a symmetric recency weight centred on the
held-out month with a half-life of 0.5 years. The held-out month is predicted at q10, q50, q90.
One file per series is written; a series whose file exists is skipped.

    uv run python ml-experiments/00041-*/e00041_hindsight.py [--timing]
"""

from __future__ import annotations

import os
import sys
import time
from datetime import UTC, datetime, timedelta
from typing import Final

import numpy as np
import polars as pl
import xgboost as xgb
from e00041_common import FEATURES_PATH, SERIES_IDS, WORK_DIR, logged_intervals

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
RANDOM_EXCLUDE = os.environ.get("RANDOM_EXCLUDE") == "1"
OUT_DIR = WORK_DIR / ("hindsight_random" if RANDOM_EXCLUDE else "hindsight")
CHECK_SERIES = (3, 8, 10, 14, 17, 25)


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


def _matrix(frame: pl.DataFrame) -> xgb.DMatrix:
    return xgb.DMatrix(frame.select([pl.col(c).cast(pl.Float32) for c in COLUMNS]).to_numpy())


def fit_series(*, time_series_id: int, first_months_only: int | None = None) -> str:
    """Fit all held-out months of one series; write quantile predictions with capacity units."""
    path = OUT_DIR / f"{time_series_id}.parquet"
    if path.exists() and first_months_only is None:
        return f"{path.name}: exists"
    frame = (
        pl.scan_parquet(FEATURES_PATH)
        .filter(pl.col("time_series_id") == time_series_id, pl.col("target").is_not_null())
        .with_columns(days_to_christmas_window=_christmas_window())
        .select("valid_time", "target", "effective_capacity_mw", *COLUMNS)
        .sort("valid_time")
        .collect()
    )
    times = frame["valid_time"].dt.epoch("s").to_numpy()
    keep = np.ones(frame.height, dtype=bool)
    side = str(time_series_id)
    logged = logged_intervals().filter((pl.col("a") == side) | (pl.col("b") == side))
    rng = np.random.default_rng(time_series_id)
    for row in logged.iter_rows(named=True):
        lo = (row["start_utc"] - PAD).timestamp()
        hi = (row["end_utc"] + PAD).timestamp()
        if RANDOM_EXCLUDE:
            span = hi - lo
            lo = rng.uniform(times[0], times[-1] - span)
            hi = lo + span
        keep &= ~((times >= lo) & (times <= hi))
    pieces: list[pl.DataFrame] = []
    months = _months()[:first_months_only]
    for month in months:
        nxt = datetime(month.year + month.month // 12, month.month % 12 + 1, 1, tzinfo=UTC)
        held = (times >= month.timestamp()) & (times < nxt.timestamp())
        if not held.any():
            continue
        train_mask = keep & ~held
        train = frame.filter(pl.Series(train_mask))
        centre = (month.timestamp() + nxt.timestamp()) / 2
        age_years = np.abs(train["valid_time"].dt.epoch("s").to_numpy() - centre) / (365.25 * 86400)
        matrix = _matrix(train)
        matrix.set_label(train["target"].to_numpy())
        matrix.set_weight(0.5 ** (age_years / HALF_LIFE_YEARS))
        booster = xgb.train(PARAMETERS, matrix, num_boost_round=N_ROUNDS)
        valid = frame.filter(pl.Series(held))
        pred = booster.predict(_matrix(valid))
        pred = np.sort(pred, axis=1)
        cap = valid["effective_capacity_mw"].to_numpy()
        pieces.append(
            valid.select("valid_time").with_columns(
                series=pl.lit(time_series_id).cast(pl.Int64),
                obs_mw=pl.Series(valid["target"].to_numpy() * cap),
                q10_mw=pl.Series(pred[:, 0] * cap),
                q50_mw=pl.Series(pred[:, 1] * cap),
                q90_mw=pl.Series(pred[:, 2] * cap),
                in_logged=pl.Series(~keep[held]),
            )
        )
    out = pl.concat(pieces)
    if first_months_only is None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out.write_parquet(path.with_suffix(".tmp"))
        path.with_suffix(".tmp").rename(path)
    return f"{time_series_id}: {out.height:,} rows"


def _run(time_series_id: int) -> str:
    return fit_series(time_series_id=time_series_id)


def main() -> None:
    """Run all series, two processes (slot D)."""
    if "--timing" in sys.argv:
        start = time.time()
        print(fit_series(time_series_id=8, first_months_only=2), time.time() - start)
        return
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for line in pool.map(_run, CHECK_SERIES if RANDOM_EXCLUDE else SERIES_IDS):
            print(line, flush=True)


if __name__ == "__main__":
    main()
